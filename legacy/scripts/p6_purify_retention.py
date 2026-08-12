"""E25-2 —— 淨化保留率：非加性的優勢是不是在「淨化之後」才出現？

動機。E23 在無淨化下量到 Sbic / P = 0.85×，據此判定非加性沒有優勢。
但 `results.csv` 每一列都帶著 22 個淨化條件（identity / blur / noise / jpeg /
quantize，各數個強度），而所有既有報告只讀了無淨化那一格。文獻對這一點有
明確預期：NAPPure（ICCV 2025）量到既有的淨化方法是為加性雜訊設計的，
對非加性擾動（模糊／遮擋／flow-field 變形）明顯失效；stAdv 那條線也記載
「空間平滑的擾動對 JPEG 防禦較強健」。若該預期成立，S/P 的比值應該隨淨化
強度上升。

量什麼。對每個淨化條件

    retention = net(該條件) / net(無淨化)

即該條件還留下原本效果的幾分之幾。比較的是 S 與 P 的 retention，而不是 net
的絕對值——後者已由無淨化那一格決定，這裡問的是「誰掉得比較慢」。

被保留的量取哪一個。**主判定為 Δsiglip**（防禦後的編輯輸出對 prompt 的
服從度變化，越負代表編輯越不照 prompt 走）。`net_lpips` 兩次被量到與
「編輯有沒有失敗」不對應——它量的是輸出移動了多少——故不可作為主判定，
但仍逐格輸出以便與先驗批次對讀。Δniqe（畫面劣化）第三個一併輸出：
它是代價不是成果，若某一側的保留率靠劣化撐著，這一欄會顯示出來。

三個量的方向不同，保留率的定義因此逐量而異：

    Δsiglip  越負越好 → retention = Δsiglip(該條件) / Δsiglip(無淨化)
    net_lpips 越大越好 → retention = net(該條件) / net(無淨化)
    Δniqe    越小越好（是代價）→ 不計保留率，只逐格列出絕對值

分母接近零時比值不可解讀。無淨化那一格的量小於 `MIN_BASE` 時，該配對的
該量記為 NaN 並在輸出中標示，不進入判定——先驗批次曾因兩側都趨近零而
出現 −43、−98 這種比值。

判準（事先宣告，不得事後調整）：

- 「淨化強健性優勢」= S 的 retention > P 的 retention，且在 ≥3 個獨立的
  淨化算子上成立（blur / noise / jpeg / quantize 算四個獨立算子；同一算子
  的不同強度只算一個）。要求跨算子而非跨強度，是因為同一算子的相鄰強度
  高度相關，湊三個強度不構成三個獨立證據。

該判準實測後發現太鬆，此處記下而不改寫。上述寫法只要求「該算子的任一
強度佔優」，於是六個強度中的一個就能扛下整個算子，七對全部 4/4 成立，判準
失去分辨力。實際資料是分裂的：E23 的 blur 在 0.5/1.0/1.5 佔優、在 2.0/3.0
反轉，jpeg 在 30/60/85 佔優、在 45/75/95 反轉。故另列一條較嚴的
`ops_majority`：該算子多數強度佔優才計入。兩條都輸出，寬鬆那條保留是為了
讓「它為何不足」有據可查（同 E22 保留被推翻的 0.97 假設）。

成對的 run 從何而來。只比較設定逐項相同、僅 site 不同的一對，見
`PAIRS`。E15 與 E21/E23 的保真約束不同，故不跨實驗配對。
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "p6_purify_retention"

# (標籤, site S 側的 run, site P 側的 run)。兩側必須同實驗、同 τ、同步數。
PAIRS = [
    ("E15 τ=0.02（舊約束）", "e15_S_tau0.02", "e15_P_tau0.02"),
    ("E15 τ=0.05（舊約束）", "e15_S_tau0.05", "e15_P_tau0.05"),
    ("E15 τ=0.10（舊約束）", "e15_S_tau0.10", "e15_P_tau0.10"),
    ("E21 τ=0.02 bicubic 25 步", "e21_Sbic_tau0.02", "e21_P_tau0.02"),
    ("E21 τ=0.05 bicubic 25 步", "e21_Sbic_tau0.05", "e21_P_tau0.05"),
    ("E21 τ=0.10 bicubic 25 步", "e21_Sbic_tau0.10", "e21_P_tau0.10"),
    ("E23 τ=0.05 bicubic 100 步", "e23_Sbic_s100_tau0.05", "e23_P_s100_tau0.05"),
]

# 無淨化的那一格。評測時 identity 只以訓練種子跑一次（noise_split="train"），
# 未見種子的無淨化格子是 blur 強度 0（等同 identity），見 scripts/run_defense.py。
BASE = ("blur", 0.0)

# 主判定用的量。改為 Δsiglip：net_lpips 兩次被量到與「編輯有沒有失敗」
# 不對應（見模組 docstring），故降為並列輸出而非判定依據。
PRIMARY = "dsiglip"
MIN_OPS = 3

# 無淨化那一格的量小於此值時，保留率的分母不可靠，該配對的該量記為 NaN。
# 門檻取 E25 量到的 edit_effect 標準差 0.0237 的四分之一：低於它的效果本來
# 就在雜訊裡，除以它得到的比值沒有意義。
MIN_BASE = {"dsiglip": 0.006, "net_lpips": 0.01}


def arm_means(run: str):
    """回傳 {(算子, 強度): {量名: 均值}}，只取未見種子那一批。"""
    path = ROOT / "runs" / run / "results.csv"
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if r["noise_split"] == "heldout"]
    if not rows:
        raise FileNotFoundError(f"{run} 沒有任何 heldout 列，無法比較")
    by = defaultdict(list)
    for r in rows:
        by[(r["purify"], float(r["strength"]))].append(r)
    out = {}
    for key, rs in by.items():
        out[key] = {
            "n": len(rs),
            "net_lpips": float(np.mean([float(r["net_lpips"]) for r in rs])),
            "dsiglip": float(np.mean([float(r["edit_siglip_b"]) - float(r["edit_siglip_a"])
                                      for r in rs])),
            # Δniqe 是代價不是成果，故不計保留率，只逐格列出。缺欄時拋出而
            # 不填 0：先驗批次曾有整整一族欄位從未被讀過而無人察覺。
            "dniqe": float(np.mean([float(r["edit_niqe_b"]) - float(r["edit_niqe_a"])
                                    for r in rs])),
        }
    return out


def retention(cur: float, base: float, quantity: str) -> float:
    """保留率。分母太小則回傳 NaN——比值在該處由量測雜訊主導。

    `dsiglip` 越負越好、`net_lpips` 越大越好，但兩者的保留率都是「現值 ÷
    無淨化值」，故方向由分子分母自帶，此處不必再分支。要分支的只有
    「分母夠不夠大」，而那個門檻逐量而異。
    """
    if abs(base) < MIN_BASE[quantity]:
        return float("nan")
    return cur / base


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, verdicts = [], []

    for label, run_s, run_p in PAIRS:
        S, P = arm_means(run_s), arm_means(run_p)
        if BASE not in S or BASE not in P:
            raise KeyError(f"{label}：找不到無淨化基準格 {BASE}，保留率無從定義")
        base_s, base_p = S[BASE], P[BASE]

        usable = (abs(base_s[PRIMARY]) >= MIN_BASE[PRIMARY]
                  and abs(base_p[PRIMARY]) >= MIN_BASE[PRIMARY])

        print(f"\n=== {label} ===")
        print(f"  S={run_s}  P={run_p}   n={base_s['n']}/{base_p['n']}")
        print(f"  無淨化 Δsiglip：S {base_s['dsiglip']:+.4f}   P {base_p['dsiglip']:+.4f}"
              f"{'' if usable else '   ← 分母低於門檻，本配對不計入判定'}")
        print(f"  無淨化 net_lpips：S {base_s['net_lpips']:.4f}   P {base_p['net_lpips']:.4f}"
              f"（並列參考，非判定依據）")
        print(f"{'算子':>9s}{'強度':>7s}{'S Δsig':>10s}{'P Δsig':>10s}"
              f"{'S 保留':>9s}{'P 保留':>9s}{'S net 保留':>11s}{'P net 保留':>11s}"
              f"{'S Δniqe':>10s}{'P Δniqe':>10s}")

        ops_favouring_s, per_op = set(), defaultdict(list)
        for key in sorted(set(S) & set(P), key=lambda k: (k[0], k[1])):
            if key == BASE:
                continue
            s, p = S[key], P[key]
            rs = retention(s[PRIMARY], base_s[PRIMARY], PRIMARY)
            rp = retention(p[PRIMARY], base_p[PRIMARY], PRIMARY)
            rs_net = retention(s["net_lpips"], base_s["net_lpips"], "net_lpips")
            rp_net = retention(p["net_lpips"], base_p["net_lpips"], "net_lpips")
            # NaN 的比較恆為 False，不可靠的格子因此自動不計入，不需另設分支
            favour = usable and rs > rp
            if favour:
                ops_favouring_s.add(key[0])
            if usable and not (np.isnan(rs) or np.isnan(rp)):
                per_op[key[0]].append(favour)
            print(f"{key[0]:>9s}{key[1]:>7g}{s[PRIMARY]:>+10.4f}{p[PRIMARY]:>+10.4f}"
                  f"{100 * rs:>8.1f}%{100 * rp:>8.1f}%"
                  f"{100 * rs_net:>10.1f}%{100 * rp_net:>10.1f}%"
                  f"{s['dniqe']:>+10.3f}{p['dniqe']:>+10.3f}")
            rows.append({
                "pair": label, "run_s": run_s, "run_p": run_p,
                "purify": key[0], "strength": key[1], "usable": usable,
                "s_dsiglip": s["dsiglip"], "p_dsiglip": p["dsiglip"],
                "s_retention": rs, "p_retention": rp,
                "s_net": s["net_lpips"], "p_net": p["net_lpips"],
                "s_net_retention": rs_net, "p_net_retention": rp_net,
                "s_dniqe": s["dniqe"], "p_dniqe": p["dniqe"],
            })

        majority = sorted(op for op, v in per_op.items() if sum(v) * 2 > len(v))
        ok = len(ops_favouring_s) >= MIN_OPS
        ok_maj = len(majority) >= MIN_OPS
        verdicts.append((label, sorted(ops_favouring_s), ok, majority, ok_maj,
                         {op: f"{sum(v)}/{len(v)}" for op, v in sorted(per_op.items())}))
        print(f"  → 寬鬆（任一強度）：{sorted(ops_favouring_s)}"
              f"（{len(ops_favouring_s)}/4）{'成立' if ok else '不成立'}")
        print(f"  → 嚴格（多數強度）：{majority}"
              f"（{len(majority)}/4）{'成立' if ok_maj else '不成立'}"
              f"   逐算子佔優比例 "
              + " ".join(f"{op} {sum(v)}/{len(v)}" for op, v in sorted(per_op.items())))

    _write_csv(OUT / "summary.csv", rows)

    print("\n=== 判準彙總（S 保留率 > P 保留率，需 ≥"
          f"{MIN_OPS} 個獨立淨化算子）===")
    print(f"{'':4s}{'寬鬆':>6s}{'嚴格':>6s}  {'配對':32s}逐算子佔優比例")
    for label, ops, ok, majority, ok_maj, ratios in verdicts:
        print(f"{'':4s}{'成立' if ok else '不成立':>6s}"
              f"{'成立' if ok_maj else '不成立':>6s}  {label:32s}"
              + " ".join(f"{op} {r}" for op, r in ratios.items()))

    n_skip = sum(1 for r in rows if not r["usable"])
    if n_skip:
        print(f"\n注意：{n_skip} 格因無淨化基準的 Δsiglip 低於 {MIN_BASE[PRIMARY]}"
              "而未計入判定。分母在雜訊量級時，保留率不可解讀。")
    print("\n判定用的是 Δsiglip 的保留率；net_lpips 兩欄並列輸出僅供與先驗批次"
          "\n對讀，不參與判定——它量的是輸出移動了多少，不是編輯有沒有失敗。"
          "\nΔniqe 兩欄是代價不是成果：某一側若靠畫面劣化撐住效果，會在該欄顯示。")
    print(f"\n寫入 {OUT}")


def _write_csv(path: Path, rows) -> None:
    if not rows:
        raise ValueError(f"沒有任何資料可寫入 {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
