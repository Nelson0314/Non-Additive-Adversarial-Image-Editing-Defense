"""E25-2 —— 淨化保留率：非加性的優勢是不是在「淨化之後」才出現？

**動機。** E23 在無淨化下量到 Sbic / P = 0.85×，據此判定非加性沒有優勢。
但 `results.csv` 每一列都帶著 22 個淨化臂（identity / blur / noise / jpeg /
quantize，各數個強度），而所有既有報告只讀了無淨化那一格。文獻對這一點有
明確預期：NAPPure（ICCV 2025）量到既有的淨化方法是為**加性**雜訊設計的，
對非加性擾動（模糊／遮擋／flow-field 變形）明顯失效；stAdv 那條線也記載
「空間平滑的擾動對 JPEG 防禦較強健」。若該預期成立，S/P 的比值應該隨淨化
強度上升。

**量什麼。** 對每個淨化臂

    retention = net(該臂) / net(無淨化)

即該臂還留下原本效果的幾分之幾。比較的是 S 與 P 的 retention，而不是 net
的絕對值——後者已由無淨化那一格決定，這裡問的是「誰掉得比較慢」。

**為什麼要同時用 Δsiglip 重算一次。** E25-1 顯示 `net_lpips` 與語意軸給出
不同的圖像（726 格中語意失敗 0 格）。若兩個被保留量給出相反的排序，那表示
「保留率」保留的是視覺偏移而非防禦，結論不能只憑 net 下。兩者一併列出。

**判準（事先宣告，不得事後調整）：**

- 「淨化強健性優勢」= S 的 retention > P 的 retention，且在 **≥3 個獨立的
  淨化算子**上成立（blur / noise / jpeg / quantize 算四個獨立算子；同一算子
  的不同強度只算一個）。要求跨算子而非跨強度，是因為同一算子的相鄰強度
  高度相關，湊三個強度不構成三個獨立證據。

**該判準實測後發現太鬆，此處記下而不改寫。** 上述寫法只要求「該算子的**任一**
強度佔優」，於是六個強度中的一個就能扛下整個算子，七對全部 4/4 成立，判準
失去分辨力。實際資料是分裂的：E23 的 blur 在 0.5/1.0/1.5 佔優、在 2.0/3.0
反轉，jpeg 在 30/60/85 佔優、在 45/75/95 反轉。故另列一條較嚴的
`ops_majority`：該算子**多數強度**佔優才計入。兩條都輸出，寬鬆那條保留是為了
讓「它為何不足」有據可查（同 E22 保留被推翻的 0.97 假設）。

**成對的 run 從何而來。** 只比較設定逐項相同、僅 site 不同的一對，見
`PAIRS`。E15 與 E21/E23 的保真約束不同，故不跨實驗配對。
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
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

QUANTITIES = ("net_lpips", "dsiglip")
MIN_OPS = 3


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
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, verdicts = [], []

    for label, run_s, run_p in PAIRS:
        S, P = arm_means(run_s), arm_means(run_p)
        if BASE not in S or BASE not in P:
            raise KeyError(f"{label}：找不到無淨化基準格 {BASE}，保留率無從定義")
        base_s, base_p = S[BASE], P[BASE]

        print(f"\n=== {label} ===")
        print(f"  S={run_s}  P={run_p}   n={base_s['n']}/{base_p['n']}")
        print(f"  無淨化 net：S {base_s['net_lpips']:.4f}   P {base_p['net_lpips']:.4f}"
              f"   S/P {base_s['net_lpips'] / base_p['net_lpips']:.2f}×")
        print(f"{'算子':>9s}{'強度':>7s}{'S net':>9s}{'P net':>9s}"
              f"{'S 保留':>9s}{'P 保留':>9s}{'S/P':>7s}{'Δsiglip S':>11s}{'Δsiglip P':>11s}")

        ops_favouring_s, per_op = set(), defaultdict(list)
        for key in sorted(set(S) & set(P), key=lambda k: (k[0], k[1])):
            if key == BASE:
                continue
            s, p = S[key], P[key]
            rs = base_s["net_lpips"] and s["net_lpips"] / base_s["net_lpips"]
            rp = base_p["net_lpips"] and p["net_lpips"] / base_p["net_lpips"]
            if rs > rp:
                ops_favouring_s.add(key[0])
            per_op[key[0]].append(rs > rp)
            print(f"{key[0]:>9s}{key[1]:>7g}{s['net_lpips']:>9.4f}{p['net_lpips']:>9.4f}"
                  f"{100 * rs:>8.1f}%{100 * rp:>8.1f}%"
                  f"{s['net_lpips'] / p['net_lpips'] if p['net_lpips'] else float('nan'):>7.2f}"
                  f"{s['dsiglip']:>+11.4f}{p['dsiglip']:>+11.4f}")
            rows.append({
                "pair": label, "run_s": run_s, "run_p": run_p,
                "purify": key[0], "strength": key[1],
                "s_net": s["net_lpips"], "p_net": p["net_lpips"],
                "s_retention": rs, "p_retention": rp,
                "s_over_p": s["net_lpips"] / p["net_lpips"] if p["net_lpips"] else float("nan"),
                "s_dsiglip": s["dsiglip"], "p_dsiglip": p["dsiglip"],
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

    print("\n注意：保留率的分子分母都是 net_lpips，而 E25-1 已顯示 net_lpips 量的是"
          "\n『輸出移動了多少』而非『編輯有沒有失敗』（726 格語意失敗 0 格）。"
          "\n故本表證明的是『非加性的視覺偏移對淨化較耐受』，"
          "\n**不是**『非加性的防禦對淨化較耐受』。兩者不可混為一談。")
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
