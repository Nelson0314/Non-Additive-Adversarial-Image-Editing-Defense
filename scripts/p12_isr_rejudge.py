"""以 ISR 聯集判準重判既有的所有 run。

判準（規格 §4，源自 arXiv:2512.14320）：

    ISR = 語意失敗 OR 感知劣化

語意失敗沿用 E25 §1.2 的定義（配對差、n≥2、|mean| > sd，只用 SigLIP——
CLIP 的 edit_effect 對照未通過，見 E25 §1.1）。感知劣化是新增的那一半：
編輯輸出本身的無參考品質相對未防禦編輯下降超過門檻。

為什麼值得重判既有資料。`results.csv` 自 E2 起就有 `edit_niqe_a` 與
`edit_niqe_b`（`src/metrics/suite.py::full` 對 a、b 各報一次無參考品質），
而全專案沒有任何腳本讀過它們——這與 E25 發現 `edit_siglip_*` 從未被讀過是
同一型的疏漏。實測 `runs/e29c_P_tau0.10` 的 identity 那一格是 3.104 → 4.675
（NIQE 越低越好），即防禦後的輸出確實較差。那個差值算不算「不能用」，
由 p11 的人眼階梯定門檻。

執行：python scripts/p12_isr_rejudge.py --degrade_tau <p11 定出的值>
成本：只讀 CSV，秒級。
"""

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# results.csv 中已有的無參考指標。目前只有 niqe；p11 若選出別的指標，
# 須先讓 MetricSuite.full 也報那一個，才能在此重判既有資料。
DEGRADE_HIGHER_IS_WORSE = {"niqe": True}


def judge_cell(deltas_sem, deltas_deg, degrade_tau):
    """一格（一個 run × 一個淨化臂）的 ISR 判定。

    `deltas_sem`：逐影像的 SigLIP(防禦後編輯) − SigLIP(未防禦編輯)。
    `deltas_deg`：逐影像的劣化量，已由呼叫端依指標方向轉為「越大越劣化」。
    `degrade_tau`：劣化門檻，由 p11 的人眼階梯定出。

    n≥2 是必要條件而非保險：n=1 時 pstdev 恆為 0，`|mean| > sd` 對任何負值
    自動成立，E25 曾因此在 `e6_stepsP` 與 `e6_stepsLA` 上產生 24 格假陽性。
    劣化那一側同樣要求 n≥2，理由相同——單張影像的劣化量沒有離散度可比，
    兩軸用不同的樣本數門檻會讓 ISR 的兩半不對等。
    """
    if not deltas_sem or not deltas_deg:
        raise ValueError(
            "ISR 判定需要至少一筆配對差。空清單不得回傳 False——"
            "那會把「沒有資料」與「判定為未擋下」混為一談"
        )
    if len(deltas_sem) != len(deltas_deg):
        raise ValueError(
            f"兩軸的樣本數不符（{len(deltas_sem)} vs {len(deltas_deg)}），"
            "它們必須來自同一批影像"
        )
    n = len(deltas_sem)
    m_sem, sd_sem = mean(deltas_sem), pstdev(deltas_sem)
    m_deg, sd_deg = mean(deltas_deg), pstdev(deltas_deg)
    semantic_fail = bool(n >= 2 and m_sem < 0 and abs(m_sem) > sd_sem)
    degrade_fail = bool(n >= 2 and m_deg > degrade_tau)
    return {
        "n": n,
        "mean_dsiglip": m_sem, "sd_dsiglip": sd_sem,
        "mean_ddegrade": m_deg, "sd_ddegrade": sd_deg,
        "semantic_fail": semantic_fail,
        "degrade_fail": degrade_fail,
        "isr": bool(semantic_fail or degrade_fail),
    }


def collect(run_dir: Path, degrade_metric: str = "niqe"):
    """回傳 {(purify, strength, noise_split): (sem_deltas, deg_deltas)}。

    CSV 一律以 utf-8 讀：`stop_reason` 欄含中文，Windows 預設 cp950 會
    UnicodeDecodeError。

    缺欄時拋出而非略過：早於該欄位加入的 run 是真實資訊，必須在報告中
    列為「無此軸的資料」，靜默跳過會讓分母悄悄變小。
    """
    path = run_dir / "results.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    if degrade_metric not in DEGRADE_HIGHER_IS_WORSE:
        raise ValueError(
            f"未知的劣化指標 {degrade_metric!r}。方向未定義時無法把它轉成"
            "「越大越劣化」，判定會反向"
        )
    a_key, b_key = f"edit_{degrade_metric}_a", f"edit_{degrade_metric}_b"
    if a_key not in rows[0] or b_key not in rows[0]:
        raise KeyError(f"{path} 缺 {a_key}／{b_key} 欄")

    higher_is_worse = DEGRADE_HIGHER_IS_WORSE[degrade_metric]
    out = {}
    for r in rows:
        if not r.get("edit_siglip_a") or not r.get("edit_siglip_b"):
            continue
        if not r.get(a_key) or not r.get(b_key):
            continue
        key = (r["purify"], r["strength"], r["noise_split"])
        sem = float(r["edit_siglip_b"]) - float(r["edit_siglip_a"])
        raw = float(r[b_key]) - float(r[a_key])
        deg = raw if higher_is_worse else -raw
        out.setdefault(key, ([], []))
        out[key][0].append(sem)
        out[key][1].append(deg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/p12_isr_rejudge")
    ap.add_argument("--degrade_metric", default="niqe")
    ap.add_argument("--degrade_tau", type=float, default=1.0,
                    help="感知劣化門檻，由 p11 的人眼階梯定出。"
                         "預設 1.0 只是佔位，正式判定必須傳入定錨值")
    args = ap.parse_args()

    runs_root = ROOT / args.runs
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        try:
            cells = collect(run_dir, args.degrade_metric)
        except KeyError as e:
            # 早於該欄位加入的 run。列為「無此軸的資料」而非靜默跳過。
            skipped.append((run_dir.name, str(e)))
            continue
        for (purify, strength, split), (sem, deg) in sorted(cells.items()):
            r = judge_cell(sem, deg, args.degrade_tau)
            rows.append({"run": run_dir.name, "purify": purify,
                         "strength": strength, "noise_split": split, **r})

    if not rows:
        raise RuntimeError("沒有任何可判定的格。請確認 runs/ 內有 results.csv")

    with open(out / "isr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_isr = sum(r["isr"] for r in rows)
    n_sem = sum(r["semantic_fail"] for r in rows)
    n_deg = sum(r["degrade_fail"] for r in rows)
    lines = [
        "# E31 ISR 重判（既有 run）",
        "",
        f"判準：語意失敗 OR 感知劣化（{args.degrade_metric}，"
        f"門檻 {args.degrade_tau}）。",
        "",
        f"- 可判定的格數：{len(rows)}（{len({r['run'] for r in rows})} 個 run）",
        f"- 語意失敗：{n_sem}",
        f"- 感知劣化：{n_deg}",
        f"- **ISR 成立：{n_isr}**",
        f"- 無此軸資料的 run：{len(skipped)}",
        "",
        "## 乾淨攻擊（purify=identity、strength=0.0）",
        "",
        "這一列是正式判準所在的位置：未經淨化的攻擊。",
        "",
        "| run | split | n | ΔSigLIP | Δ劣化 | 語意 | 劣化 | ISR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in [r for r in rows if r["purify"] == "identity"
              and r["strength"] == "0.0"]:
        lines.append(
            f"| {r['run']} | {r['noise_split']} | {r['n']} | "
            f"{r['mean_dsiglip']:+.4f} | {r['mean_ddegrade']:+.3f} | "
            f"{'是' if r['semantic_fail'] else '否'} | "
            f"{'是' if r['degrade_fail'] else '否'} | "
            f"{'**是**' if r['isr'] else '否'} |")
    if skipped:
        lines += ["", "## 無此軸資料的 run", ""]
        lines += [f"- `{n}`：{e}" for n, e in skipped]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[p12] {len(rows)} 格，語意 {n_sem}、劣化 {n_deg}、ISR {n_isr}；"
          f"另有 {len(skipped)} 個 run 無此軸資料")
    print(f"[p12] 寫出 {out / 'isr.csv'} 與 {out / 'summary.md'}")


if __name__ == "__main__":
    main()
