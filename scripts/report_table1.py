"""把任一批 run 整理成 Lo et al. (CVPR 2024) Table 1 的格式。

    python scripts/report_table1.py --out docs/RESULTS_TABLE1.md

判準是該表的五個指標，全部量「免疫後的編輯輸出」與「原始編輯結果」之間的
不相似度。**方向與影像品質相反**：PSNR／SSIM／VIFp／FSIM 越低代表免疫越
成功，LPIPS 越高代表越成功。

為什麼一定要併列 L∞
─────────────────────────────────────────────────────────────────────

Lo 的協定把全部方法固定在 L∞ ≤ 0.06 上比較，那是「匹配失真」的執行方式。
本專案既有的 run 用的是 LPIPS 綁定約束，實際 L∞ 由 0.06 到 0.85 都有——
`e15_S_tau0.05`（空間變形）是 0.5654，即 κ 的 9.4 倍。不併列 L∞ 就會拿
不同預算的數字互相比較，那是無效的比較。

**L∞ 對非加性方法本身也不是好的失真尺**（把一條邊緣移動不到一個像素，
L∞ 可接近 1 而人眼幾乎看不出來），這正是本專案改用 LPIPS 的理由。此處
併列兩者，讓「哪一組數字可以互相比較」在表面上就看得出來，不必回去翻設定。
"""

import argparse
import csv
import glob
import os
import re
import statistics as st
from pathlib import Path

# 論文 Table 1 的原始數字。SD v1.4 那三欄。
PUBLISHED = [
    ("[Lo] PhotoGuard encoder attack", 0.06,
     dict(psnr=18.8437, ssim=0.6318, vif_p=0.2118, fsim=0.7757, lpips=0.4131)),
    ("[Lo] PhotoGuard diffusion attack", 0.06,
     dict(psnr=18.2617, ssim=0.6504, vif_p=0.2656, fsim=0.7693, lpips=0.4056)),
    ("[Lo] semantic attack", 0.06,
     dict(psnr=15.1487, ssim=0.4470, vif_p=0.1462, fsim=0.6584, lpips=0.5901)),
]
COLS = ["psnr", "ssim", "vif_p", "fsim", "lpips"]
ARROW = {"psnr": "↓", "ssim": "↓", "vif_p": "↓", "fsim": "↓", "lpips": "↑"}


def collect(runs_root: Path, pool_prompts: bool = True):
    """讀 runs/*/results.csv 的乾淨攻擊列，回傳逐 (run, site) 的平均。

    乾淨攻擊 = `purify == 'identity'` 且淨化強度為 0。這一列才是「沒有任何
    後處理時的免疫效果」，也就是 Table 1 量的東西；其餘各列是抗淨化，
    Lo 的表沒有那一項。

    `pool_prompts` 會把 `<run>_p1` 併入 `<run>`。論文補充材料 §A 的 Table 1
    是「每個物件兩個編輯 prompt」一起平均，兩個 prompt 分開的數字都不是
    該表的對照對象。兩半的影像數與種子數相同，故直接對所有列取平均，
    與先分半再平均等值。
    """
    groups = {}
    for f in sorted(glob.glob(str(runs_root / "*" / "results.csv"))):
        run = os.path.basename(os.path.dirname(f))
        if pool_prompts:
            run = re.sub(r"_p\d+$", "", run)
        for r in csv.DictReader(open(f, encoding="utf-8")):
            # lo_baseline 的 results.csv 沒有 purify 欄（該協定不含淨化），
            # 其餘 run 有。缺欄視為乾淨攻擊，有欄則必須是 identity + 0。
            if "purify" in r:
                if r["purify"] != "identity" or (r.get("strength") or "0") != "0.0":
                    continue
            key = (run, r.get("attack") or r.get("site") or "")
            vals = {}
            for c in COLS:
                v = r.get(f"edit_{c}")
                vals[c] = float(v) if v not in (None, "") else None
            li = r.get("pert_linf") or r.get("defimg_linf")
            lp = r.get("pert_lpips") or r.get("defimg_lpips")
            vals["_linf"] = float(li) if li else None
            vals["_lpips"] = float(lp) if lp else None
            groups.setdefault(key, []).append(vals)
    out = []
    for (run, site), vs in groups.items():
        row = {"label": f"{run} / {site}" if site else run, "n": len(vs)}
        for c in COLS + ["_linf", "_lpips"]:
            got = [v[c] for v in vs if v[c] is not None]
            row[c] = st.mean(got) if got else None
        out.append(row)
    return out


def c_a_in_prompt(content: str, prompt: str) -> bool:
    """c_a 是否以**完整詞**出現在編輯 prompt 裡。

    必須看詞邊界，不能用子字串：本專案的類別裡 `man` 是 `woman` 的子字串，
    子字串比對會把 man 類的「a woman」判成「c_a 有出現」，方向剛好相反。
    """
    return re.search(rf"\b{re.escape(content.strip())}\b", prompt,
                     flags=re.IGNORECASE) is not None


def by_prompt(runs_root: Path, base: str):
    """把 `base` 與 `base_p*` 的列拆成逐 (攻擊, 編輯 prompt) 的平均。

    存在的理由：semantic attack 的輸入 c_a 由防禦方選，可能出現也可能不出現
    在攻擊方寫的編輯 prompt 裡。論文 §4.3 把「不出現時仍有效」當成優點提出，
    等於承認兩種情況的難度不同。合併後的那張表看不出這件事，但它決定
    semantic 這根柱子能不能與論文對照，必須單獨列出。兩根 PhotoGuard
    不使用 c_a，其逐 prompt 差異只反映 prompt 本身的編輯幅度。
    """
    groups = {}
    for f in sorted(glob.glob(str(runs_root / "*" / "results.csv"))):
        run = os.path.basename(os.path.dirname(f))
        if re.sub(r"_p\d+$", "", run) != base:
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            key = (r["attack"], c_a_in_prompt(r["content"], r["prompt"]))
            groups.setdefault(key, []).append(
                {c: float(r[f"edit_{c}"]) for c in COLS})
    out = []
    for (atk, in_prompt), vs in sorted(groups.items()):
        row = {"attack": atk, "c_a_in_prompt": in_prompt, "n": len(vs)}
        for c in COLS:
            row[c] = st.mean(v[c] for v in vs)
        out.append(row)
    return out


def render_by_prompt(rows):
    head = ("| 攻擊 | c_a 出現在編輯 prompt 裡 | 情境 | n | "
            + " | ".join(f"{c.upper()} {ARROW[c]}" for c in COLS) + " |")
    lines = [head, "|" + "---|" * (4 + len(COLS))]
    for r in sorted(rows, key=lambda r: (r["attack"], r["c_a_in_prompt"])):
        scen = "改動其他區域（編輯 prompt 2）" if r["c_a_in_prompt"] \
            else "改掉該內容（編輯 prompt 1）"
        lines.append(
            f"| `{r['attack']}` | {'是' if r['c_a_in_prompt'] else '否'} | "
            f"{scen} | {r['n']} | "
            + " | ".join(fmt(r[c], 4 if c != "psnr" else 2) for c in COLS) + " |"
        )
    return "\n".join(lines)


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def render(rows, kappa):
    head = "| 方法 | n | L∞ | ×κ | LPIPS(擾動) | " + " | ".join(
        f"{c.upper()} {ARROW[c]}" for c in COLS) + " |"
    sep = "|" + "---|" * (5 + len(COLS))
    lines = [head, sep]
    for r in sorted(rows, key=lambda r: -(r["lpips"] or 0)):
        ratio = f"{r['_linf']/kappa:.1f}×" if r["_linf"] else "—"
        lines.append(
            f"| `{r['label']}` | {r['n']} | {fmt(r['_linf'])} | {ratio} | "
            f"{fmt(r['_lpips'])} | "
            + " | ".join(fmt(r[c], 4 if c != "psnr" else 2) for c in COLS)
            + " |"
        )
    lines.append("|" + " |" * (5 + len(COLS)))
    for label, k, v in PUBLISHED:
        lines.append(
            f"| **{label}** | 150×20 | ≤{k} | 1.0× | — | "
            + " | ".join(fmt(v[c], 4 if c != "psnr" else 2) for c in COLS)
            + " |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs/RESULTS_TABLE1.md")
    ap.add_argument("--kappa", type=float, default=0.06)
    ap.add_argument("--min_n", type=int, default=1)
    ap.add_argument("--split_base", default="lo_baseline",
                    help="要額外做逐編輯 prompt 分解的 run 名稱（不含 _p1 尾綴）")
    args = ap.parse_args()

    rows = [r for r in collect(Path(args.runs)) if r["n"] >= args.min_n]
    if not rows:
        raise SystemExit(f"{args.runs} 底下沒有可判讀的乾淨攻擊列")

    body = render(rows, args.kappa)
    split = by_prompt(Path(args.runs), args.split_base)
    split_md = (
        f"\n## 逐編輯 prompt 的分解（`{args.split_base}`）\n\n"
        "論文補充材料 §A 的每個物件有兩個編輯 prompt，上表是兩者一起平均。\n"
        "此處拆開，因為 semantic attack 的輸入 c_a 由防禦方選定，可能出現、\n"
        "也可能不出現在攻擊方寫的編輯 prompt 裡；論文 §4.3 把「不出現時仍\n"
        "有效」列為額外的優點，等於承認兩種情況難度不同。兩根 PhotoGuard\n"
        "不使用 c_a，其兩列的差異只反映 prompt 本身的編輯幅度，可作為對照。\n\n"
        + render_by_prompt(split) + "\n"
    ) if split else ""
    text = f"""# Table 1 對照：本專案 vs Lo et al. (CVPR 2024)

<!-- 由 scripts/report_table1.py 產生，不要手改 -->

判準為 Lo et al., *Distraction is All You Need*, CVPR 2024, Table 1 的五個
指標，全部量「免疫後的編輯輸出」與「原始編輯結果」之間的不相似度。
箭頭是**免疫成功**的方向，與影像品質的方向相反。

`×κ` 是該列實際的 L∞ 相對論文預算 κ = {args.kappa} 的倍數。**只有 ×κ 接近
1.0 的列可以與論文的數字直接比較**；倍數明顯大於 1 的列是在更大的失真
預算上取得的成績，不構成對照。缺 VIFp／FSIM 的列（顯示為 —）是 2026-08-03
之前跑的，當時 `suite.pairwise` 還沒有這兩項。

{body}
{split_md}
## 讀這張表的三個前提

1. **資料集不同。** 論文是用擴散模型生成的 150 張圖（dog／horse／man，
   每類 2 個編輯 prompt），每格平均 20 個隨機種子；本專案用的是自己的
   影像。跨資料集的絕對值不構成嚴格比較，只有同一列內的相對關係是乾淨的。
2. **論文未公布 strength 與 guidance_scale。** 本專案依 E26 的結論一律用
   guidance_scale = 7.5（w = 1 時 SD v1.4 幾乎不服從 prompt，該設定下
   量不出編輯效果）。
3. **這張表只有第 1 族判準。** 語意軸與感知劣化軸見 `docs/LEDGER.md`；
   三族之間的等級相關為 0.140 / −0.207 / 0.014（n = 217），即這張表的
   高分不預測另外兩族的高分。
"""
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\n寫出 {args.out}")


if __name__ == "__main__":
    main()
