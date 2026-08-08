#!/usr/bin/env python
"""跨批次的人眼比對頁：同一張圖、同一個 τ，兩個批次的 `x_def` 並排。

    python scripts/compare_batches.py --old runs/v14 --new runs/v14r --tau 0.2 \
        --images bird_03 cat_02 dog_03 --conditions N1 N2 N3 R photoguard_c \
        --out runs/v14r_vs_v14_tau0.2.html

## 為什麼需要它，而不是用段 4 的 compare.html

`run_report` 產生的 `compare.html` 是**單一批次內**的比對：同一批的不同條件、
不同 τ、不同淨化。它答不出「換了一組門檻之後，同一個條件在同一個 LPIPS 上
看起來有沒有變差」——那需要把兩個批次的同一格擺在一起。

`HANDOVER_2026-08-08` §8.3 第 3 項就是這個問題：放寬 `tau_acut`（0.04 → 0.16）
買到的是訓練自由度，代價可能是同一 LPIPS 下可見失真變差。使用者先前判讀
「τ=0.20 可接受」是在**受約束的 φ** 上做的，那個判讀不自動延續到放寬後。
段 3 要 7 小時，值得先看再決定。

## 目錄慣例

兩個批次都吃 `<批次根>[_<影像>]/<條件>/<影像>/` 這個結構。分片批次
（`v14_dog_03`）與合併批次（`v14_merged`）都適用，故 `--old`／`--new` 給的是
**不含影像後綴的字首**，本腳本自己依序試 `<字首>_<影像>`、`<字首>_merged`、
`<字首>` 三種。找不到就在該格標紅，不靜默略過——少一張圖而版面照樣排得出來
是這一頁最容易發生的誤導。

影像以**相對路徑**寫進 `<img src>`，與 `compare.html` 同一作法：產物留在
`runs/` 內、隨 repo 一起入版控，離線可看，不必內嵌 base64 把檔案撐大。
"""
from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path
from typing import Dict, List, Optional

# 表格中要列出的數值欄位。取的是「人眼判讀時會想同時看到的量」：達成的 τ
# 用來確認兩批真的在同一個預算上（否則整頁的比較不成立），其餘是
# `RESULTS_2026-08-08` §1.1 那張表的五個指標，以及位移場自己的幅度。
METRICS = [
    ("tau_achieved", "達成 τ", 4),
    ("scale_k", "縮放 k", 3),
    ("fid_psnr", "PSNR", 2),
    ("fid_ssim", "SSIM", 4),
    ("fid_dists", "DISTS", 4),
    ("fid_linf", "L∞", 3),
    ("disp_max_px", "位移max px", 2),
]

CSS = """
body{font:14px/1.5 system-ui,sans-serif;margin:1.5rem;max-width:100rem}
h1{margin:0 0 .2rem}
table{border-collapse:collapse;margin:.4rem 0 1rem;width:100%}
th,td{border-bottom:1px solid #8884;padding:.25rem;vertical-align:top}
th[scope=row]{text-align:left;white-space:nowrap;font-weight:600}
img{width:100%;max-width:15rem;height:auto;display:block;background:#8881}
td.miss{color:#c33;text-align:center;font-weight:600}
td.num{white-space:nowrap;font-variant-numeric:tabular-nums;color:#666;font-size:.85em}
tr.metrics td{border-bottom:2px solid #8884}
.lead{background:#8881;padding:.6rem .8rem;border-radius:.3rem}
.index{font-size:.85em;line-height:1.9;color:#666;margin-bottom:1.5rem}
section{margin-bottom:2.5rem}
@media(prefers-color-scheme:dark){body{background:#111;color:#ddd}
  td.num,.index{color:#999}}
"""


def resolve_batch_dir(prefix: str, image_id: str) -> Optional[Path]:
    """依序試分片、合併、單一目錄三種慣例。"""
    for cand in (f"{prefix}_{image_id}", f"{prefix}_merged", prefix):
        p = Path(cand)
        if p.is_dir():
            return p
    return None


def cell_dir(prefix: str, image_id: str, condition: str) -> Optional[Path]:
    root = resolve_batch_dir(prefix, image_id)
    if root is None:
        return None
    d = root / condition / image_id
    return d if d.is_dir() else None


def read_metrics(d: Optional[Path], tau: str) -> Dict[str, str]:
    if d is None:
        return {}
    p = d / f"fidelity_tau{tau}.csv"
    if not p.is_file():
        return {}
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else {}


def img_td(d: Optional[Path], name: str, out_dir: Path) -> str:
    if d is None:
        return '<td class="miss">批次目錄不存在</td>'
    p = d / name
    if not p.is_file():
        return f'<td class="miss">缺 {html.escape(name)}</td>'
    rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
    return f'<td><img loading="lazy" src="{html.escape(rel)}" alt="{html.escape(str(p))}"></td>'


def fmt(row: Dict[str, str], key: str, nd: int) -> str:
    v = row.get(key)
    if v in (None, ""):
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except ValueError:
        return html.escape(str(v))


def build(args) -> str:
    out_dir = Path(args.out).resolve().parent
    tau = args.tau
    parts: List[str] = [
        f"<title>{html.escape(args.title)}</title>",
        f"<style>{CSS}</style>",
        f"<h1>{html.escape(args.title)}</h1>",
        f"<p>τ = {html.escape(tau)} · 舊批次 <code>{html.escape(args.old)}</code>"
        f" · 新批次 <code>{html.escape(args.new)}</code></p>",
        f'<p class="lead">{args.lead}</p>',
    ]
    idx = " · ".join(f'<a href="#i-{i}">{html.escape(i)}</a>' for i in args.images)
    parts.append(f'<div class="index">{idx}</div>')

    for image_id in args.images:
        parts.append(f'<section id="i-{html.escape(image_id)}">'
                     f"<h2>{html.escape(image_id)}</h2><table>")
        parts.append("<tr><th>條件</th><th>原圖</th>"
                     f"<th>{html.escape(args.old_label)}</th>"
                     f"<th>{html.escape(args.new_label)}</th>"
                     f"<th>殘差 · {html.escape(args.old_label)}</th>"
                     f"<th>殘差 · {html.escape(args.new_label)}</th></tr>")
        for cond in args.conditions:
            old_d = cell_dir(args.old, image_id, cond)
            new_d = cell_dir(args.new, image_id, cond)
            # 原圖取新批次的；兩批的 orig.png 是同一個資料集檔案。
            orig_d = new_d or old_d
            parts.append(
                f'<tr><th scope="row">{html.escape(cond)}</th>'
                + img_td(orig_d, "orig.png", out_dir)
                + img_td(old_d, f"x_def_tau{tau}.png", out_dir)
                + img_td(new_d, f"x_def_tau{tau}.png", out_dir)
                + img_td(old_d, f"residual_tau{tau}.png", out_dir)
                + img_td(new_d, f"residual_tau{tau}.png", out_dir)
                + "</tr>"
            )
            om, nm = read_metrics(old_d, tau), read_metrics(new_d, tau)
            def line(row):
                if not row:
                    return "（無 fidelity CSV）"
                return " · ".join(f"{lab} {fmt(row, k, nd)}"
                                  for k, lab, nd in METRICS)
            parts.append(
                '<tr class="metrics"><td></td><td class="num"></td>'
                f'<td class="num" colspan="2">{html.escape(args.old_label)}：'
                f"{line(om)}<br>{html.escape(args.new_label)}：{line(nm)}</td>"
                '<td class="num" colspan="2"></td></tr>'
            )
        parts.append("</table></section>")
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="舊批次的目錄字首，例如 runs/v14")
    ap.add_argument("--new", required=True, help="新批次的目錄字首，例如 runs/v14r")
    ap.add_argument("--tau", default="0.2", help="τ 的字面值，須與檔名一致（0.2 不是 0.20）")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="跨批次人眼比對")
    ap.add_argument("--old-label", default="舊")
    ap.add_argument("--new-label", default="新")
    ap.add_argument("--lead", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(args), encoding="utf-8")
    print(f"寫入 {out}")


if __name__ == "__main__":
    main()
