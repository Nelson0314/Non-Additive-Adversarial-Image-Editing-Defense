"""把像素臂的批次輸出成 `compare.html`：每一格都必須有影像可看。

專案規則是「判準以人眼為主、數值指標為輔；指標與人眼矛盾時以人眼為準」。
本頁的存在就是為了讓那個判定做得起來，所以三張圖（防禦圖、未防禦的編輯、
防禦後的編輯）一律並列，指標放在圖底下當註腳而不是當標題。

不挑選指標：`results.csv` 有的欄位全部列出。挑選過的表格看起來永遠是好的。

用法：
    python scripts/phase_compare_page.py --run runs/phaseA_full
    python scripts/phase_compare_page.py --run runs/phaseB --layout condition
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

FID_KEYS = ("fid_dists", "fid_lpips", "fid_psnr", "fid_ssim", "fid_clip_img",
            "fid_nima", "fid_cnniqa")
EDIT_KEYS = ("edit_lpips", "edit_clip_drop", "edit_siglip_drop")

CSS = """
body{font:13px/1.5 -apple-system,'Segoe UI',sans-serif;margin:24px;background:#fafafa;color:#222}
h1{font-size:18px} h2{font-size:15px;margin:28px 0 8px}
table{border-collapse:collapse;margin-bottom:24px}
td,th{border:1px solid #ddd;padding:6px;vertical-align:top;background:#fff}
th{background:#f0f0f0;font-weight:600;text-align:left;white-space:nowrap}
img{display:block;width:190px;height:190px;object-fit:cover;border:1px solid #ccc}
.trip{display:flex;gap:4px}
.cap{font-size:11px;color:#666;text-align:center;margin-top:2px}
.m{font-size:11px;color:#333;margin-top:6px;white-space:pre;font-family:ui-monospace,monospace}
.warn{color:#b00;font-weight:600}
.note{background:#fff8e1;border:1px solid #e0d0a0;padding:10px;margin:12px 0;max-width:920px}
"""


def cell_html(run: Path, row: dict, tag: str) -> str:
    name = row["image"]
    trip = []
    for sub, cap in (("def", "防禦圖"), ("edit_orig", "未防禦的編輯"),
                     ("edit_def", "防禦後的編輯")):
        p = f"{name}__{tag}__{sub}.png"
        exists = (run / p).exists()
        trip.append(
            f'<div><img src="{html.escape(p)}" alt="{html.escape(p)}">'
            f'<div class="cap">{cap}{"" if exists else " <span class=warn>缺圖</span>"}</div></div>'
        )
    lines = []
    if row.get("budget_reached"):
        flag = " <span class=warn>unreachable</span>" if str(
            row.get("unreachable", "")).lower() == "true" else ""
        lines.append(f'DISTS 目標 {row.get("budget_target","")} → 實得 '
                     f'{row["budget_reached"]}{flag}')
        lines.append(f'半徑 {row.get("radius","")}')
    for k in FID_KEYS + EDIT_KEYS:
        if row.get(k) not in (None, ""):
            lines.append(f"{k:<16}{row[k]}")
    for k in ("amp_dev", "active_fraction", "total_seconds", "stage1_seconds"):
        if row.get(k) not in (None, ""):
            lines.append(f"{k:<16}{row[k]}")
    return (f'<div class="trip">{"".join(trip)}</div>'
            f'<div class="m">{html.escape(chr(10).join(lines))}</div>')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--layout", choices=("budget", "condition"), default="budget",
                    help="budget：每個預算點一張表（像素臂）；condition：單表（latent 臂）")
    args = ap.parse_args()
    out = args.out or (args.run / "compare.html")

    with (args.run / "results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{args.run / 'results.csv'} 沒有任何列")

    has_budget = "budget_target" in rows[0] and args.layout == "budget"
    images = sorted({r["image"] for r in rows})
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    budgets = (list(dict.fromkeys(r["budget_target"] for r in rows))
               if has_budget else [None])

    parts = [f"<style>{CSS}</style>",
             f"<h1>{html.escape(args.run.name)} · compare</h1>",
             '<div class="note">判準以人眼為主、數值指標為輔。指標與人眼矛盾時'
             '以人眼為準並記錄。<br>三張圖分別是防禦圖、未防禦的編輯、防禦後的'
             '編輯——<b>未防禦的編輯必須真的成功</b>，否則抗編輯那一欄的分母'
             '不成立（DEC-022）。</div>']

    for b in budgets:
        if b is not None:
            parts.append(f"<h2>DISTS 預算 {html.escape(str(b))}</h2>")
        parts.append("<table><tr><th>影像</th>"
                     + "".join(f"<th>{html.escape(c)}</th>" for c in conds)
                     + "</tr>")
        for img in images:
            parts.append(f"<tr><th>{html.escape(img)}</th>")
            for c in conds:
                match = [r for r in rows if r["image"] == img
                         and r["condition"] == c
                         and (b is None or r["budget_target"] == b)]
                if not match:
                    parts.append("<td>—</td>")
                    continue
                r = match[0]
                tag = (f'{c}__d{float(r["budget_target"]):g}'
                       if has_budget else c)
                parts.append(f"<td>{cell_html(args.run, r, tag)}</td>")
            parts.append("</tr>")
        parts.append("</table>")

    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"寫入 {out}（{len(images)} 影像 × {len(conds)} 條件 × "
          f"{len(budgets)} 預算）")


if __name__ == "__main__":
    main()
