#!/usr/bin/env python
"""人眼判定「編輯成功了沒有」的比對頁：條件 × 種子的編輯輸出並排。

    python scripts/edit_success_page.py --batch runs/v14r_merged \
        --images bird_03 cat_02 dog_03 --out runs/v14r_edit_success.html

## 為什麼另做一頁

`compare.html` 一列給一格，六張圖依因果鏈排列，適合追一格的來龍去脈。但
「這個防禦到底有沒有擋住編輯」要看的是**同一個條件跨種子的成敗比例**，以及
**各條件之間的比例差**——那需要把條件放列、種子放欄。使用者 2026-08-08 的
框定是：成敗在圖上很直白，先用眼睛定出真值，再去找與真值一致的量。

本頁只排版既有產物，不做任何計算，也不佔 GPU。

## 為什麼一定要看跨種子

同一個未防禦的攻擊在不同種子上會成功也會失敗（v14r 的 dog_03 實測：seed 0
成功、seed 1 與 3 失敗、seed 2 部分）。**攻擊本身是逐種子的成敗擲骰**，
所以任何只看單一種子的判讀都不成立，而把成敗平均成一個連續量會讓變異來自
兩堆的混合比例而不是防禦。

## τ 的來源

防禦側的編輯輸出在 2026-08-08 之前**不帶 τ**，四個 τ 互相覆寫（見
`executors._run_eval` 的 before/after）。故舊批次每一格只剩一個 τ，而那是
哪一個必須從旁邊的 `metrics_*.json` 讀，不能假設。本頁逐格讀出並印在標頭；
讀不到就標明，不猜。
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import grid                             # noqa: E402

CSS = """
body{font:14px/1.5 system-ui,sans-serif;margin:1.5rem;max-width:120rem}
h1{margin:0 0 .2rem}
table{border-collapse:collapse;margin:.4rem 0 1.2rem}
th,td{border-bottom:1px solid #8884;padding:.2rem;vertical-align:top}
th[scope=row]{text-align:left;white-space:nowrap;font-weight:600;padding-right:.6rem}
img{width:11rem;height:auto;display:block;background:#8881}
td.miss{color:#c33;text-align:center;font-weight:600;font-size:.85em}
tr.ctrl th[scope=row]{color:#06c}
tr.ctrl td{border-bottom:2px solid #06c8}
.lead{background:#8881;padding:.6rem .8rem;border-radius:.3rem;max-width:60rem}
.note{color:#888;font-size:.9em}
section{margin-bottom:2.5rem}
@media(prefers-color-scheme:dark){body{background:#111;color:#ddd}
  .note{color:#999}}
"""


def find_edit(cell_dir: Path, seed: int) -> Optional[Path]:
    """帶 τ 的新檔名優先，取不到再退回舊名。理由同 `compare_page._row_images`。"""
    hits = sorted(cell_dir.glob(f"edit_tau*_seed{seed}.png"))
    if hits:
        return hits[0]
    legacy = cell_dir / f"edit_seed{seed}.png"
    return legacy if legacy.is_file() else None


def tau_of(cell_dir: Path, seed: int) -> Optional[str]:
    """該格的編輯輸出屬於哪一個 τ。由 `metrics_*.json` 讀，不由檔名猜。"""
    for pat in (f"metrics_tau*_seed{seed}.json", f"metrics_seed{seed}.json"):
        for p in sorted(cell_dir.glob(pat)):
            try:
                return f"{float(json.load(p.open(encoding='utf-8'))['tau']):g}"
            except (KeyError, ValueError, OSError):
                return None
    return None


def img_td(p: Optional[Path], out_dir: Path, label: str) -> str:
    if p is None:
        return f'<td class="miss">缺 {html.escape(label)}</td>'
    rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
    return (f'<td><img loading="lazy" src="{html.escape(rel)}" '
            f'alt="{html.escape(label)}"></td>')


def build(batch: Path, images: List[str], conds: List[str], seeds: List[int],
          purify: str, out_path: Path, lead: str) -> str:
    out_dir = out_path.resolve().parent
    parts = [f"<title>編輯成敗判讀 · {html.escape(batch.name)}</title>",
             f"<style>{CSS}</style>",
             f"<h1>編輯成敗判讀 · {html.escape(batch.name)}</h1>",
             f'<p class="lead">{lead}</p>']

    for img in images:
        prompt = None
        rows = []
        for cond in conds:
            sub = "control" if cond == "control" else cond
            cdir = batch / sub / img / "purify" / purify
            taus, tds = set(), []
            for s in seeds:
                p = find_edit(cdir, s)
                tds.append(img_td(p, out_dir, f"{cond} seed{s}"))
                t = tau_of(cdir, s)
                if t:
                    taus.add(t)
                if prompt is None:
                    mj = sorted(cdir.glob(f"metrics_*seed{s}.json"))
                    if mj:
                        try:
                            prompt = json.load(
                                mj[0].open(encoding="utf-8")).get("prompt")
                        except OSError:
                            pass
            tag = "ctrl" if cond == "control" else ""
            label = "對照（未防禦）" if cond == "control" else cond
            tau_note = ("τ=" + "/".join(sorted(taus))) if taus else "τ 不明"
            rows.append(
                f'<tr class="{tag}"><th scope="row">{html.escape(label)}'
                f'<br><span class="note">{html.escape(tau_note)}</span></th>'
                + "".join(tds) + "</tr>")

        orig = batch / conds[1] / img / "orig.png" if len(conds) > 1 else None
        head = (f'<section><h2>{html.escape(img)}'
                + (f' · prompt {html.escape(repr(prompt))}' if prompt else "")
                + "</h2>")
        if orig and orig.is_file():
            rel = os.path.relpath(orig, out_dir).replace(os.sep, "/")
            head += (f'<p><img src="{html.escape(rel)}" alt="原圖" '
                     'style="width:14rem"><span class="note">原圖</span></p>')
        parts.append(head + "<table><tr><th>條件</th>"
                     + "".join(f"<th>seed {s}</th>" for s in seeds)
                     + "</tr>" + "".join(rows) + "</table></section>")
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+",
                    default=["control", *grid.CONDITIONS])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--purify", default="identity_0",
                    help="淨化算子目錄名。identity_0 即未淨化")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lead", default=(
        "<b>先用眼睛定真值：每一格問「編輯成功了嗎」。</b>"
        "第一列是未防禦的對照——攻擊本身就會逐種子成敗不定，"
        "所以要看的是同一列的成敗比例，以及各條件之間的比例差，"
        "不是任何單一格。"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build(args.batch, args.images, args.conditions, args.seeds,
              args.purify, args.out, args.lead), encoding="utf-8")
    print(f"寫入 {args.out}")


if __name__ == "__main__":
    main()
