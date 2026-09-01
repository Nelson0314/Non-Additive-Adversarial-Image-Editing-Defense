"""隨機相位的 retention 為什麼高得不合理——查證頁。

    python scripts/report_phase_rand.py --out reports/phase-rand

問題（使用者 2026-08-18 提出）：隨機相位淨化過後的 `retention` 比值突然跳到
6.15，比任何條件都高。是它真的超級抗淨化，還是圖被改壞了？

本頁不重跑任何實驗，只把既有的四組數字並排：

  1. 比值（現象本身）
  2. 分子——各條件淨化後的**絕對**位移量
  3. 分母——未淨化的位移量與它的 seed 標準差，以及 3σ 閘
  4. 空白地板——把**原圖**直接淨化再編輯，量算子自己造成的位移

第 4 組是決定性的：它回答「那個位移裡有多少根本不是防禦造成的」。

圖表與版面沿用 `report_main.py`，兩頁的樣式必須一致，否則同一個量在兩頁上
長得不一樣會讓人以為是不同的東西。
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_main import (  # noqa: E402
    COLOR, CSS, LABEL, PURIF, PURIFIED, RUN, hbars, img_tag, lines, pur_png,
    raw, rd, stamp_radius, table, fmt,
)

FOCUS = ["phase_rand", "phase", "add"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("reports/phase-rand"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = rd(RUN / "results.csv")
    res: Dict[str, Dict[str, dict]] = {}
    for r in rows:
        res.setdefault(r["condition"], {})[r["image"]] = r
    imgs = sorted({r["image"] for r in rows})
    stamp_radius(res)
    conds = [c for c in ("phase", "phase_rand", "add", "photoguard_c", "mist",
                         "dia_r", "apa_weak") if c in res]

    ret: List[dict] = []
    for f in sorted(RUN.glob("retention_*.csv")):
        ret += rd(f)
    floor: List[dict] = []
    for f in sorted(RUN.glob("floor_*.csv")):
        floor += rd(f)
    if not ret:
        raise SystemExit("找不到 retention_*.csv")

    per: Dict[str, Dict[str, List[float]]] = {}
    rat: Dict[str, Dict[str, List[float]]] = {}
    for r in ret:
        per.setdefault(r["purifier"], {}).setdefault(
            r["condition"], []).append(float(r["effect_mean"]))
        if r["purifier"] != "identity" and r["retention"] not in ("", "nan"):
            rat.setdefault(r["purifier"], {}).setdefault(
                r["condition"], []).append(float(r["retention"]))
    pur = [p for p in PURIF if p in per]
    rpur = [p for p in pur if p != "identity"]
    fl: Dict[str, List[float]] = {}
    for r in floor:
        fl.setdefault(r["purifier"], []).append(float(r["effect_mean"]))

    P: List[str] = []
    A = P.append
    A('<p class="meta">不重跑任何實驗，只把既有的四組數字並排。'
      f'{len(imgs)} 張影像 · {len(rpur)} 個淨化算子 · 每格 3 個編輯 seed。</p>')

    # ---- 1 現象 ----
    A('<h2>1　現象：retention 比值</h2>')
    A('<p class="meta">retention = 淨化後的位移量 ÷ 未淨化的位移量。'
      '隨機相位在 noise0.05 上到 6.15，是紋理重相位的 4.5 倍。</p>')
    A(lines(rpur, [(LABEL[c], COLOR[c],
                    {p: (st.fmean(rat[p][c]) if rat[p].get(c) else None)
                     for p in rpur}) for c in conds], ylabel="retention 比值"))
    A(raw("retention 比值",
          table(["淨化算子"] + [LABEL[c] for c in conds],
                [[p] + [fmt(st.fmean(rat[p][c]) if rat[p].get(c) else None, 2)
                        for c in conds] for p in rpur])))

    # ---- 2 分子 ----
    A('<h2>2　分子：淨化後的絕對位移量</h2>')
    A('<p class="meta">同一批資料，把比值換成分子本身。'
      '如果隨機相位真的超級抗淨化，這條線應該高於其他所有條件。</p>')
    series = [(LABEL[c], COLOR[c],
               {p: (st.fmean(per[p][c]) if per[p].get(c) else None) for p in pur})
              for c in conds]
    if fl:
        series.append(("空白地板（原圖直接淨化）", COLOR["none"],
                       {p: (st.fmean(fl[p]) if p in fl else None) for p in pur}, True))
    A(lines(pur, series, ylabel="位移量 (LPIPS)"))
    A(raw("絕對位移量",
          table(["淨化算子"] + [LABEL[c] for c in conds] + (["空白地板"] if fl else []),
                [[p] + [fmt(st.fmean(per[p][c]) if per[p].get(c) else None)
                        for c in conds]
                 + ([fmt(st.fmean(fl[p]) if p in fl else None)] if fl else [])
                 for p in pur])))

    # ---- 3 分母 ----
    A('<h2>3　分母：未淨化的位移量與 3σ 閘</h2>')
    A('<p class="meta">分母是同一格未淨化時的位移量。它若小到與 seed 之間的'
      '雜訊同量級，比值就沒有意義——`phase_retention.py` 的 3σ 閘'
      '（mean ≥ 3·sd）就是在擋這件事。</p>')
    A(hbars([(LABEL[c], COLOR[c],
              st.fmean([float(x["effect_identity_mean"]) for x in ret
                        if x["condition"] == c and x["purifier"] == "identity"]))
             for c in conds]))
    drows = []
    for c in conds:
        v = [x for x in ret if x["condition"] == c and x["purifier"] == "identity"]
        m = st.fmean(float(x["effect_identity_mean"]) for x in v)
        sd = st.fmean(float(x["effect_identity_sd"]) for x in v)
        bad = len({x["image"] for x in ret
                   if x["condition"] == c and x["usable"].strip().lower() == "false"})
        drows.append([LABEL[c], f"{m:.4f}", f"{sd:.4f}", f"{m / sd:.1f}",
                      f"{bad} / {len(imgs)}"])
    A(table(["方法", "分母（未淨化位移量）", "seed 標準差", "mean / sd",
             "3σ 閘不通過的影像"], drows))

    # ---- 4 地板 ----
    A('<h2>4　空白地板：那個位移有多少不是防禦造成的</h2>')
    if not fl:
        A('<p class="miss" style="aspect-ratio:auto;padding:26px">'
          '尚未產出 floor_*.csv</p>')
    else:
        A('<p class="meta">把<b>原圖</b>直接過同一個淨化算子再編輯——沒有任何防禦。'
          '量到的位移量就是算子自己造成的地板。</p>')
        A(lines(rpur, [("空白地板", COLOR["none"],
                        {p: st.fmean(fl[p]) for p in rpur}, True)] +
                [(LABEL[c], COLOR[c],
                  {p: (st.fmean(per[p][c]) if per[p].get(c) else None)
                   for p in rpur}) for c in FOCUS if c in per[rpur[0]]],
                ylabel="位移量 (LPIPS)"))
        A("<h3>4.1 地板佔比</h3>")
        A('<p class="meta">地板 ÷ 該條件淨化後的位移量。越接近 1 代表那個數字'
          '幾乎全是淨化本身造成的。</p>')
        A(lines(rpur, [(LABEL[c], COLOR[c],
                        {p: (st.fmean(fl[p]) / st.fmean(per[p][c])
                             if per[p].get(c) and st.fmean(per[p][c]) > 0 else None)
                         for p in rpur}) for c in conds], ylabel="地板佔比"))
        A(raw("地板佔比",
              table(["淨化算子", "空白地板"] + [LABEL[c] for c in conds],
                    [[p, fmt(st.fmean(fl[p]))]
                     + [fmt(st.fmean(fl[p]) / st.fmean(per[p][c]), 2)
                        if per[p].get(c) else "—" for c in conds] for p in rpur],
                    right_from=1)))

    # ---- 5 有沒有把圖改壞 ----
    A('<h2>5　隨機相位有沒有把圖改壞</h2>')
    A('<p class="meta">若比值高是因為影像被破壞，它的失真應該最大。</p>')

    def avg(c: str, k: str):
        v = [float(res[c][i][k]) for i in imgs if i in res.get(c, {})]
        return st.fmean(v) if v else None

    A(hbars([(LABEL[c], COLOR[c], avg(c, "fid_lpips")) for c in conds]))
    A(raw("保真度",
          table(["方法", "LPIPS↓", "DISTS↓", "PSNR↑", "SSIM↑"],
                [[LABEL[c], fmt(avg(c, "fid_lpips")), fmt(avg(c, "fid_dists")),
                  fmt(avg(c, "fid_psnr"), 2), fmt(avg(c, "fid_ssim"))]
                 for c in conds])))

    # ---- 6 影像 ----
    A('<h2>6　影像：隨機相位 vs 紋理重相位 vs 加性</h2>')
    A('<p class="meta">上排是淨化後的防禦圖、下排是它再被編輯的結果。'
      '若隨機相位真的抗淨化，它的編輯結果應該與其他兩者明顯不同。</p>')
    focus = [c for c in FOCUS if c in res]
    if not PURIFIED.exists():
        A(f'<p class="miss" style="aspect-ratio:auto;padding:26px">{PURIFIED} 不存在</p>')
    else:
        gpur = [p for p in pur
                if (PURIFIED / f"{imgs[0]}__{focus[0]}__{p}__pur.png").exists()]
        gcols = f'<colgroup><col class="prow"><col span="{len(focus)}"></colgroup>'
        ghead = "".join(f"<th>{LABEL[c]}</th>" for c in focus)
        for im in imgs:
            A(f'<h3>{im}</h3>')
            grows = []
            for pf in gpur:
                cells = []
                for c in focus:
                    stem = f"{im}__{c}__{pf}"
                    cells.append(
                        f'<td>{img_tag(pur_png(stem, "pur"), side=230, quality=80)}'
                        f'{img_tag(pur_png(stem, "edit"), side=230, quality=80)}</td>')
                grows.append(f'<tr><th class="rowh">{pf}</th>{"".join(cells)}</tr>')
            A(f'<div class="tw"><table class="imgs pur">{gcols}'
              f'<tr><th></th>{ghead}</tr>{"".join(grows)}</table></div>')

    html = ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>隨機相位的 retention</title><style>{CSS}</style></head>'
            '<body><main><h1>隨機相位的 retention 為什麼高得不合理</h1>'
            + "".join(P) + "</main></body></html>")
    p = args.out / "index.html"
    p.write_text(html, encoding="utf-8")
    print(f"  {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
