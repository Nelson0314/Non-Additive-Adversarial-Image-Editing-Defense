#!/usr/bin/env python
"""ip3（inpainting）的實驗報告頁：把全部證據放進一頁，含影像。

    python scripts/report_ip3.py --out docs/report_ip3.html

版面沿用 `v14r 階段一重做 · 實驗報告` 的設計系統（同一組 CSS 變數、明暗雙
主題、serif 標題／mono 數字），使兩份報告可並排閱讀。

影像一律縮到 `--px` 後以 JPEG 內嵌成 data URI：artifact 的上限是 16 MB，而
本批的注意力圖有九千餘張。**選圖的規則寫在各節的 `figcaption` 裡**，全部
原圖仍在 `runs/` 內入版控，頁面標明路徑。

圖表以 inline SVG 產生，不引入繪圖相依。
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ["bird_02", "cat_01", "horse_02"]
CONDS = ["control", "N1", "N2", "N3", "R", "photoguard_c", "mist", "dia_r"]
NONADD = ["N1", "N2", "N3"]
BASE = ["photoguard_c", "mist", "dia_r"]
MAIN_TAU = "0.2"
TAUS = ["0.05", "0.1", "0.2", "0.35"]

LABEL = {
    "control": "未防禦", "N1": "N1 注意力", "N2": "N2 輸出距離",
    "N3": "N3 APA", "R": "R 隨機對照", "photoguard_c": "PhotoGuard-c",
    "mist": "Mist", "dia_r": "DIA-R",
}
KIND = {"control": "neutral", "N1": "ours", "N2": "ours", "N3": "ours",
        "R": "ctrl", "photoguard_c": "base", "mist": "base", "dia_r": "base"}


# ---------------------------------------------------------------------------
# 影像內嵌
# ---------------------------------------------------------------------------

def img_uri(path: Path, px: int = 200, quality: int = 72) -> Optional[str]:
    """縮放並轉成 JPEG 的 data URI。檔案不存在時回 None（缺圖看得出來）。"""
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    if max(im.size) > px:
        im.thumbnail((px, px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fig(path: Path, title: str, sub: str = "", px: int = 200,
        chip: str = "", chip_cls: str = "neutral") -> str:
    uri = img_uri(path, px)
    if uri is None:
        return (f'<figure><div class="missing">缺圖</div>'
                f'<figcaption><span class="t">{title}</span>'
                f'<span class="m">{path.name}</span></figcaption></figure>')
    c = (f'<span class="chip {chip_cls}">{chip}</span>' if chip else "")
    s = f'<span>{sub}</span>' if sub else ""
    return (f'<figure><img alt="{title}" src="{uri}">'
            f'<figcaption><span class="t">{title}</span>{s}{c}</figcaption>'
            f'</figure>')


# ---------------------------------------------------------------------------
# 資料
# ---------------------------------------------------------------------------

def read_csv(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fnum(r: Dict, k: str) -> Optional[float]:
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return None


def margin_rate(rows: List[Dict], tau: str, cond: str,
                img: Optional[str] = None) -> Optional[tuple]:
    """(成功數, 總數)。control 沒有 τ 這個軸，逐 τ 共用。"""
    sel = [r for r in rows
           if r["condition"] == cond and r["purify_dir"] == "identity_0"
           and (img is None or r["image_id"] == img)
           and (cond == "control" or str(r["tau"]) == tau)]
    if not sel:
        return None
    return sum(int(r["success"]) for r in sel), len(sel)


def svg_bars(pairs: Sequence[tuple], baseline: float, w: int = 760,
             h: int = 210, label: str = "") -> str:
    """橫向長條圖。`baseline` 畫一條參考線（未防禦的成功率）。"""
    if not pairs:
        return ""
    pad_l, pad_r, pad_t, pad_b = 118, 46, 14, 26
    iw = w - pad_l - pad_r
    ih = h - pad_t - pad_b
    n = len(pairs)
    bh = ih / n * 0.66
    gap = ih / n
    mx = max(max(v for _, v in pairs), baseline, 0.01) * 1.12
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}">']
    bx = pad_l + iw * baseline / mx
    out.append(f'<line x1="{bx:.1f}" y1="{pad_t}" x2="{bx:.1f}" '
               f'y2="{pad_t + ih}" class="ref"/>')
    out.append(f'<text x="{bx:.1f}" y="{h - 8}" class="reft" '
               f'text-anchor="middle">未防禦 {baseline * 100:.0f}%</text>')
    for i, (name, v) in enumerate(pairs):
        y = pad_t + i * gap + (gap - bh) / 2
        bw = max(iw * v / mx, 1)
        cls = "bar-" + KIND.get(name, "neutral")
        out.append(f'<rect x="{pad_l}" y="{y:.1f}" width="{bw:.1f}" '
                   f'height="{bh:.1f}" class="{cls}"/>')
        out.append(f'<text x="{pad_l - 8}" y="{y + bh * 0.78:.1f}" '
                   f'class="cat" text-anchor="end">{LABEL.get(name, name)}</text>')
        out.append(f'<text x="{pad_l + bw + 6:.1f}" y="{y + bh * 0.78:.1f}" '
                   f'class="val">{v * 100:.0f}%</text>')
    out.append("</svg>")
    return "".join(out)


def svg_lines(series: Dict[str, List[tuple]], w: int = 760, h: int = 240,
              xlab: str = "", ylab: str = "") -> str:
    """折線圖，x 為 τ。"""
    pad_l, pad_r, pad_t, pad_b = 56, 96, 14, 34
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    xs = sorted({x for v in series.values() for x, _ in v})
    ys = [y for v in series.values() for _, y in v]
    if not ys:
        return ""
    lo, hi = min(ys), max(ys)
    if hi == lo:
        hi = lo + 1e-6
    pad = (hi - lo) * 0.1
    lo, hi = lo - pad, hi + pad

    def px(x):
        return pad_l + iw * (xs.index(x) / max(len(xs) - 1, 1))

    def py(y):
        return pad_t + ih * (1 - (y - lo) / (hi - lo))

    out = [f'<svg viewBox="0 0 {w} {h}" role="img">']
    out.append(f'<line x1="{pad_l}" y1="{py(0) if lo < 0 < hi else pad_t + ih:.1f}" '
               f'x2="{pad_l + iw}" y2="{py(0) if lo < 0 < hi else pad_t + ih:.1f}" '
               f'class="ref"/>')
    for x in xs:
        out.append(f'<text x="{px(x):.1f}" y="{h - 10}" class="cat" '
                   f'text-anchor="middle">τ={x}</text>')
    for name, pts in series.items():
        pts = sorted(pts, key=lambda t: xs.index(t[0]))
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
                     for i, (x, y) in enumerate(pts))
        cls = "ln-" + KIND.get(name, "neutral")
        out.append(f'<path d="{d}" class="{cls}"/>')
        for x, y in pts:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.6" '
                       f'class="{cls} dot"/>')
        lx, ly = pts[-1]
        out.append(f'<text x="{px(lx) + 8:.1f}" y="{py(ly) + 4:.1f}" '
                   f'class="lbl {cls}">{LABEL.get(name, name)}</text>')
    out.append(f'<text x="{pad_l}" y="{pad_t + 2}" class="cat">{ylab}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 版面
# ---------------------------------------------------------------------------

CSS = """
:root{
  --ground:#F2F3F5; --surface:#FFFFFF; --surface-2:#EAECEF;
  --ink:#14171C; --ink-soft:#5B6472; --ink-faint:#8A93A0;
  --rule:#D8DCE2; --accent:#0E5A61; --accent-soft:#D6E6E7;
  --bad:#A8231F; --bad-soft:#F6E0DF; --good:#1F7A4C; --good-soft:#DCEDE3;
  --warn:#A5680A; --warn-soft:#F6EBD6;
  --ours:#0E5A61; --base:#A5680A; --ctrl:#5B6472; --neutral:#8A93A0;
  --shadow:0 1px 2px rgba(20,23,28,.06),0 8px 24px rgba(20,23,28,.05);
  --serif:Georgia,"Iowan Old Style","Source Serif 4",Cambria,serif;
  --sans:system-ui,-apple-system,"Segoe UI","Noto Sans TC","PingFang TC",sans-serif;
  --mono:ui-monospace,"Cascadia Mono",Consolas,"SFMono-Regular",monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#12141A; --surface:#191C23; --surface-2:#21252E;
  --ink:#E7EAEF; --ink-soft:#9AA4B2; --ink-faint:#6E7789;
  --rule:#2A2F3A; --accent:#3FB3AE; --accent-soft:#123033;
  --bad:#E37B77; --bad-soft:#33201F; --good:#5CB783; --good-soft:#16281E;
  --warn:#D9AC5C; --warn-soft:#2C2416;
  --ours:#3FB3AE; --base:#D9AC5C; --ctrl:#9AA4B2; --neutral:#6E7789;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
}}
:root[data-theme="dark"]{
  --ground:#12141A; --surface:#191C23; --surface-2:#21252E;
  --ink:#E7EAEF; --ink-soft:#9AA4B2; --ink-faint:#6E7789;
  --rule:#2A2F3A; --accent:#3FB3AE; --accent-soft:#123033;
  --bad:#E37B77; --bad-soft:#33201F; --good:#5CB783; --good-soft:#16281E;
  --warn:#D9AC5C; --warn-soft:#2C2416;
  --ours:#3FB3AE; --base:#D9AC5C; --ctrl:#9AA4B2; --neutral:#6E7789;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:400 16px/1.65 var(--sans);-webkit-font-smoothing:antialiased}
.wrap{max-width:78rem;margin:0 auto;padding:3rem 1.5rem 6rem}
.measure{max-width:42rem}
h1,h2,h3{font-family:var(--serif);text-wrap:balance;margin:0}
h1{font-size:clamp(1.9rem,4.4vw,3rem);line-height:1.12;letter-spacing:-.015em}
h2{font-size:clamp(1.35rem,2.6vw,1.85rem);line-height:1.2}
h3{font-size:1.08rem;line-height:1.3;margin-top:.4rem}
p{margin:0}
.eyebrow{font:600 .72rem/1 var(--sans);letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent)}
section{margin-top:3.5rem;display:flex;flex-direction:column;gap:1.25rem}
.head{display:flex;flex-direction:column;gap:.4rem;
  border-top:2px solid var(--ink);padding-top:.8rem}
.sub{color:var(--ink-soft)}
.lede{font-size:1.12rem;line-height:1.6}
code,.num{font-family:var(--mono);font-size:.92em}
.masthead{display:flex;flex-direction:column;gap:1.25rem;
  padding-bottom:2rem;border-bottom:1px solid var(--rule)}
.meta{display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;
  font:500 .8rem/1.4 var(--mono);color:var(--ink-soft)}
.meta b{color:var(--ink);font-weight:600}
.verdicts{display:grid;gap:.75rem;
  grid-template-columns:repeat(auto-fit,minmax(15rem,1fr))}
.card{background:var(--surface);border:1px solid var(--rule);
  border-radius:.35rem;padding:1rem 1.1rem;box-shadow:var(--shadow);
  display:flex;flex-direction:column;gap:.45rem}
.card .k{font:600 .72rem/1 var(--sans);letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-faint)}
.card .v{font-family:var(--serif);font-size:1.15rem;line-height:1.25}
.card.is-good{border-left:3px solid var(--good)}
.card.is-bad{border-left:3px solid var(--bad)}
.card.is-warn{border-left:3px solid var(--warn)}
.tw{overflow-x:auto;border:1px solid var(--rule);border-radius:.35rem;
  background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
caption{text-align:left;padding:.7rem .9rem;color:var(--ink-soft);
  font-size:.85rem;border-bottom:1px solid var(--rule)}
th,td{padding:.45rem .7rem;text-align:right;white-space:nowrap;
  border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
thead th{background:var(--surface-2);font:600 .78rem/1.3 var(--sans);
  color:var(--ink-soft)}
tbody tr:last-child td{border-bottom:none}
tr.hi td{background:var(--accent-soft)}
td.bad{color:var(--bad);font-weight:600}
td.good{color:var(--good);font-weight:600}
.plate{display:grid;gap:.6rem;
  grid-template-columns:repeat(auto-fit,minmax(9rem,1fr))}
.plate.n4{grid-template-columns:repeat(4,minmax(0,1fr))}
.plate.n5{grid-template-columns:repeat(5,minmax(0,1fr))}
.plate.n3{grid-template-columns:repeat(3,minmax(0,1fr))}
figure{margin:0;display:flex;flex-direction:column;gap:.3rem}
figure img{width:100%;height:auto;display:block;border-radius:.25rem;
  border:1px solid var(--rule);background:var(--surface-2)}
.missing{aspect-ratio:1;display:grid;place-items:center;border-radius:.25rem;
  border:1px dashed var(--rule);color:var(--ink-faint);font-size:.8rem}
figcaption{font:500 .74rem/1.3 var(--sans);color:var(--ink-soft);
  display:flex;flex-direction:column;gap:.1rem}
figcaption .t{color:var(--ink);font-weight:600}
.chip{align-self:flex-start;font:600 .66rem/1 var(--sans);
  padding:.2rem .4rem;border-radius:.2rem}
.chip.ok{background:var(--bad-soft);color:var(--bad)}
.chip.no{background:var(--good-soft);color:var(--good)}
.chip.neutral{background:var(--surface-2);color:var(--ink-soft)}
.m{font-family:var(--mono);font-size:.7rem;color:var(--ink-faint)}
.callout{border-left:3px solid var(--accent);background:var(--surface);
  border-radius:0 .35rem .35rem 0;padding:.9rem 1.1rem;
  display:flex;flex-direction:column;gap:.5rem}
.callout.bad{border-left-color:var(--bad)}
.callout.warn{border-left-color:var(--warn)}
.callout .k{font:600 .72rem/1 var(--sans);letter-spacing:.1em;
  text-transform:uppercase;color:var(--accent)}
.callout.bad .k{color:var(--bad)} .callout.warn .k{color:var(--warn)}
.eq{font-family:var(--mono);font-size:.85rem;line-height:1.7;
  background:var(--surface);border:1px solid var(--rule);border-radius:.35rem;
  padding:.85rem 1rem;overflow-x:auto;white-space:pre;color:var(--ink)}
ul,ol{margin:0;padding-left:1.3rem;display:flex;flex-direction:column;gap:.4rem}
li::marker{color:var(--accent)}
svg{width:100%;height:auto;background:var(--surface);
  border:1px solid var(--rule);border-radius:.35rem}
svg text{font:500 .72rem var(--sans);fill:var(--ink-soft)}
svg text.val{fill:var(--ink);font-weight:600}
svg text.reft{fill:var(--ink-faint);font-size:.66rem}
svg .ref{stroke:var(--ink-faint);stroke-dasharray:3 3;stroke-width:1}
svg rect.bar-ours{fill:var(--ours)} svg rect.bar-base{fill:var(--base)}
svg rect.bar-ctrl{fill:var(--ctrl)} svg rect.bar-neutral{fill:var(--neutral)}
svg path{fill:none;stroke-width:2}
svg .ln-ours{stroke:var(--ours);fill:var(--ours)}
svg .ln-base{stroke:var(--base);fill:var(--base)}
svg .ln-ctrl{stroke:var(--ctrl);fill:var(--ctrl)}
svg .ln-neutral{stroke:var(--neutral);fill:var(--neutral)}
svg text.lbl{font-size:.68rem;font-weight:600}
svg text.lbl.ln-ours{fill:var(--ours)} svg text.lbl.ln-base{fill:var(--base)}
svg text.lbl.ln-ctrl{fill:var(--ctrl)} svg text.lbl.ln-neutral{fill:var(--neutral)}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.tw:focus-visible,.eq:focus-visible{outline-offset:0}
footer{margin-top:4rem;padding-top:1.25rem;border-top:1px solid var(--rule);
  color:var(--ink-faint);font-size:.85rem}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--px", type=int, default=340)
    args = ap.parse_args()

    R = args.runs
    M = R / "ip3_merged"
    grid = read_csv(M / "grid.csv")
    mfull = read_csv(M / "margin_full.csv")
    mmask = read_csv(M / "margin_mask.csv")
    seam = read_csv(M / "seam.csv")
    masks = read_csv(R / "ip3" / "masks" / "masks.csv")
    editable = read_csv(R / "ip3" / "calib" / "editable_filter.csv")
    reach = read_csv(R / "ip3" / "calib" / "warp_reach.csv")
    cov = {r["image_id"]: float(r["coverage"]) for r in masks}

    def shard(img: str) -> Path:
        return R / f"ip3_{img}"

    H: List[str] = []
    A = H.append

    # ---- masthead ----
    A('<div class="wrap"><header class="masthead">')
    A('<p class="eyebrow">批次 ip3 · SD inpainting · 512² · fp32</p>')
    A('<h1>第二階段：威脅模型換成 inpainting，<br>三個處置都成立，主張仍不成立</h1>')
    A('<p class="lede measure">白盒非加性抗文字編輯防禦。本批把攻擊由全圖 '
      'img2img 換成 inpainting，並施行兩個事前查出的處置——保真門檻正比於預算、'
      '位移場在遮罩內歸零。<b>三個處置都按設計生效</b>：約束不再提前綁死訓練、'
      '位移場不再把預算花在會被覆寫的區域、遮罩涵蓋率落在可用範圍。'
      '但用與人眼一致的判定量測，<b>沒有任何條件把編輯成功率壓到未防禦之下</b>，'
      '包含三個加性 baseline。</p>')
    A('<div class="meta">'
      '<span><b>執行</b> 2026-08-08 08:54 – 08-09 03:40</span>'
      f'<span><b>格點</b> {len(grid)} 列，零失敗</span>'
      '<span><b>影像</b> bird_02 · cat_01 · horse_02</span>'
      '<span><b>分支</b> claude/e20-fidelity-constraint</span></div>')
    A('</header>')

    # ---- 結論 ----
    A('<section><div class="head"><p class="eyebrow">摘要</p>'
      '<h2>四項結論</h2></div><div class="verdicts">')
    A('<div class="card is-good"><span class="k">確立</span>'
      '<span class="v">門檻正比於 τ 解開了訓練約束</span>'
      '<span class="sub">六個位移場訓練格的 <code>fid_acut</code> 最大值 '
      '0.032–0.089，門檻 0.16，<b>沒有一格被罰</b>。第一階段是六格精確頂在 '
      '0.0401–0.0404。</span></div>')
    A('<div class="card is-good"><span class="k">確立</span>'
      '<span class="v">遮罩閘沒有讓位移場失去可達性</span>'
      '<span class="sub">加閘後可達 LPIPS 仍為 0.364–0.456，'
      '<code>TAUS</code> 最大的 0.35 可達。</span></div>')
    A('<div class="card is-bad"><span class="k">否定</span>'
      '<span class="v">沒有任何條件把編輯成功率壓到未防禦之下</span>'
      '<span class="sub">遮罩內判定、τ=0.20：未防禦 47%，'
      'N1 40%、N2 53%、N3 30%，三個加性 baseline 全部更高。</span></div>')
    A('<div class="card is-warn"><span class="k">改判</span>'
      '<span class="v">學習率格點的上緣不是瓶頸</span>'
      '<span class="sub">250 步實測：用掉越多失真預算，防禦目標越差。'
      '段 0 選的 0.1 是對的。</span></div>')
    A('</div></section>')

    # ---- 變因 ----
    A('<section><div class="head"><p class="eyebrow">變因</p>'
      '<h2>相對第一階段改了什麼</h2></div>')
    A('<div class="tw"><table><caption>五個變因，與各自的理由</caption>'
      '<thead><tr><th>項目</th><th>第一階段 v14</th><th>本批 ip3</th>'
      '<th style="text-align:left">依據</th></tr></thead><tbody>'
      '<tr><td>威脅模型</td><td>img2img，strength 0.6</td>'
      '<td class="num">inpainting</td>'
      '<td style="text-align:left">strength 是我們自己定的自由度，五篇原始碼都沒有；'
      '換成 inpainting 後該自由度消失，且三篇 baseline 回到原生形態</td></tr>'
      '<tr><td><code>tau_acut</code></td><td class="num">0.04</td>'
      '<td class="num">0.16</td>'
      '<td style="text-align:left">改為 <code>0.8 × τ_train</code>；'
      '0.04 是在 τ=0.05 上由人眼定出的絕對值</td></tr>'
      '<tr><td><code>tau_chroma</code></td><td class="num">0.8</td>'
      '<td class="num">3.2</td>'
      '<td style="text-align:left"><code>16 × τ_train</code></td></tr>'
      '<tr><td>位移場的作用域</td><td>整張圖</td><td>遮罩外</td>'
      '<td style="text-align:left">遮罩內攻擊方會整片覆寫，'
      '擾動零防禦價值卻全額計入保真成本</td></tr>'
      '<tr><td>影像</td><td>bird_03 · cat_02 · dog_03</td>'
      '<td>bird_02 · cat_01 · horse_02</td>'
      '<td style="text-align:left">依遮罩涵蓋率與可編輯性重新選，見下節</td></tr>'
      '</tbody></table></div></section>')

    # ---- 遮罩 ----
    A('<section><div class="head"><p class="eyebrow">前置</p>'
      '<h2>遮罩：三次嘗試才定案</h2></div>')
    A('<p class="measure">inpainting 引入一個 img2img 沒有的變因：攻擊方要重畫'
      '哪一塊。涵蓋率太大則脈絡不足以約束模型（退化回第一階段），太小則編輯'
      '本來就沒改到什麼。遮罩由模型自己對該詞的 cross-attention 產生。</p>')
    A('<div class="plate n3">')
    for im in IMAGES:
        A(fig(M / "masks" / f"{im}_overlay.png",
              f"{im}", f"涵蓋率 {cov.get(im, float('nan')):.3f}", px=300))
    A('</div>')
    A('<p class="sub" style="font-size:.85rem">變暗的矩形是攻擊方要重畫的區域。'
      '純遮罩看不出對位，故一律看疊圖。</p>')
    A('<div class="callout"><span class="k">選圖是規則不是挑選</span>'
      '<p>涵蓋率落在 [0.15, 0.45]、每類取一張，並以疊圖確認框含住物件。'
      '判準必須在批次實際使用的 seed 上量——遮罩由一次加噪前向決定，'
      '<code>dog_02</code> 在 seed 0 與 7 下量到 0.4475 與 0.8750，差一倍。'
      '全 24 張的量測在 <code>runs/ip2_maskprobe3/</code>：'
      '<b>人像 8 張全部 0.92–1.00，整組不可用</b>。</p></div>')

    # 可編輯性
    A('<div class="tw"><table><caption>段 0 對六張候選同時量涵蓋率與可編輯性；'
      '段 1 取通過的三張，每類一張</caption>'
      '<thead><tr><th>影像</th><th>涵蓋率</th><th>可編輯性 SigLIP</th>'
      '<th>通過</th><th>warp_reach</th><th style="text-align:left">採用</th>'
      '</tr></thead><tbody>')
    rmap = {r["image_id"]: float(r["lpips_at_bound"]) for r in reach}
    for r in editable:
        im = r["image_id"]
        eff = float(r["effect_siglip"])
        ok = r["passed"] == "True"
        used = im in IMAGES
        A(f'<tr{" class=hi" if used else ""}><td>{im}</td>'
          f'<td class="num">{float(r.get("coverage", 0) or 0) or cov.get(im, 0):.3f}</td>'
          f'<td class="num {"good" if ok else "bad"}">{eff:+.4f}</td>'
          f'<td>{"✓" if ok else "✗"}</td>'
          f'<td class="num">{rmap.get(im, float("nan")):.3f}</td>'
          f'<td style="text-align:left">{"採用" if used else ""}</td></tr>')
    A('</tbody></table></div></section>')

    # ---- 段 1 訓練 ----
    A('<section><div class="head"><p class="eyebrow">段 1</p>'
      '<h2>訓練：約束確實解開了</h2></div>')
    A('<p class="measure">判準事前宣告：<code>fid_acut</code> 若改頂在 0.16，'
      '代表約束仍綁定、只是位置移了；要看到它明顯低於 0.16 才是真的解開。</p>')
    rows = []
    for im in IMAGES:
        for c in NONADD:
            t = read_csv(shard(im) / c / im / "train.csv")
            if not t:
                continue
            acut = [fnum(r, "fid_acut") for r in t]
            acut = [v for v in acut if v is not None]
            last = t[-1]
            meta = shard(im) / c / im / "meta.json"
            stop = ""
            if meta.exists():
                stop = json.load(meta.open(encoding="utf-8")).get(
                    "stop_reason", "") or "跑滿上限"
            rows.append((im, c, len(t), max(acut) if acut else float("nan"),
                         fnum(last, "fid_lpips"), stop))
    A('<div class="tw"><table><caption>六個位移場格（N1／N2）與三個生成路徑格'
      '（N3）。門檻 <code>tau_acut</code> = 0.16</caption>'
      '<thead><tr><th>影像 · 條件</th><th>步數</th><th>max acut</th>'
      '<th>末端 LPIPS</th><th style="text-align:left">停止原因</th></tr></thead>'
      '<tbody>')
    for im, c, n, a, l, stop in rows:
        cls = "good" if a < 0.16 else "bad"
        A(f'<tr><td>{im} · {c}</td><td class="num">{n}</td>'
          f'<td class="num {cls}">{a:.4f}</td>'
          f'<td class="num">{l:.4f}</td>'
          f'<td style="text-align:left;white-space:normal;font-size:.8rem">'
          f'{stop[:52]}</td></tr>')
    A('</tbody></table></div>')
    A('<h3>訓練曲線</h3>'
      '<p class="sub" style="font-size:.85rem">逐格由 <code>optimize</code> '
      '落盤，每張圖三個條件。橫軸為步數。</p>')
    A('<div class="plate n3">')
    for im in IMAGES:
        for c in NONADD:
            A(fig(shard(im) / c / im / "history.png", f"{im} · {c}",
                  px=520))
    A('</div>')
    A('<div class="callout"><span class="k">解開了，但不是因為訓練到了預算</span>'
      '<p>N1／N2 的末端 LPIPS 是 0.103–0.178，仍未到主表的 0.20，'
      '但<b>原因換了</b>：不再是被 <code>tau_acut</code> 擋下，而是 bird_02 與 '
      'horse_02 跑滿 250 步上限、cat_01 平台停止。四格的末段每步改善量都已低於'
      '平台停止的容差——它們早就收斂了，之所以還跑滿是因為停止準則要求'
      '「約束至少啟動過一次」，而那兩張圖的失真從未碰到任何門檻。</p></div>')
    A('</section>')

    # ---- 對比圖 ----
    A('<section><div class="head"><p class="eyebrow">影像</p>'
      '<h2>主表 τ = 0.20 的逐條件對照</h2></div>')
    A('<p class="measure">每一列一個條件：<b>防禦圖</b>（攻擊方拿到的東西）、'
      '<b>殘差</b>（防禦改了什麼，已放大顯示）、'
      '<b>編輯輸出</b>（攻擊方 inpainting 的結果，identity 未淨化、seed 0）。'
      '第一列是未防禦的對照。</p>')
    for im in IMAGES:
        A(f'<h3>{im}　<span class="m">'
          f'{cov.get(im, 0):.3f} 涵蓋率</span></h3>')
        A('<div class="plate n3">')
        A(fig(shard(im) / "N2" / im / "orig.png", "原圖", "未經任何處理",
              px=args.px))
        A(fig(M / "masks" / f"{im}_overlay.png", "遮罩", "攻擊方要重畫的區域",
              px=args.px))
        A(fig(shard(im) / "control" / im / "purify" / "identity_0"
              / "edit_seed0.png", "未防禦的編輯", "分母", px=args.px))
        A('</div>')
        A('<div class="plate n3" style="margin-top:.6rem">')
        for c in CONDS[1:]:
            d = shard(im) / c / im
            A(fig(d / f"x_def_tau{MAIN_TAU}.png", f"{LABEL[c]} · 防禦圖",
                  px=args.px))
            A(fig(d / f"residual_tau{MAIN_TAU}.png", f"{LABEL[c]} · 殘差",
                  px=args.px))
            A(fig(d / "purify" / "identity_0"
                  / f"edit_tau{MAIN_TAU}_seed0.png",
                  f"{LABEL[c]} · 編輯輸出", px=args.px))
        A('</div>')
    A('<div class="callout warn"><span class="k">N3 有幾格結構上不存在</span>'
      '<p>N3 走生成路徑，<code>x_def</code> 必經 <code>decode(encode(x))</code>，'
      '該來回本身就有一個逐影像的 LPIPS 下限：'
      f'bird_02 <b>0.1853</b>、cat_01 <b>0.0937</b>、horse_02 <b>0.2103</b>。'
      '低於下限的 τ 結構上不可能達成，那些格標為 <code>skipped</code> 而非 '
      '<code>failed</code>。故 horse_02 的 N3 只有 τ=0.35 存在，'
      '主表 τ=0.20 的 N3 是 n=10 而非 15——本頁該處的「缺圖」是這個原因，'
      '不是資料遺失。<b>缺圖看得出來，錯圖看不出來</b>，故不以別的 τ 頂替。</p>'
      '</div>')
    A('</section>')

    # ---- τ 階梯 ----
    A('<section><div class="head"><p class="eyebrow">失真預算</p>'
      '<h2>同一個 φ 沿射線縮放到四個 τ</h2></div>')
    A('<p class="measure">段 2 把訓練出的 φ 沿其方向縮放，使每個條件的失真'
      '精確落在各個 τ 上。這是「匹配失真下比較」的實現方式。</p>')
    for c in ["N2", "photoguard_c"]:
        A(f'<h3>{LABEL[c]}　bird_02</h3><div class="plate n4">')
        for t in TAUS:
            A(fig(shard("bird_02") / c / "bird_02" / f"x_def_tau{t}.png",
                  f"τ = {t}", px=args.px))
        A('</div>')
    A('</section>')

    # ---- attention ----
    A('<section><div class="head"><p class="eyebrow">機制</p>'
      '<h2>Cross-attention：N1 的著力點有沒有動</h2></div>')
    A('<p class="measure">N1 的目標是把注意力質量導向語意無資訊的 token，'
      '使編輯無從落點。下面是攻擊方 inpainting 過程中全部 attn2 層的聚合圖'
      '（τ=0.20、identity、seed 0）。若 N1 生效，它與未防禦的分佈應該不同。</p>')
    for im in IMAGES:
        A(f'<h3>{im}</h3><div class="plate n4">')
        A(fig(shard(im) / "control" / im / "purify" / "identity_0" / "attn"
              / "seed0_agg.png", "未防禦", "分母", px=args.px))
        for c in ["N1", "N2", "photoguard_c"]:
            A(fig(shard(im) / c / im / "purify" / "identity_0" / "attn"
                  / f"tau{MAIN_TAU}_seed0_agg.png", LABEL[c], px=args.px))
        A('</div>')
    A('<p class="sub" style="font-size:.85rem">全部逐層圖（3113 張／分片）在 '
      '<code>runs/ip3_&lt;影像&gt;/&lt;條件&gt;/&lt;影像&gt;/purify/&lt;算子&gt;/attn/</code>，'
      '檔名帶 τ 與 seed。</p>')
    A('</section>')

    # ---- 判定 ----
    A('<section><div class="head"><p class="eyebrow">判定</p>'
      '<h2>類別 margin：編輯成功了沒有</h2></div>')
    A('<div class="eq">margin(y) = SigLIP(y, 目標類) − SigLIP(y, 原類)\n'
      '編輯成功 := margin(y) &gt; 0</div>')
    A('<p class="measure">同一張圖對兩個 prompt，畫質與風格的變化同時影響兩項'
      '而抵消，符號就是判定。校準：三張原圖的 margin 為 −0.0763／−0.0518／'
      '−0.0662，全部正確判成原類，決策邊界在零成立。</p>')
    for tag, rows_, title, note in [
        ("mask", mmask, "遮罩內判定（inpainting 的正確讀出量）",
         "攻擊方只重畫遮罩內，對整張圖算會把訊號稀釋 2.5–3.6 倍。"),
        ("full", mfull, "全圖判定（與文獻的 CLIP 欄同一取法）",
         "保留作為對照：兩種讀出量的差別本身就是結果。")]:
        base = margin_rate(rows_, MAIN_TAU, "control")
        if not base:
            continue
        b = base[0] / base[1]
        pairs = []
        for c in CONDS[1:]:
            v = margin_rate(rows_, MAIN_TAU, c)
            if v:
                pairs.append((c, v[0] / v[1]))
        A(f'<h3>{title}</h3><p class="sub" style="font-size:.85rem">{note}'
          f'　τ=0.20、identity、n={base[1]}（3 影像 × 5 seed）。'
          f'虛線為未防禦的成功率——<b>要低於它才是防禦</b>。</p>')
        A(svg_bars(pairs, b, label=title))
    A('<div class="callout bad"><span class="k">主結果</span>'
      '<p>兩種讀出量給出同一個結論：<b>沒有任何條件把編輯成功率壓到未防禦'
      '之下</b>。n=15 時二項標準誤約 ±13 個百分點，故 47% 對 40% 是 0.5σ、'
      '不可解讀；但 dia_r 的 87% 對 47% 約 3σ，那是真的把攻擊變得更容易成功。'
      '三個加性 baseline 在此判定下全部高於未防禦。</p></div>')

    # 逐 τ 的折線
    series = {}
    for c in CONDS:
        pts = []
        for t in TAUS:
            v = margin_rate(mmask, t, c)
            if v:
                pts.append((t, v[0] / v[1]))
        if pts:
            series[c] = pts
    A('<h3>逐 τ 的編輯成功率（遮罩內判定）</h3>')
    A('<p class="sub" style="font-size:.85rem">未防禦（灰）沒有 τ 這個軸，'
      '在每個 τ 上都是同一個值，作為共用的分母。</p>')
    A(svg_lines(series, ylab="編輯成功率"))
    A('</section>')

    # ---- 三層 ----
    A('<section><div class="head"><p class="eyebrow">判準</p>'
      '<h2>同一批資料，三層判準給出三個結論</h2></div>')
    A('<p class="measure">文獻的判準沒有交集。第 1 層（位移）是與全部 baseline '
      '逐欄對照的唯一途徑；第 2 層（語意）是本專案真正要問的；第 3 層'
      '（感知劣化）的存在是為了讓「靠把輸出弄糟撐起來的免疫」暴露出來。</p>')
    # **全部由 grid.csv 現算**，不抄任何數字進原始碼：抄過來的表在資料變動
    # 之後不會跟著變，而頁面看起來完全正常。
    def gmean(cond: str, col: str) -> Optional[float]:
        v = [fnum(r, col) for r in grid
             if r["condition"] == cond and r["purify_kind"] == "identity"
             and str(r["tau"]).startswith(MAIN_TAU)]
        v = [x for x in v if x is not None]
        return statistics.fmean(v) if v else None

    A('<div class="tw"><table><caption>τ = 0.20、identity，逐格由 '
      '<code>grid.csv</code> 現算。第 1 層的 LPIPS 越高代表防禦把編輯推得'
      '越遠——依這一層，每個條件都「有效」</caption>'
      '<thead><tr><th>條件</th><th>L1 LPIPS↑</th><th>L1 PSNR↓</th>'
      '<th>L1 SSIM↓</th><th>L2 ΔSigLIP↑</th><th>L2 ΔCLIP↑</th>'
      '<th>L3 銳利度比</th><th>margin 成功率</th></tr></thead><tbody>')
    ctl = margin_rate(mmask, MAIN_TAU, "control")
    A(f'<tr><td>未防禦</td><td class="num">—</td><td class="num">—</td>'
      f'<td class="num">—</td><td class="num">—</td><td class="num">—</td>'
      f'<td class="num">—</td>'
      f'<td class="num">{ctl[0]}/{ctl[1]}</td></tr>')
    for c in CONDS[1:]:
        lp, ps, ss = (gmean(c, "edit_lpips"), gmean(c, "edit_psnr"),
                      gmean(c, "edit_ssim"))
        sg, cl = gmean(c, "effect_siglip"), gmean(c, "effect_clip")
        sr = gmean(c, "fid_acutance_ratio")
        if lp is None:
            continue
        v = margin_rate(mmask, MAIN_TAU, c)
        vs = f"{v[0]}/{v[1]}" if v else "—"
        worse = v and (v[0] / v[1]) > (ctl[0] / ctl[1])

        def f4(x, n=4):
            return "—" if x is None else f"{x:.{n}f}"

        A(f'<tr><td>{LABEL[c]}</td><td class="num">{f4(lp)}</td>'
          f'<td class="num">{f4(ps, 2)}</td><td class="num">{f4(ss)}</td>'
          f'<td class="num {"bad" if (sg or 0) < 0 else "good"}">'
          f'{"" if sg is None else f"{sg:+.4f}"}</td>'
          f'<td class="num {"bad" if (cl or 0) < 0 else "good"}">'
          f'{"" if cl is None else f"{cl:+.4f}"}</td>'
          f'<td class="num">{f4(sr)}</td>'
          f'<td class="num {"bad" if worse else ""}">{vs}</td></tr>')
    A('</tbody></table></div>')
    A('<p class="sub" style="font-size:.85rem">第 3 層的 ΔNIQE（編輯輸出側'
      '的無參考品質變化）在 <code>runs/ip3_merged/protocols_mask/protocols.md</code>，'
      '本表列的是防禦側的銳利度比。</p>')
    A('<div class="callout warn"><span class="k">三個相反的讀法</span>'
      '<p>第 1 層：每個條件都把編輯輸出推開 LPIPS 0.16–0.34，依 DAYN／DIA／'
      'PhotoGuard 的欄位全部「有效」。第 2 層的連續量：多數為<b>負</b>，'
      '即防禦後的編輯反而更接近 prompt。第 3 層：ΔNIQE 全部為負，'
      '防禦後的輸出比未防禦的<b>更乾淨</b>——這一批的免疫不是靠劣化撐的，'
      '但也沒有免疫。第 2 層的判定（margin）則說沒有一個條件低於未防禦。</p></div>')
    A('</section>')

    # ---- 接縫 ----
    A('<section><div class="head"><p class="eyebrow">inpainting 專屬</p>'
      '<h2>接縫不連續度</h2></div>')
    A('<div class="eq">ring := 膨脹(mask) ∧ ¬侵蝕(mask)      寬 4 px 的邊界帶\n'
      'seam := mean(|∇I| 帶上) / mean(|∇I| 帶外)</div>')
    A('<p class="measure">inpainting 的結構優勢是「生成內容必須與未遮罩的脈絡'
      '一致」，防禦生效的症狀是接不起來，在邊界上表現為梯度異常升高。'
      '只有「防禦側減對照側」可解讀。</p>')
    sm = defaultdict(list)
    for r in seam:
        if r["purify_dir"] != "identity_0":
            continue
        v = fnum(r, "seam")
        if v is None or v != v:
            continue
        sm[(str(r["tau"]), r["condition"])].append(v)
    ctlseam = {}
    for (t, c), v in sm.items():
        if c == "control":
            ctlseam.setdefault("", []).extend(v)
    cb = statistics.fmean(ctlseam[""]) if ctlseam.get("") else 0.0
    pairs = []
    for c in CONDS[1:]:
        v = sm.get((MAIN_TAU, c))
        if v:
            pairs.append((c, statistics.fmean(v) - cb))
    A('<div class="tw"><table><caption>τ = 0.20、identity。正值＝比未防禦'
      '更接不起來</caption><thead><tr><th>條件</th><th>Δseam</th>'
      '<th>n</th></tr></thead><tbody>')
    for c, d in pairs:
        A(f'<tr><td>{LABEL[c]}</td>'
          f'<td class="num {"good" if d > 0 else "bad"}">{d:+.4f}</td>'
          f'<td class="num">{len(sm[(MAIN_TAU, c)])}</td></tr>')
    A('</tbody></table></div>')
    A('<div class="callout warn"><span class="k">這個量有一個混淆項</span>'
      '<p>分母是邊界帶<b>以外</b>的平均梯度，而編輯輸出在遮罩外就是防禦圖'
      '本身貼回去的。加性擾動是全圖高頻雜訊，會把分母整個抬高、比值被系統性'
      '壓低。三個加性 baseline 的大負值多半來自這個效應而不是「接得更好」。'
      '<b>此量只在同一種擾動族內可比。</b></p></div>')
    A('</section>')

    # ---- 限制 ----
    A('<section><div class="head"><p class="eyebrow">限制</p>'
      '<h2>否證與適用範圍</h2></div>')
    A('<ul class="measure">'
      '<li><b>攻擊本身就很弱。</b> 未防禦的編輯成功率在全圖判定下只有 20%'
      '（3/15）。防禦效果的上限就是攻擊本身的效果量，分母這麼小時任何'
      '「壓低成功率」的空間都極其有限。<code>editable_filter</code> 的門檻是 '
      '0.0，等於只擋掉負效果，沒有擋掉「太弱」。</li>'
      '<li><b>N1 的目標函數在 SD v1.4 上幾乎沒有空間。</b> '
      'shared token 質量起手就是 0.9415，理論上限 1.0，全部可推距離只有 '
      '5.85%；250 步只走掉 0.0014。</li>'
      '<li><b>N3 沒有加閘。</b> 它走生成路徑，擾動經 VAE 解碼後不是逐像素'
      '定域的，無法以同一方式歸零。故 N3 仍把一部分預算花在會被覆寫的區域，'
      '而 N1／N2／R 與兩篇加性 baseline 不會。</li>'
      '<li><b>學習率不是瓶頸。</b> 250 步實測 lr 0.1／0.3／1.0 的末端 '
      '<code>L_def</code> 為 0.79780／0.81473／0.83041，'
      '而末 50 步的步間標準差只有 0.00059。<b>用掉越多失真預算，防禦目標'
      '越差</b>，單調。</li>'
      '<li><b>n = 15。</b> 三張影像 × 五個 seed。二項標準誤 ±13 個百分點，'
      '本頁任何小於該量的差距都不可解讀。</li>'
      '</ul></section>')

    A('<footer><p>證據全部在 <code>runs/ip3</code>、'
      '<code>runs/ip3_&lt;影像&gt;</code>、<code>runs/ip3_merged</code>，'
      '含 2940 列 <code>grid.csv</code>、逐格 <code>compare.html</code>、'
      '三份讀出量 CSV 與三層彙總。本頁由 '
      '<code>scripts/report_ip3.py</code> 產生。</p></footer>')
    A('</div>')

    html = (f"<title>ip3 inpainting · 實驗報告</title>\n<style>{CSS}</style>\n"
            + "\n".join(H))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    kb = len(html.encode("utf-8")) / 1024
    print(f"寫入 {args.out}（{kb:.0f} KB）")
    if kb > 15000:
        print("警告：超過 artifact 的 16 MB 上限，需降低 --px")


if __name__ == "__main__":
    main()
