"""色散度那條軸的報告頁。**純資料呈現，頁上不放任何文字說明。**

版面元件（逐張矩陣、三張表、三張圖）沿用 `mainline_matrix.build_context`，
只換掉標題與模板，並改畫這條線自己的 pipeline：主線那兩張網路圖畫的是紋理
重相位與 DCT-Shield，放在這裡是錯的圖。

用法：
    python scripts/dispersion_report.py --tables runs/ip2p_dispersion/tables \\
        --gallery runs/gallery_dispersion --defense runs/ip2p_dispersion \\
        --out report_dispersion.html
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mainline_architecture as MA  # noqa: E402
import mainline_charts as MC  # noqa: E402
import mainline_matrix as MM  # noqa: E402


def pipeline() -> str:
    """這一批實際跑的那條路徑，一格一個運算。"""
    p = []
    p.append("<text x='20' y='24' class='ttl ours'>"
             "逐頻帶位移（色散變形）</text>")
    p.append(MA.node(20, 52, 92, 50, "原圖 x", "3×512×512"))
    p.append(MA.node(140, 52, 146, 50, "反射填補 ＋ unfold",
                     "32×32, hop 8 → 61×61 視窗"))
    p.append(MA.node(314, 52, 128, 50, "× Hann ＋ rfft2",
                     "複數 3×3721×32×17"))
    p.append(MA.edge([(112, 77), (140, 77)]))
    p.append(MA.edge([(286, 77), (314, 77)]))

    p.append(MA.node(140, 156, 146, 50, "頻帶編號 k(ω)",
                     "log₂ 半徑等分, r ≥ 0.12"))
    p.append(MA.node(140, 232, 146, 50, "粗網格隨機場",
                     "16×16 → 雙三次 → 61×61"))
    p.append(MA.edge([(66, 102), (66, 181), (140, 181)]))
    p.append(MA.edge([(66, 181), (66, 257), (140, 257)]))

    p.append(MA.node(314, 194, 128, 50, "θ = −2π⟨f, u_k(b)⟩",
                     "平移定理，無近似"))
    p.append(MA.edge([(286, 181), (300, 181), (300, 219), (314, 219)]))
    p.append(MA.edge([(286, 257), (300, 257), (300, 219), (314, 219)]))

    p.append(MA.param(508, 219, "K：色散度"))
    p.append(MA.edge([(442, 219), (470, 219)]))

    p.append(MA.op(508, 77, "⊗"))
    p.append(MA.edge([(442, 77), (495, 77)]))
    p.append(MA.edge([(508, 200), (508, 90)], label="exp(iθ)", lx=548, ly=150))

    p.append(MA.node(560, 52, 132, 50, "irfft2 × Hann",
                     "重疊相加 ÷ OLA(w²)"))
    p.append(MA.edge([(521, 77), (560, 77)]))
    p.append(MA.node(720, 52, 104, 50, "防禦圖 x′", "3×512×512", cls="n out"))
    p.append(MA.edge([(692, 77), (720, 77)]))

    p.append(MA.node(720, 156, 104, 62, "淨化算子 T",
                     "JPEG 90/75/50/30", cls="n key"))
    p.append("<text x='772' y='206' class='ns mid'>"
             "模糊 σ1/σ2 · 裁切 10%/15%</text>")
    p.append(MA.edge([(772, 102), (772, 156)]))

    p.append(MA.node(720, 250, 104, 50, "IP2P 編輯", "s_T 7.5 · s_I 1.5",
                     cls="n key2"))
    p.append(MA.edge([(772, 218), (772, 250)]))

    p.append(MA.node(880, 250, 150, 50, "LPIPS(編輯(x), 編輯(T(x′)))",
                     "扣同影像同算子的空白地板", cls="n out"))
    p.append(MA.edge([(824, 275), (880, 275)]))
    p.append(MA.node(880, 156, 150, 50, "DISTS / LPIPS / PSNR",
                     "SSIM · VIFp · L∞ · RMS", cls="n out"))
    p.append(MA.edge([(824, 77), (852, 77), (852, 181), (880, 181)]))

    return ("<svg viewBox='0 0 1060 320' class='arch' role='img'>"
            + "".join(p) + "</svg>")


def phase_axis() -> str:
    """K 這個旋鈕實際改變了什麼：相位對頻率的形狀。"""
    w, h, pad = 1060, 250, 34
    panels = [("K = 1", 1), ("K = 4", 4), ("K = 每個頻格獨立", 0)]
    pw = (w - pad * 2 - 60) / 3
    out = [f"<svg viewBox='0 0 {w} {h}' class='chart' role='img'>"]
    rng = __import__("random").Random(7)
    for i, (title, k) in enumerate(panels):
        x0 = pad + i * (pw + 30)
        y0, y1 = 56, h - 44
        cy = (y0 + y1) / 2
        out.append(f"<text x='{x0}' y='{y0 - 22}' class='ptitle "
                   f"{'ours' if k else 'dct'}'>{title}</text>")
        out.append(f"<line x1='{x0}' y1='{cy}' x2='{x0 + pw}' y2='{cy}' "
                   "class='ax'/>")
        out.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' class='ax'/>")
        pts = []
        n = 96
        for j in range(n + 1):
            f = j / n
            if k == 0:
                v = rng.uniform(-1, 1)
            else:
                band = min(int(f * k), k - 1)
                slope = [0.9, -0.7, 0.55, -0.85][band % 4]
                v = max(-1.0, min(1.0, slope * (f * k - band) * 2 - slope))
            pts.append(f"{x0 + f * pw:.1f},{cy - v * (y1 - y0) / 2 * 0.9:.1f}")
        cls = "ln" if k else "ln dash"
        out.append(f"<polyline points='{' '.join(pts)}' class='{cls}' "
                   f"style='stroke:var(--{'accent' if k else 'alt'})'/>")
        out.append(f"<text x='{x0 + pw / 2}' y='{y1 + 26}' class='axlab mid'>"
                   "空間頻率 →</text>")
    out.append(f"<text x='{pad}' y='{h - 8}' class='axlab'>"
               "縱軸：相位偏移 θ</text>")
    out.append("</svg>")
    return "".join(out)


TEMPLATE = """<title>色散度：從古典位移場到逐頻格相位</title>
<style>
{chart_css}
:root{{--ink:#16121B;--ink-2:#4C4555;--ink-3:#7A7182;--paper:#F2EEF1;
 --surface:#FCFAFB;--line:#DDD4DB;--accent:#A9124F;--alt:#0B6B76;--good:#146B34;
 --thumb:{thumb}px}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
 --ink:#EEE8ED;--ink-2:#B5ABB8;--ink-3:#8B8191;--paper:#121016;--surface:#1C1721;
 --line:#302935;--accent:#F26AA1;--alt:#5FC9D2;--good:#5FCB8C}}}}
:root[data-theme="dark"]{{--ink:#EEE8ED;--ink-2:#B5ABB8;--ink-3:#8B8191;
 --paper:#121016;--surface:#1C1721;--line:#302935;--accent:#F26AA1;
 --alt:#5FC9D2;--good:#5FCB8C}}
{page_css}
</style>
<header><div class="hwrap">
<p class="eyebrow">白盒頻域／相位抗文字編輯防禦 · InstructPix2Pix</p>
<h1>色散度：從古典位移場到逐頻格相位</h1>
<p class="lede mono">10 張 · {ncond} 條件 · {npur} 算子 · 1 種子 ·
DISTS 0.131–0.133</p>
</div></header>

<section class="sec"><h2>Pipeline</h2>{pipeline}</section>
<section class="sec"><h2>K 改變了什麼</h2>{phase_axis}</section>
<section class="sec"><h2>淨增益對壓縮強度</h2>{curve}</section>
<section class="sec"><h2>失真換到多少效果</h2>{bar}</section>
<section class="sec"><h2>各條件的失真高度</h2>{scatter}</section>

<section class="sec">
<h2>逐張影像</h2>
<div class="bar">
<div class="tabs">{tabs}</div>
<div class="ctl"><label for="sz">縮圖</label>
<input id="sz" type="range" min="48" max="{embed}" step="4" value="{thumb}">
<span class="mono" id="szv">{thumb}</span>px
<button id="fit" class="mini">符合畫面</button></div>
<div class="seg"><button id="be" class="on">編輯輸出</button>
<button id="bp">淨化後的防禦圖</button></div>
</div>
{panels}
</section>
<section class="tables">
<h2>防禦圖的失真</h2>
{t1}
<h2>編輯輸出的位移與語意</h2>
{t2}
<h2>抗淨化的淨增益（扣空白地板）</h2>
{t3}
{warn}{dropped}
</section>
{script}
"""


def main() -> None:
    ap = MM.build_parser()
    args = ap.parse_args()
    ctx = MM.build_context(args)
    ctx["pipeline"] = pipeline()
    ctx["phase_axis"] = phase_axis()
    ctx["page_css"] = _slice(MM.TEMPLATE, "*{{box-sizing", "</style>")
    ctx["script"] = _slice(MM.TEMPLATE, "<script>", None)
    MM.write_page(args, TEMPLATE, ctx)


def _slice(src: str, start: str, end: str | None) -> str:
    """從主線模板取一段原樣重用。**找不到就拋錯，不靜默回空字串**——
    版面元件靜默消失的症狀是「頁面看起來怪」，不是錯誤訊息。"""
    i = src.find(start)
    if i < 0:
        raise SystemExit(f"主線模板裡找不到 {start!r}，版面元件無法重用")
    if end is None:
        return src[i:]
    j = src.find(end, i)
    if j < 0:
        raise SystemExit(f"主線模板裡找不到 {end!r}")
    return src[i:j]


if __name__ == "__main__":
    main()
