"""報告頁的圖表與管線圖，全部輸出成**內嵌 SVG**（無外部相依、無 JS 繪圖庫）。

四張圖，各回答一件事：

1. `quality_curve`  淨增益對 JPEG 品質的折線——交叉點落在哪裡。
2. `distortion_bar` 各條件的防禦圖失真長條——比較是不是在同一個高度上做的。
3. `tradeoff`       失真對低品質淨增益的散布——「贏是不是因為失真高」的直答。
4. `pipeline`       四條管線的圖解：我方有／無量化交付、DCT-Shield 原生／抗 JPEG。

為什麼用 SVG 而不是 matplotlib：報告頁是單一 HTML，圖必須內嵌；SVG 是文字、
可被 CSS 主題變數上色，深淺色兩種底色下都讀得到，而點陣圖做不到。
"""

from __future__ import annotations

import html
from typing import Dict, List, Sequence

# 品質軸用「壓縮強度」由弱到強排，與閱讀方向一致（左邊壓得輕、右邊壓得重）。
JPEG_ORDER = ["identity", "jpeg90", "jpeg75", "jpeg50", "jpeg30"]
JPEG_TICK = {"identity": "未淨化", "jpeg90": "90", "jpeg75": "75",
             "jpeg50": "50", "jpeg30": "30"}


def _series_color(tag: str, i: int) -> str:
    """我方一律用主色系、對照組用次色系，同族之間靠深淺分。"""
    ours = ["var(--s1)", "var(--s2)", "var(--s3)"]
    rival = ["var(--r1)", "var(--r2)", "var(--r3)", "var(--r4)"]
    pool = ours if tag.startswith("ours") else rival
    return pool[i % len(pool)]


def _axis(x0, y0, x1, y1) -> str:
    return (f"<line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' class='ax'/>"
            f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' class='ax'/>")


def quality_curve(tags: Sequence[str], label: Dict[str, str],
                  gain: Dict[str, Dict[str, float]]) -> str:
    """淨增益 vs JPEG 品質。**交叉點是這張圖唯一要看的東西。**"""
    W, H = 760, 380
    L, R, T, B = 66, 250, 22, 46
    px = [p for p in JPEG_ORDER if all(p in gain.get(t, {}) for t in tags)]
    if not px or not tags:
        return ""
    ymax = max(gain[t][p] for t in tags for p in px)
    ymax = (int(ymax * 10) + 1) / 10
    xs = [L + i * (W - L - R) / (len(px) - 1) for i in range(len(px))]

    def y(v):
        return T + (H - T - B) * (1 - v / ymax)

    out = [f"<svg viewBox='0 0 {W} {H}' class='chart' role='img'>"]
    for k in range(6):
        v = ymax * k / 5
        out.append(f"<line x1='{L}' y1='{y(v):.1f}' x2='{W - R}' y2='{y(v):.1f}'"
                   " class='grid'/>")
        out.append(f"<text x='{L - 8}' y='{y(v) + 4:.1f}' class='tick end'>"
                   f"{v:.1f}</text>")
    out.append(_axis(L, T, W - R, H - B))
    for i, p in enumerate(px):
        out.append(f"<text x='{xs[i]:.1f}' y='{H - B + 18}' class='tick mid'>"
                   f"{JPEG_TICK.get(p, p)}</text>")
    out.append(f"<text x='{(L + W - R) / 2:.0f}' y='{H - B + 38}' "
               "class='axlab mid'>JPEG 品質（左弱右強）</text>")
    out.append(f"<text x='16' y='16' class='axlab'>淨增益 ↑</text>")

    for i, t in enumerate(tags):
        c = _series_color(t, i)
        pts = " ".join(f"{xs[j]:.1f},{y(gain[t][p]):.1f}" for j, p in enumerate(px))
        out.append(f"<polyline points='{pts}' class='ln' style='stroke:{c}'/>")
        for j, p in enumerate(px):
            out.append(f"<circle cx='{xs[j]:.1f}' cy='{y(gain[t][p]):.1f}' r='3.4'"
                       f" style='fill:{c}'/>")
        ly = T + 14 + i * 19
        out.append(f"<line x1='{W - R + 8}' y1='{ly - 4}' x2='{W - R + 28}'"
                   f" y2='{ly - 4}' class='ln' style='stroke:{c}'/>")
        out.append(f"<text x='{W - R + 34}' y='{ly}' class='leg'>"
                   f"{html.escape(label.get(t, t))}</text>")
    out.append("</svg>")
    return "".join(out)


def distortion_bar(tags: Sequence[str], label: Dict[str, str],
                   fid: Dict[str, dict]) -> str:
    """各條件的 DISTS。等失真對照畫成同一條虛線，一眼看出誰跟誰可比。"""
    rows = [(t, fid[t]["fid_dists"]) for t in tags if t in fid]
    if not rows:
        return ""
    W = 760
    BARH, GAP, L, T = 22, 9, 250, 16
    H = T + len(rows) * (BARH + GAP) + 34
    vmax = max(v for _, v in rows) * 1.12
    out = [f"<svg viewBox='0 0 {W} {H}' class='chart' role='img'>"]
    for k in range(5):
        v = vmax * k / 4
        x = L + (W - L - 20) * v / vmax
        out.append(f"<line x1='{x:.1f}' y1='{T}' x2='{x:.1f}' y2='{H - 30}'"
                   " class='grid'/>")
        out.append(f"<text x='{x:.1f}' y='{H - 14}' class='tick mid'>{v:.2f}</text>")
    for i, (t, v) in enumerate(rows):
        y0 = T + i * (BARH + GAP)
        w = (W - L - 20) * v / vmax
        c = _series_color(t, i if t.startswith("ours") else i - 3)
        out.append(f"<rect x='{L}' y='{y0}' width='{w:.1f}' height='{BARH}'"
                   f" rx='3' style='fill:{c}'/>")
        out.append(f"<text x='{L - 9}' y='{y0 + BARH - 6}' class='tick end'>"
                   f"{html.escape(label.get(t, t))}</text>")
        out.append(f"<text x='{L + w + 7:.1f}' y='{y0 + BARH - 6}' class='val'>"
                   f"{v:.4f}</text>")
    out.append(f"<text x='{(L + W) / 2:.0f}' y='{H - 2}' class='axlab mid'>"
               "DISTS（越低越不明顯）</text>")
    out.append("</svg>")
    return "".join(out)


def tradeoff(tags: Sequence[str], label: Dict[str, str], fid: Dict[str, dict],
             gain: Dict[str, Dict[str, float]], purifier: str = "jpeg30") -> str:
    """失真對低品質淨增益的散布。

    **這張圖就是「贏是不是因為失真高」的直答**：若優勢只是預算，所有點會落在
    同一條上升曲線上；兩族分成兩條曲線才代表機制不同。
    """
    pts = [(t, fid[t]["fid_dists"], gain[t][purifier])
           for t in tags if t in fid and purifier in gain.get(t, {})]
    if len(pts) < 2:
        return ""
    W, H = 760, 380
    L, R, T, B = 66, 250, 22, 46
    xmax = max(p[1] for p in pts) * 1.15
    ymax = max(p[2] for p in pts) * 1.18

    def X(v):
        return L + (W - L - R) * v / xmax

    def Y(v):
        return T + (H - T - B) * (1 - v / ymax)

    out = [f"<svg viewBox='0 0 {W} {H}' class='chart' role='img'>"]
    for k in range(5):
        out.append(f"<line x1='{L}' y1='{Y(ymax * k / 4):.1f}' x2='{W - R}'"
                   f" y2='{Y(ymax * k / 4):.1f}' class='grid'/>")
        out.append(f"<text x='{L - 8}' y='{Y(ymax * k / 4) + 4:.1f}'"
                   f" class='tick end'>{ymax * k / 4:.2f}</text>")
        out.append(f"<text x='{X(xmax * k / 4):.1f}' y='{H - B + 18}'"
                   f" class='tick mid'>{xmax * k / 4:.2f}</text>")
    out.append(_axis(L, T, W - R, H - B))
    # 兩族各連一條線，看的是斜率與截距的差
    for fam, sel in (("ours", lambda t: t.startswith("ours")),
                     ("dct", lambda t: not t.startswith("ours"))):
        fp = sorted((p for p in pts if sel(p[0])), key=lambda p: p[1])
        if len(fp) < 2:
            continue
        c = "var(--s1)" if fam == "ours" else "var(--r1)"
        out.append("<polyline points='"
                   + " ".join(f"{X(x):.1f},{Y(y):.1f}" for _, x, y in fp)
                   + f"' class='ln dash' style='stroke:{c}'/>")
    for i, (t, x, y) in enumerate(pts):
        c = _series_color(t, i)
        out.append(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='5.5'"
                   f" style='fill:{c}'/>")
        ly = T + 14 + i * 19
        out.append(f"<circle cx='{W - R + 16}' cy='{ly - 4}' r='4.5'"
                   f" style='fill:{c}'/>")
        out.append(f"<text x='{W - R + 34}' y='{ly}' class='leg'>"
                   f"{html.escape(label.get(t, t))}</text>")
    out.append(f"<text x='{(L + W - R) / 2:.0f}' y='{H - B + 38}' class='axlab mid'>"
               "防禦圖的 DISTS →</text>")
    out.append(f"<text x='16' y='16' class='axlab'>"
               f"{JPEG_TICK.get(purifier, purifier)} 的淨增益 ↑</text>")
    out.append("</svg>")
    return "".join(out)


# ---- 管線圖 --------------------------------------------------------------
_BOXW, _BOXH, _VGAP, _HGAP = 128, 40, 30, 26


def _box(x, y, text, cls="bx", w=_BOXW, h=_BOXH) -> str:
    lines = text.split("|")
    dy = -(len(lines) - 1) * 7
    tspans = "".join(
        f"<tspan x='{x + w / 2:.0f}' dy='{14 if i else dy + 5}'>{html.escape(l)}"
        "</tspan>" for i, l in enumerate(lines))
    return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='6'"
            f" class='{cls}'/><text x='{x + w / 2:.0f}' y='{y + h / 2:.0f}'"
            f" class='bxt mid'>{tspans}</text>")


def _arrow(x, y0, y1) -> str:
    return (f"<line x1='{x}' y1='{y0}' x2='{x}' y2='{y1 - 7}' class='arw'/>"
            f"<path d='M{x - 4},{y1 - 7} L{x + 4},{y1 - 7} L{x},{y1} Z'"
            " class='arwh'/>")


def pipeline() -> str:
    """四條管線並排。**差別只在被框起來的那一步。**"""
    cols = [
        ("本方法 · 無量化交付", "ours", [
            "原圖 x", "重疊窗化 FFT|32×32, hop 8",
            "相位旋轉 θ|× 幅度增益 g|＋ 加性下限", "iFFT ＋ 重疊相加",
            "**交付連續值影像**", "攻擊方 JPEG 重壓"]),
        ("本方法 · 量化交付", "ours", [
            "原圖 x", "重疊窗化 FFT|32×32, hop 8",
            "相位旋轉 θ|× 幅度增益 g|＋ 加性下限", "iFFT ＋ 重疊相加",
            "**JPEG 往返 QD=0.85**|前向直通估計|交付量化後的圖",
            "攻擊方 JPEG 重壓"]),
        ("DCT-Shield 原生 q0.95", "dct", [
            "原圖 x", "JPEG 編碼 q_alg=0.95|RGB→YCbCr, 8×8 DCT",
            "**量化整數係數**|三個通道", "δ 加在整數係數上|‖δ‖∞ ≤ ε, ε ≥ 1",
            "JPEG 解碼|交付量化後的圖", "攻擊方 JPEG 重壓"]),
        ("DCT-Shield 抗 JPEG q0.85", "dct", [
            "原圖 x", "JPEG 編碼 q_alg=0.85|RGB→YCbCr, 8×8 DCT",
            "**量化整數係數**|僅 Y 通道", "δ 加在整數係數上|‖δ‖∞ ≤ ε, ε ≥ 1",
            "JPEG 解碼|交付量化後的圖", "攻擊方 JPEG 重壓"]),
    ]
    nrow = max(len(c[2]) for c in cols)
    W = len(cols) * _BOXW + (len(cols) + 1) * _HGAP
    H = 46 + nrow * (_BOXH + _VGAP) + 34
    out = [f"<svg viewBox='0 0 {W} {H}' class='chart pipe' role='img'>"]
    for ci, (title, fam, steps) in enumerate(cols):
        x = _HGAP + ci * (_BOXW + _HGAP)
        out.append(f"<text x='{x + _BOXW / 2:.0f}' y='26' class='ptitle mid "
                   f"{fam}'>{html.escape(title)}</text>")
        for si, st in enumerate(steps):
            y = 46 + si * (_BOXH + _VGAP)
            key = st.startswith("**")
            cls = "bx key" if key else "bx"
            if fam == "dct" and key:
                cls = "bx key2"
            out.append(_box(x, y, st.replace("**", ""), cls))
            if si:
                out.append(_arrow(x + _BOXW / 2, y - _VGAP, y))
    out.append(f"<text x='{_HGAP}' y='{H - 10}' class='cap'>"
               "粗框是四條路徑真正不同的地方。DCT-Shield **沒有非量化的版本**"
               "——它的擾動由構造就長在量化整數係數上；本方法的擾動長在連續值"
               "像素上，量化交付是接在後面的一步，可以關掉。</text>")
    out.append("</svg>")
    return "".join(out)


CHART_CSS = """
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--line);stroke-width:1}
.chart .ax{stroke:var(--ink-3);stroke-width:1.2}
.chart .ln{fill:none;stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}
.chart .ln.dash{stroke-width:1.4;stroke-dasharray:5 4;opacity:.6}
.chart .tick{fill:var(--ink-3);font:11px ui-monospace,monospace}
.chart .val{fill:var(--ink-2);font:11px ui-monospace,monospace}
.chart .leg{fill:var(--ink-2);font:12px "IBM Plex Sans",sans-serif}
.chart .axlab{fill:var(--ink-3);font:11.5px "IBM Plex Sans",sans-serif}
.chart .mid{text-anchor:middle}.chart .end{text-anchor:end}
.chart .bx{fill:var(--surface);stroke:var(--line);stroke-width:1.2}
.chart .bx.key{fill:color-mix(in srgb,var(--accent) 12%,var(--surface));
 stroke:var(--accent);stroke-width:2.2}
.chart .bx.key2{fill:color-mix(in srgb,var(--alt) 12%,var(--surface));
 stroke:var(--alt);stroke-width:2.2}
.chart .bxt{fill:var(--ink);font:11px "IBM Plex Sans",sans-serif}
.chart .arw{stroke:var(--ink-3);stroke-width:1.2}
.chart .arwh{fill:var(--ink-3)}
.chart .ptitle{font:600 12.5px "IBM Plex Sans",sans-serif}
.chart .ptitle.ours{fill:var(--accent)}.chart .ptitle.dct{fill:var(--alt)}
.chart .cap{fill:var(--ink-3);font:11.5px "IBM Plex Sans",sans-serif}
.chart.pipe{max-width:720px}
:root{--s1:#A9124F;--s2:#D4497B;--s3:#E88AAE;
 --r1:#0B6B76;--r2:#2E9AA5;--r3:#66C3CB;--r4:#A5DDE2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --s1:#F26AA1;--s2:#D4497B;--s3:#A9124F;
 --r1:#5FC9D2;--r2:#2E9AA5;--r3:#0B6B76;--r4:#075159}}
:root[data-theme="dark"]{--s1:#F26AA1;--s2:#D4497B;--s3:#A9124F;
 --r1:#5FC9D2;--r2:#2E9AA5;--r3:#0B6B76;--r4:#075159}
"""
