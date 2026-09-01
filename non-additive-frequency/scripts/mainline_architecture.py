"""兩個方法的**網路圖**（訊號流、張量形狀、可學參數、梯度回路），內嵌 SVG。

不是流程圖。每一條線都對應 `src/residual/texture_rephase.py::_rephase` 與
`src/baselines/dct_shield.py` 裡真正的一步，張量形狀取自 `size=512, block=32,
hop=8`（`pad = 16`、`padded = 544`、`side = 65`、`L = 4225` 塊、
`nbins = (32, 17)`）。

畫這張圖要傳達的**唯一一件結構性差異**：

    本方法   擾動長在**連續值影像**上（相位旋轉保幅度、乘性增益、加性下限），
             量化交付是接在最後、**可以拿掉**的一個方塊。
    DCT-Shield  擾動 δ 直接加在**量化後的整數係數**上，量化在參數化**裡面**，
             構造上沒有「不量化」的版本。

所以本方法的圖上有一個**旁路開關**，DCT-Shield 的圖上沒有——那個有無就是
`ours_pg_q` 對 `ours_pg_m` 這一組對照在圖上的樣子。
"""

from __future__ import annotations

import html

# ---- 基本繪圖單元 --------------------------------------------------------


def node(x, y, w, h, title, sub="", cls="n") -> str:
    """一個方塊。`sub` 用 `|` 分行，通常放張量形狀。"""
    t = (f"<text x='{x + w / 2:.0f}' y='{y + (h / 2 - 6 if sub else h / 2 + 4):.0f}'"
         f" class='nt mid'>{html.escape(title)}</text>")
    if sub:
        lines = sub.split("|")
        t += "".join(
            f"<text x='{x + w / 2:.0f}' y='{y + h / 2 + 8 + i * 11:.0f}'"
            f" class='ns mid'>{html.escape(l)}</text>"
            for i, l in enumerate(lines))
    return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='7'"
            f" class='{cls}'/>{t}")


def param(cx, cy, text) -> str:
    """可學參數畫成平行四邊形，與運算方塊區分開。"""
    w, h, sk = 78, 30, 9
    x, y = cx - w / 2, cy - h / 2
    pts = (f"{x + sk},{y} {x + w},{y} {x + w - sk},{y + h} {x},{y + h}")
    return (f"<polygon points='{pts}' class='pm'/>"
            f"<text x='{cx}' y='{cy + 4:.0f}' class='pt mid'>"
            f"{html.escape(text)}</text>")


def op(cx, cy, sym, cls="op") -> str:
    """逐元素運算子（⊗ 乘、⊕ 加、∠ 相位旋轉）。"""
    return (f"<circle cx='{cx}' cy='{cy}' r='13' class='{cls}'/>"
            f"<text x='{cx}' y='{cy + 5:.0f}' class='opt mid'>{sym}</text>")


def edge(pts, cls="e", label="", lx=None, ly=None) -> str:
    """折線邊，`pts` 是 (x, y) 串列；終點自動畫箭頭。"""
    d = " ".join(f"{x:.0f},{y:.0f}" for x, y in pts)
    (x2, y2), (x1, y1) = pts[-1], pts[-2]
    if abs(x2 - x1) > abs(y2 - y1):
        s = 1 if x2 > x1 else -1
        head = f"M{x2 - 7 * s},{y2 - 4.5} L{x2 - 7 * s},{y2 + 4.5} L{x2},{y2} Z"
    else:
        s = 1 if y2 > y1 else -1
        head = f"M{x2 - 4.5},{y2 - 7 * s} L{x2 + 4.5},{y2 - 7 * s} L{x2},{y2} Z"
    out = f"<polyline points='{d}' class='{cls}'/><path d='{head}' class='{cls}h'/>"
    if label:
        out += (f"<text x='{lx}' y='{ly}' class='el mid'>"
                f"{html.escape(label)}</text>")
    return out


# ---- 本方法 --------------------------------------------------------------

def ours() -> str:
    W, H = 1060, 560
    o = [f"<svg viewBox='0 0 {W} {H}' class='arch' role='img'>"]
    o.append("<text x='20' y='24' class='ttl ours'>本方法 · 紋理重相位"
             "（＋可選的量化交付）</text>")

    # 主路徑上排
    o.append(node(20, 60, 92, 50, "原圖 x", "3×512×512"))
    o.append(node(148, 60, 132, 50, "反射填補 ＋ unfold",
                  "32×32, hop 8 → 4225 塊"))
    o.append(node(316, 60, 118, 50, "× Hann ＋ rfft2", "複數 3×4225×32×17"))
    o.append(edge([(112, 85), (148, 85)]))
    o.append(edge([(280, 85), (316, 85)]))

    # 幅度／相位分裂
    o.append(node(478, 26, 104, 40, "幅度 |X|", "不受任何參數影響", "n dim"))
    o.append(node(478, 96, 104, 40, "相位 ∠X", ""))
    o.append(edge([(434, 85), (456, 85), (456, 46), (478, 46)]))
    o.append(edge([(434, 85), (456, 85), (456, 116), (478, 116)]))

    # 閘（由原圖算出並固定）
    o.append(node(148, 214, 132, 46, "結構張量 → 紋理閘", "m_tex：4225（逐塊）"))
    o.append(node(316, 214, 152, 46, "徑向帶通 r ≥ 0.12",
                  "× JPEG 亮度表^0.25 → m_f：32×17"))
    o.append(edge([(66, 110), (66, 237), (148, 237)]))
    o.append(edge([(280, 237), (316, 237)]))
    o.append(op(500, 237, "⊗"))
    o.append(edge([(468, 237), (487, 237)]))
    o.append(edge([(214, 260), (214, 292), (500, 292), (500, 250)]))
    o.append("<text x='214' y='306' class='el'>閘由**原圖**算出後凍結，"
             "不隨防禦圖漂移</text>".replace("**", ""))

    # θ 與旋轉
    o.append(param(500, 168, "θ 可學"))
    o.append(op(608, 116, "∠"))
    o.append(edge([(582, 116), (595, 116)]))
    o.append(edge([(513, 224), (560, 224), (560, 129), (596, 129)],
                  label="θ · m", lx=560, ly=186))
    o.append(edge([(500, 183), (500, 210)], cls="e thin"))

    # 增益
    o.append(param(700, 168, "g 可學"))
    o.append(op(700, 116, "⊗"))
    o.append(edge([(621, 116), (687, 116)], label="exp(i·θm)", lx=654, ly=106))
    o.append(edge([(700, 153), (700, 129)], cls="e thin"))
    o.append("<text x='700' y='198' class='el mid'>exp(g·m)　乘性</text>")

    # 加性下限
    o.append(param(820, 168, "a 可學"))
    o.append(op(820, 116, "⊕"))
    o.append(edge([(713, 116), (807, 116)]))
    o.append(edge([(820, 153), (820, 129)], cls="e thin"))
    o.append("<text x='820' y='198' class='el mid'>a·P·s　加性下限</text>")
    o.append("<text x='820' y='212' class='el mid'>P = 帶通 × 亮度表</text>")

    # 幅度旁路（虛線，強調不變）
    o.append(edge([(582, 46), (900, 46), (900, 96)], cls="e dash",
                  label="幅度原封不動送進重組", lx=740, ly=38))

    # 合成
    o.append(node(858, 96, 84, 40, "重組", "|X|·e^{iφ'}"))
    o.append(edge([(833, 116), (858, 116)]))
    o.append(node(858, 214, 84, 56, "irfft2 → 窗",
                  "→ fold ÷ Σw²|→ clamp[0,1]"))
    o.append(edge([(900, 136), (900, 214)]))

    # 量化交付開關
    o.append(f"<rect x='700' y='330' width='242' height='84' rx='9'"
             " class='sw'/>")
    o.append("<text x='821' y='352' class='swt mid'>量化交付（可拿掉的一步）</text>")
    o.append(node(714, 362, 214, 42, "JPEG 往返 QD = 0.85",
                  "前向：真的 round｜反向：恆等（STE）", "n key"))
    o.append(edge([(900, 270), (900, 330)]))
    o.append(edge([(700, 372), (640, 372), (640, 452), (762, 452)],
                  cls="e dash", label="旁路 = 無量化", lx=640, ly=436))
    o.append(edge([(821, 414), (821, 452)]))
    o.append(node(762, 452, 118, 42, "交付 x′", "3×512×512", "n out"))

    # 損失與回傳
    o.append(node(430, 440, 150, 46, "SD VAE 編碼器 E",
                  "L = ‖E(x′)‖₂（DCT-Shield §4.2）"))
    o.append(edge([(762, 473), (580, 473)]))
    o.append(edge([(430, 452), (300, 452), (300, 168), (462, 168)],
                  cls="e grad", label="sign 梯度上升 → 投影回可行集",
                  lx=300, ly=340))
    o.append("<text x='300' y='356' class='el mid grad'>1000 步，更新 θ / g / a"
             "</text>")
    o.append("</svg>")
    return "".join(o)


# ---- DCT-Shield ----------------------------------------------------------

def dct() -> str:
    W, H = 1060, 300
    o = [f"<svg viewBox='0 0 {W} {H}' class='arch' role='img'>"]
    o.append("<text x='20' y='24' class='ttl dct'>DCT-Shield"
             "（ICCV 2025）· 擾動長在量化整數係數上</text>")

    o.append(node(20, 62, 92, 50, "原圖 x", "3×512×512"))
    o.append(node(148, 62, 140, 50, "RGB→YCbCr ＋ 4:2:0",
                  "色度 2×2 平均池化"))
    o.append(node(324, 62, 118, 50, "8×8 分塊 DCT", "64×64 塊"))
    o.append(node(478, 62, 150, 50, "÷ Q(q_alg) ＋ round",
                  "**量化整數係數 C**".replace("**", ""), "n key2"))
    o.append(edge([(112, 87), (148, 87)]))
    o.append(edge([(288, 87), (324, 87)]))
    o.append(edge([(442, 87), (478, 87)]))

    o.append(op(690, 87, "⊕"))
    o.append(param(690, 158, "δ 可學"))
    o.append(edge([(628, 87), (677, 87)]))
    o.append(edge([(690, 143), (690, 100)], cls="e thin"))
    o.append("<text x='690' y='186' class='el mid'>‖δ‖∞ ≤ ε，**ε ≥ 1**"
             .replace("**", "") + "</text>")
    o.append("<text x='690' y='200' class='el mid'>至少變動一個量化階</text>")
    o.append("<text x='690' y='214' class='el mid'>抗 JPEG 版只動 Y 通道</text>")

    o.append(node(752, 62, 140, 50, "× Q → iDCT", "→ YCbCr→RGB"))
    o.append(edge([(703, 87), (752, 87)]))
    o.append(node(928, 62, 112, 50, "交付 x′", "永遠是量化後的圖", "n out"))
    o.append(edge([(892, 87), (928, 87)]))

    o.append(node(324, 210, 150, 46, "SD VAE 編碼器 E", "L = ‖E(x′)‖₂"))
    o.append(edge([(984, 112), (984, 233), (474, 233)]))
    o.append(edge([(324, 222), (240, 222), (240, 158), (651, 158)],
                  cls="e grad", label="Algorithm 1，1000 步", lx=240, ly=190))

    o.append("<text x='20' y='276' class='cap'>"
             "量化在參數化**裡面**：δ 是加在 round 之後的整數上，所以"
             "**構造上沒有「不量化」的版本**——上圖那個旁路開關在這裡不存在。"
             .replace("**", "") + "</text>")
    o.append("</svg>")
    return "".join(o)


ARCH_CSS = """
.arch{width:100%;height:auto;display:block;overflow:visible}
.arch .n{fill:var(--surface);stroke:var(--line);stroke-width:1.3}
.arch .n.dim{fill:none;stroke-dasharray:4 3;stroke:var(--ink-3)}
.arch .n.key{fill:color-mix(in srgb,var(--accent) 14%,var(--surface));
 stroke:var(--accent);stroke-width:2.2}
.arch .n.key2{fill:color-mix(in srgb,var(--alt) 14%,var(--surface));
 stroke:var(--alt);stroke-width:2.2}
.arch .n.out{fill:color-mix(in srgb,var(--ink) 6%,var(--surface));
 stroke:var(--ink-2);stroke-width:1.6}
.arch .sw{fill:none;stroke:var(--accent);stroke-width:1.2;stroke-dasharray:6 4}
.arch .swt{fill:var(--accent);font:600 11px "IBM Plex Sans",sans-serif}
.arch .nt{fill:var(--ink);font:600 12px "IBM Plex Sans",sans-serif}
.arch .ns{fill:var(--ink-3);font:10.5px ui-monospace,Menlo,monospace}
.arch .pm{fill:color-mix(in srgb,var(--accent) 20%,var(--surface));
 stroke:var(--accent);stroke-width:1.3}
.arch .pt{fill:var(--ink);font:600 11px "IBM Plex Sans",sans-serif}
.arch .op{fill:var(--surface);stroke:var(--ink-2);stroke-width:1.6}
.arch .opt{fill:var(--ink);font:600 14px "IBM Plex Sans",sans-serif}
.arch .e{fill:none;stroke:var(--ink-2);stroke-width:1.5;stroke-linejoin:round}
.arch .eh{fill:var(--ink-2);stroke:none}
.arch .e.thin{stroke-width:1.2;stroke:var(--accent)}
.arch .e.thinh,.arch .thinh{fill:var(--accent)}
.arch .e.dash{stroke-dasharray:6 4;stroke:var(--ink-3)}
.arch .dashh{fill:var(--ink-3)}
.arch .e.grad{stroke:var(--accent);stroke-dasharray:2 3;stroke-width:1.6}
.arch .gradh{fill:var(--accent)}
.arch .el{fill:var(--ink-3);font:10.5px "IBM Plex Sans",sans-serif}
.arch .el.grad{fill:var(--accent)}
.arch .mid{text-anchor:middle}
.arch .ttl{font:600 13.5px "IBM Plex Sans",sans-serif}
.arch .ttl.ours{fill:var(--accent)}.arch .ttl.dct{fill:var(--alt)}
.arch .cap{fill:var(--ink-3);font:11.5px "IBM Plex Sans",sans-serif}
"""
