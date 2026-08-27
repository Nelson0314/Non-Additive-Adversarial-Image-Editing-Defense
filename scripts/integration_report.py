"""整併版的對比圖與架構圖，輸出單一自足的 HTML。**不跑 GPU。**

為什麼要這一支
────────────────────────────────────────────────────────────────────
「參數住的空間」與「交出去的東西住的空間」不一致這件事，用數字表講不清楚
——它是一張圖的事。本檔把三件事畫在一起：

1. 失真–效果的取捨曲線，四個參數化族放在同一張圖上；
2. 交付要付的那一筆，兩個域各一組；
3. 現行的兩段式管線與整併版的架構圖。

資料來源
────────────────────────────────────────────────────────────────────
`runs/ip2p_mainline/tables/`（相位族與 DCT-Shield，十張）與
`runs/ip2p_dct_nonadd/*/results.csv`、`runs/ip2p_warp_hard/*/results.csv`
（十張）。**位移一律取 `edit_lpips`、失真一律取 `fid_dists`，同一個來源**
——`net_gain.csv` 的 identity 欄是三種子平均，與這裡的單種子讀數不可混用。

擋下數是 CLIP 代理（門檻 0.8445，`docs/EVALUATION.md` 的現行標準），
**金標準仍是人眼**，跨方法的擋下率一律人眼定案。

輸出
────────────────────────────────────────────────────────────────────
`report_integration.html`（專案根目錄，不入版控）。圖全部是內嵌 SVG，
沒有外部資源。

用法：python scripts/integration_report.py
"""

from __future__ import annotations

import csv
import glob
import os
import statistics as st
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
BAND = (0.1286, 0.1447)          # 失真帶，docs/PENDING.md


# ---------------------------------------------------------------------------
# 資料
# ---------------------------------------------------------------------------

def load_points() -> Dict[str, dict]:
    fd = {r["tag"]: r for r in csv.DictReader(
        open(ROOT / "runs/ip2p_mainline/tables/defense_fidelity.csv", encoding="utf-8"))}
    ed = {r["tag"]: r for r in csv.DictReader(
        open(ROOT / "runs/ip2p_mainline/tables/edit_displacement.csv", encoding="utf-8"))}
    out: Dict[str, dict] = {}
    for t in ed:
        out[t] = dict(dists=float(fd[t]["fid_dists"]), psnr=float(fd[t]["fid_psnr"]),
                      linf=float(fd[t]["fid_linf"]), disp=float(ed[t]["edit_lpips"]),
                      blocked=None)
    for p in (sorted(glob.glob(str(ROOT / "runs/ip2p_dct_nonadd/*/results.csv")))
              + sorted(glob.glob(str(ROOT / "runs/ip2p_warp_hard/*/results.csv")))
              + sorted(glob.glob(str(ROOT / "runs/ip2p_dct_unified/*/results.csv")))):
        tag = os.path.basename(os.path.dirname(p))
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        if not rows:
            continue

        def mean(key: str) -> float:
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            return round(st.mean(vals), 5) if vals else float("nan")

        clip = [float(r["edit_clip_sim"]) for r in rows if r.get("edit_clip_sim")]
        out[tag] = dict(dists=mean("fid_dists"), psnr=mean("fid_psnr"),
                        linf=mean("fid_linf"), disp=mean("edit_lpips"),
                        blocked=sum(1 for c in clip if c < 0.8445), n=len(rows))
    return out


def interpolate(pts: List[Tuple[float, float]], x: float):
    """在取捨曲線上線性內插。**落在範圍外回傳 None，不外插**（DEC-029）。"""
    pts = sorted(pts)
    if x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


# ---------------------------------------------------------------------------
# 繪圖：極小的 SVG 工具，不用任何外部函式庫
# ---------------------------------------------------------------------------

W, H = 760, 440
PAD_L, PAD_R, PAD_T, PAD_B = 66, 24, 26, 52


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Axes:
    def __init__(self, xlim, ylim, xlabel, ylabel, w=W, h=H):
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        self.w, self.h = w, h
        self.parts: List[str] = []
        self.xlabel, self.ylabel = xlabel, ylabel

    def px(self, x: float) -> float:
        return PAD_L + (x - self.x0) / (self.x1 - self.x0) * (self.w - PAD_L - PAD_R)

    def py(self, y: float) -> float:
        return self.h - PAD_B - (y - self.y0) / (self.y1 - self.y0) * (self.h - PAD_T - PAD_B)

    def band(self, lo: float, hi: float, label: str):
        x, w = self.px(lo), self.px(hi) - self.px(lo)
        self.parts.append(
            f'<rect x="{x:.1f}" y="{PAD_T}" width="{w:.1f}" '
            f'height="{self.h - PAD_T - PAD_B:.1f}" class="band"/>'
            f'<text x="{x + w / 2:.1f}" y="{PAD_T + 14}" class="bandlab" '
            f'text-anchor="middle">{_esc(label)}</text>')

    def grid(self, xticks, yticks, xfmt="{:.2f}", yfmt="{:.2f}"):
        for t in xticks:
            x = self.px(t)
            self.parts.append(
                f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" '
                f'y2="{self.h - PAD_B}" class="grid"/>'
                f'<text x="{x:.1f}" y="{self.h - PAD_B + 18}" class="tick" '
                f'text-anchor="middle">{xfmt.format(t)}</text>')
        for t in yticks:
            y = self.py(t)
            self.parts.append(
                f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{self.w - PAD_R}" '
                f'y2="{y:.1f}" class="grid"/>'
                f'<text x="{PAD_L - 10}" y="{y + 4:.1f}" class="tick" '
                f'text-anchor="end">{yfmt.format(t)}</text>')
        self.parts.append(
            f'<text x="{(PAD_L + self.w - PAD_R) / 2:.1f}" y="{self.h - 12}" '
            f'class="axlab" text-anchor="middle">{_esc(self.xlabel)}</text>'
            f'<text x="16" y="{(PAD_T + self.h - PAD_B) / 2:.1f}" class="axlab" '
            f'text-anchor="middle" transform="rotate(-90 16 '
            f'{(PAD_T + self.h - PAD_B) / 2:.1f})">{_esc(self.ylabel)}</text>')

    def series(self, pts, cls: str, label: str, marker: str = "circle",
               labels: List[str] = None):
        pts = sorted(pts)
        if len(pts) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{self.px(x):.1f},{self.py(y):.1f}"
                         for i, (x, y) in enumerate(pts))
            self.parts.append(f'<path d="{d}" class="line {cls}"/>')
        for i, (x, y) in enumerate(pts):
            cx, cy = self.px(x), self.py(y)
            if marker == "square":
                self.parts.append(
                    f'<rect x="{cx - 5:.1f}" y="{cy - 5:.1f}" width="10" height="10" '
                    f'class="mark {cls}"/>')
            elif marker == "diamond":
                self.parts.append(
                    f'<path d="M{cx:.1f},{cy - 6.5:.1f}L{cx + 6.5:.1f},{cy:.1f}'
                    f'L{cx:.1f},{cy + 6.5:.1f}L{cx - 6.5:.1f},{cy:.1f}Z" '
                    f'class="mark {cls}"/>')
            else:
                self.parts.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" class="mark {cls}"/>')
            if labels and labels[i]:
                self.parts.append(
                    f'<text x="{cx + 9:.1f}" y="{cy - 8:.1f}" class="ptlab">'
                    f'{_esc(labels[i])}</text>')

    def annotate(self, x, y, text, dx=0, dy=0, anchor="start"):
        self.parts.append(
            f'<text x="{self.px(x) + dx:.1f}" y="{self.py(y) + dy:.1f}" '
            f'class="note" text-anchor="{anchor}">{_esc(text)}</text>')

    def arrow(self, x0, y0, x1, y1, cls="arrow"):
        self.parts.append(
            f'<line x1="{self.px(x0):.1f}" y1="{self.py(y0):.1f}" '
            f'x2="{self.px(x1):.1f}" y2="{self.py(y1):.1f}" class="{cls}" '
            f'marker-end="url(#ah)"/>')

    def svg(self) -> str:
        return (f'<svg viewBox="0 0 {self.w} {self.h}" class="chart" '
                f'role="img"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" '
                f'refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                f'<path d="M0,0L10,5L0,10z" class="ahead"/></marker></defs>'
                + "".join(self.parts) + "</svg>")


def legend(items: List[Tuple[str, str]]) -> str:
    out = ['<div class="legend">']
    for cls, text in items:
        out.append(f'<span class="lg"><i class="sw {cls}"></i>{_esc(text)}</span>')
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 三張圖
# ---------------------------------------------------------------------------

def chart_tradeoff(p: Dict[str, dict]) -> str:
    ax = Axes((0.0, 0.30), (0.0, 0.78), "失真 DISTS（越低越好）",
              "未淨化位移（越高越好）")
    ax.band(*BAND, "失真帶")
    ax.grid([0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    phase = [(p[t]["dists"], p[t]["disp"]) for t in ("ours_ph_n", "ours_pg_n", "ours_pg_m")]
    ax.series(phase, "c-phase", "相位族", labels=["r0.9", "", "r2.0"])

    phase_q = [(p[t]["dists"], p[t]["disp"]) for t in ("ours_ph_q", "ours_pg_q")]
    ax.series(phase_q, "c-phaseq", "相位族＋量化交付", marker="square",
              labels=["", "量化交付"])

    plane = [(p[t]["dists"], p[t]["disp"])
             for t in ("nd_plane_12", "nd_plane_18", "nd_plane_25", "nd_plane_31")]
    ax.series(plane, "c-plane", "DCT 學習平面", labels=["", "", "", "θ=π"])

    ax.series([(p["nd_plane_25_qd"]["dists"], p["nd_plane_25_qd"]["disp"])],
              "c-planeq", "DCT 學習平面＋事後投影交付", marker="square",
              labels=["事後投影交付"])
    ax.arrow(p["nd_plane_31"]["dists"], p["nd_plane_31"]["disp"],
             p["nd_plane_25_qd"]["dists"], p["nd_plane_25_qd"]["disp"])

    gain = [(p[t]["dists"], p[t]["disp"]) for t in ("nd_gain_11", "nd_gain_16")]
    ax.series(gain, "c-gain", "DCT 非正交乘性", marker="diamond",
              labels=["", "非正交"])

    warp = [(p[t]["dists"], p[t]["disp"]) for t in ("w_adam_rand", "w_adam_g64")]
    ax.series(warp, "c-warp", "位移場", marker="diamond",
              labels=["grid 16", "grid 64"])

    ax.series([(p["dct_aj85"]["dists"], p["dct_aj85"]["disp"])], "c-dct",
              "DCT-Shield", marker="square", labels=["DCT-Shield"])
    ax.series([(p["nd_rand_25"]["dists"], p["nd_rand_25"]["disp"])], "c-rand",
              "隨機對照", labels=["隨機"])
    return ax.svg() + legend([
        ("c-phase", "相位族（重疊 STFT，不交付）"),
        ("c-phaseq", "相位族 ＋ 量化交付"),
        ("c-plane", "DCT 學習平面（浮點係數，不交付）"),
        ("c-planeq", "DCT 學習平面 ＋ 事後投影交付"),
        ("c-gain", "DCT 非正交乘性（打它用的對照）"),
        ("c-warp", "位移場硬訓練"),
        ("c-dct", "DCT-Shield（論文設定）"),
        ("c-rand", "同角度隨機解"),
    ])


def chart_delivery_cost(p: Dict[str, dict]) -> str:
    """交付要付的那一筆，兩個域各一組。**兩組的比法不同，圖上要標明。**"""
    w, h = 760, 330
    ax = Axes((0, 4), (0, 0.80), "", "未淨化位移", w=w, h=h)
    ax.grid([], [0.2, 0.4, 0.6, 0.8])
    bars = [
        (0.55, 0.7012, "c-phase", "相位族", "不交付"),
        (1.25, 0.5561, "c-phaseq", "相位族", "量化交付"),
        (2.35, p["nd_plane_31"]["disp"], "c-plane", "DCT 學習平面", "不交付"),
        (3.05, p["nd_plane_25_qd"]["disp"], "c-planeq", "DCT 學習平面", "事後投影"),
    ]
    for x, y, cls, top, bot in bars:
        x0, x1 = ax.px(x - 0.28), ax.px(x + 0.28)
        y1 = ax.py(y)
        ax.parts.append(
            f'<rect x="{x0:.1f}" y="{y1:.1f}" width="{x1 - x0:.1f}" '
            f'height="{ax.py(0) - y1:.1f}" class="bar {cls}"/>'
            f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 - 8:.1f}" class="barval" '
            f'text-anchor="middle">{y:.4f}</text>'
            f'<text x="{(x0 + x1) / 2:.1f}" y="{ax.py(0) + 20:.1f}" class="tick" '
            f'text-anchor="middle">{_esc(bot)}</text>')
    ax.parts.append(
        f'<text x="{ax.px(0.9):.1f}" y="{ax.py(0) + 42:.1f}" class="axlab" '
        f'text-anchor="middle">相位族（重疊 STFT）</text>'
        f'<text x="{ax.px(2.7):.1f}" y="{ax.py(0) + 42:.1f}" class="axlab" '
        f'text-anchor="middle">DCT 學習平面</text>')
    ax.annotate(0.9, 0.745, "等失真內插：−21%", anchor="middle")
    ax.annotate(2.7, 0.745, "兩軸皆被支配（不需內插）", anchor="middle")
    return ax.svg()


def chart_length_preserving(p: Dict[str, dict]) -> str:
    """保長 vs 非正交，兩個失真錨點。內插的那一側標明是內插值。"""
    plane = [(p[t]["dists"], p[t]["disp"])
             for t in ("nd_plane_12", "nd_plane_18", "nd_plane_25", "nd_plane_31")]
    rows = []
    for tag in ("nd_gain_11", "nd_gain_16"):
        d = p[tag]["dists"]
        got = interpolate(plane, d)
        rows.append((d, p[tag]["disp"], got))
    w, h = 760, 300
    ax = Axes((0, 2), (0, 0.62), "", "未淨化位移", w=w, h=h)
    ax.grid([], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    for i, (d, g, pl) in enumerate(rows):
        base = 0.5 + i
        for j, (val, cls, name) in enumerate(
                ((g, "c-gain", "非正交乘性"), (pl, "c-plane", "保長旋轉（內插）"))):
            x = base + (j - 0.5) * 0.34
            x0, x1 = ax.px(x - 0.15), ax.px(x + 0.15)
            y1 = ax.py(val)
            ax.parts.append(
                f'<rect x="{x0:.1f}" y="{y1:.1f}" width="{x1 - x0:.1f}" '
                f'height="{ax.py(0) - y1:.1f}" class="bar {cls}"/>'
                f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 - 8:.1f}" class="barval" '
                f'text-anchor="middle">{val:.4f}</text>'
                f'<text x="{(x0 + x1) / 2:.1f}" y="{ax.py(0) + 20:.1f}" '
                f'class="tick" text-anchor="middle">{_esc(name)}</text>')
        ax.parts.append(
            f'<text x="{ax.px(base):.1f}" y="{ax.py(0) + 42:.1f}" class="axlab" '
            f'text-anchor="middle">等失真 DISTS {d:.4f}　→　保長是 '
            f'{pl / g:.2f} 倍</text>')
    return ax.svg()


# ---------------------------------------------------------------------------
# 架構圖
# ---------------------------------------------------------------------------

def _box(x, y, w, h, title, sub, cls=""):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" class="ab {cls}"/>',
           f'<text x="{x + w / 2}" y="{y + (20 if sub else h / 2 + 5)}" '
           f'class="abt" text-anchor="middle">{_esc(title)}</text>']
    if sub:
        for i, line in enumerate(sub.split("\n")):
            out.append(f'<text x="{x + w / 2}" y="{y + 38 + i * 14}" class="abs" '
                       f'text-anchor="middle">{_esc(line)}</text>')
    return "".join(out)


def _arrow(x0, y, x1, cls="aflow"):
    return (f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="{cls}" '
            f'marker-end="url(#ah2)"/>')


def architecture() -> str:
    w, h = 980, 470
    p = [f'<svg viewBox="0 0 {w} {h}" class="chart arch" role="img">'
         '<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" '
         'markerWidth="7" markerHeight="7" orient="auto">'
         '<path d="M0,0L10,5L0,10z" class="ahead"/></marker></defs>']

    p.append('<text x="14" y="24" class="archh">現行：兩段拼接</text>')
    y = 44
    p.append(_box(14, y, 96, 66, "原圖", ""))
    p.append(_arrow(112, y + 33, 140))
    p.append(_box(142, y, 168, 66, "切塊 · 加窗 · FFT",
                  "32×32、hop 8\n每像素被 16 塊蓋到", "amine"))
    p.append(_arrow(312, y + 33, 340))
    p.append(_box(342, y, 168, 66, "轉相位 · 乘增益",
                  "參數住在這裡\n約 59 萬個", "aparam"))
    p.append(_arrow(512, y + 33, 540))
    p.append(_box(542, y, 150, 66, "重疊相加", "連續值影像", "amine"))
    p.append(_arrow(694, y + 33, 722))
    p.append(_box(724, y, 150, 66, "JPEG 壓縮", "交付格式在這裡", "adeliver"))
    p.append(_arrow(876, y + 33, 904))
    p.append(_box(906, y, 60, 66, "交付", ""))
    p.append(f'<path d="M426,{y + 70}L426,{y + 96}L799,{y + 96}L799,{y + 70}" '
             f'class="agap"/>')
    p.append(f'<text x="612" y="{y + 116}" class="agapt" text-anchor="middle">'
             '參數住的空間 ≠ 交出去的東西住的空間 → 交付時要投影一次</text>')
    p.append(f'<text x="612" y="{y + 136}" class="agapt bad" text-anchor="middle">'
             '實測代價：等失真下未淨化位移 0.7012 → 0.5561（−21%）</text>')

    p.append('<text x="14" y="252" class="archh">整併：一個模組、一個空間</text>')
    y = 272
    p.append(_box(14, y, 96, 66, "原圖", ""))
    p.append(_arrow(112, y + 33, 140))
    p.append(_box(142, y, 200, 66, "JPEG 編碼",
                  "→ 量化後的整數係數\n8×8、4:2:0", "amine"))
    p.append(_arrow(344, y + 33, 372))
    p.append(_box(374, y, 236, 66, "學平面 · 旋轉 · 取整",
                  "參數住在這裡＝交付格式\n約 62 萬個", "aboth"))
    p.append(_arrow(612, y + 33, 640))
    p.append(_box(642, y, 200, 66, "JPEG 解碼", "交出去的就是這組整數", "adeliver"))
    p.append(_arrow(844, y + 33, 872))
    p.append(_box(874, y, 60, 66, "交付", ""))
    p.append(f'<text x="474" y="{y + 96}" class="agapt good" text-anchor="middle">'
             '沒有事後投影——那 21% 不該存在。這是可證偽的預測，不是修辭。</text>')
    p.append(f'<text x="474" y="{y + 116}" class="agapt" text-anchor="middle">'
             'θ = 0 時輸出等於壓縮圖，不是原圖；旋轉之後必須再取整，'
             '故動作仍在 DCT-Shield 的空間裡。</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------------------
# 頁面
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b66;--line:#e3e3df;--card:#fff;
--band:rgba(120,140,200,.13);--phase:#2f6f4f;--phaseq:#69a37f;--plane:#b4531f;
--planeq:#dd9a5e;--gain:#8a6fb0;--warp:#9a9a92;--dct:#2b5d9b;--rand:#b9b9b2;
--good:#1f6f45;--bad:#a33a2a;}
@media(prefers-color-scheme:dark){:root{--bg:#171716;--fg:#eceae4;--mut:#a2a099;
--line:#33322f;--card:#1f1f1d;--band:rgba(140,160,220,.16);--phase:#6cc196;
--phaseq:#3f8a63;--plane:#e08a4e;--planeq:#a56338;--gain:#b39ad8;
--warp:#7a7a74;--dct:#6ba3e8;--rand:#5c5c57;--good:#6cc196;--bad:#e0745e;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.75 -apple-system,"Segoe UI","Noto Sans TC","PingFang TC",
"Microsoft JhengHei",sans-serif;}
main{max-width:1040px;margin:0 auto;padding:44px 22px 96px}
h1{font-size:29px;line-height:1.3;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:21px;margin:52px 0 12px;padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:28px 0 8px}
p{margin:12px 0}
.lede{color:var(--mut);margin:0 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px 18px 12px;margin:20px 0}
.chart{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}
.band{fill:var(--band)}
.bandlab{fill:var(--mut);font-size:11px}
.tick{fill:var(--mut);font-size:11px}
.axlab{fill:var(--fg);font-size:12px}
.ptlab{fill:var(--mut);font-size:11px}
.note{fill:var(--mut);font-size:12px}
.barval{fill:var(--fg);font-size:12px;font-weight:600}
.line{fill:none;stroke-width:2}
.mark{stroke-width:0}
.arrow{stroke:var(--bad);stroke-width:1.6;stroke-dasharray:4 3}
.ahead{fill:var(--bad)}
.bar{stroke:none}
.c-phase{stroke:var(--phase);fill:var(--phase)}
.c-phaseq{stroke:var(--phaseq);fill:var(--phaseq)}
.c-plane{stroke:var(--plane);fill:var(--plane)}
.c-planeq{stroke:var(--planeq);fill:var(--planeq)}
.c-gain{stroke:var(--gain);fill:var(--gain)}
.c-warp{stroke:var(--warp);fill:var(--warp);stroke-dasharray:5 4}
.c-dct{stroke:var(--dct);fill:var(--dct)}
.c-rand{stroke:var(--rand);fill:var(--rand)}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;margin:10px 2px 4px;
font-size:13px;color:var(--mut)}
.lg{display:flex;align-items:center;gap:7px}
.sw{width:13px;height:13px;border-radius:3px;display:inline-block}
table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}
th,td{border-bottom:1px solid var(--line);padding:8px 9px;text-align:left;
vertical-align:top}
th{font-weight:600;color:var(--mut);font-size:12.5px}
td.num{font-variant-numeric:tabular-nums;text-align:right}
.win{color:var(--good);font-weight:600}
.lose{color:var(--bad);font-weight:600}
.arch .ab{fill:var(--card);stroke:var(--line);stroke-width:1.5}
.arch .amine{stroke:var(--phase)}
.arch .aparam{stroke:var(--plane);stroke-width:2.5}
.arch .adeliver{stroke:var(--dct);stroke-width:2.5}
.arch .aboth{stroke:var(--good);stroke-width:3}
.arch .abt{fill:var(--fg);font-size:13.5px;font-weight:600}
.arch .abs{fill:var(--mut);font-size:11.5px}
.arch .archh{fill:var(--fg);font-size:15px;font-weight:700}
.arch .aflow{stroke:var(--mut);stroke-width:1.6}
.arch .agap{fill:none;stroke:var(--bad);stroke-width:1.6;stroke-dasharray:5 4}
.arch .agapt{fill:var(--mut);font-size:12.5px}
.arch .agapt.bad{fill:var(--bad)}
.arch .agapt.good{fill:var(--good)}
.foot{color:var(--mut);font-size:13px;margin-top:44px}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;
background:var(--band);padding:1px 5px;border-radius:4px}
"""


def page(p: Dict[str, dict]) -> str:
    plane = [(p[t]["dists"], p[t]["disp"])
             for t in ("nd_plane_12", "nd_plane_18", "nd_plane_25", "nd_plane_31")]
    g11 = interpolate(plane, p["nd_gain_11"]["dists"])
    g16 = interpolate(plane, p["nd_gain_16"]["dists"])

    def row(tag, label, note=""):
        d = p[tag]
        blocked = "—" if d.get("blocked") is None else f'{d["blocked"]}/10'
        return (f"<tr><td>{_esc(label)}</td>"
                f'<td class="num">{d["dists"]:.4f}</td>'
                f'<td class="num">{d["psnr"]:.2f}</td>'
                f'<td class="num">{d["linf"]:.4f}</td>'
                f'<td class="num">{d["disp"]:.4f}</td>'
                f'<td class="num">{blocked}</td>'
                f"<td>{_esc(note)}</td></tr>")

    unified = sorted(t for t in p if t.startswith("du_"))
    if unified:
        urows = "".join(row(t, t, "整併版") for t in unified)
        ustatus = ("<p>整併版的第一批數字已到，列在下表最後幾列。"
                   "判準 U1 的比較對象是 <code>nd_plane_25_qd</code>。</p>")
    else:
        urows = ""
        ustatus = ('<p>整併版的四個工作點已經寫好並排在機器上，'
                   '等分階段訓練那一批讓出卡就會自動開跑；跑完這一頁重跑一次'
                   '就會多出它的列與曲線。</p>')

    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>整併：把相位旋轉與量化交付放進同一個模組</title>
<style>{CSS}</style></head><body><main>

<h1>整併：把相位旋轉與量化交付放進同一個模組</h1>
<p class="lede">十張影像、InstructPix2Pix、1000 步 sign-PGD。位移取
<code>edit_lpips</code>、失真取 <code>fid_dists</code>，全部同一個來源。
擋下數是 CLIP 代理（門檻 0.8445），金標準仍是人眼。</p>

<h2>一、一句話</h2>
<p>現行做法把擾動設計在一個空間、交付在另一個空間，<strong>交付那一刻要投影
一次，而投影會削掉一塊</strong>。這一塊在兩個域上都量到了。整併版把參數搬進
交付格式本身，於是那筆投影不存在——這是可證偽的預測。</p>

<h2>二、四個參數化族放在同一張圖上</h2>
<div class="card">{chart_tradeoff(p)}</div>
<p>三件事值得指出。<strong>其一</strong>，DCT 學習平面（橘）整條曲線都在相位族
（綠）下方，等失真下最好也只到 0.93 倍——換域本身沒有換到效果。
<strong>其二</strong>，交付那一步（虛線箭頭）把學習平面往<strong>右下</strong>
推：失真更高、位移更低，兩軸皆被支配，不需要任何內插就看得出來。
<strong>其三</strong>，位移場（灰）的細網格終於進得了失真帶，但衝過頭到 0.279，
而它在那裡的位移還不如相位族在一半失真上的讀數。</p>

<h2>三、交付要付的那一筆，兩個域都要付</h2>
<div class="card">{chart_delivery_cost(p)}</div>
<p>左邊是等失真內插出來的（相位族兩條曲線在失真上重疊，可以內插）；右邊不需要
內插——交付版比不交付版<strong>失真更高而位移更低</strong>。</p>
<p><strong>輸的原因不是「沒學到落腳點」</strong>：交付後的擾動保留率是 0.9486，
與相位族的 0.9431 同級。輸的是<strong>可行集變小</strong>本身。這正是整併版
要拿回來的東西。</p>

<h2>四、保長不是負擔，是資產</h2>
<div class="card">{chart_length_preserving(p)}</div>
<p>非正交的逐係數乘性增益是我們自己設計來打保長旋轉的對照組。兩個失真點上
結果都相反：保長是它的 <strong>{g16 / p["nd_gain_16"]["disp"]:.2f} 倍</strong>
與 <strong>{g11 / p["nd_gain_11"]["disp"]:.2f} 倍</strong>。
這是「非加性」在 DCT 域唯一的正面證據，也是整併版值得做的理由之一——
要整併的那個東西本身有價值。</p>

<h2>五、架構：差別在參數住哪裡</h2>
<div class="card">{architecture()}</div>
<p>DCT-Shield 的架構之所以簡單又有效，關鍵是它把參數直接放在交付格式裡——
參數即交付，沒有投影，抗 JPEG 是白送的。整併版把同一個選擇套到我們的
參數化上。</p>
<p><strong>兩件不可以誇大的事。</strong>其一，旋轉之後必須再取整，所以交出去的
動作仍是「一個整數的係數位移」，也就是 DCT-Shield 的動作空間；我們能主張的是
<em>約束</em>不同（位移被限制在通過原係數的球面上），不是<em>動作</em>不同。
其二，「DCT 的相位就是係數的正負號」是 Ito &amp; Kiya (2007) 的既有結論，
本專案的貢獻上限是「把它連續化成保長旋轉，而且平面是學出來的」。</p>

<h2>六、全部工作點</h2>
<table><thead><tr><th>工作點</th><th>DISTS↓</th><th>PSNR↑</th><th>L∞↓</th>
<th>位移↑</th><th>擋下</th><th>說明</th></tr></thead><tbody>
{row("ours_ph_n", "相位 r0.9（不交付）")}
{row("ours_pg_n", "相位＋增益 r0.9（不交付）")}
{row("ours_pg_m", "相位＋增益 r2.0（不交付）", "階段一的參照")}
{row("ours_ph_q", "相位 r0.9 ＋量化交付")}
{row("ours_pg_q", "相位＋增益 r0.9 ＋量化交付")}
{row("dct_aj85", "DCT-Shield（§6.3 論文設定）", "對手")}
{row("nd_plane_12", "DCT 學習平面 θ1.2")}
{row("nd_plane_18", "DCT 學習平面 θ1.8")}
{row("nd_plane_25", "DCT 學習平面 θ2.5")}
{row("nd_plane_31", "DCT 學習平面 θ=π")}
{row("nd_plane_25_qd", "DCT 學習平面 θ2.5 ＋事後投影交付", "U1 的比較對象")}
{row("nd_gain_11", "DCT 非正交乘性 g1.1", "打保長用的對照")}
{row("nd_gain_16", "DCT 非正交乘性 g1.6", "打保長用的對照")}
{row("nd_rand_25", "同角度隨機解", "P4 對照")}
{row("w_adam_rand", "位移場 grid 16")}
{row("w_adam_g64", "位移場 grid 64", "到得了帶但衝過頭")}
{urows}
</tbody></table>

<h2>七、整併版的四條判準（跑之前就寫下）</h2>
{ustatus}
<table><thead><tr><th>判準</th><th>內容</th></tr></thead><tbody>
<tr><td>U1</td><td><strong>主判準。</strong>等失真下的未淨化位移要高於
<code>nd_plane_25_qd</code>（0.1617／0.5679／6-of-10）。達不到就代表那 21% 不是
事後投影造成的，「參數住在交付格式裡就會比較好」這條推理要整個重寫。</td></tr>
<tr><td>U2</td><td>仍要贏同上界的隨機解。每一個新參數化都要有這一格。</td></tr>
<tr><td>U3</td><td><code>delta_within_1</code> 決定新穎性怎麼寫：比例高就代表
動的幾乎全落在 DCT-Shield 的 ε=1 球裡，只能主張「約束不同」。固定配對版在帶內
工作點是 0.927。</td></tr>
<tr><td>U4</td><td><code>zero_coef_frac</code> 是可行集稀薄程度的直接讀數。
<strong>高不等於容量不夠</strong>——固定配對版的成對零比例 0.7538，但那些配對
只帶 0.79% 的能量。要與 U1 一起讀。</td></tr>
</tbody></table>

<h3>兩筆已知的代價，先寫在這裡</h3>
<p><strong>殘差是稀疏高振幅尖峰。</strong>學習平面在等 DISTS 下多付 3.4 dB
PSNR 與 2.2 倍 L∞（上表的 L∞ 欄直接看得到）。DISTS 打平不代表失真打平，
換到 PSNR 或 L∞ 對齊會明顯落後。</p>
<p><strong>空間選擇性變粗。</strong>8×8 格點的徑向解析度只有 8 階，而 32×32 的
rfft2 有 17 階。空間選擇性正是本方法對 DCT-Shield 的主要構造差異。</p>

<p class="foot">重跑：<code>python scripts/integration_report.py</code>。
數值來源在 <code>runs/ip2p_mainline/tables/</code>、
<code>runs/ip2p_dct_nonadd/</code>、<code>runs/ip2p_warp_hard/</code>、
<code>runs/ip2p_dct_unified/</code>。本頁不入版控，圖全部是內嵌 SVG。</p>
</main></body></html>
"""


def main() -> None:
    p = load_points()
    out = ROOT / "report_integration.html"
    out.write_text(page(p), encoding="utf-8")
    print(f"寫出 {out}（{out.stat().st_size / 1024:.0f} KB、{len(p)} 個工作點）")


if __name__ == "__main__":
    main()
