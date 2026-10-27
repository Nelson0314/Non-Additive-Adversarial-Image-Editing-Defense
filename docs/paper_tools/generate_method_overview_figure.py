"""論文第 1 頁 teaser（方法總覽，條件 color3d：雙角度色度旋轉 LUT）：(a) LUT 參數化、(b) 編碼器位移最佳化、(c) 淨化與編輯結果。

遠端 CPU 執行（不用 GPU）：`~/venvs/wacv/bin/python generate_method_overview_figure.py --out-dir <目錄>`。
影像、LUT 參數與頻譜一律讀既有產物；輸出兩個變體 `method_overview_spectrum.{pdf,png}`（含擾動頻譜）與
`method_overview.{pdf,png}`（不含頻譜，空間給結果對照），寬 6.9 in。
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Polygon, Arc
from PIL import Image

REPO = Path.home() / "image-immunization"
TRIAL = REPO / "color/trials/lut3d_defense_and_naturalness"
COND = "color3d"  # 雙角度色度旋轉（原條件名 color3d_smooth_polar10_hue3）
IMAGE = "woman_00"
PROMPT = "p3"  # "Let the person wear a bowtie"（color/data/portraits/prompts.yaml）
PROMPT_LABEL = "“wear a bowtie”"
PURIFIER = "blur2"
BASE = REPO / "baseline/artifacts/aligned"

PATHS = {
    "x": REPO / "color/data/portraits/woman" / f"{IMAGE}.png",
    "x_def": TRIAL / f"artifacts/defenses/{COND}/{IMAGE}__{COND}__def.png",
    "carrier": TRIAL / f"artifacts/defenses/{COND}/{IMAGE}__carrier.pt",
    "x_add": BASE / f"defenses/dct_shield/{IMAGE}__dct_shield__def.png",
    "edit_ours": TRIAL / f"artifacts/purified_edits/{COND}/{PURIFIER}/ip2p_{COND}_{PURIFIER}/{IMAGE}__{PROMPT}.png",
    "edit_none": TRIAL / f"artifacts/purified_edits/undefended/{PURIFIER}/ip2p_undefended_{PURIFIER}/{IMAGE}__{PROMPT}.png",
    "edit_add": BASE / f"purified_edits/dct_shield/{PURIFIER}/ip2p_dct_shield_{PURIFIER}/{IMAGE}__{PROMPT}.png",
    "spectrum": TRIAL / f"results/perturbation_spectrum/{COND}/spectrum.csv",
}

W, H = 6.9, 2.55
GRAY_BOX = "#e4e4e4"
C_A, C_B, C_C = "#1f8a4c", "#2445c7", "#c0392b"
C_ADD = "#e08a1e"


def load_rgb(p, size=512):
    im = Image.open(p).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def meridian_angles(c, o):
    """以黑點為原點、灰軸 u 為極軸：o 相對 c 的極角變化（c 的子午面內）與方位角變化（繞灰軸），單位度。"""
    u = torch.full_like(c, 3 ** -0.5)
    n = torch.cross(u, c, dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
    on = (o * n).sum(1, keepdim=True)
    om = o - on * n
    cos = (om * c).sum(1) / (om.norm(dim=1) * c.norm(dim=1)).clamp_min(1e-12)
    polar = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    azimuth = torch.rad2deg(torch.atan2(on[:, 0].abs(), torch.cross(u, om, dim=1).norm(dim=1)))
    return polar, azimuth


def load_lut():
    for p in (TRIAL, REPO / "color/src", REPO / "color/vendor"):
        sys.path.insert(0, str(p))
    import generate_color3d_defenses as g

    cfg = g.TARGETS[COND]  # 與 generate_color3d_defenses.main 相同的類別設定
    if not cfg.get("smooth_angle") or "hue" not in cfg or cfg.get("desat"):
        raise RuntimeError(f"{COND} 不是雙角度 SmoothAngleLut：{cfg}")
    L = g.SmoothAngleLut
    L.tan_theta, L.sigma = math.tan(math.radians(cfg["theta"])), cfg["sigma"]
    L.tan_hue, L.desat = math.tan(math.radians(cfg["hue"])), False
    state = torch.load(PATHS["carrier"], map_location="cpu")
    lut = L(grid=int(state["grid"]))
    lut.reset(torch.zeros(1, 3, 8, 8), seed=0)
    lut.load_state_dict(state)
    with torch.no_grad():
        table = lut.lut()
        err = float((table - lut.identity - state["delta"]).abs().max())
        if err > 1e-5:
            raise RuntimeError(f"重算的 LUT 與 carrier 的 delta 不一致：max |Δ| = {err:.3g}")
        x = torch.from_numpy(load_rgb(PATHS["x"])).permute(2, 0, 1)[None]
        y = lut.render(x)[0].permute(1, 2, 0).numpy()
        d = np.abs(y - load_rgb(PATHS["x_def"])).mean() * 255
        if d > 1.0:
            raise RuntimeError(f"以 LUT 重繪的防禦圖與存檔不一致：平均 |Δ| = {d:.3f}／255")
        # 截到 [0, 1] 之前的輸出（SmoothAngleLut.lut 的雙角度分支），並確認截斷後與 lut() 相同
        v = lut._blur(lut.delta)[0].reshape(3, -1).T
        v = v - (v * lut.axis).sum(1, keepdim=True) * lut.axis
        vh = (v * lut.hue_dir).sum(1, keepdim=True)
        vp = v - vh * lut.hue_dir
        n = vp.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dd = lut.axis + vp * (lut.tan_theta * torch.tanh(n / lut.tan_theta) / n)
        ud = torch.cross(torch.full_like(dd, 3 ** -0.5), dd, dim=1)
        r = ud.norm(dim=1, keepdim=True)
        hd = torch.where(r > 1e-6, ud / r.clamp_min(1e-12), torch.zeros_like(ud))
        b = (lut.tan_hue * r).clamp_min(1e-12)
        v = dd - lut.axis + b * torch.tanh(vh / b) * hd
        pre = lut.axis + v
        pre = lut.cn * pre / pre.norm(dim=1, keepdim=True)
        out = table[0].reshape(3, -1).T
        if float((pre.clamp(0, 1) - out).abs().max()) > 1e-5:
            raise RuntimeError("截斷前輸出的重算與 SmoothAngleLut.lut() 不一致")
        c = lut.identity[0].reshape(3, -1).T
        # 方位角只在距灰軸 ≥ 1e-3（RGB 單位，約 0.26／255）的顏色上有定義；float32 下更近的點無法量到 0.1° 精度
        def off_axis(z):
            return torch.cross(torch.full_like(z, 3 ** -0.5), z, dim=1).norm(dim=1) > 1e-3

        tp, ta = math.degrees(math.atan(lut.tan_theta)), math.degrees(math.atan(lut.tan_hue))
        c64, pre64, out64 = c.double(), pre.double(), out.double()
        p0, a0 = meridian_angles(c64, pre64)
        m0 = off_axis(c64)
        ma0 = m0 & off_axis(pre64)
        p0, a0 = p0[m0], a0[ma0]
        if float(p0.max()) > tp + 0.01 or float(a0.max()) > ta + 0.01:  # 0.01°：float32 輸出在距灰軸 1e-3 處的角度解析度
            raise RuntimeError(f"截斷前角度超出上界：極角 {float(p0.max()):.3f}°、方位角 {float(a0.max()):.3f}°")
        p1, a1 = meridian_angles(c64, out64)
        ma1 = m0 & off_axis(out64)
        p1, a1 = p1[m0 & (out64.norm(dim=1) > 1e-3)], a1[ma1]
        print(f"LUT 檢查：max |Δ| = {err:.2e}；重繪平均 |Δ| = {d:.3f}/255")
        print(f"  截斷前：極角變化最大 {float(p0.max()):.3f}°（上界 {tp:g}°）、方位角變化最大 {float(a0.max()):.3f}°（上界 {ta:g}°；"
              f"輸出距灰軸 < 1e-3 的 {int((m0 & ~off_axis(pre64)).sum())} 個格點不計）")
        print(f"  截斷到 [0, 1] 後：極角最大 {float(p1.max()):.3f}°、方位角最大 {float(a1.max()):.3f}°；"
              f"極角 > {tp:g}° 的格點 {int((p1 > tp + 1e-3).sum())}、方位角 > {ta:g}° 的格點 {int((a1 > ta + 1e-3).sum())}／{int(ma1.sum())}")
    return dict(G=lut.grid, ident=c.numpy(), param=state["param"][0].reshape(3, -1).T.numpy(), field=v.numpy(), out=out.numpy(),
                theta=tp, hue=ta)


def read_spectrum():
    rows = {}
    with open(PATHS["spectrum"]) as h:
        r = csv.reader(h)
        head = next(r)
        freqs = np.array([float(s[1:]) for s in head[5:]])
        for row in r:
            rows[(row[0], row[1])] = np.array([float(s) for s in row[5:]])
    return freqs, rows


class Canvas:
    """以英吋為座標的覆蓋層，負責箭頭、方框與文字。"""

    def __init__(self, fig):
        self.fig = fig
        self.ax = fig.add_axes([0, 0, 1, 1], zorder=10)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.set_axis_off()
        self.ax.patch.set_alpha(0)

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / W, y / H, w / W, h / H], **kw)

    def image(self, arr, x, y, s, edge="black", lw=0.6, cmap=None, vmin=None, vmax=None):
        a = self.axes(x, y, s, s)
        a.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="lanczos" if cmap is None else "nearest")
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor(edge)
            sp.set_linewidth(lw)
        return a

    def box(self, cx, cy, w, h, text, fc=GRAY_BOX, fs=7, weight="normal"):
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                         fc=fc, ec="black", lw=0.6))
        self.ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, weight=weight, linespacing=1.0)

    def trapezoid(self, cx, cy, w, h, text):
        pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 4), (cx + w / 2, cy + h / 4), (cx - w / 2, cy + h / 2)]
        self.ax.add_patch(Polygon(pts, closed=True, fc=GRAY_BOX, ec="black", lw=0.6))
        self.ax.text(cx, cy, text, ha="center", va="center", fontsize=8)

    def arrow(self, pts, color="black", ls="-", lw=0.7):
        pts = np.asarray(pts, dtype=float)
        if len(pts) > 2:
            self.ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, ls=ls, lw=lw, solid_capstyle="butt")
        self.ax.annotate("", xy=pts[-1], xytext=pts[-2],
                         arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=color, lw=lw, ls=ls,
                                         shrinkA=0, shrinkB=0))

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", 7)
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        self.ax.text(x, y, s, **kw)

    def bracket(self, x0, x1, y, color):
        self.ax.plot([x0, x0, x1, x1], [y + 0.05, y, y, y + 0.05], color=color, lw=1.0)


def lattice_axes(cv, x, y, s):
    a = cv.axes(x, y, s, s, projection="3d")
    a.set_axis_off()
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_zlim(0, 1)
    a.view_init(elev=22, azim=-58)
    a.set_box_aspect((1, 1, 1), zoom=1.25)
    a.patch.set_alpha(0)
    e = [0, 1]
    for i in e:
        for j in e:
            a.plot([0, 1], [i, i], [j, j], color="0.55", lw=0.4)
            a.plot([i, i], [0, 1], [j, j], color="0.55", lw=0.4)
            a.plot([i, i], [j, j], [0, 1], color="0.55", lw=0.4)
    return a


def sub_index(G, step=8):
    k = np.arange(0, G, step)
    bb, gg, rr = np.meshgrid(k, k, k, indexing="ij")
    return (bb * G * G + gg * G + rr).ravel()


def draw_field(a, pts, vec, colors, length=0.16):
    n = np.linalg.norm(vec, axis=1)
    vec = vec * (length / max(np.percentile(n, 95), 1e-12))
    a.quiver(pts[:, 0], pts[:, 1], pts[:, 2], vec[:, 0], vec[:, 1], vec[:, 2], colors=colors,
             arrow_length_ratio=0.35, linewidths=0.45, normalize=False)


def chroma_diagram(cv, lut):
    """雙角度示意（角度放大）：左為子午面側視（灰軸為極軸，極角變化 Δθ），右為沿灰軸俯視（方位角變化 Δφ）；‖c‖ 不變。"""
    k = int(np.argmin(np.linalg.norm(lut["ident"] - np.array([0.86, 0.62, 0.52]), axis=1)))
    cols = [lut["ident"][k], np.clip(lut["out"][k], 0, 1)]
    style = dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=0.8, shrinkA=0, shrinkB=0)

    def frame(x, y, w, h, xlim, ylim):
        a = cv.axes(x, y, w, h)
        a.set_xlim(*xlim)
        a.set_ylim(*ylim)
        a.set_aspect("equal")
        a.set_axis_off()
        a.patch.set_alpha(0)
        return a

    def rays(a, o, pts, arc_r, arc_t, label, label_xy):
        a.annotate("", xy=pts[0], xytext=o, arrowprops=dict(color="0.4", **style))
        a.annotate("", xy=pts[1], xytext=o, arrowprops=dict(color="black", **style))
        a.add_patch(Arc(o, 2 * arc_r, 2 * arc_r, theta1=arc_t[0], theta2=arc_t[1], color="black", lw=0.7))
        a.scatter(pts[:, 0], pts[:, 1], s=26, c=cols, edgecolors="black", linewidths=0.4, zorder=5)
        a.text(*label_xy, label, fontsize=7, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.3))

    # 側視：子午面內，以黑點為原點、灰軸為極軸
    a = frame(0.02, 1.36, 0.60, 0.98, (-0.15, 0.85), (-0.12, 1.5))
    t0, t1, L = 15, 40, 1.15
    pts = np.array([[L * math.sin(math.radians(t)), L * math.cos(math.radians(t))] for t in (t0, t1)])
    a.plot([0, 0], [0, 1.35], color="0.55", lw=0.8)
    a.add_patch(Arc((0, 0), 2 * L, 2 * L, theta1=90 - 55, theta2=90 - 5, color="0.55", lw=0.6, ls=(0, (2, 1.5))))
    rays(a, (0, 0), pts, 0.62, (90 - t1, 90 - t0), r"$\Delta\theta$", (0.36, 0.78))
    a.scatter([0], [0], s=12, c="black", zorder=5)
    a.text(0.05, -0.05, "black", fontsize=7, ha="left", va="top")
    a.text(-0.05, 1.35, "gray", fontsize=7, ha="right", va="top", rotation=90)
    a.text(pts[0, 0] + 0.06, pts[0, 1] + 0.02, "$c$", fontsize=8, va="bottom")
    a.text(pts[1, 0] + 0.07, pts[1, 1], "$c'$", fontsize=8, va="center")
    a.text(0.35, 1.5, rf"$\Delta\theta<{lut['theta']:g}^\circ$", fontsize=7, ha="center", va="top")

    # 俯視：沿灰軸看，繞灰軸的方位角
    a = frame(0.64, 1.48, 0.50, 0.82, (-1.0, 1.0), (-1.0, 1.75))
    r, f0, f1 = 0.85, -30, 75
    pts = np.array([[r * math.cos(math.radians(f)), r * math.sin(math.radians(f))] for f in (f0, f1)])
    a.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=0, theta2=360, color="0.55", lw=0.6, ls=(0, (2, 1.5))))
    fm = math.radians((f0 + f1) / 2)
    rays(a, (0, 0), pts, 0.34, (f0, f1), r"$\Delta\phi$", (0.66, 0.16))
    a.scatter([0], [0], s=16, c="0.6", edgecolors="black", linewidths=0.4, zorder=5)
    a.text(-0.12, 0, "gray", fontsize=7, ha="right", va="center")
    a.text(0, 1.72, rf"$\Delta\phi<{lut['hue']:g}^\circ$", fontsize=7, ha="center", va="top")
    cv.text(0.88, 1.43, r"$\|c'\|=\|c\|$")


def spectrum_plot(cv, x, y, w, h, freqs, rows):
    a = cv.axes(x, y, w, h)
    for cond, col in ((COND, C_B), ("dct_shield", C_ADD)):
        a.loglog(freqs, rows[(cond, "none")], color=col, lw=0.9)
        a.loglog(freqs, rows[(cond, PURIFIER)], color=col, lw=0.9, ls=(0, (2.2, 1.2)))
    a.set_xlim(freqs[0], 0.5)
    a.tick_params(labelsize=7, length=2, width=0.5, pad=1)
    a.set_yticks([])
    a.minorticks_off()
    a.set_xticks([0.01, 0.1])
    a.set_xticklabels(["0.01", "0.1"])
    a.set_xlabel("frequency", fontsize=7, labelpad=0)
    a.set_ylabel("power", fontsize=7, labelpad=1)
    for sp in a.spines.values():
        sp.set_linewidth(0.5)
    a.text(0.0, 1.03, "ours", color=C_B, fontsize=7, ha="left", va="bottom", transform=a.transAxes)
    a.text(1.0, 1.03, "DCT-Shield", color=C_ADD, fontsize=7, ha="right", va="bottom", transform=a.transAxes)
    a.plot([], [], color="black", lw=0.9, label="before")
    a.plot([], [], color="black", lw=0.9, ls=(0, (2.2, 1.2)), label="after blur")
    a.legend(loc="lower left", fontsize=7, frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.2, handletextpad=0.4,
             borderaxespad=0.1)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman"],
                         "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})

    lut = load_lut()
    x, x_def = load_rgb(PATHS["x"]), load_rgb(PATHS["x_def"])
    z, z_def = encode_latents(x, x_def)
    freqs, rows = read_spectrum()

    fig = plt.figure(figsize=(W, H))
    cv = Canvas(fig)
    S, yb, yt = 0.55, 0.52, 1.62
    ym = (yb + S + yt) / 2

    # (a) 色空間參數化
    idx = sub_index(lut["G"])
    pts = lut["ident"][idx]
    a = lattice_axes(cv, 0.0, 0.47, 0.72)
    draw_field(a, pts, lut["param"][idx], "0.35")
    cv.text(0.36, 0.44, "free params $P$", va="top")
    cv.box(0.94, yb + S / 2, 0.42, 0.34, "Gaussian\n$\\sigma=4$")
    a = lattice_axes(cv, 1.17, 0.47, 0.72)
    draw_field(a, pts, lut["field"][idx], np.clip(pts, 0, 1) * 0.85)
    cv.text(1.53, 0.44, "rotation field", va="top")
    a = lattice_axes(cv, 1.17, 1.52, 0.72)
    out = np.clip(lut["out"][idx], 0, 1)
    a.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=out, s=7, depthshade=False, edgecolors="none")
    cv.text(1.53, 2.30, "LUT $T$ ($33^3$)", va="bottom")
    chroma_diagram(cv, lut)
    cv.arrow([(0.70, yb + S / 2), (0.73, yb + S / 2)])
    cv.arrow([(1.15, yb + S / 2), (1.22, yb + S / 2)])
    cv.arrow([(1.53, 1.12), (1.53, 1.55)])
    cv.text(1.47, 1.335, "rotate $c$", ha="right")

    # (b) 編碼器位移最佳化
    xc = 2.22
    cv.image(x, xc, yb, S)
    cv.image(x_def, xc, yt, S)
    cv.text(xc - 0.05, yb + S / 2, "$x$", fontsize=9, ha="right")
    cv.text(xc - 0.05, yt + S / 2, r"$\tilde{x}$", fontsize=9, ha="right")
    cv.box(xc + S / 2, ym, 0.40, 0.22, "apply $T$")
    cv.arrow([(xc + S / 2, yb + S), (xc + S / 2, ym - 0.11)])
    cv.arrow([(xc + S / 2, ym + 0.11), (xc + S / 2, yt)])
    cv.arrow([(1.90, 1.80), (2.02, 1.80), (2.02, ym), (xc + S / 2 - 0.20, ym)])
    ex = 3.06
    for yy in (yt + S / 2, yb + S / 2):
        cv.trapezoid(ex, yy, 0.30, 0.44, "$E$")
        cv.arrow([(xc + S, yy), (ex - 0.15, yy)])
    cv.text(ex, ym, "frozen\nencoder", linespacing=1.0)
    hs, hx = 0.40, 3.28
    lo, hi = np.percentile(np.stack([z[0], z_def[0]]), [1, 99])
    for zz, yy, lab in ((z_def, yt, r"$E(\tilde{x})$"), (z, yb, "$E(x)$")):
        cv.image(zz[0], hx, yy + (S - hs) / 2, hs, cmap="magma", vmin=lo, vmax=hi, lw=0.5)
        cv.arrow([(ex + 0.15, yy + S / 2), (hx, yy + S / 2)])
    cv.text(hx + hs / 2, yt + (S + hs) / 2 + 0.02, r"$E(\tilde{x})$", va="bottom")
    cv.text(hx + hs / 2, yb + (S - hs) / 2 - 0.02, "$E(x)$", va="top")
    lx = 4.05
    cv.box(lx, ym, 0.70, 0.36, "maximize\n" + r"$\|E(\tilde{x})-E(x)\|_2$", fc="#dfe6fb")
    cv.arrow([(hx + hs, yt + S / 2), (lx, yt + S / 2), (lx, ym + 0.18)])
    cv.arrow([(hx + hs, yb + S / 2), (lx, yb + S / 2), (lx, ym - 0.18)])
    cv.arrow([(lx + 0.35, ym), (4.46, ym), (4.46, 0.27), (0.36, 0.27), (0.36, 0.33)], color=C_B, ls=(0, (2.5, 1.5)), lw=0.8)
    cv.text(2.4, 0.27, r"$\nabla_P$", color=C_B, fontsize=7, va="center",
            bbox=dict(fc="white", ec="none", pad=0.5))

    # (c) 淨化與編輯
    px = 4.80
    cv.arrow([(xc + S / 2, yt + S), (xc + S / 2, 2.38), (px, 2.38), (px, yt + S / 2 + 0.24)])
    cv.box(px, yt + S / 2, 0.52, 0.48, "JPEG\nblur\nVAE-Trans")
    cv.text(px, yt - 0.02, "purifier", va="top")
    ix = 5.16
    cv.image(load_rgb(PATHS["x_pur"]), ix, yt, S)
    cv.text(ix + S / 2, yt - 0.02, "blur", va="top")
    cv.arrow([(px + 0.26, yt + S / 2), (ix, yt + S / 2)])
    ex2 = 5.98
    cv.box(ex2, yt + S / 2 + 0.06, 0.40, 0.24, "IP2P")
    cv.text(ex2, yt + S / 2 - 0.10, PROMPT_LABEL, va="top", style="italic", linespacing=1.0)
    cv.arrow([(ix + S, yt + S / 2 + 0.06), (ex2 - 0.20, yt + S / 2 + 0.06)])
    ox = 6.30
    cv.image(load_rgb(PATHS["edit_ours"]), ox, yt, S, edge=C_C, lw=1.0)
    cv.arrow([(ex2 + 0.20, yt + S / 2 + 0.06), (ox, yt + S / 2 + 0.06)])
    cv.text(ox + S / 2, yt + S + 0.02, "ours", va="bottom", color=C_C)
    s2 = 0.50
    for k, (key, lab) in enumerate((("edit_none", "no defense"), ("edit_add", "DCT-Shield"))):
        xx = 4.55 + k * (s2 + 0.06)
        cv.image(load_rgb(PATHS[key]), xx, yb + 0.03, s2)
        cv.text(xx + s2 / 2, yb, lab, va="top")
    spectrum_plot(cv, 5.92, yb + 0.12, 0.93, 0.76, freqs, rows)

    # 階段標籤
    for x0, x1, col, lab in ((0.05, 1.95, C_A, "(a) Color-space parameterization"),
                             (2.10, 4.45, C_B, "(b) Encoder-displacement optimization"),
                             (4.55, 6.85, C_C, "(c) Purification and editing")):
        cv.bracket(x0, x1, 0.17, col)
        cv.text((x0 + x1) / 2, 0.07, lab, color=col, weight="bold")

    fig.savefig(args.out_dir / "method_overview.pdf")
    fig.savefig(args.out_dir / "method_overview.png", dpi=300)
    print("輸出", args.out_dir)


if __name__ == "__main__":
    main()
