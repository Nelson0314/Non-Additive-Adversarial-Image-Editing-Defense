"""Fig. 2（fig:overview）的資料圖塊：產生 TikZ 片段與色度盤底圖，由 method_overview/method_overview.tex 組版。

輸出（--out-dir）：
  lattice_raw.tex、lattice_smooth.tex  (a) 33³ 晶格取 5³ 節點的向量箭頭。示意：固定種子的隨機向量場與其
                                        σ = 4 格的高斯平滑（與 SmoothAngleLut._blur 相同的可分離核與 replicate 邊界）。
  disk_background.png                   (b) 色度盤底色：‖c‖ = NORM 的球面沿灰軸俯視，半徑為極角 θ、角度為方位角 φ；
                                        色域外透明。
  disk_arrows.tex                       (b) 同一球面上 c → T(c) 的位移，T 為 woman_00、color3d 的實際 LUT（三線性查表）。
  disk_points.tex                       (b) 兩個同色像素的 c、c′ 在色度盤上的位置與色票顏色。

遠端 CPU 執行（basic-1）：
  cd ~/paper_figures && ~/venvs/wacv/bin/python in/generate_method_overview_panels.py --out-dir out_overview
再把輸出複製到 docs/paper_tools/method_overview/，於該目錄執行 tectonic -X compile method_overview.tex。
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path.home() / "image-immunization"
TRIAL = REPO / "color/trials/lut3d_defense_and_naturalness"
COND, IMG = "color3d", "woman_00"
PIXELS = ((228, 85), (423, 378))  # 影像座標 (x, y)，在 x 中同為 RGB (170, 123, 101)
GRID, SIGMA, NODES, NODE_STEP = 33, 4.0, 5, 2  # 示意晶格：5³ 個節點，間隔 2 格（σ 的一半）
DISK_THETA = 45.0  # 色度盤外緣的極角（度）
DISK_RADIUS = 1.6  # 色度盤外緣在圖上的半徑（cm）
ARROW_SCALE = 1.0  # 色度盤箭頭的放大倍率

G_AXIS = np.ones(3) / math.sqrt(3)
E1 = np.array([2.0, -1.0, -1.0]) / math.sqrt(6)  # 紅色方向在灰軸垂直面上的投影，φ = 0
E2 = np.cross(G_AXIS, E1)
# 晶格的等角投影（cm）：R 往左下、G 往右下、B 往上
PR, PG, PB = np.array([-0.62, -0.36]), np.array([0.62, -0.36]), np.array([0.0, 0.72])


def load_lut():
    for p in (TRIAL, REPO / "color/src", REPO / "color/vendor"):
        sys.path.insert(0, str(p))
    import generate_color3d_defenses as g

    cfg = g.TARGETS[COND]
    cls = g.SmoothAngleLut
    cls.tan_theta, cls.sigma = math.tan(math.radians(cfg["theta"])), cfg["sigma"]
    cls.tan_hue, cls.desat = math.tan(math.radians(cfg["hue"])), False
    state = torch.load(TRIAL / f"artifacts/defenses/{COND}/{IMG}__carrier.pt", map_location="cpu")
    lut = cls(grid=int(state["grid"]))
    lut.reset(torch.zeros(1, 3, 8, 8), seed=0)
    lut.load_state_dict(state)
    return lut


def lookup(table, colors):
    """與 SmoothAngleLut.render 相同的三線性查表；colors 為 N×3。"""
    grid = torch.from_numpy(colors * 2 - 1).float().view(1, 1, 1, -1, 3)
    return F.grid_sample(table, grid, mode="bilinear", align_corners=True)[0, :, 0, 0].T.double().numpy()


def to_rgb(norm, theta, phi):
    t, p = np.radians(theta), np.radians(phi)
    return norm * (np.cos(t)[..., None] * G_AXIS + np.sin(t)[..., None] * (np.cos(p)[..., None] * E1 + np.sin(p)[..., None] * E2))


def to_polar(c):
    u = c / np.linalg.norm(c, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(u @ G_AXIS, -1, 1))), np.degrees(np.arctan2(u @ E2, u @ E1))


def disk_xy(theta, phi):
    r = DISK_RADIUS * theta / DISK_THETA
    return r * np.cos(np.radians(phi)), r * np.sin(np.radians(phi))


def in_gamut(c):
    return np.all((c >= 0) & (c <= 1), axis=-1)


def lattice(lut, field, path):
    """5³ 節點的向量箭頭；箭頭長度以該場第 90 百分位正規化。"""
    idx = 8 + NODE_STEP * np.arange(NODES)  # 取中段，避開 replicate 邊界
    v = field[0].numpy()  # 3 × B × G × R
    last = NODES - 1  # 只畫可見的三個外表面（R、G、B 最大的面），避免前後節點在投影中交錯
    vectors = {(i, j, k): v[:, idx[k], idx[j], idx[i]] for i in range(NODES) for j in range(NODES) for k in range(NODES)
               if last in (i, j, k)}
    scale = 0.17 / np.percentile([np.linalg.norm(w) for w in vectors.values()], 90)
    step = 1.0 / (NODES - 1)
    lines = []
    for a, b in (((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)),
                 ((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)), ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),
                 ((1, 0, 1), (1, 1, 1)), ((0, 1, 1), (1, 1, 1))):
        p, q = (np.array(a) @ np.stack([PR, PG, PB]), np.array(b) @ np.stack([PR, PG, PB]))
        lines.append(rf"\draw[lattice] ({p[0]:.3f},{p[1]:.3f}) -- ({q[0]:.3f},{q[1]:.3f});")
    for (i, j, k), w in sorted(vectors.items(), key=lambda item: item[0][0] + item[0][1] - item[0][2]):
        p = np.array([i, j, k]) * step @ np.stack([PR, PG, PB])
        d = scale * (w @ np.stack([PR, PG, PB]))
        lines.append(rf"\fill[node] ({p[0]:.3f},{p[1]:.3f}) circle (0.45pt);")
        lines.append(rf"\draw[vec] ({p[0]:.3f},{p[1]:.3f}) -- ++({d[0]:.3f},{d[1]:.3f});")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lut = load_lut()
    with torch.no_grad():
        table = lut.lut()
        noise = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 3, GRID, GRID, GRID))).float()
        smooth = lut._blur(noise)
    lattice(lut, noise, args.out_dir / "lattice_raw.tex")
    lattice(lut, smooth, args.out_dir / "lattice_smooth.tex")

    x = np.asarray(Image.open(TRIAL / f"artifacts/defenses/{COND}/{IMG}__orig.png").convert("RGB"))
    protected = np.asarray(Image.open(TRIAL / f"artifacts/defenses/{COND}/{IMG}__{COND}__def.png").convert("RGB"))
    colors = {tuple(x[py, px]) for px, py in PIXELS}
    outputs = {tuple(protected[py, px]) for px, py in PIXELS}
    if len(colors) != 1 or len(outputs) != 1:
        raise ValueError(f"像素在 x 或 x̃ 中不同色：{colors}, {outputs}")
    c8, t8 = np.array(colors.pop(), dtype=float), np.array(outputs.pop(), dtype=float)
    c = c8 / 255.0
    norm = float(np.linalg.norm(c))

    # 色度盤底色
    n = 480
    xs = np.linspace(-DISK_THETA, DISK_THETA, n)
    X, Y = np.meshgrid(xs, xs[::-1])
    theta, phi = np.hypot(X, Y), np.degrees(np.arctan2(Y, X))
    rgb = to_rgb(norm, theta, phi)
    alpha = (in_gamut(rgb) & (theta <= DISK_THETA)).astype(float)
    image = np.dstack([np.clip(rgb, 0, 1), alpha])
    Image.fromarray(np.round(image * 255).astype(np.uint8), "RGBA").save(args.out_dir / "disk_background.png")

    # 位移場
    pts = []
    for t in np.arange(7.5, DISK_THETA, 7.5):
        k = max(6, int(round(2 * math.pi * t / 7.5)))
        pts += [(t, p + (t / 7.5 % 2) * 180 / k) for p in np.linspace(-180, 180, k, endpoint=False)]
    th, ph = np.array(pts).T
    src = to_rgb(norm, th, ph)
    keep = in_gamut(src)
    th, ph, src = th[keep], ph[keep], src[keep]
    dst = lookup(table, src)
    t1, p1 = to_polar(dst)
    x0, y0 = disk_xy(th, ph)
    x1, y1 = disk_xy(t1, p1)
    lines = [rf"\draw[disp] ({a:.3f},{b:.3f}) -- ({a + ARROW_SCALE * (u - a):.3f},{b + ARROW_SCALE * (w - b):.3f});"
             for a, b, u, w in zip(x0, y0, x1, y1)]
    Path(args.out_dir / "disk_arrows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 兩個同色像素：c 與 x̃ 中的 c′
    tc, pc = to_polar(c)
    tt, pt = to_polar(t8 / 255.0)
    cx, cy = disk_xy(tc, pc)
    qx, qy = disk_xy(tt, pt)
    rgb_line = lambda v: ",".join(str(int(round(e))) for e in v)
    Path(args.out_dir / "disk_points.tex").write_text(
        "\n".join([
            rf"\coordinate (dc) at ({cx:.3f},{cy:.3f});",
            rf"\coordinate (dcp) at ({qx:.3f},{qy:.3f});",
            rf"\xglobal\definecolor{{colorc}}{{RGB}}{{{rgb_line(c8)}}}",
            rf"\xglobal\definecolor{{colorcp}}{{RGB}}{{{rgb_line(t8)}}}",
            rf"\gdef\disknorm{{{norm:.2f}}}",
        ]) + "\n", encoding="utf-8")
    dphi = (p1 - ph + 180) % 360 - 180
    print(f"‖c‖ = {norm:.4f}；c (θ, φ) = ({float(tc):.2f}, {float(pc):.2f})，c′ = ({float(tt):.2f}, {float(pt):.2f})")
    print(f"色度盤箭頭 {len(th)} 支；|Δθ| 最大 {np.abs(t1 - th).max():.2f}°，|Δφ| 最大 {np.abs(dphi).max():.2f}°"
          f"（θ ≥ 15° 時 {np.abs(dphi[th >= 15]).max():.2f}°）")


if __name__ == "__main__":
    main()
