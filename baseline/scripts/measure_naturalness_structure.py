"""防禦圖對原圖的結構量測（不經對齊）與容量遞增的全域色彩對齊後失真（CPU）。

結構量測（防禦圖 y 對原圖 x，皆在 BT.601 亮度 Y ∈ [0,1] 上計算）：
    ssim_structure  SSIM 的 structure 項 s = (σ_xy + C3)/(σ_x σ_y + C3)，C3 = (0.03)²/2，11×11 高斯窗 σ=1.5，取全圖平均
    fsim_pc         FSIM 只取 phase congruency 項：Σ S_PC·PC_m / Σ PC_m，S_PC 的 T1 = 0.85，PC 由 piq.fsim 的內部函式計算
                    （與 piq.fsim 相同的下取樣、尺度與方向設定）
    edge_ssim_sobel Sobel 梯度幅值（除以理論上限 4√2 映到 [0,1]）之間的 SSIM（piq.ssim）
    edge_ssim_canny Canny（cv2，門檻 100／200，8 位元 Y）二值邊緣圖之間的 SSIM
WCT2（arXiv:1903.09760）的協定以 HED 邊緣圖計算 SSIM；HED 權重未取得，以 Sobel 與 Canny 代替。

對齊（防禦圖以全域最小平方對齊回原圖，輸出量化到 8 位元），容量依序遞增：
    none         不對齊
    lab_affine   Lab 3×4 仿射（color trial `tools/measure_aligned_perceptual_quality.py` 的 fit_lab_affine）
    root_poly    sRGB 值上的二階 root-polynomial 3×6（Finlayson, Mackiewicz & Hurlbert 2015 的項 R、G、B、√RG、√GB、√RB；
                 遠端環境未安裝 colour-science，依同一組項以 numpy 最小平方實作）
    lut9         9³ 三線性 LUT（同一 trial 檔的 fit_lut，曲率正則 0.1）
每一級報 LPIPS（piq.LPIPS）與 DISTS（piq.DISTS），防禦圖對原圖，CPU。

用法（遠端 baseline 專案根）：python scripts/measure_naturalness_structure.py <輸出目錄>
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import piq
import torch
import torch.nn.functional as F
from piq.fsim import _construct_filters, _phase_congruency
from piq.functional import similarity_map

HOME = Path.home()
T = HOME / "image-immunization" / "color" / "trials" / "lut3d_defense_and_naturalness"
sys.path.insert(0, str(T / "tools"))
ARGV = sys.argv[:]
sys.argv = [ARGV[0], "cpu"]  # 該模組在載入時以 sys.argv[1] 決定裝置
import measure_aligned_perceptual_quality as apq  # noqa: E402
sys.argv = ARGV

apq.dev = torch.device("cpu")
torch.set_num_threads(4)

ALIGNED = HOME / "image-immunization" / "baseline" / "artifacts" / "aligned" / "defenses"
IMAGES = ("man_00", "woman_01")
ARMS = ("photoguard_encoder", "mist", "dia_pt", "dct_shield")
COLOR = ("color3d", "color3d_orig")
GRADING = ("kodak_portra_400", "creative_teal_orange")


def original(im):
    return T / "artifacts" / "defenses" / "color3d" / f"{im}__orig.png"


def defended(cond, im):
    if cond == "original":
        return original(im)
    if cond in ARMS:
        return ALIGNED / cond / f"{im}__{cond}__def.png"
    if cond in COLOR:
        return T / "artifacts" / "defenses" / cond / f"{im}__{cond}__def.png"
    return T / "artifacts" / "perceptual_quality" / "grading_luts" / cond / f"{im}.png"


def luma(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def gaussian(size=11, sigma=1.5):
    g = torch.arange(size, dtype=torch.float64) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_structure(x, y):
    w = gaussian()
    x, y = luma(x).double(), luma(y).double()
    mx, my = F.conv2d(x, w), F.conv2d(y, w)
    vx = (F.conv2d(x * x, w) - mx ** 2).clamp_min(0)
    vy = (F.conv2d(y * y, w) - my ** 2).clamp_min(0)
    cxy = F.conv2d(x * y, w) - mx * my
    c3 = (0.03 ** 2) / 2
    return float(((cxy + c3) / (vx.sqrt() * vy.sqrt() + c3)).mean())


def fsim_pc(x, y):
    a, b = luma(x) * 255, luma(y) * 255
    k = max(1, round(min(a.shape[-2:]) / 256))
    a, b = F.avg_pool2d(a, k), F.avg_pool2d(b, k)
    filters = _construct_filters(a, 4, 4, 6, 2, 0.55, 1.2)  # piq.fsim 的預設
    pa = _phase_congruency(a, filters=filters, scales=4, orientations=4, k=2.0)
    pb = _phase_congruency(b, filters=filters, scales=4, orientations=4, k=2.0)
    pm = torch.maximum(pa, pb)
    return float((similarity_map(pa, pb, 0.85) * pm).sum() / pm.sum())


def sobel(x):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None]
    y = luma(x)
    gx, gy = F.conv2d(y, kx, padding=1), F.conv2d(y, kx.transpose(-1, -2), padding=1)
    return ((gx ** 2 + gy ** 2).sqrt() / (4 * 2 ** 0.5)).clamp(0, 1)


def canny(x):
    y = (luma(x)[0, 0].numpy() * 255).round().astype(np.uint8)
    return torch.from_numpy(cv2.Canny(y, 100, 200).astype(np.float32) / 255)[None, None]


def fit_root_poly(y, x):
    def terms(v):
        r, g, b = v[:, 0], v[:, 1], v[:, 2]
        return np.stack([r, g, b, np.sqrt(r * g), np.sqrt(g * b), np.sqrt(r * b)], 1)
    yy = y[0].flatten(1).T.double().numpy()
    xx = x[0].flatten(1).T.double().numpy()
    coef, *_ = np.linalg.lstsq(terms(yy), xx, rcond=None)  # (6, 3)
    out = torch.from_numpy(terms(yy) @ coef).float().T.reshape(1, 3, *y.shape[-2:])
    return apq.quantize(out)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    lpips, dists = piq.LPIPS(), piq.DISTS()
    structure, alignment = [], []
    for cond in ("original",) + COLOR + ARMS + GRADING:
        for im in IMAGES:
            x, y = apq.tens(original(im)), apq.tens(defended(cond, im))
            if x.shape != y.shape:
                raise SystemExit(f"{cond} {im} 尺寸 {tuple(y.shape)} 與原圖 {tuple(x.shape)} 不同")
            with torch.no_grad():
                structure.append({
                    "condition": cond, "image": im,
                    "ssim_structure": round(ssim_structure(x, y), 5),
                    "fsim_pc": round(fsim_pc(x, y), 5),
                    "edge_ssim_sobel": round(float(piq.ssim(sobel(x), sobel(y), data_range=1.0)), 5),
                    "edge_ssim_canny": round(float(piq.ssim(canny(x), canny(y), data_range=1.0)), 5),
                    "defended_png": defended(cond, im).as_posix(), "original_png": original(im).as_posix(),
                    "device": "cpu"})
                for name, fit in (("none", lambda a, b: a), ("lab_affine", apq.fit_lab_affine),
                                  ("root_poly", fit_root_poly), ("lut9", apq.fit_lut)):
                    z = fit(y, x)
                    alignment.append({"condition": cond, "image": im, "alignment": name,
                                      "lpips": round(float(lpips(z, x)), 5), "dists": round(float(dists(z, x)), 5),
                                      "device": "cpu"})
            print(f"[DONE] {cond} {im}", flush=True)
    for name, rows in (("structure.csv", structure), ("alignment.csv", alignment)):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"[ALLDONE] {out}", flush=True)


if __name__ == "__main__":
    main()
