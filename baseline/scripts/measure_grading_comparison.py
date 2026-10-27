"""color3d_phr 防禦圖與全域調色（RawTherapee HaldCLUT、隨機參數化調色）對原圖的影響，以同一組讀數量測（CPU）。

條件（皆對 data/portraits 8 張原圖量）：
    color3d_phr                 color trial `artifacts/defenses/color3d_phr/<影像>__color3d_phr__def.png`
    rt_<名稱>               color trial `artifacts/perceptual_quality/grading_luts/<名稱>/<影像>.png`（20 個 HaldCLUT）
    random_raw_<k>          每張 20 組隨機參數化調色，不縮放（k = 0..19）
    random_matched_<k>      同一組參數乘上逐張的倍率 α，使防禦圖 LPIPS 等於該張 color3d_phr 的防禦圖 LPIPS（二分 14 次）

隨機調色（sRGB [0,1]，依序施加，輸出量化到 8 位元）；範圍取 Lightroom／RawTherapee 基本面板的常用區間：
    曝光 EV ∈ [−1, 1]：線性 RGB 乘 2^EV
    白平衡：色溫 τ ∈ [−0.2, 0.2]（線性 R ×(1+τ)、B ×(1−τ)），色調 κ ∈ [−0.1, 0.1]（線性 G ×(1−κ)）
    對比 S 曲線 c ∈ [−0.4, 0.4]：y = x − c·sin(2πx)/(2π)，逐通道施加於 sRGB 值（|c| < 1 時單調）
    飽和度 s ∈ [−0.4, 0.4]：Lab 彩度乘 (1+s)
    色相旋轉 h ∈ [−15°, 15°]：Lab 的 (a, b) 旋轉 h
    參數獨立均勻抽樣；種子 = SEED + 影像序號（SEED = 20261024），參數逐列寫入 CSV。
    縮放組把五個參數同乘 α（α ∈ [0, 8]）；α 達上限仍未到目標時照實記錄 α 與 LPIPS。

讀數（調色圖或防禦圖 y 對原圖 x）：
    結構：ssim_structure、fsim_pc、edge_ssim_sobel、edge_ssim_canny（定義見 measure_naturalness_structure.py）
    對齊後 LPIPS、DISTS：none、lab_affine、root_poly、lut9（同上）
    pieapp（pyiqa，不對齊）
    無參考：musiq（musiq_koniq）、laion_aes、clipiqa（pyiqa），報 y 的值與 y − x 的差
    colorfulness_ratio：Hasler–Süsstrunk M(y) / M(x)（color trial measure_grading_lut_quality.py 的定義）

用法（遠端 baseline 專案根）：python scripts/measure_grading_comparison.py <輸出目錄> <影像…>
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import piq
import pyiqa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_naturalness_structure as ns  # noqa: E402
from immunization_core.color.space import lab_to_rgb, rgb_to_lab  # noqa: E402

apq = ns.apq
IMAGES = ("man_00", "man_01", "man_02", "man_03", "woman_00", "woman_01", "woman_02", "woman_03")
GRADING = ("kodak_portra_160", "kodak_portra_400", "kodak_ektar_100", "kodak_kodachrome_64", "kodak_elite_chrome_200",
           "fuji_400h", "fuji_superia_400", "fuji_velvia_50", "fuji_provia_100f", "fuji_astia_100f", "agfa_vista_200",
           "polaroid_690", "creative_teal_orange", "creative_bleach_bypass", "creative_soft_warming",
           "creative_crisp_warm", "creative_crisp_winter", "creative_fall_colors", "creative_late_sunset",
           "creative_moonlight")
SEED = 20261024
DRAWS = 20
RANGES = {"ev": 1.0, "temperature": 0.2, "tint": 0.1, "contrast": 0.4, "saturation": 0.4, "hue_deg": 15.0}
ALPHA_MAX = 8.0
NR = {"musiq": "musiq", "laion_aes": "laion_aes", "clipiqa": "clipiqa"}


def to_linear(v):
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def to_srgb(v):
    v = v.clamp(0, 1)
    return torch.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def grade(x, p, alpha=1.0):
    q = {k: v * alpha for k, v in p.items()}
    lin = to_linear(x.double())
    gain = torch.tensor([1 + q["temperature"], 1 - q["tint"], 1 - q["temperature"]], dtype=torch.float64)
    lin = lin * gain[None, :, None, None] * 2 ** q["ev"]
    y = to_srgb(lin)
    y = (y - q["contrast"] * torch.sin(2 * math.pi * y) / (2 * math.pi)).clamp(0, 1)
    lab = rgb_to_lab(y)
    h = math.radians(q["hue_deg"])
    a, b = lab[:, 1], lab[:, 2]
    a, b = (a * math.cos(h) - b * math.sin(h)) * (1 + q["saturation"]), (a * math.sin(h) + b * math.cos(h)) * (1 + q["saturation"])
    y = lab_to_rgb(torch.stack([lab[:, 0], a, b], 1))
    return apq.quantize(y.float())


def draw(rng):
    return {k: float(rng.uniform(-r, r)) for k, r in RANGES.items()}


def colorfulness(x01):
    r, g, b = (x01[0] * 255).double()
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(torch.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * torch.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def main():
    out = Path(sys.argv[1])
    images = sys.argv[2:] or IMAGES
    out.mkdir(parents=True, exist_ok=True)
    lpips, dists = piq.LPIPS(), piq.DISTS()
    pieapp = pyiqa.create_metric("pieapp", device="cpu", as_loss=False)
    nr = {k: pyiqa.create_metric(v, device="cpu", as_loss=False) for k, v in NR.items()}
    rows, params = [], []

    def measure(cond, group, im, x, y, nr_x, extra):
        with torch.no_grad():
            row = {"condition": cond, "group": group, "image": im,
                   "ssim_structure": round(ns.ssim_structure(x, y), 5), "fsim_pc": round(ns.fsim_pc(x, y), 5),
                   "edge_ssim_sobel": round(float(piq.ssim(ns.sobel(x), ns.sobel(y), data_range=1.0)), 5),
                   "edge_ssim_canny": round(float(piq.ssim(ns.canny(x), ns.canny(y), data_range=1.0)), 5)}
            for name, fit in (("none", lambda a, b: a), ("lab_affine", apq.fit_lab_affine),
                              ("root_poly", ns.fit_root_poly), ("lut9", apq.fit_lut)):
                z = fit(y, x)
                row[f"lpips_{name}"] = round(float(lpips(z, x)), 5)
                row[f"dists_{name}"] = round(float(dists(z, x)), 5)
            row["pieapp"] = round(float(pieapp(y, x)), 5)
            for k, m in nr.items():
                v = float(m(y))
                row[k] = round(v, 5)
                row[f"{k}_delta"] = round(v - nr_x[k], 5)
            row["colorfulness_ratio"] = round(colorfulness(y) / colorfulness(x), 5)
        rows.append({**row, **extra, "device": "cpu"})

    for im in images:
        index = IMAGES.index(im)
        x = apq.tens(ns.original(im))
        with torch.no_grad():
            nr_x = {k: float(m(x)) for k, m in nr.items()}
            target = float(lpips(apq.tens(ns.defended("color3d_phr", im)), x))
        measure("color3d_phr", "color3d_phr", im, x, apq.tens(ns.defended("color3d_phr", im)), nr_x, {})
        for g in GRADING:
            y = apq.tens(ns.T / "artifacts" / "perceptual_quality" / "grading_luts" / g / f"{im}.png")
            measure(f"rt_{g}", "rt", im, x, y, nr_x, {})
        rng = np.random.default_rng(SEED + index)
        for k in range(DRAWS):
            p = draw(rng)
            measure(f"random_raw_{k}", "random_raw", im, x, grade(x, p), nr_x, {"alpha": 1.0})
            lo, hi = 0.0, ALPHA_MAX
            with torch.no_grad():
                if float(lpips(grade(x, p, hi), x)) < target:
                    lo = hi
                else:
                    for _ in range(14):
                        mid = (lo + hi) / 2
                        if float(lpips(grade(x, p, mid), x)) < target:
                            lo = mid
                        else:
                            hi = mid
            alpha = lo if lo == ALPHA_MAX else (lo + hi) / 2
            measure(f"random_matched_{k}", "random_matched", im, x, grade(x, p, alpha), nr_x,
                    {"alpha": round(alpha, 5), "target_lpips": round(target, 5)})
            params.append({"image": im, "draw": k, "seed": SEED + index, **{k2: round(v, 6) for k2, v in p.items()},
                           "alpha_matched": round(alpha, 5), "target_lpips": round(target, 5)})
            print(f"[DRAW] {im} {k} alpha={alpha:.3f}", flush=True)
        print(f"[DONE] {im}", flush=True)
        for name, data in ((f"grading_comparison_{'_'.join(images)}.csv", rows),
                           (f"random_grading_parameters_{'_'.join(images)}.csv", params)):
            fields = list(dict.fromkeys(k for r in data for k in r))
            with (out / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(data)
    print(f"[ALLDONE] {out}", flush=True)


if __name__ == "__main__":
    main()
