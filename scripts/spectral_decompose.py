"""重現 PAD（Zhou et al., ICML 2023）第 3 節的量化研究，換成擴散編輯的讀數。

問題
────────────────────────────────────────────────────────────────────
PAD 在 CIFAR-10／ResNet-18 上把對抗樣本拆成「只有幅度被擾動」與「只有相位
被擾動」兩半，發現**相位那一半用更小的雜訊換到更大的破壞**（該篇表 1：
準確率 6.12% 對 24.32%，L2 雜訊 0.9385 對 0.9931）。

本腳本把同一個分解搬到本專案的讀數上：對每一張已存的防禦圖 `x_def`，
以原圖 `x` 為參照造出

    只有幅度  x_amp = F⁻¹( ξ_{x_def} , φ_x     )
    只有相位  x_pha = F⁻¹( ξ_x       , φ_{x_def} )

三個版本（`full` = `x_def` 本身、`amp`、`pha`）各自跑 SDEdit，量

    位移量 = LPIPS( 編輯(原圖), 編輯(該版本) )

**這回答的是 FND-040 欠的問題**：任何保護擾動裡真正在起作用的，是不是它的
相位那一半。若成立，「相位重要」就從本方法的一個性質，升級成所有保護擾動
的共通性質。

為什麼加性 baseline 也要進來
────────────────────────────────────────────────────────────────────
DEC-025 把加性 baseline 從**比較表**上擱置。本腳本不是比較表——它把那些
擾動當成**被解剖的對象**。少了它們，這個研究只剩下拆解自己的方法，而
紋理重相位的幅度偏移由構造壓得很小，結果會退化成恆等式而非發現。

夾取
────────────────────────────────────────────────────────────────────
交叉互換後的影像可能超出 [0,1]。**夾取會破壞「幅度逐位保留」**，故本腳本
一律記錄 `clip_fraction`（被夾住的像素比例）與夾取後的 `amp_dev`
（相對原圖的全域幅度譜偏移）。`amp_dev` 對 `pha` 版本若不接近 0，代表夾取
已經吃掉這個分解的定義，該列不可解讀。

用法
    python scripts/spectral_decompose.py --run runs/s0817/merged \\
        --data data/set0817 --edit-strength 0.7 --out runs/spectral/dec_0.csv \\
        --images cat_00 shiba_00
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torchvision.utils as vutils  # noqa: E402

from apa_baseline import (  # noqa: E402
    EDIT_GUIDANCE, EDIT_SEED, EDIT_STEPS, EDIT_STRENGTH, MODEL_NAME,
    RESOLUTION, head_keep, load_dataset,
)
from phase_retention import cell_of  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.residual.spectral_split import (  # noqa: E402
    amplitude_deviation, decompose,
)
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

VARIANTS = ("full", "amp", "pha", "amp_s", "pha_s")


def _overflow(v: torch.Tensor) -> dict:
    """夾取前超出 [0,1] 的程度。**只報「有幾成像素超出」會誤導**：自然照片
    本來就有大量恰好落在 0 或 1 的飽和像素，任何擾動都會把它們推出界，於是
    比例很高而幅度極小。故同時報超出量的平均與最大值。"""
    over = (v - v.clamp(0.0, 1.0)).abs()
    return {"clip_fraction": float((over > 0).double().mean()),
            "clip_mean": float(over.mean()),
            "clip_max": float(over.max())}


def _match_distortion(x01: torch.Tensor, delta: torch.Tensor, target: float,
                      suite, *, rounds: int = 12, hi: float = 4.0) -> tuple:
    """把擾動縮放到 `DISTS(x01, clamp(x01 + s·delta)) = target`。

    存在理由：交叉互換後的影像失真**高於**原始防禦圖（PAD 表 1 自己記過
    這件事——分解版本的 L∞ 反而比完整對抗樣本大），故「哪個半邊位移量比較
    大」在未對齊的失真上不可解讀。PAD 的處置是把重組樣本夾回同一個 L∞ 球
    再測一次（該篇的 `Accuracy_com`）；本專案的預算軸是 DISTS（DEC-015），
    故改對 DISTS 對齊。

    單調性：`s` 增大時失真單調上升（夾取只會讓它更慢上升，不會反轉），
    故二分搜尋有意義。達不到目標時回傳 `hi` 並標記 unreachable。
    """
    def at(s: float):
        x = (x01 + s * delta).clamp(0.0, 1.0)
        return x, float(suite.pairwise(x01, x)["dists"])

    x_hi, d_hi = at(hi)
    if d_hi < target:
        return x_hi, hi, True                      # unreachable
    lo, best, best_err = 0.0, x_hi, abs(d_hi - target)
    best_s = hi
    for _ in range(rounds):
        mid = 0.5 * (lo + hi)
        x_m, d_m = at(mid)
        if abs(d_m - target) < best_err:
            best, best_err, best_s = x_m, abs(d_m - target), mid
        if d_m < target:
            lo = mid
        else:
            hi = mid
    return best, best_s, False


def build_variants(x01: torch.Tensor, x_def: torch.Tensor, suite) -> dict:
    """回傳 {版本名: (影像, 診斷 dict)}。分解在 float64 上做以免累積誤差。

    五個版本：`full` 是防禦圖本身；`amp`／`pha` 是 PAD 的原始分解（失真未
    對齊）；`amp_s`／`pha_s` 把同一個擾動縮放到與 `full` 相同的 DISTS。
    """
    d = decompose(x01.double(), x_def.double())
    if d["imag_max"] > 1e-6:
        raise ValueError(
            f"逆轉換的虛部殘量 {d['imag_max']:.3e} 過大——輸入可能不是實數影像")

    target = float(suite.pairwise(x01, x_def)["dists"])
    out = {"full": (x_def, {"clip_fraction": 0.0, "clip_mean": 0.0,
                            "clip_max": 0.0, "scale": 1.0, "unreachable": False})}
    for key, name in (("amp_only", "amp"), ("pha_only", "pha")):
        v = d[key]
        diag = _overflow(v)
        out[name] = (v.clamp(0.0, 1.0).to(x_def),
                     {**diag, "scale": 1.0, "unreachable": False})
        delta = v.to(x_def) - x01
        x_s, s, unreach = _match_distortion(x01, delta, target, suite)
        out[f"{name}_s"] = (x_s, {**_overflow(x01 + s * delta),
                                  "scale": s, "unreachable": unreach})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/set0817"))
    ap.add_argument("--edit-strength", type=float, default=EDIT_STRENGTH)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-png", action="store_true", default=True,
                    help="存下每個版本的影像。判準以人眼為主，每一格都要有圖可看")
    args = ap.parse_args()
    out = args.out or (args.run / "spectral_decompose.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    with (args.run / "results.csv").open(encoding="utf-8") as fh:
        cells = [cell_of(r) for r in csv.DictReader(fh)]
    if args.images:
        keep = set(args.images)
        cells = [c for c in cells if c["image"] in keep]
    if args.conditions:
        keep = set(args.conditions)
        cells = [c for c in cells if c["condition"] in keep]
    if not cells:
        raise SystemExit("沒有符合條件的列")

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    seeds = [EDIT_SEED + k for k in range(args.seeds)]
    dataset = {d["name"]: d for d in load_dataset(args.data)}
    edit_orig_cache: dict = {}

    def edit(x01, item, seed):
        noise = sd.sample_edit_noise(sd.encode_image(x01), seed=seed)
        emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
        with torch.no_grad():
            return sd.sdedit(x01.clamp(0, 1), emb, noise, EDIT_STEPS,
                             strength=args.edit_strength,
                             guidance_scale=EDIT_GUIDANCE,
                             emb_uncond=emb_u, keep01=head_keep(item, x01))

    rows = []
    for cell in cells:
        item = dataset[cell["image"]]
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        def_png = args.run / f"{cell['image']}__{cell['tag']}__def.png"
        if not def_png.exists():
            raise FileNotFoundError(f"缺少防禦圖 {def_png}")
        x_def = load_image_tensor(def_png, sd.device, size=RESOLUTION)

        for seed in seeds:
            key = (cell["image"], seed)
            if key not in edit_orig_cache:
                edit_orig_cache[key] = edit(x01, item, seed)

        print(f"=== {cell['image']} / {cell['tag']} ===", flush=True)
        t0 = time.time()
        variants = build_variants(x01, x_def, suite)

        for name in VARIANTS:
            x_v, diag = variants[name]
            fid = suite.pairwise(x01, x_v)
            vals = []
            for seed in seeds:
                e = edit(x_v, item, seed)
                vals.append(float(suite.pairwise(
                    edit_orig_cache[(cell["image"], seed)], e)["lpips"]))
            if args.save_png:
                vutils.save_image(
                    x_v.clamp(0, 1),
                    out.parent / f"{cell['image']}__{cell['tag']}__{name}.png")
            rows.append({
                "image": cell["image"], "condition": cell["condition"],
                "budget_target": cell["budget"], "variant": name,
                "effect_mean": round(statistics.fmean(vals), 5),
                "effect_sd": round(statistics.stdev(vals), 5) if len(vals) > 1 else "",
                "fid_lpips": round(fid["lpips"], 5),
                "fid_dists": round(fid["dists"], 5),
                "fid_psnr": round(fid["psnr"], 3),
                "fid_ssim": round(fid["ssim"], 5),
                "fid_linf": round(fid["linf"], 5),
                "fid_rms": round(fid["rms"], 5),
                "amp_dev": round(amplitude_deviation(x01.double(), x_v.double()), 6),
                "clip_fraction": round(diag["clip_fraction"], 6),
                "clip_mean": round(diag["clip_mean"], 8),
                "clip_max": round(diag["clip_max"], 6),
                "scale": round(diag["scale"], 5),
                "unreachable": diag["unreachable"],
                "edit_strength": args.edit_strength,
                "seconds": round(time.time() - t0, 1),
            })
        write_csv(out, rows)
        got = {r["variant"]: r["effect_mean"] for r in rows[-len(VARIANTS):]}
        print(f"    位移量 {got}", flush=True)

    print(f"\n表：{out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
