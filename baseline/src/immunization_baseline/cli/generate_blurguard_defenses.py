"""BlurGuard（`pgd_freq`）防禦圖：直接呼叫官方程式碼（github.com/jsu-kim/BlurGuard，commit e1b2d5e）。

沿用的官方元件與設定（`BlurGuard/configs/attack/base.yaml`、`code/blurguard.py`）：
    - `attacks.LF_PGD(attack_type="pgd_freq")`：擾動經 SAM 分區的高斯低通後投影到 L∞ ε，
      損失為 `‖get_first_stage_encoding(encode_first_stage(x_adv))‖²` 的平均，前 50 步只更新 σ。
    - `epsilon` 16（/255，程式內換算到 [-1,1]）、`steps` 150、`learning_rate` 0.06、`eps_sigma` 0、
      `sigma_weighting` 10；SAM `vit_h`、`points_per_side=2`，其餘遮罩參數同官方 `generate_and_process_masks`。
    - 輸入以 PIL bicubic 縮到 512²，轉到 [-1,1]。官方未固定亂數種子；這裡每張以 `torch.manual_seed(0)` 固定（主種子）。

與官方的差異：官方以 ldm 的 SD v1.4 checkpoint（`../ckpt/model.ckpt`）提供第一階段編碼器，該檔不在本機；
這裡以 diffusers 的 `CompVis/stable-diffusion-v1-4` VAE（同一組權重）實作 `encode_first_stage` 與
`get_first_stage_encoding`（後驗取樣乘 0.18215，與 ldm 相同）。`attacks.py` 頂層匯入但 `pgd_freq` 未使用的
`torch_dct` 在 import 前以空模組代替。`utils.py` 另需本環境沒有的 `colorama`、`sewar`（`pgd_freq` 皆不使用），
因此其中的 `generate_and_process_masks` 逐行複製為本檔的 `_generate_and_process_masks`，不匯入 `utils`。
`--eps-scale` 不等於 1 時列標 `modified_from_paper=True`（等失真臂）。

輸出版面與 `generate_defenses` 相同：`<輸出目錄>/<影像>__orig.png`、`<影像>__blurguard__def.png`、
`results_<tag>.csv`（求解設定與秒數；防禦圖失真由 `measure_additional_metrics --stage fidelity` 另量）。

用法（遠端 baseline 專案根，需一張卡）：
    python -m immunization_baseline.cli.generate_blurguard_defenses --data-root data/portraits \\
        --output-dir artifacts/defenses/blurguard [--eps-scale 1.5] [--images man_00 ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import types
from pathlib import Path

BLURGUARD_CODE = Path.home() / "thirdparty" / "BlurGuard" / "BlurGuard" / "code"
SAM_CHECKPOINT = BLURGUARD_CODE / "segment_anything" / "sam_vit_h_4b8939.pth"
CONDITION = "blurguard"
EPSILON = 16
STEPS = 150
LEARNING_RATE = 0.06
EPS_SIGMA = 0
SIGMA_WEIGHTING = 10
POINTS_PER_SIDE = 2
INPUT_SIZE = 512
SD_MODEL = "CompVis/stable-diffusion-v1-4"
VAE_SCALE = 0.18215


def _import_official():
    sys.modules.setdefault("torch_dct", types.ModuleType("torch_dct"))
    sys.path.insert(0, str(BLURGUARD_CODE))
    import attacks
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    return attacks, sam_model_registry, SamAutomaticMaskGenerator


def _generate_and_process_masks(SamAutomaticMaskGenerator, model, image, device, points_per_side,
                                pred_iou_thresh=0.86, stability_score_thresh=0.92, crop_n_layers=1,
                                crop_n_points_downscale_factor=2, min_mask_region_area=100):
    """官方 `code/utils.py` 的 `generate_and_process_masks`（commit e1b2d5e）逐行複製。"""
    import numpy as np
    import torch
    mask_generator = SamAutomaticMaskGenerator(
        model=model, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh, crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area)
    masks = mask_generator.generate(image)
    masks_dict = {}
    sorted_masks = sorted(masks, key=lambda x: x['area'])
    used_mask = np.zeros_like(masks[0]['segmentation'], dtype=np.uint8)
    for i, mask_dict in enumerate(sorted_masks):
        segmentation = mask_dict['segmentation']
        mask_array = np.where(segmentation, 1, 0).astype(np.uint8)
        mask_array = np.where(used_mask == 0, mask_array, 0)
        used_mask = np.where(mask_array > 0, 1, used_mask)
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        masks_dict[f'mask{i+1}'] = mask_tensor
    unused_mask = np.where(used_mask == 0, 1, 0).astype(np.uint8)
    if np.any(unused_mask):
        unused_mask_tensor = torch.tensor(unused_mask, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        masks_dict[f'mask{len(masks_dict) + 1}'] = unused_mask_tensor
    del mask_generator
    return masks_dict


class _FirstStage:
    """ldm `LatentDiffusion` 的兩個第一階段方法，以 diffusers VAE 實作。"""

    def __init__(self, vae):
        self.vae = vae

    def encode_first_stage(self, x):
        return self.vae.encode(x).latent_dist

    def get_first_stage_encoding(self, posterior):
        return VAE_SCALE * posterior.sample()


class _Net:
    def __init__(self, vae):
        self.model = _FirstStage(vae)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--eps-scale", type=float, default=1.0)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--tag", default="blurguard")
    args = ap.parse_args(argv)

    import cv2
    import numpy as np
    import PIL
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image

    attacks, sam_model_registry, SamAutomaticMaskGenerator = _import_official()
    device = "cuda"
    # 官方 build_sam 以相對於 code/ 的固定路徑讀權重（官方以 code/ 為工作目錄執行），建構時暫時切換工作目錄。
    import os
    cwd = os.getcwd()
    os.chdir(BLURGUARD_CODE)
    try:
        sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT)).to(device=device)
    finally:
        os.chdir(cwd)
    vae = AutoencoderKL.from_pretrained(SD_MODEL, subfolder="vae").to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    net = _Net(vae)
    epsilon = EPSILON * args.eps_scale
    eps_internal = epsilon / 255.0 * 2

    images = sorted(p for cls in ("man", "woman") for p in (args.data_root / cls).glob("*.png"))
    if args.images:
        images = [p for p in images if p.stem in set(args.images)]
        if len(images) != len(set(args.images)):
            raise SystemExit(f"--images 有不在 {args.data_root} 的影像")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in images:
        name = path.stem
        t0 = time.time()
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        masks = _generate_and_process_masks(SamAutomaticMaskGenerator, sam, rgb, device,
                                            points_per_side=POINTS_PER_SIDE)
        img = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), resample=PIL.Image.BICUBIC)
        img.save(args.output_dir / f"{name}__orig.png")
        x = torch.from_numpy(np.array(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].to(device)
        torch.manual_seed(0)
        attack = attacks.LF_PGD(net, eps_internal, STEPS, targeted=True, masks_dict=masks, filename=name,
                                eps_sigma=EPS_SIGMA, sigma_weighting=SIGMA_WEIGHTING,
                                learning_rate=LEARNING_RATE, attack_type="pgd_freq")
        x_adv, adv_loss, _, sigma_loss, sigmas = attack.perturb(x, torch.zeros_like(x))
        out = torch.clamp((x_adv[0] + 1.0) / 2.0, 0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
        Image.fromarray(out.astype(np.uint8)).save(args.output_dir / f"{name}__{CONDITION}__def.png")
        rows.append({
            "image": name, "class": path.parent.name, "condition": CONDITION,
            "eps": epsilon, "eps_pixel01": epsilon / 255.0, "eps_native": EPSILON, "eps_scale": args.eps_scale,
            "norm": "linf", "steps": STEPS, "learning_rate": LEARNING_RATE, "eps_sigma": EPS_SIGMA,
            "sigma_weighting": SIGMA_WEIGHTING, "num_masks": len(masks),
            "final_adv_loss": round(adv_loss[-1], 6), "final_sigma_loss": round(sigma_loss[-1], 6),
            "modified_from_paper": args.eps_scale != 1.0,
            "modification_note": ("first-stage encoder from diffusers SD v1.4 VAE"
                                  + ("" if args.eps_scale == 1.0 else f"; epsilon scaled by {args.eps_scale}")),
            "spec_source": "BlurGuard e1b2d5e configs/attack/base.yaml", "total_seconds": round(time.time() - t0, 1),
            "data_root": str(args.data_root),
        })
        print(f"[DONE] {name} masks={len(masks)} adv={adv_loss[-1]:.4f} ({rows[-1]['total_seconds']}s)", flush=True)
    with open(args.output_dir / f"results_{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[ALLDONE] {args.output_dir}（{len(rows)} 張）", flush=True)


if __name__ == "__main__":
    main()
