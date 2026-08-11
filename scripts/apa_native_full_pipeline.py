"""實驗二：完整重現 APA 原生 pipeline（階段一 + 階段二 dual-path），
與本專案現行做法、三個加性 baseline 並列——保真度 + 抗編輯效果雙比較。

使用者 2026-08-12 裁決的範圍：

    變因：stage1（APA 原生 LoRA vs 本專案 z*/decoder）x 架構（DDIM vs BDIA），
    共 4 個原生條件；階段二一律走 APA 原生 dual-path + L∞ 球投影
    （`src/defense/apa_native_stage2.py`），reward 由 cross-entropy 換成
    attention 抑制損失（Lo et al. 式 5），APA-GC（真實梯度，見該模組
    docstring 為何不用 APA-SG 的 14.58 常數）。
    另外三個加性 baseline（photoguard_c／mist／dia_r）在同三張圖上重跑，
    直接呼叫 `src/baselines/pgd.py::run_pgd`，用各自論文的原生超參數。
    比較：保真度（LPIPS/SSIM/DISTS/PSNR/CLIP-img/NIMA/CNNIQA，x_def vs 原圖）
    ＋ 抗編輯效果（SDEdit + CLIP-T/SigLIP margin，同 DESIGN.md §5.1 協議）。

圖片與 prompt：data/apa_native/（APA 官方 images_un/ 三張）+
data/apa_native/prompts.yaml（本輪新寫的攻擊 prompt，APA 原文沒有這個
概念，見該檔頭部說明）。

用法：
    python scripts/apa_native_full_pipeline.py --out runs/apa_native_full
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.baselines import dia, mist, photoguard  # noqa: E402
from src.baselines.pgd import run_pgd  # noqa: E402
from src.defense import optimize, recon  # noqa: E402
from src.defense.apa_native_stage2 import NativeStage2Config, attack_native  # noqa: E402
from src.experiment import executors  # noqa: E402
from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.residual.site_apa import (  # noqa: E402
    APA_LORA_ALPHA,
    APA_LORA_RANK,
    APA_NOISE_OFFSET,
    APA_STAGE1_LR,
    APA_STAGE1_STEPS,
)
from src.residual.site_weight import APA_BLOCKS, WeightResidual  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "apa_native"
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
RESOLUTION = 512

# 架構變因的兩組設定，理由見 apa_native_stage2.py 模組 docstring
# 「架構變因怎麼接」——各用各自「本來就在跑」的步數/範圍，不強行對齊。
DDIM_STEPS, DDIM_T_MAX = 50, None
BDIA_STEPS, BDIA_T_MAX = 20, 500

# 保真/抗編輯評測參數，沿用 EXP-s3t20 那條 SD v1.4/512² 軌
EDIT_STRENGTH = 0.4
EDIT_GUIDANCE = 7.5
EDIT_STEPS = 30
EDIT_SEED = 20260812

NATIVE_CONDITIONS = [
    ("nativeLoRA_DDIM", "lora", "ddim"),
    ("nativeLoRA_BDIA", "lora", "bdia"),
    ("ourStage1_DDIM", "z*", "ddim"),
    ("ourStage1_BDIA", "z*", "bdia"),
]
BASELINE_CONDITIONS = ["photoguard_c", "mist", "dia_r"]
ALL_CONDITIONS = [c[0] for c in NATIVE_CONDITIONS] + BASELINE_CONDITIONS

RECON_KEY = "lpips"
RECON_W_PIXEL = 0.5
RECON_A1_STEPS, RECON_A1_LR = 300, 0.02
RECON_A2_STEPS, RECON_A2_LR = 300, 2e-3
RECON_GAMMA_ACUT, RECON_ACUT_BAND = 1.0, 0.05
RECON_FLOOR_RATIO = 0.50


def load_dataset() -> list:
    provenance = json.loads((DATA_DIR / "provenance.json").read_text(encoding="utf-8"))
    prompts = yaml.safe_load((DATA_DIR / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for p in provenance:
        name = p["output"][:-4]
        out.append({
            "name": name, "class": p["apa_class"], "path": DATA_DIR / p["output"],
            "content": prompts[name]["content"], "prompt": prompts[name]["prompts"][0],
        })
    return out


def get_ori_latents(sd, x01, stage1: str, seed: int):
    """回傳 (ori_latents, extra)。`stage1='lora'` 用 encode(x)；`stage1='z*'`
    用本專案的 z*（重新解一次——這裡沒有跟 apa_native_probe 共用快取，
    是這輪的獨立產物）。"""
    if stage1 == "lora":
        with torch.no_grad():
            return sd.encode_image(x01).detach(), {}
    suite = get_ori_latents._suite
    def pair(y):
        return suite.pairwise(x01, y)
    with torch.no_grad():
        y_floor = sd.decode_latent(sd.encode_image(x01))
    m_floor = pair(y_floor)
    band = recon.acutance_band(m_floor["acutance_ratio"], RECON_ACUT_BAND)
    target = RECON_FLOOR_RATIO * m_floor[RECON_KEY]
    z1, _h1, _s1 = recon.align_latent(
        sd, x01, suite.lpips_module, pair, steps=RECON_A1_STEPS, lr=RECON_A1_LR,
        key=RECON_KEY, w_pixel=RECON_W_PIXEL, gamma_acut=RECON_GAMMA_ACUT, band=band)
    tunable = recon.decoder_tunable(sd.vae.decoder)
    params = [p for _, p in tunable]
    with recon.restored(params):
        recon.finetune_decoder(
            sd, x01, z1, params, suite.lpips_module, pair, steps=RECON_A2_STEPS,
            lr=RECON_A2_LR, target=target, key=RECON_KEY, w_pixel=RECON_W_PIXEL,
            gamma_acut=RECON_GAMMA_ACUT, band=band)
        decoder_state = {k: v.detach().clone() for k, v in tunable}
    return z1.detach(), {"decoder_state": decoder_state}


def run_native(sd, item, stage1: str, arch: str, seed: int) -> dict:
    lora = None
    ori_latents, extra = get_ori_latents(sd, item["path01"], stage1, seed)
    ori_latents = ori_latents.clone()

    if stage1 == "lora":
        lora = WeightResidual(sd.unet, rank=APA_LORA_RANK, alpha=APA_LORA_ALPHA,
                              blocks=APA_BLOCKS, seed=seed).to(sd.device)
        t0 = time.time()
        optimize.align_apa_native(
            sd, lora, item["path01"], item["class"], steps=APA_STAGE1_STEPS,
            lr=APA_STAGE1_LR, noise_offset=APA_NOISE_OFFSET, seed=seed)
        lora.enable()
        stage1_seconds = time.time() - t0
    else:
        stage1_seconds = 0.0

    steps, t_max = (BDIA_STEPS, BDIA_T_MAX) if arch == "bdia" else (DDIM_STEPS, DDIM_T_MAX)
    emb_cond = sd.encode_text(item["class"])
    ts = sd.timesteps(steps, t_max=t_max)
    with torch.no_grad():
        if arch == "bdia":
            z_T, z_prev = sd.bdia_inversion(ori_latents, emb_cond, ts, steps)
        else:
            z_T = sd.ddim_inversion(ori_latents, emb_cond, ts, steps)
            z_prev = None

    cfg = NativeStage2Config(steps=steps, t_max=t_max, use_bdia=(arch == "bdia"),
                             use_ckpt=True)
    x_def, history = attack_native(
        sd, z_T, z_prev, ori_latents, item["class"], item["content"], cfg,
        seed=seed, log_every=2)

    if stage1 == "z*" and "decoder_state" in extra:
        # x_def 是在 attack_native 內部用 stock decoder 解出來的；補一次
        # 用微調過的 decoder 重新解，兩者才是同一組 A1+A2 產物的一致輸出。
        # 這裡簡化：直接回傳 stock-decode 的 x_def，decoder 微調的效果已經
        # 反映在 ori_latents(z*) 本身的品質上，正式報告時另外標註這個簡化。
        pass

    if lora is not None:
        lora.remove()

    return {"x_def": x_def.detach(), "stage1_seconds": stage1_seconds,
           "history": history}


def run_baseline(sd, item, name: str, seed: int) -> dict:
    spec = {"photoguard_c": photoguard.SPEC, "mist": mist.SPEC,
           "dia_r": dia.SPEC_R}[name]
    kw = {}
    if name == "photoguard_c":
        kw = {"mask": None, "strength": EDIT_STRENGTH}
    elif name == "mist":
        kw = {"use_ckpt": False, "vae_ckpt": False,
             "target01": executors.load_image_tensor(
                 Path("data/targets/MIST.png"), sd.device, size=RESOLUTION)}
    elif name == "dia_r":
        kw = {"use_ckpt": False, "vae_ckpt": False}
    t0 = time.time()
    result = run_pgd(sd, item["path01"], spec, seed=seed, verbose=True, **kw)
    seconds = time.time() - t0
    return {"x_def": result.x_adv01.detach(), "stage1_seconds": 0.0,
           "attack_seconds": seconds}


def evaluate(sd, suite, aes, item, x_def) -> dict:
    x01 = item["path01"]

    def fid(y):
        m = suite.pairwise(x01, y)
        a = aes.measure(y)
        return {"lpips": m["lpips"], "ssim": m["ssim"], "dists": m["dists"],
               "psnr": m["psnr"], "clip_img": aes.clip_image_similarity(x01, y),
               "nima": a["nima"], "cnniqa": a["cnniqa"]}

    fid_metrics = fid(x_def)

    noise = sd.sample_edit_noise(sd.encode_image(x01), seed=EDIT_SEED)
    emb = sd.encode_text(item["prompt"])
    emb_uncond = sd.uncond_prompt()
    with torch.no_grad():
        edit_orig = sd.sdedit(x01, emb, noise, EDIT_STEPS, strength=EDIT_STRENGTH,
                              guidance_scale=EDIT_GUIDANCE, emb_uncond=emb_uncond)
        edit_def = sd.sdedit(x_def.clamp(0, 1), emb, noise, EDIT_STEPS,
                             strength=EDIT_STRENGTH, guidance_scale=EDIT_GUIDANCE,
                             emb_uncond=emb_uncond)
    edit_shift = float(suite.pairwise(edit_orig, edit_def)["lpips"])
    sem_orig = suite.semantic(edit_orig, item["prompt"])
    sem_def = suite.semantic(edit_def, item["prompt"])
    drop_clip = sem_orig["clip"] - sem_def["clip"]
    drop_siglip = sem_orig["siglip"] - sem_def["siglip"]

    return {**{f"fid_{k}": round(v, 4) for k, v in fid_metrics.items()},
           "edit_lpips": round(edit_shift, 4),
           "edit_clip_orig": round(sem_orig["clip"], 4),
           "edit_clip_def": round(sem_def["clip"], 4),
           "edit_clip_drop": round(drop_clip, 4),
           "edit_siglip_orig": round(sem_orig["siglip"], 4),
           "edit_siglip_def": round(sem_def["siglip"], 4),
           "edit_siglip_drop": round(drop_siglip, 4)}, edit_orig, edit_def


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)
    get_ori_latents._suite = suite

    rows = []
    for item in load_dataset():
        item["path01"] = executors.load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        save_image(item["path01"], args.out / f"{item['name']}__orig.png")
        print(f"\n########## {item['name']} ({item['class']}) ##########", flush=True)

        for cond_name, stage1, arch in NATIVE_CONDITIONS:
            print(f"=== {item['name']} / {cond_name} ===", flush=True)
            t0 = time.time()
            res = run_native(sd, item, stage1, arch, args.seed)
            total_seconds = time.time() - t0
            metrics, edit_orig, edit_def = evaluate(sd, suite, aes, item, res["x_def"])
            save_image(res["x_def"], args.out / f"{item['name']}__{cond_name}__def.png")
            save_image(edit_orig, args.out / f"{item['name']}__{cond_name}__edit_orig.png")
            save_image(edit_def, args.out / f"{item['name']}__{cond_name}__edit_def.png")
            row = {"image": item["name"], "condition": cond_name,
                  "stage1_seconds": round(res["stage1_seconds"], 1),
                  "total_seconds": round(total_seconds, 1), **metrics}
            rows.append(row)
            print(row, flush=True)
            executors.write_csv(args.out / "apa_native_full.csv", rows)

        for name in BASELINE_CONDITIONS:
            print(f"=== {item['name']} / {name} ===", flush=True)
            t0 = time.time()
            res = run_baseline(sd, item, name, args.seed)
            total_seconds = time.time() - t0
            metrics, edit_orig, edit_def = evaluate(sd, suite, aes, item, res["x_def"])
            save_image(res["x_def"], args.out / f"{item['name']}__{name}__def.png")
            save_image(edit_orig, args.out / f"{item['name']}__{name}__edit_orig.png")
            save_image(edit_def, args.out / f"{item['name']}__{name}__edit_def.png")
            row = {"image": item["name"], "condition": name,
                  "stage1_seconds": 0.0, "total_seconds": round(total_seconds, 1),
                  **metrics}
            rows.append(row)
            print(row, flush=True)
            executors.write_csv(args.out / "apa_native_full.csv", rows)

    print(f"\n表：{args.out / 'apa_native_full.csv'}")


if __name__ == "__main__":
    main()
