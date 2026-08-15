"""主線驅動：弱 baseline 與三個加性 baseline 的訓練與評測。

弱 baseline（DEC-023）＝ **完全原生 APA 的階段一與階段二機制，只把 reward
換成 targeted output**：

    階段一   APA 官方 LoRA（Eq.6 denoising MSE、AdamW 1e-4、200 步固定）
    階段二   dual-path attack guidance + latent L∞ 球 + sign/L1 動量
    reward   −‖D(z̄_0) − y_target‖²
    架構     DDIM（淺噪聲帶：50 格排程只執行前 11 格、T_a=10）

三個加性 baseline（photoguard_c／mist／dia_r）用各自論文的原生超參數，
直接呼叫 `src/baselines/pgd.py::run_pgd`。

評測：保真度（x_def 對原圖）＋ 抗編輯（SDEdit 後對「未防禦的編輯」比較）。
**未防禦的編輯必須真的成功**，否則抗編輯那一欄的分母不成立（DEC-022）。

用法：
    python scripts/apa_baseline.py --out runs/<批次> --data data/lo_aligned \
        --images horse_00 man_00 bird_03
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.baselines import dia, mist, photoguard  # noqa: E402
from src.baselines.pgd import run_pgd  # noqa: E402
from src.defense.apa_stage1 import align_apa_native  # noqa: E402
from src.defense.apa_native_stage2 import NativeStage2Config, attack_native  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402
from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.residual.apa_port import (  # noqa: E402
    APA_LORA_ALPHA, APA_LORA_RANK, APA_NOISE_OFFSET,
    APA_STAGE1_LR, APA_STAGE1_STEPS,
)
from src.residual.lora_weights import APA_BLOCKS, WeightResidual  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "apa_native"
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
RESOLUTION = 512
TARGET_IMAGE = "data/targets/gray.png"

# 評測設定。strength 0.55 由 DEC-022 定出——0.4 下有影像的未防禦編輯
# 根本沒發生，分母不成立。換資料集時**必須重新確認未防禦的編輯有成功**。
EDIT_STRENGTH = 0.55
EDIT_GUIDANCE = 7.5
EDIT_STEPS = 30
EDIT_SEED = 20260812

CONDITIONS = ["apa_weak", "photoguard_c", "mist", "dia_r"]

# latent 臂（規格 §3）：與 apa_weak 完全相同，只把 latent 的 L∞ 球換成 紋理重相位的
# 相位參數化。不列入 CONDITIONS 的預設，要用 --conditions 明給。
PARAMETERIZATION = {"apa_weak": "linf", "apa_phase": "phase"}


def load_dataset(root: Path = None, prompt_index: int = 0) -> list:
    """`root` 給定時讀 lo_aligned 版面（每類一子目錄）；否則讀 data/apa_native。

    `prompt_index` 選 `prompts.yaml` 的第幾個編輯 prompt。0 是「改掉指定內容」
    （論文的 A dog -> A cat），1 是「保留該內容、改動其他區域」（A dog in the
    park）。兩者是不同的惡意情境，防禦的難度不同——第二個 prompt 不要求模型
    改掉主體，故未防禦的編輯改動較小，抗編輯那一欄的分母也較小。
    **換 prompt 必須重新確認未防禦的編輯有成功**（DEC-022）。
    """
    if root is not None:
        spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
        return [{"name": img.stem, "class": spec[c]["content"], "path": img,
                 "content": spec[c]["content"],
                 "prompt": spec[c]["prompts"][prompt_index],
                 "prompt_index": prompt_index}
                for c in sorted(spec) for img in sorted((root / c).glob("*.png"))]
    prov = json.loads((DATA_DIR / "provenance.json").read_text(encoding="utf-8"))
    prompts = yaml.safe_load((DATA_DIR / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for p in prov:
        n = p["output"][:-4]
        out.append({"name": n, "class": p["apa_class"], "path": DATA_DIR / p["output"],
                    "content": prompts[n]["content"],
                    "prompt": prompts[n]["prompts"][prompt_index],
                    "prompt_index": prompt_index})
    return out


def run_weak_baseline(sd, item, y_target, seed: int,
                      parameterization: str = "linf") -> dict:
    """階段一（官方 LoRA）＋ 階段二（dual-path + 球 + sign），reward 為 targeted。"""
    x01 = item["path01"]
    with torch.no_grad():
        ori_latents = sd.encode_image(x01).detach().clone()

    lora = WeightResidual(sd.unet, rank=APA_LORA_RANK, alpha=APA_LORA_ALPHA,
                          blocks=APA_BLOCKS, seed=seed).to(sd.device)
    t0 = time.time()
    align_apa_native(sd, lora, x01, item["class"], steps=APA_STAGE1_STEPS,
                              lr=APA_STAGE1_LR, noise_offset=APA_NOISE_OFFSET, seed=seed)
    lora.enable()
    stage1_seconds = time.time() - t0

    cfg = NativeStage2Config(parameterization=parameterization)
    ts = sd.timesteps(cfg.schedule_steps, t_max=cfg.t_max)
    emb_cond = sd.encode_text(item["class"])
    with torch.no_grad():
        z_T = sd.ddim_inversion(ori_latents, emb_cond, ts, cfg.steps)
    x_def, history = attack_native(sd, z_T, None, ori_latents, item["class"],
                                   y_target, cfg)
    lora.remove()
    return {"x_def": x_def.detach(), "stage1_seconds": stage1_seconds,
            "history": history}


def run_additive(sd, item, name: str, seed: int) -> dict:
    spec = {"photoguard_c": photoguard.SPEC, "mist": mist.SPEC,
            "dia_r": dia.SPEC_R}[name]
    kw = {}
    if name == "photoguard_c":
        kw = {"mask": None, "strength": EDIT_STRENGTH}
    elif name == "mist":
        # fused mode 把兩次 VAE 編碼與一次完整 UNet 前向放在同一張圖上。
        kw = {"use_ckpt": True, "vae_ckpt": True,
              "target01": load_image_tensor(
                  Path("data/targets/MIST.png"), sd.device, size=RESOLUTION)}
    elif name == "dia_r":
        # DIA-R 把整條反演加整條重建留在同一張圖上，兩個開關都必須開，
        # 否則 OOM（`executors.baseline_kwargs` 記過同一條規則）。
        kw = {"use_ckpt": True, "vae_ckpt": True}
    return {"x_def": run_pgd(sd, item["path01"], spec, seed=seed, **kw).x_adv01.detach(),
            "stage1_seconds": 0.0}


def evaluate(sd, suite, aes, item, x_def):
    x01 = item["path01"]
    m = suite.pairwise(x01, x_def)
    a = aes.measure(x_def)
    fid = {"lpips": m["lpips"], "ssim": m["ssim"], "dists": m["dists"],
           "psnr": m["psnr"], "clip_img": aes.clip_image_similarity(x01, x_def),
           "nima": a["nima"], "cnniqa": a["cnniqa"]}

    noise = sd.sample_edit_noise(sd.encode_image(x01), seed=EDIT_SEED)
    emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
    with torch.no_grad():
        edit_orig = sd.sdedit(x01, emb, noise, EDIT_STEPS, strength=EDIT_STRENGTH,
                              guidance_scale=EDIT_GUIDANCE, emb_uncond=emb_u)
        edit_def = sd.sdedit(x_def.clamp(0, 1), emb, noise, EDIT_STEPS,
                             strength=EDIT_STRENGTH, guidance_scale=EDIT_GUIDANCE,
                             emb_uncond=emb_u)
    so = suite.semantic(edit_orig, item["prompt"])
    sd_ = suite.semantic(edit_def, item["prompt"])
    return ({**{f"fid_{k}": round(v, 4) for k, v in fid.items()},
             "edit_lpips": round(float(suite.pairwise(edit_orig, edit_def)["lpips"]), 4),
             "edit_clip_orig": round(so["clip"], 4),
             "edit_clip_def": round(sd_["clip"], 4),
             "edit_clip_drop": round(so["clip"] - sd_["clip"], 4),
             "edit_siglip_orig": round(so["siglip"], 4),
             "edit_siglip_def": round(sd_["siglip"], 4),
             "edit_siglip_drop": round(so["siglip"] - sd_["siglip"], 4)},
            edit_orig, edit_def)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--target", type=Path, default=Path(TARGET_IMAGE))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="用 prompts.yaml 的第幾個編輯 prompt（0 改內容、1 改場景）")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)
    y_target = load_image_tensor(args.target, sd.device,
                                           size=RESOLUTION)

    dataset = load_dataset(args.data, prompt_index=args.prompt_index)
    if args.images:
        dataset = [d for d in dataset if d["name"] in args.images]
    conds = args.conditions or CONDITIONS

    rows = []
    for item in dataset:
        item["path01"] = load_image_tensor(item["path"], sd.device,
                                                     size=RESOLUTION)
        save_image(item["path01"], args.out / f"{item['name']}__orig.png")
        print(f"\n########## {item['name']} ({item['class']}) ##########", flush=True)
        for cond in conds:
            print(f"=== {item['name']} / {cond} ===", flush=True)
            t0 = time.time()
            res = (run_weak_baseline(sd, item, y_target, args.seed,
                                     PARAMETERIZATION[cond])
                   if cond in PARAMETERIZATION
                   else run_additive(sd, item, cond, args.seed))
            metrics, eo, ed = evaluate(sd, suite, aes, item, res["x_def"])
            for tag, img in (("def", res["x_def"]), ("edit_orig", eo), ("edit_def", ed)):
                save_image(img, args.out / f"{item['name']}__{cond}__{tag}.png")
            row = {"image": item["name"], "condition": cond,
                   "stage1_seconds": round(res["stage1_seconds"], 1),
                   "total_seconds": round(time.time() - t0, 1), **metrics}
            rows.append(row)
            print(row, flush=True)
            write_csv(args.out / "results.csv", rows)
    print(f"\n表：{args.out / 'results.csv'}")


if __name__ == "__main__":
    main()
