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

# 排程／步數不再逐架構分開設定：兩者共用 `NativeStage2Config` 的
# `schedule_steps=50`／`steps=11`／`guidance_steps=10`（APA-GC 的原生
# 操作點），使「架構」是唯一的變因。2026-08-12 移除 DDIM_STEPS 等四個
# 常數，理由見 `run_native` 內的註解與 apa_native_stage2 的 `steps` 欄位。

# 保真/抗編輯評測參數，沿用 EXP-s3t20 那條 SD v1.4/512² 軌
# 2026-08-12 由 0.4 改為 0.55（使用者裁決，依 `scripts/apa_native_edit_sweep.py`
# 的掃描結果）。0.4 下 butterfly 的未防禦編輯與原圖幾乎沒有差別——攻擊本身
# 沒有發生，而防禦強弱是拿「防禦後的編輯」對「未防禦的編輯」比的，分母不成立
# 時整組抗編輯數字都沒有意義（`data/lo_aligned/prompts.yaml` 記過同一個失效
# 模式）。掃描實測：0.55 下 butterfly 出現明顯的帝王蝶＋紅玫瑰、coot 變成
# 白天鵝，且**兩者構圖仍認得出是同一張照片**；0.7 則構圖整個換掉，反而讓
# 「編輯成功與否」不再能歸因到防禦。prompt 不動，問題只在強度。
EDIT_STRENGTH = 0.55
EDIT_GUIDANCE = 7.5
EDIT_STEPS = 30
EDIT_SEED = 20260812

NATIVE_CONDITIONS = [
    # (名稱, 階段一, 架構, 階段二覆寫)
    ("nativeLoRA_DDIM", "lora", "ddim", {}),
    ("nativeLoRA_BDIA", "lora", "bdia", {}),
    ("ourStage1_DDIM", "z*", "ddim", {}),
    ("ourStage1_BDIA", "z*", "bdia", {}),

    # ---- 2026-08-12 新增三組（使用者裁決，與上面四格同一批跑）----
    #
    # 【實驗二】階段二的訓練方法。M1 = 上面四格（ball + sign），M4 = apa_pj
    # （影像空間投影 + Adam），兩者已有數據；這裡補中間兩格，使
    # 「保真控制」與「更新規則」兩個因子可以各自分離：
    #   M1→M2 只換保真控制、M2→M3 只換更新規則、M3→M4 只換施加方式。
    #
    # M2 是刻意設計的**對照組**：sign 丟掉梯度大小，故位移量恆為 µ×iter，
    # 與 reward 裡放什麼無關。它應該與 M1 幾乎同失真、只有方向不同；
    # 若實測不是如此，代表對機制的理解有誤，要先查清楚再往下走。
    ("M2_soft_sign_DDIM", "z*", "ddim",
     dict(fidelity_mode="soft", update_rule="sign")),
    ("M2_soft_sign_BDIA", "z*", "bdia",
     dict(fidelity_mode="soft", update_rule="sign")),
    ("M3_soft_adam_DDIM", "z*", "ddim",
     dict(fidelity_mode="soft", update_rule="adam")),
    ("M3_soft_adam_BDIA", "z*", "bdia",
     dict(fidelity_mode="soft", update_rule="adam")),

    # 【新臂 A】換目標函數：targeted_output（PhotoGuard-c／Mist 的形式）。
    # 動機：本輪八條件中唯一在兩張圖、兩個語意指標上都為正的是 Mist，
    # 而它用的正是這個形式；FND-024 已排除「大失真本身就夠」，故剩下的
    # 可解釋變因是目標函數。階段二機制其餘部分維持原生。
    ("A_targeted_DDIM", "z*", "ddim", dict(reward_mode="targeted")),
    ("A_targeted_BDIA", "z*", "bdia", dict(reward_mode="targeted")),

    # 【新臂 B】**連 reward 都用原生**：替代分類器的 cross-entropy
    # （ResNet-50、ImageNet 標籤取自 APA 官方 data.json），untargeted。
    # 這一臂問的是一個沒人量過的問題：APA 原文的攻擊目標是「誤導分類器」，
    # 與本專案的威脅模型（抗文字引導編輯）不同，兩者有沒有交集？
    ("B_classifier_DDIM", "z*", "ddim", dict(reward_mode="classifier")),
    ("B_classifier_BDIA", "z*", "bdia", dict(reward_mode="classifier")),

    # ---- 「完全原生 APA」錨點與它的單因子變體（2026-08-12）----
    #
    # 先前每一格都至少改了一處（`nativeLoRA_*` 換了 reward、`B_classifier_*`
    # 換了階段一），於是「只改一處」的比較沒有基準可對。這裡補上真正的
    # 原生格：APA 自己的階段一 LoRA ＋ 自己的 reward（分類器 CE）＋
    # 自己的約束（latent 球）＋ 自己的更新規則（sign）。
    ("NATIVE_full_DDIM", "lora", "ddim", dict(reward_mode="classifier")),
    ("NATIVE_full_BDIA", "lora", "bdia", dict(reward_mode="classifier")),
    # 只改 reward → targeted
    ("NF_targeted_DDIM", "lora", "ddim", dict(reward_mode="targeted")),
    ("NF_targeted_BDIA", "lora", "bdia", dict(reward_mode="targeted")),
    # 只改約束 → soft（λ 固定取掃描的中點 8.0，避免再多一個變因）
    ("NF_soft_DDIM", "lora", "ddim",
     dict(reward_mode="classifier", fidelity_mode="soft", dists_lambda=8.0)),
    ("NF_soft_BDIA", "lora", "bdia",
     dict(reward_mode="classifier", fidelity_mode="soft", dists_lambda=8.0)),
    # 只改更新規則 → Adam（**保留 latent 球**，這樣真的只有一處不同）
    ("NF_adam_DDIM", "lora", "ddim",
     dict(reward_mode="classifier", update_rule="adam")),
    ("NF_adam_BDIA", "lora", "bdia",
     dict(reward_mode="classifier", update_rule="adam")),
]
# ---- 2026-08-12 使用者指定的 2×3×2 格點（horse／man／bird）----
# 維度一 階段一：APA 原生 LoRA vs 我方 z*+decoder
# 維度二 loss  ：latent（推離原圖 latent）／target（推向固定目標圖）／CLIP
# 維度三 約束+更新：原生（latent 球 + sign + L1 動量）vs 我方（DISTS 進 loss + Adam）
# 架構固定 DDIM——本輪已測得 DDIM 在此淺噪聲帶上 4/4 優於 BDIA（FND-029），
# 再帶一個已知的變因進來只會稀釋這 12 組的可讀性。
for _s1, _s1tag in (("lora", "L"), ("z*", "Z")):
    for _rw in ("latent", "targeted", "clip"):
        for _fm, _ur, _mtag in (("ball", "sign", "native"), ("soft", "adam", "ours")):
            NATIVE_CONDITIONS.append((
                f"G_{_s1tag}_{_rw}_{_mtag}", _s1, "ddim",
                dict(reward_mode=_rw, fidelity_mode=_fm, update_rule=_ur,
                     **({"dists_lambda": 8.0} if _fm == "soft" else {}))))

BASELINE_CONDITIONS = ["photoguard_c", "mist", "dia_r"]
ALL_CONDITIONS = [c[0] for c in NATIVE_CONDITIONS] + BASELINE_CONDITIONS

# soft 模式的 λ 掃描值。**不做二分**：M2 的位移量依機制推論與 λ 無關，
# 二分在它上面不會收斂；M3 則用掃描點事後挑「DISTS 最接近 apa_pj」的那一格
# 做匹配失真比較，這比二分省一半機時而結論相同。
DISTS_LAMBDA_SWEEP = (0.5, 2.0, 8.0, 32.0, 128.0)

# targeted 臂的目標影像，與專案既有 `targeted_output` 條件同一張
# （`executors.RunConfig.target_image` 的預設）。
TARGET_IMAGE = "data/targets/gray.png"

RECON_KEY = "lpips"
RECON_W_PIXEL = 0.5
RECON_A1_STEPS, RECON_A1_LR = 300, 0.02
RECON_A2_STEPS, RECON_A2_LR = 300, 2e-3
RECON_GAMMA_ACUT, RECON_ACUT_BAND = 1.0, 0.05
RECON_FLOOR_RATIO = 0.50


def load_dataset(root: Path = None) -> list:
    """`root` 給定時讀 `data/lo_aligned` 版面（每類一個子目錄、無 ImageNet
    標籤）；不給時讀 `data/apa_native`（平放 + provenance.json）。

    兩種版面分開處理而不是統一：APA 那批的類別名與標籤來自官方 `data.json`，
    是重現的一部分；lo_aligned 是本專案自己的資料集，沒有也不需要標籤
    （分類器 reward 不在用它的格點裡）。
    """
    if root is not None:
        prompts = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
        out = []
        for cls in sorted(prompts):
            for img in sorted((root / cls).glob("*.png")):
                out.append({"name": img.stem, "class": prompts[cls]["content"],
                            "path": img, "content": prompts[cls]["content"],
                            "prompt": prompts[cls]["prompts"][0], "label": None})
        return out
    provenance = json.loads((DATA_DIR / "provenance.json").read_text(encoding="utf-8"))
    prompts = yaml.safe_load((DATA_DIR / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for p in provenance:
        name = p["output"][:-4]
        out.append({
            "name": name, "class": p["apa_class"], "path": DATA_DIR / p["output"],
            "content": prompts[name]["content"], "prompt": prompts[name]["prompts"][0],
            "label": p["apa_label"],
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


def run_native(sd, item, stage1: str, arch: str, seed: int,
               overrides: dict = None, res=None) -> dict:
    """`overrides` 覆寫 `NativeStage2Config` 的欄位（reward_mode／
    fidelity_mode／update_rule／dists_lambda）。`res` 帶 reward 需要的
    外部素材（分類器、目標影像、可微 DISTS），由呼叫端建一次共用——
    ResNet-50 與 DISTS 各自都要載權重，逐格重建會讓每格多付數秒。"""
    overrides = overrides or {}
    res = res or {}
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

    # **兩種架構共用同一個排程與同一個反演深度**，差別只在遞迴公式。
    # 這是 2026-08-12 的修訂：先前 DDIM 走 50 步全範圍、BDIA 走 20 步
    # t_max=500，兩者的噪聲帶與步數都不同，「架構」這個變因因此混進了
    # 噪聲深度與步數兩個額外變因。現行取 APA-GC 的原生操作點
    # （50 格排程只執行前 11 格、T_a=10），兩種架構同時套用。
    emb_cond = sd.encode_text(item["class"])
    cfg = NativeStage2Config(use_bdia=(arch == "bdia"), use_ckpt=True,
                             **overrides)
    ts = sd.timesteps(cfg.schedule_steps, t_max=cfg.t_max)
    with torch.no_grad():
        if arch == "bdia":
            z_T, z_prev = sd.bdia_inversion(ori_latents, emb_cond, ts, cfg.steps)
        else:
            z_T = sd.ddim_inversion(ori_latents, emb_cond, ts, cfg.steps)
            z_prev = None
    label = (torch.tensor([item["label"]], device=sd.device)
             if item.get("label") is not None else None)
    x_def, history = attack_native(
        sd, z_T, z_prev, ori_latents, item["class"], item["content"], cfg,
        seed=seed, log_every=2, x01=item["path01"],
        y_target=res.get("y_target"), clf=res.get("clf"), label=label,
        dists_module=res.get("dists_module"), clip_pack=res.get("clip_pack"))

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
        # mist 的 fused mode 把兩次 VAE 編碼與一次完整 UNet 前向放在同一張
        # 計算圖上，故兩個 checkpoint 開關都要給（`executors.baseline_kwargs`
        # 同一條規則）。checkpoint 是數值中性的，只換記憶體與時間。
        kw = {"use_ckpt": True, "vae_ckpt": True,
             "target01": executors.load_image_tensor(
                 Path("data/targets/MIST.png"), sd.device, size=RESOLUTION)}
    elif name == "dia_r":
        # **兩個都必須為 True。** DIA-R 把整條反演再加整條重建留在同一張圖上，
        # 圖上另有一次 VAE 編碼與一次 VAE 解碼。
        #
        # 2026-08-12 修正。before：兩者皆 False，三張圖全部以
        # `torch.OutOfMemoryError`（23.54/23.56 GiB 用盡）中止於 UNet 的 FFN。
        # `executors.baseline_kwargs` 早已記載這條規則（2026-08-07 那一筆
        # 補的正是「只包 UNet 仍不夠，dia_r 改在 vae.decode OOM」），
        # 本腳本沒走那個函式而自行組 kwargs，於是漏掉。
        kw = {"use_ckpt": True, "vae_ckpt": True}
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
    ap.add_argument("--images", nargs="+", default=None,
                    help="只跑這些影像（供多 GPU 平行分片用），預設全部三張")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="只跑這些條件名稱（見 ALL_CONDITIONS），預設全部")
    ap.add_argument("--data", type=Path, default=None,
                    help="改讀 lo_aligned 版面的資料集根目錄（每類一子目錄）")
    ap.add_argument("--lam", type=float, default=None,
                    help="soft 模式只跑這一個 λ。不給時跑 DISTS_LAMBDA_SWEEP 全部")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)
    get_ori_latents._suite = suite

    # reward 需要的外部素材建一次共用。分類器與目標影像只在對應的臂才會
    # 被用到，但建構成本低（ResNet-50 一次載入），比逐格判斷要不要建簡單。
    shared = {
        "dists_module": suite.dists_module,
        "y_target": executors.load_image_tensor(
            Path(TARGET_IMAGE), sd.device, size=RESOLUTION),
        "clf": None,
    }

    dataset = load_dataset(args.data)
    if args.images:
        dataset = [it for it in dataset if it["name"] in args.images]
    native_conditions = NATIVE_CONDITIONS
    baseline_conditions = BASELINE_CONDITIONS
    if args.conditions:
        native_conditions = [c for c in NATIVE_CONDITIONS if c[0] in args.conditions]
        baseline_conditions = [c for c in BASELINE_CONDITIONS if c in args.conditions]
    if any(c[3].get("reward_mode") == "classifier" for c in native_conditions):
        from src.defense.apa_native_stage2 import load_source_classifier
        shared["clf"] = load_source_classifier(sd.device)

    # soft 模式的格子展開成 λ 掃描：同一個條件名稱配多個 λ，落盤時帶
    # `dists_lambda` 欄位，事後才能挑「DISTS 最接近 apa_pj」的那一格做
    # 匹配失真比較。ball 模式不吃 λ，維持單格。
    expanded = []
    for name, st1, arch, ov in native_conditions:
        if (ov.get("fidelity_mode") == "soft" and args.lam is None
                and "dists_lambda" not in ov):
            for lam in DISTS_LAMBDA_SWEEP:
                expanded.append((f"{name}_lam{lam:g}", st1, arch,
                                 {**ov, "dists_lambda": lam}))
        elif ov.get("fidelity_mode") == "soft":
            # λ 的來源有兩個且**條件自帶的優先**：`--lam` 是整批覆寫，
            # 條件裡的 `dists_lambda` 是該條件定義的一部分（例如 2×3×2 格點
            # 固定取 8.0，避免多一個變因）。
            #
            # 2026-08-12 修正。before：這一支無條件格式化 `args.lam`，而
            # 條件自帶 λ 時 `args.lam` 是 None，於是以
            # `TypeError: unsupported format string passed to NoneType.__format__`
            # 中止——9 個 shard 全在載完模型之後才炸。
            lam = args.lam if args.lam is not None else ov["dists_lambda"]
            suffix = "" if args.lam is None else f"_lam{lam:g}"
            expanded.append((f"{name}{suffix}", st1, arch,
                             {**ov, "dists_lambda": lam}))
        else:
            expanded.append((name, st1, arch, ov))
    native_conditions = expanded

    rows = []
    for item in dataset:
        item["path01"] = executors.load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        save_image(item["path01"], args.out / f"{item['name']}__orig.png")
        print(f"\n########## {item['name']} ({item['class']}) ##########", flush=True)

        if any(c[3].get("reward_mode") == "clip" for c in native_conditions):
            from src.defense.apa_native_stage2 import build_clip_pack
            shared["clip_pack"] = build_clip_pack(suite, item["content"], sd.device)

        for cond_name, stage1, arch, ov in native_conditions:
            print(f"=== {item['name']} / {cond_name} ===", flush=True)
            t0 = time.time()
            res = run_native(sd, item, stage1, arch, args.seed,
                             overrides=ov, res=shared)
            total_seconds = time.time() - t0
            metrics, edit_orig, edit_def = evaluate(sd, suite, aes, item, res["x_def"])
            save_image(res["x_def"], args.out / f"{item['name']}__{cond_name}__def.png")
            save_image(edit_orig, args.out / f"{item['name']}__{cond_name}__edit_orig.png")
            save_image(edit_def, args.out / f"{item['name']}__{cond_name}__edit_def.png")
            row = {"image": item["name"], "condition": cond_name,
                  "reward_mode": ov.get("reward_mode", "attn"),
                  "fidelity_mode": ov.get("fidelity_mode", "ball"),
                  "update_rule": ov.get("update_rule", "sign"),
                  "dists_lambda": ov.get("dists_lambda", ""),
                  "stage1_seconds": round(res["stage1_seconds"], 1),
                  "total_seconds": round(total_seconds, 1), **metrics}
            rows.append(row)
            print(row, flush=True)
            executors.write_csv(args.out / "apa_native_full.csv", rows)

        for name in baseline_conditions:
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
