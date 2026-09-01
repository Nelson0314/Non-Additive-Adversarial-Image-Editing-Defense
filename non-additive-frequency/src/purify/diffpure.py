"""DiffPure 淨化算子與其 resize 對照（Nie et al., ICML 2022，`NVlabs/DiffPure`）。

出處與參數查證見 `docs/_audit_purify.md` §3。摘要：

- 演算法：先一次加噪到 `t`（`x = x0·√ᾱ_{t-1} + e·√(1−ᾱ_{t-1})`），再逆向去噪回 0。
  guided 版逐步呼叫 `diffusion.p_sample(..., clip_denoised=True)`；SDE 版以
  `torchsde.sdeint_adjoint(..., method='euler')` 解 reverse-VP-SDE（論文主結果）。
- `t`：CIFAR-10 為 100、**ImageNet 為 150**、CelebA-HQ 為 500。`--t 400` 是 argparse
  預設但被所有實驗腳本覆寫，不是任何一組實驗設定。
- 檢查點：ImageNet 用 guided-diffusion 的 `256x256_diffusion_uncond.pt`，
  **官方三個檢查點最高 256×256**，沒有 512² 或 1024² 的設定可依循。
- 值域 `[-1, 1]`，形狀 `(B,C,H,W)`。

**解析度裁決（`reference/SOURCE_AUDIT.md` §9 第 4 項）**：本專案在 1024² 下評測，
作法為「降取樣到 256 → 淨化 → 升回原尺寸」，並**額外提供 `resize_only`**：
同樣的降升取樣但不做擴散淨化，用來把 resize 本身的破壞力與擴散淨化分開。
兩者共用本模組的 `resize_roundtrip`，故降升取樣參數必然一致。

**我方指定的部分（DiffPure 原文與原始碼皆無 resize，故無來源可依循）**：
插值方法 `bicubic`、降取樣開 antialias、每一段後 clamp 回 `[0,1]`。
此三項須在論文與報表標註為我方指定。

**可微性**：SDE 版以 adjoint 法反傳，原生可微；guided 版包在 `torch.no_grad()` 內，
不可微。本專案在權重到位前不宣稱任何一種，`differentiable` 取 False，訓練側走
`straight_through`。`resize_only` 只有 `F.interpolate`，原生可微。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# 依原始碼（run_scripts/*）：資料集 → --t
DIFFPURE_T = {"cifar10": 100, "imagenet": 150, "celebahq": 500}
DIFFPURE_T_DEFAULT = DIFFPURE_T["imagenet"]  # 本專案為自然影像，取 ImageNet 那組
DIFFPURE_SAMPLE_STEP = 1                     # eval_sde_adv.py argparse 預設
DIFFPURE_CHECKPOINT = "256x256_diffusion_uncond.pt"
DIFFPURE_CHECKPOINT_SOURCE = "https://github.com/openai/guided-diffusion"

# 以下三項為我方指定（DiffPure 無 resize 步驟，無來源）
DIFFPURE_RESOLUTION = 256      # 檢查點的原生解析度
RESIZE_MODE = "bicubic"
RESIZE_ANTIALIAS = True


def resize_roundtrip(
    x01: torch.Tensor,
    inner=None,
    size: int = DIFFPURE_RESOLUTION,
    mode: str = RESIZE_MODE,
    antialias: bool = RESIZE_ANTIALIAS,
) -> torch.Tensor:
    """降取樣到 `size`²、（可選）套用 `inner`、再升回原尺寸。

    `inner=None` 即 `resize_only` 對照組；`inner` 傳入擴散淨化即 DiffPure。
    兩條路徑共用同一函式，確保降升取樣參數逐項一致。
    """
    if x01.dim() != 4:
        raise ValueError(f"需要 (B,C,H,W) 張量，收到 {tuple(x01.shape)}")
    h, w = x01.shape[-2:]
    small = F.interpolate(x01, size=(size, size), mode=mode, antialias=antialias).clamp(0, 1)
    if inner is not None:
        small = inner(small)
    return F.interpolate(small, size=(h, w), mode=mode, antialias=antialias).clamp(0, 1)


def resize_only(x01: torch.Tensor, **kw) -> torch.Tensor:
    """DiffPure 的解析度對照：只做同樣的降升取樣，不淨化。原生可微。"""
    return resize_roundtrip(x01, inner=None, **kw)


def resize_params() -> dict:
    """回報本模組實際使用的降升取樣參數，供對照組一致性檢查。"""
    return {"size": DIFFPURE_RESOLUTION, "mode": RESIZE_MODE, "antialias": RESIZE_ANTIALIAS}


# `NVlabs/DiffPure` 的 `configs/imagenet.yml` 的 `model:` 區塊，逐欄照抄
# （2026-08-05 由 raw 檔核對）。這些值覆寫 `model_and_diffusion_defaults()`。
# **不得憑 `256x256_diffusion_uncond.pt` 這個檔名推測參數**：guided-diffusion
# 同一解析度有 `learn_sigma`／`attention_resolutions` 不同的多組設定，
# 猜錯會讓 `load_state_dict` 以形狀不符中止（有症狀），或更糟——形狀碰巧
# 相符而權重對應到別的層（無症狀）。
#
# 這組設定已與檢查點交叉核對（2026-08-05，不需下載即可驗證）：
# `create_model_and_diffusion(**cfg)` 的 `state_dict` 共 552,814,086 個元素，
# fp32 下 2,211,256,344 bytes；檢查點的 HTTP Content-Length 為 2,211,383,297，
# 差 126,953 bytes 即 zip 容器與 pickle 的開銷（比值 0.99994）。
# 由 `tests/test_purify_new_ops.py::test_DiffPure的設定與檢查點大小相符` 釘住。
DIFFPURE_MODEL_CONFIG = {
    "attention_resolutions": "32,16,8",
    "class_cond": False,
    "diffusion_steps": 1000,
    "rescale_timesteps": True,
    "timestep_respacing": "1000",
    "image_size": 256,
    "learn_sigma": True,
    "noise_schedule": "linear",
    "num_channels": 256,
    "num_head_channels": 64,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_fp16": False,     # 見 `_load_guided`：本專案在 fp32 下跑淨化
    "use_scale_shift_norm": True,
}

# 檢查點的預設位置。以環境變數覆寫，因為 GPU 機器與本機的路徑不同，
# 而把路徑寫進入庫檔案會讓「這批資料用的是哪份權重」隨機器而變。
DIFFPURE_CKPT_ENV = "DIFFPURE_CKPT"

_CACHE: dict = {}


def diffpure_checkpoint_path(ckpt=None):
    """檢查點路徑。`ckpt` > 環境變數 `DIFFPURE_CKPT` > `None`。"""
    import os
    from pathlib import Path

    if ckpt:
        return Path(ckpt)
    env = os.environ.get(DIFFPURE_CKPT_ENV)
    return Path(env) if env else None


def has_diffpure_weights(ckpt=None) -> bool:
    """檢查點與 `guided_diffusion` 套件是否都到位。

    兩者都要檢查。只檢查其一的話，缺的那一項會在**跑到那一格時**才炸，
    而那已經是數小時機時之後——`Purifier.available` 的存在就是為了讓
    `annotate_unavailable` 在跑之前就把那些格標成 skipped。
    """
    from importlib.util import find_spec

    p = diffpure_checkpoint_path(ckpt)
    if p is None or not p.exists():
        return False
    return find_spec("guided_diffusion") is not None


def _load_guided(ckpt=None, device=None):
    """載入 guided-diffusion 的 256² 無條件模型。**同一個行程只載入一次。**

    模型約 2.2 GB，逐格重載會讓 4,050 格的評測完全被 I/O 綁住。
    以 (路徑, 裝置) 為鍵快取。

    `use_fp16=False`：`convert_to_fp16` 只在 CUDA 上有意義，而淨化是評測
    路徑的一部分，其數值必須與精度無關才不會在 `precision_equiv` 之外
    另外引入一個變因。速度差異相對 4,050 格的總成本可忽略。
    """
    import torch as _torch
    from guided_diffusion.script_util import (
        create_model_and_diffusion, model_and_diffusion_defaults,
    )

    p = diffpure_checkpoint_path(ckpt)
    if p is None or not p.exists():
        raise FileNotFoundError(
            f"找不到 DiffPure 的檢查點。請設環境變數 {DIFFPURE_CKPT_ENV} 指向 "
            f"{DIFFPURE_CHECKPOINT}，或以 `scripts/fetch_diffpure.py` 下載。"
            f"（來源 {DIFFPURE_CHECKPOINT_SOURCE}，2.2 GB，不入版控）"
        )
    key = (str(p), str(device))
    if key in _CACHE:
        return _CACHE[key]

    cfg = model_and_diffusion_defaults()
    cfg.update(DIFFPURE_MODEL_CONFIG)
    model, diffusion = create_model_and_diffusion(**cfg)
    state = _torch.load(p, map_location="cpu")
    # `strict=True`（預設）：少一個鍵或多一個鍵都表示設定與檢查點不配對，
    # 而 `strict=False` 會讓未載入的層留在隨機初始化上——淨化仍然跑得完，
    # 輸出也仍是一張圖，只是那不是 DiffPure。
    model.load_state_dict(state)
    model.requires_grad_(False).eval()
    if device is not None:
        model.to(device)
    _CACHE[key] = (model, diffusion)
    return model, diffusion


def diffpure_real(x01: torch.Tensor, t: int = DIFFPURE_T_DEFAULT,
                  ckpt=None, sample_step: int = DIFFPURE_SAMPLE_STEP,
                  seed=None) -> torch.Tensor:
    """真實 DiffPure（guided 版）。輸入輸出 `(B,3,H,W)`、RGB、`[0,1]`。

    逐行對應 `NVlabs/DiffPure` 的 `runners/diffpure_guided.py`
    `GuidedDiffusion.image_editing_sample`（2026-08-05 由 raw 檔核對）：

        a = (1 - betas).cumprod(0)
        x = x0·√a[t-1] + e·√(1 − a[t-1])              # 一次加噪到 t
        for i in reversed(range(t)):                   # 逐步去噪回 0
            x = diffusion.p_sample(model, x, i, clip_denoised=True)["sample"]

    值域為 `[-1,1]`，故此處先 `x·2−1`、最後 `(x+1)/2`。
    解析度由 `resize_roundtrip` 處理（**我方指定**，DiffPure 原文無 resize）。

    `t` 取 150（ImageNet 那組）。`sample_step` 為外層重複次數，原始碼
    argparse 預設 1；大於 1 時原作把各次結果 `cat` 起來當成多個樣本，
    本專案要的是單一淨化影像，故只取最後一次——這一點與原作的**用途**
    不同（它做的是對抗防禦的隨機平滑），已在此註明。
    """
    if x01.dim() != 4:
        raise ValueError(f"需要 (B,C,H,W) 張量，收到 {tuple(x01.shape)}")
    model, diffusion = _load_guided(ckpt, device=x01.device)
    betas = torch.from_numpy(diffusion.betas).float().to(x01.device)
    a = (1.0 - betas).cumprod(dim=0)
    if not 0 < t <= len(a):
        raise ValueError(f"t={t} 超出 betas 的長度 {len(a)}")

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def inner(small01: torch.Tensor) -> torch.Tensor:
        x0 = small01 * 2.0 - 1.0
        with torch.no_grad():
            for _ in range(max(1, sample_step)):
                e = torch.randn(x0.shape, generator=gen,
                                dtype=x0.dtype).to(x0.device)
                x = x0 * a[t - 1].sqrt() + e * (1.0 - a[t - 1]).sqrt()
                for i in reversed(range(t)):
                    tt = torch.full((x.shape[0],), i, device=x.device,
                                    dtype=torch.long)
                    x = diffusion.p_sample(
                        model, x, tt, clip_denoised=True, denoised_fn=None,
                        cond_fn=None, model_kwargs=None)["sample"]
                x0 = x
        return ((x0 + 1.0) / 2.0).clamp(0, 1)

    return resize_roundtrip(x01, inner=inner)
