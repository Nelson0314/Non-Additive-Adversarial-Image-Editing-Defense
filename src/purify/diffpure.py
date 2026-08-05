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

**解析度裁決（`SOURCE_AUDIT_2026-08-05.md` §9 第 4 項）**：本專案在 1024² 下評測，
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


def has_diffpure_weights(ckpt=None) -> bool:
    """權重與相依套件是否到位。目前恆為 False，見 `diffpure_real` 的說明。"""
    return False


def diffpure_real(x01: torch.Tensor, t: int = DIFFPURE_T_DEFAULT, ckpt=None) -> torch.Tensor:
    """真實 DiffPure。**目前無法執行**，缺項逐條列於例外訊息中。

    介面已定案：輸入輸出為 `(B,3,H,W)`、RGB、`[0,1]`；內部應換算到 `[-1,1]`，
    在 256² 上跑 `t` 步，再由 `resize_roundtrip` 升回原尺寸。
    """
    raise NotImplementedError(
        "DiffPure 無法執行，缺以下項目（不得以其他擴散模型或自寫近似替代）：\n"
        f"  1. 檢查點 {DIFFPURE_CHECKPOINT}（guided-diffusion 的 256×256 無條件模型，"
        f"來源 {DIFFPURE_CHECKPOINT_SOURCE}，MIT）。\n"
        "  2. `guided_diffusion` 套件本身（`create_model_and_diffusion`、`p_sample`），"
        "或 NVlabs/DiffPure repo 內的 `runners/diffpure_guided.py`。\n"
        "  3. 若要用論文主結果的 SDE 版，另需 `torchsde`"
        "（`sdeint_adjoint`, method='euler', dt=1e-3），目前環境未安裝。\n"
        f"  設定已定案：t={t}（ImageNet），sample_step={DIFFPURE_SAMPLE_STEP}，"
        "clip_denoised=True，值域 [-1,1]，"
        f"解析度經 resize_roundtrip（{RESIZE_MODE}, antialias={RESIZE_ANTIALIAS}, "
        f"{DIFFPURE_RESOLUTION}²）——最後這一項為我方指定，非 DiffPure 原設定。"
    )
