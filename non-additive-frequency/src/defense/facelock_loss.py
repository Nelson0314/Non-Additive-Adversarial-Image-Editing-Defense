"""身分損失接在 **VAE 的一次往返**上，不經過 UNet。

移植自 *Edit Away and My Face Will not Stay*（CVPR 2025，arXiv:2411.16832），
逐行對照官方 `methods.py::facelock`。

為什麼這個組合本專案沒有跑過
────────────────────────────────────────────────────────────────────
判準（臉）與模組（走到哪裡）是兩個獨立的軸，本專案此前只填了對角線：

    損失                    經過的模組                 目標
    encoder_target          VAE 編碼器                 latent 的位置
    latent_norm             VAE 編碼器                 latent 的長度
    image_guidance          VAE 編碼器 → UNet          影像引導項消失
    identity                VAE 編碼器 → UNet → 解碼   x̂₀ 上的臉
    **本項**                **VAE 編碼器 → 解碼器**    **重建圖上的臉**

`latent_norm` 有「只走 VAE」那一半但沒有臉；`identity` 有臉那一半但走了 UNet
（而且實測明顯較差：8/25 對代理損失的 14/25）。**兩者都不是這一格。**

不經過 UNet 的代價與好處都要講清楚：單張成本掉一到兩個數量級（沒有擴散取樣
的前向與反向），但目標也從「編輯會壞掉」換成「這張圖過一次自編碼器就不是
那個人」——**後者是前者的必要條件而不是充分條件**，模型是否真的會在編輯時
沿用那個被破壞的重建，這個損失不保證。

一個本專案特有、原論文沒有的結構
────────────────────────────────────────────────────────────────────
FaceLock 的擾動是**全圖**的 L∞ 球，臉本身也被改。本專案的威脅模型把受保護
主體凍結（`S ∩ M = ∅`，支撐外逐位元等於原圖），**臉在像素上完全沒有被動到**。

於是 `D(E(x'))` 的臉只能經由 **VAE 的感受野**被支撐內的內容影響——衣物上的
圖樣要隔著空間去改變重建出來的臉。**這一批要問的就是那件事做不做得到。**
做不到的話這個損失在本專案的載體上恆為常數，梯度為零，`stuck_at_init` 會滿。

式子
────────────────────────────────────────────────────────────────────
論文正文（式 3）寫的是兩項，且是**梯度上升**：

    δ = argmax_{‖δ‖∞ ≤ ε}  f_FR( D(E(x+δ)), x ) + λ · f_FE( D(E(x+δ)), x )

官方程式碼另有兩項與兩個啟動排程，**正文未載**（`SOURCE_AUDIT` 意義下的
「原始碼有、論文沒有」）：

    loss = − FR(D(E(x')), x) · [i ≥ 0.35N]
         + w_lat · ‖E(x') − E(x)‖²
         + w_lpips · LPIPS(D(E(x')), x) · [i > 0.25N]

`run_param_pgd` 一律**最小化**，故本模組回傳的是上式取負：

    L(x') =  cos_id( D(E(x'))|_face , x|_face ) · [i ≥ 0.35N]
          −  w_lat  · ‖E(x') − E(x)‖²
          −  w_lpips· LPIPS( D(E(x')), x ) · [i > 0.25N]

**符號寫反不會拋錯**，只會安靜地把防禦圖推向「更像本人」而曲線看起來仍在
下降。守門在 `tests/test_facelock_loss.py`。

三個不可比的地方，出表時要標
────────────────────────────────────────────────────────────────────
1. **預算**。原文 `eps=0.02`／`step=0.003`／100 步，官方程式碼預設
   `eps=0.03`／`step=0.01`。本專案的補丁族沒有 L∞ 球，支撐內只夾在 [0,1]，
   支撐外逐位元不動。**兩者的「預算」不是同一個量。**
2. **臉的裁切**。本模組走 `identity_loss.face_box` 的固定方框（可微），
   FaceLock 走 `cvlface` 的對齊器。`src/metrics/identity.py` 的讀數走 MTCNN。
   **三條路徑的絕對值都不可互比**，本項只用來最佳化。
3. **辨識器**。本模組用 facenet-pytorch 的 InceptionResnetV1（VGGFace2），
   FaceLock 用 AdaFace ViT-Base ＋ KPRPE（WebFace4M）。
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from src.defense.identity_loss import embed_crop, face_box

#: 官方 `methods.py` 的兩個權重。論文正文只給 λ 一個符號，未給數值。
FACELOCK_W_LATENT = 0.2
FACELOCK_W_LPIPS = 1.0
#: 官方 `methods.py` 的兩個啟動比例。**正文未載。**
FACELOCK_START_FR = 0.35
FACELOCK_START_LPIPS = 0.25

_LPIPS = None


def _lpips_net(device):
    """VGG backbone——與 FaceLock 的 `lpips.LPIPS(net='vgg')` 相同。

    **不共用 `src/metrics/suite.py` 的實例**：那一份在 `no_grad` 底下用，
    這裡要梯度；共用時別處改了 `requires_grad_` 會互相影響而看不出來。
    """
    global _LPIPS
    if _LPIPS is None:
        import lpips as _l

        _LPIPS = _l.LPIPS(net="vgg").eval().to(device)
        for p in _LPIPS.parameters():
            p.requires_grad_(False)
    return _LPIPS


def make_facelock_loss(
    ip2p,
    *,
    x_clean: torch.Tensor,
    face_mask: torch.Tensor,
    steps: int,
    box_margin: float = 0.35,
    w_latent: float = FACELOCK_W_LATENT,
    w_lpips: float = FACELOCK_W_LPIPS,
    start_fr: float = FACELOCK_START_FR,
    start_lpips: float = FACELOCK_START_LPIPS,
    decode_ckpt: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `loss(x_def01) -> 純量`，**要最小化**。

    掛在回傳物件上的 `_advance(i)` 供 `run_param_pgd` 的 `step_hook` 呼叫，
    兩個啟動排程靠它知道現在第幾步。**沒接上時兩項恆為開啟**——那是一個
    會靜默改變目標的差異，故 `_advance` 未被呼叫過時 `step()` 回傳 `steps`
    （視為已過所有啟動點），並在 `geometry()` 那一側由呼叫端記錄。
    """
    if steps < 1:
        raise ValueError(f"steps 必須至少為 1，收到 {steps}")
    for name, w in (("w_latent", w_latent), ("w_lpips", w_lpips)):
        if w < 0.0:
            raise ValueError(f"{name} 必須非負，收到 {w}")
    for name, s in (("start_fr", start_fr), ("start_lpips", start_lpips)):
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"{name} 必須落在 [0,1]，收到 {s}")

    device = x_clean.device
    box = face_box(face_mask, margin=box_margin)

    with torch.no_grad():
        ref_id = embed_crop(x_clean, box, device).detach()
        ref_lat = ip2p.encode_image(x_clean).detach()

    state = {"i": None}

    def _advance(i: int) -> None:
        state["i"] = int(i)

    def _on(start: float) -> bool:
        # `_advance` 沒被接上時視為全開，見 docstring。
        i = state["i"]
        return True if i is None else i >= start * steps

    def loss(x_def: torch.Tensor) -> torch.Tensor:
        lat = ip2p.encode_image(x_def)
        rec = ip2p.decode_latent(lat, use_ckpt=decode_ckpt)
        # `decode_latent` 回的是 [-1,1]，與 `encode_image` 的輸入約定相反。
        rec01 = (rec + 1.0) * 0.5
        total = x_def.sum() * 0.0            # 保持在計算圖上，形狀為純量

        if _on(start_fr):
            emb = embed_crop(rec01.clamp(0, 1), box, device)
            total = total + torch.dot(emb, ref_id) / (
                emb.norm() * ref_id.norm() + 1e-12)
        if w_latent > 0.0:
            total = total - w_latent * (lat - ref_lat).pow(2).mean()
        if w_lpips > 0.0 and _on(start_lpips):
            d = _lpips_net(device)(
                rec01.clamp(0, 1) * 2.0 - 1.0, x_clean * 2.0 - 1.0)
            total = total - w_lpips * d.mean()
        return total

    loss._advance = _advance                                  # type: ignore[attr-defined]
    loss._facelock_box = box                                  # type: ignore[attr-defined]
    return loss
