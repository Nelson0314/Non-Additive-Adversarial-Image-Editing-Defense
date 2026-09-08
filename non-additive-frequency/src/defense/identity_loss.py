"""把**判準本身**寫進損失：讓模型建出來的那張臉，不再是原來那個人。

為什麼需要它
────────────────────────────────────────────────────────────────────
現行的四個損失都是代理量：`encoder_target`／`latent_norm`／`image_guidance`
把影像條件推向某個點，`edit_divergence` 要求「離原圖的結果遠」。**沒有一個
提到身分**，而判準是身分（`src/metrics/identity.py`）。

這個分歧不是理論上的顧慮，是本專案已經量到的事：內容約束那一批四個工作點
（自由學 14/25、低頻替換 12/25、凍結亮度 7/25、低頻替換＋平舖 5/25）在
`best_eval`（代理量）上的排序，與在身分讀數上的排序**並不一致**；而位移與
身分讀數已經分歧過三次。最直接的補救是讓訓練與判準量同一件事。

    L_id(x') = cos( E_id(x) , E_id( D(x̂₀(z_t ; c_I(x'))) ) )      ← 要最小化

`x̂₀` 是取樣器每一步實際在往哪裡走的那個「模型認為的乾淨影像」，與
`edit_divergence` 用的是同一個量（同一份 `_x0`）。差別在拿它做什麼：那一個
比兩個 `x̂₀` 的像素距離，本項把它**解碼成影像、裁出臉、算身分嵌入**。

三個構造上的選擇，每一個都會靜默失效
────────────────────────────────────────────────────────────────────
1. **參照端是原圖的臉，不是 `x̂₀(原圖)` 的臉。** 判準寫的是
   `cos(E_id(x), E_id(A(x')))`，參照端是**原始照片**。用 `x̂₀(原圖)` 當參照
   會把「模型重建得像不像」也算進去，那不是要量的東西。參照嵌入是常數，
   建構時算一次。

2. **裁臉用固定框，不用 MTCNN。** MTCNN 的偵測不可微，而且它在編輯輸出上
   會偵測失敗（那正是我們要的結果之一）——偵測失敗時損失沒有定義，訓練就
   斷了。故框由**原圖的臉部遮罩**算出來並固定：防禦圖的臉逐位元不動，框是
   已知的。編輯後臉會移動，故框往外放寬 `box_margin`。
   **代價要說清楚**：這個嵌入與 `identity.py` 的讀數不是同一條路徑（那一個
   走 MTCNN 的對齊裁切），故 `L_id` 的絕對值不可與 `id_def` 直接比較。
   它只用來最佳化，判準仍然由事後的讀數給。

3. **符號**：`run_param_pgd` 一律最小化，故回傳的是**餘弦本身**（越小越不像
   同一個人）。寫成負的不會拋錯，只會安靜地把防禦圖推向「更像本人」，
   而報表上的曲線看起來仍然在下降。

雙目標：毀身分、保場景
────────────────────────────────────────────────────────────────────
`layout_weight > 0` 時另加一項：

    L = L_id + λ · mean_{臉之外}( ‖ x̂₀_z(x') − x̂₀_z(x) ‖² )

第二項要求「模型從防禦圖建出來的場景，在臉以外要與從原圖建出來的一樣」。
**在 latent 域算，不解碼**——多一次 VAE decode 的代價比多一次 UNet 前傳還高，
而佈局在 latent 域已經看得出來。

存在理由是產物的形狀：現行方法的成功幾乎都是「輸出整個被毀掉」（變成紅色
人台、商品照），那是鈍器。加上這一項是要求「同一個場景、同一件衣服、同一個
姿勢，但那個人不是他」——那才是這個威脅模型真正要的東西，而且它同時把
「防禦方自己把攻擊做完了」那一類產物排除掉（`iou_person` 此前只是旁證）。
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from src.defense.fixedpoint_loss import _null_embedding, _scheduler_of
from src.defense.image_guidance_loss import ZT_MODES

# facenet-pytorch 的 `post_process=True` 做的固定正規化是 `(x*255 − 127.5)/128`。
# 從 [0,1] 出發就是 `(x − 0.5)/0.50196`。**這個常數要與那個套件一致**，
# 差一點不會拋錯，只會讓嵌入落在訓練分布之外而餘弦全部貼近零。
_FACENET_MEAN, _FACENET_STD = 0.5, 128.0 / 255.0
_FACE_SIDE = 160

_NET = None


def _embedder(device):
    """InceptionResnetV1（vggface2），與 `src/metrics/identity.py` 同一組權重。

    **不共用那個模組的 `_NET`**：那一份在 `torch.no_grad` 底下用，這裡要梯度。
    共用時如果有人在別處改了 `requires_grad_`，兩邊會互相影響而看不出來。
    """
    global _NET
    if _NET is None:
        from facenet_pytorch import InceptionResnetV1

        _NET = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in _NET.parameters():
            p.requires_grad_(False)
    return _NET


def face_box(mask: torch.Tensor, margin: float = 0.35
             ) -> Tuple[int, int, int, int]:
    """(1,1,H,W) 的臉部遮罩 → 放寬之後的方形框 `(top, left, side, side)`。

    取**方形**是因為嵌入網路吃 160×160；先取遮罩的外接矩形，再擴成以它為中心
    的方形並往外放寬 `margin`，最後夾回畫面內。

    遮罩全為零時拋錯——那代表這張影像上沒有偵測到臉，而「沒有臉」的情況下
    這個損失沒有定義。**不回退到整張畫面**：那會讓損失變成「整張圖像不像那個
    人」，與旗標名稱說的不是同一件事。
    """
    idx = torch.nonzero(mask[0, 0] > 0.5)
    if idx.numel() == 0:
        raise ValueError(
            "臉部遮罩是空的：這張影像上沒有可用的臉，identity 損失沒有定義。"
            "**不回退到整張畫面**——那會變成另一個目標而旗標名稱不變。")
    y0, x0 = (int(v) for v in idx.min(dim=0).values)
    y1, x1 = (int(v) for v in idx.max(dim=0).values)
    h, w = mask.shape[-2:]
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    side = max(y1 - y0, x1 - x0) * (1.0 + 2.0 * margin)
    side = int(min(side, min(h, w)))
    top = int(min(max(cy - side / 2.0, 0), h - side))
    left = int(min(max(cx - side / 2.0, 0), w - side))
    return top, left, side, side


def embed_crop(x01: torch.Tensor, box, device) -> torch.Tensor:
    """(1,3,H,W) ＋ 固定框 → (512,) 的身分向量。**可微**。

    裁切與雙線性縮放都是可微的；`identity.py` 走的 MTCNN 對齊裁切不是，
    這是兩條路徑的唯一差別，故兩者的數值不可直接比較（見模組 docstring）。
    """
    top, left, sh, sw = box
    crop = x01[..., top:top + sh, left:left + sw]
    crop = F.interpolate(crop, size=(_FACE_SIDE, _FACE_SIDE),
                         mode="bilinear", align_corners=False)
    norm = (crop.clamp(0, 1) - _FACENET_MEAN) / _FACENET_STD
    return _embedder(device)(norm.to(device))[0]


def make_identity_loss(
    ip2p,
    *,
    zt_mode: str,
    x_clean: torch.Tensor,
    face_mask: torch.Tensor,
    t_min: int = 1,
    t_max: int = 1000,
    samples: int = 1,
    seed: int = 0,
    box_margin: float = 0.35,
    layout_weight: float = 0.0,
    decode_ckpt: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `loss(x_def01) -> 純量`，**要最小化**。

    值越小代表「模型從防禦圖建出來的臉」離原本那個人越遠。
    `layout_weight > 0` 時另加場景保持項，見模組 docstring。
    """
    if zt_mode not in ZT_MODES:
        raise ValueError(f"未知的 zt_mode：{zt_mode!r}，必須是 {ZT_MODES}")
    if not 1 <= t_min <= t_max:
        raise ValueError(f"需要 1 <= t_min <= t_max，收到 {t_min}／{t_max}")
    if samples < 1:
        raise ValueError(f"samples 必須為正整數，收到 {samples}")
    if layout_weight < 0:
        raise ValueError(f"layout_weight 不可為負，收到 {layout_weight}")

    unet, device = ip2p.unet, ip2p.device
    sched = _scheduler_of(ip2p)
    abar = sched.alphas_cumprod.to(device=device, dtype=torch.float32)
    if t_max > len(abar):
        raise ValueError(f"t_max={t_max} 超出排程長度 {len(abar)}")
    null_emb = _null_embedding(ip2p)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    box = face_box(face_mask, box_margin)
    with torch.no_grad():
        # 參照端是**原始照片**的臉，與判準的參照端一致。常數，算一次。
        ref = embed_crop(x_clean.to(device), box, device).detach()
        z_src = ip2p.encode_image(x_clean).detach()
        z_img_src = ip2p.image_latents(x_clean).detach()
        # 臉之外的權重，降到 latent 解析度。`1 − 臉` 而不是 `臉的補集的硬遮罩`
        # ——羽化帶要平滑過渡，硬邊界會在邊上製造一圈高梯度。
        out_w = (1.0 - face_mask.clamp(0, 1)).to(device)

    def _zt(ref_t, step, eps):
        if zt_mode == "noise":
            return eps
        a = abar[step].to(ref_t.dtype)
        return z_src.to(ref_t.dtype) * a.sqrt() + eps * (1.0 - a).sqrt()

    def _x0(z_t, eps_hat, step):
        a = abar[step].to(z_t.dtype)
        return (z_t - (1.0 - a).sqrt() * eps_hat) / a.sqrt()

    def _terms(x_def01, step, eps):
        z_img = ip2p.image_latents(x_def01)
        z_t = _zt(z_img, step, eps)
        tt = torch.tensor([step], device=device, dtype=torch.long)
        emb = null_emb.to(z_t.dtype)
        eps_def = unet(torch.cat([z_t, z_img], dim=1), tt,
                       encoder_hidden_states=emb).sample
        x0_def = _x0(z_t, eps_def, step)
        img = ip2p.decode_latent(x0_def, use_ckpt=decode_ckpt)
        cos = F.cosine_similarity(ref[None], embed_crop(img, box, device)[None])[0]
        if layout_weight <= 0.0:
            return cos
        with torch.no_grad():
            # 原圖那一支不依賴 x_def，是常數。
            eps_src = unet(torch.cat([z_t, z_img_src.to(z_t.dtype)], dim=1), tt,
                           encoder_hidden_states=emb).sample.detach()
            x0_src = _x0(z_t, eps_src, step).detach()
        w = F.adaptive_avg_pool2d(out_w.to(x0_def.dtype), x0_def.shape[-2:])
        denom = w.sum().clamp_min(1.0) * x0_def.shape[1]
        layout = (((x0_def - x0_src) ** 2) * w).sum() / denom
        return cos + layout_weight * layout

    def make_fixed(n_draws: int, eval_seed: int):
        """決定性的評估函數。理由與 `image_guidance` 相同：訓練用的損失每一步
        重抽 `(t, ε)`，逐步值本來就會抖，拿它判收斂會判錯。"""
        g2 = torch.Generator(device="cpu").manual_seed(int(eval_seed))
        steps_fixed = [int(torch.randint(t_min - 1, t_max, (1,), generator=g2))
                       for _ in range(n_draws)]
        eps_fixed = [None] * n_draws

        def fixed(x_def01: torch.Tensor) -> torch.Tensor:
            total = None
            for k, step in enumerate(steps_fixed):
                if eps_fixed[k] is None:
                    shape = ip2p.image_latents(x_def01).shape
                    eps_fixed[k] = torch.randn(
                        shape, generator=g2, dtype=torch.float32).to(device)
                term = _terms(x_def01, step, eps_fixed[k].to(x_def01.dtype))
                total = term if total is None else total + term
            return total / len(steps_fixed)

        return fixed

    def loss(x_def01: torch.Tensor) -> torch.Tensor:
        total = None
        for _ in range(samples):
            step = int(torch.randint(t_min - 1, t_max, (1,), generator=gen))
            eps = torch.randn(ip2p.image_latents(x_def01).shape,
                              generator=gen, dtype=torch.float32).to(device)
            term = _terms(x_def01, step, eps.to(x_def01.dtype))
            total = term if total is None else total + term
        return total / samples

    loss.make_fixed = make_fixed
    loss.face_box = box
    return loss
