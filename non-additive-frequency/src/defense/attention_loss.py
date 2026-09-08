"""**改道**而不是破壞：把編輯指令的注意力吸到標記上，模型就去改標記而不改人。

與現行四個損失的分界
────────────────────────────────────────────────────────────────────
`image_guidance`／`latent_norm`／`encoder_target`／`edit_divergence` 全部是
**破壞**：讓模型讀不到這張圖、或讓它建出來的東西離原圖遠。那是全域的目標，
於是最佳化獎勵全域的擾動——實測到的形狀正是如此：效果由「碰到多少 latent
token」決定，而把權重攤平到整張畫面（碰到 100% 的 token）的成績最好，
產物就長成 PhotoGuard／Mist 那一族的樣子。

本項的機制不同：**犧牲區**。cross-attention 決定「指令要改畫面的哪裡」，
如果補丁把那份注意力吸走，模型就會去改補丁而不是改人。產物會是「花紋變成
紅色，人沒事」——那與「整張圖被毀掉」是不同的東西，也與 L∞ 那一族在機制上
可以區分。

    L_attn(x') = − E_{t, c_T} [ 注意力落在補丁上的比例 ]      ← 要最小化

**這是加項不是取代**：單獨用它時最佳化可以靠「把補丁變得極端顯眼」拿到高分，
而那不保證人沒被改。故它一律與主損失相加，權重由 `--attn-weight` 給。

實作：為什麼要自己算 softmax
────────────────────────────────────────────────────────────────────
diffusers 的 `AttnProcessor2_0` 走 `scaled_dot_product_attention`（融合核心），
注意力權重**根本沒有被具現化**，掛 hook 也拿不到。故本模組換上一個自己算
`softmax(QK^T/√d)` 的 processor。代價是慢與吃記憶體，故：

- 只在 **cross-attention**（`encoder_hidden_states` 不是 None）上收集；
  self-attention 的 key 是影像自己，「指令落在哪裡」在那裡沒有定義。
- 只收 **latent 邊長 ≤ `max_side`** 的層。SD 的 UNet 在 64×64 有 4096 個位置，
  一層的注意力矩陣就是 4096×77×heads，全部收下來會爆記憶體；而佈局資訊主要
  在 16×16 與 32×32 那幾層。
- processor 用完**一定要還原**。不還原的話後續的編輯會繼續走這條慢路徑，
  輸出不變、只有速度變慢，**看不出來**。

一個必須主動聲明的限制
────────────────────────────────────────────────────────────────────
「注意力落在哪裡」與「輸出被改在哪裡」不是同一件事，兩者的關聯是本專案還沒
量過的假設。`scripts/attention_probe.py` 就是在量它。這個損失在那個前提被
證實之前是探索性質，報表上要標明。
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

# 只收邊長不超過這個值的 cross-attention 層。64 那一層的矩陣是 4096×77，
# 收下來會讓一步 PGD 的峰值記憶體翻倍，而佈局資訊主要在較粗的層。
MAX_SIDE = 32


class CrossAttnShare:
    """收集 cross-attention 落在指定區域上的比例。**用完必須 `restore()`。**"""

    def __init__(self, unet, region: torch.Tensor):
        self.unet = unet
        self.region = region
        self.shares: List[torch.Tensor] = []
        self._saved = None

    def install(self):
        self.shares = []
        outer = self

        class _Proc:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                         attention_mask=None, **kw):
                is_cross = encoder_hidden_states is not None
                ctx = encoder_hidden_states if is_cross else hidden_states
                q = attn.to_q(hidden_states)
                k = attn.to_k(ctx)
                v = attn.to_v(ctx)
                q, k, v = (attn.head_to_batch_dim(t) for t in (q, k, v))
                probs = attn.get_attention_scores(q, k, attention_mask)
                if is_cross:
                    outer._accumulate(probs)
                out = attn.batch_to_head_dim(torch.bmm(probs, v))
                return attn.to_out[1](attn.to_out[0](out))

        self._saved = self.unet.attn_processors
        self.unet.set_attn_processor(_Proc())
        return self

    def _accumulate(self, probs: torch.Tensor) -> None:
        """(heads, hw, tokens) → 這一層落在區域上的**文字驅動程度**比例。

        為什麼不能用 `probs.sum(dim=-1)`
        ────────────────────────────────────────────────────────────────
        cross-attention 的 softmax 是**對 token 維**正規化的，故
        `probs.sum(dim=-1)` 每一個位置恆等於 1，區域比例會退化成**區域的面積**
        ——量到的與注意力無關。實測踩過：三個臂的「注意力比例」分別是 0.2719、
        0.1339、0.7131，正好等於它們的支撐面積，而且原圖與防禦圖逐位元相同、
        四個時間步也相同。數字看起來完全正常。

        改用 `1 − A[pos, 0]`：第 0 個 token 是 BOS，它在 CLIP 文字條件裡扮演
        **注意力的洩流口**——一個位置若沒有被任何內容詞驅動，它的注意力就大量
        落在 BOS 上。故 `1 − A[pos, BOS]` 是「這個位置有多少受文字內容驅動」，
        而那正是「指令會改哪裡」要問的東西。這個量**沿位置變化**，不會退化成面積。
        """
        hw = probs.shape[1]
        side = int(round(hw ** 0.5))
        if side * side != hw or side > MAX_SIDE:
            return
        per_pos = (1.0 - probs[:, :, 0]).mean(dim=0)     # (hw,)
        grid = per_pos.reshape(1, 1, side, side)
        w = F.adaptive_avg_pool2d(self.region.to(grid), (side, side))
        total = grid.sum().clamp_min(1e-12)
        self.shares.append((grid * w).sum() / total)

    def restore(self):
        if self._saved is not None:
            self.unet.set_attn_processor(self._saved)
            self._saved = None

    def mean_share(self) -> Optional[torch.Tensor]:
        if not self.shares:
            return None
        return torch.stack(self.shares).mean()


def make_attention_term(
    ip2p,
    *,
    region,
    text_embeds: torch.Tensor,
    weight: float,
    zt_mode: str,
    x_clean: torch.Tensor,
    t_min: int = 1,
    t_max: int = 1000,
    seed: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `term(x_def01) -> 純量`，**加到主損失上**（已含負號與權重）。

    `region` 是像素域的 (1,1,H,W) 遮罩，通常就是補丁的支撐。**它也可以是一個
    回傳該遮罩的 callable**，此時在每次 `term()` 呼叫時才解析。

    為什麼要允許 callable：補丁的 `support` 是在 `PatchParam.reset()` 裡才建的，
    而 `reset()` 發生在 `run_param_pgd` **內部**——組損失的時候它還是 `None`。
    直接傳值等於在還沒建好的時候取它。`scripts/ip2p_run.py` 的
    `_with_consistency` 記過同一個坑（「檢查屬性存不存在，不檢查它的值」），
    而這一支當初沒有照做，於是 `attn` 那個臂**從來沒有跑起來過**：
    守門每次都在 `support is None` 上把自己擋下。
    `text_embeds` 是 (K,L,D) 的一疊指令嵌入——注意力要對**文字**定義，
    故這裡不能用空字串：空字串的注意力落在哪裡與「指令會改哪裡」無關。
    """
    from src.defense.fixedpoint_loss import _scheduler_of

    if weight <= 0:
        raise ValueError(f"weight 必須為正，收到 {weight}")
    if text_embeds.dim() != 3 or text_embeds.shape[0] < 1:
        raise ValueError(
            f"text_embeds 必須是 (K,L,D) 且 K>=1，收到 {tuple(text_embeds.shape)}")

    unet, device = ip2p.unet, ip2p.device
    abar = _scheduler_of(ip2p).alphas_cumprod.to(device=device,
                                                 dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 9973)
    emb_all = text_embeds.to(device).detach()
    with torch.no_grad():
        z_src = ip2p.encode_image(x_clean).detach()

    def term(x_def01: torch.Tensor) -> torch.Tensor:
        z_img = ip2p.image_latents(x_def01)
        step = int(torch.randint(t_min - 1, t_max, (1,), generator=gen))
        eps = torch.randn(z_img.shape, generator=gen, dtype=torch.float32
                          ).to(device=device, dtype=z_img.dtype)
        if zt_mode == "noise":
            z_t = eps
        else:
            a = abar[step].to(z_img.dtype)
            z_t = z_src.to(z_img.dtype) * a.sqrt() + eps * (1.0 - a).sqrt()
        k = int(torch.randint(emb_all.shape[0], (1,), generator=gen))
        emb = emb_all[k:k + 1].to(z_t.dtype)
        tt = torch.tensor([step], device=device, dtype=torch.long)

        # 在這裡才解析：見上方 `region` 的說明。
        reg = region() if callable(region) else region
        if reg is None:
            raise RuntimeError(
                "region 解析出 None：補丁的 support 還沒建好。"
                "**不回傳零**——回零會讓這一項安靜地失效，"
                "而 CSV 上的 attn_weight 仍然寫著一個非零值。")
        col = CrossAttnShare(unet, reg).install()
        try:
            unet(torch.cat([z_t, z_img], dim=1), tt, encoder_hidden_states=emb)
            share = col.mean_share()
        finally:
            # **一定要還原**：不還原時後續的編輯會繼續走這條慢路徑而輸出不變。
            col.restore()
        if share is None:
            raise RuntimeError(
                "沒有收到任何 cross-attention 層——UNet 的 processor 介面可能"
                "換了。**不回傳零**：回零會讓這一項安靜地失效，而 CSV 上的"
                "attn_weight 仍然寫著一個非零值。")
        # 符號：PGD 最小化，故要把「落在補丁上的比例」推高就回傳負的。
        return -weight * share

    return term
