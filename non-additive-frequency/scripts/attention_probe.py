"""編輯指令的注意力落在哪裡。**只讀已存的防禦圖，不訓練、不重跑攻擊。**

為什麼要問
────────────────────────────────────────────────────────────────────
現行的損失（`image_guidance`）要求「UNet 對防禦圖的反應與沒有影像條件時相同」
——那是一個**全域**的目標，於是它獎勵全域的擾動。實測到的形狀正是如此：
效果由「碰到多少個 latent token」決定（同樣 23% 的面積，集中或散成六塊都是
14/25；權重攤平到整張畫面、碰到 100% 的 token 則是 23/25），而攤平整張畫面的
產物就是 PhotoGuard／Mist 那一族的樣子。

另一條路是**劫持**而不是破壞：讓編輯指令的 cross-attention 落到標記上，
而不是落到人身上。那條路若可行，小面積就夠——但前提是「補丁真的吸得到
注意力」。這支探針就是在動手寫新損失之前先驗證那個前提。

量什麼
────────────────────────────────────────────────────────────────────
對每一個 cross-attention 層，取指令 token 對影像位置的注意力圖 `A[token, hw]`，
把它攤回 (H, W) 之後在三塊區域上求和：

    attn_patch     補丁支撐（防禦圖與原圖不同的地方）
    attn_face      受保護的臉（ATR 的 Face + Hair）
    attn_rest      其餘

量的不是「這個位置收到多少注意力」——cross-attention 的 softmax 是**對 token
維**正規化的，每個位置的總和恆等於 1，那樣算出來的比例會退化成**區域面積**
（實測踩過：三個臂的數字正好等於各自的支撐面積 0.2719／0.1339／0.7131，而且
原圖與防禦圖逐位元相同、四個時間步也相同，看起來完全正常）。

改量 `1 − A[pos, BOS]`：第 0 個 token 是 BOS，在 CLIP 文字條件裡扮演注意力的
**洩流口**——一個位置若沒有被任何內容詞驅動，注意力就大量落在它身上。故這個
量是「這個位置有多少受文字內容驅動」，而且沿位置變化。三塊區域的比例相加為 1。
原圖與防禦圖各算一份，主讀數是**差**：`attn_patch_def − attn_patch_orig`
就是補丁吸走了多少。

`attn_patch_orig` 不可省——補丁所在的區域（衣服）本來就會吸走一些注意力，
不扣掉就分不出「補丁吸走了注意力」與「那塊區域本來就受注意」。這與
`identity.py` 的 `id_orig` 是同一條理由。

實作上的兩個坑
────────────────────────────────────────────────────────────────────
1. **要掛在 processor 上，不能改 forward 的回傳。** diffusers 的
   `Attention.forward` 不回傳注意力權重，`AttnProcessor2_0` 走的是
   `scaled_dot_product_attention`（融合核心，權重根本沒有被具現化）。故本檔
   自己算一次 `softmax(QK^T/√d)`——**只在探測時**，不影響任何訓練路徑。
2. **只取 cross-attention**（`attn2`，`encoder_hidden_states` 不是 None）。
   self-attention 的 key 是影像自己，「指令落在哪裡」在那裡沒有定義。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.ip2p import IP2PWrapper  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

# 取樣軌跡上的哪幾步。IP2P 由純噪聲起步，早期步決定佈局、晚期步決定細節；
# 兩端都取才知道注意力的分佈會不會隨 t 改變。本專案指定，故逐列寫進 CSV。
PROBE_STEPS = (999, 700, 400, 100)


def _load(p: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


class _Collector:
    """把每一個 cross-attention 層的 `softmax(QK^T/√d)` 收下來。

    只在 `with` 區塊內生效，離開就把 processor 還原——**不還原的話後續的
    編輯會繼續走這條慢路徑**，而輸出不變、只有速度變慢，看不出來。
    """

    def __init__(self, unet):
        self.unet = unet
        self.maps = []
        self._saved = None

    def __enter__(self):
        self.maps = []
        collector = self

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
                    # (batch·heads, hw, tokens) → 對 head 取平均後留著
                    collector.maps.append(probs.detach().float().cpu())
                out = torch.bmm(probs, v)
                out = attn.batch_to_head_dim(out)
                out = attn.to_out[0](out)
                return attn.to_out[1](out)

        self._saved = self.unet.attn_processors
        self.unet.set_attn_processor(_Proc())
        return self

    def __exit__(self, *exc):
        self.unet.set_attn_processor(self._saved)
        return False


def _region_share(prob: torch.Tensor, masks: dict, hw: int) -> dict:
    """一張 (hw, tokens) 的注意力圖 → 三塊區域各佔多少比例。

    **不可以用 `prob.sum(dim=-1)`**：softmax 是對 token 維正規化的，那樣每個
    位置恆等於 1，比例會退化成**區域面積**而與注意力無關。實測踩過一次，
    三個臂的數字正好等於各自的支撐面積且原圖與防禦圖完全相同。

    改用 `1 − A[pos, BOS]`：第 0 個 token 是 BOS，是 CLIP 文字條件裡的注意力
    洩流口；一個位置若沒有被內容詞驅動，注意力就大量落在它身上。故這個量是
    「這個位置有多少受文字內容驅動」，而且**沿位置變化**。
    """
    side = int(round(hw ** 0.5))
    if side * side != hw:
        return {}
    per_pos = 1.0 - prob[:, 0]                       # (hw,)
    grid = per_pos.reshape(1, 1, side, side)
    out = {}
    total = max(float(grid.sum()), 1e-12)
    for name, m in masks.items():
        small = F.adaptive_avg_pool2d(m, (side, side))
        out[name] = float((grid * small).sum()) / total
    return out


@torch.no_grad()
def probe_one(ip2p, x_orig, x_def, instruction, subject_mask, steps=PROBE_STEPS):
    """一張影像 → 逐步、逐區域的注意力比例。"""
    from src.metrics.naturalness import support_from_pair

    sup = support_from_pair(x_orig, x_def)
    face = subject_mask
    rest = (1.0 - sup.clamp(0, 1) - face.clamp(0, 1)).clamp(0, 1)
    masks = {"patch": sup, "face": face, "rest": rest}

    dev = ip2p.device
    emb = ip2p.pipe._encode_prompt(instruction, dev, 1, True, "")
    # `_encode_prompt` 回傳 [text, uncond, uncond] 或 [uncond, text]，形狀依版本
    # 而異。**取最後一份文字嵌入之前先檢查批次大小**，猜錯會量到空字串的注意力
    # 而那看起來完全正常。
    text_emb = emb[:1] if emb.shape[0] == 1 else emb[-1:]

    rows = []
    for img, tag in ((x_orig, "orig"), (x_def, "def")):
        z_img = ip2p.image_latents(img.to(dev))
        for step in steps:
            g = torch.Generator(device="cpu").manual_seed(step)
            eps = torch.randn(z_img.shape, generator=g).to(dev, z_img.dtype)
            sched = ip2p.pipe.scheduler
            abar = sched.alphas_cumprod.to(dev)[step]
            z_src = ip2p.encode_image(img.to(dev))
            z_t = z_src * abar.sqrt() + eps * (1 - abar).sqrt()
            tt = torch.tensor([step], device=dev, dtype=torch.long)
            with _Collector(ip2p.unet) as col:
                ip2p.unet(torch.cat([z_t, z_img], dim=1), tt,
                          encoder_hidden_states=text_emb.to(z_t.dtype))
            for li, p in enumerate(col.maps):
                heads = p.shape[0]
                share = _region_share(p.mean(0) if heads > 1 else p[0],
                                      masks, p.shape[1])
                if not share:
                    continue
                rows.append({"which": tag, "step": step, "layer": li,
                             "hw": p.shape[1], **share})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="一個已跑完的格目錄（含 __orig.png 與 __def.png）")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from src.defense.carrier_mask import face_subject_mask

    orig = sorted(args.run.glob("*__orig.png"))
    if not orig:
        raise SystemExit(f"{args.run} 底下沒有 *__orig.png")
    image = orig[0].name.split("__")[0]
    defs = sorted(args.run.glob(f"{image}__*__def.png"))
    if not defs:
        raise SystemExit(f"{args.run} 底下沒有 {image} 的防禦圖")

    ip2p = IP2PWrapper(dtype=torch.float32)
    x0, xd = _load(orig[0]), _load(defs[0])
    face = face_subject_mask(x0.to(ip2p.device)).cpu()
    rows = probe_one(ip2p, x0, xd, args.instruction, face)
    for r in rows:
        r["image"] = image
        r["run"] = args.run.name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)

    # 逐層太細，摘要按 step 聚合——主問題是「補丁吸不吸得到注意力」，
    # 那是一個跨層的量。
    import statistics as st
    print(f"\n{image}：注意力落在補丁上的比例（step 越大越早）")
    print("%6s %10s %10s %10s" % ("step", "orig", "def", "差"))
    for step in PROBE_STEPS:
        a = [r["patch"] for r in rows if r["which"] == "orig" and r["step"] == step]
        b = [r["patch"] for r in rows if r["which"] == "def" and r["step"] == step]
        if a and b:
            print("%6d %10.4f %10.4f %+10.4f"
                  % (step, st.median(a), st.median(b), st.median(b) - st.median(a)))
    print(f"\n寫出 {args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
