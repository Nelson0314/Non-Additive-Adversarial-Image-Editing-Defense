"""擷取 UNet 的 cross-attention 分佈 — 供「破壞文字與影像的綁定」目標使用。

這個目標在攻擊什麼

文字引導編輯之所以能改動指定的內容，是因為 UNet 的 cross-attention 把
prompt 的每個 token 綁定到影像的特定區域：query 來自影像 latent 的空間
位置，key/value 來自文字嵌入，softmax 後的分佈 A[q, τ] 決定「第 q 個位置
要聽第 τ 個 token 的」。既有的目標函數把編輯結果推離原編輯，是在輸出
端量測；此處改為在該綁定本身上施力（Xu et al., "Immunizing Images from
Text to Image Editing via Adversarial Cross-Attention", arXiv 2509.10359，
ACM MM 2025 採取相同的著力點）。

擷取方式：forward pre-hook，不換 attention processor

換掉 processor 會連帶改變 UNet 自己的注意力計算路徑（SDPA 融合核心改為
展開的 QKᵀ），數值與記憶體行為都會變，於是「有沒有開這個目標」就不再是
單一變因。此處改用 forward pre-hook 只讀取該層的輸入，再以該層自己的
`to_q` / `to_k` / `get_attention_scores` 另算一次注意力分佈。UNet 的前向
完全不受影響，代價是每層多一次 q、k 投影與一次 QKᵀ。

SD v1.4 的 attn2 其 `group_norm` 與 `norm_cross` 均為 None（實測），但此處
仍照 diffusers `AttnProcessor.__call__` 的順序處理該兩者，否則換模型時會
算出不同的東西。
"""

from typing import List, Optional

import torch


class CrossAttentionRecorder:
    """context manager：記錄一次 UNet 前向中所有 cross-attention 層的分佈。

    用法：

        rec = CrossAttentionRecorder(sd.unet)
        with rec:
            eps = sd._eps(z, t, emb)
        maps = rec.maps          # list，每層一個 (B, Q_l, T)

    每層的 query 數 Q_l 不同（對應 64²、32²、16²、8² 等解析度），故不合併
    成單一張量：合併需要重採樣，那會把「哪一個解析度的綁定被破壞」這個
    資訊抹掉。呼叫端應逐層算完再平均。

    分佈已對 head 取平均。保留逐 head 會讓張量大 8 倍，而目標是「這個位置
    聽哪個 token」，head 之間的分工不是本研究要區分的對象。
    """

    def __init__(self, unet, average_heads: bool = True):
        self.unet = unet
        self.average_heads = average_heads
        self.maps: List[torch.Tensor] = []
        self._handles = []
        self._layers = [
            m for n, m in unet.named_modules() if n.endswith("attn2")
        ]
        if not self._layers:
            raise RuntimeError(
                "在 UNet 中找不到任何 attn2 層；此模型可能不是 SD 架構，"
                "cross-attention 目標無法套用"
            )

    def _make_hook(self):
        def hook(module, args, kwargs):
            # attn2 的呼叫可能用位置引數也可能用關鍵字，兩種都要接
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            enc = kwargs.get("encoder_hidden_states", None)
            if enc is None and len(args) > 1:
                enc = args[1]
            if enc is None:
                # encoder_hidden_states 為 None 表示這層退化成 self-attention，
                # 沒有文字綁定可言，跳過而非誤記成一筆
                return None

            h = hidden
            if module.group_norm is not None:
                h = module.group_norm(h.transpose(1, 2)).transpose(1, 2)
            e = enc
            if module.norm_cross is not None:
                e = module.norm_encoder_hidden_states(e)

            q = module.head_to_batch_dim(module.to_q(h))
            k = module.head_to_batch_dim(module.to_k(e))
            probs = module.get_attention_scores(q, k, None)   # (B·H, Q, T)

            if self.average_heads:
                bh, qn, tn = probs.shape
                probs = probs.reshape(bh // module.heads, module.heads, qn, tn)
                probs = probs.mean(dim=1)
            self.maps.append(probs)
            return None

        return hook

    def __enter__(self):
        self.maps = []
        for layer in self._layers:
            self._handles.append(
                layer.register_forward_pre_hook(self._make_hook(), with_kwargs=True)
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def clear(self):
        self.maps = []


def token_span(tokenizer, prompt: str) -> tuple:
    """回傳 prompt 的內容 token 在 77 格中的區間 (start, end)，不含 BOS/EOS。

    CLIP 的 tokenizer 產生 [BOS] tok₁ … tok_n [EOS] 之後補 padding。padding
    與 EOS 位置在 SD 中仍會分到可觀的注意力質量（是已知現象），但它們不
    承載 prompt 的語意；把它們算進「內容 token」會讓指標被一個與 prompt
    無關的常數項稀釋。故此處明確界定區間，讓損失只作用在真正的內容上。

    空 prompt 時回傳 (1, 1)，即空區間；呼叫端須據此改用不依賴內容 token
    的模式，而不是對空區間取平均得到 nan。
    """
    ids = tokenizer(prompt, padding=False, truncation=True,
                    max_length=tokenizer.model_max_length).input_ids
    # ids = [BOS] ... [EOS]，內容為中間那段
    return (1, max(1, len(ids) - 1))


def attention_divergence(
    maps_def: List[torch.Tensor],
    maps_ref: List[torch.Tensor],
    span: Optional[tuple] = None,
) -> torch.Tensor:
    """兩組注意力分佈的差異，逐層算完再平均。

    以對稱 KL 而非 L2：注意力是機率分佈，L2 對「質量從哪個 token 搬到哪個
    token」不敏感——把 0.9 平均攤到 76 個 token 與把 0.9 整塊搬到另一個
    token，兩者 L2 相近，但前者是綁定被破壞、後者是綁定被改指向，是完全
    不同的事。KL 分得出來。

    `span` 為內容 token 的區間；給定時只比較該區間並重新正規化，理由見
    `token_span`。逐層平均而非逐層加總：層數隨模型而異，取平均後這個量
    在不同模型之間才可比。
    """
    if len(maps_def) != len(maps_ref):
        raise ValueError(
            f"兩次前向記錄到的層數不同（{len(maps_def)} vs {len(maps_ref)}），"
            "表示某一次的 hook 未完整覆蓋，數值不可比"
        )
    eps = 1e-8
    terms = []
    for a, b in zip(maps_def, maps_ref):
        if span is not None:
            a = a[..., span[0]:span[1]]
            b = b[..., span[0]:span[1]]
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(eps)
            b = b / b.sum(dim=-1, keepdim=True).clamp_min(eps)
        a = a.clamp_min(eps)
        b = b.clamp_min(eps)
        kl_ab = (a * (a.log() - b.log())).sum(dim=-1)
        kl_ba = (b * (b.log() - a.log())).sum(dim=-1)
        terms.append((kl_ab + kl_ba).mean() * 0.5)
    return torch.stack(terms).mean()


def attention_content_suppression(
    maps: List[torch.Tensor], span: tuple
) -> torch.Tensor:
    """1 − 內容 token 分到的注意力質量，逐層算完再平均。越大代表綁定越弱。

    這個模式存在的理由是 `divergence` 從 φ=0 起不了步。KL(A_φ ‖ A_ref) 在
    φ=0 時兩個分佈逐元素相同，KL = 0，而 0 是 KL 的最小值，故梯度在該點
    精確為零（實測 grad_norm = 0.000e+00，見 `docs/RESULTS_E20_fidelity.md` §9
    與 `tests/test_pipeline.py::test_divergence模式在phi等於零時無梯度`）。
    任何「與未防禦參照的散度」形式的目標都有這個性質，換一種散度並不能解決。

    解法是換一個最佳點不在 φ=0 的量。此處取內容 token 分到的注意力質量：
    它在 φ=0 時是一個由影像與 prompt 決定的中間值，不是極值，故梯度一般非零。
    這也正是 cross-attention 免疫那條線的著力點（Xu et al., arXiv:2509.10359；
    *Dual Attention Guided Defense*, arXiv:2512.14333）——降低 prompt 內容與
    影像區域之間的注意力質量，而不是把分佈推離某個參照。

    質量取 softmax 後未重新正規化的和，故值域為 [0, 1]，本量亦然。有界
    使它不需要 hinge：無界的最大化才會發散（見 `objective.py` 的 L_def）。

    `span` 為必要參數而非可選。沒有內容 token 時本目標沒有定義，靜默落回
    「全部 77 格」會得到恆等於 0 的常數（全部 token 的質量和恆為 1），
    最佳化會不會產生任何更新——那正是本模式要修掉的失效。
    """
    if span is None:
        raise ValueError(
            "attention_content_suppression 需要內容 token 的區間 span。"
            "prompt 沒有內容 token 時本目標無定義，不可落回全域："
            "全部 token 的注意力質量和恆為 1，該目標會變成常數 0"
        )
    if span[1] <= span[0]:
        raise ValueError(
            f"內容 token 區間為空：{span}。空 prompt 不適用本模式"
        )
    terms = []
    for a in maps:
        mass = a[..., span[0]:span[1]].sum(dim=-1)
        terms.append((1.0 - mass).mean())
    return torch.stack(terms).mean()


def attention_entropy(
    maps: List[torch.Tensor], span: Optional[tuple] = None
) -> torch.Tensor:
    """注意力分佈在 token 維度上的熵，逐層算完再平均。

    分佈趨近均勻時熵最大，代表沒有任何 token 主導任何位置，即綁定被瓦解。
    此模式不需要參考分佈，故與 prompt 的具體內容關聯較弱——用來檢驗
    「破壞綁定」與「把綁定改指向別處」哪一種比較能跨 prompt 泛化。
    """
    eps = 1e-8
    terms = []
    for a in maps:
        if span is not None:
            a = a[..., span[0]:span[1]]
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(eps)
        a = a.clamp_min(eps)
        terms.append((-(a * a.log()).sum(dim=-1)).mean())
    return torch.stack(terms).mean()
