"""擷取 UNet 的 cross-attention 分佈 — 供「破壞文字與影像的綁定」目標使用。

這個目標在攻擊什麼

文字引導編輯之所以能改動指定的內容，是因為 UNet 的 cross-attention 把
prompt 的每個 token 綁定到影像的特定區域：query 來自影像 latent 的空間
位置，key/value 來自文字嵌入，softmax 後的分佈 A[q, τ] 決定「第 q 個位置
要聽第 τ 個 token 的」。既有的目標函數把編輯結果推離原編輯，是在輸出
端量測；此處改為在該綁定本身上施力。

這正是 **Lo et al., "Distraction is All You Need: Memory-Efficient Image
Immunization against Diffusion-Based Image Editing", CVPR 2024** 的著力點，
本專案以該論文為對齊基準（Xu et al., arXiv 2509.10359, ACM MM 2025 為
同一路線的後續）。2026-08-03 依該論文補入三個函式，實作其式 (3)(4)(5)：

| 函式 | 對應 | 與本模組既有函式的差別 |
|---|---|---|
| `aggregate_token_attention` | 式 (3) | 保留空間維度，不壓成純量 |
| `attention_region_mask` | 式 (4) | 本模組原本沒有空間遮罩的概念 |
| `masked_attention_l1` | 式 (5) | 取 L1 而非注意力質量的補數 |

既有的 `attention_content_suppression` 與之最接近，但有兩處實質不同：
它對**全部** query 位置取平均（Lo 只取遮罩區內），且量的是「1 − 內容
token 的質量」（Lo 直接取聚合反應的 L1）。兩者都保留——前者是本專案的
變體，後者是文獻基準，網格中並列。

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
    精確為零（實測 grad_norm = 0.000e+00，見 `docs/RESULTS_E13-E23.md` §9
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


def aggregate_token_attention(
    maps: List[torch.Tensor], span: tuple, side: Optional[int] = None
) -> torch.Tensor:
    """Lo et al. (CVPR 2024) 式 (3) 的注意力聚合：跨層上採樣後逐像素相加。

        Att(x, c_a) = Σ_{l=1..L} upsample( A_l(x, c_a) )

    與本模組其他函式的差別是**輸出的形狀**。`attention_content_suppression`
    等函式把每層各自壓成一個純量再平均，得到的是「綁定強度」這個全域量；
    此處保留空間維度，得到一張 (B, side, side) 的圖，因為 Lo 的方法需要在
    這張圖上取遮罩（式 4）並只在遮罩內施力（式 5）。壓成純量就沒有遮罩可言。

    實際運算，逐層做：

    1. 取內容 token 的切片 `A_l[..., span[0]:span[1]]`，在 token 維度求和，
       得到 (B, Q_l)——即「該位置分給這個詞的注意力總質量」。
    2. 把 Q_l 還原成 (h_l, w_l)。UNet 各層的解析度不同（512² 影像對應
       64²／32²／16²／8²），此處要求 Q_l 為完全平方數，否則拋出。**不作
       任何長寬比推測**：猜錯會讓注意力圖被轉置而遮罩落在錯的區域，且不會
       有任何症狀顯示出來。非方形影像須由呼叫端另行提供形狀。
    3. 以雙三次插值上採樣到 `side`。論文明確指定 bicubic，理由是 UNet 的
       全卷積性質讓中間座標局部對應到初始尺寸特徵圖的鄰域。
    4. 全部層相加（不是平均）。論文式 (3) 是 Σ，故值域是 [0, L] 而非 [0, 1]。

    `side` 為 None 時取各層中最大的邊長，即最高解析度那一層的尺寸。
    """
    if span is None or span[1] <= span[0]:
        raise ValueError(
            f"aggregate_token_attention 需要非空的內容 token 區間，收到 {span}。"
            "Lo 的方法是對「要保護的那個詞」施力，沒有該詞就沒有攻擊目標"
        )
    if not maps:
        raise ValueError("注意力圖清單為空；CrossAttentionRecorder 未記錄到任何層")

    import math

    planes = []
    for a in maps:
        b, qn, _ = a.shape
        h = int(math.isqrt(qn))
        if h * h != qn:
            raise ValueError(
                f"注意力圖的 query 數 {qn} 不是完全平方數，無法還原成方形空間"
                "格點。本函式只支援方形影像；非方形須由呼叫端提供 (h, w)"
            )
        mass = a[..., span[0]:span[1]].sum(dim=-1)     # (B, Q_l)
        planes.append(mass.reshape(b, 1, h, h))

    tgt = side or max(p.shape[-1] for p in planes)
    up = [
        torch.nn.functional.interpolate(
            p, size=(tgt, tgt), mode="bicubic", align_corners=False
        )
        for p in planes
    ]
    return torch.stack(up).sum(dim=0).squeeze(1)       # (B, side, side)


def attention_region_mask(att_ref: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Lo et al. 式 (4) 的二值遮罩：M = I(Att(x, c_a) > τ)。

    遮罩由**原圖**的注意力圖決定，故對防禦參數為常數；此處回傳的張量已
    脫離計算圖。它標出「模型認為這個詞在影像的哪一塊」，攻擊只在該區內
    壓低注意力。

    論文沒有給出 τ 的數值。本實作把 `att_ref` 先除以自身的最大值再比較，
    即 τ 作用在正規化後的 [0, 1] 尺度上。**這是一個有記錄的偏離**，理由是
    式 (3) 的 Att 是 L 層相加，值域上界隨 UNet 的層數而變（SD v1.4 的
    attn2 有 16 層），絕對門檻會隨模型換代而失去意義，也無法跨影像比較。
    正規化後 τ = 0.5 的意思是「注意力達到該影像峰值一半以上的區域」。

    以峰值正規化帶來一個結構性保證：峰值那一格的正規化值恆為 1，而 τ < 1，
    故**遮罩永遠至少含有峰值，不可能為空**。下方的空遮罩檢查因此在目前的
    正規化下不可達；保留它是為了在有人改掉正規化方式時，讓「L1 恆為 0、
    梯度恆為零、最佳化靜默地什麼都不做」這個失效有一個具名的攔截點，而不是
    跑完才發現什麼都沒動。此性質由 `tests/test_lo_protocol.py` 釘住。
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau 須落在開區間 (0, 1)，收到 {tau}")
    ref = att_ref.detach()
    peak = ref.amax(dim=(-2, -1), keepdim=True)
    if float(peak.min()) <= 0:
        raise ValueError(
            "參照注意力圖的最大值為 0，表示該內容 token 沒有分到任何注意力質量；"
            "無法據以取遮罩"
        )
    mask = (ref / peak > tau).to(ref.dtype)
    if float(mask.sum()) == 0:
        raise ValueError(
            f"τ = {tau} 產生了空遮罩。後續的 masked L1 會恆為 0 而梯度恆為零，"
            "最佳化將靜默地不做任何事"
        )
    return mask


def masked_attention_l1(
    att: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Lo et al. 式 (5) 的注意力抑制損失：L = ‖Att(M ⊙ x_adv, c_a)‖₁。

    最小化它，即把遮罩區內該詞的注意力反應壓下去——論文稱為把模型的注意力
    「分散」（distract），使它找不到該編輯哪一塊。

    式 (5) 的寫法 `Att(M ⊙ x_adv, c_a)` 字面上是「先遮罩影像再算注意力」，
    但式 (4) 把 M 定義在注意力圖的空間格點上，且補充材料 Figure B 展示的是
    「注意力衰減」。本實作採後者：**先算注意力圖，再以 M 取該區域**。
    兩種讀法在梯度上不同（前者會讓遮罩外的像素完全收不到梯度），此處採用
    的版本讓整張影像都可以被更新，只是損失只計遮罩內的反應。此偏離已記錄。

    取和而非取平均，與「component-wise ℓ1-norm」一致。尺度隨遮罩大小而變，
    但 Algorithm 1 的更新是 `sign(grad)`，對損失的整體縮放不敏感，故此處的
    尺度選擇不影響最佳化軌跡。
    """
    if att.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            f"注意力圖 {tuple(att.shape)} 與遮罩 {tuple(mask.shape)} 的空間"
            "尺寸不符；遮罩必須在與 att 相同的格點上取"
        )
    return (att * mask).abs().sum()


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
