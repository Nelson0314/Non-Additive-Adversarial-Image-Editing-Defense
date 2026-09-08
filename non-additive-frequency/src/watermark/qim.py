"""區塊 DCT 的量化索引調變（QIM）浮水印 — `anti-purify/start.md` §7.4 的最小可跑版本。

為什麼有這個模組
────────────────────────────────────────────────────────────────────
本專案量到的天花板是：影像引導殘差有 64% 落在受保護主體自身的 latent token
上，因此**任何讓主體逐像素不變的防禦都被封頂**，擋不住編輯。§7.4 換掉的是
問題本身——不再要求「編不出來」，而是「編得出來但編過的事實藏不住」。

嵌進去的東西不需要在攻擊方那一端起作用，只需要在**我方的讀出端**還讀得回來。
脆弱那一半（擴散重生把位元正確率打到 50% 附近）文獻已有量測；難的是**另一半**
——被 JPEG 與模糊之後不能跟著崩潰，否則「讀不回來」就無法區分是被編輯過還是
只是被壓縮過，整個舉證就沒有意義。本模組只負責後面那一半是否成立，且是可量的。

§7.4 已裁定最小可跑版本**不需要訓練**：DCT 域的量化索引調變，逐圖封閉式計算。
從零訓練編碼解碼器做不到——WAM 用 MS-COCO 118000 張、8×V100 兩天，本專案的
資料集差三個數量級。

方法
────────────────────────────────────────────────────────────────────
Dither modulation QIM（Chen & Wornell, IEEE Trans. Inf. Theory 2001）：對載體
係數 c 與位元 b ∈ {0,1}，取兩個互相錯開 Δ/2 的量化格

    d_b = (b − 1/2)·Δ/2                    d_0 = −Δ/4、d_1 = +Δ/4
    c'  = Δ·round((c − d_b)/Δ) + d_b

萃取取最近的格：`argmin_b |c − Q_b(c)|`。無攻擊時 c' 恰在格點上，判定邊界距它
Δ/4，故 **Δ/4 就是這個方案能吃下的最大係數擾動**。這個量是本模組所有取捨的
單位：Δ 調大則抗壓縮，同時失真線性上升。

Δ 大到蓋過載體本身時，這個方案其實退化成正負號調變：|c| < Δ/2 的係數一律被
量到第 0 個格子，嵌入後的值就是 ±Δ/4，位元只編碼在正負號上。低通影像的中頻
係數大多如此。這**不是錯誤**，但它使「萃取端的 Δ 填錯」的症狀變成時好時壞
（實測嵌入用 Δ=32 而萃取用 24／31／48 全部仍是 1.000，用 12 則整批反相成
0.000），比「一律掉到 0.5」更難察覺。這是 Δ 必須逐列寫進 CSV、不可事後由
讀數回推的直接理由。

載體為什麼是 luma，以及殘差為什麼加在三個通道上
────────────────────────────────────────────────────────────────────
JPEG 的色度走 4:2:0 次取樣（見 `src/baselines/jpeg_codec.py`），把位元放進
Cb/Cr 等於先自砍一半頻寬再進量化器，所以載體取 luma。

但實作**不做 RGB↔YCbCr 的往返**：`ycbcr_to_rgb` 不是 `rgb_to_ycbcr` 的精確逆
（JFIF 正逆常數各自四捨五入到小數第六位，該檔 docstring 已載明只互逆到
1.2e-6），往返誤差會直接吃掉 Δ/4 的餘裕，而且症狀是「位元偶爾錯一個」這種
看起來像雜訊的東西。改成：算出 luma 平面上要的改動量 r，再把 **r 同時加到
R、G、B 三個通道**。BT.601 的 luma 列係數和恰為 1（0.299+0.587+0.114），
兩條色度列的係數和恰為 0，所以這個等量改動使

    ΔY = r（精確）        ΔCb = ΔCr = 0（精確）

也就是浮水印是純亮度紋理、不動色度，且讀出端只要用同一組權重取 luma 就能
逐位元拿回 r，中間沒有任何一次有損轉換。

頻帶：為什麼是中頻
────────────────────────────────────────────────────────────────────
- **DC (0,0) 不能用。** 它是區塊均值，任何全域色調、曝光、白平衡調整都會整片
  平移它；本專案的淨化清單裡 `auto_levels`、`gray_world`、`clahe` 全都做這件事。
  它同時也是視覺上最敏感的一格（區塊均值改動會看成塊狀瑕疵）。
- **最高頻不能用。** JPEG 的 luma 量化步長在 (7,7) 於 q=75 是 50、q=30 是 164；
  Δ 要大過那個量級才活得下來，而那時失真已經不能看。模糊也是先殺這一段。
- **中頻是唯一還有餘裕的一段。** 同一張表在 (1,2)／(2,1)／(2,2) 於 q=75 是
  7／7／8、q=30 是 23／22／27，Δ=48（餘裕 12）就吃得下 q=30 的 ±13.5。

**「多放幾個係數等於免費的冗餘」是錯的**，這一點實測過：把頻帶由四個係數擴成
`band_coeffs(3, 5)` 的十五個，冗餘倍數多 60 倍，Δ=32 下 JPEG q=30 的位元正確率
反而由 0.668 掉到 0.543。原因是該帶含 (0,5) 與 (5,0)，q=30 的步長是 66 與 40，
這些槽的輸出接近亂數，而多數決對「一致地壞掉」的多數票沒有辦法。挑係數要看
量化步長，不是看對角線編號。

已知會失效的四件事（全部量得到，不要當成 bug）
────────────────────────────────────────────────────────────────────
0. **QIM 對振幅縮放沒有不變性。** 模糊不是加性擾動，而是把係數乘上一個小於 1
   的頻率響應 a(u,v)，造成的偏移是 (1−a)·|c|，**正比於載體本身的大小**。
   於是同一組 (Δ, 頻帶) 在不同影像上的抗模糊表現不同：中頻能量低的自然影像
   幾乎不受影響，中頻能量高的影像（極端例：均勻雜訊圖）在 sigma=2.0 就掉到
   0.56。這是 QIM 這個方案的性質，不是實作缺陷；文獻的對策是有理擴張 QIM
   （RDM），本模組沒有做，只把它量出來。
1. **裁切。** 區塊 DCT 綁在固定的 8×8 格點上，裁掉的邊長只要不是 8 的倍數，
   讀出端就把整張圖切在不同的位置上，讀到的係數與嵌入時的無關。實測
   `crop_resize` 0.1 一律在 0.36–0.61 之間游走，也就是**亂猜**。本模組
   不處理這件事：§7.4 指出解法是嵌入模板加上解碼端估幾何（我方擁有解碼器
   才做得到），那是另一個工作，不在此。
2. **飽和區塊。** 殘差要加回像素後夾回 `[0,1]`，純黑或純白的區塊夾不動，
   位元就寫不進去。實測一張半白半黑的合成圖在 Δ=4 只有 0.688。`embed` 的
   `verify` 會把這件事變成明確的例外，而不是後面某個讀數莫名其妙。
3. **參數填錯。** 萃取端的 `delta`、`coeffs`、`repeat`、`n_bits` 只要有一個與
   嵌入端不同，讀數就是 0.5 附近，而 0.5 附近**也正是「被擴散編輯打掉了」的
   長相**。兩者無法由讀數本身分辨，所以這四個參數一律沒有預設值、必須明給，
   而且驅動程式必須把它們逐列寫進 CSV 欄位。

值域與形狀
────────────────────────────────────────────────────────────────────
對外一律 (1,3,H,W)、`[0,1]`、H 與 W 為 8 的倍數（本專案是 512×512）。
`embed` 的輸出**已經落在 uint8 網格上**（`round(x·255)/255`），因為本專案存
PNG，先在這裡量化才能讓「存檔再讀回」是恆等，而不是另一道未量測的攻擊。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from src.baselines.jpeg_codec import block_dct, block_idct, dct_matrix

# BT.601 full-range 的 luma 權重，與 `src/baselines/jpeg_codec.py` 的
# `_RGB2YCC` 第一列逐字相同。**不要在這裡另寫一組**：兩處不一致時嵌入端與
# JPEG 端就不是在同一個平面上作用，而症狀只會是抗壓縮讀數偏低。
LUMA_WEIGHTS: Tuple[float, float, float] = (0.299, 0.587, 0.114)

BLOCK = 8


def band_coeffs(lo: int, hi: int) -> Tuple[Tuple[int, int], ...]:
    """回傳 `lo <= u + v <= hi` 的所有 (u, v)，依 (u+v, u) 排序。

    這是**產生候選集的工具，不是推薦值**。同一條反對角上各格的 JPEG 量化步長
    差很多（q=30 時 (1,4) 是 43、(4,1) 是 37、(0,5) 是 66），用對角線圈出來的帶
    會混進步長大一個量級的格，見模組 docstring 的實測。
    """
    if not 0 <= lo <= hi <= 2 * (BLOCK - 1):
        raise ValueError(
            f"需要 0 <= lo <= hi <= {2 * (BLOCK - 1)}，收到 lo={lo}、hi={hi}")
    return tuple(sorted(
        ((u, v) for u in range(BLOCK) for v in range(BLOCK) if lo <= u + v <= hi),
        key=lambda uv: (uv[0] + uv[1], uv[0])))


def slot_count(h: int, w: int, n_coeffs: int) -> int:
    """(h, w) 的影像在每區塊用 `n_coeffs` 個係數時的載體槽總數。"""
    if h % BLOCK or w % BLOCK:
        raise ValueError(f"高寬必須是 {BLOCK} 的倍數，收到 {h}×{w}")
    if n_coeffs < 1:
        raise ValueError(f"每區塊至少要一個係數，收到 {n_coeffs}")
    return (h // BLOCK) * (w // BLOCK) * n_coeffs


def _check_image(x01: torch.Tensor) -> None:
    if x01.dim() != 4 or x01.shape[0] != 1 or x01.shape[1] != 3:
        raise ValueError(f"需要 (1,3,H,W) 的影像，收到 {tuple(x01.shape)}")
    h, w = x01.shape[-2:]
    if h % BLOCK or w % BLOCK:
        raise ValueError(f"高寬必須是 {BLOCK} 的倍數，收到 {h}×{w}")


def _check_coeffs(coeffs: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    out = tuple((int(u), int(v)) for u, v in coeffs)
    if not out:
        raise ValueError("coeffs 不可為空；它是必填參數，沒有預設頻帶")
    for u, v in out:
        if not (0 <= u < BLOCK and 0 <= v < BLOCK):
            raise ValueError(f"係數索引必須落在 [0,{BLOCK})，收到 ({u},{v})")
    if len(set(out)) != len(out):
        raise ValueError(f"coeffs 有重複的格：{out}")
    return out


def _check_payload(delta: float, repeat: int) -> None:
    if not delta > 0:
        raise ValueError(f"delta 必須為正，收到 {delta}")
    if int(repeat) != repeat or repeat < 1:
        raise ValueError(f"repeat 必須是正整數，收到 {repeat}")


def _luma255(x01: torch.Tensor) -> torch.Tensor:
    """(1,3,H,W) 的 `[0,1]` RGB 轉成 (1,1,H,W) 的 `[0,255]` luma。"""
    w = torch.tensor(LUMA_WEIGHTS, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 * 255.0 * w).sum(dim=1, keepdim=True)


def _slot_plan(n_slots: int, n_bits: int, repeat: int
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """回傳 (被使用的槽索引, 該槽承載的位元索引)，長度皆為 `n_bits * repeat`。

    槽的編號是 `((by * wb) + bx) * K + k`，也就是區塊光柵序、區塊內再依
    `coeffs` 的順序。用到的槽以 `(j * n_slots) // M` **均勻攤在整張圖上**，
    而不是塞滿前 M 個槽：後者會把浮水印全部堆在影像上緣，失真在空間上不均，
    而且下緣被裁掉或被局部編輯時整段位元一起消失。`M == n_slots` 時這個式子
    退化成恆等，也就是「用滿」是這個規則的特例而不是另一條路徑。

    第 j 個被使用的槽承載第 `j % n_bits` 個位元，故同一個位元的 `repeat` 份
    副本彼此相距 `n_bits` 個槽，落在不同的區塊上。
    """
    m = n_bits * repeat
    j = torch.arange(m, dtype=torch.long)
    return (j * n_slots) // m, j % n_bits


def _quantise(c: torch.Tensor, bit: torch.Tensor, delta: float) -> torch.Tensor:
    """DM-QIM 的格點量化。`bit` 與 `c` 同形狀，值為 0 或 1。"""
    d = (bit.to(c.dtype) - 0.5) * (delta / 2.0)
    return torch.round((c - d) / delta) * delta + d


def _gather_slots(coef: torch.Tensor, coeffs: Tuple[Tuple[int, int], ...]
                  ) -> torch.Tensor:
    """(1,hb,wb,8,8) 的係數轉成依槽編號攤平的 (n_slots,) 向量。"""
    picked = torch.stack([coef[0, :, :, u, v] for u, v in coeffs], dim=-1)
    return picked.reshape(-1)


def embed(x01: torch.Tensor, bits: torch.Tensor, *,
          delta: float,
          coeffs: Sequence[Tuple[int, int]],
          repeat: int,
          verify: bool = True) -> torch.Tensor:
    """把 `bits` 嵌進 `x01`，回傳同形狀、已落在 uint8 網格上的影像。

    `delta`、`coeffs`、`repeat` **都沒有預設值**：§7.4 沒有指定，而填錯的症狀
    與「浮水印被編輯打掉」完全一樣（見模組 docstring 第 3 點）。

    `verify=True` 時會對自己的輸出跑一次 `extract` 並在對不上時拋出。這不是
    保險絲而是契約檢查：對不上代表這組 (Δ, 頻帶, 影像) 根本承載不了這個酬載
    （最常見的成因是飽和區塊夾不動，見模組 docstring 第 2 點），此時回傳一張
    讀不出來的圖只會讓錯誤延後到某個位元正確率讀數上，且無法與攻擊區分。
    驅動程式若要把這件事量成欄位而不是中止整批，才應該關掉它。
    """
    _check_image(x01)
    coeffs = _check_coeffs(coeffs)
    _check_payload(delta, repeat)

    bits = torch.as_tensor(bits).reshape(-1).long()
    if bits.numel() == 0:
        raise ValueError("bits 不可為空")
    if not torch.all((bits == 0) | (bits == 1)):
        raise ValueError("bits 只能是 0 或 1")

    h, w = x01.shape[-2:]
    n_slots = slot_count(h, w, len(coeffs))
    n_bits = int(bits.numel())
    need = n_bits * repeat
    if need > n_slots:
        raise ValueError(
            f"{n_bits} 個位元乘 repeat={repeat} 需要 {need} 個槽，"
            f"但 {h}×{w} 的影像在 {len(coeffs)} 個係數下只有 {n_slots} 個。"
            f"請降低位元數或 repeat，或增加 coeffs")

    dev, dt = x01.device, x01.dtype
    d = dct_matrix(dev, dt)
    y = _luma255(x01)
    coef = block_dct(y - 128.0, d)
    hb, wb = coef.shape[1], coef.shape[2]

    slots, bit_idx = _slot_plan(n_slots, n_bits, repeat)
    slots, bit_idx = slots.to(dev), bit_idx.to(dev)
    flat = _gather_slots(coef, coeffs).clone()
    flat[slots] = _quantise(flat[slots], bits.to(dev)[bit_idx], delta)

    coef = coef.clone()
    picked = flat.reshape(hb, wb, len(coeffs))
    for k, (u, v) in enumerate(coeffs):
        coef[0, :, :, u, v] = picked[:, :, k]

    # 只把 luma 的改動量取出來加回三個通道，見模組 docstring：這使 ΔY 精確
    # 等於 r 而 ΔCb 與 ΔCr 精確為 0，中間沒有 RGB↔YCbCr 的有損往返。
    residual = (block_idct(coef, d) + 128.0) - y
    out = (x01 * 255.0 + residual).clamp(0.0, 255.0)
    out = torch.round(out) / 255.0

    if verify:
        got = extract(out, n_bits, delta=delta, coeffs=coeffs, repeat=repeat)
        wrong = int((got != bits.to(got.device)).sum())
        if wrong:
            raise RuntimeError(
                f"嵌入後自我萃取有 {wrong}/{n_bits} 個位元對不上（delta={delta}、"
                f"coeffs={coeffs}、repeat={repeat}）。Δ/4 = {delta / 4:g} 是本方案"
                f"能吃下的最大係數擾動，而 uint8 量化與 [0,1] 夾取已經吃掉超過"
                f"這個量；最常見的成因是影像有大片飽和（純黑或純白）區塊。"
                f"提高 delta 或改用不飽和的影像，不要略過這個檢查")
    return out


def extract(x01: torch.Tensor, n_bits: int, *,
            delta: float,
            coeffs: Sequence[Tuple[int, int]],
            repeat: int) -> torch.Tensor:
    """由 `x01` 讀回 `n_bits` 個位元，回傳 (n_bits,) 的 0/1 `long` 張量。

    每個位元的 `repeat` 份副本各自硬判定後取**多數決**。`repeat` 為偶數而票數
    恰好平手時，改用軟證據（各副本到兩個格點的距離差之和）決定；平手時硬票
    本身沒有資訊，退回距離是唯一還有內容的量。這一步不影響 `repeat` 為奇數的
    情形，也不影響無攻擊時的結果（無攻擊時所有副本一致）。
    """
    _check_image(x01)
    coeffs = _check_coeffs(coeffs)
    _check_payload(delta, repeat)
    if int(n_bits) != n_bits or n_bits < 1:
        raise ValueError(f"n_bits 必須是正整數，收到 {n_bits}")
    n_bits = int(n_bits)

    h, w = x01.shape[-2:]
    n_slots = slot_count(h, w, len(coeffs))
    need = n_bits * repeat
    if need > n_slots:
        raise ValueError(
            f"{n_bits} 個位元乘 repeat={repeat} 需要 {need} 個槽，"
            f"但 {h}×{w} 的影像在 {len(coeffs)} 個係數下只有 {n_slots} 個")

    dev, dt = x01.device, x01.dtype
    d = dct_matrix(dev, dt)
    coef = block_dct(_luma255(x01) - 128.0, d)
    flat = _gather_slots(coef, coeffs)

    slots, bit_idx = _slot_plan(n_slots, n_bits, repeat)
    c = flat[slots.to(dev)]
    zeros = torch.zeros_like(c)
    c0 = _quantise(c, zeros, delta)
    c1 = _quantise(c, zeros + 1, delta)
    # 到 b=0 的距離減到 b=1 的距離：大於 0 表示比較靠近 b=1。
    margin = (c - c0).abs() - (c - c1).abs()
    hard = (margin > 0).to(dt)

    bit_idx = bit_idx.to(dev)
    votes = torch.zeros(n_bits, device=dev, dtype=dt).index_add_(0, bit_idx, hard)
    soft = torch.zeros(n_bits, device=dev, dtype=dt).index_add_(0, bit_idx, margin)
    half = repeat / 2.0
    return torch.where(votes == half, soft > 0, votes > half).long()


def bit_accuracy(a: torch.Tensor, b: torch.Tensor) -> float:
    """兩個等長位元向量相同的比例。隨機猜測的期望值是 0.5，不是 0。"""
    a = torch.as_tensor(a).reshape(-1).long()
    b = torch.as_tensor(b).reshape(-1).long()
    if a.numel() != b.numel():
        raise ValueError(f"兩個位元向量長度不同：{a.numel()} 與 {b.numel()}")
    if a.numel() == 0:
        raise ValueError("位元向量不可為空")
    return float((a == b.to(a.device)).to(torch.float64).mean())


def random_bits(n_bits: int, seed: int) -> torch.Tensor:
    """固定種子的酬載。**種子必須進 CSV 欄位**：位元序列不入版控，
    重跑時只有 (n_bits, seed) 這一對能把同一個酬載還原出來。"""
    if int(n_bits) != n_bits or n_bits < 1:
        raise ValueError(f"n_bits 必須是正整數，收到 {n_bits}")
    g = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, 2, (int(n_bits),), generator=g, dtype=torch.long)
