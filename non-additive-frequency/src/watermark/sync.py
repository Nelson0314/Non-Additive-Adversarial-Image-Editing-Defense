"""解碼端的幾何對齊：把裁切那一欄從「亂猜」救回來。

為什麼需要這一項
────────────────────────────────────────────────────────────────────
`runs/watermark_qim/README.md` 量到本方案唯一真正的破口：**裁切後的位元
正確率是 0.44–0.52，比擴散編輯的 0.73–0.79 還低**。於是現行的判別式會把
「有人裁掉邊framing」誤判成「有人用擴散模型改過」——一個會誤報的偵測器
沒有舉證價值。

根因不是浮水印被破壞，而是**沒有人替它對回去**：區塊 DCT 綁在固定的 8×8
格點上，裁切把格點整個推開，讀到的係數與嵌入的係數無關。擾動本身原封不動
地通過了。

`anti-purify/start.md` §7.4 的關鍵論點正是這一點——穩健浮水印工具箱裡
「嵌入模板 ＋ 解碼端估幾何」這一項，建立在**我方控制解碼器**上，而防護擾動
沒有那個東西。既然這條路承認讀出端存在，那一項就回來可用。

作法：以萃取信心度搜尋幾何
────────────────────────────────────────────────────────────────────
不嵌同步模板，改用**萃取自己的信心度**當對齊準則，因為它不花額外的失真預算。

對齊正確時，載體係數會坐在 QIM 的格點中心附近，`|margin|` 接近 `delta/2`；
對齊錯誤時係數大致均勻分布，`|margin|` 趨近 `delta/4`。所以

    信心度 = mean(|margin|) ÷ (delta/2)

在正確的幾何上會出現明顯的峰。搜尋空間很小（尺度 × 區塊格點位移），
每個候選只要一次區塊 DCT，成本可以忽略。

三件必須說清楚的事
────────────────────────────────────────────────────────────────────
1. **這一項只對「幾何可逆」的攻擊有用。** 裁切之後重新取樣是可逆的（把倍率
   除回去），但裁掉的內容不會回來——落在畫面外的槽讀到的是別的東西，故
   `_extract_core` 帶一個逐槽的有效遮罩，只用還在畫面內的副本投票。
2. **信心度是在被檢查的那張圖上算的，不需要原圖。** 若需要原圖，這一項在
   舉證情境下就沒有意義了。
3. **搜到的幾何要逐列寫進 CSV**（尺度、位移、信心度）。對錯了會安靜地讀出
   一組垃圾位元，與「被編輯打掉」的長相相同。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.baselines.jpeg_codec import block_dct, dct_matrix

from src.watermark.qim import (
    BLOCK, _check_coeffs, _check_payload, _gather_slots, _luma255, _quantise,
    _slot_plan, slot_count,
)


def _extract_core(x01: torch.Tensor, n_bits: int, *, delta: float,
                  coeffs: Tuple[Tuple[int, int], ...], repeat: int,
                  valid: Optional[torch.Tensor] = None
                  ) -> Tuple[torch.Tensor, float]:
    """萃取並回報信心度。`valid` 是逐槽的 0/1 遮罩（None = 全部有效）。

    與 `qim.extract` 的判定規則相同（硬票多數決、平手看軟證據），差別只在
    多了遮罩與信心度。**不重寫量化與槽規劃**，那兩者由 `qim` 提供，兩邊
    各寫一份的話對齊搜尋會與正式萃取用不同的規則。
    """
    coeffs = _check_coeffs(coeffs)
    _check_payload(delta, repeat)
    h, w = x01.shape[-2:]
    n_slots = slot_count(h, w, len(coeffs))
    dev, dt = x01.device, x01.dtype

    d = dct_matrix(dev, dt)
    coef = block_dct(_luma255(x01) - 128.0, d)
    flat = _gather_slots(coef, coeffs)

    slots, bit_idx = _slot_plan(n_slots, n_bits, repeat)
    slots, bit_idx = slots.to(dev), bit_idx.to(dev)
    c = flat[slots]
    zeros = torch.zeros_like(c)
    margin = ((c - _quantise(c, zeros, delta)).abs()
              - (c - _quantise(c, zeros + 1, delta)).abs())

    if valid is None:
        wgt = torch.ones_like(margin)
    else:
        wgt = valid.to(dev, dt).reshape(-1)[slots]

    hard = (margin > 0).to(dt) * wgt
    votes = torch.zeros(n_bits, device=dev, dtype=dt).index_add_(0, bit_idx, hard)
    seen = torch.zeros(n_bits, device=dev, dtype=dt).index_add_(0, bit_idx, wgt)
    soft = torch.zeros(n_bits, device=dev, dtype=dt).index_add_(
        0, bit_idx, margin * wgt)
    bits = torch.where(votes * 2 == seen, soft > 0, votes * 2 > seen).long()

    denom = float(wgt.sum())
    conf = (float((margin.abs() * wgt).sum()) / denom / (delta / 2.0)
            if denom > 0 else 0.0)
    return bits, conf


def _pad_to_block(n: int) -> int:
    """向下取到 `BLOCK` 的倍數。畫布必須整除 8，否則區塊 DCT 定義不出來。"""
    return (n // BLOCK) * BLOCK


def undo_magnification(y01: torch.Tensor, scale: float, oy: int, ox: int,
                       canvas: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """把放大過的圖縮回去、貼回原尺寸的畫布，回傳 (畫布, 逐像素有效遮罩)。

    `scale > 1` 代表觀測到的圖被放大了（裁切後重新取樣就是這樣），故縮回去
    的邊長是 `round(canvas / scale)`。貼在 `(oy, ox)`，其餘補畫布均值——
    補零會在邊界造成強烈的假邊，那個假邊自己就會在 DCT 上產生高信心度。
    """
    side = int(round(canvas / scale))
    side = max(BLOCK, min(canvas, side))
    small = F.interpolate(y01, size=(side, side), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0, 1)
    oy = max(0, min(canvas - side, int(oy)))
    ox = max(0, min(canvas - side, int(ox)))
    out = y01.new_full((1, 3, canvas, canvas), float(y01.mean()))
    out[..., oy:oy + side, ox:ox + side] = small
    mask = y01.new_zeros((1, 1, canvas, canvas))
    mask[..., oy:oy + side, ox:ox + side] = 1.0
    return out, mask


def _block_valid(mask: torch.Tensor, n_coeffs: int) -> torch.Tensor:
    """逐像素遮罩 → 逐槽有效旗標。**整個 8×8 區塊都在畫面內才算有效。**

    只要有一個像素在外面，那個區塊的所有係數都被邊界汙染，該槽的讀數沒有
    意義；半有效的區塊放進投票只會把雜訊算進去。
    """
    pooled = F.avg_pool2d(mask, BLOCK)          # 區塊內的有效比例
    full = (pooled >= 1.0).to(mask.dtype)       # 1.0 = 整塊都在裡面
    hb, wb = full.shape[-2:]
    return full.view(hb, wb, 1).expand(hb, wb, n_coeffs).reshape(-1)


def search_alignment(y01: torch.Tensor, n_bits: int, *, delta: float,
                     coeffs: Sequence[Tuple[int, int]], repeat: int,
                     scales: Sequence[float],
                     sync: Optional[torch.Tensor] = None,
                     offset_radius: int = BLOCK,
                     canvas: Optional[int] = None):
    """掃尺度與位移，回傳**前導碼吻合度**最高的那一組。

    位移要搜到整格以上，不能只搜格點相位：整格的平移不改變係數的值，卻把槽
    與位元的對應整個推移，而信心度看不出來（見模組 docstring 的表）。

    `sync` 是雙方講好的已知前導碼，對應酬載的前幾個位元。給了就以吻合率
    評分、信心度破平手；沒給就退回純信心度，**而純信心度已知分不出整格
    錯位**，故只在確定沒有幾何攻擊時才可以省略。

    回傳 `(bits, info)`，`info` 帶 `scale`／`oy`／`ox`／`confidence`／
    `sync_match`／`valid_frac`，全部要進 CSV——對錯了會安靜地讀出一組垃圾
    位元，而那與「被編輯打掉」的長相相同。
    """
    coeffs = _check_coeffs(coeffs)
    canvas = int(canvas or y01.shape[-1])
    if canvas % BLOCK:
        raise ValueError(f"畫布邊長必須是 {BLOCK} 的倍數，收到 {canvas}")
    if not len(scales):
        raise ValueError("scales 不可為空；要搜哪些尺度是必填的")

    best = None
    for s in scales:
        side = int(round(canvas / float(s)))
        centre = (canvas - side) // 2
        for dy in range(-offset_radius, offset_radius + 1):
            for dx in range(-offset_radius, offset_radius + 1):
                oy, ox = centre + dy, centre + dx
                if not (0 <= oy <= canvas - side and 0 <= ox <= canvas - side):
                    continue
                cv, mask = undo_magnification(y01, float(s), oy, ox, canvas)
                valid = _block_valid(mask, len(coeffs))
                if float(valid.sum()) < n_bits:      # 有效副本比位元數還少
                    continue
                bits, conf = _extract_core(
                    cv, n_bits, delta=delta, coeffs=coeffs, repeat=repeat,
                    valid=valid)
                if sync is None:
                    match = 0.0
                else:
                    k = int(sync.numel())
                    match = float((bits[:k].cpu() == sync.reshape(-1).long()
                                   ).to(torch.float64).mean())
                # 前導碼吻合率優先，信心度只破平手：整格錯位的候選信心度同樣
                # 是滿的，只有前導碼分得出來。
                score = (match, conf)
                if best is None or score > best[2]:
                    best = (bits, {"scale": float(s), "oy": int(oy),
                                   "ox": int(ox), "confidence": conf,
                                   "sync_match": match,
                                   "valid_frac": float(valid.mean())}, score)
    if best is None:
        raise ValueError(
            "沒有任何候選幾何通過有效槽數的下限——scales 給的範圍可能整個錯了。")
    return best[0], best[1]
