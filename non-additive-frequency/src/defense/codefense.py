"""共防禦參照：讓量測的兩側吃同一個防禦變換。

問題
────────────────────────────────────────────────────────────────────
現行的非幾何參照是

    effect(p) = LPIPS( 編輯(原圖), 編輯(p(防禦圖)) )

它預設「防禦圖與原圖在內容上是同一張」。**防禦一旦可見，這個前提就不成立**：
位移被內容差異灌水，而空白地板扣不掉那一份——地板本身依賴防禦是什麼。

作法
────────────────────────────────────────────────────────────────────
專案在幾何類算子上已經解過同型的問題：`crop_resize` 改變取景，於是兩側都吃
同一個算子，地板由構造為 0。把防禦本身當成那個算子：

    effect_codefense(p) = LPIPS( 編輯(p(D(x))), p(D(編輯(x))) )

左側是攻擊方實際拿到的；右側是「防禦沒有起作用時他**應該**拿到的」——真正的
編輯，再經過同樣的防禦與同樣的淨化。三個性質：

1. `D = identity` 且 `p = identity` 時**恰為 0**。
2. `D = identity`、`p ≠ identity` 時它就是該算子的空白地板，且是對稱的
   （兩側都吃 p），比現行地板少一項不對稱。
3. 右側**不需要任何額外的編輯**：`編輯(原圖)` 已在快取裡，只是多套一次 `D`
   與一次 `p`。故這個讀數的 GPU 成本近似為零。

適用範圍
────────────────────────────────────────────────────────────────────
`D` 必須能套用在**任意影像**上，才有 `D(編輯(x))` 這個東西。

- **色彩族**（`color_curve`／`color_grid`）是逐點或逐像素查表的映射，
  參數與原圖無關，套到編輯圖上定義良好 → `exact`。
- **相位族**（`phase`／`phase_gain`／…）的兩個閘由**原圖自己的頻譜**算出，
  同一組參數套到另一張影像上不是同一個算子 → `not_applicable`。
  這不是實作缺口，是那一族沒有這個物件。
- **空白地板**（`condition = none`）的 `D` 就是恆等 → `identity`。

回傳 `None` 表示 `not_applicable`；呼叫端照實記在 `codefense_status` 欄，
不得靜默當成恆等。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch

from src.defense.color_param import (
    ColorCurveParam, ColorCurveRandomParam, ColorGridParam, ColorGridRandomParam,
)

# 條件名 → 類別。只有能套用在任意影像上的參數化才在這裡。
POINTWISE_FAMILIES = {
    "color_curve": ColorCurveParam,
    "color_curve_rand": ColorCurveRandomParam,
    "color_grid": ColorGridParam,
    "color_grid_rand": ColorGridRandomParam,
}

STATUS_EXACT = "exact"
STATUS_IDENTITY = "identity"
STATUS_NOT_APPLICABLE = "not_applicable"


def _int_field(row: dict, key: str, default: int) -> int:
    """CSV 讀回來一律是字串，空字串代表那一批沒有這一欄。"""
    v = row.get(key, "")
    return default if v in ("", None) else int(float(v))


def build_codefense(row: dict, run_dir: Path, image: str,
                    device, dtype) -> Tuple[Optional[Callable], str]:
    """由 `results.csv` 的一列 ＋ 存下的參數重建 `D`。

    回傳 `(D, status)`。`D` 是 `(1,3,H,W) → (1,3,H,W)` 的可呼叫物。

    **參數檔缺了就拋錯，不退回恆等**：退回恆等會讓表上多出一堆看起來像
    「地板」的列，而它們其實是漏讀。跑掃描時要帶 `--save-weights`。
    """
    cond = row.get("condition", "")
    if cond in ("none", ""):
        return (lambda x: x), STATUS_IDENTITY
    cls = POINTWISE_FAMILIES.get(cond)
    if cls is None:
        return None, STATUS_NOT_APPLICABLE

    radius = float(row.get("radius", 0.0) or 0.0)
    is_random = cond.endswith("_rand")
    # 隨機對照的 `params()` 是空的，`--save-weights` 存不到東西。它們由
    # **種子**完整決定，故改由 `defense_seed` 重抽。抽出來的張量形狀與影像
    # 尺寸無關（曲線是 (1,3,K)、網格是 (1,12,D,G,G)），所以拿一個小的探針
    # 影像重建與拿原尺寸重建逐位元相同。
    tensors = None
    if not is_random:
        wpath = run_dir / f"{image}__{cond}__w.pt"
        if not wpath.exists():
            raise FileNotFoundError(
                f"共防禦參照需要防禦端的參數，但找不到 {wpath}。"
                f"產生防禦圖那一批要帶 --save-weights。")
        tensors = torch.load(wpath, map_location="cpu", weights_only=True)
        if len(tensors) != 1:
            raise ValueError(
                f"{wpath} 存了 {len(tensors)} 個張量，色彩族只該有一個。")

    extra = ({"draw": row.get("color_rand_draw") or "corner"}
             if is_random else {})
    if cond.startswith("color_curve"):
        param = cls(radius=radius,
                    pieces=_int_field(row, "color_pieces", 64),
                    bound_mode=row.get("color_bound_mode") or "symmetric",
                    **extra)
    else:
        param = cls(radius=radius,
                    grid=_int_field(row, "color_grid", 8),
                    luma_bins=_int_field(row, "color_luma_bins", 8),
                    **extra)

    seed_field = row.get("defense_seed", "")
    if is_random and seed_field in ("", None):
        raise ValueError(
            f"{cond} 由種子完整決定，但這一列沒有 defense_seed 欄——"
            f"該批是加上這一欄之前跑的，共防禦參照無法重建。")
    probe = torch.zeros(1, 3, 8, 8, device=device, dtype=dtype)
    param.reset(probe, seed=int(float(seed_field)) if is_random else 0)
    if not is_random:
        live = param.params()[0]
        stored = tensors[0].to(device=device, dtype=dtype)
        if live.shape != stored.shape:
            raise ValueError(
                f"存下的形狀 {tuple(stored.shape)} 與由 CSV 重建的 "
                f"{tuple(live.shape)} 不合——構造設定與存檔時不同，"
                f"載進去的是別的東西。")
        with torch.no_grad():
            live.copy_(stored)

    def apply(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return param.render(x.to(device=device, dtype=dtype))

    return apply, STATUS_EXACT
