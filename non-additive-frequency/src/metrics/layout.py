"""編輯前後的**空間佈局**還是不是同一件事。

為什麼需要它
────────────────────────────────────────────────────────────────────
身分讀數（`identity.py`）回答「還認不認得出是誰」，位移（LPIPS）回答「像素
變了多少」。兩者都判不準一種失效：**人還在、佈局也還在，但被整個換成別人**
——實測 `task_obj_remove_284852` 與 `410264` 的編輯輸出是**服飾型錄照**，
原本的人不見了，而 LPIPS 只說「變很多」，身分讀數只說「相似度低」，
兩者都沒說出「這張圖已經不是同一個場景」。

Trippodo et al.（ACM MM 2025，arXiv:2509.10359）對既有免疫度量提出同樣的
質疑，並提出 semantic IoU：用分割遮罩量空間佈局被破壞的程度。本模組是那個
想法在本專案的實作。

用哪個分割器
────────────────────────────────────────────────────────────────────
ATR 人體解析（`carrier_mask.parse_atr`），**因為它已經在跑**——載體與臉部
主體都由它給，多算一次 IoU 幾乎沒有成本。它是以人為中心的 18 類，而本專案
現行的保護對象正是人，故「這個人的佈局還在不在」正是要問的東西。

三個數
────────────────────────────────────────────────────────────────────
    iou_person   非背景像素的 IoU。人還在不在、位置有沒有變
    iou_classes  逐類 IoU 在「兩邊至少一邊出現過」的類別上取平均。
                 衣服變成別的東西、頭髮不見了這類改變都會反映在這裡
    n_classes    上面那個平均用了幾類——**分母要一起報**，只有一兩類時
                 平均值不可解讀

**兩邊都沒有人時 IoU 未定義**，回 `None` 而不是 1.0 或 0.0：那是「兩張圖都
沒有人」，與「佈局完全一致」是不同的事。
"""

from __future__ import annotations

from typing import Optional

import torch


def _iou(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """兩個布林遮罩的 IoU。聯集為空時回 `None`（未定義，不是 1.0）。"""
    union = float((a | b).sum())
    if union == 0.0:
        return None
    return float((a & b).sum()) / union


def layout_row(seg_a: torch.Tensor, seg_b: torch.Tensor) -> dict:
    """兩張 ATR 類別圖 → 逐列寫進 CSV 的三個欄位。

    `seg_a` 取「編輯（原圖）」、`seg_b` 取「編輯（防禦圖）」：問的是防禦有沒有
    讓攻擊的輸出偏離它原本會長成的樣子，故參照是**攻擊在原圖上的結果**，
    不是原圖本身。
    """
    if seg_a.shape != seg_b.shape:
        raise ValueError(
            f"兩張類別圖的形狀不同：{tuple(seg_a.shape)} 與 {tuple(seg_b.shape)}")
    person = _iou(seg_a != 0, seg_b != 0)
    ious = []
    for c in torch.unique(torch.cat([seg_a.reshape(-1), seg_b.reshape(-1)])):
        if int(c) == 0:
            continue
        v = _iou(seg_a == c, seg_b == c)
        if v is not None:
            ious.append(v)
    return {
        "iou_person": "" if person is None else round(person, 5),
        "iou_classes": "" if not ious else round(sum(ious) / len(ious), 5),
        "n_classes": len(ious),
    }


def layout_of(edit_orig: torch.Tensor, edit_def: torch.Tensor,
              device=None) -> dict:
    """兩張影像 (1,3,H,W) → 佈局讀數。內部各跑一次 ATR 解析。"""
    from src.defense.carrier_mask import parse_atr

    return layout_row(parse_atr(edit_orig, device=device),
                      parse_atr(edit_def, device=device))
