"""載體在**還沒開始最佳化之前**長什麼樣：邊界渲染與外觀讀數。

**所以順序必須是**：先把載體推到它自己約束的邊界、完全不最佳化、渲染出來
檢查；通不過就換強度或換載體，不進訓練。這道檢查不用 GPU，也不用載入任何
擴散模型——它問的是「這個載體在它能走到最遠的地方長什麼樣」，與目標函數無關。

三個可以量的外觀條件（是讀數，不是門檻）
────────────────────────────────────────────────────────────────────
1. **高頻殘差比** `hf_ratio_rgb_total`：大於 1 表示載體在往高頻加東西。
   量法沿用 `lowfreq_color.highfreq_report`，與淨化端看到的是同一組頻帶。
2. **主體仍是同一個人**：防禦圖自己與原圖的人臉餘弦。整圖濾鏡會改變主體的
   顏色，那是「顏色濾鏡」的定義；但改到認不出人就越界了。**原圖本來就偵測不到
   臉時這一條不適用**（留空），那是「沒有臉可保護」，與「防禦把臉毀了」不同——
   兩者混在一起會讓沒有人的影像全部被誤擋。
3. **主體的像素沒有被畫上東西**：載體支撐之外必須逐位元相同。衣物載體有
   支撐，整圖濾鏡沒有（支撐即全圖），故這一項只對有支撐的載體檢查。

**「看起來自然」不在這三個數裡，那是人眼判定。** 本模組把邊界圖存下來讓
使用者看，三個數只是幫忙定位，不構成任何通過或不通過。
"""

from __future__ import annotations

from typing import Optional

import torch

from src.defense.lowfreq_color import highfreq_report

MAX_HIGHFREQ_RATIO = 1.0
# VGGFace2 上「同一人」的常見參照，取自 `identity.SAME_PERSON_REFERENCE`。
# **只作讀表的參照**，不是門檻。
MIN_SUBJECT_IDENTITY = 0.55

GATE_FIELDS = (
    "gate_psnr", "gate_dists", "gate_deltaE00",
    "gate_hf_rgb_total", "gate_hf_lab_L", "gate_hf_lab_a", "gate_hf_lab_b",
    "gate_subject_identity", "gate_face_in_original",
    "gate_outside_support_max_abs",
    "gate_hf_below_reference", "gate_id_above_reference", "gate_support_exact",
)


@torch.no_grad()
def box_corner_render(param, x01: torch.Tensor, seed: int, *,
                    draw: str = "corner") -> torch.Tensor:
    """把可學參數推到 ±radius 盒角、投影之後渲染，**完全不最佳化**。

    這張圖回答的是「可學參數走到盒角時長什麼樣」。它**不是**可達集合的極端
    點：顏色載體的顏色位移主要由 `T0` 與 `amplitude` 決定，兩者都不在
    `delta` 的盒子裡（見 `carrier_objectives.box_corner_init`）。
    """
    from src.defense.carrier_objectives import box_corner_init

    param.reset(x01, seed)
    box_corner_init(param, x01, seed, draw=draw)
    return param.render(x01).detach()


@torch.no_grad()
def gate_row(x01: torch.Tensor, x_def: torch.Tensor, *,
             support: Optional[torch.Tensor] = None,
             suite=None, device=None) -> dict:
    """原圖與防禦圖 → 逐列寫進 CSV 的守門讀數。

    `support` 是載體的支撐 (1,1,H,W)；給了就檢查支撐外逐位元相同。整圖濾鏡
    沒有支撐概念，傳 `None`，該欄留空而不是填 0——「沒有這個約束」與
    「約束通過」是不同的事。
    """
    from src.metrics.identity import embed, similarity

    hf = highfreq_report(x01, x_def)
    row = {
        "gate_hf_rgb_total": round(hf["hf_ratio_rgb_total"], 5),
        "gate_hf_lab_L": round(hf["hf_ratio_lab_L"], 5),
        "gate_hf_lab_a": round(hf["hf_ratio_lab_a"], 5),
        "gate_hf_lab_b": round(hf["hf_ratio_lab_b"], 5),
    }
    if suite is not None:
        d = suite.pairwise(x01, x_def)
        row["gate_psnr"] = round(float(d["psnr"]), 4)
        row["gate_dists"] = round(float(d["dists"]), 5)
        row["gate_deltaE00"] = round(float(d.get("deltaE00", float("nan"))), 4)

    # 三種情況要分開，混在一起會把「這張圖沒有臉」讀成「防禦把臉毀了」：
    #   原圖就沒有臉        → 這一條不適用，留空
    #   原圖有、防禦圖沒有  → 不通過（最嚴重的那種）
    #   兩邊都有            → 比餘弦
    e_orig = embed(x01, device)
    if e_orig is None:
        ident, id_pass = None, ""
    else:
        e_def = embed(x_def, device)
        if e_def is None:
            ident, id_pass = None, False
        else:
            ident = similarity(e_orig, e_def)
            id_pass = ident >= MIN_SUBJECT_IDENTITY
    row["gate_subject_identity"] = "" if ident is None else round(ident, 5)
    row["gate_face_in_original"] = e_orig is not None

    if support is None:
        row["gate_outside_support_max_abs"] = ""
        row["gate_support_exact"] = ""
    else:
        w = support.to(x_def)
        outside = float(((x_def - x01).abs() * (1 - w)).max())
        row["gate_outside_support_max_abs"] = round(outside, 8)
        row["gate_support_exact"] = outside == 0.0

    # 這兩欄只是「與參照值比起來在哪一邊」，不是通過與否，也不影響任何流程。
    row["gate_hf_below_reference"] = row["gate_hf_rgb_total"] <= MAX_HIGHFREQ_RATIO
    row["gate_id_above_reference"] = id_pass
    return row
