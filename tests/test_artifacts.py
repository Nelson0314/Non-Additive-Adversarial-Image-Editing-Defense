"""影像留存 — `src/utils/artifacts.py`。

存在理由：留存是本專案唯一的證據來源（`CLAUDE.md`「資料保全」），而留存
失敗的代價不對稱——它發生在計算**之後**，一格跑完幾十分鐘才在存檔那行
中止，逐格紀錄只留下 traceback，訓練過程全部作廢。故凡是進得了
`save_image` 的張量型態都必須有測試釘住。
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.artifacts import _to_uint8, save_image, save_residual

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
def test_計算精度的張量都存得下來(dtype, tmp_path: Path):
    """bf16 是本輪的執行精度，而 numpy 沒有 bfloat16。

    2026-08-06：`_to_uint8` 直接對計算精度的張量呼叫 `.numpy()`，bf16 下以
    `TypeError: Got unsupported ScalarType BFloat16` 中止，N3 兩張圖因此在
    訓練跑完之後才失敗（`_train_nonadditive` 的 `save_image(result.x_base,…)`）。
    """
    x = torch.rand(1, 3, 8, 8).to(dtype)
    out = _to_uint8(x)
    assert out.dtype == np.uint8
    assert out.shape == (8, 8, 3)

    path = tmp_path / f"{dtype}.png"
    save_image(x, path)
    assert path.exists()


@pytest.mark.parametrize("dtype", DTYPES)
def test_殘差視覺化同樣不限計算精度(dtype, tmp_path: Path):
    d = (torch.rand(1, 3, 8, 8) - 0.5).to(dtype) * 0.01
    gain = save_residual(d, tmp_path / f"res_{dtype}.png")
    assert gain > 0
    assert (tmp_path / f"res_{dtype}.png").exists()


def test_轉換的數值與fp32一致():
    """轉 fp32 只是為了讓 numpy 收得下，不得改變像素值。

    以 fp32 為基準比對：bf16 的輸入本身已經量化過，故先量化再轉、與轉了
    再量化，兩者必須給出同一張圖，否則「存下來的」與「算出來的」不是同一件事。
    """
    x32 = torch.rand(1, 3, 8, 8)
    x16 = x32.to(torch.bfloat16)
    assert np.array_equal(_to_uint8(x16), _to_uint8(x16.float()))
