"""`src/utils/io.py`：由 `src/experiment/executors.py` 抽出的兩個 I/O 函式。

抽出的理由是依賴而非美觀：`scripts/apa_baseline.py`（主線）為了這兩個函式
匯入 `executors`，而 `executors` 拉進整個 `src/experiment/` 與
`src/defense/objective.py`、`src/purify/`、`src/residual/site_warp.py`。
切斷這條邊之後 `src/experiment/` 才能整包移到 `legacy/`（`PLAN.md` §6.1a）。

**行為必須逐字不變**：舊主線的批次仍在用 `executors` 的名字讀寫同一批 CSV，
搬家改變了輸出格式的話，既有 `runs/` 會變成不可比。
"""

import csv

import torch
from PIL import Image

from src.utils.device import get_device

DEV = get_device()


def test_欄位取全部列的聯集而非第一列的鍵(tmp_path):
    """只認第一列會靜默丟掉後面才出現的欄位，而 CSV 看起來仍然完整。"""
    from src.utils.io import write_csv

    p = write_csv(tmp_path / "r.csv",
                  [{"a": 1}, {"a": 2, "b": 3}, {"c": 4}])
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert list(rows[0].keys()) == ["a", "b", "c"], "欄位順序應為首次出現序"
    assert rows[2]["c"] == "4"


def test_未指定size時不縮放(tmp_path):
    """留存的產物本來就是本批解析度，預設縮放會多一次不必要的重採樣。"""
    from src.utils.io import load_image_tensor

    Image.new("RGB", (37, 37), (10, 20, 30)).save(tmp_path / "a.png")
    x = load_image_tensor(tmp_path / "a.png", DEV)
    assert x.shape == (1, 3, 37, 37)
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0


def test_指定size時縮放到正方(tmp_path):
    """外部素材的尺寸由檔案決定；不縮放的話錯誤會發生在 mse_loss 的
    broadcast 裡，看不出來源是一張沒有縮放的素材。"""
    from src.utils.io import load_image_tensor

    Image.new("RGB", (64, 32), (10, 20, 30)).save(tmp_path / "b.png")
    x = load_image_tensor(tmp_path / "b.png", DEV, size=16)
    assert x.shape == (1, 3, 16, 16)


def test_executors的舊名字仍指向同一個實作(tmp_path):
    """舊主線的批次仍用 `executors.load_image_tensor` 的名字。抽出後若變成
    兩份實作，兩邊的行為會慢慢分岔而沒有症狀。"""
    from src.experiment import executors
    from src.utils import io

    assert executors.load_image_tensor is io.load_image_tensor
    assert executors.write_csv is io.write_csv
