"""批次產物的讀寫：CSV 與影像張量。

由 `src/experiment/executors.py` 抽出。**抽出的理由是依賴而非美觀**：
`scripts/apa_baseline.py`（主線）只為了這兩個函式而匯入 `executors`，
而 `executors` 拉進整個 `src/experiment/` 與 `src/defense/objective.py`、
`src/purify/`、`src/residual/site_warp.py`——45 個主線遞移依賴裡有相當
一部分是這條邊造成的（`docs/PLAN.md` §6.1a）。

`executors` 仍以 `from src.utils.io import ...` 沿用同一份實作，不複製一份：
兩份實作會慢慢分岔，而 CSV 的欄位規則分岔之後既有 `runs/` 會變成不可比。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """欄位取全部列的聯集，依首次出現排序。

    不取第一列的鍵：`optimize` 的 history 在階段切換時會多出欄位，只認第一列
    會靜默丟掉後面的欄位，而 CSV 看起來仍然完整。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def load_image_tensor(path: Path, device, size: Optional[int] = None
                      ) -> torch.Tensor:
    """讀回 PNG 成 (1,3,H,W)、[0,1]。對照側的參照走這條。

    `size` 明給時縮放到 `size × size`。**外部素材必須經過這一步**：它們的
    尺寸由檔案決定，與本批的 `--resolution` 無關，而 latent 的邊長是影像邊長
    ÷ 8，兩者不符時錯誤發生在損失函式裡（`mse_loss` 的 broadcast），
    看不出來源是一張沒有縮放的素材。留存的產物（`x_def.png` 等）本來就
    是本批解析度，故預設不縮放。
    """
    from PIL import Image
    import torch.nn.functional as F
    import torchvision.transforms as T

    img = Image.open(path).convert("RGB")
    x = T.ToTensor()(img).unsqueeze(0).to(device)
    if size is not None and x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic",
                          antialias=True).clamp(0, 1)
    return x
