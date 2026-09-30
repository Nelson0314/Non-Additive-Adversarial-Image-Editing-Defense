"""CSV 與影像張量 I/O。

CSV 欄位依全部資料列首次出現順序聯集；write_csv() 覆寫指定檔案。write_sorted_csv()
的欄位依名稱排序。load_dataset_images() 列出資料集中 prompts.yaml 所列類別的影像。
load_image_tensor() 讀取 RGB 並回傳 (1,3,H,W)、[0,1] 張量；明給 size 時
使用 bicubic、antialias=True 縮放為正方形，保留既有的取樣與 clamp 順序。"""

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
    if size is not None and x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bicubic",
                          antialias=True).clamp(0, 1)
    return x


def load_dataset_images(root: Path, only) -> list:
    """`prompts.yaml` 中除 `edits` 以外的每個類別目錄下的 PNG；`only` 非空時只保留這些影像名。"""
    import yaml

    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    out = [{"name": img.stem, "class": cls, "path": img}
           for cls in sorted(k for k in spec if k != "edits")
           for img in sorted((root / cls).glob("*.png")) if not only or img.stem in only]
    if not out:
        raise SystemExit(f"{root} 找不到影像（--images 過濾後為空）")
    return out


def write_sorted_csv(path: Path, rows) -> None:
    """欄位取全部列的聯集並依名稱排序後覆寫 `path`。"""
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
