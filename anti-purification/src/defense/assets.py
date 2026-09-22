"""影像與色彩庫的讀取。`scripts/immunise.py` 與 `scripts/evaluate_defence.py` 共用。

`scripts/carrier_search.py` 裡有同名的區域函式，那一支是含指令的舊路徑，
保留原樣不動，兩條路徑不共用狀態。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_LIB_CACHE = {}


def load_image(path, device):
    import numpy as np
    import torch
    from PIL import Image
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def save_png(x, path):
    import numpy as np
    from PIL import Image
    a = (x.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255)
    Image.fromarray(a.round().astype(np.uint8)).save(path)


def library(spec):
    """整個雜湊驗過的色彩庫，不按類別過濾；過濾在取用端做。

    `ade20k_classes=None` 是刻意的：臉配 `person`、衣物配 `apparel`，
    取用端要在多個類別之間挑，雜湊與 Lab 單位的檢查仍走 `NCFLibrary`。
    """
    from .ncf_library import NCFLibrary
    lib = spec['library']
    key = lib['sha256']
    if key not in _LIB_CACHE:
        obj = NCFLibrary(ROOT / lib['path'], expected_sha256=lib['sha256'],
                         required_classes=lib['class_weights'],
                         ade20k_classes=None)
        _LIB_CACHE[key] = obj.records
    return _LIB_CACHE[key]


def palette_of(spec, palette_id):
    rec = next(r for r in library(spec) if r['id'] == palette_id)
    return rec['mean'], rec['covariance']


def palette_bank(spec, class_prefix):
    bank = [r for r in library(spec) if r['id'].startswith(class_prefix)]
    if not bank:
        raise SystemExit(f'色彩庫裡沒有 {class_prefix} 開頭的分布')
    return [(r['mean'], r['covariance']) for r in bank]
