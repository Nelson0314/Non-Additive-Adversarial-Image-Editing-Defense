"""結果存取 — STRUCTURE.md §5。

每次執行在 experiments/<stage>/<timestamp>/ 下產生：
config_snapshot.yaml、results.csv、summary.md、env.json。
summary.md 須明確區分引用值（DAYN Table 1）與實測值。
"""

import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from src.utils.device import get_device_info


def make_run_dir(stage: str, root: str = "experiments") -> Path:
    run_dir = Path(root) / stage / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(run_dir: Path, config: dict) -> None:
    text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    (run_dir / "config_snapshot.yaml").write_text(text, encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    """欄位取所有列鍵之聯集（保序），缺值留空。"""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_env_json(run_dir: Path) -> None:
    import importlib.metadata as md

    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "piq", "peft",
                 "opencv-contrib-python", "numpy"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "device": get_device_info(),
        "packages": packages,
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")


def save_summary(run_dir: Path, text: str) -> None:
    (run_dir / "summary.md").write_text(text, encoding="utf-8")


def save_image(path: Path, image01: torch.Tensor) -> None:
    """(1,3,H,W) [0,1] → png（無失真）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    T.ToPILImage()(image01[0].detach().cpu().clamp(0, 1)).save(path)


def load_image(path: Path) -> torch.Tensor:
    """png → (1,3,H,W) [0,1]。"""
    return T.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
