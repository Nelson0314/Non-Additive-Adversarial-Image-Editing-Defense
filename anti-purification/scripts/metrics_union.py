"""把外部文獻用過、但本專案尚未記錄的指標補齊。

為什麼需要這一支
────────────────────────────────────────────────────────────────────
主表的欄位是本專案自己定的標準清單（`src/metrics/standard.py`）：
LPIPS／SSIM／PSNR／VIFp／DISTS。但十一個外部方法的原論文各自報的指標不同，
其中 **FSIM** 出現在五篇（DAYN §4.3、DANP §V-B、SIFM §VII-A、TDAE、
DiffVax 的 `evaluate.py`），本專案的 `MetricSuite` 支援它、卻沒有寫進任何
一張結果表。**CIEDE2000** 是本專案自己的預算單位，同樣只有顏色那一列有值。

這一支只讀既有的 PNG，不重跑攻擊、不重跑編輯、不碰 GPU（FSIM 與 ΔE00 都是
傳統指標，CPU 即可）。輸出獨立成檔，不覆寫任何既有 CSV。

三個輸出
────────────────────────────────────────────────────────────────────
| 檔 | 對象 | 補的欄位 |
|---|---|---|
| `metrics_fidelity_union.csv`   | 防禦圖 vs 原圖 | `fid_fsim`、`fid_delta_e00`、`fid_mse` |
| `metrics_displacement_union.csv` | 編輯(原圖) vs 編輯(防禦圖) | `disp_fsim`、`disp_mse` |
| `metrics_retention_union.csv`  | 淨化後的同一對 | `disp_purified_fsim` |
| `metrics_aesthetic_union.csv`  | 防禦圖本身（無參考） | 美學與自然度四項 |

**美學那一組是無參考指標**：它們只看一張圖，不跟原圖比，所以原圖自己也要量一份
當參照列——`niqe 4.12` 這個數單獨擺出來沒有意義，要對著同一張原圖的值看。

`mse` 由 rms 反推不可靠（rms 的平均方式見 `src/metrics/suite.py`），故直接算。

用法
    python scripts/metrics_union.py --stage fidelity
    python scripts/metrics_union.py --stage displacement
    python scripts/metrics_union.py --stage retention
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import piq  # noqa: E402
import torch  # noqa: E402

from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512
OUT_DIR = Path("runs/report")

#: 淨化算子與其標籤，與 `scripts/purify_run.py` 的 `label()` 同式。
UNDEFENDED = "undefended"


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load(path, device):
    return load_image_tensor(str(path), device, size=RESOLUTION)


def fsim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(piq.fsim(a, b, data_range=1.0))


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean())


def delta_e00(a: torch.Tensor, b: torch.Tensor) -> float:
    from skimage.color import deltaE_ciede2000, rgb2lab
    x = a[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    y = b[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    return float(deltaE_ciede2000(rgb2lab(x), rgb2lab(y)).mean())


def stage_fidelity(device) -> None:
    """防禦圖對原圖。檔名式樣 `<name>__orig.png` 與 `<name>__<cond>__def.png`。"""
    root = Path("runs/defence_portraits")
    rows = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        condition = directory.name
        for defended in sorted(directory.glob(f"*__{condition}__def.png")):
            name = defended.name.split("__")[0]
            original = directory / f"{name}__orig.png"
            if not original.is_file():
                raise SystemExit(f"缺原圖：{original}。保真那一欄的分母缺一張就不可比")
            a, b = load(original, device), load(defended, device)
            with torch.no_grad():
                rows.append({
                    "condition": condition, "image": name,
                    "fid_fsim": round(fsim(a, b), 5),
                    "fid_delta_e00": round(delta_e00(a, b), 4),
                    "fid_mse": round(mse(a, b), 8),
                    "original_png": original.as_posix(),
                    "defended_png": defended.as_posix(),
                })
        print(f"[DONE] {condition:20s} 累計 {len(rows)} 列", flush=True)
    write_csv(OUT_DIR / "metrics_fidelity_union.csv", rows)
    print(f"[ALLDONE] {OUT_DIR / 'metrics_fidelity_union.csv'}（{len(rows)} 列）", flush=True)


def stage_displacement(device) -> None:
    """編輯(原圖) vs 編輯(防禦圖)。路徑直接取 `displacement.csv` 自己記的兩欄。"""
    source = read_csv(Path("runs/edit_defended/displacement.csv"))
    rows = []
    for i, row in enumerate(source, 1):
        a = load(row["undefended_png"], device)
        b = load(row["defended_png"], device)
        with torch.no_grad():
            rows.append({
                "condition": row["condition"], "scenario": row["scenario"],
                "image": row["image"], "prompt_index": row["prompt_index"],
                "disp_fsim": round(fsim(a, b), 5),
                "disp_mse": round(mse(a, b), 8),
            })
        if i % 64 == 0:
            print(f"[{i}/{len(source)}]", flush=True)
    write_csv(OUT_DIR / "metrics_displacement_union.csv", rows)
    print(f"[ALLDONE] {OUT_DIR / 'metrics_displacement_union.csv'}（{len(rows)} 列）", flush=True)


def cell(root: Path, condition: str, purifier: str, scenario: str,
         name: str, index: str) -> Path:
    arm = f"{scenario}_{condition}_{purifier}"
    return root / condition / purifier / arm / f"{name}__p{index}.png"


def stage_retention(device) -> None:
    """淨化後的同一對。條件與算子的組合直接照 `retention.csv` 的列。"""
    root = Path("runs/edit_purified")
    source = read_csv(root / "retention.csv")
    rows = []
    for i, row in enumerate(source, 1):
        a = cell(root, UNDEFENDED, row["purifier"], row["scenario"],
                 row["image"], row["prompt_index"])
        b = cell(root, row["condition"], row["purifier"], row["scenario"],
                 row["image"], row["prompt_index"])
        if not a.is_file() or not b.is_file():
            raise SystemExit(f"缺檔：{a if not a.is_file() else b}")
        with torch.no_grad():
            rows.append({
                "condition": row["condition"], "purifier": row["purifier"],
                "scenario": row["scenario"], "image": row["image"],
                "prompt_index": row["prompt_index"],
                "disp_purified_fsim": round(fsim(load(a, device), load(b, device)), 5),
            })
        if i % 256 == 0:
            print(f"[{i}/{len(source)}]", flush=True)
    write_csv(OUT_DIR / "metrics_retention_union.csv", rows)
    print(f"[ALLDONE] {OUT_DIR / 'metrics_retention_union.csv'}（{len(rows)} 列）", flush=True)


#: (欄名, pyiqa 模型, 越小越好)。七項都是**無參考**：只看一張圖，不跟原圖比。
#: 前四項訓練自人類的美感評分，後三項量的是「偏離自然影像統計多遠」。
AESTHETIC = (
    ("aes_laion", "laion_aes", False),
    ("aes_nima", "nima", False),
    ("aes_musiq_ava", "musiq-ava", False),
    ("aes_topiq_iaa", "topiq_iaa", False),
    ("aes_clipiqa", "clipiqa", False),
    ("aes_niqe", "niqe", True),
    ("aes_brisque", "brisque", True),
)


def stage_aesthetic(device) -> None:
    """防禦圖與原圖各量一份無參考的美學／自然度讀數。"""
    import pyiqa
    models = {}
    for column, name, _ in AESTHETIC:
        try:
            models[column] = pyiqa.create_metric(name, device=device)
        except Exception as error:  # noqa: BLE001
            print(f"[SKIP] {name}：{type(error).__name__} {error}", flush=True)

    root = Path("runs/defence_portraits")
    targets = []
    seen = set()
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        condition = directory.name
        for defended in sorted(directory.glob(f"*__{condition}__def.png")):
            name = defended.name.split("__")[0]
            targets.append((condition, name, defended))
            original = directory / f"{name}__orig.png"
            if name not in seen and original.is_file():
                seen.add(name)
                targets.append(("original", name, original))

    rows = []
    for i, (condition, name, path) in enumerate(targets, 1):
        x = load(path, device).clamp(0, 1)
        row = {"condition": condition, "image": name, "png": path.as_posix()}
        with torch.no_grad():
            for column, _, _ in AESTHETIC:
                model = models.get(column)
                row[column] = round(float(model(x)), 5) if model else ""
        rows.append(row)
        if i % 16 == 0:
            print(f"[{i}/{len(targets)}]", flush=True)
    write_csv(OUT_DIR / "metrics_aesthetic_union.csv", rows)
    print(f"[ALLDONE] {OUT_DIR / 'metrics_aesthetic_union.csv'}（{len(rows)} 列）",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True,
                        choices=("fidelity", "displacement", "retention", "aesthetic"))
    args = parser.parse_args()
    device = torch.device("cpu")
    {"fidelity": stage_fidelity,
     "displacement": stage_displacement,
     "retention": stage_retention,
     "aesthetic": stage_aesthetic}[args.stage](device)


if __name__ == "__main__":
    main()
