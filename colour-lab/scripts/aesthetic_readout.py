"""把 lab 裡所有防禦圖的美學／品質分數量一遍，逐張寫成一份 CSV。

量什麼
────────────────────────────────────────────────────────────────────
| 欄 | 來源 | 是什麼 |
|---|---|---|
| `nima` | pyiqa NIMA-AVA | 美學評分，訓練自 AVA 的人類評分，**不比對原圖** |
| `cnniqa` | pyiqa CNNIQA | 無參考的品質評分，**不比對原圖** |
| `clip_image` | CLIP ViT-B/32 影像—影像餘弦 | 防禦圖對原圖，不是 CLIP-T |
| `deltae00` | `src/defense/delta_e_torch` | 逐像素色差平均，與原圖比 |

前兩項是**無參考**的：它們問「這張圖本身像不像一張好照片」，不問「離原圖多遠」。
這正是「失真可見、但不像壞掉的圖」這個目標要的那一側。

**這不是門檻。** 本專案的 NIQE、平均色差、CVaR 尾端、臉框色差四道自動門檻都被
最佳化鑽過，NIMA 是同一類的純量聚合讀數，拿它當選解準則會是第五道。這支只把
數字擺出來。

用法
    python scripts/aesthetic_readout.py --out runs/report/aesthetic.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.defense.delta_e_torch import delta_e00_torch  # noqa: E402
from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512

#: 條件名 → 防禦圖目錄。每個目錄裡是 `<名稱>__def.png` 或 `<名稱>__<條件>__def.png`。
GROUPS = {
    "curve_control": "runs/curve_control/defended",
    "ladder_cap16": "runs/amplitude_ladder/defended/cap16",
    "ladder_cap22": "runs/amplitude_ladder/defended/cap22",
    "ladder_cap28": "runs/amplitude_ladder/defended/cap28",
    "ladder_cap36": "runs/amplitude_ladder/defended/cap36",
    "no_cast_identity_start": "runs/no_cast_ceiling/defended/no_cast_band",
    "no_cast_jittered_start": "runs/no_cast_from_random_start/defended/no_cast_band",
    "objective_operating_point": "runs/objective_no_id/defended/operating_point",
    "objective_out_no_id": "runs/objective_no_id/defended/out_no_id",
    "objective_out_no_id_diffusion10":
        "runs/objective_no_id/defended/out_no_id_diffusion10",
}


def originals(data: Path) -> dict:
    out = {}
    for sub in ("man", "woman"):
        for path in sorted((data / sub).glob("*.png")):
            out[path.stem] = path
    return out


def defended(directory: Path) -> dict:
    out = {}
    for path in sorted(directory.glob("*__def.png")):
        out[path.name.split("__")[0]] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/portraits"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = AestheticSuite(device)
    source = originals(args.data)
    print(f"裝置 {device}，原圖 {len(source)} 張", flush=True)

    rows = []
    for name, path in source.items():
        x = load_image_tensor(path, device, size=RESOLUTION)
        rows.append({"condition": "original", "image": name,
                     **{k: round(v, 4) for k, v in suite.measure(x).items()},
                     "clip_image": 1.0, "deltae00": 0.0,
                     "png": path.as_posix()})
    print(f"[DONE] original {len(rows)} 張", flush=True)

    wanted = args.groups or list(GROUPS)
    for condition in wanted:
        directory = Path(GROUPS[condition])
        if not directory.is_dir():
            print(f"[SKIP] {condition}：{directory} 不存在", flush=True)
            continue
        files = defended(directory)
        if not files:
            print(f"[SKIP] {condition}：目錄裡沒有 __def.png", flush=True)
            continue
        for name, path in files.items():
            if name not in source:
                print(f"[SKIP] {condition}/{name}：找不到對應的原圖", flush=True)
                continue
            x = load_image_tensor(source[name], device, size=RESOLUTION)
            y = load_image_tensor(path, device, size=RESOLUTION)
            with torch.no_grad():
                de = float(delta_e00_torch(x, y))
            rows.append({
                "condition": condition, "image": name,
                **{k: round(v, 4) for k, v in suite.measure(y).items()},
                "clip_image": round(suite.clip_image_similarity(x, y), 4),
                "deltae00": round(de, 3),
                "png": path.as_posix(),
            })
        print(f"[DONE] {condition} {len(files)} 張", flush=True)
        write_csv(args.out, rows)

    write_csv(args.out, rows)
    print(f"[ALLDONE] {args.out}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
