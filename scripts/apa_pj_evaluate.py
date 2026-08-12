"""把 `run_stage.py` 訓練出的 apa_pj 防禦圖，用與其他條件**同一個**評測函式量一次。

存在理由：apa_pj 走專案自己的段 1（`run_stage.py`），其餘七個條件走
`apa_native_full_pipeline.py`。兩邊若各自評測，保真度與抗編輯的數字就不
可比——同一個 SDEdit 種子、同一個 strength、同一組指標實作，這三件事必須
逐字相同。故此處只做「讀圖 → 呼叫那支腳本的 `evaluate()`」，不重新實作
任何度量。

apa_pj 的 φ 直接取訓練產物、不經段 2 射線縮放：投影模式下訓練即評測
（DEC-019，EXP-s3t20_pj 實測 `scale_k` 0.938–1.000，縮放為空操作）。

用法：
    python scripts/apa_pj_evaluate.py --out runs/apa_pj_eval \
        --shards runs/apapj_bf/apa_pj/butterfly_00 runs/apapj_ct/apa_pj/coot_00
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import apa_native_full_pipeline as pipe  # noqa: E402
from src.experiment import executors  # noqa: E402
from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402

# run_stage 的影像 id 帶 `_00` 後綴（每類一個子目錄的慣例），
# `apa_native_full_pipeline` 的資料集用的是類別名本身。
STEM = {"butterfly_00": "butterfly", "coot_00": "coot", "panda_00": "panda"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shards", nargs="+", required=True,
                    help="每個是 run_stage 產出的 <batch>/apa_pj/<image_id> 目錄")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(pipe.MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)

    dataset = {it["name"]: it for it in pipe.load_dataset()}

    rows = []
    for shard in args.shards:
        shard = Path(shard)
        image_id = shard.name
        name = STEM[image_id]
        item = dict(dataset[name])
        item["path01"] = executors.load_image_tensor(
            item["path"], sd.device, size=pipe.RESOLUTION)

        x_def = executors.load_image_tensor(
            shard / "x_def.png", sd.device, size=pipe.RESOLUTION)
        print(f"=== {name} / apa_pj ===", flush=True)
        t0 = time.time()
        metrics, edit_orig, edit_def = pipe.evaluate(sd, suite, aes, item, x_def)
        save_image(item["path01"], args.out / f"{name}__orig.png")
        save_image(x_def, args.out / f"{name}__apa_pj__def.png")
        save_image(edit_orig, args.out / f"{name}__apa_pj__edit_orig.png")
        save_image(edit_def, args.out / f"{name}__apa_pj__edit_def.png")
        row = {"image": name, "condition": "apa_pj", "stage1_seconds": 0.0,
               "total_seconds": round(time.time() - t0, 1), **metrics}
        rows.append(row)
        print(row, flush=True)

    executors.write_csv(args.out / "apa_native_full.csv", rows)
    print(f"\n表：{args.out / 'apa_native_full.csv'}")


if __name__ == "__main__":
    main()
