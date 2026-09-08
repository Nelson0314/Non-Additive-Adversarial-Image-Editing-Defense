"""編輯後還認不認得出那是誰、佈局還是不是同一件事。**只讀已存的影像。**

在「不管怎麼換，都看不出那個人是誰」這個目標下，主讀數是身分本身而不是
像素位移。位移量的是變了多少，量不到還認不認得出來——兩者已經分歧過：
`runs/ip2p_patch_carrier/` 的位移看起來像有效的防禦，而拉圖看到受保護的
工具箱照樣變成紅色。

三個數的意義見 `src/metrics/identity.py`。**`id_orig` 不可省**：攻擊自己
就已經動了身分（實測 0.77–0.96 而不是 1.0），不扣掉它就分不出「防禦壓低了
身分」與「這個編輯本來就會把人改掉」。

用法：
    python scripts/identity_probe.py --out runs/x/identity.csv \
        --entry clothing_plain=runs/ip2p_face_defence/clothing_plain_task_x
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.identity import identity_row  # noqa: E402
from src.metrics.layout import layout_of  # noqa: E402
from src.utils.io import write_csv  # noqa: E402


def _load(p: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def _triple(run: Path, image: str):
    """(原圖, 編輯(原圖), 編輯(防禦圖))。三者缺一就拋錯，不猜。"""
    orig = run / f"{image}__orig.png"
    hits = sorted(run.glob(f"{image}__*__edit_orig.png"))
    if not orig.exists() or not hits:
        raise SystemExit(f"{run} 缺 {image} 的原圖或編輯圖")
    cond = hits[0].name.split("__")[1]
    edit_def = run / f"{image}__{cond}__edit_def.png"
    if not edit_def.exists():
        raise SystemExit(f"{run} 缺 {image} 的 edit_def")
    return _load(orig), _load(hits[0]), _load(edit_def), cond


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", action="append", required=True,
                    help="label=目錄。可重複。")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows = []
    for spec in args.entry:
        if "=" not in spec:
            raise SystemExit(f"--entry 要寫成 label=目錄，收到 {spec!r}")
        label, path = spec.split("=", 1)
        run = Path(path)
        names = sorted({p.name.split("__")[0] for p in run.glob("*__orig.png")})
        if not names:
            raise SystemExit(f"{run} 底下沒有 *__orig.png")
        for image in names:
            x, eo, ed, cond = _triple(run, image)
            r = {"image": image, "label": label, "condition": cond}
            r.update(identity_row(x, eo, ed, device=args.device))
            # 佈局：身分讀數與 LPIPS 都判不準「人還在但被整個換掉」，
            # 而編輯前後的分割 IoU 直接顯示那件事。參照是**攻擊在原圖上的
            # 結果**，不是原圖——問的是防禦有沒有讓輸出偏離它原本會長成的樣子。
            r.update(layout_of(eo, ed, device=args.device))
            rows.append(r)
            print(f"{image:<30s} {label:<18s} id_orig={r['id_orig']} "
                  f"id_def={r['id_def']} 臉={r['face_found_edit_def']} "
                  f"iou_person={r['iou_person']} iou_classes={r['iou_classes']}"
                  f"({r['n_classes']} 類)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
    print(f"\n寫出 {args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
