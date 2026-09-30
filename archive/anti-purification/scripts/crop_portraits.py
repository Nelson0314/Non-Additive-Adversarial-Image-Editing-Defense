"""把使用者指定的人像切成 512² 並組成資料集目錄。

裁切怎麼定位
────────────────────────────────────────────────────────────────────
**以臉框定位，不取畫面中心。** 人像構圖常把臉放在畫面上半或三分線上，
取中心會把臉切掉一半或只留下半身。作法是：

- 邊長取 `min(H, W)`，即畫面裝得下的最大正方形，不放大原圖。
- 水平中心取臉框中心。
- 垂直中心取臉框中心**往下移 `SHOULDER_DROP × 邊長`**，讓肩膀進畫面。
  只對齊臉會讓正方形的上半是空背景、下半把肩膀切掉，而身分與位移兩個
  讀數都需要臉以外還有東西可動。
- 正方形超出畫面時夾回邊界內。

偵測不到臉時**拋錯不是退回中心裁切**：人像類的每一格讀數都以臉為前提，
一張沒有臉的圖混進來不會有症狀，只會讓那一列的 identity 欄變成雜訊。

輸出
────────────────────────────────────────────────────────────────────
`{out}/{類別}/{類別}_{序號}.png`（512²）＋ `{out}/provenance.json`
（來源檔、sha256、原尺寸、臉框、裁切框）。`prompts.yaml` 不由這一支產生。

用法（遠端）
    HF_HOME=/var/cache/huggingface python scripts/crop_portraits.py \\
        --spec man=127,150,177,214 woman=159,165,171,380 \\
        --src data/_portraits_src --out data/portraits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESOLUTION = 512
# 臉框中心往下移多少（以正方形邊長為單位）。0.18 讓臉落在上三分附近。
SHOULDER_DROP = 0.18


def face_centre(x01: torch.Tensor, device) -> tuple:
    """回傳最大臉框的 (cy, cx) 與臉框本身。偵測不到就拋錯。"""
    from src.metrics.identity import face_boxes

    boxes = face_boxes(x01, device=device)
    if not boxes:
        raise ValueError(
            "偵測不到人臉。人像類的 identity 讀數以臉為前提，"
            "**不退回中心裁切**——那會讓該列的讀數變成雜訊而沒有症狀。")
    b = max(boxes, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    x0, y0, x1, y1 = (float(v) for v in b)
    return (y0 + y1) / 2.0, (x0 + x1) / 2.0, (x0, y0, x1, y1)


def crop_box(h: int, w: int, cy: float, cx: float) -> tuple:
    """以 `(cy, cx)` 為中心、邊長 `min(h, w)` 的正方形，超界夾回。"""
    side = min(h, w)
    cy = cy + SHOULDER_DROP * side
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    return x0, y0, side


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", nargs="+", required=True,
                    help="類別=檔名逗號清單，例如 man=127,150,177,214")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    records = []
    for item in args.spec:
        cls, _, names = item.partition("=")
        if not names:
            raise SystemExit(f"--spec 要寫成 類別=檔名清單，收到 {item!r}")
        d = args.out / cls
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
        for i, stem in enumerate(names.split(",")):
            matches = sorted(args.src.glob(stem + ".*"))
            if not matches:
                raise SystemExit(f"{args.src} 找不到 {stem}.*")
            src = matches[0]
            arr = np.array(Image.open(src).convert("RGB"))
            h, w = arr.shape[:2]
            x = torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 255.0
            cy, cx, box = face_centre(x.to(device), device)
            x0, y0, side = crop_box(h, w, cy, cx)
            crop = arr[y0:y0 + side, x0:x0 + side]
            out_name = f"{cls}_{i:02d}.png"
            Image.fromarray(crop).resize((RESOLUTION, RESOLUTION),
                                         Image.LANCZOS).save(d / out_name)
            records.append({
                "output": f"{cls}/{out_name}",
                "source": src.name,
                "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "source_size": [w, h],
                "face_box": [round(v, 1) for v in box],
                "crop_xywh": [x0, y0, side, side],
                "output_size": RESOLUTION,
            })
            print(f"{src.name} {w}x{h} -> {cls}/{out_name} "
                  f"（臉框中心 {cx:.0f},{cy:.0f}；裁切 {x0},{y0} 邊長 {side}）")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "provenance.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(records)} 張 -> {args.out}，出處 {args.out / 'provenance.json'}")


if __name__ == "__main__":
    main()
