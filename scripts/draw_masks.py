"""互動式畫保留遮罩：滑鼠自己框，存成羽化過的灰階 PNG。

遮罩的用途是在編輯階段保留一塊區域：`sdedit(..., keep01=mask)` 讓遮罩為 1 的
地方每一步都被換回輸入圖在該時刻的帶噪 latent，於是那塊不被編輯（latent 混合，
不是 inpainting，UNet 仍是四通道 stock 權重）。

2026-08-17 使用者裁定人物的流程是：整張圖加防禦 -> 遮罩蓋住臉與身體 -> 編輯只
改背景。因此遮罩範圍是**整個人**而不是只有頭，形狀不是一個橢圓能框出來的，
先前寫死座標的 `make_headmasks.py` 因此作廢，改由使用者自己畫。

輸出：`<data>/keepmasks/<影像>.png`，灰階、已羽化。硬邊會在 latent 混合處留下
可見的接縫，所以存出去的一定是羽化過的，不是二值圖。

    python scripts/draw_masks.py --data data/set0817
    python scripts/draw_masks.py --data data/set0817 --images obama_00 musk_00

操作：

    左鍵拖曳      畫（依目前工具）
    e / r / b     切換 橢圓 / 矩形 / 筆刷
    [ ]           筆刷變小 / 變大
    - =           羽化半徑減 / 加
    u             復原上一筆
    c             清空
    s             存檔
    n / p         下一張 / 上一張（未存的筆畫會留著）
    q             離開
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

SIZE = 512          # 與 apa_baseline.RESOLUTION 一致
DEFAULT_FEATHER = 16

HELP = [
    "drag = draw      e/r/b = ellipse/rect/brush     [ ] = brush size",
    "- = = feather    u = undo   c = clear   s = save   n/p = next/prev   q = quit",
]


class Canvas:
    """一張影像的遮罩狀態。`strokes` 保留每一筆，才能逐筆復原。"""

    def __init__(self, name: str, img: np.ndarray, mask: Optional[np.ndarray]):
        self.name = name
        self.img = img
        # 讀回既有遮罩時當成一筆已完成的底稿：可以繼續加，也可以整個清掉。
        self.strokes: List[np.ndarray] = []
        if mask is not None:
            self.strokes.append((mask > 127).astype(np.uint8))
        self.feather = DEFAULT_FEATHER
        self.dirty = False

    def hard(self) -> np.ndarray:
        m = np.zeros((SIZE, SIZE), np.uint8)
        for s in self.strokes:
            m |= s
        return m

    def soft(self) -> np.ndarray:
        m = self.hard() * 255
        if self.feather > 0:
            k = self.feather * 2 + 1
            m = cv2.GaussianBlur(m, (k, k), self.feather / 2.0)
        return m


def stroke_shape(tool: str, p0: Tuple[int, int], p1: Tuple[int, int]) -> np.ndarray:
    m = np.zeros((SIZE, SIZE), np.uint8)
    x0, y0 = p0
    x1, y1 = p1
    if tool == "rect":
        cv2.rectangle(m, (min(x0, x1), min(y0, y1)),
                      (max(x0, x1), max(y0, y1)), 1, -1)
    else:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        rx, ry = max(abs(x1 - x0) // 2, 1), max(abs(y1 - y0) // 2, 1)
        cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    return m


def overlay(canvas: Canvas, tool: str, brush: int,
            preview: Optional[np.ndarray]) -> np.ndarray:
    soft = canvas.soft().astype(np.float32) / 255.0
    if preview is not None:
        soft = np.maximum(soft, preview.astype(np.float32))
    tint = np.zeros_like(canvas.img, np.float32)
    tint[..., 2] = 255.0                      # BGR：紅
    a = soft[..., None] * 0.45
    out = (canvas.img.astype(np.float32) * (1 - a) + tint * a).astype(np.uint8)

    cov = soft.mean() * 100
    bar = (f"{canvas.name}  tool={tool}  brush={brush}  feather={canvas.feather}"
           f"  strokes={len(canvas.strokes)}  cover={cov:.1f}%"
           f"{'  UNSAVED' if canvas.dirty else ''}")
    cv2.rectangle(out, (0, 0), (SIZE, 20), (0, 0, 0), -1)
    cv2.putText(out, bar, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, SIZE - 34), (SIZE, SIZE), (0, 0, 0), -1)
    for i, line in enumerate(HELP):
        cv2.putText(out, line, (6, SIZE - 20 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/set0817"))
    ap.add_argument("--images", nargs="+", default=None,
                    help="預設是資料集裡的全部影像")
    ap.add_argument("--scale", type=float, default=1.6,
                    help="視窗放大倍率，只影響顯示，存出來仍是 512")
    args = ap.parse_args()

    out_dir = args.data / "keepmasks"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in args.data.glob("*/*.png")
                   if p.parent.name not in ("keepmasks", "headmasks", "_source"))
    if args.images:
        want = set(args.images)
        paths = [p for p in paths if p.stem in want]
        missing = want - {p.stem for p in paths}
        if missing:
            raise FileNotFoundError(f"資料集裡沒有這些影像：{sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"{args.data} 底下找不到影像")

    canvases: List[Canvas] = []
    for p in paths:
        raw = cv2.imread(str(p))
        if raw is None:
            raise FileNotFoundError(f"讀不到影像：{p}")
        img = cv2.resize(raw, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        prev = out_dir / f"{p.stem}.png"
        m = cv2.imread(str(prev), cv2.IMREAD_GRAYSCALE) if prev.exists() else None
        if m is not None and m.shape != (SIZE, SIZE):
            m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        canvases.append(Canvas(p.stem, img, m))

    ui = {"idx": 0, "tool": "ellipse", "brush": 28, "down": False,
          "p0": (0, 0), "p1": (0, 0), "free": np.zeros((SIZE, SIZE), np.uint8)}
    win = "draw keep mask"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def to_img(x: int, y: int) -> Tuple[int, int]:
        return (int(np.clip(x / args.scale, 0, SIZE - 1)),
                int(np.clip(y / args.scale, 0, SIZE - 1)))

    def on_mouse(event, x, y, flags, _):
        px, py = to_img(x, y)
        brush = ui["brush"]
        if event == cv2.EVENT_LBUTTONDOWN:
            ui["down"] = True
            ui["p0"] = ui["p1"] = (px, py)
            ui["free"] = np.zeros((SIZE, SIZE), np.uint8)
            if ui["tool"] == "brush":
                cv2.circle(ui["free"], (px, py), brush, 1, -1)
        elif event == cv2.EVENT_MOUSEMOVE and ui["down"]:
            if ui["tool"] == "brush":
                cv2.line(ui["free"], ui["p1"], (px, py), 1, brush * 2)
                cv2.circle(ui["free"], (px, py), brush, 1, -1)
            ui["p1"] = (px, py)
        elif event == cv2.EVENT_LBUTTONUP and ui["down"]:
            ui["down"] = False
            c = canvases[ui["idx"]]
            s = (ui["free"] if ui["tool"] == "brush"
                 else stroke_shape(ui["tool"], ui["p0"], ui["p1"]))
            if s.any():
                c.strokes.append(s)
                c.dirty = True
            ui["free"] = np.zeros((SIZE, SIZE), np.uint8)

    cv2.setMouseCallback(win, on_mouse)
    for line in HELP:
        print(line)

    disp = int(SIZE * args.scale)
    while True:
        c = canvases[ui["idx"]]
        preview = None
        if ui["down"]:
            preview = (ui["free"] if ui["tool"] == "brush"
                       else stroke_shape(ui["tool"], ui["p0"], ui["p1"]))
        frame = overlay(c, ui["tool"], ui["brush"], preview)
        cv2.imshow(win, cv2.resize(frame, (disp, disp),
                                   interpolation=cv2.INTER_NEAREST))

        k = cv2.waitKey(16) & 0xFF
        if k == 255:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        if k == ord("q"):
            break
        elif k in (ord("e"), ord("r"), ord("b")):
            ui["tool"] = {"e": "ellipse", "r": "rect", "b": "brush"}[chr(k)]
        elif k == ord("["):
            ui["brush"] = max(2, ui["brush"] - 4)
        elif k == ord("]"):
            ui["brush"] = min(120, ui["brush"] + 4)
        elif k == ord("-"):
            c.feather = max(0, c.feather - 2)
            c.dirty = True
        elif k in (ord("="), ord("+")):
            c.feather = min(64, c.feather + 2)
            c.dirty = True
        elif k == ord("u") and c.strokes:
            c.strokes.pop()
            c.dirty = True
        elif k == ord("c"):
            c.strokes.clear()
            c.dirty = True
        elif k == ord("s"):
            if not c.strokes:
                print(f"{c.name}：沒有任何筆畫，不存空遮罩")
            else:
                p = out_dir / f"{c.name}.png"
                soft = c.soft()
                cv2.imwrite(str(p), soft)
                c.dirty = False
                print(f"{c.name} -> {p}  覆蓋率 {soft.mean() / 255 * 100:.1f}%"
                      f"  羽化 {c.feather}")
        elif k == ord("n"):
            ui["idx"] = (ui["idx"] + 1) % len(canvases)
        elif k == ord("p"):
            ui["idx"] = (ui["idx"] - 1) % len(canvases)

    cv2.destroyAllWindows()
    left = [x.name for x in canvases if x.dirty]
    if left:
        print(f"\n下列影像有未存檔的修改：{' '.join(left)}")
    print(f"遮罩目錄：{out_dir}")


if __name__ == "__main__":
    main()
