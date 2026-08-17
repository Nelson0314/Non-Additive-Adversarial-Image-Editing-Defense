"""互動式畫保留遮罩：沿著邊界描一圈，圍起來的區域就是遮罩。

遮罩的用途是在編輯階段保留一塊區域：`sdedit(..., keep01=mask)` 讓遮罩為 1 的
地方每一步都被換回輸入圖在該時刻的帶噪 latent，於是那塊不被編輯（latent 混合，
不是 inpainting，UNet 仍是四通道 stock 權重）。

2026-08-17 使用者裁定人物的流程是：整張圖加防禦 -> 遮罩蓋住臉與身體 -> 編輯只
改背景。因此遮罩範圍是**整個人**，形狀不是橢圓或矩形框得出來的，寫死座標的
`make_headmasks.py` 因此作廢。同日使用者再裁定輸入方式為**描邊界**而不是塗
色塊：左鍵沿著人的輪廓拖一圈，放開時自動把頭尾接起來並填滿。

輸出：`<data>/keepmasks/<影像>.png`，灰階、已羽化。硬邊會在 latent 混合處留下
可見的接縫，所以存出去的一定是羽化過的，不是二值圖。

    python scripts/draw_masks.py --data data/set0817
    python scripts/draw_masks.py --data data/set0817 --images obama_00 musk_00

操作（鍵盤事件由影像視窗接收，按鍵前先點一下視窗；中文輸入法會攔截按鍵）：

    左鍵拖曳      沿邊界描一圈 -> 放開時填滿，加進遮罩
    右鍵拖曳      同樣描一圈 -> 放開時從遮罩挖掉
    u             復原上一圈
    c             清空
    - =           羽化半徑減 / 加
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
MIN_POINTS = 3      # 少於三點圍不出面積

HELP = [
    "L-drag = trace outline (fill)   R-drag = trace outline (erase)",
    "u undo   c clear   - = feather   s save   n/p next/prev   q quit",
]


class Canvas:
    """一張影像的遮罩狀態。

    `strokes` 是 (區域, 是否為加) 的序列，依序套用；保留每一圈才能逐圈復原，
    也才能讓「挖掉」與「加上」在同一個復原堆疊裡。
    """

    def __init__(self, name: str, img: np.ndarray, mask: Optional[np.ndarray]):
        self.name = name
        self.img = img
        self.strokes: List[Tuple[np.ndarray, bool]] = []
        if mask is not None:
            # 讀回既有遮罩時當成一圈已完成的底稿：可以繼續加、挖，也可以清掉。
            self.strokes.append(((mask > 127).astype(np.uint8), True))
        self.feather = DEFAULT_FEATHER
        self.dirty = False

    def hard(self) -> np.ndarray:
        m = np.zeros((SIZE, SIZE), np.uint8)
        for s, add in self.strokes:
            if add:
                m |= s
            else:
                m &= 1 - s
        return m

    def soft(self) -> np.ndarray:
        m = self.hard() * 255
        if self.feather > 0:
            k = self.feather * 2 + 1
            m = cv2.GaussianBlur(m, (k, k), self.feather / 2.0)
        return m


def fill_outline(points: List[Tuple[int, int]]) -> np.ndarray:
    """把描出來的一圈頭尾接起來並填滿。

    `fillPoly` 本來就把點列當成封閉多邊形，所以不必自己補最後一段線；描的時候
    起點與終點沒對上也沒關係，接縫由它直接連。
    """
    m = np.zeros((SIZE, SIZE), np.uint8)
    if len(points) >= MIN_POINTS:
        cv2.fillPoly(m, [np.asarray(points, np.int32)], 1)
    return m


def overlay(canvas: Canvas, points: List[Tuple[int, int]], adding: bool) -> np.ndarray:
    soft = canvas.soft().astype(np.float32) / 255.0
    tint = np.zeros_like(canvas.img, np.float32)
    tint[..., 2] = 255.0                      # BGR：紅
    a = soft[..., None] * 0.45
    out = (canvas.img.astype(np.float32) * (1 - a) + tint * a).astype(np.uint8)

    if len(points) >= 2:
        # 描的過程中畫的是線而不是填滿：填滿要等放開，否則整張圖一直在閃。
        colr = (80, 255, 80) if adding else (80, 80, 255)
        cv2.polylines(out, [np.asarray(points, np.int32)], False, colr, 2,
                      cv2.LINE_AA)

    cov = soft.mean() * 100
    bar = (f"{canvas.name}  feather={canvas.feather}  loops={len(canvas.strokes)}"
           f"  cover={cov:.1f}%{'  UNSAVED' if canvas.dirty else ''}")
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

    ui = {"idx": 0, "down": False, "adding": True,
          "points": []}          # type: dict
    win = "draw keep mask"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def to_img(x: int, y: int) -> Tuple[int, int]:
        return (int(np.clip(x / args.scale, 0, SIZE - 1)),
                int(np.clip(y / args.scale, 0, SIZE - 1)))

    def on_mouse(event, x, y, flags, _):
        px, py = to_img(x, y)
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            ui["down"] = True
            ui["adding"] = event == cv2.EVENT_LBUTTONDOWN
            ui["points"] = [(px, py)]
        elif event == cv2.EVENT_MOUSEMOVE and ui["down"]:
            if ui["points"][-1] != (px, py):
                ui["points"].append((px, py))
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and ui["down"]:
            ui["down"] = False
            region = fill_outline(ui["points"])
            if region.any():
                c = canvases[ui["idx"]]
                c.strokes.append((region, ui["adding"]))
                c.dirty = True
            elif len(ui["points"]) >= 2:
                print(f"這一圈只有 {len(ui['points'])} 個點，圍不出面積，忽略")
            ui["points"] = []

    cv2.setMouseCallback(win, on_mouse)
    for line in HELP:
        print(line)

    disp = int(SIZE * args.scale)
    while True:
        c = canvases[ui["idx"]]
        frame = overlay(c, ui["points"] if ui["down"] else [], ui["adding"])
        cv2.imshow(win, cv2.resize(frame, (disp, disp),
                                   interpolation=cv2.INTER_NEAREST))

        k = cv2.waitKey(16) & 0xFF
        if k == 255:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        if k == ord("q"):
            break
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
