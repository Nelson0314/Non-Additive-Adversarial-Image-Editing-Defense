"""產生人物影像的頭部遮罩。

用途：`--strength 0.8` 的 SDEdit 會把整張臉換掉，未防禦的編輯連身分都不剩，
量出來的位移量就不是「編輯被推開多少」而是「人被換成另一個人多少」。遮罩讓
頭部在編輯過程中被保留（`src/models/sd.py` 的 latent 混合，不是 inpainting，
UNet 仍是四通道的 stock 權重）。

範圍是**整顆頭**，不是人臉偵測框：偵測框只從眉毛到下巴，不含頭髮與下顎。
`HEAD` 的四個數字是 (cx, cy, rx, ry)，由 512x512 影像上的網格逐張目視量得，
邊界取髮頂、下巴、兩側耳廓。2026-08-17 重量：先前的一組五張全部偏高
（y 差 11-78 px）、水平最多偏 70 px，下緣切在嘴巴上，下巴與下顎露在遮罩外。

檢核不是裝飾：YuNet 偵測到的人臉框必須完全落在頭部橢圓的外接框內。這條若不
成立就是量錯了，直接報錯而不是產出一張看起來合理的遮罩。YuNet 的模型檔不入
版控，給 --detector 指到本機的 .onnx 才會做檢核。

    python scripts/make_headmasks.py --data data/set0817
    python scripts/make_headmasks.py --data data/set0817 --detector <path>/yunet.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# (cx, cy, rx, ry)，單位是 512x512 影像上的像素。
HEAD = {
    "obama":   (250, 160,  95, 140),
    "lebron":  (225, 126,  73,  99),
    "ronaldo": (239, 186, 111, 159),
    "musk":    (249, 148,  81, 137),
    "trump":   (250, 171, 110, 149),
}

FEATHER = 16   # 高斯羽化半徑。硬邊會在 latent 混合處留下可見的接縫。
SIZE = 512


def ellipse_mask(cx: int, cy: int, rx: int, ry: int, size: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    return m.filter(ImageFilter.GaussianBlur(FEATHER))


def detect_face(det, path: Path) -> tuple[float, float, float, float]:
    import cv2
    img = cv2.imread(str(path))
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    if faces is None:
        raise ValueError(f"{path.name}：YuNet 偵測不到人臉，無法檢核")
    f = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
    x, y, w, h = (float(v) for v in f[:4])
    return x, y, x + w, y + h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/set0817"))
    ap.add_argument("--detector", type=Path, default=None,
                    help="YuNet 的 .onnx。給了才做人臉框包含性檢核。")
    args = ap.parse_args()

    out = args.data / "headmasks"
    out.mkdir(parents=True, exist_ok=True)

    det = None
    if args.detector:
        import cv2
        det = cv2.FaceDetectorYN.create(str(args.detector), "", (SIZE, SIZE), 0.6, 0.3, 5000)

    for name, (cx, cy, rx, ry) in HEAD.items():
        src = args.data / name / f"{name}_00.png"
        if not src.exists():
            raise FileNotFoundError(src)

        if det is not None:
            fx0, fy0, fx1, fy1 = detect_face(det, src)
            hx0, hy0, hx1, hy1 = cx - rx, cy - ry, cx + rx, cy + ry
            if not (hx0 <= fx0 and fx1 <= hx1 and hy0 <= fy0 and fy1 <= hy1):
                raise ValueError(
                    f"{name}：人臉框 x[{fx0:.0f},{fx1:.0f}] y[{fy0:.0f},{fy1:.0f}] "
                    f"沒有完全落在頭部框 x[{hx0},{hx1}] y[{hy0},{hy1}] 內")
            print(f"{name:9s} 人臉框 x[{fx0:.0f},{fx1:.0f}] y[{fy0:.0f},{fy1:.0f}] "
                  f"落在頭部框 x[{hx0},{hx1}] y[{hy0},{hy1}] 內")

        m = ellipse_mask(cx, cy, rx, ry, SIZE)
        m.save(out / f"{name}_00.png")
        cov = float(np.asarray(m, dtype=np.float32).mean()) / 255.0
        print(f"{name:9s} -> {out / f'{name}_00.png'}  平均權重 {cov * 100:.1f}%")


if __name__ == "__main__":
    main()
