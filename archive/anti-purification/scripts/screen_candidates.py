"""量候選影像池的每一張，輸出排序、旗標與看圖版面。**唯讀，不改資料池。**

這一支只出**數值與旗標**，最後選哪幾張由人看圖決定——與自然度同一個理由：
「這張照片主體夠不夠鮮明」沒有可信的自動門檻。數值的用途是把明顯不合用的
排到後面，不是代替看圖。

裁切
────────────────────────────────────────────────────────────────────
以**主體遮罩的質心**為中心取最大正方形再縮到 512，而不是取畫面中心。
原因是攝影構圖常把主體放在三分線上，取中心會把主體切掉一半，而那會讓
「主體鮮明」這件事在資料上就不成立。質心落在邊緣時正方形會被夾回畫面內。

量什麼
────────────────────────────────────────────────────────────────────
| 欄位 | 意義 | 為什麼要它 |
|---|---|---|
| `subject_frac` | CLIPSeg 主體佔畫面比例 | 太大則重繪區只剩一圈細邊，「保留主體、改動其餘」這個攻擊沒有動作空間；太小則身分與位移讀數動不到主體 |
| `n_parts` | 面積 ≥ 2% 的連通塊數 | 大於 1 表示畫面裡不只一個主體（或分割溢到背景），那一格的讀數會指到別的東西 |
| `largest_part_frac` | 最大連通塊佔主體的比例 | 接近 1 表示主體是完整一塊 |
| `centre_offset` | 質心離畫面中心的距離，以半邊長為單位 | 裁切後仍偏離表示主體被畫面邊界夾住 |
| `sharpness` | 灰階 Laplacian 變異數 | 對焦與放大痕跡 |
| `colour_count` | 量化到 4 bit 後的相異顏色數 | 插畫與過度後製的平坦色塊讓它偏低 |
| `edge_ratio` | 強梯度像素比例 | 線稿與銳化過頭讓它偏高 |
| `faces` | MTCNN 偵測到的臉數 | 人像類要恰好 1；多人照的身分讀數會指到別人 |

`shortlist` 是上面幾欄同時落在區間內的旗標，**是提示不是判準**。
區間見 `BANDS`，取自本專案對前一批資料的實測，不是文獻值。

用法
    python scripts/screen_candidates.py --pool data/_pool --out runs/candidate_screen
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.defense.subject_mask import _probability  # noqa: E402

RESOLUTION = 512
THRESHOLD = 0.30
PART_MIN = 0.02                      # 連通塊要算一個「主體」的最小面積
COLOUR_FLOOR = 1800
EDGE_CEILING = 0.13

# 入選區間。`subject_frac` 的上界對應**重繪區至少要佔多少**：0.60 即重繪區
# 不低於 40%。取 0.60 而不是 0.45，是因為 0.45 在人像上卡掉太多真實照片
# （75 張的人像池裡 21 張只因這一項落選，其中 14 張是比例偏高），而 40% 的
# 重繪區已經遠離前一批資料集最差的那幾張（`dog_00` 的重繪區只有 22%）。
BANDS = {
    "subject_frac": (0.18, 0.60),
    "largest_part_frac": (0.85, 1.01),
    "centre_offset": (0.0, 0.35),
    "sharpness": (120.0, 1e9),
}
COLUMNS = ["image", "class", "subject_frac", "n_parts", "largest_part_frac",
           "centre_offset", "sharpness", "colour_count", "edge_ratio",
           "faces", "shortlist", "reject_reason", "path"]


def square_crop(arr: np.ndarray, cy: float, cx: float) -> np.ndarray:
    """以 `(cy, cx)` 為中心的最大正方形裁切，超界時夾回畫面內。"""
    h, w = arr.shape[:2]
    side = min(h, w)
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    return arr[y0:y0 + side, x0:x0 + side]


def parts(mask: np.ndarray):
    """回傳 (連通塊數, 最大塊佔主體比例)。面積小於 `PART_MIN` 的不算。"""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, 0.0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    total = float(mask.sum())
    keep = [s for s in sizes if s / mask.size >= PART_MIN]
    return len(keep), (float(max(sizes)) / total if total else 0.0)


def texture(rgb: np.ndarray):
    """`sharpness` / `colour_count` / `edge_ratio` 三欄。

    `colour_count` 與 `edge_ratio` **逐字沿用 `scripts/select_subjects.py`**
    的定義，因為 `COLOUR_FLOOR` 與 `EDGE_CEILING` 兩個門檻是那一支在本專案的
    候選池上量出來的。換一個定義（例如把一階差分換成 Sobel）會讓同樣的門檻
    落在完全不同的分位上，而症狀是整批被判成插畫。

    `sharpness` 與那一支不同：它量的是**臉框內**的 Laplacian 變異數，只對人像
    有定義；這裡六類都要，故量整張。兩者的數值不可互相比較。
    """
    import cv2

    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    q = (rgb.astype(np.int32) >> 4)
    colours = int(np.unique(q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]).size)
    gm = rgb.astype(np.float64).mean(axis=2)
    gx = np.abs(np.diff(gm, axis=1, prepend=gm[:, :1]))
    gy = np.abs(np.diff(gm, axis=0, prepend=gm[:1, :]))
    edge = float(((gx + gy) > 40).mean())
    return sharp, colours, edge


def _erode(m: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return ndimage.binary_erosion(m, iterations=2)


def _grid(cells, cols: int) -> torch.Tensor:
    rows = []
    for i in range(0, len(cells), cols):
        chunk = list(cells[i:i + cols])
        while len(chunk) < cols:
            chunk.append(torch.ones_like(cells[0]))
        rows.append(torch.cat(chunk, dim=-1))
    return torch.cat(rows, dim=-2)


def _save(x: torch.Tensor, path: Path) -> None:
    import torchvision.utils as vutils
    vutils.save_image(x.clamp(0, 1), path)


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=Path("data/_pool"))
    ap.add_argument("--out", type=Path, default=Path("runs/candidate_screen"))
    ap.add_argument("--classes", nargs="+", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sheet-side", type=int, default=256)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "crops").mkdir(exist_ok=True)
    device = torch.device(args.device)
    classes = args.classes or sorted(
        d.name for d in args.pool.iterdir() if d.is_dir())

    rows, sheets = [], {}
    for cls in classes:
        for src in sorted((args.pool / cls).glob("*.jpg")):
            arr = np.array(Image.open(src).convert("RGB"))
            # 質心這一趟只要位置，不要邊界，所以先縮到長邊 512 再跑 CLIPSeg——
            # 原尺寸（長邊 1280）在 CPU 上慢六倍，而質心的差異落在一個像素內。
            h0, w0 = arr.shape[:2]
            scale = 512 / max(h0, w0)
            small = np.array(Image.fromarray(arr).resize(
                (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))),
                Image.LANCZOS))
            x = torch.from_numpy(small).permute(2, 0, 1)[None].float() / 255.0
            prob = _probability(x.to(device), cls, device)[0, 0].cpu().numpy()
            m0 = prob >= THRESHOLD
            if m0.sum() == 0:
                cy, cx = h0 / 2, w0 / 2
            else:
                ys, xs = np.nonzero(m0)
                cy, cx = float(ys.mean()) / scale, float(xs.mean()) / scale

            crop = square_crop(arr, cy, cx)
            pil = Image.fromarray(crop).resize((RESOLUTION, RESOLUTION),
                                               Image.LANCZOS)
            cr = np.array(pil)
            xc = torch.from_numpy(cr).permute(2, 0, 1)[None].float() / 255.0
            pc = _probability(xc.to(device), cls, device)[0, 0].cpu().numpy()
            mask = pc >= THRESHOLD

            frac = float(mask.mean())
            n_parts, largest = parts(mask)
            if mask.sum():
                ys, xs = np.nonzero(mask)
                off = float(np.hypot(ys.mean() - RESOLUTION / 2,
                                     xs.mean() - RESOLUTION / 2)
                            / (RESOLUTION / 2))
            else:
                off = 1.0
            sharp, colours, edge = texture(cr)

            faces = ""
            if cls in ("man", "woman"):
                from src.metrics.identity import face_boxes
                faces = len(face_boxes(xc.to(device), device=device))

            reasons = []
            for k, v in (("subject_frac", frac),
                         ("largest_part_frac", largest),
                         ("centre_offset", off), ("sharpness", sharp)):
                lo, hi = BANDS[k]
                if not (lo <= v <= hi):
                    reasons.append(f"{k}={v:.3f}")
            if n_parts != 1:
                reasons.append(f"n_parts={n_parts}")
            # `colour_count` / `edge_ratio` **逐列記錄但不當入選門檻**。
            # `select_subjects.py` 的門檻是在人像候選池上量的；本池含大量自然
            # 場景（密集枝葉、羽毛），同樣的門檻把整批真實照片判成插畫——實測
            # 39 張「只因這一項落選」的影像裡，只有 4 張不是照片。
            # 而那 4 張是石雕、石浮雕與油畫：它們**本身就是真實照片**，
            # 拍的是非生物，靠紋理統計分不出來，只能看圖。
            if faces != "" and faces != 1:
                reasons.append(f"faces={faces}")

            name = f"{cls}_{src.stem}"
            pil.save(args.out / "crops" / f"{name}.jpg", quality=92)
            rows.append({
                "image": name, "class": cls, "subject_frac": round(frac, 4),
                "n_parts": n_parts, "largest_part_frac": round(largest, 4),
                "centre_offset": round(off, 4), "sharpness": round(sharp, 1),
                "colour_count": colours, "edge_ratio": round(edge, 4),
                "faces": faces, "shortlist": int(not reasons),
                "reject_reason": ";".join(reasons),
                "path": str(src).replace("\\", "/"),
            })
            if not reasons:
                sheets.setdefault(cls, []).append((name, cr, mask))
            print(f"{name}: frac {frac:.3f} parts {n_parts} "
                  f"sharp {sharp:.0f} " + ("OK" if not reasons else str(reasons)))

    with open(args.out / "screen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    side = args.sheet_side
    for cls, items in sheets.items():
        cells = []
        for name, cr, mask in items:
            tinted = cr.astype(np.float32).copy()
            outline = mask ^ _erode(mask)
            tinted[outline] = np.array([255.0, 40.0, 40.0])
            t = torch.from_numpy(tinted).permute(2, 0, 1)[None] / 255.0
            cells.append(F.interpolate(t, size=(side, side), mode="bilinear",
                                       align_corners=False))
        _save(_grid(cells, cols=4), args.out / f"shortlist_{cls}.png")
        (args.out / f"shortlist_{cls}.txt").write_text(
            "\n".join(n for n, _, _ in items), encoding="utf-8")

    n_ok = sum(r["shortlist"] for r in rows)
    print(f"\n{len(rows)} 張，入選旗標 {n_ok} 張。"
          f"表 {args.out / 'screen.csv'}，看圖版面 {args.out}/shortlist_*.png")


if __name__ == "__main__":
    main()
