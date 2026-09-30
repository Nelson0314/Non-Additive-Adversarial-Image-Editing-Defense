"""從候選池挑主體鮮明的人像。**唯讀，不改任何資料。**

挑選的目的
────────────────────────────────────────────────────────────────────
判準是「攻擊者拿不到可用、認得出是同一個人、指令完成的照片」，所以每一張
候選都要先滿足兩件事，否則那一格的讀數是雜訊而不是防禦：

1. **有一張夠大、夠清楚的臉。** 身分是判準的主軸，臉太小則位移場與顏色場都
   動不到它，而讀數的動態範圍會塌掉。
2. **是照片，不是插畫。** 先前的 `task_env_weather_195273` 是卡通武士，
   它的身分嵌入與自然度判斷都與照片不同族，混在同一批裡讀不出東西。

自動量什麼，不自動量什麼
────────────────────────────────────────────────────────────────────
這一支只出**排序與旗標**，最後選哪幾張由人看圖決定——與自然度同一個理由：
「像不像一張真實照片」沒有可信的自動門檻。

| 欄位 | 意義 |
|---|---|
| `faces` | MTCNN 偵測到的臉數。主體鮮明要的是 1，多人照的身分讀數會指到別人 |
| `face_frac` | 最大臉框面積佔畫面的比例 |
| `face_px` | 最大臉框的邊長（幾何載體的網格間距要與它比） |
| `sharpness` | 臉框內的 Laplacian 變異數，越大越清楚 |
| `colour_count` | 量化到 4 bit 之後的相異顏色數。**插畫的平坦色塊讓它明顯偏低** |
| `edge_ratio` | 強梯度像素的比例。插畫的線稿讓它偏高 |
| `illustration_flag` | 上面兩項同時越界時掛起來，**是提示不是判準** |

`colour_count` 與 `edge_ratio` 的門檻取自本專案的候選池分布，不是文獻值；
它們只用來把疑似插畫排到後面，不用來自動剔除。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['image', 'faces', 'face_frac', 'face_px', 'sharpness',
           'colour_count', 'edge_ratio', 'illustration_flag', 'path']


def main():
    import numpy as np
    import torch

    from src.defense.assets import load_image
    from src.metrics.identity import face_boxes

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pool', type=Path, default=ROOT / 'data' / 'omniedit150')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--colour-floor', type=int, default=1800,
                    help='相異顏色數低於此值算疑似插畫')
    ap.add_argument('--edge-ceiling', type=float, default=0.13,
                    help='強梯度像素比例高於此值算疑似插畫')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for d in sorted(p for p in args.pool.iterdir() if p.is_dir()):
        png = d / f'{d.name}.png'
        if not png.exists():
            continue
        x = load_image(png, device)
        boxes = face_boxes(x, device)
        a = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        h, w = a.shape[:2]

        q = (a >> 4).astype(np.int32)
        colour_count = int(len(np.unique(q[..., 0] * 256 + q[..., 1] * 16
                                         + q[..., 2])))
        g = a.mean(axis=2)
        gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
        gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
        edge_ratio = float(((gx + gy) > 40).mean())

        if boxes:
            b = max(boxes, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            x0, y0, x1, y1 = (int(v) for v in b)
            side = float(max(x1 - x0, y1 - y0))
            frac = float((x1 - x0) * (y1 - y0)) / float(w * h)
            crop = g[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            if crop.size >= 16:
                lap = (crop[2:, 1:-1] + crop[:-2, 1:-1] + crop[1:-1, 2:]
                       + crop[1:-1, :-2] - 4 * crop[1:-1, 1:-1])
                sharp = float(lap.var())
            else:
                sharp = 0.0
        else:
            side, frac, sharp = 0.0, 0.0, 0.0

        rows.append({
            'image': d.name, 'faces': len(boxes),
            'face_frac': round(frac, 5), 'face_px': round(side, 1),
            'sharpness': round(sharp, 2), 'colour_count': colour_count,
            'edge_ratio': round(edge_ratio, 5),
            'illustration_flag': int(colour_count < args.colour_floor
                                     and edge_ratio > args.edge_ceiling),
            'path': str(png.relative_to(ROOT)).replace('\\', '/'),
        })

    rows.sort(key=lambda r: (r['illustration_flag'], r['faces'] != 1,
                             -r['face_frac']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    keep = [r for r in rows if r['faces'] == 1 and not r['illustration_flag']]
    print(f'{len(rows)} 張候選，單臉且未標插畫的有 {len(keep)} 張')
    print(f'{"image":<34}{"臉":>3}{"面積比":>8}{"邊長":>7}'
          f'{"銳利":>9}{"顏色數":>8}{"邊緣比":>8}')
    for r in keep[:24]:
        print(f'{r["image"]:<34}{r["faces"]:>3}{r["face_frac"]:>8.3f}'
              f'{r["face_px"]:>7.0f}{r["sharpness"]:>9.1f}'
              f'{r["colour_count"]:>8d}{r["edge_ratio"]:>8.3f}')
    print(f'寫出 {args.out}（{len(rows)} 列）')


if __name__ == '__main__':
    main()
