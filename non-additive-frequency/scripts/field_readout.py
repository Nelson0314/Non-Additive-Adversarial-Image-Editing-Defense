"""防禦圖本身的讀數：身分、失真、位移場的形狀。**唯讀，不含任何指令。**

為什麼要單獨一支
────────────────────────────────────────────────────────────────────
`scripts/evaluate_defence.py` 量的是**編輯輸出**上的身分，分母是同一格的未防禦
編輯。那個比值回答「攻擊者拿不拿得到同一個人」，但回答不了「這張防禦圖本身還
是不是這個人的照片」——幾何載體會把臉的形狀改掉，身分下降有多少在防禦圖上就
已經發生、有多少是編輯放大的，兩者要分開看，否則使用者判斷不了自己還願不願意
公開這張照片。

`subject_id_defended` 取的是原圖主體框的固定座標（`anchored_identity` 的同一條
路徑），偵測器不參與，所以框沒有跟著臉移動——量到的正是「這個位置上的人還是
不是同一個人」。`subject_id_defended_detected` 另外以偵測器重找臉再量，兩個都報。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['variant', 'image', 'subject_id_defended',
           'subject_id_defended_detected', 'face_box_iou_defended',
           'faces_orig', 'faces_defended',
           'psnr', 'ssim', 'lpips', 'dists', 'deltaE00', 'linf',
           'acutance_ratio', 'niqe_orig', 'niqe_defended']


def main():
    import torch

    from src.defense.assets import load_image
    from src.defense.ncf_library import sha256
    from src.metrics.identity import (embed, embed_box, face_boxes,
                                      similarity)
    from src.metrics.suite import MetricSuite

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path,
                    default=ROOT / 'data' / 'scaleup_manifest.json')
    ap.add_argument('--root', type=Path, required=True,
                    help='含 <variant>/<image>__immunised.png 的目錄')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    entries = {r['id']: r for r in manifest['images']}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    suite = MetricSuite(device=device)

    def iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return 0.0 if union <= 0 else inter / union

    rows = []
    for vdir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for png in sorted(vdir.glob('*__immunised.png')):
            image = png.name[:-len('__immunised.png')]
            entry = entries[image]
            src = ROOT / entry['path']
            if sha256(src) != entry['sha256']:
                raise ValueError(f'{image} 的輸入雜湊不符')
            x = load_image(src, device)
            y = load_image(png, device)
            boxes_x = face_boxes(x, device)
            boxes_y = face_boxes(y, device)
            if not boxes_x:
                raise ValueError(f'{image} 原圖偵測不到臉')
            anchor = max(boxes_x, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
            e_anchor_x = embed_box(x, anchor, device)
            e_anchor_y = embed_box(y, anchor, device)
            pair = suite.pairwise(x, y)
            row = {
                'variant': vdir.name, 'image': image,
                'subject_id_defended': round(
                    similarity(e_anchor_x, e_anchor_y), 5),
                'subject_id_defended_detected': round(
                    similarity(embed(x, device), embed(y, device)), 5)
                if boxes_y else '',
                'face_box_iou_defended': round(iou(anchor, max(
                    boxes_y, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))), 5)
                if boxes_y else 0.0,
                'faces_orig': len(boxes_x), 'faces_defended': len(boxes_y),
                'niqe_orig': round(suite.niqe(x), 5),
                'niqe_defended': round(suite.niqe(y), 5),
                **{k: round(v, 5) for k, v in pair.items() if k in COLUMNS},
            }
            rows.append(row)
            print(f'{vdir.name:12s} {image:28s} '
                  f'id {row["subject_id_defended"]:.4f} '
                  f'(偵測 {row["subject_id_defended_detected"]}) '
                  f'PSNR {row["psnr"]:.2f} LPIPS {row["lpips"]:.4f} '
                  f'ΔE00 {row["deltaE00"]:.2f} '
                  f'niqe {row["niqe_defended"]:.2f}/{row["niqe_orig"]:.2f}',
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'寫出 {args.out}（{len(rows)} 列）', flush=True)


if __name__ == '__main__':
    main()
