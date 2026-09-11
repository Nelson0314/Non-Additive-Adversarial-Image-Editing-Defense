"""逐張影像的特徵：用來解釋「為什麼同一個設定在不同照片上差這麼多」。

第一批量到的最大效應不是臂之間的差別，而是**逐圖的差別**：同一個臂、同一個
設定下，防禦後的主體身分從 0.135 到 0.853，而未防禦的對照全部落在
0.788–0.923。五張圖分不出那是訊號還是雜訊，也分不出是什麼決定的。

這支把可能的解釋變數逐張算出來，全部在 CPU 上、不需要攻擊模型：

    clothes_area        ATR 衣物遮罩的面積比
    face_area           受保護主體（臉與頭髮）的面積比
    n_faces             偵測到的臉數（多人畫面是另一個 regime）
    subject_box_frac    主體框佔畫面的比例
    chroma_std          Lab 色度的標準差——色彩豐富的影像可轉的角度不同
    chroma_mean_ab      色度均值的長度，離無彩多遠
    luma_std            亮度標準差
    highfreq_energy     高通殘差能量，紋理多寡
    bg_chroma_std       主體與衣物之外那塊的色度標準差
    region_contrast     碰撞目標關心的區域與其環帶的 Lab 均值距離

**本支不做判定。** 它只產生解釋變數，回歸與相關性由使用者在拿到效果讀數之後
自己做，或由另一支腳本做。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['image', 'width', 'height', 'clothes_area', 'face_area', 'n_faces',
           'subject_box_frac', 'chroma_std', 'chroma_mean_ab', 'luma_std',
           'highfreq_energy', 'bg_chroma_std', 'region_contrast_clothes',
           'region_contrast_head_ring', 'region_contrast_outside_subject']


def load_image(path):
    import numpy as np
    import torch
    from PIL import Image
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def features(x, device=None):
    import torch

    from src.defense.carrier_mask import face_subject_mask
    from src.defense.lowfreq_color import highfreq_report
    from src.defense.ncf_param import rgb_to_lab
    from src.defense.ncf_runner import ncf_support
    from src.metrics.identity import face_boxes
    from scripts.color_ceiling import collision_region

    lab = rgb_to_lab(x)
    ab = lab[:, 1:]
    clothes = ncf_support(x, 'clothes')
    face = face_subject_mask(x, device=device)
    boxes = face_boxes(x, device)
    h, w = x.shape[-2:]
    if boxes:
        b = max(boxes, key=lambda q: (q[2]-q[0])*(q[3]-q[1]))
        box_frac = float((b[2]-b[0]) * (b[3]-b[1]) / (w * h))
    else:
        box_frac = ''
    inside = torch.maximum(clothes, (face > .5).to(clothes.dtype))
    bg = (1. - inside).clamp(0., 1.)
    out = {
        'width': int(w), 'height': int(h),
        'clothes_area': round(float(clothes.mean()), 5),
        'face_area': round(float((face > .5).to(x.dtype).mean()), 5),
        'n_faces': len(boxes),
        'subject_box_frac': '' if box_frac == '' else round(box_frac, 5),
        'chroma_std': round(float(ab.reshape(2, -1).std(dim=1).norm()), 4),
        'chroma_mean_ab': round(float(ab.reshape(2, -1).mean(dim=1).norm()), 4),
        'luma_std': round(float(lab[:, 0].std()), 4),
        # `highfreq_report(x, x)` 的比值恆為 1，所以這裡取殘差的絕對能量。
        'highfreq_energy': round(float(
            (x - __import__('src.purify.ops', fromlist=['gaussian_blur'])
             .gaussian_blur(x, 2.0)).pow(2).mean()), 8),
        'bg_chroma_std': ('' if float(bg.sum()) < 2 else round(float(
            (ab * bg).reshape(2, -1).std(dim=1).norm()), 4)),
    }
    # 碰撞目標關心的三塊區域，各自與其環帶的 Lab 均值距離。
    from src.defense.collision_loss import ring_of
    for name in ('clothes', 'head_ring', 'outside_subject'):
        try:
            r = collision_region(x, name, clothes, device=device)
            ring = ring_of(r, 16)
            if float(r.sum()) < 2 or float(ring.sum()) < 2:
                out[f'region_contrast_{name}'] = ''
                continue
            mr = (lab * r).sum(dim=(0, 2, 3)) / r.sum()
            me = (lab * ring).sum(dim=(0, 2, 3)) / ring.sum()
            out[f'region_contrast_{name}'] = round(float((mr - me).norm()), 4)
        except Exception as exc:                       # noqa: BLE001
            # 區域取不出來是一個讀數（例如衣物遮罩為空），不是錯誤；照實記空值
            # 並把理由印出來，不吞掉。
            print(f'    region {name} 取不到：{exc}', flush=True)
            out[f'region_contrast_{name}'] = ''
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    from src.defense.ncf_library import sha256

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in manifest['images']:
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f"{entry['id']} 的輸入雜湊不符")
        x = load_image(src)
        row = {'image': entry['id'], **features(x)}
        rows.append(row)
        print(f"  {entry['id'][:34]:36s} 衣物 {row['clothes_area']} "
              f"臉 {row['face_area']} 臉數 {row['n_faces']} "
              f"色度σ {row['chroma_std']}", flush=True)
    write_csv(args.out / 'image_features.csv', rows)
    print(f'-> {args.out}  共 {len(rows)} 列', flush=True)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'{path} 沒有任何列可寫；不要留下空檔假裝跑過')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    unknown = [f for f in fields if f not in COLUMNS]
    if unknown:
        raise ValueError(f'這些欄位不在 COLUMNS 的合約裡：{unknown}')
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
