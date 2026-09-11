"""逐圖的差異由什麼決定：把效果讀數對影像特徵做相關。

第一批量到的最大效應不是臂之間的差別，而是**逐圖的差別**：同一個臂、同一個
設定下防禦後的主體身分從 0.135 到 0.853，而未防禦的對照全部落在 0.788–0.923。
五張圖分不出那是訊號還是雜訊；這支在放大後的批次上回答它。

做什麼
────────────────────────────────────────────────────────────────────
把 `scripts/image_features.py` 的逐圖特徵接上 `runs/color_scaleup/` 的效果讀數
（逐圖先取中位數，避免同一張圖的多列互相加權），再逐對算 Spearman 等級相關。

**用等級相關不用皮爾森。** 樣本只有 20 張，而且像「臉數」這種變數是離散的、
「色度標準差」的分布明顯偏斜；等級相關不假設線性也不假設常態。

**不設判準、不挑變數。** 所有變數對所有讀數的相關係數照報，附樣本數與
置換檢定的 p 值。n = 20 時單一係數的不確定性很大，而且做了幾十次比較，
所以 p 值旁邊一併報 Benjamini–Hochberg 校正後的值——**校正與否都不構成
「有沒有效」的判準**，那是使用者的事。
"""
import argparse
import csv
import glob
import json
import random
import statistics as st
from pathlib import Path

COLUMNS = ['readout', 'feature', 'n', 'spearman', 'p_permutation', 'p_bh',
           'readout_median', 'readout_min', 'readout_max']

READOUTS = ('subject_id_def', 'subject_id_drop', 'edit_lpips', 'clip_s_drop',
            'edit_siglip_drop', 'support_deltaE00', 'final_hf_rgb_total',
            'gate_subject_id', 'gate_clip_s')

FEATURES = ('clothes_area', 'face_area', 'n_faces', 'subject_box_frac',
            'chroma_std', 'chroma_mean_ab', 'luma_std', 'highfreq_energy',
            'bg_chroma_std', 'region_contrast_clothes',
            'region_contrast_head_ring', 'region_contrast_outside_subject')


def _rank(v):
    """平均等級，平手取平均——等級相關的標準處理。"""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** .5
    db = sum((y - mb) ** 2 for y in rb) ** .5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def permutation_p(a, b, rho, draws=20000, seed=0):
    """置換檢定：n = 20 時等級相關的解析分布不可靠，直接重抽。"""
    if rho is None:
        return None
    rng = random.Random(seed)
    c = list(b)
    hits = 0
    for _ in range(draws):
        rng.shuffle(c)
        r = spearman(a, c)
        if r is not None and abs(r) >= abs(rho) - 1e-12:
            hits += 1
    return (hits + 1) / (draws + 1)


def benjamini_hochberg(ps):
    """BH 校正。回傳與輸入同序的校正後 p。"""
    idx = [i for i, p in enumerate(ps) if p is not None]
    m = len(idx)
    out = list(ps)
    if not m:
        return out
    order = sorted(idx, key=lambda i: ps[i])
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        val = min(prev, ps[i] * m / k)
        out[i] = val
        prev = val
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', type=Path, required=True,
                    help='image_features.csv')
    ap.add_argument('--runs', type=Path, required=True,
                    help='含 color_ceiling*.csv 的目錄')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--arm', default='',
                    help='只看某一個臂；空字串表示合併所有防禦臂')
    ap.add_argument('--target', default='',
                    help='只看某一個 ΔE00 目標')
    ap.add_argument('--draws', type=int, default=20000)
    args = ap.parse_args()

    feats = {r['image']: r for r in csv.DictReader(
        args.features.open(encoding='utf-8'))}
    rows = []
    for p in sorted(glob.glob(str(args.runs / '**' / 'color_ceiling*.csv'),
                              recursive=True)):
        rows += list(csv.DictReader(open(p, encoding='utf-8')))
    if not rows:
        raise SystemExit(f'{args.runs} 底下找不到任何 color_ceiling*.csv')
    keep = [r for r in rows if r['arm'] != 'undefended']
    if args.arm:
        keep = [r for r in keep if r['arm'] == args.arm]
    if args.target:
        keep = [r for r in keep if r['delta_e_target'] == args.target]
    if not keep:
        raise SystemExit('篩選之後沒有列了；檢查 --arm 與 --target')

    images = sorted({r['image'] for r in keep} & set(feats))
    print(f'影像 {len(images)} 張，列 {len(keep)}，臂 '
          f'{sorted({r["arm"] for r in keep})}', flush=True)
    if len(images) < 8:
        print(f'  **樣本只有 {len(images)} 張**，等級相關在這個樣本數上'
              f'不可解讀；照算照報，解讀由使用者決定。', flush=True)

    out, pending = [], []
    for readout in READOUTS:
        per_image = {}
        for img in images:
            v = [float(r[readout]) for r in keep
                 if r['image'] == img and r.get(readout) not in (None, '', 'nan')]
            if v:
                per_image[img] = st.median(v)
        if len(per_image) < 4:
            continue
        common = [i for i in images if i in per_image]
        y = [per_image[i] for i in common]
        for feature in FEATURES:
            xs, ys = [], []
            for i, val in zip(common, y):
                f = feats[i].get(feature)
                if f in (None, '', 'nan'):
                    continue
                xs.append(float(f))
                ys.append(val)
            if len(xs) < 4:
                continue
            rho = spearman(xs, ys)
            pending.append({'readout': readout, 'feature': feature,
                            'n': len(xs), 'spearman': None if rho is None
                            else round(rho, 4),
                            'p_permutation': permutation_p(xs, ys, rho,
                                                           draws=args.draws),
                            'readout_median': round(st.median(ys), 5),
                            'readout_min': round(min(ys), 5),
                            'readout_max': round(max(ys), 5)})
    ps = benjamini_hochberg([r['p_permutation'] for r in pending])
    for r, p in zip(pending, ps):
        r['p_permutation'] = (None if r['p_permutation'] is None
                              else round(r['p_permutation'], 5))
        r['p_bh'] = None if p is None else round(p, 5)
        out.append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'per_image_analysis.csv', out)
    strong = sorted([r for r in out if r['spearman'] is not None],
                    key=lambda r: -abs(r['spearman']))[:15]
    print('\n等級相關最大的十五對（照報，不作判準）：')
    print(f"{'讀數':22s}{'特徵':28s}{'rho':>8s}{'p':>9s}{'p_BH':>9s}{'n':>4s}")
    for r in strong:
        print(f"{r['readout']:22s}{r['feature']:28s}{r['spearman']:>8.3f}"
              f"{r['p_permutation']:>9.4f}{r['p_bh']:>9.4f}{r['n']:>4d}")
    print(f'\n-> {args.out}  共 {len(out)} 對', flush=True)


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
