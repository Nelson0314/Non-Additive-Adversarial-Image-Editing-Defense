"""消融 ISR 分母子集合上的防禦圖美學讀數（LAION、MUSIQ-AVA），條件 Full（ImmuLUT）、B2、B3。

輸入
    --judgments   docs/isr_pilot/ablation/judgments.csv（逐格評審標籤：sheet、row、cell、purification、accessory、mismatch、degraded）
    --key         消融 ISR 拼圖的解盲表（sheet、row、condition；sheet 含 .jpg 副檔名）
    --aesthetic-dir  含 aesthetic_full.csv、aesthetic_b2.csv、aesthetic_b3.csv（image、aes_laion、aes_musiq_ava），
                  為 supp 表 4 所用的逐張讀數（pyiqa laion_aes、musiq-ava，CPU，防禦圖 8 位元 PNG 以 512² 讀入）
分母：一格＝(影像, 指令, 淨化)，淨化為 none、jpeg50、blur2、vae_trans；同一格的 Undefended 編輯成功
（accessory＝1、mismatch＝0、degraded＝0）才計入，與 measure_ablation_isr_summary.py 相同。Full、B2、B3 共用同一組分母格。
讀數：各條件防禦圖本身（淨化前）的分數。
    逐格加權：每個分母格計一次，分數為該格影像的防禦圖分數。
    影像層級：至少一格進分母的影像各計一次。
    未篩選：消融 ISR 的全部影像（crosseditor100）各計一次。
    306 張：aesthetic_*.csv 的全部影像（表 4 的樣本）。
區間：以影像為單位的 bootstrap 95% 百分位區間，1000 次，random.Random(20260812)；成對差 Full − B2、Full − B3 用同一組重抽。
輸出（--output-dir）：denominator.csv（各淨化的分母格數）、summary.csv、summary.txt。任一必要輸入或影像讀數缺少即失敗。
用法：python color/scripts/measure_ablation_isr_aesthetics.py --judgments <CSV> --key <CSV> --aesthetic-dir <目錄> --output-dir <目錄>
"""
import argparse
import csv
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

PURIFS = ["none", "jpeg50", "blur2", "vae_trans"]
CONDS = ["Full", "B2", "B3"]
FILES = {"Full": "aesthetic_full.csv", "B2": "aesthetic_b2.csv", "B3": "aesthetic_b3.csv"}
METRICS = ["aes_laion", "aes_musiq_ava"]
SEED, BOOTS = 20260812, 1000


def read(path):
    if not Path(path).is_file():
        raise SystemExit(f"缺輸入：{path}")
    return list(csv.DictReader(open(path, encoding="utf-8")))


def load_cells(judgments, key_path):
    key = {(r["sheet"].rsplit(".", 1)[0], r["row"]): r["condition"] for r in read(key_path)}
    cell = {}
    for r in read(judgments):
        if r["cell"] == "C1":
            continue
        cond = key[(r["sheet"], r["row"])]
        cell[(r["sheet"], r["purification"], cond)] = (r["accessory"] == "1" and r["mismatch"] == "0" and r["degraded"] == "0")
    sheets = sorted({s for s, _, _ in cell})
    if len(sheets) != 400 or len(cell) != 400 * 4 * 4:
        raise SystemExit(f"judgments 不完整：{len(sheets)} 張拼圖、{len(cell)} 格")
    return sheets, cell


def ci(values):
    v = sorted(values)
    return v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--aesthetic-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    sheets, cell = load_cells(a.judgments, a.key)
    aes = {c: {r["image"]: {m: float(r[m]) for m in METRICS} for r in read(a.aesthetic_dir / f)} for c, f in FILES.items()}
    denom = defaultdict(int)
    cells_of = defaultdict(int)  # 影像 → 分母格數
    images_all = sorted({s.rsplit("__", 1)[0] for s in sheets})
    for s in sheets:
        img = s.rsplit("__", 1)[0]
        for pu in PURIFS:
            if cell[(s, pu, "Undefended")]:
                denom[pu] += 1
                cells_of[img] += 1
    for c in CONDS:
        missing = [i for i in images_all if i not in aes[c]]
        if missing:
            raise SystemExit(f"{c} 缺防禦圖美學讀數：{missing[:5]}（共 {len(missing)} 張）")
    images = sorted(cells_of)
    rng = random.Random(SEED)
    boots = [[rng.choice(images) for _ in images] for _ in range(BOOTS)]
    rng_all = random.Random(SEED)
    boots_all = [[rng_all.choice(images_all) for _ in images_all] for _ in range(BOOTS)]

    def weighted(c, m, ims, w):
        return sum(aes[c][i][m] * w(i) for i in ims) / sum(w(i) for i in ims)

    schemes = {"cell_weighted": (images, boots, lambda i: cells_of[i]),
               "image_level": (images, boots, lambda i: 1),
               "unfiltered_100": (images_all, boots_all, lambda i: 1)}
    rows = []
    for name, (ims, bs, w) in schemes.items():
        n = sum(cells_of[i] for i in ims) if name == "cell_weighted" else len(ims)
        for m in METRICS:
            for c in CONDS:
                lo, hi = ci([weighted(c, m, b, w) for b in bs])
                rows.append({"scheme": name, "metric": m, "row": c, "n": n, "mean": round(weighted(c, m, ims, w), 5),
                             "ci_low": round(lo, 5), "ci_high": round(hi, 5)})
            for c in ("B2", "B3"):
                d = lambda b: weighted("Full", m, b, w) - weighted(c, m, b, w)
                lo, hi = ci([d(b) for b in bs])
                rows.append({"scheme": name, "metric": m, "row": f"Full-{c}", "n": n, "mean": round(d(ims), 5),
                             "ci_low": round(lo, 5), "ci_high": round(hi, 5)})
    for m in METRICS:
        for c in CONDS:
            vals = [v[m] for v in aes[c].values()]
            rows.append({"scheme": "all_306", "metric": m, "row": c, "n": len(vals), "mean": round(st.mean(vals), 5),
                         "ci_low": "", "ci_high": ""})
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with open(a.output_dir / "denominator.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["purification", "eligible_cells", "sheets"])
        for pu in PURIFS:
            wr.writerow([pu, denom[pu], len(sheets)])
        wr.writerow(["total", sum(denom.values()), len(sheets) * len(PURIFS)])
    with open(a.output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    lines = ["Denominator cells (undefended edit successful): "
             + ", ".join(f"{pu} {denom[pu]}" for pu in PURIFS) + f", total {sum(denom.values())}; images {len(images)}",
             f"{'scheme':15s} {'metric':14s} {'row':8s} {'n':>5s} {'mean':>8s}  95% CI"]
    for r in rows:
        ci_text = f"[{r['ci_low']}, {r['ci_high']}]" if r["ci_low"] != "" else ""
        lines.append(f"{r['scheme']:15s} {r['metric']:14s} {r['row']:8s} {r['n']:>5} {r['mean']:>8}  {ci_text}")
    (a.output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
