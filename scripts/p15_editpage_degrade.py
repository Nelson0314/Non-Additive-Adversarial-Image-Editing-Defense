"""對既有的編輯比對頁影像算感知劣化，取得 n≥2 的可判定樣本。

## 為什麼需要這一支

`p12_isr_rejudge.py` 對既有 `results.csv` 重判，發現唯一有訊號的一格是
`runs/e29c_P_tau0.10` 的 ΔNIQE = +1.571，但該 run 用 `--limit 1`，n = 1，
而判定要求 n ≥ 2（n = 1 時 `pstdev` 恆為 0，任何判定都自動成立——E25 曾因
缺這個條件產生 24 格假陽性）。

`runs/e29_edit_page/` 是另一個來源：它有 **car_00 與 car_01 兩張**，每張都
存了未防禦編輯（`*_01_orig_edited.png`）與各防禦版本的編輯結果
（`*_def_edited.png`）。那是 n = 2，判得動。

這批圖的攻擊設定是 E29 的：w=7.5、strength=0.5、τ=0.05、60 步，且用的是
**防禦訓練時見過的那個 ε**，即對防禦最有利的條件（E29 §4.1）。

## 這**不**取代網格

樣本只有兩張、只有一個運作點、只有 `untargeted` 目標。它回答的是
「既有資料在補上判準的另一半之後，結論會不會變」，不是 E31 要回答的
「有沒有任何運作點擋得下編輯」。

執行：python scripts/p15_editpage_degrade.py
成本：只讀既有 PNG，GPU 上秒級。
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.p12_isr_rejudge import judge_cell

PAGES = ["runs/e29_edit_page", "runs/e29c_edit_page"]
# 檔名形如 car_00_e29_C_lr0.3_def_edited.png / car_00_01_orig_edited.png
RE_DEF = re.compile(r"^(?P<img>\w+?_\d+)_(?P<arm>.+)_def_edited\.png$")
RE_ORIG = re.compile(r"^(?P<img>\w+?_\d+)_01_orig_edited\.png$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/p15_editpage_degrade")
    ap.add_argument("--degrade_tau", type=float, default=1.0,
                    help="感知劣化門檻。預設 1.0 是佔位，正式判定須用 p11 "
                         "的人眼階梯定出的值")
    args = ap.parse_args()

    import pyiqa

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    niqe = pyiqa.create_metric("niqe", device=device)
    from src.metrics.suite import MetricSuite
    suite = MetricSuite(device=device)

    def load(p):
        return T.ToTensor()(Image.open(p).convert("RGB")).unsqueeze(0).to(device)

    rows = []
    for page in PAGES:
        d = ROOT / page
        if not d.is_dir():
            raise FileNotFoundError(f"找不到 {d}")
        prompt = (d / "prompt.txt").read_text(encoding="utf-8").strip()
        origs = {}
        for f in sorted(d.glob("*_01_orig_edited.png")):
            m = RE_ORIG.match(f.name)
            origs[m.group("img")] = load(f)
        for f in sorted(d.glob("*_def_edited.png")):
            m = RE_DEF.match(f.name)
            if not m:
                raise ValueError(f"檔名不符預期格式：{f.name}")
            img, arm = m.group("img"), m.group("arm")
            if img not in origs:
                raise KeyError(
                    f"{f.name} 沒有對應的未防禦編輯 {img}_01_orig_edited.png。"
                    "配對差沒有對照就沒有意義"
                )
            y_def, y_orig = load(f), origs[img]
            with torch.no_grad():
                q_def, q_orig = float(niqe(y_def)), float(niqe(y_orig))
                s_def = suite.semantic(y_def, prompt)["siglip"]
                s_orig = suite.semantic(y_orig, prompt)["siglip"]
            rows.append({"page": d.name, "arm": arm, "image": img,
                         "prompt": prompt,
                         "niqe_orig": q_orig, "niqe_def": q_def,
                         "dniqe": q_def - q_orig,
                         "siglip_orig": s_orig, "siglip_def": s_def,
                         "dsiglip": s_def - s_orig})
            r = rows[-1]
            print(f"  {d.name}/{arm:<18} {img}  Δniqe={r['dniqe']:+.3f}  "
                  f"Δsiglip={r['dsiglip']:+.4f}", flush=True)

    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 逐（頁面, 臂）聚合成一格，用與正式判準同一個函數
    groups = defaultdict(lambda: ([], []))
    for r in rows:
        k = (r["page"], r["arm"])
        groups[k][0].append(r["dsiglip"])
        groups[k][1].append(r["dniqe"])

    verdicts = []
    print(f"\n{'頁面 / 臂':<44} {'n':>2} {'ΔSigLIP':>9} {'Δ劣化':>8} "
          f"{'語意':<4} {'劣化':<4} ISR")
    for (page, arm), (sem, deg) in sorted(groups.items()):
        v = judge_cell(sem, deg, args.degrade_tau)
        verdicts.append({"page": page, "arm": arm, **v})
        print(f"{page + ' / ' + arm:<44} {v['n']:>2} "
              f"{v['mean_dsiglip']:>+9.4f} {v['mean_ddegrade']:>+8.3f} "
              f"{'是' if v['semantic_fail'] else '否':<4} "
              f"{'是' if v['degrade_fail'] else '否':<4} "
              f"{'**是**' if v['isr'] else '否'}")

    with open(out / "verdicts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(verdicts[0].keys()))
        w.writeheader()
        w.writerows(verdicts)
    n = sum(v["isr"] for v in verdicts)
    print(f"\n[p15] {len(verdicts)} 格中 ISR 成立 {n} 格"
          f"（門檻 {args.degrade_tau}）")
    print(f"[p15] 寫出 {out / 'per_image.csv'} 與 {out / 'verdicts.csv'}")
    print("[p15] 提醒：n=2 是判定的下限，且只有一個運作點、一個目標函數。"
          "這不取代 E31 網格。")


if __name__ == "__main__":
    main()
