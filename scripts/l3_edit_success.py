"""未防禦的編輯本來就成功了嗎？

    python scripts/l3_edit_success.py

只對 24 張原圖各做一次 SigLIP 前向（數秒），不跑 SDEdit。

## 為什麼要問

Attention Attack（ACM MM 2025）在建資料集時明說：「We manually filtered the
dataset to ensure reliable results... **selecting images where edits are
successful**」。**在未防禦編輯本來就失敗的影像上量免疫效果沒有意義**——
防的是一個不會發生的事。

本專案沒有做這道過濾。逐圖看 `runs/figs/2026-08-04_l1_three_attacks.png`
至少有三張的未防禦編輯是失敗的：`woman_00`（→ a man，輸出仍是女性）、
`man_02`（→ a woman，仍是男性）、`bird_00`（→ a butterfly，仍是鳥）。

## 怎麼量

`runs/lo_baseline/results.csv` 已經有 `edit_siglip_a`，即
`SigLIP(y_ref, prompt)`——未防禦編輯與惡意 prompt 的相似度。缺的是同一個
prompt 對**原圖**的相似度 `SigLIP(x, prompt)`。兩者之差就是編輯效果：

    edit_effect = SigLIP(y_ref, prompt) − SigLIP(x, prompt)

正值代表編輯確實把影像推向了 prompt。E25 量過這個對照的噪聲範圍：
SigLIP 的 edit_effect 是 **+0.0276 ± 0.0237**（LEDGER 1.3），
故以 0 為界只是符號判定，判「明顯成功」應該要求超過那個標準差。

本腳本只補 `SigLIP(x, prompt)` 這一項，其餘從既有 CSV 讀。
"""

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.device import get_device  # noqa: E402

# E25 實測的 edit_effect 標準差（LEDGER 1.3）。用它當「明顯成功」的門檻，
# 而不是自己再定一個——同一個量已經量過，換一個定義只會多一個變因。
EDIT_EFFECT_SD = 0.0237


def main():
    ap = argparse.ArgumentParser(description="未防禦編輯的成功與否")
    ap.add_argument("--data", default="data/lo_aligned")
    ap.add_argument("--run", default="runs/lo_baseline")
    ap.add_argument("--out", default="runs/l3_criterion_axes/edit_success.csv")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    import yaml
    from PIL import Image
    import torchvision.transforms as T

    device = get_device()
    suite = MetricSuite(device=device)
    data = ROOT / args.data
    spec = yaml.safe_load((data / "prompts.yaml").read_text(encoding="utf-8"))

    # 未防禦編輯的 SigLIP，逐 (影像, 攻擊) 取 20 個種子的平均。
    # 三個攻擊共用同一組參照編輯，故三者的 edit_siglip_a 應該相同；
    # 取平均而非任取一個，並把跨攻擊的離散度印出來作為健全性檢查。
    refs = defaultdict(list)
    with open(ROOT / args.run / "results.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("edit_siglip_a"):
                refs[r["image"]].append(float(r["edit_siglip_a"]))

    rows = []
    with torch.no_grad():
        for cls, entry in spec.items():
            prompt = entry["prompts"][0]
            for p in sorted((data / cls).glob("*.png")):
                name = p.stem
                if name not in refs:
                    continue
                img = Image.open(p).convert("RGB").resize(
                    (args.size, args.size), Image.LANCZOS)
                x = T.ToTensor()(img).unsqueeze(0).to(device)
                s_orig = suite.semantic(x, prompt)["siglip"]
                s_edit = st.mean(refs[name])
                eff = s_edit - s_orig
                rows.append({
                    "image": name, "cls": cls, "prompt": prompt,
                    "siglip_orig": round(s_orig, 6),
                    "siglip_edit_ref": round(s_edit, 6),
                    "edit_effect": round(eff, 6),
                    "n_ref_rows": len(refs[name]),
                    "edit_worked": eff > 0,
                    "edit_worked_clearly": eff > EDIT_EFFECT_SD,
                })

    rows.sort(key=lambda r: r["edit_effect"])
    print(f"{'影像':<11}{'prompt':<26}{'原圖':>9}{'未防禦編輯':>11}"
          f"{'編輯效果':>10}  判定")
    for r in rows:
        tag = ("明顯成功" if r["edit_worked_clearly"]
               else ("勉強" if r["edit_worked"] else "**失敗**"))
        print(f"{r['image']:<11}{r['prompt']:<26}{r['siglip_orig']:>9.4f}"
              f"{r['siglip_edit_ref']:>11.4f}{r['edit_effect']:>+10.4f}  {tag}")

    n = len(rows)
    ok = sum(r["edit_worked"] for r in rows)
    clear = sum(r["edit_worked_clearly"] for r in rows)
    print(f"\n共 {n} 張：編輯效果為正 {ok} 張、"
          f"超過 E25 的標準差 {EDIT_EFFECT_SD} 的 {clear} 張、"
          f"**為負（編輯反而更遠離 prompt）{n - ok} 張**")
    print(f"平均編輯效果 {st.mean(r['edit_effect'] for r in rows):+.4f}"
          f" ± {st.pstdev(r['edit_effect'] for r in rows):.4f}"
          f"（E25 的對照值為 +0.0276 ± 0.0237）")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[l3] 寫出 {out}")
    print("讀法：判為「失敗」的影像上，任何免疫效果的量測都沒有意義——"
          "防的是一個不會發生的事。Attention Attack 建資料集時就先過濾掉它們")


if __name__ == "__main__":
    main()
