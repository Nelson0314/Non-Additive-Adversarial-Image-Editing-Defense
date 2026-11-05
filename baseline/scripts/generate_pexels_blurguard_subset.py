"""BlurGuard 的 Pexels 100 張子集：性別 × color3d_pexels 防禦圖 LPIPS（306 張上的四分位）分層，各層名額為 100 × 該層比例
（最大餘數法），必含 color trial pexels_subsets/budget_sweep.txt 的 20 張，層內其餘以 random.Random(0) 抽取。
輸出 runtime/pexels_portraits/blurguard/{images.txt,strata.csv}，並印出與 306 張的 KS 檢定。
用法（遠端 baseline 專案根，CPU）：python generate_blurguard_subset.py"""
import csv
import json
import random
from pathlib import Path

from scipy.stats import ks_2samp

N = 100
CT = Path("../color/trials/lut3d_defense_and_naturalness")
OUT = Path("runtime/pexels_portraits/blurguard")
sets = json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
lpips = {}
for s in sets:
    p = Path("runtime/pexels_portraits/followup") / (s["root"] or "") / "results/color3d/fidelity.csv"
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["condition"] == "color3d_pexels":
            lpips[r["image"]] = float(r["fid_lpips"])
assert len(lpips) == 306, len(lpips)
values = sorted(lpips.values())
q = [values[int(round(k * (len(values) - 1) / 4))] for k in (1, 2, 3)]


def stratum(im):
    g = "man" if im.startswith("man_") else "woman"
    return g, 1 + sum(lpips[im] > b for b in q)


forced = (CT / "pexels_subsets/budget_sweep.txt").read_text(encoding="utf-8").split()
strata = {}
for im in sorted(lpips):
    strata.setdefault(stratum(im), []).append(im)
share = {k: len(v) / 306 for k, v in strata.items()}
raw = {k: N * v for k, v in share.items()}
quota = {k: int(v) for k, v in raw.items()}
for k in sorted(raw, key=lambda k: raw[k] - quota[k], reverse=True)[: N - sum(quota.values())]:
    quota[k] += 1
rng = random.Random(0)
chosen, rows = [], []
for k in sorted(strata):
    must = [im for im in strata[k] if im in forced]
    rest = [im for im in strata[k] if im not in forced]
    need = quota[k] - len(must)
    if need < 0:
        raise SystemExit(f"層 {k} 的 budget_sweep 影像 {len(must)} 張超過名額 {quota[k]}")
    pick = must + rng.sample(rest, need)
    chosen += pick
    rows.append({"gender": k[0], "lpips_quartile": k[1], "population_306": len(strata[k]), "share_306": round(share[k], 4),
                 "quota": quota[k], "forced_budget_sweep": len(must), "chosen": len(pick)})
assert len(chosen) == N and set(forced) <= set(chosen)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "images.txt").write_text("\n".join(sorted(chosen)) + "\n", encoding="utf-8")
with open(OUT / "strata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
sub = [lpips[im] for im in chosen]
ks = ks_2samp(sub, list(lpips.values()))
men = sum(im.startswith("man_") for im in chosen)
print(f"[SUBSET] {N} 張（男 {men}／女 {N - men}），四分位邊界 {[round(b, 4) for b in q]}，"
      f"color3d 防禦圖 LPIPS 平均 {sum(sub) / N:.4f}（306 張 {sum(lpips.values()) / 306:.4f}），KS D={ks.statistic:.4f} p={ks.pvalue:.4f}")
