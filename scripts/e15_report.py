"""彙整 E15：非加性（site S）對加性（site P）在匹配 LPIPS 下的編輯抵抗力。

必要問題只有一個：在人眼可辨失真相同的前提下，哪一個把文字編輯破壞得
比較多。故主表只看無淨化、未見噪聲種子下的 net_lpips。

三個容易讀錯的地方，逐一列出而非藏在平均值裡：

1. 訓練種子的數字不可當結論。φ 是針對訓練用的那一組 ε 優化的，實測
   site P 過擬合 3.30–3.56 倍。主表一律用 heldout，另設一欄併列過擬合倍率。
2. net 而非 edit。淨化本身就會讓編輯結果偏離，即使 φ=0。net 已扣掉
   對照（同一淨化施加於原圖）。
3. 實際達成的 LPIPS 才是匹配軸，不是設定的 τ。兩個位置未必都剛好用滿
   預算，故一律併列 defimg_lpips。
"""
import csv
import glob
import os
from collections import defaultdict

TAUS = ["0.02", "0.05", "0.10"]
SITES = ["S", "P"]


# 未見種子下「無淨化」的那一格不叫 identity：eval_sweep() 的每個算子都含
# 強度 0 的對照，故 blur@0.0 與 noise@0.0 就是恆等。results.csv 裡唯一標成
# identity 的是訓練種子那一列。兩者都取來互相印證，數值應一致。
NOPURIFY = {("blur", "0.0"), ("noise", "0.0")}


def is_nopurify(r):
    return (r["purify"], r["strength"]) in NOPURIFY and r["noise_split"] == "heldout"


def load(site, tau):
    p = f"runs/e15_{site}_tau{tau}/results.csv"
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p)))


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


print("=" * 78)
print("主表：無淨化、未見噪聲種子。net_lpips 越大表示編輯被破壞得越多")
print("=" * 78)
hdr = ("site  tau    n   net_lpips  edit_lpips  失真lpips  psnr    linf    "
       "disp_px  過擬合")
print(hdr)
print("-" * len(hdr))
main = {}
for tau in TAUS:
    for site in SITES:
        rows = load(site, tau)
        if rows is None:
            print(f"{site:4s}  {tau:5s}  (未完成)")
            continue
        ho = [r for r in rows if is_nopurify(r)]
        tr = [r for r in rows
              if r["purify"] == "identity" and r["noise_split"] == "train"]
        net = mean([f(r, "net_lpips") for r in ho])
        ed = mean([f(r, "edit_lpips") for r in ho])
        ed_tr = mean([f(r, "edit_lpips") for r in tr])
        lp = mean([f(r, "defimg_lpips") for r in ho])
        ps = mean([f(r, "defimg_psnr") for r in ho])
        li = mean([f(r, "final_linf_total") for r in ho])
        dp = mean([f(r, "disp_mean_px") for r in ho]) if "disp_mean_px" in rows[0] else float("nan")
        over = ed_tr / ed if ed else float("nan")
        main[(site, tau)] = (net, lp)
        print(f"{site:4s}  {tau:5s}  {len(ho):2d}  {net:9.4f}  {ed:10.4f}  "
              f"{lp:9.4f}  {ps:5.2f}  {li:6.4f}  {dp:7.3f}  {over:5.2f}x")

print()
print("匹配失真後的直接對比（同 tau 下 S 相對 P 的倍率）")
for tau in TAUS:
    if ("S", tau) in main and ("P", tau) in main:
        (ns, ls), (np_, lp_) = main[("S", tau)], main[("P", tau)]
        print(f"  tau={tau}:  S net {ns:.4f} @lpips {ls:.4f}   "
              f"P net {np_:.4f} @lpips {lp_:.4f}   S/P = {ns / np_:.2f}x")

print()
print("=" * 78)
print("附帶結果：淨化下的殘存（非本階段必要目標，有更好、沒有也可）")
print("=" * 78)
for tau in ["0.05"]:
    for site in SITES:
        rows = load(site, tau)
        if rows is None:
            continue
        by = defaultdict(list)
        for r in rows:
            if r["noise_split"] != "heldout":
                continue
            by[(r["purify"], r["strength"])].append(f(r, "net_lpips"))
        base = mean([v for k in by if k in NOPURIFY for v in by[k]])
        print(f"\nsite {site} (tau={tau}), 無淨化 net={base:.4f}")
        for k in sorted(by, key=lambda k: -mean(by[k])):
            m = mean(by[k])
            print(f"   {k[0]:10s} {k[1]:>6s}  net={m:.4f}  "
                  f"殘存={100 * m / base:5.1f}%")

print()
gen = {}
for site in SITES:
    p = f"runs/e15_{site}_tau0.05/generalization.csv"
    if not os.path.exists(p):
        continue
    rows = list(csv.DictReader(open(p)))
    gen[site] = rows
if gen:
    print("=" * 78)
    print("泛化：訓練時只見過 prompt[0] 與 strength=0.5")
    print("=" * 78)
    for site, rows in gen.items():
        by = defaultdict(list)
        for r in rows:
            by[r.get("eval_strength", r.get("strength"))].append(
                f(r, "edit_lpips"))
        print(f"site {site}: " + "  ".join(
            f"s={k}: {mean(v):.4f}" for k, v in sorted(by.items())))
