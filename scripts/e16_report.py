"""E16：放寬 site S 的位移上界後，其曲線是否繼續往外延伸。

E15 顯示 site S 在 tau=0.05 與 0.10 下達成的 LPIPS 相同（0.0407 / 0.0413），
即保真度預算已不再綁定，天花板是 max_disp=1.5 像素。此處把上界放到 3.0 與
6.0，看 net 是否隨之上升，以判斷「非加性在高失真端還有沒有空間」。
"""
import csv
import os

NOPURIFY = {("blur", "0.0"), ("noise", "0.0")}


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


def fv(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


def report(label, path):
    if not os.path.exists(path):
        print(f"{label:22s}  (未完成)")
        return
    rows = list(csv.DictReader(open(path)))
    ho = [r for r in rows
          if (r["purify"], r["strength"]) in NOPURIFY
          and r["noise_split"] == "heldout"]
    tr = [r for r in rows
          if r["purify"] == "identity" and r["noise_split"] == "train"]
    if not ho:
        print(f"{label:22s}  (尚無 heldout 列)")
        return
    net = mean([fv(r, "net_lpips") for r in ho])
    ed = mean([fv(r, "edit_lpips") for r in ho])
    ed_tr = mean([fv(r, "edit_lpips") for r in tr])
    print(f"{label:22s}  n={len(ho):2d}  net={net:.4f}  "
          f"lpips={mean([fv(r, 'defimg_lpips') for r in ho]):.4f}  "
          f"psnr={mean([fv(r, 'defimg_psnr') for r in ho]):5.2f}  "
          f"disp={mean([fv(r, 'disp_mean_px') for r in ho]):.3f}  "
          f"dispmax={mean([fv(r, 'disp_max_px') for r in ho]):.3f}  "
          f"過擬合={ed_tr / ed:.2f}x")


print("位移上界             net / 達成失真 / 位移實際用量")
print("-" * 100)
report("S max_disp=1.5 (E15)", "runs/e15_S_tau0.10/results.csv")
report("S max_disp=3.0", "runs/e16_S_disp3.0/results.csv")
report("S max_disp=6.0", "runs/e16_S_disp6.0/results.csv")
print()
report("P tau=0.10 (E15 對照)", "runs/e15_P_tau0.10/results.csv")
