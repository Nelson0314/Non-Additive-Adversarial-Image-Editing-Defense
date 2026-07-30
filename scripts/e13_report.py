"""彙整 E13 site S 學習率探測結果。

判準說明：不能只看 final_shift。τ=0.05 是綁定約束，超出預算的那些 lr 其
shift 是「多花失真換來的」，不可與守住預算的相比。另外 purify_mode 預設為
rotate，相鄰步量的是不同淨化條件，單一步的數字帶有條件切換造成的跳動，
故同時報最後 5 步的平均（涵蓋 3 個算子一輪以上）。
"""
import csv
import glob
import json
import sys

rows = []
for lr in ["0.008", "0.03", "0.1", "0.3"]:
    d = "runs/e13_slr_" + lr
    r = list(csv.DictReader(open(d + "/summary.csv")))[0]
    h = json.load(open(glob.glob(d + "/*/history.json")[0]))
    tail = h[-5:]
    rows.append((
        lr,
        float(r["final_shift"]),
        float(r["final_lpips"]),
        float(r["final_psnr_total"]),
        float(r["final_linf_total"]),
        float(r["disp_mean_px"]),
        float(r["disp_max_px"]),
        sum(x["edit_shift"] for x in tail) / len(tail),
        sum(x["fid_lpips"] for x in tail) / len(tail),
    ))

hdr = ("lr      shift   lpips   psnr    linf    disp_mean disp_max"
       "  tail5_shift tail5_lpips  預算內")
print(hdr)
print("-" * len(hdr))
for (lr, s, lp, ps, li, dm, dx, t5s, t5l) in rows:
    ok = "是" if t5l <= 0.05 else "否"
    print(f"{lr:6s}  {s:.4f}  {lp:.4f}  {ps:5.2f}  {li:.4f}  "
          f"{dm:7.3f}  {dx:7.3f}   {t5s:.4f}      {t5l:.4f}      {ok}")

print()
print("對照 site P（E8, τ=0.05, 6 張圖平均）：shift 0.0378 對應 lpips 0.0378")
print("對照 site P（E2 主網格 r=16, 6 張圖）：shift 0.1133, psnr 39.79, lpips 0.0626")
