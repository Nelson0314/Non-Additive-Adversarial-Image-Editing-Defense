"""彙整 E14：LPIPS 為唯一綁定約束下，site S 與 site P 的學習率探測。

判準：tail5_lpips ≤ τ 的前提下，tail5_shift 最大者。

取最後 5 步的平均而非最後一步：purify_mode 為 rotate，相鄰步量的是不同
淨化條件，單一步帶有條件切換造成的跳動；5 步涵蓋 3 個算子一輪以上。
這是 E0d 已出現過的問題（曾以「相鄰兩步損失有沒有下降」當判準，實際比較的
是兩個不同的目標）。
"""
import csv
import glob
import json
import os

TAU = 0.05
rows = []
for site, lrs in (("S", ["0.03", "0.1", "0.3", "1.0"]),
                  ("P", ["0.008", "0.03", "0.1"])):
    for lr in lrs:
        d = f"runs/e14_{site}_{lr}"
        if not os.path.exists(d + "/summary.csv"):
            rows.append((site, lr, None))
            continue
        r = list(csv.DictReader(open(d + "/summary.csv")))[0]
        h = json.load(open(glob.glob(d + "/*/history.json")[0]))
        tail = h[-5:]
        rows.append((site, lr, {
            "shift": sum(x["edit_shift"] for x in tail) / len(tail),
            "lpips": sum(x["fid_lpips"] for x in tail) / len(tail),
            "linf": float(r["final_linf_total"]),
            "psnr": float(r["final_psnr_total"]),
            "disp": float(r.get("disp_mean_px") or 0.0),
            "sec": float(r["seconds"]),
        }))

hdr = "site  lr      tail5_shift  tail5_lpips  linf    psnr    disp_mean  秒/格  預算內"
print(hdr)
print("-" * len(hdr))
for site, lr, v in rows:
    if v is None:
        print(f"{site:4s}  {lr:6s}  (未完成)")
        continue
    ok = "是" if v["lpips"] <= TAU else "否"
    print(f"{site:4s}  {lr:6s}  {v['shift']:.4f}       {v['lpips']:.4f}"
          f"       {v['linf']:.4f}  {v['psnr']:5.2f}  {v['disp']:7.3f}"
          f"  {v['sec']:5.0f}  {ok}")

print()
for site in ("S", "P"):
    cand = [(v["shift"], lr) for s, lr, v in rows
            if s == site and v is not None and v["lpips"] <= TAU]
    if cand:
        best = max(cand)
        print(f"site {site} 選 lr={best[1]}（tail5_shift {best[0]:.4f}，"
              f"且 LPIPS 在 τ={TAU} 內）")
    else:
        print(f"site {site}：沒有任何 lr 守住 τ={TAU}，須降低 lr 或縮短步數")
