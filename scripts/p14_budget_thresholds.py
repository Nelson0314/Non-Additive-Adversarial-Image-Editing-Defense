"""逐預算重定兩道次要保真門檻（τ_acut、τ_chroma）。

## 為什麼需要

`τ_acut = 0.04`（E20）與 `τ_chroma = 0.8`（E28）都是**絕對值**，而且都是在
`τ_lpips = 0.05` 那個量級上定的。E31 要把預算拉到文獻區間（0.267–0.362），
p13 的探針顯示問題不在防禦方法：

    純白高斯雜訊（i.i.d.、無空間結構）在 LPIPS 0.20 的 acut 已是 0.0414，
    在 0.28 是 0.0875 與 chroma 1.1284——兩道門檻都被跨過。

也就是說現行門檻把可達的 LPIPS 上限壓在 0.15–0.20，與最佳化找到什麼解無關。
在那個約束集下問「文獻預算下防禦會不會成立」是問不出來的。

## 作法

兩道門檻的職責不是「限制失真量」——那是 τ_lpips 的事——而是**擋住最佳化去買
LPIPS 收費不足的那兩種失真**：模糊（E18/E19 觀察到的）與空間連貫的色偏
（E27/E28 觀察到的 site C）。職責是相對的，門檻就該是相對的。

沿用 E20 的多臂等 LPIPS 探針，只是把「預算」加成一個軸：對每一個 τ_lpips，
把五個臂全部校準到該 LPIPS，量各臂的 acut 與 chroma，再把門檻定在
「該通過的臂」與「該被擋下的臂」之間。

    acut：   通過 = {noise, warp_bicubic}   擋下 = {blur, warp_bilinear}
    chroma： 通過 = {noise, warp_bicubic}   擋下 = {chroma}

臂的歸屬直接沿用 E20 §5.2 與 E28 §1 的既有判定，不重新裁定：使用者對比對頁
的判讀是「0.4–0.6 px 的位移不明顯，模糊明顯較糟」，而雙線性重取樣本身會鈍化
（E20 實測只保留 85.0% 銳利度，雙三次為 99.9%），故雙線性歸在該擋下那一側。

## 門檻定在區間的哪個位置

不取幾何中點，而是**沿用既有人眼定錨值所在的位置**。作法是先在定錨預算
（τ_lpips = 0.05，即 E20 與 E28 當初量的那個量級）上量出通過側上緣 `p` 與
擋下側下緣 `b`，再算既有門檻 `τ₀` 在 [p, b] 這個對數區間上的位置

    pos = ln(τ₀ / p) / ln(b / p)

然後在每一個預算上以同一個 `pos` 取值。這樣做的理由是不另立標準：τ₀ 是使用者
判讀比對頁定出來的，是這裡唯一的真值；本腳本只把那個判斷延伸到別的預算，
而不是用一個新規則取代它。

實測（car_00）：τ_chroma = 0.8 落在 pos ≈ 0.50，即幾何中點——幾何平均在該軸上
獨立重現了 E28 的人眼定錨值（0.802 對 0.8）。τ_acut = 0.04 落在 pos ≈ 0.12，
明顯偏向通過側，與 E20 「site P 自身的 0.0089 與雙三次的 0.0296 都須留有餘裕」
的敘述一致。兩軸的 pos 不同，故各用各的。

幾何中點的值一併輸出供對照，不取代上述。

## 驗證

`thresholds.csv` 的 τ_lpips = 0.05 那一列依定義會重現既有值（pos 就是由它算
出來的）。真正的檢查是**兩側有沒有間隔**：`pick_threshold` 在通過側上緣不低於
擋下側下緣時直接拋出，因為那表示在該預算上這個指標分不出兩群，門檻無論定在
哪裡都會誤判其中一側。

執行：python scripts/p14_budget_thresholds.py
成本：本機 CPU 約 10 分鐘 / GPU 數分鐘。不需要 SD 權重。
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.p9_chroma_probe import calibrate, make_arm, to_tensor
from src.metrics.chroma import local_chroma_bias
from src.metrics.local_acutance import local_acutance_dev

# 臂的**實作**沿用 p9（`make_arm`），但**搜尋範圍**在此重新給，不改 p9。
# p9 的 ARMS 只需要涵蓋單一預算 τ=0.05；本腳本要跨到 0.28，兩個變形臂的
# 上界 3.0 不夠——實測 person_00 的 warp_bilinear 在 3.0 只到 LPIPS 0.2196，
# `calibrate` 因此正確地拋出而非回傳上界。
# 改 p9 的常數會動到 E28 已入庫結果的重跑條件，故在此另立。
ARMS = [
    ("blur", 0.0, 8.0),
    ("noise", 0.0, 0.5),
    ("warp_bilinear", 0.0, 20.0),
    ("warp_bicubic", 0.0, 20.0),
    ("chroma", 0.0, 6.0),
]

# 臂的歸屬。沿用 E20 §5.2 與 E28 §1 的既有判定，見模組 docstring。
PASS_ACUT = ("noise", "warp_bicubic")
BLOCK_ACUT = ("blur", "warp_bilinear")
PASS_CHROMA = ("noise", "warp_bicubic")
BLOCK_CHROMA = ("chroma",)


def pick_threshold(values: dict, pass_arms, block_arms, pos: float = 0.5):
    """回傳 (門檻, 通過側上緣, 擋下側下緣)。

    門檻取對數區間 [通過側上緣, 擋下側下緣] 上的第 `pos` 個位置，
    `pos=0.5` 即幾何平均。

    兩側若沒有間隔就拋出：那表示在這個預算上該指標分不出這兩群，門檻無論
    定在哪裡都會誤判其中一側。靜默取中點會產生一個看起來有值、實際不成立的
    門檻，而整批網格會建立在它上面。
    """
    hi_pass = max(values[a] for a in pass_arms)
    lo_block = min(values[a] for a in block_arms)
    if lo_block <= hi_pass:
        raise ValueError(
            f"通過側最大 {hi_pass:.4f} 已不低於擋下側最小 {lo_block:.4f}，"
            f"此預算上該指標分不出兩群，門檻不成立"
        )
    tau = hi_pass * (lo_block / hi_pass) ** pos
    return tau, hi_pass, lo_block


def anchor_position(tau0: float, hi_pass: float, lo_block: float) -> float:
    """既有人眼定錨值 `tau0` 在對數區間上的位置。

    落在區間外時拋出：那表示既有門檻本身已經擋掉了該通過的臂（pos<0）或
    放過了該擋下的臂（pos>1），此時「沿用它的位置」沒有意義，必須先處理
    那個矛盾而不是外推一個不成立的位置。
    """
    pos = math.log(tau0 / hi_pass) / math.log(lo_block / hi_pass)
    if not 0.0 <= pos <= 1.0:
        raise ValueError(
            f"既有門檻 {tau0} 不落在 [{hi_pass:.4f}, {lo_block:.4f}] 之間"
            f"（pos={pos:.3f}）。它已經誤判了其中一側，不可作為外推的依據"
        )
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/dayn_testset/car/car_00.png,"
                                        "data/dayn_testset/dog/dog_00.png,"
                                        "data/dayn_testset/person/person_00.png")
    ap.add_argument("--budgets", default="0.05,0.10,0.15,0.20,0.28")
    ap.add_argument("--anchor_budget", type=float, default=0.05,
                    help="既有門檻被定出來的那個預算。E20 的 τ_acut 與 E28 的"
                         "τ_chroma 都是在 τ_lpips=0.05 的量級上定的")
    ap.add_argument("--anchor_tau_acut", type=float, default=0.04)
    ap.add_argument("--anchor_tau_chroma", type=float, default=0.8)
    ap.add_argument("--out", default="runs/p14_budget_thresholds")
    args = ap.parse_args()

    import piq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    lpips = piq.LPIPS().to(device)

    def lp(a, b):
        with torch.no_grad():
            return lpips(a.clamp(0, 1).to(device), b.clamp(0, 1).to(device))

    paths = [ROOT / p.strip() for p in args.images.split(",")]
    budgets = [float(b) for b in args.budgets.split(",")]

    rows = []
    for path in paths:
        x = to_tensor(path)
        for target in budgets:
            for kind, lo, hi in ARMS:
                amt = calibrate(kind, x, lp, target, lo, hi)
                y = make_arm(kind, x, amt).clamp(0, 1)
                with torch.no_grad():
                    rows.append({
                        "image": path.stem, "budget": target, "arm": kind,
                        "amount": round(amt, 6),
                        "lpips": float(lp(x, y)),
                        "acut": float(local_acutance_dev(x, y)),
                        "chroma": float(local_chroma_bias(x, y)),
                    })
                r = rows[-1]
                print(f"  {path.stem} τ={target} {kind:<14} amt={amt:.4f} "
                      f"lpips={r['lpips']:.4f} acut={r['acut']:.4f} "
                      f"chroma={r['chroma']:.4f}", flush=True)

    with open(out / "arms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 逐預算跨影像取平均後定門檻。跨影像平均而非逐影像各定一個：門檻是
    # 實驗設定的一部分，必須對整批影像是同一個值，否則各格的約束不同。
    def bounds(target):
        acut, chroma = {}, {}
        for kind, _, _ in ARMS:
            sel = [r for r in rows if r["budget"] == target and r["arm"] == kind]
            acut[kind] = sum(r["acut"] for r in sel) / len(sel)
            chroma[kind] = sum(r["chroma"] for r in sel) / len(sel)
        return acut, chroma

    if args.anchor_budget not in budgets:
        raise ValueError(
            f"定錨預算 {args.anchor_budget} 不在 --budgets 內。位置 pos 必須"
            f"在既有門檻被定出來的那個預算上量，不可用別的預算代替"
        )
    a0, c0 = bounds(args.anchor_budget)
    pos_a = anchor_position(args.anchor_tau_acut,
                            *pick_threshold(a0, PASS_ACUT, BLOCK_ACUT)[1:])
    pos_c = anchor_position(args.anchor_tau_chroma,
                            *pick_threshold(c0, PASS_CHROMA, BLOCK_CHROMA)[1:])
    print(f"\n[定錨] τ_acut={args.anchor_tau_acut} 位於 pos={pos_a:.3f}；"
          f"τ_chroma={args.anchor_tau_chroma} 位於 pos={pos_c:.3f}"
          f"（0.5 即幾何中點）", flush=True)

    picks = []
    for target in budgets:
        acut, chroma = bounds(target)
        tau_a, a_pass, a_block = pick_threshold(acut, PASS_ACUT, BLOCK_ACUT,
                                                pos_a)
        tau_c, c_pass, c_block = pick_threshold(chroma, PASS_CHROMA,
                                                BLOCK_CHROMA, pos_c)
        geo_a = pick_threshold(acut, PASS_ACUT, BLOCK_ACUT)[0]
        geo_c = pick_threshold(chroma, PASS_CHROMA, BLOCK_CHROMA)[0]
        picks.append({
            "budget": target,
            "tau_acut": round(tau_a, 4),
            "acut_pass_max": round(a_pass, 4),
            "acut_block_min": round(a_block, 4),
            "tau_acut_geomean": round(geo_a, 4),
            "tau_chroma": round(tau_c, 4),
            "chroma_pass_max": round(c_pass, 4),
            "chroma_block_min": round(c_block, 4),
            "tau_chroma_geomean": round(geo_c, 4),
            "pos_acut": round(pos_a, 4), "pos_chroma": round(pos_c, 4),
            **{f"acut_{k}": round(v, 4) for k, v in acut.items()},
            **{f"chroma_{k}": round(v, 4) for k, v in chroma.items()},
        })

    with open(out / "thresholds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(picks[0].keys()))
        w.writeheader()
        w.writerows(picks)

    print("\n預算   τ_acut   (通過≤ / 擋下≥)        τ_chroma  (通過≤ / 擋下≥)")
    for p in picks:
        print(f"{p['budget']:<6} {p['tau_acut']:<8} "
              f"({p['acut_pass_max']:.4f} / {p['acut_block_min']:.4f})"
              f"      {p['tau_chroma']:<9} "
              f"({p['chroma_pass_max']:.4f} / {p['chroma_block_min']:.4f})")
    print(f"\n[p14] 寫出 {out / 'arms.csv'} 與 {out / 'thresholds.csv'}")
    print("[p14] 兩側都有間隔（`pick_threshold` 未拋出）即代表門檻在每一個"
          "預算上都成立。E31 網格的每一格須以對應預算的這兩個值傳入 "
          "--tau_acut / --tau_chroma。")


if __name__ == "__main__":
    main()
