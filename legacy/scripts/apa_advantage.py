#!/usr/bin/env python
"""apa 有沒有優勢：三層指標一次算完，並把「配對比較」與「絕對水準」分開。

    python scripts/apa_advantage.py --new runs/s3t20_r_merged \
        --old runs/s3t20_merged

## 三個容易誤讀的地方，本腳本各自處理一個

**配對與非配對不可混談。** `apa+A` 對 `apa` 是同影像、同淨化、同 seed 的
配對比較，樣本數 3×18×5 = 270；`apa` 對 baseline 則是跨參數化的比較。前者
可以看逐格勝率，後者只能看平均，混在一起報等於用配對的統計力去支撐非配對
的結論。

**比值不能離開絕對水準單獨讀。** 強模糊下 `apa` 對最佳 baseline 的
`edit_lpips` 比值是 2.5，但兩者的絕對值都遠低於不淨化時的 baseline——輸入
已經被模糊毀掉，那個比值比的是殘差不是防禦。故每一列同時印比值與絕對值。

**位移不等於防禦成功。** 第 3 層的 ΔNIQE 與銳利度比要與第 1 層並排看：位移
大而 ΔNIQE 也大，代表「輸出變糟」而不是「編輯被導離」。
"""

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

L1 = ["edit_lpips", "edit_psnr", "edit_ssim", "edit_vif_p", "edit_fsim"]
L2 = ["effect_clip", "effect_siglip"]
L3 = ["edit_acutance_ratio", "edit_rms"]
FID = ["fid_lpips", "fid_dists", "fid_psnr", "fid_acutance_ratio"]
HIGHER_IS_STRONGER = {"edit_lpips", "edit_mse", "effect_clip", "effect_siglip"}


def load(batch: Path, rename=None):
    rows = defaultdict(dict)
    with (batch / "grid.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["stage"] != "eval":
                continue
            cond = (rename or {}).get(r["condition"], r["condition"])
            key = (cond, r["image_id"], r["purify_kind"],
                   float(r["purify_strength"]), int(r["seed"]))
            rows[key] = r
    return rows


def dniqe(r):
    """ΔNIQE = 防禦後的編輯 − 未防禦的編輯。正值代表輸出更糟（NIQE 越低越好）。

    順序取自 `eval_executor` 的 `suite.full(y_ctrl, y_def)`：`_a` 是**未防禦**
    的那張、`_b` 是防禦後的那張。反過來寫會讓「靠劣化撐起來的免疫」在表上
    看起來像是「輸出品質反而變好」，而那個符號錯誤沒有任何症狀。
    """
    return float(r["edit_niqe_b"]) - float(r["edit_niqe_a"])


def get(r, k):
    return dniqe(r) if k == "dniqe" else float(r[k])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True)
    a = ap.parse_args(argv)

    D = load(a.old)
    D.update(load(a.new, {"apa": "apa+A", "Ra": "Ra+A"}))
    conds = ["apa", "apa+A", "Ra", "Ra+A", "photoguard_c", "mist", "dia_r"]
    purs = sorted({(k[2], k[3]) for k in D if k[0] == "apa"})

    def cells(cond, pur=None):
        return [r for k, r in D.items()
                if k[0] == cond and (pur is None or (k[2], k[3]) == pur)]

    # ---- 1. 配對：apa+A 對 apa ----
    print("=" * 78)
    print("1. 配對比較 apa+A 對 apa（同影像、同淨化、同 seed）")
    print("=" * 78)
    print(f"{'指標':<22}{'apa':>10}{'apa+A':>10}{'平均差':>10}{'apa+A 較優':>12}{'n':>5}")
    for m in ["edit_lpips", "edit_psnr", "effect_clip", "effect_siglip",
              "dniqe", "edit_acutance_ratio", "fid_lpips", "fid_dists"]:
        pairs = [(get(D[k], m), get(D[("apa+A",) + k[1:]], m))
                 for k in D if k[0] == "apa" and ("apa+A",) + k[1:] in D]
        o = [p[0] for p in pairs]
        n = [p[1] for p in pairs]
        better = sum((y > x) if m in HIGHER_IS_STRONGER else (y < x)
                     for x, y in pairs)
        arrow = "↑優" if m in HIGHER_IS_STRONGER else "↓優"
        print(f"{m + ' ' + arrow:<22}{st.fmean(o):>10.4f}{st.fmean(n):>10.4f}"
              f"{st.fmean(n) - st.fmean(o):>+10.4f}"
              f"{better / len(pairs) * 100:>11.0f}%{len(pairs):>5}")

    # ---- 2. 逐淨化：位移的絕對水準 ----
    print()
    print("=" * 78)
    print("2. edit_lpips 逐淨化（↑ 代表推得更遠）。括號內是對未淨化的保留率")
    print("=" * 78)
    head = f"{'淨化':<18}" + "".join(f"{c:>17}" for c in conds)
    print(head)
    ident = {c: st.fmean(get(r, "edit_lpips")
                         for r in cells(c, ("identity", 0.0))) for c in conds}
    for p in purs:
        line = f"{p[0] + ' ' + format(p[1], 'g'):<18}"
        for c in conds:
            cs = cells(c, p)
            if not cs:
                line += f"{'—':>17}"
                continue
            v = st.fmean(get(r, "edit_lpips") for r in cs)
            line += f"{v:>10.4f}({v / ident[c] * 100:>4.0f}%)"
        print(line)

    # ---- 3. 三層並排（全部淨化平均） ----
    print()
    print("=" * 78)
    print("3. 三層並排（18 個淨化設定 × 3 影像 × 5 seed 的平均）")
    print("=" * 78)
    ks = ["edit_lpips", "edit_psnr", "edit_ssim", "effect_clip", "effect_siglip",
          "dniqe", "edit_acutance_ratio", "fid_lpips", "fid_dists"]
    print(f"{'條件':<15}" + "".join(f"{k.replace('edit_', 'e_').replace('effect_', 'f_'):>12}"
                                    for k in ks))
    for c in conds:
        cs = cells(c)
        print(f"{c:<15}" + "".join(f"{st.fmean(get(r, k) for r in cs):>12.4f}"
                                   for k in ks))

    # ---- 4. 單位失真換到的位移 ----
    print()
    print("=" * 78)
    print("4. 每單位可見失真換到的位移（edit_lpips / fid_dists，不淨化）")
    print("=" * 78)
    for c in conds:
        cs = cells(c, ("identity", 0.0))
        e = st.fmean(get(r, "edit_lpips") for r in cs)
        f = st.fmean(get(r, "fid_dists") for r in cs)
        q = st.fmean(get(r, "dniqe") for r in cs)
        print(f"{c:<15} edit_lpips {e:.4f}  fid_dists {f:.4f}  "
              f"效率 {e / f:>6.2f}  ΔNIQE {q:>+7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
