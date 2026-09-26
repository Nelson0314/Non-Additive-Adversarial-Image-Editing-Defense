"""產生 `report/aligned_matrix.html` 吃的 `data.js`。

頁面本身不含任何數字，全部由這一支從 `results/` 的 CSV 算出來：各方法的防禦圖
失真與位移指標、逐圖的防禦圖 LPIPS、逐格的位移（給影像矩陣的列順序用），以及
十個等失真條件的中位數（頁面用它判斷哪些值要標紅並附倍率）。

影像不由這一支產生，見 `--help` 末段的重建步驟。

用法：

    python code/report_matrix_data.py --out <報告目錄>/data.js

重建整個報告頁：

1. 遠端轉影像（防禦圖與原圖無損 WebP、編輯圖 q90 有損）到 `<報告目錄>/img/`：
   原圖與防禦圖取 `runs/defence_portraits/<條件>/` 與 `runs/eps_aligned/<條件>/`，
   編輯圖取 `runs/edit_preflight/{ip2p_si18|inpaint_undefended}/`、
   `runs/edit_defended/colour_curve_ours/<場景>_colour_curve_ours/` 與
   `runs/edit_defended_aligned/<條件>/<場景>_<條件>_aligned/`。
   檔名式樣：`orig_<影像>`、`mask_<影像>`、`def_<方法>_<影像>`、
   `e_<場景><指令編號>_<方法>_<影像>`，副檔名 `.webp`。
2. 跑這一支產生 `data.js`。
3. 以 `report/aligned_matrix.html` 為 `file_path` 發佈，`root` 指向報告目錄，
   `files` 列出 `data.js` 與 `img/` 底下全部檔案。整個 artifact 上限 256 個檔、
   單一版本 64 MB。更新時只傳改動的檔，沒傳的會保留；要移除某個檔用 `null`。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

#: 十個等失真條件，依既有的位移由高到低不重要——頁面自己排。
ALIGNED = ["dia_pt", "dia_r", "dayn", "sifm", "danp", "dct_shield",
           "photoguard_c", "dct_shield_y", "photoguard_linf", "mist"]

#: 影像矩陣取的四張照片與四個指令（ip2p 兩個、inpaint 兩個）。
#: 避開了逐格看圖查出來的問題格：`inpaint/p0` 的背景、`p2` 的雪景、
#: `man_02/p3` 的分母沒有狗。
NAMES = ["man_00", "man_01", "woman_00", "woman_01"]
CELLS = [("ip2p", "0"), ("ip2p", "1"), ("inpaint", "1"), ("inpaint", "3")]

PURIFIERS = ["blur1", "blur2", "crop_resize0.1", "jpeg30", "jpeg50", "jpeg80", "rotate15"]

#: 位移指標與它們的好壞方向（對防禦方而言）。`up` 是越大越好。
METRICS = [
    ("disp_lpips_full", "LPIPS", "up", 4),
    ("disp_lpips_subject", "LPIPS 主體", "up", 4),
    ("disp_lpips_background", "LPIPS 背景", "up", 4),
    ("disp_dists", "DISTS", "up", 4),
    ("disp_rms", "rms", "up", 4),
    ("disp_linf", "L inf", "up", 4),
    ("disp_ssim", "SSIM", "down", 4),
    ("disp_psnr", "PSNR", "down", 2),
    ("disp_vif_p", "VIFp", "down", 4),
    ("clip_pair", "CLIP", "down", 4),
    ("siglip_pair", "SigLIP", "down", 4),
]


def load(rel: str) -> list:
    with (paths.RESULTS / rel).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(rows, key, dp=5):
    return round(st.mean(float(r[key]) for r in rows), dp)


def build() -> dict:
    nat_disp = load("displacement.csv")
    ali_disp = load("aligned/displacement_aligned.csv")
    deltae = {r["condition"]: float(r["deltae00"]) for r in load("aligned/deltae.csv")}
    fid_native = {}
    for r in load("metrics_fidelity_union.csv"):
        fid_native.setdefault(r["condition"], []).append(float(r["fid_delta_e00"]))

    def agg(rows, cond):
        sel = [r for r in rows if r["condition"] == cond]
        out = {k: mean(sel, k) for k, _, _, _ in METRICS}
        out["blocked"] = sum(1 for r in sel if r["blocked"] == "True")
        out["cells"] = len(sel)
        return out

    def retained(rel, cond, purifier=None):
        v = [float(r["retained"]) for r in load(rel)
             if r["condition"] == cond and r["retained"]
             and (purifier is None or r["purifier"] == purifier)]
        return round(st.mean(v), 4) if v else None

    methods = []
    for cond in ALIGNED + ["colour_curve_ours"]:
        aligned = cond in ALIGNED
        solver = load("aligned/defence_%s_aligned.csv" % cond if aligned
                      else "defence_%s.csv" % cond)
        ret_src = "aligned/retention_aligned.csv" if aligned else "retention.csv"
        m = agg(ali_disp if aligned else nat_disp, cond)
        m.update({
            "name": cond,
            "arm": "aligned" if aligned else "native",
            "scale": float(solver[0]["eps_scale"]) if aligned else None,
            "fid_lpips": mean(solver, "fid_lpips", 4),
            "fid_psnr": mean(solver, "fid_psnr", 2),
            "fid_rms": mean(solver, "fid_rms", 5),
            "fid_linf": mean(solver, "fid_linf", 5),
            "deltae": deltae.get(cond) if aligned
                      else round(st.mean(fid_native[cond]), 3),
            "ret": retained(ret_src, cond),
            "ret_p": {p: retained(ret_src, cond, p) for p in PURIFIERS},
            "fid_by_image": {r["image"]: round(float(r["fid_lpips"]), 4)
                             for r in solver if r["image"] in NAMES},
        })
        methods.append(m)

    cellmap = {(r["condition"], r["scenario"], r["image"], r["prompt_index"]): r
               for r in ali_disp}
    cellmap.update({(r["condition"], r["scenario"], r["image"], r["prompt_index"]): r
                    for r in nat_disp if r["condition"] == "colour_curve_ours"})

    cells = []
    for scenario, index in CELLS:
        prompt = next(r["prompt"] for r in ali_disp
                      if r["scenario"] == scenario and r["prompt_index"] == index)
        for name in NAMES:
            per = {}
            for m in methods:
                r = cellmap.get((m["name"], scenario, name, index))
                if r:
                    per[m["name"]] = {k: round(float(r[k]), 5) for k, _, _, _ in METRICS}
            cells.append({"scenario": scenario, "pi": index, "prompt": prompt,
                          "image": name, "per": per})

    base = [m for m in methods if m["arm"] == "aligned"]
    return {
        "metrics": [{"key": k, "label": l, "dir": d, "dp": p} for k, l, d, p in METRICS],
        "methods": methods,
        "cells": cells,
        "names": NAMES,
        "purifiers": PURIFIERS,
        "order": [m["name"] for m in sorted(methods, key=lambda x: -x["disp_lpips_full"])],
        "anchor": [m for m in methods if m["name"] == "colour_curve_ours"][0]["fid_lpips"],
        # 頁面用中位數判斷哪些值要標紅：與它差兩倍以上（PSNR 用 6 dB）就標。
        "median": {k: round(st.median([m[k] for m in base]), 5)
                   for k in ("fid_lpips", "fid_psnr", "fid_rms", "fid_linf", "deltae")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("window.MT = " + json.dumps(data, ensure_ascii=False) + ";\n",
                        encoding="utf-8")
    print(f"[ALLDONE] {args.out}（{len(data['methods'])} 方法、{len(data['cells'])} 格）",
          flush=True)


if __name__ == "__main__":
    main()
