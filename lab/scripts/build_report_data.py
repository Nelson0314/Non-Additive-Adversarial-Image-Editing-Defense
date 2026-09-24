"""把讀數 CSV 併成報告頁的 `data.js`。

四份來源：`displacement.csv`（位移）、`retention.csv`（保留率）、
`fidelity.csv`（四個失真指標，逐張）、各臂的 `defence/*/results.csv`
（求解端自己的欄位，逐臂不同）。

**還沒跑完的臂列在 `pending`**：頁面照樣把它們的列畫出來，格子寫「還沒跑完」，
不是靜默省略。這樣表的形狀在資料進來之前就固定下來，補資料時不需要改版面。

用法
    python lab/scripts/build_report_data.py --out lab/report/data.js
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
from pathlib import Path

import yaml

IMAGES = ["man_00", "man_01", "man_02", "man_03",
          "woman_00", "woman_01", "woman_02", "woman_03"]

#: 依 ip2p 全圖位移由強到弱；`pending` 的臂排在最後。
#: **`style_random` 與 `style_low` 已退役**，不在這一串裡。理由見
#: `lab/docs/DESIGN.md` 的「已退役：直接交付 SDEdit 輸出」。
#: 臉框長出空間權重場的四個臂已移除，理由見 DESIGN.md「已移除」一節。
ARMS = ["inpaint_outside_face", "style_affine", "curve_dual_chroma", "style_opt",
        "inpaint_bg", "ab_warp", "ab_prism", "ab_prism_random_r1",
        "ab_prism_random_r2", "ab_prism_random_r3", "style_warp"]
PENDING = []
PURIFIERS = ["jpeg80", "jpeg50", "jpeg30", "blur1", "blur2",
             "crop_resize0.1", "rotate15"]
NONGEO = PURIFIERS[:5]

#: 主表等失真對齊的錨（另一批的數字，只當參照）。
ANCHOR = 0.3344

REFERENCE = {
    "colour_curve_ours": {
        "disp": {"ip2p": {"full": 0.3821, "subj": 0.3867, "blocked": 3},
                 "inpaint": {"full": 0.4431, "subj": 0.3317, "blocked": 4}},
        "ret": {"jpeg80": 1.018, "jpeg50": 1.045, "jpeg30": 1.065,
                "blur1": 0.980, "blur2": 0.996, "crop_resize0.1": 1.053,
                "rotate15": 0.906, "nongeo": 1.021},
        "psnr": 16.67, "lpips": 0.3367},
    "diffvax": {"ret": {"jpeg80": 0.927, "jpeg50": 0.902, "jpeg30": 0.862,
                        "blur1": 0.880, "blur2": 0.734, "crop_resize0.1": 0.597,
                        "rotate15": 0.689, "nongeo": 0.861}},
    "dct_shield_y": {"disp": {"ip2p": {"full": 0.6411, "subj": 0.6714, "blocked": 31},
                              "inpaint": {"full": 0.6287, "subj": 0.6166, "blocked": 2}}},
    # 主表那條線的等失真對齊（另一批）：倍率、對齊後 LPIPS、ΔE00、PSNR、
    # 以及同條件內只換照片的 rms 倍率。最後一欄是「預算寫在哪個域」的證據。
    "aligned": [
        {"c": "photoguard_linf", "s": 0.086, "l": 0.3082, "d": 0.544, "p": 46.60, "rms": 1.03},
        {"c": "sifm", "s": 0.289, "l": 0.3107, "d": 1.139, "p": 42.99, "rms": 1.08},
        {"c": "mist", "s": 0.093, "l": 0.3149, "d": 0.539, "p": 45.82, "rms": 1.05},
        {"c": "dct_shield_y", "s": 0.073, "l": 0.3164, "d": 1.243, "p": 38.74, "rms": 1.96},
        {"c": "dia_pt", "s": 0.418, "l": 0.3203, "d": 1.094, "p": 43.53, "rms": 1.10},
        {"c": "photoguard_c", "s": 0.491, "l": 0.3247, "d": 0.611, "p": 47.07, "rms": 1.00},
        {"c": "dct_shield", "s": 0.211, "l": 0.3276, "d": 1.553, "p": 39.93, "rms": 1.65},
        {"c": "danp", "s": 0.323, "l": 0.3345, "d": 1.150, "p": 42.31, "rms": 1.08},
        {"c": "dayn", "s": 0.430, "l": 0.3379, "d": 1.598, "p": 40.45, "rms": 1.08},
        {"c": "dia_r", "s": 0.969, "l": 0.3425, "d": 1.460, "p": 40.71, "rms": 1.07},
    ],
}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def read(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("lab/results"))
    ap.add_argument("--defence", type=Path, default=Path("lab/runs/defence"))
    ap.add_argument("--data", type=Path, default=Path("lab/data/portraits"))
    ap.add_argument("--out", type=Path, default=Path("lab/report/data.js"))
    args = ap.parse_args()

    defence = {}
    for arm in ARMS:
        p = args.defence / arm / "results.csv"
        defence[arm] = ({r["image"]: {k: num(v) for k, v in r.items()}
                         for r in read(p)} if p.is_file() else {})

    fid = {}
    for r in read(args.results / "fidelity.csv"):
        fid.setdefault(r["arm"], {})[r["image"]] = {
            k: num(v) for k, v in r.items() if k not in ("arm", "image")}

    disp = read(args.results / "displacement.csv")
    ret = read(args.results / "retention.csv")

    cells = {}
    for r in disp:
        (cells.setdefault(r["condition"], {}).setdefault(r["scenario"], {})
         .setdefault(r["image"], {})[r["prompt_index"]]) = {
            "full": round(float(r["disp_lpips_full"]), 4),
            "subj": round(float(r["disp_lpips_subject"]), 4),
            "bg": round(float(r["disp_lpips_background"]), 4),
            "blocked": r["blocked"] == "True",
            "siglip": round(float(r["siglip_pair"]), 4),
            "prompt": r["prompt"]}

    # 逐格只留 jpeg30（報告頁只收那一道的圖），其餘走彙總
    retj = {}
    for r in ret:
        if r["purifier"] != "jpeg30":
            continue
        (retj.setdefault(r["condition"], {}).setdefault(r["scenario"], {})
         .setdefault(r["image"], {})[r["prompt_index"]]) = {
            "ret": round(float(r["retained"]), 4) if r["retained"] else None,
            "purified": round(float(r["disp_purified"]), 4),
            "blocked": r["blocked"] == "True"}

    dispagg = {}
    for c in sorted(cells):
        row = {}
        for sc in ("ip2p", "inpaint"):
            if sc not in cells[c]:      # 只跑 ip2p 的臂
                row[sc] = {"full": None, "subj": None, "bg": None,
                           "blocked": 0, "n": 0}
                continue
            cs = [v for im in cells[c][sc].values() for v in im.values()]
            row[sc] = {"full": round(st.median(x["full"] for x in cs), 4),
                       "subj": round(st.median(x["subj"] for x in cs), 4),
                       "bg": round(st.median(x["bg"] for x in cs), 4),
                       "blocked": sum(1 for x in cs if x["blocked"]),
                       "n": len(cs)}
        dispagg[c] = row

    retagg = {}
    for c in sorted({r["condition"] for r in ret}):
        row = {}
        for p in PURIFIERS:
            v = [float(r["retained"]) for r in ret
                 if r["condition"] == c and r["purifier"] == p and r["retained"]]
            row[p] = round(st.median(v), 4) if v else None
        ng = [float(r["retained"]) for r in ret
              if r["condition"] == c and r["purifier"] in NONGEO and r["retained"]]
        row["nongeo"] = round(st.median(ng), 4) if ng else None
        retagg[c] = row

    # 失真彙總。`psnr_range` 與 `rms_ratio` 是同一個量的兩種寫法
    # （`PSNR 全距 = 20·log₁₀(rms 倍率)`，恆等式）；兩欄並列是為了讓
    # 「預算寫在哪個域」那一段可以直接讀，不是兩個獨立的讀數。
    fagg = {}
    for arm, per in fid.items():
        vals = list(per.values())
        fagg[arm] = {k: round(st.median(v[k] for v in vals), 4)
                     for k in ("lpips", "deltaE00", "psnr", "linf", "rms")}
        fagg[arm].update({
            "n": len(vals),
            "psnr_range": round(max(v["psnr"] for v in vals)
                                - min(v["psnr"] for v in vals), 2),
            "rms_ratio": round(max(v["rms"] for v in vals)
                               / min(v["rms"] for v in vals), 2),
            "budget": vals[0]["budget"]})

    spec = yaml.safe_load((args.data / "prompts.yaml").read_text(encoding="utf-8"))
    out = {
        "images": IMAGES, "arms": ARMS, "pending": PENDING,
        "purifiers": PURIFIERS, "nongeo": NONGEO, "anchor": ANCHOR,
        "prompts": {"ip2p": spec["edits"]["ip2p"],
                    "inpaint": spec["edits"]["inpaint"]},
        "defence": defence, "fid": fid, "fagg": fagg,
        "disp": cells, "retj": retj, "dispagg": dispagg, "retagg": retagg,
        "reference": REFERENCE,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("window.LAB=" + json.dumps(
        out, ensure_ascii=False, separators=(",", ":")) + ";", encoding="utf-8")
    print(f"{args.out} {os.path.getsize(args.out)/1000:.1f} KB  "
          f"臂 {len(ARMS)} 個，待跑 {PENDING}", flush=True)


if __name__ == "__main__":
    main()
