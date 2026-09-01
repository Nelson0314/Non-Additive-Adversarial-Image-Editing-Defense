"""把報告頁要的所有數字收成一份 JSON。**不跑 GPU**，只讀已存的 CSV。

三個來源，三套協定，**分開輸出不合併**：

    現行批次   runs/ip2p_{ig_loss,ig_lowdist,blur_band,split_band,sbsurv_*}
               六算子（identity/jpeg75/jpeg30/blur1/blur2/crop_resize0.1）、
               三種子、兩張影像。
    對照組批次 runs/ip2p_mainline + runs/ip2p_mainline_purify
               DCT-Shield 的 ε 掃描與本方法在**同一批**裡，十張影像、
               JPEG 軸五個算子。本檔一律**限縮到那兩張影像**再取平均。
    頭對頭批次 runs/ip2p_purify_headtohead
               八個算子（多了 gridpure／adverse_cleaner／jpeg_then_resize75），
               十三張影像。本檔一律**限縮到那兩張影像**再取平均。

位移場那一族另外收 `runs/ip2p_warp`（同失真隨機對照）與 `runs/ip2p_warp_hard`，
同樣限縮到那兩張影像。

**幾何欄的參照**：三個來源的 `crop_resize*`／`jpeg_then_resize*` 都是舊參照
（`LPIPS(編輯(原圖), 編輯(p(防禦圖)))`、地板非 0）。現行協定對幾何類改取
「同一個算子淨化過的原圖」為參照、地板由構造為 0。本檔照實輸出 `reference`
欄，不代為換算。

用法：python scripts/collect_report_data.py --out report_data.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

IMAGES = ["task_attr_mod_color_11699", "task_attr_mod_color_6205"]
IMSET = set(IMAGES)
LPIPS_CEILING = 0.772          # runs/readout_ceiling/，45 對的中位數
DEF_FIELDS = ["fid_dists", "fid_lpips", "fid_psnr", "fid_ssim", "fid_vif_p",
              "fid_rms", "fid_linf", "edit_lpips", "edit_clip_sim",
              "edit_siglip_sim"]
KEEP = ["condition", "radius", "loss", "spectral_floor", "floor_survival",
        "floor_r_max", "r_min", "r_max", "gain_ratio", "freq_weight",
        "freq_weight_power", "purify_aware", "flow_tau", "warp_grid",
        "update", "stop_reason", "stopped_at", "steps", "defense_steps",
        "eps", "dct_q_alg", "quantile", "hop", "block", "ig_zt"]


def num(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def defence_dir(d: Path, restrict: bool) -> dict | None:
    f = d / "results.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(f.open(encoding="utf-8")))
    if restrict:
        rows = [r for r in rows if r["image"] in IMSET]
    if not rows:
        return None
    rec = {"n": len(rows), "images": sorted({r["image"] for r in rows})}
    for k in DEF_FIELDS:
        v = [num(r, k) for r in rows]
        v = [x for x in v if x is not None]
        if v:
            rec[k] = round(st.mean(v), 5)
    for k in KEEP:
        vs = sorted({r.get(k, "") for r in rows if r.get(k, "") != ""})
        if vs:
            rec[k] = vs[0] if len(vs) == 1 else "|".join(vs)
    return rec


def defence_batch(root: Path, restrict: bool = True) -> dict:
    if not root.is_dir():
        return {}
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        rec = defence_dir(d, restrict)
        if rec:
            out[d.name] = rec
    return out


def _cells(path: Path, restrict: bool):
    """`{(image, purifier): effect}` 與 `{purifier: reference}`。"""
    eff, ref = {}, {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if restrict and r["image"] not in IMSET:
            continue
        v = num(r, "effect_mean")
        if v is None:
            continue
        eff[(r["image"], r["purifier"])] = v
        ref[r["purifier"]] = r.get("reference", "orig")
    return eff, ref


def net_gain(cond_cells, floor_cells, ref) -> dict:
    """兩個**絕對值**：總增益 = effect(p)，淨增益 = effect(p) − 空白地板。

    比例讀數（保留率、佔可達範圍）在分母塌陷或參照改變時不可解讀，
    故本檔不輸出（`docs/DECISIONS.md`）。
    """
    out = {}
    for p in sorted({q for _, q in cond_cells}):
        tot = [v for (i, q), v in cond_cells.items() if q == p]
        fl = [v for (i, q), v in floor_cells.items() if q == p]
        net = [v - floor_cells[(i, p)] for (i, q), v in cond_cells.items()
               if q == p and (i, p) in floor_cells]
        if not tot:
            continue
        out[p] = {
            "total": round(st.mean(tot), 4),
            "floor": round(st.mean(fl), 4) if fl else None,
            "net": round(st.mean(net), 4) if net else None,
            "n": len(tot),
            "reference": ref.get(p, "orig"),
        }
    return out


def purify_batch(root: Path, restrict: bool = True) -> dict:
    """`runs/<batch>/purify/*_all.csv` 這一種佈局。"""
    gal = root / "purify"
    if not gal.is_dir():
        return {}
    tags, refs = {}, {}
    for f in sorted(gal.glob("*_all.csv")):
        tags[f.name[: -len("_all.csv")]], refs[f.name[: -len("_all.csv")]] = \
            _cells(f, restrict)
    floor = tags.pop("floor", None)
    refs.pop("floor", None)
    if floor is None:
        return {}
    return {t: net_gain(c, floor, refs[t]) for t, c in tags.items()}


def purify_flat(root: Path, suffixes: list[str], restrict: bool = True) -> dict:
    """`runs/<batch>/<tag>_<片>.csv` 這一種佈局（mainline／headtohead）。

    片是影像的分組（`color`／`object`／`scene`）。限縮到那兩張影像之後只有
    `color` 片有列，但仍然把每一片都讀進來再過濾——**不靠檔名猜哪一片有**。
    """
    if not root.is_dir():
        return {}
    tags, refs = defaultdict(dict), {}
    for f in sorted(root.glob("*.csv")):
        stem = f.stem
        tag = None
        for s in suffixes:
            if stem.endswith("_" + s):
                tag = stem[: -len(s) - 1]
                break
        if tag is None:
            continue
        e, rf = _cells(f, restrict)
        tags[tag].update(e)
        refs.setdefault(tag, {}).update(rf)
    floor = tags.pop("floor", None)
    refs.pop("floor", None)
    if not floor:
        return {}
    return {t: net_gain(c, floor, refs.get(t, {}))
            for t, c in tags.items() if c}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("report_data.json"))
    args = ap.parse_args()

    R = Path("runs")
    data = {
        "lpips_ceiling": LPIPS_CEILING,
        "images": IMAGES,
        "phase": {},
        "warp": {},
        "baseline": {},
    }

    # ── 相位族：現行六算子協定 ──────────────────────────────────────
    for b in ["ip2p_ig_loss", "ip2p_ig_lowdist", "ip2p_blur_band",
              "ip2p_split_band", "ip2p_sbsurv_long", "ip2p_sbsurv_converged"]:
        d = defence_batch(R / b)
        if not d:
            print(f"[缺] runs/{b}")
            continue
        data["phase"][b] = {"defence": d, "purify": purify_batch(R / b)}

    # ── 位移場族 ────────────────────────────────────────────────────
    for b in ["ip2p_stadv", "ip2p_stadv_radius", "ip2p_warp",
              "ip2p_warp_hard"]:
        d = defence_batch(R / b)
        if not d:
            print(f"[缺] runs/{b}")
            continue
        data["warp"][b] = {"defence": d, "purify": purify_batch(R / b)}

    # ── 對照組：DCT-Shield ──────────────────────────────────────────
    data["baseline"]["ip2p_mainline"] = {
        "defence": defence_batch(R / "ip2p_mainline"),
        "purify": purify_flat(R / "ip2p_mainline_purify",
                              ["color", "object", "scene"]),
    }
    # 新批次：三組等失真配對在**現行六算子協定**上（含模糊與裁切），
    # 而且留了 `--gallery`。佈局與 `runs/<batch>/purify/*_all.csv` 不同，
    # CSV 直接放在批次目錄底下。
    mh = R / "ip2p_matched_headtohead"
    if mh.is_dir():
        tags, refs = {}, {}
        for f in sorted(mh.glob("*_all.csv")):
            t = f.name[: -len("_all.csv")]
            tags[t], refs[t] = _cells(f, True)
        floor = tags.pop("floor", None)
        refs.pop("floor", None)
        if floor:
            data["baseline"]["ip2p_matched_headtohead"] = {
                "defence": {},
                "purify": {t: net_gain(c, floor, refs.get(t, {}))
                           for t, c in tags.items() if c},
            }

    data["baseline"]["ip2p_purify_headtohead"] = {
        "defence": {},
        "purify": purify_flat(R / "ip2p_purify_headtohead",
                              ["color", "object", "scene"]),
    }

    for fam in ("phase", "warp", "baseline"):
        for b, v in data[fam].items():
            print(f"{fam:<9s} {b:<26s} 防禦 {len(v['defence']):>2d}  "
                  f"抗淨化 {len(v['purify']):>2d}")

    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"寫出 {args.out}（{args.out.stat().st_size/1000:.0f} KB）")


if __name__ == "__main__":
    main()
