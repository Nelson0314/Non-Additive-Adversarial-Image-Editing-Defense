"""把探針批的讀數與影像整理成可以逐張看的三份產物。

產出
────────────────────────────────────────────────────────────────────
| 檔 | 粒度 | 內容 |
|---|---|---|
| `runs/colour_probe/summary.csv` | 取樣點 | θ 四欄、導數係數三欄、保真五欄、位移三欄、blocked 格數 |
| `runs/colour_probe/paired.csv` | 取樣點 | 對 `curve_control` 的逐格配對差中位與改善格數 |
| `runs/report/colour_probe/tiles/` | 影像 | 200 px、品質 72 的 JPEG，配 `index.json` |

為什麼配對差要逐格相減
────────────────────────────────────────────────────────────────────
兩個臂的位移都隨「哪一張圖、哪一句指令」大幅變動，直接比兩個中位數等於把
那個變異當成雜訊吞掉。以 `(影像, 指令序號)` 為鍵逐格相減之後再取中位，比較
的才是同一格上的差。改善格數同理，數的是逐格 `取樣點 − curve_control > 0`
的格數。

**門檻不在這裡判。** 本專案既有的判定門檻（配對差中位 ≥ 0.02 且 64 格裡改善
≥ 42 格）是為 ip2p 與 inpaint 合計 64 格定的，這一批只跑 ip2p 32 格，格數
那一半不適用。本腳本只把數字擺出來，不標成立與否。

`curve_control` 的欄位從哪裡來
────────────────────────────────────────────────────────────────────
它不是探針的取樣點，沒有 θ，保真讀數取自 `runs/curve_control/curve_control.csv`
（求解器自己寫的），低頻佔比與 blur 殘存取自
`runs/curve_control/curve_control_perturbation_band.csv`（事後以同一份
`src/metrics/perturbation_band.py` 在 CPU 上補算，與探針同一條量測路徑）。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

CONTROL = "curve_control"
#: 縮圖的長邊像素與 JPEG 品質。
TILE_PX = 200
TILE_QUALITY = 72
#: 看圖只取第一句指令那一格，四句全放會讓版面到不了「逐張看」的密度。
TILE_PROMPT_INDEX = "0"


def read(path: Path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def med(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    return statistics.median(v) if v else ""


def rounded(value, digits):
    return round(value, digits) if value != "" else ""


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_summary(probe, disp, control, control_band, order, paired):
    by_point = {r['point']: r for r in paired}
    rows = []
    for name in order:
        p = [r for r in probe if r["point"] == name]
        d = [r for r in disp if r["condition"] == name]
        if name == CONTROL:
            fidelity, band = control, control_band
        else:
            fidelity, band = p, p
        theta = p[0] if p else {}
        rows.append({
            "point": name,
            "t0": theta.get("t0", ""), "t1": theta.get("t1", ""),
            "t2": theta.get("t2", ""), "s": theta.get("s", ""),
            "d0": theta.get("d0", ""), "d1": theta.get("d1", ""),
            "d2": theta.get("d2", ""),
            "radius_fraction": theta.get("radius_fraction", ""),
            "images": len({r["image"] for r in fidelity}),
            "edit_cells": len(d),
            "deltaE00_median": rounded(med(fidelity, "deltaE00"), 3),
            "psnr_median": rounded(med(fidelity, "psnr"), 3),
            "lpips_median": rounded(med(fidelity, "lpips"), 5),
            "low_freq_share_median": rounded(med(band, "low_freq_share"), 5),
            "blur_retention_median": rounded(med(band, "blur_retention"), 5),
            "displacement_full_median": rounded(med(d, "disp_lpips_full"), 5),
            "displacement_subject_median":
                rounded(med(d, "disp_lpips_subject"), 5),
            "displacement_background_median":
                rounded(med(d, "disp_lpips_background"), 5),
            "blocked_cells": sum(1 for r in d
                                 if str(r.get("blocked", "")).lower() == "true"),
            # 對 curve_control 自己的配對差恆為 0，留空而不是填 0：
            # 「沒有這個量」與「差為零」是兩件事。
            "paired_diff_vs_control_median":
                by_point.get(name, {}).get("paired_diff_full_median", ""),
            "improved_cells_vs_control":
                by_point.get(name, {}).get("improved_cells_full", ""),
            "paired_total_cells":
                by_point.get(name, {}).get("total_cells", ""),
        })
    return rows


def build_paired(disp, order):
    """逐格 `取樣點 − curve_control`，鍵是 (影像, 指令序號)。"""
    def index(name):
        return {(r["image"], r["prompt_index"]): r
                for r in disp if r["condition"] == name}

    base = index(CONTROL)
    if not base:
        raise SystemExit(f"位移 CSV 裡沒有 {CONTROL}，配對差算不了")

    rows = []
    for name in order:
        if name == CONTROL:
            continue
        arm = index(name)
        keys = sorted(set(arm) & set(base))
        missing = sorted(set(base) - set(arm))
        if missing:
            # 缺格就不能只算其餘的：分母不同的兩個中位數不可比。
            raise SystemExit(f"{name} 相對 {CONTROL} 缺這些格：{missing}")
        entry = {"point": name, "cells": len(keys)}
        for label, key in (("full", "disp_lpips_full"),
                           ("subject", "disp_lpips_subject"),
                           ("background", "disp_lpips_background")):
            diffs = [float(arm[k][key]) - float(base[k][key]) for k in keys]
            entry[f"paired_diff_{label}_median"] = round(
                statistics.median(diffs), 5)
            entry[f"improved_cells_{label}"] = sum(1 for v in diffs if v > 0)
        entry["total_cells"] = len(keys)
        entry["improved_fraction_full"] = round(
            entry["improved_cells_full"] / len(keys), 4)
        rows.append(entry)
    return rows


def defended_png(name: str, image: str, runs: Path):
    if name == CONTROL:
        hits = sorted(runs.glob(
            f"curve_control/*/{image}__{CONTROL}__defended.png"))
        return hits[0] if hits else None
    path = runs / "colour_probe" / "points" / name / f"{image}__{name}__def.png"
    return path if path.is_file() else None


def build_tiles(summary, disp, probe, runs: Path, data: Path, out: Path):
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    images = sorted({r["image"] for r in probe})
    disp_cell = {(r["condition"], r["image"], r["prompt_index"]): r
                 for r in disp}
    de_cell = {(r["point"], r["image"]): r["deltaE00"] for r in probe}
    control_disp = {r["point"]: r["displacement_full_median"] for r in summary}
    # θ 逐張帶進 index.json：看圖的人要能當場對回是哪一組參數，
    # 不必再去翻 summary.csv。curve_control 沒有 θ，四欄留 None。
    theta_of = {r["point"]: {k: (float(r[k]) if r[k] != "" else None)
                             for k in ("t0", "t1", "t2", "s")}
                for r in summary}

    entries = []

    def emit(src: Path, filename: str, meta: dict):
        if src is None or not Path(src).is_file():
            raise SystemExit(f"看圖素材缺檔：{src}（{filename}）")
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((TILE_PX, TILE_PX), Image.LANCZOS)
            im.save(out / filename, "JPEG", quality=TILE_QUALITY)
        entries.append({"file": filename, **meta})

    for image in images:
        cls = image.rsplit("_", 1)[0]
        emit(data / cls / f"{image}.png", f"orig__{image}.jpg",
             {"kind": "original", "point": None, "image": image,
              "prompt_index": None, "displacement": None, "deltaE00": None,
              "theta": None})
        emit(runs / "colour_probe_preflight" / "ip2p_si18"
             / f"{image}__p{TILE_PROMPT_INDEX}.png",
             f"edit_undefended__{image}.jpg",
             {"kind": "edit_undefended", "point": None, "image": image,
              "prompt_index": TILE_PROMPT_INDEX, "displacement": None,
              "deltaE00": None, "theta": None})

    for row in summary:
        name = row["point"]
        for image in images:
            emit(defended_png(name, image, runs),
                 f"def__{name}__{image}.jpg",
                 {"kind": "defended", "point": name, "image": image,
                  "prompt_index": None,
                  "displacement": row["displacement_full_median"],
                  "deltaE00": de_cell.get((name, image)),
                  "theta": theta_of.get(name)})
            cell = disp_cell.get((name, image, TILE_PROMPT_INDEX))
            emit(runs / "colour_probe_edits" / name / "ip2p_si18"
                 / f"{image}__p{TILE_PROMPT_INDEX}.png",
                 f"edit_p{TILE_PROMPT_INDEX}__{name}__{image}.jpg",
                 {"kind": "edit_defended", "point": name, "image": image,
                  "prompt_index": TILE_PROMPT_INDEX,
                  "displacement": (float(cell["disp_lpips_full"])
                                   if cell else None),
                  "deltaE00": de_cell.get((name, image)),
                  "theta": theta_of.get(name)})

    index = {
        "tile_pixels": TILE_PX,
        "jpeg_quality": TILE_QUALITY,
        "prompt_index": TILE_PROMPT_INDEX,
        "prompt_note": "指令逐字在 data/portraits/prompts.yaml 的 edits.ip2p",
        "displacement_note": ("edit_defended 的 displacement 是該格的 "
                              "disp_lpips_full；defended 的是該取樣點 32 格的"
                              "中位，兩者粒度不同，不要混著讀"),
        "point_displacement_median": control_disp,
        "tiles": entries,
    }
    (out.parent / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data/portraits"))
    ap.add_argument("--tiles", type=Path,
                    default=Path("runs/report/colour_probe/tiles"))
    args = ap.parse_args()

    probe = read(args.runs / "colour_probe" / "colour_probe.csv")
    disp = read(args.runs / "colour_probe_edits" / "displacement.csv")
    control = read(args.runs / "curve_control" / "curve_control.csv")
    band = read(args.runs / "curve_control"
                / "curve_control_perturbation_band.csv")
    if not (probe and disp and control and band):
        raise SystemExit("四份輸入缺其一，先把批次跑完再彙整")

    order = []
    for r in probe:
        if r["point"] not in order:
            order.append(r["point"])
    order = [n for n in order if n != CONTROL] + [CONTROL]

    paired = build_paired(disp, order)
    write_csv(args.runs / "colour_probe" / "paired.csv", paired)
    summary = build_summary(probe, disp, control, band, order, paired)
    write_csv(args.runs / "colour_probe" / "summary.csv", summary)
    tiles = build_tiles(summary, disp, probe, args.runs, args.data, args.tiles)

    print(f"summary.csv {len(summary)} 列")
    print(f"paired.csv  {len(paired)} 列")
    print(f"tiles       {len(tiles)} 張 -> {args.tiles}")


if __name__ == "__main__":
    main()
