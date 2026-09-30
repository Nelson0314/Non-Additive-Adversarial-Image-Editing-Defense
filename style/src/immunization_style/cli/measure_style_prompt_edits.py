"""風格載體防禦的攻擊端讀數。

每格三張編輯：U = edit(x)（未防禦分母）、E_ref = edit(x_ref)、E_def = edit(x_def)。欄位（皆分全圖／主體／背景，
與 `immunization_core.pipelines.displacement` 同一個 LPIPS 與遮罩）：

    disp_lpips_*                  編輯結果 LPIPS：LPIPS(U, E)
    disp_reference_lpips_*        對風格參照編輯的編輯結果 LPIPS：LPIPS(E_ref, E_def)，只量最佳化在風格之上多出的部分
    edit_change_lpips_*           編輯前後改變量：LPIPS(輸入, E)
    edit_change_undefended_lpips_* 未防禦的編輯前後改變量：LPIPS(x, U)

未防禦分母 U 取自 `--undefended-edits-dir`（預設 `artifacts/undefended_edits/ip2p_si18`）。
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import piq
import torch

from immunization_core.io import load_image_tensor, write_sorted_csv
from immunization_core.metrics.regional import RegionalLPIPS, split_displacement
from immunization_core.pipelines.masks import subject_mask
from immunization_style import layout

RESOLUTION = 512


def edits(directory: Path, images=None):
    out = {}
    for path in sorted(directory.rglob("preflight.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["scenario"] == "ip2p" and (not images or r["image"] in images):
                    out[(r["image"], r["prompt_index"])] = (
                        path.parent / r["arm"] / f"{r['image']}__p{r['prompt_index']}.png", r["prompt"],
                        r["input_png"])
    return out


def read_groups(root, ref_root, styles, strengths, images=None):
    """先核對每個指定組別，避免空讀數覆寫既有 CSV。"""
    groups = []
    for style in styles:
        ref_dir = ref_root / f"ref_{style}"
        ref = edits(ref_dir, images)
        if not ref:
            raise SystemExit(f"沒有符合的參照編輯格：{ref_dir}；images={sorted(images or [])}")
        for strength in ["ref", *strengths]:
            directory = ref_dir if strength == "ref" else root / f"{strength}_{style}"
            cur = ref if strength == "ref" else edits(directory, images)
            if not cur:
                raise SystemExit(f"沒有符合的編輯格：{directory}；images={sorted(images or [])}")
            missing = sorted(set(cur) - set(ref))
            if missing:
                raise SystemExit(f"{ref_dir} 缺少 {directory} 的參照格：{missing}")
            groups.append((style, strength, ref, cur))
    if not groups:
        raise SystemExit(f"沒有指定的讀數組別：{root}")
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edits-root", dest="edits", type=Path, required=True)
    ap.add_argument("--styles", nargs="+", required=True)
    ap.add_argument("--strengths", nargs="+", default=["capped", "uncapped"])
    ap.add_argument("--undefended-edits-dir", dest="undefended", type=Path, default=layout.UNDEFENDED_EDITS / "ip2p_si18")
    ap.add_argument("--data-root", dest="data", type=Path, default=layout.PORTRAITS)
    ap.add_argument("--output-csv", dest="out", type=Path, required=True)
    ap.add_argument("--reference-edits-root", dest="ref_edits", type=Path, default=None, help="ref_<style> 所在的根目錄，預設同 --edits-root")
    ap.add_argument("--images", nargs="+", default=None)
    args = ap.parse_args()
    images = set(args.images) if args.images else None
    groups = read_groups(args.edits, args.ref_edits or args.edits,
                         args.styles, args.strengths, images)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regional = RegionalLPIPS(piq.LPIPS().to(device))

    def load(p):
        if not Path(p).is_file():
            raise SystemExit(f"找不到 {p}")
        return load_image_tensor(Path(p), device, size=RESOLUTION)

    rows = []
    for style, strength, ref, cur in groups:
        for (name, k), (path, prompt, inp) in sorted(cur.items()):
            mask = subject_mask(load(args.data / "masks" / f"{name}.png")[:, :1])
            e = load(path)
            with torch.no_grad():
                u = load(args.undefended / f"{name}__p{k}.png")
                d = split_displacement(regional, u, e, mask)
                c = split_displacement(regional, load(inp), e, mask)
                c0 = split_displacement(regional, load(args.data / name.split("_")[0] / f"{name}.png"), u, mask)
                row = {"style": style, "strength": strength, "image": name, "prompt_index": k,
                       "prompt": prompt, **{f"disp_{a}": round(float(v), 5) for a, v in d.items()},
                       **{f"edit_change_{a}": round(float(v), 5) for a, v in c.items()},
                       **{f"edit_change_undefended_{a}": round(float(v), 5) for a, v in c0.items()}}
                if strength != "ref":
                    p = split_displacement(regional, load(ref[(name, k)][0]), e, mask)
                    row.update({f"disp_reference_{a}": round(float(v), 5) for a, v in p.items()})
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_sorted_csv(args.out, rows)

    keys = [k for k in rows[0] if k.startswith(("disp_", "edit_change_")) and not k.startswith("disp_reference_")]
    keys += [k for k in rows[-1] if k.startswith("disp_reference_")]
    print(f"{'style':<11}{'strength':<10}" + "".join(f"{k:>34}" for k in keys))
    for style in args.styles:
        for strength in ["ref", *args.strengths]:
            sel = [r for r in rows if r["style"] == style and r["strength"] == strength]
            if not sel:
                continue
            print(f"{style:<11}{strength:<16}n={len(sel):<3}" + "".join(
                f"{statistics.mean(r[k] for r in sel):>34.4f}" if k in sel[0] else f"{'':>34}"
                for k in keys))
    print(f"{len(rows)} 列 -> {args.out}")


if __name__ == "__main__":
    main()
