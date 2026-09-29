"""產生主表報告(原生預算、12 條件)的 `data.js` 與縮圖。

與 `report_matrix_data.py`(等失真臂、11 條件)是兩份不同的資料，理由與協定
差異見 `main_table/README.md` 的「三組資料的關係」。這一支對應
`results/displacement.csv`、`results/defence_<條件>.csv`、`results/retention.csv`
三張主讀數 CSV，外加 `metrics_*_union.csv` 四張(含新加的 `metrics_vmaf_union.csv`)，以及
FLUX 與 UltraEdit（SD3）兩個跨編輯器的獨立一節。

縮圖直接從本機 `images/` 讀,轉 WebP 存進 `<報告目錄>/img/`(不經過遠端)。

用法:
    python code/report_main_data.py --out report/main
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

CONDITIONS = ["dct_shield_y", "mist", "dct_shield", "photoguard_linf", "danp",
              "sifm", "dayn", "dia_pt", "dia_r", "photoguard_c",
              "color", "diffvax"]

NAMES = ["man_00", "man_01", "woman_00", "woman_01"]
CELLS = [("ip2p", "0"), ("ip2p", "1"), ("inpaint", "1"), ("inpaint", "3")]
PURIFIERS = ["blur1", "blur2", "crop_resize0.1", "jpeg30", "jpeg50", "jpeg80", "rotate15"]

METRICS = [
    ("disp_lpips_full", "LPIPS", "up", 4),
    ("disp_lpips_subject", "LPIPS 主體", "up", 4),
    ("disp_lpips_background", "LPIPS 背景", "up", 4),
    ("disp_dists", "DISTS", "up", 4),
    ("disp_fsim", "FSIM", "down", 4),
    ("disp_rms", "rms", "up", 4),
    ("disp_linf", "L inf", "up", 4),
    ("disp_ssim", "SSIM", "down", 4),
    ("disp_psnr", "PSNR", "down", 2),
    ("disp_vif_p", "VIFp", "down", 4),
    ("disp_vmaf", "VMAF", "down", 2),
    ("clip_pair", "CLIP", "down", 4),
    ("siglip_pair", "SigLIP", "down", 4),
]

#: FLUX 全表沒有 disp_fsim／disp_vmaf(還沒算),多一欄 id_orig(FaceNet，
#: FLUX 自己讀數才有的東西，ip2p/inpaint 那張表沒有可比欄位)。
FLUX_METRICS = [
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

#: UltraEdit 全表：同一組位移欄位，另有 id_orig 與淨化後保留率。
ULTRA_PROMPTS = ["0", "1", "2", "3"]
ULTRA_STRIP_COLS = ["undefended"] + CONDITIONS
ULTRA_THUMB = 208

AESTHETIC = [
    ("aes_laion", "LAION Aesthetic", "up"),
    ("aes_nima", "NIMA", "up"),
    ("aes_musiq_ava", "MUSIQ-AVA", "up"),
    ("aes_topiq_iaa", "TOPIQ-IAA", "up"),
    ("aes_clipiqa", "CLIP-IQA", "up"),
    ("aes_niqe", "NIQE", "down"),
    ("aes_brisque", "BRISQUE", "down"),
]


def load(rel: str) -> list:
    with (paths.RESULTS / rel).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(rows, key, dp=5):
    vals = [float(r[key]) for r in rows if r.get(key, "") not in ("", None)]
    return round(st.mean(vals), dp) if vals else None


def build_data() -> dict:
    disp = load("displacement.csv")
    ret = load("retention.csv")
    fid_union = load("metrics_fidelity_union.csv")
    disp_union = load("metrics_displacement_union.csv")
    ret_union = load("metrics_retention_union.csv")
    vmaf = load("metrics_vmaf_union.csv")
    aes = load("metrics_aesthetic_union.csv")

    vmaf_fid = {(r["condition"], r["image"]): float(r["vmaf"])
                for r in vmaf if r["pairing"] == "fidelity"}
    vmaf_disp = {}
    for r in vmaf:
        if r["pairing"] == "displacement":
            vmaf_disp.setdefault(r["condition"], []).append(float(r["vmaf"]))
    vmaf_ret = {}
    for r in vmaf:
        if r["pairing"] == "retention":
            vmaf_ret.setdefault((r["condition"], r["purifier"]), []).append(float(r["vmaf"]))

    fsim_fid = {(r["condition"], r["image"]): float(r["fid_fsim"]) for r in fid_union}
    deltae_fid = {}
    for r in fid_union:
        deltae_fid.setdefault(r["condition"], []).append(float(r["fid_delta_e00"]))
    fsim_disp = {}
    for r in disp_union:
        fsim_disp.setdefault(r["condition"], []).append(float(r["disp_fsim"]))
    fsim_ret = {}
    for r in ret_union:
        fsim_ret.setdefault(r["condition"], []).append(float(r["disp_purified_fsim"]))

    def agg_disp(cond):
        sel = [r for r in disp if r["condition"] == cond]
        out = {k: mean(sel, k) for k, _, _, _ in METRICS if k != "disp_fsim" and k != "disp_vmaf"}
        out["disp_fsim"] = round(st.mean(fsim_disp.get(cond, [0])), 5) if fsim_disp.get(cond) else None
        out["disp_vmaf"] = round(st.mean(vmaf_disp.get(cond, [0])), 3) if vmaf_disp.get(cond) else None
        out["blocked"] = sum(1 for r in sel if r["blocked"] == "True")
        out["cells"] = len(sel)
        return out

    def retained(cond, purifier=None):
        v = [float(r["retained"]) for r in ret
             if r["condition"] == cond and r["retained"]
             and (purifier is None or r["purifier"] == purifier)]
        return round(st.mean(v), 4) if v else None

    methods = []
    for cond in CONDITIONS:
        solver = load(f"defence_{cond}.csv")
        m = agg_disp(cond)
        m.update({
            "name": cond,
            "fid_lpips": mean(solver, "fid_lpips", 4),
            "fid_psnr": mean(solver, "fid_psnr", 2),
            "fid_rms": mean(solver, "fid_rms", 5),
            "fid_linf": mean(solver, "fid_linf", 5),
            "fid_ssim": mean(solver, "fid_ssim", 4),
            "fid_vif_p": mean(solver, "fid_vif_p", 4),
            "fid_dists": mean(solver, "fid_dists", 4),
            "fid_fsim": round(st.mean(v for (c, i), v in fsim_fid.items() if c == cond), 5),
            "fid_delta_e00": round(st.mean(deltae_fid.get(cond, [0])), 3) if deltae_fid.get(cond) else None,
            "fid_vmaf": round(st.mean(v for (c, i), v in vmaf_fid.items() if c == cond), 3),
            "ret": retained(cond),
            "ret_p": {p: retained(cond, p) for p in PURIFIERS},
            "ret_fsim": round(st.mean(fsim_ret.get(cond, [0])), 5) if fsim_ret.get(cond) else None,
            "ret_vmaf": round(st.mean(v for p in PURIFIERS for v in vmaf_ret.get((cond, p), [])), 3)
                        if any((cond, p) in vmaf_ret for p in PURIFIERS) else None,
            "ret_vmaf_p": {p: (round(st.mean(vmaf_ret[(cond, p)]), 3) if (cond, p) in vmaf_ret else None)
                          for p in PURIFIERS},
            "fid_by_image": {r["image"]: round(float(r["fid_lpips"]), 4)
                             for r in solver if r["image"] in NAMES},
        })
        methods.append(m)

    cellmap = {(r["condition"], r["scenario"], r["image"], r["prompt_index"]): r for r in disp}
    cells = []
    for scenario, index in CELLS:
        prompt = next(r["prompt"] for r in disp
                      if r["scenario"] == scenario and r["prompt_index"] == index)
        for name in NAMES:
            per = {}
            for m in methods:
                r = cellmap.get((m["name"], scenario, name, index))
                if r:
                    per[m["name"]] = round(float(r["disp_lpips_full"]), 4)
            cells.append({"scenario": scenario, "pi": index, "prompt": prompt,
                          "image": name, "per": per})

    order = [m["name"] for m in sorted(methods, key=lambda x: -(x["disp_lpips_full"] or 0))]
    median_keys = [k for k, _, _, _ in METRICS] + [
        "fid_lpips", "fid_psnr", "fid_rms", "fid_linf", "fid_ssim",
        "fid_vif_p", "fid_dists", "fid_fsim", "fid_delta_e00", "fid_vmaf",
        "ret", "ret_fsim", "ret_vmaf"]
    median = {k: round(st.median([m[k] for m in methods if m[k] is not None]), 5)
              for k in median_keys if any(m[k] is not None for m in methods)}

    # ---- 美術指標:真圖逐條件排序 ----
    aes_by_image = {}
    for name in NAMES:
        per_metric = {}
        rows = [r for r in aes if r["image"] == name
                and r["condition"] in CONDITIONS + ["original"]]
        for key, label, direction in AESTHETIC:
            entries = [{"condition": r["condition"], "value": round(float(r[key]), 4)}
                      for r in rows if r.get(key, "") != ""]
            entries.sort(key=lambda e: e["value"], reverse=(direction == "up"))
            per_metric[key] = entries
        aes_by_image[name] = per_metric

    return {
        "metrics": [{"key": k, "label": l, "dir": d, "dp": p} for k, l, d, p in METRICS],
        "methods": methods,
        "cells": cells,
        "names": NAMES,
        "purifiers": PURIFIERS,
        "order": order,
        "median": median,
        "aesthetics": {
            "metrics": [{"key": k, "label": l, "dir": d} for k, l, d in AESTHETIC],
            "images": NAMES,
            "rows": aes_by_image,
        },
        "flux": build_flux(),
        "ultra": build_ultraedit(),
        "xeditor": build_xeditor(),
    }


def build_flux() -> dict:
    """FLUX 全表(`edit_flux_preview.py --arm`)的讀數，獨立一節。

    協定跟 ip2p/inpaint 不同(guidance 3.5、1024×1024、無 strength)，數字
    不放進同一張聚合表——理由與「三組資料的關係」同一套：協定不同的位移
    不能直接比大小，見 `main_table/README.md`。
    """
    disp = load("displacement_flux.csv")

    def agg(cond):
        sel = [r for r in disp if r["condition"] == cond]
        out = {k: mean(sel, k) for k, _, _, _ in FLUX_METRICS}
        idr = load(f"flux_full_{cond}.csv")
        out["id_orig"] = mean(idr, "id_orig", 4)
        out["blocked"] = sum(1 for r in sel if r["blocked"] == "True")
        out["cells"] = len(sel)
        out["name"] = cond
        return out

    methods = [agg(c) for c in CONDITIONS]
    order = [m["name"] for m in sorted(methods, key=lambda x: -(x["disp_lpips_full"] or 0))]
    median_keys = [k for k, _, _, _ in FLUX_METRICS] + ["id_orig"]
    median = {k: round(st.median([m[k] for m in methods if m[k] is not None]), 5)
              for k in median_keys if any(m[k] is not None for m in methods)}
    return {
        "metrics": [{"key": k, "label": l, "dir": d, "dp": p} for k, l, d, p in FLUX_METRICS],
        "methods": methods,
        "order": order,
        "median": median,
    }


def build_ultraedit() -> dict:
    """UltraEdit（SD3）全表（`edit_ultraedit_full.py`）的讀數，獨立一節。

    協定：`add` 句型、guidance 2.5、image guidance 1.5、512×512、50 步，與
    ip2p/inpaint/FLUX 都不同，數字不放進同一張聚合表。位移與保留率由
    `edit_displacement.py`／`edit_retention.py` 原樣算出，欄位與主表同一組。
    """
    disp = load("displacement_ultraedit.csv")
    ret = load("retention_ultraedit.csv")
    edits = {c: load(f"ultraedit_full/{c}.csv") for c in ["undefended"] + CONDITIONS}

    def retained(cond, purifier=None):
        v = [float(r["retained"]) for r in ret
             if r["condition"] == cond and r["retained"]
             and (purifier is None or r["purifier"] == purifier)]
        return round(st.mean(v), 4) if v else None

    def agg(cond):
        sel = [r for r in disp if r["condition"] == cond]
        out = {k: mean(sel, k) for k, _, _, _ in FLUX_METRICS}
        plain = [r for r in edits[cond] if r["purifier"] == "none"]
        out["id_orig"] = mean(plain, "id_orig", 4)
        out["blocked"] = sum(1 for r in sel if r["blocked"] == "True")
        out["cells"] = len(sel)
        out["ret"] = retained(cond)
        out["ret_p"] = {p: retained(cond, p) for p in PURIFIERS}
        out["name"] = cond
        return out

    methods = [agg(c) for c in CONDITIONS]
    base = [r for r in edits["undefended"] if r["purifier"] == "none"]
    order = [m["name"] for m in sorted(methods, key=lambda x: -(x["disp_lpips_full"] or 0))]
    median_keys = [k for k, _, _, _ in FLUX_METRICS] + ["id_orig", "ret"]
    median = {k: round(st.median([m[k] for m in methods if m[k] is not None]), 5)
              for k in median_keys if any(m[k] is not None for m in methods)}
    prompts = {r["prompt_index"]: r["prompt"] for r in base}
    return {
        "metrics": [{"key": k, "label": l, "dir": d, "dp": p} for k, l, d, p in FLUX_METRICS],
        "methods": methods,
        "order": order,
        "median": median,
        "undefended_id_orig": mean(base, "id_orig", 4),
        "prompts": [prompts[i] for i in ULTRA_PROMPTS],
        "prompt_index": ULTRA_PROMPTS,
        "strip_cols": ULTRA_STRIP_COLS,
    }


def build_xeditor() -> list:
    """同一方法在四個攻擊模型上的位移（LPIPS，全圖）與編輯後身分，逐條件一列。

    四欄的協定各不相同（解析度、步數、guidance），並列只供對照，不是同一把尺。
    """
    def disp_mean(rows, cond, scenario=None):
        v = [float(r["disp_lpips_full"]) for r in rows if r["condition"] == cond
             and (scenario is None or r["scenario"] == scenario)]
        return round(st.mean(v), 4) if v else None

    main = load("displacement.csv")
    flux = load("displacement_flux.csv")
    ultra = load("displacement_ultraedit.csv")
    out = []
    for cond in CONDITIONS:
        fl = load(f"flux_full_{cond}.csv")
        ue = [r for r in load(f"ultraedit_full/{cond}.csv") if r["purifier"] == "none"]
        out.append({
            "name": cond,
            "fid_lpips": mean(load(f"defence_{cond}.csv"), "fid_lpips", 4),
            "ip2p": disp_mean(main, cond, "ip2p"),
            "inpaint": disp_mean(main, cond, "inpaint"),
            "flux": disp_mean(flux, cond),
            "ultra": disp_mean(ultra, cond),
            "flux_id": mean(fl, "id_orig", 4),
            "ultra_id": mean(ue, "id_orig", 4),
        })
    out.sort(key=lambda m: -(m["ip2p"] or 0))
    base_fl = load("flux_full_undefended.csv")
    base_ue = [r for r in load("ultraedit_full/undefended.csv") if r["purifier"] == "none"]
    out.append({"name": "undefended", "fid_lpips": None, "ip2p": None, "inpaint": None,
                "flux": None, "ultra": None, "flux_id": mean(base_fl, "id_orig", 4),
                "ultra_id": mean(base_ue, "id_orig", 4)})
    return out


def to_webp(src: Path, dst: Path, size: int = 220) -> None:
    from PIL import Image
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(dst, "WEBP", quality=82, method=6)


def build_images(out: Path) -> None:
    img_dir = out / "img"
    root = paths.IMAGES
    n = 0

    def defended_png(condition: str, name: str) -> Path:
        d = root / "defence_portraits" / condition
        matches = sorted(d.glob(f"{name}__{condition}__def.png"))
        if not matches:
            raise SystemExit(f"缺防禦圖: {d}/{name}__{condition}__def.png")
        return matches[0]

    def edit_png(condition: str, scenario: str, name: str, pi: str) -> Path:
        if condition == "undefended":
            arm = "ip2p_si18" if scenario == "ip2p" else "inpaint_undefended"
            return root / "edit_preflight" / arm / f"{name}__p{pi}.png"
        arm = f"{scenario}_{condition}"
        return root / "edit_defended" / condition / arm / f"{name}__p{pi}.png"

    for name in NAMES:
        orig = root / "defence_portraits" / "mist" / f"{name}__orig.png"
        to_webp(orig, img_dir / f"orig_{name}.webp")
        to_webp(root / "masks" / f"{name}.png", img_dir / f"mask_{name}.webp")
        n += 2
        for cond in CONDITIONS:
            to_webp(defended_png(cond, name), img_dir / f"def_{cond}_{name}.webp")
            n += 1

    for scenario, pi in CELLS:
        for name in NAMES:
            for cond in ["undefended"] + CONDITIONS:
                src = edit_png(cond, scenario, name, pi)
                if not src.is_file():
                    raise SystemExit(f"缺編輯圖: {src}")
                to_webp(src, img_dir / f"e_{scenario}{pi}_{cond}_{name}.webp")
                n += 1

    # FLUX 全表:重用 report/flux_full/img/ 已經轉好的縮圖(同一批來源，
    # 已經是 webp)，只搬 p0 那組，加 flux_ 前綴避免跟上面的檔名混在一起。
    # 那份報告之後才加進來的條件(color)沒有現成縮圖，改從 images/flux_full/ 轉。
    import shutil
    flux_src = paths.BASELINES / "report" / "flux_full" / "img"
    for name in NAMES:
        for cond in ["undefended"] + CONDITIONS:
            src = flux_src / f"{cond}_{name}_p0.webp"
            png = root / "flux_full" / cond / f"{name}__p0.png"
            if src.is_file():
                shutil.copy(src, img_dir / f"flux_{cond}_{name}_p0.webp")
            elif png.is_file():
                to_webp(png, img_dir / f"flux_{cond}_{name}_p0.webp")
            else:
                raise SystemExit(f"缺 FLUX 縮圖與原圖: {src}、{png}")
            n += 1
    # UltraEdit 全表：從本機 images/ultraedit_full/ 轉（遠端產物拉回來的同一版面）。
    # 每列（影像 × 指令）拼成一張橫條，欄序為 ULTRA_STRIP_COLS，頁面以 CSS 位移取格：
    # 已發布的 artifact 每版上限 511 個檔，逐格一檔會超過。
    from PIL import Image
    ultra = paths.IMAGES / "ultraedit_full"
    for name in NAMES:
        for pi in ULTRA_PROMPTS:
            strip = Image.new("RGB", (ULTRA_THUMB * len(ULTRA_STRIP_COLS), ULTRA_THUMB))
            for k, cond in enumerate(ULTRA_STRIP_COLS):
                src = (ultra / "edit_preflight" / "ultraedit_undefended" if cond == "undefended"
                       else ultra / "edit_defended" / cond / f"ultraedit_{cond}") / f"{name}__p{pi}.png"
                if not src.is_file():
                    raise SystemExit(f"缺 UltraEdit 編輯圖: {src}")
                with Image.open(src) as im:
                    strip.paste(im.convert("RGB").resize((ULTRA_THUMB, ULTRA_THUMB), Image.LANCZOS),
                                (k * ULTRA_THUMB, 0))
            strip.save(img_dir / f"ultra_{name}_p{pi}.webp", "WEBP", quality=82, method=6)
            n += 1
    print(f"[IMAGES] {n} 張 -> {img_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()
    data = build_data()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data.js").write_text(
        "window.MAIN = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"[DATA] {len(data['methods'])} 條件、{len(data['cells'])} 格 -> {args.out / 'data.js'}",
          flush=True)
    if not args.skip_images:
        build_images(args.out)


if __name__ == "__main__":
    main()
