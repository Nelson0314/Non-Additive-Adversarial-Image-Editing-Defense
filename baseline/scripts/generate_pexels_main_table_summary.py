"""Pexels 人像 306 張的主表彙整：等失真九個條件與 color3d_pexels，每條件一列，ip2p、主種子。

來源（<根> 為 image_sets.json 的 root 欄）：
    等失真臂  runtime/pexels_portraits/aligned/<根>/results/：fidelity.csv、aesthetic.csv（photoguard_encoder 為 results/photoguard_encoder/）、
              displacement_<條件>.csv、retention_<條件>.csv
    color3d   color trial results/color3d_pexels/{displacement,retention}.csv；防禦圖讀數在 followup/<根>/results/color3d/
    後續量測  runtime/pexels_portraits/followup/<根>/results/<aligned|color3d>/：retention_vae_trans_<條件>.csv、semantic_identity{,_vae_trans}.csv、
              arcface_pair.csv、survival_{seven,vae_trans}.csv（color3d 七道為 followup/results/color3d_all/survival_seven.csv）；
              followup/<根>/results/clip_score.csv
    分布指標  runtime/pexels_portraits/summary/distribution_metrics/pexels_distribution.csv

主表 main_table.csv 的欄：
    defense_lpips、defense_ssim、defense_psnr：防禦圖對原圖（fidelity.csv 的 fid_lpips、fid_ssim、fid_psnr），交付 PNG、CPU
    aes_laion、aes_musiq_ava：防禦圖美學
    disp_none、disp_seven、disp_worst、disp_vae_trans：編輯結果 LPIPS（未淨化、七道平均、每格取七道最小值後平均、VAE-Trans）
    disp_weakest、disp_weakest_purifier：逐道先對全部格平均，取平均最低的一道（論文 Weakest）；逐道平均見 per_purifier_displacement.csv
    retained_seven、retained_seven_ratio、retained_vae_trans：逐格 disp_purified/disp_plain 的平均、七道平均之比、VAE-Trans 逐格平均
    siglip_seven、siglip_vae_trans：兩張編輯的 SigLIP 影像餘弦（siglip_pair）
    clip_i_seven、clip_i_vae_trans：兩張編輯的 CLIP 影像餘弦（clip_pair，論文 CLIP-I）
    clip_s_none、clip_s_seven、clip_s_vae_trans：cos(E_img(編輯), E_txt(指令))，CLIP（clip_score.csv 的 clip_s，論文 CLIP-S）
    arcface_original_seven：ArcFace(原圖, 編輯結果)；arcface_pair_seven、arcface_pair_vae_trans：ArcFace(E_u, E_d)；只計有臉的格，*_missing 為無臉格數
    survival_seven、survival_vae_trans：LPIPS(P(x̃),P(x))/LPIPS(x̃,x) 的逐張平均
    fid_none、fid_seven、fid_vae_trans、kid_*：編輯端（seven 為七組的平均）；kid_negative_groups 為 KID 負值的組數
    <欄>_std：影像間樣本標準差。逐張值為該張所有格的平均（4 句；七道欄為 4 句 × 7 道；disp_weakest_std 只取 disp_weakest_purifier
              那一道；disp_worst_std 為每格七道最小值的逐張平均；ArcFace 只計有臉的格，無臉的影像不計）。平均欄為逐格平均，
              格數齊全時與逐張值的平均相同。分布指標與 retained_seven_ratio 沒有逐張值，無 _std。
補充表 main_table_supplementary.csv：其餘五項防禦圖美學（aes_nima、aes_topiq_iaa、aes_clipiqa、aes_niqe、aes_brisque）與其 _std、
    fid_distortion、kid_distortion（防禦圖對原圖），以及 clipiqa_defense、clipiqa_none、clipiqa_seven、clipiqa_vae_trans（A 集合的 CLIP-IQA）。
兩張表的末列 original 為原圖美學。
任一條件的影像數不是 306 或格數不齊時失敗。
`--without-vae-trans` 時不讀 VAE-Trans 與 CLIP-S 的來源，這些欄留空，其餘檢查不變。

輸出：runtime/pexels_portraits/summary/{main_table.csv,main_table_supplementary.csv,per_purifier_displacement.csv,distribution_groups.csv,README.md}
用法（遠端 baseline 專案根）：python scripts/generate_pexels_main_table_summary.py [--without-vae-trans]
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

N_IMAGES = 306
SEVEN = ("blur1", "blur2", "crop_resize0.1", "jpeg30", "jpeg50", "jpeg80", "rotate15")
ALIGNED = ("mist", "dia_pt", "sifm", "dia_r", "dct_shield_y", "dct_shield", "dayn", "danp", "photoguard_encoder")
COLOR3D = "color3d_pexels"
FIDELITY = ("lpips", "ssim", "psnr")
AESTHETIC_MAIN = ("aes_laion", "aes_musiq_ava")
AESTHETIC_SUPPLEMENTARY = ("aes_nima", "aes_topiq_iaa", "aes_clipiqa", "aes_niqe", "aes_brisque")
AESTHETIC = AESTHETIC_MAIN + AESTHETIC_SUPPLEMENTARY
CT = Path("../color/trials/lut3d_defense_and_naturalness")
OUT = Path("runtime/pexels_portraits/summary")


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values):
    values = list(values)
    if not values:
        raise SystemExit("空集合無法平均")
    return sum(values) / len(values)


def per_image_std(cells):
    """cells：{(影像, …): 值}；逐張平均後取影像間樣本標準差。"""
    by_image = defaultdict(list)
    for key, value in cells.items():
        by_image[key[0]].append(value)
    return statistics.stdev(mean(v) for v in by_image.values())


def require(condition, label, got, want):
    if got != want:
        raise SystemExit(f"{condition} 的 {label} 為 {got}，應為 {want}")


def faces(cells):
    return {k: v for k, v in cells.items() if v is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--without-vae-trans", action="store_true")
    vae = not ap.parse_args().without_vae_trans
    sets = json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
    conditions = ALIGNED + (COLOR3D,)
    fidelity = {c: {} for c in conditions}            # 影像 -> {lpips, ssim, psnr}
    aesthetic = {c: {} for c in conditions}           # 影像 -> 列
    displacement = {c: {} for c in conditions}        # (影像, 句) -> LPIPS
    retention = {c: defaultdict(dict) for c in conditions}   # (影像, 句) -> 道 -> (disp_purified, retained)
    vae_retention = {c: {} for c in conditions}       # (影像, 句) -> (disp_purified, retained)
    pair = defaultdict(dict)        # (指標, 條件, seven|vae_trans) -> (影像, 句, 道) -> 值；指標 siglip、clip
    arc_original = defaultdict(dict)  # 條件 -> (影像, 句, 道) -> 值或 None
    arc_pair = defaultdict(dict)      # (條件, kind) -> (影像, 句, 道) -> 值或 None
    survival = defaultdict(dict)      # (條件, kind) -> (影像, 道) -> 值
    clip_s = defaultdict(dict)        # (條件, none|seven|vae_trans) -> (影像, 句, 道) -> 值
    originals = {}
    for row in read(CT / "results/color3d_pexels/displacement.csv"):
        displacement[COLOR3D][(row["image"], row["prompt_index"])] = float(row["disp_lpips_full"])
    for row in read(CT / "results/color3d_pexels/retention.csv"):
        retention[COLOR3D][(row["image"], row["prompt_index"])][row["purifier"]] = (float(row["disp_purified"]), float(row["retained"]))
    for row in read("runtime/pexels_portraits/followup/results/color3d_all/survival_seven.csv"):
        survival[(COLOR3D, "seven")][(row["image"], row["purifier"])] = float(row["survival"])

    def kind_of(purifier):
        return "seven" if purifier in SEVEN else purifier

    for s in sets:
        sub = Path(s["root"]) if s["root"] else Path()
        R = Path("runtime/pexels_portraits/aligned") / sub / "results"
        F = Path("runtime/pexels_portraits/followup") / sub / "results"
        for c in ALIGNED:
            directory = R / c if c == "photoguard_encoder" else R
            for row in read(directory / "fidelity.csv"):
                if row["condition"] == c:
                    fidelity[c][row["image"]] = {k: float(row[f"fid_{k}"]) for k in FIDELITY}
            for row in read(directory / "aesthetic.csv"):
                if row["condition"] == c:
                    aesthetic[c][row["image"]] = row
                elif row["condition"] == "original":
                    originals[row["image"]] = row
            for row in read(R / f"displacement_{c}.csv"):
                if row["scenario"] == "ip2p":
                    displacement[c][(row["image"], row["prompt_index"])] = float(row["disp_lpips_full"])
            for row in read(R / f"retention_{c}.csv"):
                if row["scenario"] == "ip2p" and row["purifier"] in SEVEN:
                    retention[c][(row["image"], row["prompt_index"])][row["purifier"]] = (float(row["disp_purified"]), float(row["retained"]))
        for row in read(F / "color3d/fidelity.csv"):
            if row["condition"] == COLOR3D:
                fidelity[COLOR3D][row["image"]] = {k: float(row[f"fid_{k}"]) for k in FIDELITY}
        for row in read(F / "color3d/aesthetic.csv"):
            if row["condition"] == COLOR3D:
                aesthetic[COLOR3D][row["image"]] = row
        for group, members in (("aligned", ALIGNED), ("color3d", (COLOR3D,))):
            for c in members if vae else ():
                for row in read(F / group / f"retention_vae_trans_{c}.csv"):
                    if row["scenario"] == "ip2p":
                        vae_retention[c][(row["image"], row["prompt_index"])] = (float(row["disp_purified"]), float(row["retained"]))
            for name in ("semantic_identity.csv", "semantic_identity_vae_trans.csv") if vae else ("semantic_identity.csv",):
                for row in read(F / group / name):
                    c, kind = row["condition"], kind_of(row["purifier"])
                    key = (row["image"], row["prompt_index"], row["purifier"])
                    if c == "undefended" or kind not in ("seven", "vae_trans"):
                        continue
                    for metric in ("siglip", "clip"):
                        if row[f"{metric}_pair"] != "":
                            pair[(metric, c, kind)][key] = float(row[f"{metric}_pair"])
                    if kind == "seven":
                        arc_original[c][key] = float(row["arcface_id"]) if row["arcface_id"] != "" else None
            for row in read(F / group / "arcface_pair.csv"):
                key = (row["image"], row["prompt_index"], row["purifier"])
                arc_pair[(row["condition"], kind_of(row["purifier"]))][key] = float(row["arcface_pair"]) if row["arcface_pair"] != "" else None
        for row in read(F / "aligned/survival_seven.csv"):
            survival[(row["condition"], "seven")][(row["image"], row["purifier"])] = float(row["survival"])
        for group in ("aligned", "color3d") if vae else ():
            for row in read(F / group / "survival_vae_trans.csv"):
                survival[(row["condition"], "vae_trans")][(row["image"], row["purifier"])] = float(row["survival"])
        for row in read(F / "clip_score.csv") if vae else ():
            if row["condition"] in conditions:
                clip_s[(row["condition"], kind_of(row["purifier"]))][(row["image"], row["prompt_index"], row["purifier"])] = float(row["clip_s"])

    groups = read(OUT / "distribution_metrics/pexels_distribution.csv")
    distribution = defaultdict(dict)
    for row in groups:
        c = COLOR3D if row["condition"] == "color3d" else row["condition"]
        purifier = "distortion" if row["end"] == "distortion" else row["purifier"]
        distribution[c][purifier] = row

    table, supplementary, per_purifier = [], [], []
    for c in conditions:
        n_cells = N_IMAGES * 4
        require(c, "防禦圖失真影像數", len(fidelity[c]), N_IMAGES)
        require(c, "防禦圖美學影像數", len(aesthetic[c]), N_IMAGES)
        require(c, "未淨化格數", len(displacement[c]), n_cells)
        cells = retention[c]
        require(c, "七道格數", sum(len(v) for v in cells.values()), n_cells * 7)
        seven = {(k[0], k[1], p): v for k, by in cells.items() for p, v in by.items()}
        kinds = ("seven", "vae_trans") if vae else ("seven",)
        for key in kinds:
            require(c, f"存續比 {key} 列數", len(survival[(c, key)]), N_IMAGES * (7 if key == "seven" else 1))
            for metric in ("siglip", "clip"):
                require(c, f"{metric}_pair {key} 格數", len(pair[(metric, c, key)]), n_cells * (7 if key == "seven" else 1))
            require(c, f"ArcFace(E_u,E_d) {key} 格數", len(arc_pair[(c, key)]), n_cells * (7 if key == "seven" else 1))
        require(c, "ArcFace 七道格數", len(arc_original[c]), n_cells * 7)
        if vae:
            require(c, "VAE-Trans 格數", len(vae_retention[c]), n_cells)
            for key, n in (("none", n_cells), ("seven", n_cells * 7), ("vae_trans", n_cells)):
                require(c, f"CLIP-S {key} 格數", len(clip_s[(c, key)]), n)

        row = {"condition": c, "n_images": N_IMAGES}
        extra = {"condition": c, "n_images": N_IMAGES}

        def put(target, name, values):
            """values：{(影像, …): 值}；寫入平均與影像間標準差。"""
            target[name] = mean(values.values())
            target[f"{name}_std"] = per_image_std(values)

        def blank(target, name):
            target[name] = target[f"{name}_std"] = ""

        for k in FIDELITY:
            put(row, f"defense_{k}", {(i,): v[k] for i, v in fidelity[c].items()})
        for target, columns in ((row, AESTHETIC_MAIN), (extra, AESTHETIC_SUPPLEMENTARY)):
            for column in columns:
                put(target, column, {(i,): float(r[column]) for i, r in aesthetic[c].items()})

        put(row, "disp_none", displacement[c])
        put(row, "disp_seven", {k: v[0] for k, v in seven.items()})
        by_purifier = {}
        for p in SEVEN:
            values = {k: v[0] for k, v in seven.items() if k[2] == p}
            by_purifier[p] = mean(values.values())
            per_purifier.append({"condition": c, "purifier": p, "n_cells": len(values), "disp_purified": round(by_purifier[p], 5),
                                 "retained": round(mean(v[1] for k, v in seven.items() if k[2] == p), 5)})
        weakest = min(by_purifier, key=by_purifier.get)
        put(row, "disp_weakest", {k: v[0] for k, v in seven.items() if k[2] == weakest})
        row["disp_weakest_purifier"] = weakest
        put(row, "disp_worst", {k: min(v[0] for v in by.values()) for k, by in cells.items()})
        if vae:
            put(row, "disp_vae_trans", {k: v[0] for k, v in vae_retention[c].items()})
        else:
            blank(row, "disp_vae_trans")
        put(row, "retained_seven", {k: v[1] for k, v in seven.items()})
        row["retained_seven_ratio"] = row["disp_seven"] / row["disp_none"]
        if vae:
            put(row, "retained_vae_trans", {k: v[1] for k, v in vae_retention[c].items()})
        else:
            blank(row, "retained_vae_trans")
        for metric, name in (("siglip", "siglip"), ("clip", "clip_i")):
            for key in ("seven", "vae_trans"):
                (put(row, f"{name}_{key}", pair[(metric, c, key)]) if key in kinds else blank(row, f"{name}_{key}"))
        for key in ("none", "seven", "vae_trans"):
            (put(row, f"clip_s_{key}", clip_s[(c, key)]) if vae else blank(row, f"clip_s_{key}"))
        put(row, "arcface_original_seven", faces(arc_original[c]))
        row["arcface_original_seven_missing"] = len(arc_original[c]) - len(faces(arc_original[c]))
        for key in ("seven", "vae_trans"):
            if key in kinds:
                put(row, f"arcface_pair_{key}", faces(arc_pair[(c, key)]))
                row[f"arcface_pair_{key}_missing"] = len(arc_pair[(c, key)]) - len(faces(arc_pair[(c, key)]))
            else:
                blank(row, f"arcface_pair_{key}")
                row[f"arcface_pair_{key}_missing"] = ""
        for key in ("seven", "vae_trans"):
            (put(row, f"survival_{key}", survival[(c, key)]) if key in kinds else blank(row, f"survival_{key}"))

        d = distribution[c]
        required = ("distortion", "none") + (("vae_trans",) if vae else ()) + SEVEN
        for key in required:
            if key not in d:
                raise SystemExit(f"{c} 缺分布指標組 {key}")
            require(c, f"分布指標 {key} 影像數", int(d[key]["n_images"]), N_IMAGES)
        for metric in ("fid", "kid"):
            row[f"{metric}_none"] = float(d["none"][metric])
            row[f"{metric}_seven"] = mean(float(d[p][metric]) for p in SEVEN)
            row[f"{metric}_vae_trans"] = float(d["vae_trans"][metric]) if vae else ""
            extra[f"{metric}_distortion"] = float(d["distortion"][metric])
        row["kid_negative_groups"] = sum(float(d[key]["kid"]) < 0 for key in required)
        extra["clipiqa_defense"] = float(d["distortion"]["clipiqa_a"])
        extra["clipiqa_none"] = float(d["none"]["clipiqa_a"])
        extra["clipiqa_seven"] = mean(float(d[p]["clipiqa_a"]) for p in SEVEN)
        extra["clipiqa_vae_trans"] = float(d["vae_trans"]["clipiqa_a"]) if vae else ""
        table.append({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})
        supplementary.append({k: round(v, 5) if isinstance(v, float) else v for k, v in extra.items()})

    require("原圖", "美學影像數", len(originals), N_IMAGES)
    for rows, columns in ((table, AESTHETIC_MAIN), (supplementary, AESTHETIC_SUPPLEMENTARY)):
        original = {"condition": "original", "n_images": N_IMAGES}
        for column in columns:
            values = {(i,): float(r[column]) for i, r in originals.items()}
            original[column] = round(mean(values.values()), 5)
            original[f"{column}_std"] = round(per_image_std(values), 5)
        rows.append({k: original.get(k, "") for k in rows[0]})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("main_table.csv", table), ("main_table_supplementary.csv", supplementary),
                       ("per_purifier_displacement.csv", per_purifier), ("distribution_groups.csv", groups)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    (OUT / "README.md").write_text(
        f"# Pexels 主表彙整\n\n由 `scripts/generate_pexels_main_table_summary.py` 產生；欄的定義與來源見該程式開頭。\n\n"
        + ("" if vae else "- 本版不含 VAE-Trans 與 CLIP-S：`*_vae_trans`、`clip_s_*` 欄留空。\n") +
        f"- 資料組：Pexels 人像 {N_IMAGES} 張；等失真九個條件與 `color3d_pexels`；ip2p、四句、主種子。\n"
        f"- `main_table.csv`：每條件一列，`<欄>_std` 為影像間樣本標準差；末列 `original` 為原圖美學。\n"
        f"- `main_table_supplementary.csv`：其餘防禦圖美學、防禦圖對原圖的 FID／KID、CLIP-IQA。\n"
        f"- `per_purifier_displacement.csv`：每條件逐道淨化的編輯結果 LPIPS 與保留率平均（1,224 格／道）。\n"
        f"- `distribution_groups.csv`：逐組 FID、KID、CLIP-IQA，含 `n_images`、`kid_negative`。\n"
        f"- `distribution_metrics/pexels_clipiqa.csv`：逐張 CLIP-IQA。\n", encoding="utf-8")
    print(f"[ALLDONE] {OUT / 'main_table.csv'}（{len(table)} 列）", flush=True)


if __name__ == "__main__":
    main()
