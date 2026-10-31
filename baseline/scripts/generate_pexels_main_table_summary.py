"""Pexels 人像 306 張的主表彙整：等失真九個條件與 color3d_pexels，每條件一列，ip2p、主種子。

來源（<根> 為 image_sets.json 的 root 欄）：
    等失真臂  runtime/pexels_portraits/aligned/<根>/results/：fidelity.csv、aesthetic.csv（photoguard_encoder 為 results/photoguard_encoder/）、
              displacement_<條件>.csv、retention_<條件>.csv
    color3d   color trial results/color3d_pexels/{displacement,retention}.csv；防禦圖讀數在 followup/<根>/results/color3d/
    後續量測  runtime/pexels_portraits/followup/<根>/results/<aligned|color3d>/：retention_vae_trans_<條件>.csv、semantic_identity{,_vae_trans}.csv、
              arcface_pair.csv、survival_{seven,vae_trans}.csv（color3d 七道為 followup/results/color3d_all/survival_seven.csv）
    分布指標  runtime/pexels_portraits/summary/distribution_metrics/pexels_distribution.csv

欄（數值皆先逐格或逐張再對條件平均）：
    防禦圖 LPIPS 與七項美學：交付 PNG、CPU
    disp_none、disp_seven、disp_worst、disp_vae_trans：編輯結果 LPIPS（未淨化、七道平均、每格取七道最小值後平均、VAE-Trans）
    retained_seven、retained_seven_ratio、retained_vae_trans：逐格 disp_purified/disp_plain 的平均、七道平均之比、VAE-Trans 逐格平均
    siglip_seven、siglip_vae_trans：兩張編輯的 SigLIP 影像餘弦（siglip_pair）
    arcface_original_seven：ArcFace(原圖, 編輯結果)；arcface_pair_seven、arcface_pair_vae_trans：ArcFace(E_u, E_d)；只計有臉的格，*_missing 為無臉格數
    survival_seven、survival_vae_trans：LPIPS(P(x̃),P(x))/LPIPS(x̃,x) 的逐張平均
    clipiqa_defense、clipiqa_none、clipiqa_seven、clipiqa_vae_trans：A 集合（防禦圖或防禦後編輯）的 CLIP-IQA
    fid_*、kid_*：distortion（防禦圖對原圖）、none、seven（七組的平均）、vae_trans；kid_negative_groups 為 KID 負值的組數
任一條件的影像數不是 306 或格數不齊時失敗。

輸出：runtime/pexels_portraits/summary/{main_table.csv,distribution_groups.csv,README.md}
用法（遠端 baseline 專案根）：python scripts/generate_pexels_main_table_summary.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

N_IMAGES = 306
SEVEN = ("blur1", "blur2", "crop_resize0.1", "jpeg30", "jpeg50", "jpeg80", "rotate15")
ALIGNED = ("mist", "dia_pt", "sifm", "dia_r", "dct_shield_y", "dct_shield", "dayn", "danp", "photoguard_encoder")
COLOR3D = "color3d_pexels"
AESTHETIC = ("aes_laion", "aes_nima", "aes_musiq_ava", "aes_topiq_iaa", "aes_clipiqa", "aes_niqe", "aes_brisque")
CT = Path("../color/trials/lut3d_defense_and_naturalness")
OUT = Path("runtime/pexels_portraits/summary")


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values):
    if not values:
        raise SystemExit("空集合無法平均")
    return sum(values) / len(values)


def require(condition, label, got, want):
    if got != want:
        raise SystemExit(f"{condition} 的 {label} 為 {got}，應為 {want}")


def main():
    sets = json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
    fidelity, aesthetic, displacement, retention, vae_retention = {}, {}, {}, {}, {}
    siglip, arc_original, arc_pair, survival = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    originals = {}
    for c in ALIGNED + (COLOR3D,):
        fidelity[c], aesthetic[c], displacement[c], retention[c], vae_retention[c] = {}, {}, {}, defaultdict(dict), {}
    for row in read(CT / "results/color3d_pexels/displacement.csv"):
        displacement[COLOR3D][(row["image"], row["prompt_index"])] = float(row["disp_lpips_full"])
    for row in read(CT / "results/color3d_pexels/retention.csv"):
        retention[COLOR3D][(row["image"], row["prompt_index"])][row["purifier"]] = (float(row["disp_purified"]), float(row["retained"]))
    for row in read("runtime/pexels_portraits/followup/results/color3d_all/survival_seven.csv"):
        survival[(COLOR3D, "seven")].append(float(row["survival"]))

    for s in sets:
        sub = Path(s["root"]) if s["root"] else Path()
        R = Path("runtime/pexels_portraits/aligned") / sub / "results"
        F = Path("runtime/pexels_portraits/followup") / sub / "results"
        for c in ALIGNED:
            directory = R / c if c == "photoguard_encoder" else R
            for row in read(directory / "fidelity.csv"):
                if row["condition"] == c:
                    fidelity[c][row["image"]] = float(row["fid_lpips"])
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
                fidelity[COLOR3D][row["image"]] = float(row["fid_lpips"])
        for row in read(F / "color3d/aesthetic.csv"):
            if row["condition"] == COLOR3D:
                aesthetic[COLOR3D][row["image"]] = row
        for group, conditions in (("aligned", ALIGNED), ("color3d", (COLOR3D,))):
            for c in conditions:
                for row in read(F / group / f"retention_vae_trans_{c}.csv"):
                    if row["scenario"] == "ip2p":
                        vae_retention[c][(row["image"], row["prompt_index"])] = (float(row["disp_purified"]), float(row["retained"]))
            for name in ("semantic_identity.csv", "semantic_identity_vae_trans.csv"):
                for row in read(F / group / name):
                    c = row["condition"]
                    kind = "seven" if row["purifier"] in SEVEN else row["purifier"]
                    if c != "undefended" and row["siglip_pair"] != "":
                        siglip[(c, kind)].append(float(row["siglip_pair"]))
                    if c != "undefended" and row["purifier"] in SEVEN:
                        arc_original[c].append(float(row["arcface_id"]) if row["arcface_id"] != "" else None)
            for row in read(F / group / "arcface_pair.csv"):
                kind = "seven" if row["purifier"] in SEVEN else row["purifier"]
                arc_pair[(row["condition"], kind)].append(float(row["arcface_pair"]) if row["arcface_pair"] != "" else None)
        for row in read(F / "aligned/survival_seven.csv"):
            survival[(row["condition"], "seven")].append(float(row["survival"]))
        for group in ("aligned", "color3d"):
            for row in read(F / group / "survival_vae_trans.csv"):
                survival[(row["condition"], "vae_trans")].append(float(row["survival"]))

    groups = read(OUT / "distribution_metrics/pexels_distribution.csv")
    distribution = defaultdict(dict)
    for row in groups:
        c = COLOR3D if row["condition"] == "color3d" else row["condition"]
        purifier = "distortion" if row["end"] == "distortion" else row["purifier"]
        distribution[c][purifier] = row

    table = []
    for c in ALIGNED + (COLOR3D,):
        n_cells = N_IMAGES * 4
        require(c, "防禦圖失真影像數", len(fidelity[c]), N_IMAGES)
        require(c, "防禦圖美學影像數", len(aesthetic[c]), N_IMAGES)
        require(c, "未淨化格數", len(displacement[c]), n_cells)
        require(c, "VAE-Trans 格數", len(vae_retention[c]), n_cells)
        cells = retention[c]
        require(c, "七道格數", sum(len(v) for v in cells.values()), n_cells * 7)
        seven = [v for by_purifier in cells.values() for v in by_purifier.values()]
        for key in (("seven"), ("vae_trans")):
            require(c, f"存續比 {key} 列數", len(survival[(c, key)]), N_IMAGES * (7 if key == "seven" else 1))
        original_faces = [x for x in arc_original[c] if x is not None]
        pair_seven = arc_pair[(c, "seven")]
        pair_vae = arc_pair[(c, "vae_trans")]
        require(c, "ArcFace 七道格數", len(arc_original[c]), n_cells * 7)
        require(c, "ArcFace(E_u,E_d) 七道格數", len(pair_seven), n_cells * 7)
        require(c, "ArcFace(E_u,E_d) VAE-Trans 格數", len(pair_vae), n_cells)
        row = {"condition": c, "n_images": N_IMAGES, "defense_lpips": mean(list(fidelity[c].values()))}
        for column in AESTHETIC:
            row[column] = mean([float(r[column]) for r in aesthetic[c].values()])
        row.update({
            "disp_none": mean(list(displacement[c].values())),
            "disp_seven": mean([v[0] for v in seven]),
            "disp_worst": mean([min(v[0] for v in by_purifier.values()) for by_purifier in cells.values()]),
            "disp_vae_trans": mean([v[0] for v in vae_retention[c].values()]),
            "retained_seven": mean([v[1] for v in seven]),
            "retained_seven_ratio": mean([v[0] for v in seven]) / mean(list(displacement[c].values())),
            "retained_vae_trans": mean([v[1] for v in vae_retention[c].values()]),
            "siglip_seven": mean(siglip[(c, "seven")]),
            "siglip_vae_trans": mean(siglip[(c, "vae_trans")]),
            "arcface_original_seven": mean(original_faces),
            "arcface_original_seven_missing": len(arc_original[c]) - len(original_faces),
            "arcface_pair_seven": mean([x for x in pair_seven if x is not None]),
            "arcface_pair_seven_missing": sum(x is None for x in pair_seven),
            "arcface_pair_vae_trans": mean([x for x in pair_vae if x is not None]),
            "arcface_pair_vae_trans_missing": sum(x is None for x in pair_vae),
            "survival_seven": mean(survival[(c, "seven")]),
            "survival_vae_trans": mean(survival[(c, "vae_trans")]),
        })
        d = distribution[c]
        required = ("distortion", "none", "vae_trans") + SEVEN
        for key in required:
            if key not in d:
                raise SystemExit(f"{c} 缺分布指標組 {key}")
            require(c, f"分布指標 {key} 影像數", int(d[key]["n_images"]), N_IMAGES)
        for metric, column in (("fid", "fid"), ("kid", "kid"), ("clipiqa", "clipiqa_a")):
            name = {"distortion": "defense" if metric == "clipiqa" else "distortion"}
            row[f"{metric}_{name['distortion']}"] = float(d["distortion"][column])
            row[f"{metric}_none"] = float(d["none"][column])
            row[f"{metric}_seven"] = mean([float(d[p][column]) for p in SEVEN])
            row[f"{metric}_vae_trans"] = float(d["vae_trans"][column])
        row["kid_negative_groups"] = sum(float(d[key]["kid"]) < 0 for key in required)
        table.append({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})

    require("原圖", "美學影像數", len(originals), N_IMAGES)
    original_row = {"condition": "original", "n_images": N_IMAGES}
    for column in AESTHETIC:
        original_row[column] = round(mean([float(r[column]) for r in originals.values()]), 5)
    table.append({k: original_row.get(k, "") for k in table[0]})

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "main_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    with open(OUT / "distribution_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(groups[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(groups)
    (OUT / "README.md").write_text(
        f"# Pexels 主表彙整\n\n由 `scripts/generate_pexels_main_table_summary.py` 產生；欄的定義與來源見該程式開頭。\n\n"
        f"- 資料組：Pexels 人像 {N_IMAGES} 張；等失真九個條件與 `color3d_pexels`；ip2p、四句、主種子。\n"
        f"- `main_table.csv`：每條件一列，末列 `original` 為原圖美學。\n"
        f"- `distribution_groups.csv`：逐組 FID、KID、CLIP-IQA，含 `n_images`、`kid_negative`。\n"
        f"- `distribution_metrics/pexels_clipiqa.csv`：逐張 CLIP-IQA。\n", encoding="utf-8")
    print(f"[ALLDONE] {OUT / 'main_table.csv'}（{len(table)} 列）", flush=True)


if __name__ == "__main__":
    main()
