"""Pexels 人像 306 張的主表彙整：等失真九個條件與 color3d_pexels，每條件一列，ip2p、主種子。

來源（<根> 為 image_sets.json 的 root 欄）：
    等失真臂  runtime/pexels_portraits/aligned/<根>/results/：fidelity.csv、aesthetic.csv（photoguard_encoder 為 results/photoguard_encoder/）、
              displacement_<條件>.csv、retention_<條件>.csv
    color3d   color trial results/color3d_pexels/{displacement,retention}.csv；防禦圖讀數在 followup/<根>/results/color3d/
    後續量測  runtime/pexels_portraits/followup/<根>/results/<aligned|color3d>/：retention_vae_trans_<條件>.csv、semantic_identity{,_vae_trans}.csv、
              arcface_pair.csv、survival_{seven,vae_trans}.csv（color3d 七道為 followup/results/color3d_all/survival_seven.csv）、
              edit_fidelity{,_vae_trans}.csv（scripts/measure_pexels_edit_fidelity.py）；followup/<根>/results/clip_score.csv
    分布指標  runtime/pexels_portraits/summary/distribution_metrics/pexels_distribution.csv

主表 main_table.csv 的欄（none＝淨化前、seven＝七道、vae_trans＝VAE-Trans）：
    defense_lpips：防禦圖對原圖（fidelity.csv 的 fid_lpips），交付 PNG、CPU；aes_laion、aes_musiq_ava：防禦圖美學
    disp_{none,seven,vae_trans}：編輯結果 LPIPS（E_u 對 E_d，兩者經同一淨化）；disp_worst：每格取七道最小值
    disp_ssim_*、disp_psnr_*：E_u 對 E_d 的 SSIM、PSNR（512²、整張，同編輯結果 LPIPS 的讀圖與配對）；none 取位移表的 disp_ssim、disp_psnr
    retained_{seven,vae_trans}：逐格 disp_purified/disp_plain；retained_seven_ratio＝disp_seven／disp_none
    siglip_*、clip_i_*：兩張編輯的 SigLIP、CLIP 影像餘弦（siglip_pair、clip_pair；CLIP-I）
    clip_s_*：cos(E_img(編輯), E_txt(指令))，CLIP（clip_score.csv 的 clip_s；CLIP-S）
    arcface_original_seven：ArcFace(原圖, 編輯結果)；arcface_pair_*：ArcFace(E_u, E_d)；只計有臉的格，*_missing 為無臉格數
    survival_*：LPIPS(P(x̃),P(x))/LPIPS(x̃,x)；fid_*、kid_*：編輯端；kid_negative_groups 為 KID 負值的組數
  平均與區間（逐張值＝該張 4 句的平均；ArcFace 只計有臉的格與影像）：
    七道欄（*_seven，disp_worst 除外）：先算每一道的逐張值在影像間的平均，得 7 個逐道平均（per_purifier.csv）；
        欄值為 7 個的平均，<欄>_std 為 7 個的樣本標準差 s_p，<欄>_ci95＝t(0.975, 6)·s_p／√7（t＝2.446912）。
        fid_seven、kid_seven 為 7 組各自的值，同一規則。
    其餘欄：欄值為逐格平均（格數齊全時等於逐張值的平均），<欄>_std 為逐張值的影像間樣本標準差 s，<欄>_ci95＝1.96·s／√n，
        n 為有逐張值的影像數（ArcFace 為有臉的影像數，其餘 306）。
    retained_seven_ratio、fid_none、fid_vae_trans、kid_none、kid_vae_trans 沒有區間。
補充表 main_table_supplementary.csv：defense_ssim、defense_psnr（防禦圖對原圖，fid_ssim、fid_psnr）；其餘五項防禦圖美學
    （aes_nima、aes_topiq_iaa、aes_clipiqa、aes_niqe、aes_brisque）；disp_weakest、disp_ssim_weakest、disp_psnr_weakest
    （Weakest＝7 個逐道 disp 平均最低的一道，名稱在 disp_weakest_purifier，區間為該道逐張值的影像間 1.96·s／√306）；
    fid_distortion、kid_distortion（防禦圖對原圖）；clipiqa_*（A 集合的 CLIP-IQA）。皆附 _std、_ci95（影像間；分布指標除外）。
逐道表 per_purifier.csv：每條件 × 每道的逐道平均（上述七道欄的 7 個值）與 FID、KID。
兩張表的末列 original 為原圖美學。任一條件的影像數不是 306 或格數不齊時失敗。
`--without-vae-trans` 時不讀 VAE-Trans 與 CLIP-S 的來源，這些欄留空，其餘檢查不變。

輸出：runtime/pexels_portraits/summary/{main_table.csv,main_table_supplementary.csv,per_purifier.csv,distribution_groups.csv,README.md}
用法（遠端 baseline 專案根）：python scripts/generate_pexels_main_table_summary.py [--without-vae-trans]
"""
import argparse
import csv
import json
import math
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
T_0975_DF6 = 2.446912   # Student t 分布 0.975 分位，自由度 6
Z_0975 = 1.96
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


def require(condition, label, got, want):
    if got != want:
        raise SystemExit(f"{condition} 的 {label} 為 {got}，應為 {want}")


def faces(cells):
    return {k: v for k, v in cells.items() if v is not None}


def per_image(cells):
    """{(影像, …): 值} -> {影像: 該張各格的平均}。"""
    by_image = defaultdict(list)
    for key, value in cells.items():
        by_image[key[0]].append(value)
    return {i: mean(v) for i, v in by_image.items()}


def image_stat(cells):
    """逐格平均，以及逐張值的影像間樣本標準差與 1.96·s／√n。"""
    values = list(per_image(cells).values())
    s = statistics.stdev(values)
    return mean(cells.values()), s, Z_0975 * s / math.sqrt(len(values))


def purifier_means(cells):
    """{(影像, …, 道): 值} -> {道: 逐張值的影像間平均}；7 道皆須存在。"""
    means = {p: mean(per_image({k: v for k, v in cells.items() if k[-1] == p}).values())
             for p in SEVEN if any(k[-1] == p for k in cells)}
    if tuple(means) != SEVEN:
        raise SystemExit(f"七道不齊：{sorted(means)}")
    return means


def purifier_stat(values):
    """7 個逐道值的平均、樣本標準差與 t(0.975, 6)·s／√7。"""
    values = list(values)
    s = statistics.stdev(values)
    return mean(values), s, T_0975_DF6 * s / math.sqrt(len(values))


def put(target, name, stat):
    target[name], target[f"{name}_std"], target[f"{name}_ci95"] = stat


def blank(target, name):
    target[name] = target[f"{name}_std"] = target[f"{name}_ci95"] = ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--without-vae-trans", action="store_true")
    vae = not ap.parse_args().without_vae_trans
    sets = json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
    conditions = ALIGNED + (COLOR3D,)
    fidelity = {c: {} for c in conditions}            # 影像 -> {lpips, ssim, psnr}
    aesthetic = {c: {} for c in conditions}           # 影像 -> 列
    # metric[(指標, 條件, none|seven|vae_trans)] = {(影像, 句, 道): 值}；survival 的鍵為 (影像, 道)
    metric = defaultdict(dict)
    originals = {}

    def kind_of(purifier):
        return "seven" if purifier in SEVEN else purifier

    def add_displacement(c, row):
        key = (row["image"], row["prompt_index"], "none")
        metric[("disp", c, "none")][key] = float(row["disp_lpips_full"])
        for m in ("ssim", "psnr"):
            metric[(f"disp_{m}", c, "none")][key] = float(row[f"disp_{m}"])

    def add_retention(c, row, kind):
        key = (row["image"], row["prompt_index"], row["purifier"])
        metric[("disp", c, kind)][key] = float(row["disp_purified"])
        metric[("retained", c, kind)][key] = float(row["retained"])

    for row in read(CT / "results/color3d_pexels/displacement.csv"):
        add_displacement(COLOR3D, row)
    for row in read(CT / "results/color3d_pexels/retention.csv"):
        add_retention(COLOR3D, row, "seven")
    for row in read("runtime/pexels_portraits/followup/results/color3d_all/survival_seven.csv"):
        metric[("survival", COLOR3D, "seven")][(row["image"], row["purifier"])] = float(row["survival"])

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
                    add_displacement(c, row)
            for row in read(R / f"retention_{c}.csv"):
                if row["scenario"] == "ip2p" and row["purifier"] in SEVEN:
                    add_retention(c, row, "seven")
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
                        add_retention(c, row, "vae_trans")
            suffixes = ("", "_vae_trans") if vae else ("",)
            for suffix in suffixes:
                for row in read(F / group / f"semantic_identity{suffix}.csv"):
                    c, kind = row["condition"], kind_of(row["purifier"])
                    key = (row["image"], row["prompt_index"], row["purifier"])
                    if c == "undefended" or kind not in ("seven", "vae_trans"):
                        continue
                    for m, name in (("siglip", "siglip"), ("clip", "clip_i")):
                        if row[f"{m}_pair"] != "":
                            metric[(name, c, kind)][key] = float(row[f"{m}_pair"])
                    if kind == "seven":
                        metric[("arcface_original", c, kind)][key] = float(row["arcface_id"]) if row["arcface_id"] != "" else None
                for row in read(F / group / f"edit_fidelity{suffix}.csv"):
                    kind = kind_of(row["purifier"])
                    if kind in ("seven", "vae_trans"):
                        for m in ("ssim", "psnr"):
                            metric[(f"disp_{m}", row["condition"], kind)][(row["image"], row["prompt_index"], row["purifier"])] = float(row[f"disp_{m}"])
            for row in read(F / group / "arcface_pair.csv"):
                key = (row["image"], row["prompt_index"], row["purifier"])
                metric[("arcface_pair", row["condition"], kind_of(row["purifier"]))][key] = (
                    float(row["arcface_pair"]) if row["arcface_pair"] != "" else None)
        for row in read(F / "aligned/survival_seven.csv"):
            metric[("survival", row["condition"], "seven")][(row["image"], row["purifier"])] = float(row["survival"])
        for group in ("aligned", "color3d") if vae else ():
            for row in read(F / group / "survival_vae_trans.csv"):
                metric[("survival", row["condition"], "vae_trans")][(row["image"], row["purifier"])] = float(row["survival"])
        for row in read(F / "clip_score.csv") if vae else ():
            if row["condition"] in conditions:
                metric[("clip_s", row["condition"], kind_of(row["purifier"]))][
                    (row["image"], row["prompt_index"], row["purifier"])] = float(row["clip_s"])

    groups = read(OUT / "distribution_metrics/pexels_distribution.csv")
    distribution = defaultdict(dict)
    for row in groups:
        c = COLOR3D if row["condition"] == "color3d" else row["condition"]
        distribution[c]["distortion" if row["end"] == "distortion" else row["purifier"]] = row

    n_cells = N_IMAGES * 4
    # (指標, 段) -> 期望格數；survival 每張每道一列
    expected = {}
    for kind, k in (("none", 1), ("seven", 7), ("vae_trans", 1)):
        for m in ("disp", "disp_ssim", "disp_psnr", "retained", "siglip", "clip_i", "clip_s", "arcface_original", "arcface_pair"):
            expected[(m, kind)] = n_cells * k
        expected[("survival", kind)] = N_IMAGES * k
    COLUMNS = (  # (指標, 段, 是否只在含 VAE-Trans 時)
        [("disp", "none", False), ("disp", "seven", False), ("disp", "vae_trans", True)]
        + [(f"disp_{m}", kind, kind == "vae_trans") for m in ("ssim", "psnr") for kind in ("none", "seven", "vae_trans")]
        + [("retained", "seven", False), ("retained", "vae_trans", True)]
        + [(m, kind, kind == "vae_trans") for m in ("siglip", "clip_i") for kind in ("seven", "vae_trans")]
        + [("clip_s", kind, True) for kind in ("none", "seven", "vae_trans")]
        + [("arcface_original", "seven", False), ("arcface_pair", "seven", False), ("arcface_pair", "vae_trans", True)]
        + [("survival", "seven", False), ("survival", "vae_trans", True)])

    table, supplementary, per_purifier = [], [], []
    for c in conditions:
        require(c, "防禦圖失真影像數", len(fidelity[c]), N_IMAGES)
        require(c, "防禦圖美學影像數", len(aesthetic[c]), N_IMAGES)
        row = {"condition": c, "n_images": N_IMAGES}
        extra = {"condition": c, "n_images": N_IMAGES}
        put(row, "defense_lpips", image_stat({(i,): v["lpips"] for i, v in fidelity[c].items()}))
        for k in ("ssim", "psnr"):
            put(extra, f"defense_{k}", image_stat({(i,): v[k] for i, v in fidelity[c].items()}))
        for target, columns in ((row, AESTHETIC_MAIN), (extra, AESTHETIC_SUPPLEMENTARY)):
            for column in columns:
                put(target, column, image_stat({(i,): float(r[column]) for i, r in aesthetic[c].items()}))

        by_purifier = {}
        for m, kind, needs_vae in COLUMNS:
            name = f"{m}_{kind}"
            if needs_vae and not vae:
                blank(row, name)
                if m.startswith("arcface"):
                    row[f"{name}_missing"] = ""
                continue
            cells = metric[(m, c, kind)]
            require(c, f"{name} 格數", len(cells), expected[(m, kind)])
            if m.startswith("arcface"):
                row[f"{name}_missing"] = len(cells) - len(faces(cells))
                cells = faces(cells)
            if kind == "seven":
                by_purifier[m] = purifier_means(cells)
                put(row, name, purifier_stat(by_purifier[m].values()))
            else:
                put(row, name, image_stat(cells))
            if name == "disp_seven":
                worst = defaultdict(list)
                for (i, pi, _), v in cells.items():
                    worst[(i, pi)].append(v)
                require(c, "七道最小值格數", sum(len(v) == 7 for v in worst.values()), n_cells)
                put(row, "disp_worst", image_stat({k: min(v) for k, v in worst.items()}))
        row["retained_seven_ratio"] = row["disp_seven"] / row["disp_none"]

        weakest = min(by_purifier["disp"], key=by_purifier["disp"].get)
        extra["disp_weakest_purifier"] = weakest
        for m in ("disp", "disp_ssim", "disp_psnr"):
            name = "disp_weakest" if m == "disp" else f"{m}_weakest"
            put(extra, name, image_stat({k: v for k, v in metric[(m, c, "seven")].items() if k[-1] == weakest}))

        d = distribution[c]
        required = ("distortion", "none") + (("vae_trans",) if vae else ()) + SEVEN
        for key in required:
            if key not in d:
                raise SystemExit(f"{c} 缺分布指標組 {key}")
            require(c, f"分布指標 {key} 影像數", int(d[key]["n_images"]), N_IMAGES)
        for m in ("fid", "kid"):
            row[f"{m}_none"] = float(d["none"][m])
            put(row, f"{m}_seven", purifier_stat(float(d[p][m]) for p in SEVEN))
            row[f"{m}_vae_trans"] = float(d["vae_trans"][m]) if vae else ""
            extra[f"{m}_distortion"] = float(d["distortion"][m])
            by_purifier[m] = {p: float(d[p][m]) for p in SEVEN}
        row["kid_negative_groups"] = sum(float(d[key]["kid"]) < 0 for key in required)
        extra["clipiqa_defense"] = float(d["distortion"]["clipiqa_a"])
        extra["clipiqa_none"] = float(d["none"]["clipiqa_a"])
        extra["clipiqa_seven"] = mean(float(d[p]["clipiqa_a"]) for p in SEVEN)
        extra["clipiqa_vae_trans"] = float(d["vae_trans"]["clipiqa_a"]) if vae else ""
        for p in SEVEN:
            per_purifier.append({"condition": c, "purifier": p,
                                 **{m: round(v[p], 5) for m, v in by_purifier.items()}})
        table.append({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})
        supplementary.append({k: round(v, 5) if isinstance(v, float) else v for k, v in extra.items()})

    require("原圖", "美學影像數", len(originals), N_IMAGES)
    for rows, columns in ((table, AESTHETIC_MAIN), (supplementary, AESTHETIC_SUPPLEMENTARY)):
        original = {"condition": "original", "n_images": N_IMAGES}
        for column in columns:
            put(original, column, tuple(round(x, 5) for x in image_stat({(i,): float(r[column]) for i, r in originals.items()})))
        rows.append({k: original.get(k, "") for k in rows[0]})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("main_table.csv", table), ("main_table_supplementary.csv", supplementary),
                       ("per_purifier.csv", per_purifier), ("distribution_groups.csv", groups)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    (OUT / "README.md").write_text(
        f"# Pexels 主表彙整\n\n由 `scripts/generate_pexels_main_table_summary.py` 產生；欄的定義、平均與區間的算法見該程式開頭。\n\n"
        + ("" if vae else "- 本版不含 VAE-Trans 與 CLIP-S：`*_vae_trans`、`clip_s_*` 欄留空。\n") +
        f"- 資料組：Pexels 人像 {N_IMAGES} 張；等失真九個條件與 `color3d_pexels`；ip2p、四句、主種子。\n"
        f"- `main_table.csv`：每條件一列；`<欄>_std`、`<欄>_ci95` 七道欄為跨 7 道、其餘為跨影像；末列 `original` 為原圖美學。\n"
        f"- `main_table_supplementary.csv`：防禦圖 SSIM／PSNR、其餘美學、Weakest、防禦圖對原圖的 FID／KID、CLIP-IQA。\n"
        f"- `per_purifier.csv`：每條件逐道的平均（七道欄的 7 個值）與 FID、KID。\n"
        f"- `distribution_groups.csv`：逐組 FID、KID、CLIP-IQA，含 `n_images`、`kid_negative`。\n", encoding="utf-8")
    print(f"[ALLDONE] {OUT / 'main_table.csv'}（{len(table)} 列）", flush=True)


if __name__ == "__main__":
    main()
