"""逐條件的 8 張平均彙整表：防禦圖失真、美學、編輯結果 LPIPS（淨化前、七道各道與平均、最低一道）、保留率、指令方向語意。

讀取（<根> 為 results 或 results/aligned）：
    <根>/additional_metrics/fidelity.csv     fid_lpips（防禦圖對原圖；PNG、CPU）
    <根>/additional_metrics/aesthetic.csv    aes_*（七項無參考美學；含 original 列）
    <根>/displacement.csv                    disp_lpips_full（場景 ip2p）
    <根>/retention.csv                       disp_purified、retained（場景 ip2p，七道）
    <根>/semantic_identity.csv               clip_dir、siglip_dir（場景 ip2p；none 與七道）
欄位：
    edit_lpips_<算子>        該道淨化後的編輯結果 LPIPS（8 張 × 4 句平均）
    edit_lpips_purified_mean 七道的平均；edit_lpips_purified_min 與 purifier_min 為七道中平均值最低的一道
    retained_mean            保留率（淨化後／淨化前）七道平均
    clip_dir_none、clip_dir_purified_mean（siglip_dir 同）  淨化前與七道平均
原圖列（condition=original）只填美學欄。

用法：python scripts/generate_condition_summary.py <根> <輸出 CSV>
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PURIFIERS = ("blur1", "blur2", "crop_resize0.1", "jpeg30", "jpeg50", "jpeg80", "rotate15")
AESTHETIC = ("aes_laion", "aes_nima", "aes_musiq_ava", "aes_topiq_iaa", "aes_clipiqa", "aes_niqe", "aes_brisque")


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean_by(rows, key, value, keep=lambda r: True):
    acc = defaultdict(list)
    for r in rows:
        if keep(r) and r[value] != "":
            acc[key(r)].append(float(r[value]))
    return {k: statistics.mean(v) for k, v in acc.items()}


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    ip2p = lambda r: r.get("scenario", "ip2p") == "ip2p"  # noqa: E731
    fid = mean_by(read(root / "additional_metrics/fidelity.csv"), lambda r: r["condition"], "fid_lpips")
    aes_rows = read(root / "additional_metrics/aesthetic.csv")
    aes = {m: mean_by(aes_rows, lambda r: r["condition"], m) for m in AESTHETIC}
    disp = mean_by(read(root / "displacement.csv"), lambda r: r["condition"], "disp_lpips_full", ip2p)
    ret_rows = read(root / "retention.csv")
    pur = mean_by(ret_rows, lambda r: (r["condition"], r["purifier"]), "disp_purified", ip2p)
    ret = mean_by(ret_rows, lambda r: r["condition"], "retained", ip2p)
    sem_rows = read(root / "semantic_identity.csv")
    sem = {m: mean_by(sem_rows, lambda r: (r["condition"], "none" if r["purifier"] == "none" else "purified"), m, ip2p)
           for m in ("clip_dir", "siglip_dir")}

    rows = [{"condition": "original", **{m: round(aes[m]["original"], 4) for m in AESTHETIC if "original" in aes[m]}}]
    for c in sorted(disp):
        per = {p: pur[(c, p)] for p in PURIFIERS}
        worst = min(per, key=per.get)
        rows.append({
            "condition": c, "defense_lpips": round(fid[c], 4),
            **{m: round(aes[m][c], 4) for m in AESTHETIC},
            "edit_lpips_none": round(disp[c], 4),
            **{f"edit_lpips_{p}": round(v, 4) for p, v in per.items()},
            "edit_lpips_purified_mean": round(statistics.mean(per.values()), 4),
            "edit_lpips_purified_min": round(per[worst], 4), "purifier_min": worst,
            "retained_mean": round(ret[c], 4),
            **{f"{m}_{s}": round(sem[m][(c, s)], 4) for m in ("clip_dir", "siglip_dir") for s in ("none", "purified")},
        })
    fields = list(dict.fromkeys(k for r in rows for k in r))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
