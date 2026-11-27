"""消融降強度擴充（crosseditor100 全部 100 張）的防禦圖讀數：防禦圖 LPIPS、到不了目標帶的張數、LAION、MUSIQ-AVA。

樣本：color/results/ablation_strength_crosseditor100/images.txt（100 張）。條件：Full（color3d_scale2p5_pexels）、
B2_lpips0p85、B3_lpips0p55、B3_lpips0p40（目標 t ＝ r · Full 同一張的防禦圖 LPIPS，容差 ±0.015）。
讀法同 measure_ablation_strength_summary.py：
    防禦圖 LPIPS   trial results/<條件>/fidelity.csv 的 lpips，只取 arm 為該條件的列
    LAION、MUSIQ   Full 為 color3d_scale2p5_pexels/per_image/aesthetic.csv；其餘為 <條件>/additional/aesthetic.csv；
                   只取 condition 為該條件的列（另有 condition＝original 的原圖列）；欄 aes_laion、aes_musiq_ava
    同一影像重複即失敗。無防禦圖的影像須記為 failed_no_feasible_checkpoint，計為到不了目標帶；其他原因缺圖或缺讀數即失敗。
輸出 color/results/ablation_strength_crosseditor100/：per_image.csv、summary.csv、summary.txt。
用法（遠端，CPU）：python color/scripts/measure_ablation_strength_crosseditor100_summary.py
"""
import csv
from pathlib import Path
from statistics import mean

COLOR = Path(__file__).resolve().parents[1]
TRIAL = COLOR / "trials/lut3d_defense_and_naturalness"
OUT = COLOR / "results/ablation_strength_crosseditor100"
FULL = "color3d_scale2p5_pexels"
TOL = 0.015
CONDS = [("color_baseline_huesat_lpips0p85_pexels", "B2", 0.85), ("color_baseline_affine_lpips0p55_pexels", "B3", 0.55),
         ("color_baseline_affine_lpips0p40_pexels", "B3", 0.40)]
METRICS = ["aes_laion", "aes_musiq_ava"]


def by_image(path, cols, key, condition):
    """只取 key 欄等於 condition 的列（aesthetic.csv 每張影像另有 condition＝original 的原圖列）；同一影像重複即失敗。"""
    if not path.is_file():
        raise SystemExit(f"缺 {path}")
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r[key] != condition:
            continue
        if r["image"] in out:
            raise SystemExit(f"{path} 的 {condition} {r['image']} 重複")
        out[r["image"]] = {c: float(r[c]) for c in cols}
    if not out:
        raise SystemExit(f"{path} 沒有 {key}＝{condition} 的列")
    return out


def main():
    images = (OUT / "images.txt").read_text(encoding="utf-8").split()
    full_lpips = by_image(TRIAL / "results" / FULL / "fidelity.csv", ["lpips"], "arm", FULL)
    full_aes = by_image(TRIAL / "results" / FULL / "per_image" / "aesthetic.csv", METRICS, "condition", FULL)
    rows, summary = [], []
    for img in images:
        if img not in full_lpips or img not in full_aes:
            raise SystemExit(f"Full 缺 {img} 的讀數")
    summary.append({"condition": FULL, "row": "Full", "ratio": 1.0, "n_images": len(images), "n_delivered": len(images),
                    "n_unreachable": 0, "defense_lpips": round(mean(full_lpips[i]["lpips"] for i in images), 4),
                    "target_lpips": "", "max_abs_gap_to_target": "",
                    "laion": round(mean(full_aes[i]["aes_laion"] for i in images), 4),
                    "musiq": round(mean(full_aes[i]["aes_musiq_ava"] for i in images), 4),
                    "full_laion_same_images": "", "full_musiq_same_images": "", "unreachable_images": ""})
    for cond, row, ratio in CONDS:
        res = TRIAL / "results" / cond
        lp = by_image(res / "fidelity.csv", ["lpips"], "arm", cond) if (res / "fidelity.csv").is_file() else {}
        aes = by_image(res / "additional" / "aesthetic.csv", METRICS, "condition", cond) if lp else {}
        delivered, unreachable = [], []
        for img in images:
            target = ratio * full_lpips[img]["lpips"]
            if (TRIAL / "artifacts_pexels/defense_shards" / cond / img / f"{img}__{cond}__def.png").is_file():
                if img not in lp or img not in aes:
                    raise SystemExit(f"{cond} {img} 有防禦圖但缺讀數")
                delivered.append(img)
                rows.append({"condition": cond, "row": row, "ratio": ratio, "image": img, "status": "delivered",
                             "full_lpips": full_lpips[img]["lpips"], "target_lpips": round(target, 4),
                             "defense_lpips": lp[img]["lpips"], "gap_to_target": round(lp[img]["lpips"] - target, 4),
                             "laion": aes[img]["aes_laion"], "musiq": aes[img]["aes_musiq_ava"],
                             "full_laion": full_aes[img]["aes_laion"], "full_musiq": full_aes[img]["aes_musiq_ava"]})
            else:
                shard = TRIAL / "artifacts_pexels/defense_shards" / cond / img / "results.csv"
                out = [r.get("outcome") for r in csv.DictReader(open(shard, encoding="utf-8"))] if shard.is_file() else []
                if out != ["failed_no_feasible_checkpoint"]:
                    raise SystemExit(f"{cond} {img} 沒有防禦圖，也未記為 failed_no_feasible_checkpoint（{shard}）")
                unreachable.append(img)
                rows.append({"condition": cond, "row": row, "ratio": ratio, "image": img, "status": "no_feasible_checkpoint",
                             "full_lpips": full_lpips[img]["lpips"], "target_lpips": round(target, 4), "defense_lpips": "",
                             "gap_to_target": "", "laion": "", "musiq": "", "full_laion": full_aes[img]["aes_laion"],
                             "full_musiq": full_aes[img]["aes_musiq_ava"]})
        d = [r for r in rows if r["condition"] == cond and r["status"] == "delivered"]
        avg = (lambda k: round(mean(r[k] for r in d), 4) if d else "")
        summary.append({"condition": cond, "row": row, "ratio": ratio, "n_images": len(images), "n_delivered": len(delivered),
                        "n_unreachable": len(unreachable), "defense_lpips": avg("defense_lpips"), "target_lpips": avg("target_lpips"),
                        "max_abs_gap_to_target": round(max(abs(r["gap_to_target"]) for r in d), 4) if d else "",
                        "laion": avg("laion"), "musiq": avg("musiq"), "full_laion_same_images": avg("full_laion"),
                        "full_musiq_same_images": avg("full_musiq"), "unreachable_images": " ".join(unreachable)})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("per_image.csv", rows), ("summary.csv", summary)):
        with open(OUT / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    lines = [f"消融降強度：crosseditor100 全部 100 張，防禦圖 LPIPS 目標 r · Full 同一張，容差 ±{TOL}；",
             "防禦圖 LPIPS 為 piq.LPIPS（8 位元 PNG）；LAION、MUSIQ-AVA 為防禦圖本身的分數，平均只計交付影像。",
             f"{'列':<14}{'r':>6}{'交付':>6}{'無可行':>8}{'防禦圖LPIPS':>12}{'目標':>8}{'LAION':>8}{'MUSIQ':>8}{'Full LAION':>12}{'Full MUSIQ':>12}"]
    for s in summary:
        lines.append(f"{s['row']:<14}{s['ratio']:>6}{s['n_delivered']:>6}{s['n_unreachable']:>8}{s['defense_lpips']!s:>12}"
                     f"{s['target_lpips']!s:>8}{s['laion']!s:>8}{s['musiq']!s:>8}{s['full_laion_same_images']!s:>12}"
                     f"{s['full_musiq_same_images']!s:>12}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
