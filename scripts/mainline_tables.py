"""主線頭對頭的三張表：防禦失真、編輯位移、抗淨化淨增益。

**不跑 GPU、不重算任何數字**，只彙總已存的 CSV。

三張表分別回答一件事
────────────────────────────────────────────────────────────────────
1. **防禦圖付了多少失真** —— `fid_*`：LPIPS／SSIM／PSNR／VIFp／DISTS／L∞／RMS。
   七項一起報是 `DECISIONS.md` 的統一清單，缺欄位就拋錯不容許靜默省略。
2. **編輯輸出被推開多少** —— `edit_*` 五項 ＋ CLIP／SigLIP 兩個語意讀數。
   位移與語意並列，因為兩者對「防禦成功」的預測力不同（前者 AUC 0.983、
   後者是人眼標記校準過的門檻）。
3. **抗淨化的淨增益** —— **一律扣空白地板**（`DECISIONS.md`）：淨化算子自己
   就會把編輯推開，不扣掉它，「淨化後位移較大」無法排除「該算子本來就把編輯
   推得比較開」這個平庸解釋。

**全部用淨增益（差值），不用比例。** 比值在分母塌陷時不可解讀
（`FND-037`／`FND-039` 量到保留率與分母的相關係數 −0.83／−0.900），而本批
有多個弱工作點，分母塌陷不是理論上的顧慮。

用法：
    python scripts/mainline_tables.py --defense runs/ip2p_mainline \\
        --purify runs/ip2p_mainline_purify --out runs/ip2p_mainline/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import write_csv  # noqa: E402

FID = ["fid_dists", "fid_lpips", "fid_psnr", "fid_ssim", "fid_vif_p",
       "fid_linf", "fid_rms"]
EDIT = ["edit_lpips", "edit_dists", "edit_psnr", "edit_ssim", "edit_vif_p",
        "edit_clip_sim", "edit_siglip_sim"]
# 報表上的順序：本方法在前、對照組在後，同族相鄰。
ORDER = ["ours_ph_q", "ours_pg_q", "ours_ph_q20", "ours_pg_q20",
         "ours_ph_n", "ours_pg_n", "ours_pg_m",
         "ours_ph_qd60", "ours_ph_qd45", "ours_ph_qd35",
         "dct_native", "dct_aj85", "dct_aj75",
         "dct_aj85_eps1.5", "dct_aj85_eps2.2",
         "dct_aj85_eps3.2", "dct_aj85_eps4.5",
         "dct_aj50_eps0.22", "dct_aj50_eps0.40", "dct_aj50_eps0.65",
         "dct_aj30_eps0.13", "dct_aj30_eps0.25", "dct_aj30_eps0.42"]
LABEL = {
    "ours_ph_q": "本方法 純相位 ＋量化 r0.9",
    "ours_pg_q": "本方法 相位+增益 ＋量化 r0.9",
    "ours_ph_q20": "本方法 純相位 ＋量化 r=pi",
    "ours_pg_q20": "本方法 相位+增益 ＋量化 r2.0",
    "ours_ph_n": "本方法 純相位 無量化 r0.9",
    "ours_pg_n": "本方法 相位+增益 無量化 r0.9",
    "ours_pg_m": "本方法 相位+增益 無量化 r2.0",
    "dct_native": "DCT-Shield 原生 q0.95",
    "dct_aj85": "DCT-Shield 抗JPEG q0.85",
    "dct_aj75": "DCT-Shield 抗JPEG q0.75",
    # 預算匹配組：把 DCT-Shield 抗 JPEG 版的 eps 拉高到我方的失真高度。
    # **這不是論文設定**（§6.3 的 eps = 1.0），一律標「預算匹配」，不可以
    # 拿去代表 DCT-Shield 的論文結果。
    "dct_aj85_eps1.5": "DCT-Shield 預算匹配 eps1.5",
    "dct_aj85_eps2.2": "DCT-Shield 預算匹配 eps2.2",
    "dct_aj85_eps3.2": "DCT-Shield 預算匹配 eps3.2",
    "dct_aj85_eps4.5": "DCT-Shield 預算匹配 eps4.5",
    # 品質旗鈕包絡：兩邊各自把自己的壓縮品質旗鈕往下放，再逐一攻擊品質比較。
    # **對手那六個的 eps 全部 < 1**，DCT-Shield §4.2 的抗 JPEG 條件因此失效
    # （`DCTShieldSpec` 自動標 `modified_from_paper`）——那不是我們改的，是
    # 「低 Q_alg 又要留在失真帶內」這個要求逼出來的，理由見
    # `scripts/mainline_quality_envelope.sh` 的檔頭。
    "ours_ph_qd60": "本方法 純相位 ＋量化 QD0.60 r0.9",
    "ours_ph_qd45": "本方法 純相位 ＋量化 QD0.45 r0.9",
    "ours_ph_qd35": "本方法 純相位 ＋量化 QD0.35 r0.9",
    "dct_aj50_eps0.22": "DCT-Shield q0.50 eps0.22（eps<1）",
    "dct_aj50_eps0.40": "DCT-Shield q0.50 eps0.40（eps<1）",
    "dct_aj50_eps0.65": "DCT-Shield q0.50 eps0.65（eps<1）",
    "dct_aj30_eps0.13": "DCT-Shield q0.30 eps0.13（eps<1）",
    "dct_aj30_eps0.25": "DCT-Shield q0.30 eps0.25（eps<1）",
    "dct_aj30_eps0.42": "DCT-Shield q0.30 eps0.42（eps<1）",
}
PUR_ORDER = ["identity", "jpeg90", "jpeg75", "jpeg50", "jpeg30",
             "blur1", "blur2", "crop_resize0.1", "crop_resize0.15"]


def read(path: Path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if not vals:
        raise KeyError(f"欄位 {key} 全空——這一批的 CSV 與預期的欄位不符")
    return st.fmean(vals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defense", type=Path, required=True)
    ap.add_argument("--purify", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 表一與表二：防禦失真、編輯位移 --------------------------------
    fid_rows, edit_rows, per_image = [], [], {}
    for tag in ORDER:
        p = args.defense / tag / "results.csv"
        if not p.exists():
            continue
        rr = read(p)
        if not rr:
            continue
        per_image[tag] = rr
        nb = sum(1 for r in rr if r.get("blocked") == "True")
        base = {"tag": tag, "label": LABEL.get(tag, tag), "n": len(rr)}
        fid_rows.append({**base, **{k: round(mean(rr, k), 5) for k in FID}})
        edit_rows.append({**base, **{k: round(mean(rr, k), 5) for k in EDIT},
                          "siglip_blocked": f"{nb}/{len(rr)}"})
    write_csv(args.out / "defense_fidelity.csv", fid_rows)
    write_csv(args.out / "edit_displacement.csv", edit_rows)

    # ---- 表三：抗淨化淨增益（扣地板） ----------------------------------
    floor, cells = {}, {}
    for p in sorted(args.purify.glob("*.csv")):
        stem = p.stem
        tag = stem.rsplit("_", 1)[0]
        for r in read(p):
            key = (r["image"], r["purifier"])
            if tag == "floor":
                floor.setdefault(key, float(r["effect_mean"]))
            else:
                cells.setdefault((tag, r["purifier"]), []).append(
                    (r["image"], float(r["effect_mean"])))

    gain_rows, dropped = [], []
    for (tag, pur), vals in cells.items():
        keep = []
        for img, eff in vals:
            if (img, pur) not in floor:
                dropped.append(f"{tag}/{img}/{pur}：缺地板")
                continue
            keep.append(eff - floor[(img, pur)])
        if not keep:
            continue
        gain_rows.append({
            "tag": tag, "label": LABEL.get(tag, tag), "purifier": pur,
            "n": len(keep), "net_gain": round(st.fmean(keep), 5),
            "net_gain_sd": round(st.stdev(keep) if len(keep) > 1 else 0.0, 5),
            "floor": round(st.fmean(floor[(i, pur)] for i, _ in vals
                                    if (i, pur) in floor), 5),
            "wins_over_floor": sum(1 for g in keep if g > 0),
        })
    gain_rows.sort(key=lambda r: (ORDER.index(r["tag"]) if r["tag"] in ORDER else 99,
                                  PUR_ORDER.index(r["purifier"])
                                  if r["purifier"] in PUR_ORDER else 99))
    write_csv(args.out / "net_gain.csv", gain_rows)

    # ---- 給報告頁的 JSON ----------------------------------------------
    grid = {}
    for r in gain_rows:
        grid.setdefault(r["tag"], {})[r["purifier"]] = r["net_gain"]
    (args.out / "tables.json").write_text(json.dumps({
        "fidelity": fid_rows, "edit": edit_rows,
        "gain": grid, "purifiers": PUR_ORDER, "order": ORDER, "label": LABEL,
    }, ensure_ascii=False), encoding="utf-8")

    # ---- 印出來 --------------------------------------------------------
    print("表一 · 防禦圖的失真（10 張平均）")
    print(f"{'條件':<26s}" + "".join(f"{k[4:]:>9s}" for k in FID))
    for r in fid_rows:
        print(f"{r['label']:<26s}" + "".join(f"{r[k]:9.4f}" for k in FID))

    print("\n表二 · 編輯輸出的位移與語意")
    print(f"{'條件':<26s}" + "".join(f"{k[5:]:>10s}" for k in EDIT) + f"{'擋下':>8s}")
    for r in edit_rows:
        print(f"{r['label']:<26s}" + "".join(f"{r[k]:10.4f}" for k in EDIT)
              + f"{r['siglip_blocked']:>8s}")

    print("\n表三 · 抗淨化的淨增益（扣空白地板，**不是比例**）")
    print(f"{'條件':<26s}" + "".join(f"{p.replace('crop_resize','crop'):>9s}"
                                     for p in PUR_ORDER))
    for tag in ORDER:
        if tag not in grid:
            continue
        print(f"{LABEL.get(tag, tag):<26s}"
              + "".join(f"{grid[tag].get(p, float('nan')):9.4f}" for p in PUR_ORDER))
    if dropped:
        print(f"\n被排除的格子（缺地板）：{len(dropped)} 個")
        for d in dropped[:8]:
            print("  " + d)
    print(f"\n表：{args.out}")


if __name__ == "__main__":
    main()
