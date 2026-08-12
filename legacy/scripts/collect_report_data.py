"""把三個條件的資料整理成報告要用的形狀，輸出 JSON。"""
import csv
import glob
import json
import os
import statistics as st

ARMS = [("attn", "s3t20_pj_merged", "apa_pj", "A · 注意力抑制"),
        ("target", "s3t20_tj_merged", "apa_tj", "B · 目標輸出"),
        ("random", "s3t20_r_merged", "Ra", "C · 隨機對照")]
REF = [("photoguard_c", "PhotoGuard-c"), ("dia_r", "DIA-R"), ("mist", "Mist")]
KEEP = [("identity", 0.0), ("blur", 0.25), ("blur", 0.5), ("blur", 0.75),
        ("noise", 0.005), ("noise", 0.01), ("quantize", 128.0),
        ("quantize", 64.0), ("quantize", 32.0), ("quantize", 16.0),
        ("jpeg", 75.0), ("jpeg", 30.0), ("crop_resize", 0.1),
        ("adverse_cleaner", 0.0), ("diffpure", 150.0)]
L1 = ["edit_lpips", "edit_psnr", "edit_ssim", "edit_vif_p", "edit_fsim"]
FID = ["fid_lpips", "fid_dists", "fid_psnr", "fid_acutance_ratio"]


def load(batch, cond):
    d = {}
    for r in csv.DictReader(open(f"runs/{batch}/grid.csv", encoding="utf-8")):
        if r["stage"] == "eval" and r["condition"] == cond:
            d[(r["image_id"], r["purify_kind"], float(r["purify_strength"]),
               int(r["seed"]))] = r
    return d


D = {k: load(b, c) for k, b, c, _ in ARMS}
D.update({k: load("s3t20_merged", k) for k, _ in REF})
out = {"labels": {k: lab for k, _, _, lab in ARMS},
       "ref_labels": dict(REF)}


def sub(d, pur=None, img=None):
    return [v for k, v in d.items()
            if (pur is None or (k[1], k[2]) == pur)
            and (img is None or k[0] == img)]


def agg(rows, keys):
    return {k: st.fmean(float(r[k]) for r in rows) for k in keys}


# 主表：identity 上的位移與保真
out["main"] = {}
for k in list(D):
    rows = sub(D[k], ("identity", 0.0))
    if not rows:
        continue
    a = agg(rows, L1 + FID)
    a["acut_dev"] = st.fmean(abs(1 - float(r["fid_acutance_ratio"])) for r in rows)
    a["dniqe"] = st.fmean(float(r["edit_niqe_b"]) - float(r["edit_niqe_a"])
                          for r in rows)
    a["n"] = len(rows)
    out["main"][k] = a

# 逐影像
out["per_image"] = {}
for k in list(D):
    out["per_image"][k] = {}
    for img in ("horse_00", "horse_03", "woman_03"):
        rows = sub(D[k], ("identity", 0.0), img)
        if rows:
            out["per_image"][k][img] = agg(rows, L1 + FID)

# 逐淨化的位移
out["purify"] = {}
for k in list(D):
    out["purify"][k] = {}
    for p in KEEP:
        rows = sub(D[k], p)
        if rows:
            out["purify"][k][f"{p[0]}_{p[1]:g}"] = st.fmean(
                float(r["edit_lpips"]) for r in rows)

# 配對比較（三個條件互比，同影像同淨化同 seed）
out["paired"] = {}
base = D["attn"]
for k in ("target", "random"):
    pairs = [(base[key], D[k][key]) for key in D[k] if key in base]
    if not pairs:
        continue
    out["paired"][k] = {
        "n": len(pairs),
        "attn": st.fmean(float(x["edit_lpips"]) for x, _ in pairs),
        "other": st.fmean(float(y["edit_lpips"]) for _, y in pairs),
        "win": sum(float(y["edit_lpips"]) > float(x["edit_lpips"])
                   for x, y in pairs) / len(pairs),
    }

# scale_k 與步數
out["train"] = {}
for k, b, c, _ in ARMS:
    e = {}
    for q in glob.glob(f"runs/{b}/{c}/*/meta_tau0.04.json"):
        e[os.path.basename(os.path.dirname(q))] = {
            "scale_k": json.load(open(q, encoding="utf-8"))["scale_k"]}
    for q in glob.glob(f"runs/{b}/{c}/*/meta.json"):
        img = os.path.basename(os.path.dirname(q))
        d = json.load(open(q, encoding="utf-8"))
        e.setdefault(img, {}).update(
            {"steps": d.get("steps_used"),
             "stop": str(d.get("stop_reason"))[:40],
             "seconds": d.get("seconds_optimize"),
             "L_def_final": d.get("final_L_def")})
    out["train"][k] = e

json.dump(out, open(os.path.join(
    os.path.expandvars(r"$TEMP\claude\C--WACV-s3"
                       r"\f97b0be2-7c2c-4175-8705-a671a63a1017\scratchpad"),
    "report_data.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("main:", {k: round(v["edit_lpips"], 4) for k, v in out["main"].items()})
print("paired:", {k: (round(v["attn"], 4), round(v["other"], 4),
                      f"{v['win']*100:.0f}%") for k, v in out["paired"].items()})
print("scale_k:", {k: {i: round(x.get("scale_k", float('nan')), 3)
                       for i, x in v.items()} for k, v in out["train"].items()})
