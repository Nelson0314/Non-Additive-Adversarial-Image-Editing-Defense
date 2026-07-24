"""report_v2：v2 分析報告產生器 — 逐方法、附標準差、絕對 lpips 主表、
淨化強度客觀排序、交叉分析、hybrid 假設檢驗、三項判準檢核。

輸入既有的 stage1 / stage2 / （選用）stage0 run 目錄，產出：
- RESULTS_v2.md：可直接用於 meeting 的完整報告（表 1~5 + 交叉分析 + 保留事項）
- report_v2_tables.csv：上述表格之機器可讀彙整
- cross_lpips_vs_strength.png：絕對 lpips vs 淨化強度（呈現可能的交叉）

設計要點（見專案 v2 spec）：
- 主表為**淨化後絕對 lpips**（= stage2 results.csv 之 lpips 欄，淨化後殘存的
  防禦強度），非相對 drop；相對 drop 僅作輔助表。
- 淨化強度軸 = LPIPS(purified_protected, protected)，即「淨化把受保護影像改動
  多少」，在所有方法上平均後排序；**不以任何方法的 drop 當強度**（循環論證）。
- 一律逐方法呈現，**不出現「非加性平均」欄**；群體比較於分析段落另述。
- crop_resize 因幾何改變導致 lpips 基準位移，另列獨立區塊，不參與勝方判定。

用法：
  python scripts/report_v2.py --stage1-dir experiments/stage1/<ts> \
      --stage2-dir experiments/stage2/<ts> [--stage0-dir experiments/stage0/<ts>] \
      [--out RESULTS_v2.md]
"""

import argparse
import csv
import statistics
from pathlib import Path

from common import REPO_ROOT  # noqa: E402（設定 sys.path）

from src.metrics.quality import lpips_distance
from src.utils.io import load_csv, load_image, load_json

# 方法分類與呈現順序（僅列資料中實際存在者）
ADDITIVE = ["pg_enc", "pg_diff"]
NONADD = ["advdiff", "apa", "hybrid"]

# 勝方判定：加性最佳與非加性最佳之絕對 lpips 差距 < 此值視為「持平」
WIN_TOL = 0.02
# crop_resize 系列不參與強度排序與勝方判定（幾何改變使 lpips 基準位移）
CROP_FAMILY = "crop_resize"


def agg(values):
    """回傳 (mean, std, n)；std 為樣本標準差（n<2 時為 0）。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return m, s, len(vals)


def fmt(mean, std, nd=4):
    if mean is None:
        return "—"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def col_floats(rows, col):
    out = []
    for r in rows:
        v = r.get(col)
        if v in (None, ""):
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def present(methods_in_data, order):
    return [m for m in order if m in methods_in_data]


def load_fairness(stage0_dir):
    """stage0 fairness.csv → {method: {"type","lpips":(m,s,n),"psnr":...,"linf":...}}。"""
    path = Path(stage0_dir) / "fairness.csv"
    if not path.exists():
        return None
    rows = load_csv(path)
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    out = {}
    for m, rs in by_method.items():
        out[m] = {
            "type": rs[0].get("type", ""),
            "lpips": agg(col_floats(rs, "lpips")),
            "psnr": agg(col_floats(rs, "psnr")),
            "linf": agg(col_floats(rs, "linf")),
        }
    return out


def compute_purification_strength(s1_dir, s2_dir, manifest, labels, methods):
    """對每個 (label, method, image) 算 LPIPS(purified, protected)。
    回傳 (strength_by_label, per_method_by_label)：
      strength_by_label[label] = (mean, std, n)  # 所有方法所有影像平均
      per_method[label][method] = (mean, std, n)
    找不到影像者略過（回報 n）。"""
    s1_dir, s2_dir = Path(s1_dir), Path(s2_dir)
    prot_paths = manifest["protected"]  # method -> image_id -> {"path": ...}
    samples = manifest["samples"]
    all_vals = {label: [] for label in labels}
    per_method = {label: {m: [] for m in methods} for label in labels}
    prot_cache = {}
    for mkey in methods:
        for sample in samples:
            iid = sample["image_id"]
            sid = iid.replace("/", "__")
            rel = prot_paths.get(mkey, {}).get(iid, {}).get("path")
            if rel is None:
                continue
            prot_path = s1_dir / rel
            if not prot_path.exists():
                continue
            ckey = (mkey, iid)
            if ckey not in prot_cache:
                prot_cache[ckey] = load_image(prot_path)
            protected = prot_cache[ckey]
            for label in labels:
                pur_path = s2_dir / "purified" / mkey / label / f"{sid}.png"
                if not pur_path.exists():
                    continue
                d = lpips_distance(load_image(pur_path), protected)
                all_vals[label].append(d)
                per_method[label][mkey].append(d)
    strength = {label: agg(all_vals[label]) for label in labels}
    per_method_agg = {label: {m: agg(per_method[label][m]) for m in methods}
                      for label in labels}
    return strength, per_method_agg


def winner(label_abs, methods_present):
    """回傳 (勝方字串, best_add, best_non)；以絕對 lpips 越高越好。"""
    add = [(m, label_abs[m][0]) for m in methods_present
           if m in ADDITIVE and label_abs[m][0] is not None]
    non = [(m, label_abs[m][0]) for m in methods_present
           if m in NONADD and label_abs[m][0] is not None]
    if not add or not non:
        return "資料不足", None, None
    best_add = max(add, key=lambda t: t[1])
    best_non = max(non, key=lambda t: t[1])
    diff = best_non[1] - best_add[1]
    if abs(diff) < WIN_TOL:
        return f"持平（{best_add[0]}≈{best_non[0]}）", best_add, best_non
    if diff > 0:
        return f"非加性（{best_non[0]}）", best_add, best_non
    return f"加性（{best_add[0]}）", best_add, best_non


def main():
    parser = argparse.ArgumentParser(description="v2 分析報告產生器")
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--stage2-dir", required=True)
    parser.add_argument("--stage0-dir", default=None)
    parser.add_argument("--out", default=None, help="RESULTS_v2.md 輸出路徑（預設寫入 stage2 dir）")
    parser.add_argument("--edit-method", default="sdedit", help="主要條件（有 DAYN 錨點）")
    args = parser.parse_args()

    s1_dir, s2_dir = Path(args.stage1_dir), Path(args.stage2_dir)
    manifest = load_json(s1_dir / "manifest.json")
    s1_rows = load_csv(s1_dir / "results.csv")
    s2_rows = load_csv(s2_dir / "results.csv")
    out_path = Path(args.out) if args.out else s2_dir / "RESULTS_v2.md"

    edit = args.edit_method
    s1_rows = [r for r in s1_rows if r["edit_method"] == edit]
    s2_rows = [r for r in s2_rows if r["edit_method"] == edit]

    methods_in_data = list(dict.fromkeys(r["method"] for r in s1_rows))
    methods = present(methods_in_data, ADDITIVE + NONADD)

    # 淨化標籤（保序）與家族對照
    labels = list(dict.fromkeys(r["purify"] for r in s2_rows))
    family_of = {r["purify"]: r.get("purify_family", "") for r in s2_rows}
    noncrop = [lb for lb in labels if family_of.get(lb) != CROP_FAMILY]
    crop = [lb for lb in labels if family_of.get(lb) == CROP_FAMILY]

    metric_keys = [k for k in ("lpips", "psnr", "ssim", "vifp", "fsim", "clip")
                   if s1_rows and k in s1_rows[0]]

    # --- 聚合 ---
    # clean（stage1）：per method → metric 之 (mean,std,n)
    clean_by_method = {}
    for m in methods:
        mrows = [r for r in s1_rows if r["method"] == m]
        clean_by_method[m] = {k: agg(col_floats(mrows, k)) for k in metric_keys}

    # stage2：per (label, method) → 絕對 lpips 與 drop_lpips
    abs_lpips = {lb: {} for lb in labels}
    drop_lpips = {lb: {} for lb in labels}
    for lb in labels:
        for m in methods:
            mrows = [r for r in s2_rows if r["purify"] == lb and r["method"] == m]
            abs_lpips[lb][m] = agg(col_floats(mrows, "lpips"))
            drop_lpips[lb][m] = agg(col_floats(mrows, "drop_lpips"))

    # 淨化強度（非 crop 才排序；crop 仍算但獨立呈現）
    strength, strength_pm = compute_purification_strength(
        s1_dir, s2_dir, manifest, labels, methods)
    ordered = sorted(noncrop, key=lambda lb: (strength[lb][0] is None,
                                              strength[lb][0] or 0.0))

    fairness = load_fairness(args.stage0_dir) if args.stage0_dir else None

    # --- 判準檢核 ---
    # 判準 1：公平性（非加性 LPIPS(prot,orig) 不顯著高於加性）
    crit1_result, crit1_basis = _check_fairness(fairness, methods)
    # 強淨化 = 非 crop 中強度 >= 中位數者
    strong_labels = _strong_set(ordered, strength)
    # 判準 2：對稱比較（強淨化下 ≥1 非加性同時 > pg_enc 且 > pg_diff）
    crit2_result, crit2_basis, sat_labels = _check_symmetry(
        strong_labels, abs_lpips, methods)
    # 判準 3：一致性（判準 2 於多個強淨化成立）
    crit3_result, crit3_basis = _check_consistency(sat_labels, strong_labels)

    placeholder = _is_placeholder(manifest, s1_dir)

    # --- 產出 Markdown ---
    md = _render_md(
        manifest, methods, metric_keys, clean_by_method, ordered, crop,
        strength, strength_pm, abs_lpips, drop_lpips, fairness, edit,
        (crit1_result, crit1_basis), (crit2_result, crit2_basis),
        (crit3_result, crit3_basis), placeholder)
    out_path.write_text(md, encoding="utf-8")

    _write_tables_csv(s2_dir / "report_v2_tables.csv", methods, ordered, crop,
                      strength, abs_lpips, drop_lpips)
    _plot_cross(s2_dir / "cross_lpips_vs_strength.png", ordered, strength,
                abs_lpips, methods)
    print(f"report_v2 完成 → {out_path}")
    print(f"判準1 公平性={crit1_result} | 判準2 對稱={crit2_result} | 判準3 一致={crit3_result}")


def _strong_set(ordered, strength):
    vals = [strength[lb][0] for lb in ordered if strength[lb][0] is not None]
    if not vals:
        return []
    med = statistics.median(vals)
    return [lb for lb in ordered
            if strength[lb][0] is not None and strength[lb][0] >= med]


def _check_fairness(fairness, methods):
    if not fairness:
        return "資料不足", "未提供 stage0 fairness.csv（Phase A 未跑或未指定 --stage0-dir）"
    add_l = [fairness[m]["lpips"][0] for m in methods
             if m in ADDITIVE and m in fairness and fairness[m]["lpips"][0] is not None]
    non = {m: fairness[m]["lpips"][0] for m in methods
           if m in NONADD and m in fairness and fairness[m]["lpips"][0] is not None}
    if not add_l or not non:
        return "資料不足", "加性或非加性之 LPIPS(prot,orig) 缺失"
    add_max = max(add_l)
    worst = max(non.items(), key=lambda kv: kv[1])
    # 「顯著高於」：非加性最大值 > 加性最大值 × 1.15
    if worst[1] > add_max * 1.15:
        return ("不滿足",
                f"非加性 {worst[0]} 之 LPIPS(prot,orig)={worst[1]:.4f} 顯著高於"
                f"加性最大 {add_max:.4f}（>1.15×）：耐淨化優勢可能來自改動更多，"
                f"須重新校準後再跑")
    return ("滿足",
            f"非加性最大 {worst[0]}={worst[1]:.4f} 未顯著高於加性最大 {add_max:.4f}"
            f"（≤1.15×）")


def _check_symmetry(strong_labels, abs_lpips, methods):
    if not strong_labels:
        return "資料不足", "無強淨化標籤（淨化強度未算出）", []
    has_add = all(m in methods for m in ("pg_enc", "pg_diff"))
    if not has_add:
        return "資料不足", "pg_enc 或 pg_diff 缺席，無法對稱比較", []
    sat = []
    for lb in strong_labels:
        pe = abs_lpips[lb].get("pg_enc", (None,))[0]
        pd = abs_lpips[lb].get("pg_diff", (None,))[0]
        if pe is None or pd is None:
            continue
        for m in NONADD:
            v = abs_lpips[lb].get(m, (None,))[0]
            if v is not None and v > pe and v > pd:
                sat.append((lb, m, v, pe, pd))
                break
    if sat:
        detail = "；".join(f"{lb}: {m}={v:.3f} > pg_enc={pe:.3f}, pg_diff={pd:.3f}"
                          for lb, m, v, pe, pd in sat)
        return "滿足", detail, sat
    return ("不滿足",
            "無任何強淨化下有非加性方法同時高於 pg_enc 與 pg_diff", [])


def _check_consistency(sat_labels, strong_labels):
    n_sat = len({lb for lb, *_ in sat_labels})
    n_strong = len(strong_labels)
    if n_strong == 0:
        return "資料不足", "無強淨化標籤"
    if n_sat >= 2 and n_sat >= (n_strong + 1) // 2:
        return "滿足", f"對稱比較於 {n_sat}/{n_strong} 個強淨化成立（過半且≥2）"
    if n_sat >= 1:
        return ("不滿足",
                f"對稱比較僅於 {n_sat}/{n_strong} 個強淨化成立，未達過半／單一手段")
    return "不滿足", f"對稱比較於 0/{n_strong} 個強淨化成立"


def _is_placeholder(manifest, s1_dir):
    """placeholder flag 取自 stage1 之 config_snapshot.yaml。"""
    import yaml
    snap = s1_dir / "config_snapshot.yaml"
    if snap.exists():
        c = yaml.safe_load(snap.read_text(encoding="utf-8"))
        return bool(c.get("base", {}).get("data", {}).get("is_placeholder"))
    return False


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _cls(m):
    return "加性" if m in ADDITIVE else "非加性"


def _render_md(manifest, methods, metric_keys, clean_by_method, ordered, crop,
               strength, strength_pm, abs_lpips, drop_lpips, fairness, edit,
               c1, c2, c3, placeholder):
    n_img = len(manifest["samples"])
    seeds_any = next(iter(manifest.get("seeds", {}).values()), [])
    n_seed = len(seeds_any)
    L = []
    L += ["# RESULTS_v2：非加性 vs 加性防禦 — 淨化耐受性（逐方法、附標準差）", ""]
    L += [f"- 模型：{manifest.get('protect_model')}｜編輯：{edit}（img2img，主要條件）",
          f"- 規模：{n_img} 影像 × {n_seed} seeds｜方法：{', '.join(methods)}",
          "- **標準差為跨影像（seed 已於 stage 內平均）**；n 為該條件之影像×prompt 數。"]
    if placeholder:
        L += ["- **placeholder 資料集**：數值不可與真實資料集直接比較，本輪僅驗方向。"]
    L += [""]

    # 表 1 公平性
    L += ["## 表 1：公平性檢核（stage0，前提）", ""]
    if fairness:
        rows = []
        for m in methods:
            f = fairness.get(m)
            if not f:
                rows.append([m, _cls(m), "—", "—", "—"])
                continue
            linf = f["linf"][0]
            linf_s = f"{linf:.4f}" if linf is not None else "—"
            if m in NONADD:
                linf_s += "（參考）"
            rows.append([m, _cls(m), fmt(*f["lpips"][:2]), fmt(*f["psnr"][:2], nd=2), linf_s])
        L += _table(["方法", "類型", "LPIPS(prot,orig)↓", "PSNR(prot,orig)↑", "L∞ 差異"], rows)
    else:
        L += ["_未提供 stage0 fairness.csv；此表為整份結果之前提，須由 Phase A 補齊。_"]
    L += ["", f"**判準 1（公平性）：{c1[0]}** — {c1[1]}", ""]

    # 表 2 clean 防禦強度
    L += ["## 表 2：Clean 防禦強度（stage1，箭頭為對防禦有利方向）", ""]
    arrow = {"lpips": "↑", "psnr": "↓", "ssim": "↓", "vifp": "↓", "fsim": "↓", "clip": "↓"}
    head = ["方法"] + [f"{k}{arrow.get(k,'')}" for k in metric_keys]
    rows = []
    for m in methods:
        cells = [m] + [fmt(*clean_by_method[m][k][:2]) for k in metric_keys]
        rows.append(cells)
    L += _table(head, rows)
    L += [""]

    # 表 3 主表：淨化後絕對 lpips（依強度排序）
    L += ["## 表 3（主表）：淨化後絕對 lpips（越高＝防禦殘存越多）", "",
          "淨化手段依客觀強度 `LPIPS(purified, protected)`（所有方法平均）由弱至強排序。", ""]
    head = ["淨化手段", "強度", *methods, "勝方"]
    rows = []
    # clean 參考列
    rows.append(["（clean 參考）", "0.000",
                 *[fmt(*clean_by_method[m]["lpips"][:2]) for m in methods], "—"])
    for lb in ordered:
        st = strength[lb][0]
        st_s = f"{st:.3f}" if st is not None else "—"
        win, _, _ = winner(abs_lpips[lb], methods)
        rows.append([lb, st_s, *[fmt(*abs_lpips[lb][m][:2]) for m in methods], win])
    L += _table(head, rows)
    L += [""]
    if crop:
        L += ["### crop_resize（獨立區塊，不參與勝方判定）", "",
              "_幾何改變使 lpips 基準位移，drop_valid=False；不與 jpeg/blur/advclean 混讀。_", ""]
        rows = []
        for lb in crop:
            st = strength[lb][0]
            st_s = f"{st:.3f}" if st is not None else "—"
            rows.append([lb, st_s, *[fmt(*abs_lpips[lb][m][:2]) for m in methods]])
        L += _table(["淨化手段", "強度", *methods], rows)
        L += [""]

    # 表 4 輔助：相對 drop
    L += ["## 表 4（輔助）：相對 drop_lpips（移除比例；起點不同故不作主要判斷）", ""]
    head = ["淨化手段", *methods]
    rows = []
    for lb in ordered + crop:
        rows.append([lb, *[fmt(*drop_lpips[lb][m][:2]) for m in methods]])
    L += _table(head, rows)
    L += [""]

    # 交叉分析
    L += ["## 交叉分析（依強度由弱至強）", ""]
    prev = None
    cross_note = []
    for lb in ordered:
        win, ba, bn = winner(abs_lpips[lb], methods)
        side = "非加性" if win.startswith("非加性") else ("加性" if win.startswith("加性") else "持平")
        if prev is not None and side != prev and side in ("加性", "非加性") and prev in ("加性", "非加性"):
            cross_note.append(f"- 交叉發生於 **{lb}** 附近（強度 {strength[lb][0]:.3f}）："
                              f"由「{prev}佔優」轉為「{side}佔優」")
        prev = side
    if cross_note:
        L += ["偵測到勝方交叉："] + cross_note
    else:
        L += ["- 於各強度區間未偵測到加性/非加性勝方交叉（勝方大致一致）。"]
    L += ["", f"**判準 2（對稱比較）：{c2[0]}** — {c2[1]}",
          f"**判準 3（一致性）：{c3[0]}** — {c3[1]}", ""]

    # hybrid 假設檢驗
    L += ["## hybrid 假設檢驗（是否優於 advdiff 與 apa 中較佳者）", ""]
    L += _hybrid_check(methods, clean_by_method, ordered, abs_lpips, strength)
    L += [""]

    # 判準檢核彙整
    L += ["## 判準檢核彙整", ""]
    L += _table(["判準", "結果", "依據"], [
        ["1. 公平性", c1[0], c1[1]],
        ["2. 對稱比較", c2[0], c2[1]],
        ["3. 一致性", c3[0], c3[1]],
    ])
    L += ["", "> 方向成立需三項同時滿足；任一不滿足即如實記錄，未調整判準或挑選子集。", ""]

    # 保留事項
    L += ["## 保留事項", ""]
    if placeholder:
        L += ["- placeholder 資料集：數值僅驗方向，不可與真實資料集直接比較。"]
    L += [f"- 樣本量：{n_img} 影像 × {n_seed} seeds；標準差為跨影像（seed 已平均），"
          "非逐 seed，故誤差範圍偏保守。",
          "- 淨化強度以 LPIPS(purified, protected) 定義，不以任何方法的 drop 當強度"
          "（避免循環論證）。",
          "- crop_resize 因幾何改變另列，不參與勝方判定。"]
    return "\n".join(L) + "\n"


def _hybrid_check(methods, clean_by_method, ordered, abs_lpips, strength):
    if "hybrid" not in methods or not all(m in methods for m in ("advdiff", "apa")):
        return ["_advdiff / apa / hybrid 未同時存在，無法檢驗。_"]
    out = []
    # clean 維度
    h = clean_by_method["hybrid"]["lpips"][0]
    a = clean_by_method["advdiff"]["lpips"][0]
    p = clean_by_method["apa"]["lpips"][0]
    best = max(a, p)
    better_name = "advdiff" if a >= p else "apa"
    if h is not None and h > best:
        out.append(f"- **clean 強度**：hybrid={h:.4f} > max(advdiff={a:.4f}, apa={p:.4f})＝"
                   f"優於較佳者 {better_name}。")
    else:
        out.append(f"- **clean 強度**：hybrid={h:.4f} 未超過 max(advdiff={a:.4f}, apa={p:.4f})＝"
                   f"{best:.4f}（{better_name}）；假設「兼具兩者長處」在 clean 維度**不成立**，"
                   f"表現接近性質平均。")
    # 耐淨化維度（強淨化下 hybrid 絕對 lpips 是否 > max(advdiff, apa)）
    strong = _strong_set(ordered, strength)
    win_cnt = 0
    total = 0
    for lb in strong:
        hv = abs_lpips[lb].get("hybrid", (None,))[0]
        av = abs_lpips[lb].get("advdiff", (None,))[0]
        pv = abs_lpips[lb].get("apa", (None,))[0]
        if None in (hv, av, pv):
            continue
        total += 1
        if hv > max(av, pv):
            win_cnt += 1
    if total:
        verdict = "成立" if win_cnt > total / 2 else "不成立"
        out.append(f"- **耐淨化**：強淨化下 hybrid 絕對 lpips 高於 max(advdiff, apa) 於 "
                   f"{win_cnt}/{total} 個手段；假設**{verdict}**。")
    else:
        out.append("- **耐淨化**：強淨化資料不足，無法判定。")
    return out


def _write_tables_csv(path, methods, ordered, crop, strength, abs_lpips, drop_lpips):
    rows = []
    for lb in ordered + crop:
        st = strength[lb][0]
        for m in methods:
            am, asd, an = abs_lpips[lb][m]
            dm, dsd, dn = drop_lpips[lb][m]
            rows.append({
                "purify": lb, "strength": st, "method": m,
                "abs_lpips_mean": am, "abs_lpips_std": asd, "n": an,
                "drop_lpips_mean": dm, "drop_lpips_std": dsd,
                "in_strength_order": lb in ordered,
            })
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _plot_cross(path, ordered, strength, abs_lpips, methods):
    xs = [strength[lb][0] for lb in ordered]
    if not xs or any(x is None for x in xs):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        ys = [abs_lpips[lb][m][0] for lb in ordered]
        style = "--o" if m in ADDITIVE else "-s"
        ax.plot(xs, ys, style, label=m)
    ax.set_xlabel("purification strength = LPIPS(purified, protected)")
    ax.set_ylabel("abs lpips after purify (higher = defense survives)")
    ax.set_title("Defense survival vs purification strength")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
