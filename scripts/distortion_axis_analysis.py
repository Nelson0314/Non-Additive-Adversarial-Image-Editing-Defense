"""兩個失真軸（LPIPS 與 DISTS）對「防禦效果」的預測力比較。**不跑 GPU。**

為什麼要這支
────────────────────────────────────────────────────────────────────
`DECISIONS.md` 要求等 DISTS 與等 LPIPS 兩個錨點都要報，理由是「實測兩者對
同一組影像的判定經常相反」。這支把「經常相反」量化成一個可查的數：把 IP2P
線上**全部**已跑過的工作點放進同一張圖，問「每單位失真換到多少防禦效果」
這個比值在各方法之間有多穩。

若某個軸上各方法的比值一致，該軸就是一把公平的尺——兩個方法的差別只剩
「站得多遠」，而那是可達性問題。若比值差好幾倍，該軸對不同型態的擾動開的
是不同的價，用它當錨點會把定價差異讀成方法差異。

讀數
────────────────────────────────────────────────────────────────────
    slope_through_origin   sum(x*y)/sum(x*x)，即通過原點的最小平方斜率
    slope_ols / intercept  一般最小平方（截距不強制為零）
    cv_across_conditions   各條件的原點斜率之變異係數 = sd / mean

**原點斜率會隨強度遞減**（同一方法在低失真處效率較高），故跨方法比較必須
限制在重疊的失真區間內，否則比到的是「誰的掃描點比較弱」。`--x-min` 就是
做這件事的，預設 0 表示不限制，報表會同時給限制前後兩組數字。

用法：
    python scripts/distortion_axis_analysis.py \
        --run runs/ip2p_fair_comparison --out runs/distortion_axis_analysis
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import write_csv  # noqa: E402

EFFECT_KEY = "edit_lpips"
AXES = ("fid_lpips", "fid_dists")
MIN_IMAGES = 20          # 少於這個張數的工作點不進迴歸，避免單張主導

# 每一個會改變擾動的設定欄位，以及該欄位**還不存在時**批次必然用的值。
#
# 為什麼需要這張表：`runs/ip2p_fair_comparison` 底下共存六個 schema（26／27／
# 30／32／33／37 欄），一路加欄位加上來且嚴格巢狀。直接合併會把不同設定的
# 列併成一組而**沒有任何症狀**——這正是 r_min 當初漏記的失效方式。
#
# 表裡的值不是「看起來合理的預設」，而是**被構造決定的**：欄位不存在意味著
# 該功能當時還沒實作，於是它只可能是關閉的那個值。唯一的例外是 `r_min`，
# 它在成為旗標之前是寫死的 0.12（見 METHOD.md 的定案表與 FND-042）。
#
# 遇到不在這張表裡的新欄位一律拋錯，不猜。
SETTING_DEFAULTS = {
    "condition": None,          # None = 必須存在，沒有歷史預設
    "radius": None,
    "r_min": "0.12",            # 成為旗標之前寫死的值
    "r_max": "inf",             # 帶通上界，加入前是純高通
    "block": "32",
    "hop": "16",                # 恆為 block//2，成為欄位前沒有別的可能
    "gl_iters": "0",
    "pixel_gate_sigma": "0.0",
    "quantile": "0.5",
    "gate_edge_power": "1.0",
    "loss": "encoder_target",
    "gain_ratio": "0.0",
    "purify_aware": "none",
    "defense_steps": "",        # 空字串 = 未記錄，見下方的 note 欄
    # DJSMA 那一支在改名為現行實作之前用的是另一組旋鈕。留在鍵裡是為了讓
    # 那四個 topk 工作點不被併成一組；同時標記它們來自已被取代的實作。
    "wm_topk": "",
    "wm_eps": "",
    "wm_block_frac": "",
    "wm_q_attack": "",
    "wm_tau": "",
    "wm_mu": "",
    "wm_q_embed": "",
}

# 這些欄位只要出現，該列就來自已被取代的浮水印實作（現行的 `dct_wm` 走
# DJSMASpec，旋鈕是 tau／mu／diagonals）。那批數字無法由現行程式重跑。
SUPERSEDED_MARKERS = ("wm_topk", "wm_eps", "wm_block_frac", "wm_q_attack")


def load_rows(root: Path, attacker: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(root.glob("*/*/results.csv")) + sorted(root.glob("*/results.csv")):
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("attacker") != attacker:
                    continue
                r["__src"] = str(path.relative_to(root))
                rows.append(r)
    if not rows:
        raise SystemExit(f"{root} 底下沒有 attacker={attacker} 的列")
    return rows


def settings_of(row: dict) -> Dict[str, str]:
    """一列的完整設定，缺席的欄位由 `SETTING_DEFAULTS` 補上歷史值。

    **欄位缺席不等於可以忽略**：缺席時該功能還沒實作，故值是被決定的，補上
    它就能與新批次並排。反過來，出現了表裡沒有的設定欄位就直接拋錯——那代表
    有人加了旋鈕卻沒有登記，而合併之後不會有症狀。
    """
    out: Dict[str, str] = {}
    for key, fallback in SETTING_DEFAULTS.items():
        if key in row:
            out[key] = row[key]
        elif fallback is None:
            raise SystemExit(f"{row['__src']} 缺必要欄位 {key}")
        else:
            out[key] = fallback
    return out


def work_points(rows: Sequence[dict]) -> List[dict]:
    """摺成工作點。一個工作點 = 一組完全相同的設定，跨影像取平均。"""
    known = set(SETTING_DEFAULTS) | {
        "image", "attacker", "instruction", "task", "unreachable",
        "modified_from_paper", "edit_steps", "s_t", "s_i", "edit_seed",
        "total_seconds", "seconds", "__src", "wm_diagonals",
    }
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        stray = [k for k in r
                 if k not in known and not k.startswith(("fid_", "edit_"))]
        if stray:
            raise SystemExit(
                f"{r['__src']} 有未登記的欄位 {stray}——若那是新的旋鈕，"
                "請加進 SETTING_DEFAULTS 並寫明它還不存在時的值")
        settings = settings_of(r)
        groups.setdefault(tuple(settings.items()), []).append(r)

    out: List[dict] = []
    for key, group in groups.items():
        n = len({g["image"] for g in group})
        if n < MIN_IMAGES:
            continue
        point = dict(key)
        point["n_images"] = n
        point["superseded_implementation"] = any(
            m in group[0] for m in SUPERSEDED_MARKERS)
        point[EFFECT_KEY] = statistics.fmean(float(g[EFFECT_KEY]) for g in group)
        for axis in AXES:
            point[axis] = statistics.fmean(float(g[axis]) for g in group)
        out.append(point)
    return sorted(out, key=lambda p: (p["condition"], p[AXES[0]]))


def _method(condition: str) -> str:
    """把條件歸到「方法」——同一個方法的不同旋鈕仍是同一把尺量的東西。"""
    if condition.startswith("dct_shield"):
        return "dct_shield"
    if condition.startswith("dct_wm"):
        return "djsma"
    if condition.startswith("advdrop"):
        return "advdrop"
    if condition == "phase_rand":
        return "phase_rand"
    return "texture_rephase"


def fit(points: Sequence[dict], axis: str) -> dict:
    xs = [p[axis] for p in points]
    ys = [p[EFFECT_KEY] for p in points]
    through_origin = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    row = {"axis": axis, "n_points": len(points),
           "slope_through_origin": round(through_origin, 4),
           "x_min": round(min(xs), 5), "x_max": round(max(xs), 5)}
    if len(points) >= 3:
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx
        row["slope_ols"] = round(slope, 4)
        row["intercept_ols"] = round(my - slope * mx, 5)
        syy = sum((y - my) ** 2 for y in ys)
        row["r2"] = round((sxy ** 2) / (sxx * syy), 4) if syy > 0 else ""
    else:
        row["slope_ols"] = row["intercept_ols"] = row["r2"] = ""
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--attacker", default="instruct-pix2pix")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    points = work_points(load_rows(args.run, args.attacker))
    write_csv(args.out / "work_points.csv", [
        {**p, "method": _method(p["condition"]),
         **{f"ratio_{a}": round(p[EFFECT_KEY] / p[a], 4) for a in AXES}}
        for p in points])

    fits: List[dict] = []
    for axis in AXES:
        # 兩個 x 區間：全部工作點，以及**各方法共同覆蓋**的重疊區間。
        # 原點斜率隨強度遞減，只看全部工作點會把「誰的掃描點比較弱」讀成
        # 「誰的效率比較高」。重疊區間由資料自己決定，不需要為每個軸手動
        # 指定門檻（兩個軸的量級本來就差一個數量級）。
        spans = {}
        for point in points:
            lo, hi = spans.get(_method(point["condition"]), (None, None))
            v = point[axis]
            spans[_method(point["condition"])] = (
                v if lo is None else min(lo, v),
                v if hi is None else max(hi, v))
        usable = [v for v in spans.values() if v[0] != v[1]]
        overlap_lo = max(v[0] for v in usable) if usable else 0.0
        overlap_hi = min(v[1] for v in usable) if usable else float("inf")
        cuts = {("all", 0.0, float("inf"))}
        if overlap_lo < overlap_hi:
            cuts.add(("overlap", overlap_lo, overlap_hi))
        for label, cut, cut_hi in sorted(cuts):
            pool = [p for p in points if cut <= p[axis] <= cut_hi]
            by_method: Dict[str, List[dict]] = {}
            for p in pool:
                by_method.setdefault(_method(p["condition"]), []).append(p)
            slopes = {}
            for method, group in sorted(by_method.items()):
                if len(group) < 2:
                    continue
                row = fit(group, axis)
                row.update({"method": method, "window": label,
                            "x_cut_lo": round(cut, 5),
                            "x_cut_hi": ("" if cut_hi == float("inf")
                                         else round(cut_hi, 5))})
                fits.append(row)
                slopes[method] = row["slope_through_origin"]
            if len(slopes) >= 2:
                vals = list(slopes.values())
                fits.append({
                    "method": "__across_methods__", "axis": axis,
                    "window": label, "x_cut_lo": round(cut, 5),
                    "x_cut_hi": ("" if cut_hi == float("inf")
                                 else round(cut_hi, 5)),
                    "n_points": len(slopes),
                    "slope_through_origin": round(statistics.fmean(vals), 4),
                    "cv_across_conditions": round(
                        statistics.stdev(vals) / statistics.fmean(vals), 4),
                    "spread_ratio": round(max(vals) / min(vals), 3),
                })
    write_csv(args.out / "axis_fits.csv", fits)

    print(f"工作點 {len(points)} 個 → {args.out}")
    for row in fits:
        if row["method"] != "__across_methods__":
            continue
        print(f"  {row['axis']:11s} {row['window']:8s} "
              f"[{row['x_cut_lo']}, {row['x_cut_hi'] or '∞'}]  "
              f"跨方法變異係數 {row['cv_across_conditions']:.3f}  "
              f"最大/最小 {row['spread_ratio']:.2f}x")


if __name__ == "__main__":
    main()
