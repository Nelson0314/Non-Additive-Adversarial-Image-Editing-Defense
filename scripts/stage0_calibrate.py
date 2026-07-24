"""stage0：相似性約束校準 — SPEC.md §6.1、STRUCTURE.md §4.2。

1. 對 PhotoGuard encoder 與 diffusion（κ=0.06）各計算 LPIPS(protected, original)
   平均。L_ref 採 encoder 值（preflight 假設 1 裁定），兩者皆列於輸出；
   若兩者差距 >10%，警告並須重新決定基準（TWCC_CHECKLIST 驗證項）。
2. 對非加性方法掃描各自的相似性控制參數（configs/nonadditive.yaml 之
   stage0_scan；SG 與 GC 獨立範圍，假設 2 裁定），記錄各設定之 LPIPS 平均。
3. 選出使 LPIPS 最接近 L_ref 之設定；**若選中值落在掃描區間端點，
   警告「須擴展範圍重掃」**（首輪範圍為推測值）。
   - L_ref 寫入 configs/nonadditive.yaml 之 common.similarity_budget（僅改該行，保留註解）
   - 各方法選定值寫入 configs/nonadditive_calibrated.yaml（stage1 自動合併）
4. 輸出 calibration.csv 與 calibration_curve.png 至 experiments/stage0/<timestamp>/

用法：python scripts/stage0_calibrate.py [--smoke] [--model M] [--max-images N]
      [--skip-pg-diff]（省略 diffusion 基準，僅限快速驗證）
"""

import argparse
import re
import statistics
import time

import piq

from common import REPO_ROOT, apply_smoke, load_configs  # noqa: E402（先設定 sys.path）

from src.data.dataset import load_dataset
from src.metrics.quality import lpips_distance
from src.models.sd_wrapper import SDWrapper
from src.protect import build_protection
from src.utils.device import get_device
from src.utils.io import (
    make_run_dir, save_config_snapshot, save_csv, save_env_json, save_summary,
)
from src.utils.seed import set_seed

# 掃描鈕之 config 位置：方法鍵 → (區塊, 參數名)。hybrid 之 eps_a 寫入 hybrid
# 區塊（build_protection 合併時覆蓋 apa 之值）
KNOB_SECTION = {"advdiff": "advdiff", "apa_sg": "apa", "apa_gc": "apa", "hybrid": "hybrid"}


def protect_metrics(method, data) -> list:
    """逐影像 {image_id, lpips, psnr, linf}（protected vs original）。
    lpips 驅動校準；psnr/linf 供公平性檢核（L∞ 對加性方法為約束上界，
    對非加性方法僅供參考）。"""
    dev = get_device()
    out = []
    for sample in data:
        orig = sample["image"]
        protected = method.protect(orig, sample["concept"])
        x = protected.detach().float().clamp(0, 1).to(dev)
        y = orig.detach().float().clamp(0, 1).to(dev)
        out.append({
            "image_id": sample["image_id"],
            "lpips": lpips_distance(protected, orig),
            "psnr": piq.psnr(x, y, data_range=1.0).item(),
            "linf": (x - y).abs().max().item(),
        })
    return out


def mean_lpips(metrics) -> float:
    return sum(m["lpips"] for m in metrics) / len(metrics)


def write_similarity_budget(l_ref: float) -> None:
    path = REPO_ROOT / "configs" / "nonadditive.yaml"
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"(similarity_budget:\s*)[^\s#]+", rf"\g<1>{l_ref:.4f}", text, count=1
    )
    if n != 1:
        raise RuntimeError("configs/nonadditive.yaml 中找不到 similarity_budget 行")
    path.write_text(new_text, encoding="utf-8")


def set_knob(cfgs: dict, method_key: str, knob: str, value) -> None:
    section = KNOB_SECTION[method_key]
    cfgs["nonadditive"].setdefault(section, {})[knob] = value


def main():
    parser = argparse.ArgumentParser(description="stage0 相似性約束校準")
    parser.add_argument("--model", default=None, help="覆蓋 protect_model（本地測試用）")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="本地 tiny 模型流程驗證（縮減參數）")
    parser.add_argument("--skip-pg-diff", action="store_true", help="省略 diffusion 基準")
    parser.add_argument("--pg-diff-images", type=int, default=5,
                        help="diffusion 基準之影像數（假設 1 裁定：5 張；成本約為 encoder 50 倍）")
    parser.add_argument("--no-write-config", action="store_true", help="不回寫 configs/")
    args = parser.parse_args()

    cfgs = load_configs()
    model_name = args.model or cfgs["base"]["model"]["protect_model"]
    sd = SDWrapper(model_name)
    if args.smoke:
        apply_smoke(cfgs, sd)
    set_seed(cfgs["base"]["runtime"]["seed"])

    scan_cfg = cfgs["nonadditive"]["stage0_scan"]
    scan = {m: (c["knob"], c["values"][:2] if args.smoke else c["values"])
            for m, c in scan_cfg.items()}
    data = load_dataset(cfgs["base"], max_images=args.max_images or (2 if args.smoke else None))
    run_dir = make_run_dir("stage0")
    print(f"stage0: model={model_name}, images={len(data)}, run_dir={run_dir}")

    # 1. 基準：encoder 與 diffusion 兩者皆算（假設 1 裁定：encoder 為採用值，
    #    diffusion 併列供 10% 差距檢核）
    rows, warnings = [], []
    fairness_metrics = {}  # 顯示方法名 → 逐影像 metric list（採用/校準後設定）
    scan_metrics = {}      # (method_key, value) → 逐影像 metric list
    t0 = time.time()
    enc_metrics = protect_metrics(
        build_protection("pg_enc", sd, cfgs["additive"], cfgs["nonadditive"]), data)
    l_enc = mean_lpips(enc_metrics)
    fairness_metrics["pg_enc"] = enc_metrics
    rows.append({"method": "pg_enc", "knob": "epsilon",
                 "value": cfgs["additive"]["photoguard"]["epsilon"], "lpips": l_enc,
                 "adopted": True, "at_boundary": ""})
    print(f"L_ref(encoder) = {l_enc:.4f}  [{time.time() - t0:.0f}s]")
    l_diff = None
    if not args.skip_pg_diff:
        t0 = time.time()
        diff_metrics = protect_metrics(
            build_protection("pg_diff", sd, cfgs["additive"], cfgs["nonadditive"]),
            data[: args.pg_diff_images])
        l_diff = mean_lpips(diff_metrics)
        fairness_metrics["pg_diff"] = diff_metrics
        rows.append({"method": "pg_diff", "knob": "epsilon",
                     "value": cfgs["additive"]["photoguard"]["epsilon"], "lpips": l_diff,
                     "adopted": False, "at_boundary": ""})
        gap = abs(l_diff - l_enc) / max(l_enc, 1e-8)
        print(f"L(diffusion)   = {l_diff:.4f}（與 encoder 差 {gap:.1%}）")
        if gap > 0.10:
            warnings.append(
                f"encoder 與 diffusion 之 LPIPS 差距 {gap:.1%} > 10%："
                "L_ref 採 encoder 之決定須重新檢視（假設 1 之驗證條件）")
    l_ref = l_enc

    # 2. 非加性掃描（範圍取自 config stage0_scan；SG/GC 獨立）
    chosen = {}
    for method_key, (knob, values) in scan.items():
        curve = []
        for v in values:
            set_knob(cfgs, method_key, knob, v)
            method = build_protection(method_key, sd, cfgs["additive"], cfgs["nonadditive"])
            t0 = time.time()
            mets = protect_metrics(method, data)
            lp = mean_lpips(mets)
            scan_metrics[(method_key, v)] = mets
            curve.append((v, lp))
            rows.append({"method": method_key, "knob": knob, "value": v, "lpips": lp,
                         "adopted": False, "at_boundary": ""})
            print(f"{method_key} {knob}={v}: LPIPS={lp:.4f}  [{time.time() - t0:.0f}s]")
        best_v, best_lp = min(curve, key=lambda p: abs(p[1] - l_ref))
        at_boundary = best_v in (min(values), max(values))
        chosen[method_key] = {"knob": knob, "value": best_v, "lpips": best_lp,
                              "at_boundary": at_boundary}
        for r in rows:
            if r["method"] == method_key and r["value"] == best_v:
                r["adopted"] = True
                r["at_boundary"] = at_boundary
        if at_boundary:
            warnings.append(
                f"{method_key} 選中值 {knob}={best_v} 位於掃描區間端點"
                f"（範圍 {values}）：首輪範圍為推測值，須擴展範圍重掃")

    # 公平性：採用/校準後設定之逐影像 LPIPS/PSNR/L∞（非加性取各自選定 knob）
    variant = cfgs["nonadditive"]["apa"].get("variant", "gc")
    for mkey, disp in {"advdiff": "advdiff", f"apa_{variant}": "apa", "hybrid": "hybrid"}.items():
        c = chosen.get(mkey)
        if c is not None and (mkey, c["value"]) in scan_metrics:
            fairness_metrics[disp] = scan_metrics[(mkey, c["value"])]
    CLS = {"pg_enc": "加性", "pg_diff": "加性",
           "advdiff": "非加性", "apa": "非加性", "hybrid": "非加性"}
    fair_rows = [{"method": disp, "type": CLS.get(disp, ""), "image_id": m["image_id"],
                  "lpips": m["lpips"], "psnr": m["psnr"], "linf": m["linf"]}
                 for disp, mets in fairness_metrics.items() for m in mets]
    save_csv(run_dir / "fairness.csv", fair_rows)

    # 3. 輸出與回寫
    save_csv(run_dir / "calibration.csv", rows)
    _plot(rows, l_ref, run_dir)
    save_config_snapshot(run_dir, cfgs)
    save_env_json(run_dir)

    if not args.no_write_config:
        write_similarity_budget(l_ref)
        variant = cfgs["nonadditive"]["apa"].get("variant", "gc")
        apa_chosen = chosen.get(f"apa_{variant}")
        overlay = ["# stage0_calibrate.py 產出：各非加性方法之相似性校準值（stage1 自動合併）",
                   f"# L_ref = {l_ref:.4f}（encoder）"
                   + (f"；diffusion = {l_diff:.4f}" if l_diff is not None else "")]
        if "advdiff" in chosen:
            overlay += ["advdiff:", f"  eps_latent: {chosen['advdiff']['value']}"]
        if apa_chosen:
            overlay += [f"apa:  # variant={variant}", f"  eps_a: {apa_chosen['value']}"]
        if "hybrid" in chosen:
            overlay += ["hybrid:", f"  eps_a: {chosen['hybrid']['value']}"]
        (REPO_ROOT / "configs" / "nonadditive_calibrated.yaml").write_text(
            "\n".join(overlay) + "\n", encoding="utf-8")

    lines = [
        "# stage0 校準摘要", "",
        f"- 模型：{model_name}（smoke={args.smoke}；smoke 數值僅驗流程）",
        f"- 影像數：{len(data)}"
        + ("（placeholder 資料，與真實資料集結果不可直接比較）"
           if cfgs["base"]["data"].get("is_placeholder") else ""),
        f"- **L_ref = {l_ref:.4f}**（實測值：PhotoGuard encoder，採用值）",
        (f"- L(diffusion) = {l_diff:.4f}（實測值，併列供假設 1 之 10% 檢核）"
         if l_diff is not None else "- L(diffusion)：本次省略（--skip-pg-diff）"), "",
        "| 方法 | 參數 | 選定值 | LPIPS | 端點警告 |", "|---|---|---|---|---|",
    ]
    for m, c in chosen.items():
        lines.append(f"| {m} | {c['knob']} | {c['value']} | {c['lpips']:.4f} |"
                     f" {'是——須擴展重掃' if c['at_boundary'] else '否'} |")

    # 公平性表（表 1）：非加性 LPIPS(prot,orig) 不應顯著高於加性，否則耐淨化
    # 優勢可能來自「改動更多」而非機制。此表是整份 v2 結果的前提。
    def _ms(vals):
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        return m, s
    lines += ["", "## 公平性檢核（表 1；非加性 LPIPS 不應顯著高於加性）", "",
              "| 方法 | 類型 | LPIPS(prot,orig)↓ | PSNR(prot,orig)↑ | L∞ 差異 | n |",
              "|---|---|---|---|---|---|"]
    for disp in ("pg_enc", "pg_diff", "advdiff", "apa", "hybrid"):
        mets = fairness_metrics.get(disp)
        if not mets:
            continue
        lm, ls = _ms([m["lpips"] for m in mets])
        pm, ps = _ms([m["psnr"] for m in mets])
        xm, xs = _ms([m["linf"] for m in mets])
        linf_s = f"{xm:.4f}" + ("（參考）" if disp in ("advdiff", "apa", "hybrid") else "")
        lines.append(f"| {disp} | {CLS.get(disp, '')} | {lm:.4f} ± {ls:.4f} |"
                     f" {pm:.2f} ± {ps:.2f} | {linf_s} | {len(mets)} |")

    if warnings:
        lines += ["", "## 警告", ""] + [f"- {w}" for w in warnings]
    save_summary(run_dir, "\n".join(lines) + "\n")
    for w in warnings:
        print(f"[警告] {w}")
    print(f"完成。chosen={chosen}")


def _plot(rows, l_ref, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for method_key in sorted({r["method"] for r in rows} - {"pg_enc", "pg_diff"}):
        pts = sorted((r["value"], r["lpips"]) for r in rows if r["method"] == method_key)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method_key)
    ax.axhline(l_ref, linestyle="--", color="gray", label=f"L_ref={l_ref:.3f}")
    ax.set_xlabel("similarity knob value")
    ax.set_ylabel("LPIPS(protected, original)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "calibration_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
