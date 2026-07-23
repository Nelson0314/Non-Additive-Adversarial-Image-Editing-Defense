"""stage0：相似性約束校準 — SPEC.md §6.1、STRUCTURE.md §4.2。

1. 對 PhotoGuard encoder（κ=0.06）計算 LPIPS(protected, original) 平均，得基準 L_ref
2. 對非加性方法掃描各自的相似性控制參數（advdiff: eps_latent；apa/hybrid: eps_a），
   記錄各設定之 LPIPS 平均
3. 選出使 LPIPS 最接近 L_ref 之設定：
   - L_ref 寫入 configs/nonadditive.yaml 之 common.similarity_budget（僅改該行，保留註解）
   - 各方法選定值寫入 configs/nonadditive_calibrated.yaml（stage1 自動合併），
     避免以 yaml 重寫整份 nonadditive.yaml 而破壞註解
4. 輸出 calibration.csv 與 calibration_curve.png 至 experiments/stage0/<timestamp>/

用法：python scripts/stage0_calibrate.py [--smoke] [--model M] [--max-images N]
"""

import argparse
import re
import time

from common import REPO_ROOT, apply_smoke, load_configs  # noqa: E402（先設定 sys.path）

from src.data.dataset import load_dataset
from src.metrics.quality import lpips_distance
from src.models.sd_wrapper import SDWrapper
from src.protect import build_protection
from src.utils.io import (
    make_run_dir, save_config_snapshot, save_csv, save_env_json, save_summary,
)
from src.utils.seed import set_seed

# 掃描之相似性控制參數（由嚴至寬）。正式值域為初始猜測，依 TWCC 實測調整。
SCAN = {
    "advdiff": ("eps_latent", [0.05, 0.1, 0.2, 0.4]),
    "apa": ("eps_a", [0.1, 0.2, 0.4, 0.8]),
    "hybrid": ("eps_a", [0.1, 0.2, 0.4, 0.8]),
}


def mean_protect_lpips(method, data) -> float:
    vals = []
    for sample in data:
        protected = method.protect(sample["image"], sample["concept"])
        vals.append(lpips_distance(protected, sample["image"]))
    return sum(vals) / len(vals)


def write_similarity_budget(l_ref: float) -> None:
    path = REPO_ROOT / "configs" / "nonadditive.yaml"
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"(similarity_budget:\s*)[^\s#]+", rf"\g<1>{l_ref:.4f}", text, count=1
    )
    if n != 1:
        raise RuntimeError("configs/nonadditive.yaml 中找不到 similarity_budget 行")
    path.write_text(new_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="stage0 相似性約束校準")
    parser.add_argument("--model", default=None, help="覆蓋 protect_model（本地測試用）")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="本地 tiny 模型流程驗證（縮減參數）")
    parser.add_argument("--no-write-config", action="store_true", help="不回寫 configs/")
    args = parser.parse_args()

    cfgs = load_configs()
    model_name = args.model or cfgs["base"]["model"]["protect_model"]
    sd = SDWrapper(model_name)
    if args.smoke:
        apply_smoke(cfgs, sd)
    set_seed(cfgs["base"]["runtime"]["seed"])

    scan = {m: (k, vals[:2]) for m, (k, vals) in SCAN.items()} if args.smoke else SCAN
    data = load_dataset(cfgs["base"], max_images=args.max_images or (2 if args.smoke else None))
    run_dir = make_run_dir("stage0")
    print(f"stage0: model={model_name}, images={len(data)}, run_dir={run_dir}")

    # 1. 基準 L_ref（PhotoGuard encoder，κ 依 additive.yaml）
    t0 = time.time()
    pg = build_protection("pg_enc", sd, cfgs["additive"], cfgs["nonadditive"])
    l_ref = mean_protect_lpips(pg, data)
    print(f"L_ref (PhotoGuard encoder) = {l_ref:.4f}  [{time.time() - t0:.0f}s]")

    # 2. 非加性掃描
    rows = [{"method": "pg_enc", "knob": "epsilon",
             "value": cfgs["additive"]["photoguard"]["epsilon"], "lpips": l_ref}]
    chosen = {}
    for method_key, (knob, values) in scan.items():
        curve = []
        for v in values:
            section = "advdiff" if method_key == "advdiff" else "apa"
            cfgs["nonadditive"][section][knob] = v
            method = build_protection(method_key, sd, cfgs["additive"], cfgs["nonadditive"])
            t0 = time.time()
            lp = mean_protect_lpips(method, data)
            curve.append((v, lp))
            rows.append({"method": method_key, "knob": knob, "value": v, "lpips": lp})
            print(f"{method_key} {knob}={v}: LPIPS={lp:.4f}  [{time.time() - t0:.0f}s]")
        best_v, best_lp = min(curve, key=lambda p: abs(p[1] - l_ref))
        chosen[method_key] = {"knob": knob, "value": best_v, "lpips": best_lp}

    # 3. 輸出與回寫
    save_csv(run_dir / "calibration.csv", rows)
    _plot(rows, l_ref, run_dir)
    save_config_snapshot(run_dir, cfgs)
    save_env_json(run_dir)

    if not args.no_write_config:
        write_similarity_budget(l_ref)
        overlay_lines = ["# stage0_calibrate.py 產出：各非加性方法之相似性校準值（stage1 自動合併）"]
        if "advdiff" in chosen:
            overlay_lines += ["advdiff:", f"  eps_latent: {chosen['advdiff']['value']}"]
        if "apa" in chosen:
            overlay_lines += ["apa:", f"  eps_a: {chosen['apa']['value']}"]
        (REPO_ROOT / "configs" / "nonadditive_calibrated.yaml").write_text(
            "\n".join(overlay_lines) + "\n", encoding="utf-8"
        )

    lines = [
        "# stage0 校準摘要", "",
        f"- 模型：{model_name}（smoke={args.smoke}）",
        f"- 影像數：{len(data)}",
        f"- **L_ref = {l_ref:.4f}**（實測值：PhotoGuard encoder，"
        f"κ={cfgs['additive']['photoguard']['epsilon']}）", "",
        "| 方法 | 參數 | 選定值 | LPIPS |", "|---|---|---|---|",
    ]
    for m, c in chosen.items():
        lines.append(f"| {m} | {c['knob']} | {c['value']} | {c['lpips']:.4f} |")
    lines += ["", "全部為實測值；hybrid 與 apa 共用 eps_a，選定值以 apa 之結果寫入 overlay。"]
    save_summary(run_dir, "\n".join(lines) + "\n")
    print(f"完成。chosen={chosen}")


def _plot(rows, l_ref, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for method_key in {r["method"] for r in rows} - {"pg_enc"}:
        pts = [(r["value"], r["lpips"]) for r in rows if r["method"] == method_key]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method_key)
    ax.axhline(l_ref, linestyle="--", color="gray", label=f"L_ref={l_ref:.3f}")
    ax.set_xlabel("similarity knob value")
    ax.set_ylabel("LPIPS(protected, original)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "calibration_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
