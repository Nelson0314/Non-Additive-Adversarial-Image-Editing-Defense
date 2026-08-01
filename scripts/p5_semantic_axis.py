"""E25-1 —— 語意軸重判：防禦到底有沒有讓編輯不服從 prompt？

為什麼要做這件事。`runs/*/results.csv` 從 E2 起就記了四個語意欄位
（`edit_clip_a` / `edit_clip_b` / `edit_siglip_a` / `edit_siglip_b`），分別是
未防禦編輯結果與防禦後編輯結果對編輯 prompt 的對齊度。全專案沒有
任何腳本讀過它們：`grep -rn clip src/ scripts/` 只會命中 `grad_clip` 與
`numpy.clip`。至今所有結論都建立在 `net_lpips = edit_lpips − ctrl_lpips`
之上，而那個量說的是「輸出移動了多少」，不是「編輯有沒有失敗」。

這個區別不是本專案自己想出來的顧慮。*Semantic Mismatch and Perceptual
Degradation: A New Perspective on Image Editing Immunity*（arXiv:2512.14320）
直接指出：以「與未防禦編輯結果的視覺距離」為免疫成功的判準是錯的，並展示了
LPIPS 距離很高但免疫完全失敗的案例。免疫成功應定義為語意不匹配（輸出
不再符合 prompt 的意圖）或感知劣化（輸出明顯是壞掉的）。本腳本量的是
前者，後者已由既有的失真指標涵蓋。

---

判定量。逐張影像取配對差

    Δ = CLIP(防禦後編輯, prompt) − CLIP(未防禦編輯, prompt)

Δ < 0 才是防禦在起作用。以配對差而非兩組獨立比較，是因為兩者共用同一張
原圖、同一個 prompt、同一顆噪聲種子，配對後才不會被影像之間的難度差異淹沒。

必要的對照，以及它為什麼必要。若 CLIP 本身分不出「編輯有沒有發生」，
則 Δ ≈ 0 只代表 CLIP 遲鈍，不代表編輯沒被擋下來，整條語意軸就不成立。故先量

    edit_effect = CLIP(未防禦編輯, prompt) − CLIP(原圖, prompt)

即「編輯本身讓對齊度上升了多少」。這同時提供了 Δ 的尺度：

    undo_frac = −Δ / edit_effect

即防禦抵銷了編輯效果的幾分之幾。undo_frac = 1 表示編輯被完全擋下，
= 0 表示完全沒擋下。沒有這個分母，Δ = −0.003 這種數字無法解讀。

對照不通過就作廢，不改判準去救它。`edit_effect` 若不顯著為正，本腳本
直接印出作廢並以非零狀態結束。

判準（n=6，與 E20 §4 同樣採保守形式）：

- 「語意失敗」= mean(Δ) < 0 且 |mean(Δ)| > sd(Δ)，即效應大於配對差本身的
  離散度。不用 t 檢定：6 張影像套用需要更多假設的統計量並不誠實。
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "p5_semantic_axis"

# 對照的通過門檻：編輯本身必須讓 CLIP 對齊度上升至少這麼多，語意軸才有尺度可言。
# 0.005 的來源是 CLIP 餘弦相似度在本專案影像上的量級（實測 0.21 上下），
# 取其約 2.5%；低於此值時 Δ 的分母不可靠。
MIN_EDIT_EFFECT = 0.005

SEM_KEYS = ("clip", "siglip")


def load_image(path: Path, device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0).to(device)


def find_runs(root: Path):
    """回傳 (run 名稱, results.csv 路徑)，只取語意欄位有實際數值的 run。"""
    out = []
    for f in sorted(root.glob("*/results.csv")):
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        if not rows or "edit_clip_a" not in rows[0]:
            continue
        if rows[0]["edit_clip_a"] in ("", "nan"):
            continue
        out.append((f.parent.name, f))
    return out


def control(runs, device, limit_cells=None):
    """對照：編輯本身讓 CLIP/SigLIP 對齊度上升多少。

    逐 cell 讀 `orig.png` 與 `edit_orig.png`，以 `MetricSuite.semantic` 計算，
    與訓練／評測共用同一份實作（兩者不得分歧）。同一個 (影像檔, prompt) 只
    算一次。
    """
    from src.metrics.suite import MetricSuite

    suite = MetricSuite(device=device)
    cache, rows, no_cells = {}, [], []
    for name, csv_path in runs:
        seen = set()
        for r in csv.DictReader(csv_path.open(encoding="utf-8")):
            sub = _cell_dir(csv_path.parent, r)
            if sub is None:
                no_cells.append(name)
                break
            cell = csv_path.parent / sub
            key = (str(cell), r["prompt"])
            if key in seen:
                continue
            seen.add(key)
            orig, edit = cell / "orig.png", cell / "edit_orig.png"
            if not (orig.exists() and edit.exists()):
                continue
            if key not in cache:
                so = suite.semantic(load_image(orig, device), r["prompt"])
                se = suite.semantic(load_image(edit, device), r["prompt"])
                cache[key] = {k: (so[k], se[k]) for k in SEM_KEYS}
            c = cache[key]
            rows.append({
                "run": name, "cell": cell.name, "prompt": r["prompt"],
                **{f"{k}_orig": c[k][0] for k in SEM_KEYS},
                **{f"{k}_edit": c[k][1] for k in SEM_KEYS},
                **{f"{k}_edit_effect": c[k][1] - c[k][0] for k in SEM_KEYS},
            })
            if limit_cells and len(cache) >= limit_cells:
                return rows, sorted(set(no_cells))
    return rows, sorted(set(no_cells))


def _cell_dir(run_dir: Path, row: dict):
    """由 results.csv 的一列還原出 cell 目錄名；該 run 未留存逐圖目錄時回傳 None。

    命名為 `{image}__{site}__r{rank}`（見 scripts/run_defense.py）。此處以
    實際存在的目錄比對而非只靠字串拼接，因為 rank 欄位在部分 site 為 0。

    回傳 None 與拋出是兩件不同的事，必須分開。有些早期 run（如 `e2_la`）
    整個目錄下一個 cell 都沒有——逐圖影像從未留存。那是資料不存在，對照本來
    就算不出來，記錄下來即可。反之，若目錄存在卻對不上命名規則，那是命名規則
    變動，靜默跳過會讓對照少掉一整批 run 而看不出來，故仍然拋出。
    """
    want = f"{row['image']}__{row['site']}__r{row['rank']}"
    if (run_dir / want).is_dir():
        return want
    cands = [d.name for d in run_dir.iterdir()
             if d.is_dir() and d.name.startswith(f"{row['image']}__{row['site']}__")]
    if not cands and not any(d.is_dir() for d in run_dir.iterdir()):
        return None
    if len(cands) != 1:
        raise FileNotFoundError(
            f"{run_dir.name} 中無法由 {want!r} 唯一決定 cell 目錄，候選 {cands}。"
            "命名規則變動時必須明確處理，不可猜"
        )
    return cands[0]


def delta_table(runs):
    """逐 run × 淨化臂的配對差 Δ。回傳明細列。"""
    out = []
    for name, csv_path in runs:
        by = defaultdict(list)
        for r in csv.DictReader(csv_path.open(encoding="utf-8")):
            if r["noise_split"] != "heldout":
                continue
            by[(r["purify"], float(r["strength"]))].append(r)
        for (kind, strength), rs in sorted(by.items()):
            row = {"run": name, "site": rs[0]["site"], "purify": kind,
                   "strength": strength, "n": len(rs),
                   "net_lpips": float(np.mean([float(r["net_lpips"]) for r in rs]))}
            for k in SEM_KEYS:
                d = np.array([float(r[f"edit_{k}_b"]) - float(r[f"edit_{k}_a"])
                              for r in rs])
                row[f"d{k}"] = float(d.mean())
                row[f"d{k}_sd"] = float(d.std(ddof=0))
                # n=1 時 sd 恆為 0，`|mean| > sd` 對任何負值都自動成立。那是
                # 判準的假象，不是效應，故單張影像的格子一律判為不可判定。
                # 本專案的 e6_stepsP / e6_stepsLA 正是 n=1（步數掃描只跑一張圖）。
                row[f"d{k}_fail"] = bool(
                    len(rs) >= 2 and d.mean() < 0 and abs(d.mean()) > d.std(ddof=0)
                )
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_cells", type=int, default=None,
                    help="只算前 N 個 cell 的對照，供快速檢查用")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    runs = find_runs(ROOT / "runs")
    print(f"[p5] 有語意欄位的 run：{len(runs)} 個")

    # ---- 1. 對照 ----
    print("[p5] 計算對照：編輯本身讓對齊度上升多少（載入 CLIP 與 SigLIP，CPU）")
    ctrl, no_cells = control(runs, device, args.limit_cells)
    _write_csv(OUT / "control.csv", ctrl)
    if no_cells:
        print(f"[p5] 未留存逐圖影像、無法算對照的 run：{', '.join(no_cells)}"
              f"（其 Δ 仍計入主表，只是不參與對照的尺度估計）")

    print(f"\n=== 對照（n={len(ctrl)} 個 cell）===")
    verdicts = {}
    for k in SEM_KEYS:
        eff = np.array([r[f"{k}_edit_effect"] for r in ctrl])
        o = np.array([r[f"{k}_orig"] for r in ctrl])
        e = np.array([r[f"{k}_edit"] for r in ctrl])
        ok = eff.mean() > MIN_EDIT_EFFECT and eff.mean() > eff.std(ddof=0)
        verdicts[k] = {"orig": float(o.mean()), "edit": float(e.mean()),
                       "effect": float(eff.mean()), "effect_sd": float(eff.std(ddof=0)),
                       "pass": bool(ok)}
        print(f"  {k:7s} 原圖 {o.mean():.4f} → 未防禦編輯 {e.mean():.4f}   "
              f"edit_effect {eff.mean():+.4f} ± {eff.std(ddof=0):.4f}   "
              f"{'通過' if ok else '未通過'}")

    valid = [k for k in SEM_KEYS if verdicts[k]["pass"]]
    if not valid:
        raise SystemExit(
            "\n對照未通過：編輯本身都沒有讓對齊度顯著上升，表示這兩個 VLM 在本"
            "資料上分不出『編輯有沒有發生』。語意軸就此作廢，Δ ≈ 0 不可解讀為"
            "『編輯沒被擋下來』。此處直接中止，不改判準去救結論。"
        )
    for k in SEM_KEYS:
        if k not in valid:
            print(f"\n  ** {k} 未通過對照，不可用於本資料的語意判定。**")
            print(f"     它連『編輯有沒有發生』都分不出來（edit_effect "
                  f"{verdicts[k]['effect']:+.4f} ± {verdicts[k]['effect_sd']:.4f}），"
                  f"故其 Δ{k} ≈ 0 只代表該指標遲鈍。")
            print(f"     Δ{k} 仍寫入 CSV 供查核，但下方的判定不採用它。")

    # ---- 2. 主表 ----
    rows = delta_table(runs)
    for r in rows:
        for k in SEM_KEYS:
            eff = verdicts[k]["effect"]
            r[f"undo_frac_{k}"] = float(-r[f"d{k}"] / eff) if eff else float("nan")
    _write_csv(OUT / "summary.csv", rows)

    # 無淨化那一格（blur 強度 0，即 identity）是主結論所在
    base = [r for r in rows if r["purify"] == "blur" and r["strength"] == 0.0]
    head = "".join(f"{'Δ' + k:>10s}{'undo_frac':>11s}" for k in valid)
    print(f"\n=== 無淨化，判定依據 {'/'.join(valid)}（n={len(base)} 個 run）===")
    print(f"{'run':26s}{'site':>5s}{'net_lpips':>11s}{head}  語意失敗")
    for r in sorted(base, key=lambda r: r["run"]):
        body = "".join(f"{r['d' + k]:>+10.4f}{r['undo_frac_' + k]:>+11.3f}"
                       for k in valid)
        fail = any(r[f"d{k}_fail"] for k in valid)
        print(f"{r['run']:26s}{r['site']:>5s}{r['net_lpips']:>11.4f}{body}"
              f"  {'是' if fail else '否'}")

    n_fail = sum(any(r[f"d{k}_fail"] for k in valid) for r in rows)
    print(f"\n全部 {len(rows)} 個 (run × 淨化臂) 格子中，"
          f"依 {'/'.join(valid)} 判定為語意失敗的有 {n_fail} 格。")
    if n_fail:
        print("  這些格子是：")
        for r in rows:
            if any(r[f"d{k}_fail"] for k in valid):
                print(f"    {r['run']:26s} {r['purify']}@{r['strength']:g}  "
                      + "  ".join(f"Δ{k}={r['d' + k]:+.4f}" for k in valid))

    (OUT / "control_verdict.json").write_text(
        json.dumps(verdicts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n寫入 {OUT}")


def _write_csv(path: Path, rows) -> None:
    if not rows:
        raise ValueError(f"沒有任何資料可寫入 {path}，這代表上游篩選把一切都濾掉了")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    main()
