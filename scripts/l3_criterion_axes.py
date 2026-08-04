"""L3：把 L1 的那一批 x_adv 放到另外兩類判準上重判。

    python scripts/l3_criterion_axes.py

零 GPU，只讀 `runs/lo_baseline/results.csv`。

## 為什麼這一支獨立於 p16

`scripts/p16_criterion_correlation.py` 掃不到 `runs/lo_baseline/`，有三個
硬性原因：它要求 `purify` 欄（該協定不含淨化，沒有這一欄）、要求
`net_lpips`（同理，沒有淨化對照可扣）、要求 `env.json`（該目錄只有兩個 CSV）。
於是 p16 報出的 ρ = −0.207 至今只涵蓋 **w = 1、n = 217**，而 w = 1 那一批
依 LEDGER 3.2 是在防禦一個不存在的攻擊。

**三類判準的相關性在有效威脅模型下從未量過。** 這正是 LEDGER 9.10。

## 三類判準在本批資料上的欄位

| 類 | 量什麼 | 欄位 | 防禦成功的方向 |
|---|---|---|---|
| 1. 距離 | 免疫後的編輯輸出離原始編輯結果多遠 | `edit_lpips` | 越大越好 |
| 2. 語意 | 輸出還服不服從 prompt | `edit_siglip_b − edit_siglip_a` | 越負越好 |
| 3. 劣化 | 輸出本身還能不能看 | `edit_niqe_b − edit_niqe_a` | 越正越好（NIQE 越高越差） |

`_a` 是 `suite.full(a, b)` 的第一個引數即未防禦的編輯 `y_ref`，`_b` 是免疫後
的編輯 `y_def`（`src/metrics/suite.py:211`）。第 1 類不必扣對照：本協定沒有
淨化，`y_ref` 本身就是對照。

CLIP 一併算出但**不用於判定**：E25 §1.1 量到它分不出「編輯有沒有發生」
（edit_effect +0.0101 ± 0.0169，標準差大於均值），SigLIP 通過同一個對照
（+0.0276 ± 0.0237）。留著是為了讓「換一個對齊模型會不會翻盤」有數字可查。

## ISR 的判定

ISR = 語意不符 **或** 明顯的感知劣化（arXiv:2512.14320）。本腳本逐 (影像,
攻擊) 判定，不逐種子：n < 2 時標準差恆為 0，任何「|mean| > sd」的判定自動
成立，E25 曾因此產生 24 格假陽性（LEDGER 1.4）。每格有 20 個種子，故以
20 個種子的平均與標準差判定。

劣化那一半的門檻不由本腳本發明。`runs/p11_degrade_ladder/` 是為了定它而做
的階梯，但使用者尚未判讀（LEDGER 9.4）。故本腳本**只報分佈與逐格的值**，
把門檻當成參數掃過去，讓「門檻定在哪會改變結論」這件事本身看得見。
"""

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.p16_criterion_correlation import spearman  # noqa: E402

# 劣化門檻的掃描點，單位是 NIQE 的絕對差。不取單一值的理由見模組 docstring。
DEGRADE_TAUS = (0.0, 0.5, 1.0, 2.0, 4.0)


def load(run_dir: Path):
    """回傳逐列的 (影像, 攻擊, 距離, Δsiglip, Δniqe, Δclip)。"""
    f = run_dir / "results.csv"
    if not f.exists():
        raise SystemExit(f"{f} 不存在")
    need = ("edit_lpips", "edit_siglip_a", "edit_siglip_b",
            "edit_niqe_a", "edit_niqe_b", "edit_clip_a", "edit_clip_b")
    rows, skipped = [], 0
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if any(not r.get(k) for k in need):
                skipped += 1
                continue
            rows.append({
                "image": r["image"],
                "attack": r["attack"],
                "dist": float(r["edit_lpips"]),
                "dsiglip": float(r["edit_siglip_b"]) - float(r["edit_siglip_a"]),
                "dniqe": float(r["edit_niqe_b"]) - float(r["edit_niqe_a"]),
                "dclip": float(r["edit_clip_b"]) - float(r["edit_clip_a"]),
            })
    if skipped:
        # 不靜默略過：缺欄與「該欄本來就沒算」在外部分不出來
        print(f"[l3] 略過 {skipped} 列（三類欄位不齊）", flush=True)
    if not rows:
        raise SystemExit(f"{f} 沒有任何三類欄位齊全的列")
    return rows


def cells(rows):
    """把逐種子的列聚成逐 (影像, 攻擊) 的格，回傳每格的平均與標準差。

    判定必須在格的層級做，不是逐種子：n < 2 時 `pstdev` 恆為 0，
    任何「|mean| > sd」的判定自動成立（LEDGER 1.4）。
    """
    g = defaultdict(list)
    for r in rows:
        g[(r["image"], r["attack"])].append(r)
    out = []
    for (img, atk), rs in sorted(g.items()):
        c = {"image": img, "attack": atk, "n_seeds": len(rs)}
        for k in ("dist", "dsiglip", "dniqe", "dclip"):
            vals = [r[k] for r in rs]
            c[f"{k}_mean"] = st.mean(vals)
            c[f"{k}_sd"] = st.pstdev(vals)
        out.append(c)
    return out


def judge(c, degrade_tau):
    """ISR 的兩半，回傳 (語意失敗, 劣化失敗)。

    語意失敗：Δsiglip 的平均為負且其絕對值大於標準差。這是 E25 定下的規則
    ——只看符號會把噪聲當訊號。
    劣化失敗：Δniqe 的平均大於門檻且大於標準差。
    """
    sem = c["dsiglip_mean"] < 0 and abs(c["dsiglip_mean"]) > c["dsiglip_sd"]
    deg = c["dniqe_mean"] > degrade_tau and c["dniqe_mean"] > c["dniqe_sd"]
    return sem, deg


def working_images(path: Path) -> set:
    """未防禦編輯**明顯成功**的影像集合，由 `l3_edit_success.py` 產生。

    在未防禦編輯本來就失敗的影像上量免疫效果沒有意義——防的是一個不會發生
    的事。Attention Attack（ACM MM 2025）建資料集時明說要先做這道過濾。
    本專案 24 張裡有 6 張不合格，其中 `dog_03` 的編輯效果是負的（LEDGER 1.22）。
    """
    if not path.exists():
        raise SystemExit(
            f"{path} 不存在。請先跑 scripts/l3_edit_success.py——"
            "「哪些影像的未防禦編輯真的成功」是判讀的前提，不可略過")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["image"] for r in csv.DictReader(f)
                if r["edit_worked_clearly"] == "True"}


def main():
    ap = argparse.ArgumentParser(description="L3：三類判準在有效威脅模型下的重判")
    ap.add_argument("--run", default="runs/lo_baseline")
    ap.add_argument("--out", default="runs/l3_criterion_axes")
    ap.add_argument("--only_working", action="store_true",
                    help="只留未防禦編輯明顯成功的影像（LEDGER 1.22）。"
                         "需要先跑 scripts/l3_edit_success.py")
    ap.add_argument("--edit_success",
                    default="runs/l3_criterion_axes/edit_success.csv",
                    help="--only_working 讀的過濾清單。刻意與 --out 分開："
                         "換一個輸出目錄時不該連過濾清單一起換掉")
    args = ap.parse_args()

    run_dir = ROOT / args.run
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = load(run_dir)
    if args.only_working:
        keep = working_images(ROOT / args.edit_success)
        before = len({r["image"] for r in rows})
        rows = [r for r in rows if r["image"] in keep]
        print(f"[l3] 只留未防禦編輯明顯成功的影像：{before} → {len(keep)} 張"
              f"（LEDGER 1.22）", flush=True)
    cs = cells(rows)
    n_img = len({c["image"] for c in cs})
    atks = sorted({c["attack"] for c in cs})
    print(f"[l3] {args.run}：{len(rows)} 列 → {len(cs)} 格"
          f"（{n_img} 張 × {len(atks)} 攻擊），每格 "
          f"{cs[0]['n_seeds']} 個種子\n", flush=True)

    # ---- 1. 三類相關性 ----
    #
    # 逐種子算而非逐格：相關性問的是「排序一不一致」，而每一個 (影像, 種子,
    # 攻擊) 都是一次獨立的量測。逐格會把 20 個種子的變異平均掉，n 由 1440
    # 降到 72，而那個變異正是 SDEdit 的主要變異來源。兩者都報。
    corr_rows = []
    for scope, data in (("逐種子", rows),
                        ("逐格（20 種子平均）",
                         [{"dist": c["dist_mean"], "dsiglip": c["dsiglip_mean"],
                           "dniqe": c["dniqe_mean"], "attack": c["attack"]}
                          for c in cs])):
        for atk in ["（全部）"] + atks:
            d = data if atk == "（全部）" else [r for r in data if r["attack"] == atk]
            if len(d) < 3:
                continue
            corr_rows.append({
                "scope": scope, "attack": atk, "n": len(d),
                "rho_dist_sem": round(spearman([r["dist"] for r in d],
                                               [r["dsiglip"] for r in d]), 4),
                "rho_dist_deg": round(spearman([r["dist"] for r in d],
                                               [r["dniqe"] for r in d]), 4),
                "rho_sem_deg": round(spearman([r["dsiglip"] for r in d],
                                              [r["dniqe"] for r in d]), 4),
            })

    print(f"{'範圍':<20} {'攻擊':<14} {'n':>5} "
          f"{'ρ(距離,語意)':>13} {'ρ(距離,劣化)':>13} {'ρ(語意,劣化)':>13}")
    for r in corr_rows:
        print(f"{r['scope']:<20} {r['attack']:<14} {r['n']:>5} "
              f"{r['rho_dist_sem']:>13.4f} {r['rho_dist_deg']:>13.4f} "
              f"{r['rho_sem_deg']:>13.4f}")

    # ---- 2. 逐攻擊的三類平均 ----
    print(f"\n{'攻擊':<14} {'距離(edit_lpips)':>17} {'Δsiglip':>12} "
          f"{'Δclip':>10} {'Δniqe':>10}")
    axis_rows = []
    for atk in atks:
        d = [c for c in cs if c["attack"] == atk]
        a = {"attack": atk, "n_cells": len(d)}
        for k in ("dist", "dsiglip", "dclip", "dniqe"):
            a[f"{k}_mean"] = round(st.mean(c[f"{k}_mean"] for c in d), 5)
            a[f"{k}_sd"] = round(st.pstdev(c[f"{k}_mean"] for c in d), 5)
        axis_rows.append(a)
        print(f"{atk:<14} {a['dist_mean']:>17.4f} {a['dsiglip_mean']:>+12.5f} "
              f"{a['dclip_mean']:>+10.5f} {a['dniqe_mean']:>+10.4f}")

    # ---- 3. ISR 逐門檻 ----
    print(f"\n{'τ_degrade':>10} {'語意失敗':>10} {'劣化失敗':>10} "
          f"{'ISR 聯集':>10}  （共 {len(cs)} 格）")
    isr_rows = []
    for tau in DEGRADE_TAUS:
        js = [judge(c, tau) for c in cs]
        sem = sum(1 for s, _ in js if s)
        deg = sum(1 for _, d in js if d)
        uni = sum(1 for s, d in js if s or d)
        isr_rows.append({"degrade_tau": tau, "n_cells": len(cs),
                         "semantic_fail": sem, "degrade_fail": deg,
                         "isr_union": uni,
                         "isr_rate": round(uni / len(cs), 4)})
        print(f"{tau:>10.1f} {sem:>10} {deg:>10} {uni:>10}")

    # ---- 寫檔 ----
    def dump(name, data):
        p = out / name
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        return p

    dump("correlation.csv", corr_rows)
    dump("axes_by_attack.csv", axis_rows)
    dump("isr_by_threshold.csv", isr_rows)
    dump("cells.csv", [{k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in c.items()} for c in cs])
    (out / "meta.json").write_text(json.dumps({
        "source_run": args.run, "n_rows": len(rows), "n_cells": len(cs),
        "n_images": n_img, "attacks": atks,
        "seeds_per_cell": cs[0]["n_seeds"],
        "degrade_taus": list(DEGRADE_TAUS),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[l3] 寫出 {out}")


if __name__ == "__main__":
    main()
