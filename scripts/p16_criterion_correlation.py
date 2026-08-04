"""同一批資料在三類判準下的相關性。

## 這回答什麼

`docs/SURVEY.md` §2 整理出這條線並存三類判準：

| 類 | 量什麼 | 本專案的欄位 |
|---|---|---|
| 1. 與未防禦編輯的距離 | 輸出移動了多少 | `net_lpips`（= `edit_lpips` − `ctrl_lpips`） |
| 2. 影像—文字對齊 | 輸出服不服從 prompt | `edit_siglip_b` − `edit_siglip_a` |
| 3. 感知劣化 | 輸出本身還能不能用 | `edit_niqe_b` − `edit_niqe_a` |

第 1 類仍是文獻的主流（2026 年的 TPAMI 論文用它），本專案已兩次獨立量到它
與編輯是否失敗不對應。但「不對應」至今是**質性的敘述**——726 格語意失敗 0 格
說的是第 2 類沒有正例，不是兩類之間沒有關係。

本腳本直接量三類之間的等級相關（Spearman ρ）。SURVEY §8 把
「判準的三類之間如何換算」列為沒有人回答過的問題，而本專案的
`runs/*/results.csv` 有現成的材料——三類的欄位自 E2 起就一起記在同一列。

## 讀法

- ρ 接近 0：兩類量的是無關的東西。第 1 類的高分不預測第 2、3 類的高分。
- ρ 顯著為負：第 1 類越高、第 2 類越高（即防禦圖被編輯後**更**服從 prompt），
  那與 ICIP 2025 的觀察方向一致。
- 必須依 `guidance_scale` 分組。w=1 的那批是在防禦一個不存在的攻擊（E26），
  混在一起算會讓相關性由該批主導——它們佔絕大多數。

執行：python scripts/p16_criterion_correlation.py
成本：只讀 CSV 與 JSON，秒級。
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def spearman(xs, ys):
    """等級相關。用等級而非原值：三類的尺度與分佈都不同，Pearson 會被
    尺度與離群值主導，而我們要問的是「排序一不一致」。

    平手以平均等級處理（`_ranks`），否則有平手時 ρ 會系統性偏高。
    """
    if len(xs) != len(ys):
        raise ValueError(f"長度不符：{len(xs)} vs {len(ys)}")
    n = len(xs)
    if n < 3:
        raise ValueError(f"n={n} 太少，等級相關沒有意義")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        raise ValueError("其中一軸完全沒有變異，相關性無定義")
    return num / (dx * dy)


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def collect(runs_root: Path):
    """回傳 {guidance_scale: [(net_lpips, dsiglip, dniqe, run, purify), ...]}。

    只取無淨化那一列（`identity`、strength 0）：淨化條件的 `net_lpips` 已扣掉
    對照，但三類的比較在乾淨攻擊上最乾淨，混入淨化會多一個變因。
    """
    groups = defaultdict(list)
    missing_env = []
    for run in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        rc, ec = run / "results.csv", run / "env.json"
        if not rc.exists():
            continue
        if not ec.exists():
            missing_env.append(run.name)
            continue
        gs = json.loads(ec.read_text(encoding="utf-8")).get("guidance_scale")
        if gs is None:
            # 該欄位是 E26 才加的。補 1.0 不是猜測：E26 §3 確立
            # `src/models/sd.py` 在那之前全專案沒有 classifier-free guidance，
            # 等同 w=1。故對缺欄的 run 補 1.0 有明確依據，但仍標為推定值，
            # 讓報告分得出哪些是記錄的、哪些是推的。
            gs = 1.0
            missing_env.append(run.name)
        with open(rc, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["purify"] != "identity" or r["strength"] != "0.0":
                    continue
                need = ("net_lpips", "edit_siglip_a", "edit_siglip_b",
                        "edit_niqe_a", "edit_niqe_b")
                if any(not r.get(k) for k in need):
                    continue
                groups[float(gs)].append((
                    float(r["net_lpips"]),
                    float(r["edit_siglip_b"]) - float(r["edit_siglip_a"]),
                    float(r["edit_niqe_b"]) - float(r["edit_niqe_a"]),
                    run.name, r["noise_split"],
                ))
    return groups, missing_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/p16_criterion_correlation")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    groups, missing = collect(ROOT / args.runs)
    if not groups:
        raise RuntimeError("沒有任何可用的格。請確認 runs/ 內有 results.csv")

    rows = []
    print(f"{'guidance':>9} {'n':>5}  {'ρ(距離, 語意)':>14} "
          f"{'ρ(距離, 劣化)':>14} {'ρ(語意, 劣化)':>14}")
    for gs in sorted(groups):
        data = groups[gs]
        net = [d[0] for d in data]
        sem = [d[1] for d in data]
        deg = [d[2] for d in data]
        r = {"guidance_scale": gs, "n": len(data),
             "rho_dist_sem": round(spearman(net, sem), 4),
             "rho_dist_deg": round(spearman(net, deg), 4),
             "rho_sem_deg": round(spearman(sem, deg), 4),
             "mean_net_lpips": round(sum(net) / len(net), 4),
             "mean_dsiglip": round(sum(sem) / len(sem), 5),
             "mean_dniqe": round(sum(deg) / len(deg), 4),
             "runs": len({d[3] for d in data})}
        rows.append(r)
        print(f"{gs:>9} {r['n']:>5}  {r['rho_dist_sem']:>14.4f} "
              f"{r['rho_dist_deg']:>14.4f} {r['rho_sem_deg']:>14.4f}")

    with open(out / "correlation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    for r in rows:
        print(f"guidance={r['guidance_scale']}：{r['runs']} 個 run、"
              f"{r['n']} 格；平均 net_lpips={r['mean_net_lpips']}、"
              f"Δsiglip={r['mean_dsiglip']:+}、Δniqe={r['mean_dniqe']:+}")
    if missing:
        print(f"\n其中 {len(missing)} 個 run 的 env.json 沒有 guidance_scale 欄"
              f"（該欄 E26 才加），已依 E26 §3 推定為 1.0——全專案在那之前"
              f"沒有 classifier-free guidance。")
        print(f"  {', '.join(missing[:8])}"
              f"{' …' if len(missing) > 8 else ''}")
    print(f"\n[p16] 寫出 {out / 'correlation.csv'}")
    print("[p16] 讀法見本檔 docstring。ρ 接近 0 代表兩類量的是無關的東西——"
          "那正是「判準換了、目標函數沒跟著換」這個問題的量化版本。")


if __name__ == "__main__":
    main()
