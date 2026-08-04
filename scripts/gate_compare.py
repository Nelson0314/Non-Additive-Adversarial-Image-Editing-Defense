"""比較兩個 site 在各自可達失真上的「相對隨機的優勢」。

    python scripts/gate_compare.py --runs runs/gate_suppress,runs/gate_S

零 GPU，只讀 CSV。

## 比較的量為什麼不是 Δsiglip 本身

兩個 site 未必能到達同一個失真。site S 的位移場受 `max_disp` 上界，
L4 在 τ_lpips = 0.10 上只做到 LPIPS 0.025（LEDGER 3.23）；即使放寬上界，
平滑位移場能產生的感知失真仍有上限。兩個條件停在不同失真上時，
直接比 Δsiglip 就不是匹配失真的比較。

故比較的量取**相對同失真隨機的優勢**：

    優勢 = Δsiglip(最佳化) − Δsiglip(同 LPIPS 的隨機)

每個 site 的隨機條件匹配的是**該 site 自己達到的**失真，故這個量在構造上
已對失真正規化。負值代表最佳化比隨機更能降低 prompt 服從度。

## 為什麼要配對

兩個條件共用逐元素相同的評測 ε，故逐種子相減才是正確的比較。配對使所需樣本數
由約 48 降到約 7（LEDGER 1.23）。本腳本一律以配對方式計算，並報 Cohen d
與達到 power 80% 所需的 n。
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

# 這些鍵在每個 run 之間必然不同或已在列識別中，不算變因軸。
# `seed` 排除的理由不同：只差 seed 的兩個 run 是重複試驗，平均它們才是對的。
NON_AXIS = {"out", "data", "model", "images", "sites", "seed"}


def variant_axes(protos: dict) -> dict:
    """比出同一個 site 內各 run 之間**真正動到的設定軸**。

    這取代「以 site 單獨分組」。原作法為「一個 site 一個 run」而寫，一旦
    同一個 site 有多個設定（例如 grid_size 32 與 128），兩者會被平均成一個
    數字，而那個數字正好抹掉要量的效應，且沒有任何症狀——與 LEDGER 第六類
    的九次缺陷同形。改成由協定檔自動比出差異，就不必在此寫死是哪個軸。

    某個鍵在一個 run 有、另一個 run 沒有，同樣算差異（記為 `<未記錄>`），
    因為「沒記錄」不等於「和對方一樣」。
    """
    keys = {k for p in protos.values() for k in p} - NON_AXIS
    axes = {k for k in keys
            if len({json.dumps(p.get(k, "<未記錄>"), ensure_ascii=False)
                    for p in protos.values()}) > 1}
    return {rd: "; ".join(f"{k}={p.get(k, '<未記錄>')}" for k in sorted(axes))
            for rd, p in protos.items()}


def load(run_dir: Path):
    """回傳 {(影像, site): {條件: {種子: Δsiglip}}} 與逐格的擾動 LPIPS。"""
    res, summ = run_dir / "results.csv", run_dir / "summary.csv"
    if not res.exists():
        raise SystemExit(f"{res} 不存在")
    d = defaultdict(lambda: defaultdict(dict))
    with open(res, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("edit_siglip_a"):
                continue
            key = (r["image"], r["site"])
            d[key][r["arm"]][r["eval_seed"]] = (
                float(r["edit_siglip_b"]) - float(r["edit_siglip_a"]))
    pert = {}
    if summ.exists():
        with open(summ, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("arm") in ("opt", "rand") and r.get("pert_lpips"):
                    pert[(r["image"], r["site"], r["arm"])] = float(
                        r["pert_lpips"])
    return d, pert


def paired(cell: dict):
    """回傳 (逐種子的優勢清單, 共用的種子數)。缺任一條件就拋出。"""
    if "opt" not in cell or "rand" not in cell:
        raise SystemExit(
            f"這一格缺條件：有 {sorted(cell)}，兩個位置都要有才能算優勢。"
            "沒有同失真的隨機對照，任何正結果都不可解讀（LEDGER 1.18）")
    seeds = sorted(set(cell["opt"]) & set(cell["rand"]))
    if len(seeds) < 3:
        raise SystemExit(
            f"兩個條件共用的評測種子只有 {len(seeds)} 個，配對統計沒有意義")
    return [cell["opt"][s] - cell["rand"][s] for s in seeds], len(seeds)


def main():
    ap = argparse.ArgumentParser(description="兩個 site 的相對隨機優勢比較")
    ap.add_argument("--runs", default="runs/gate_suppress,runs/gate_S")
    ap.add_argument("--out", default="runs/gate_compare.csv")
    args = ap.parse_args()

    run_dirs = [s.strip() for s in args.runs.split(",") if s.strip()]
    protos = {}
    for rd in run_dirs:
        f = ROOT / rd / "protocol.json"
        if not f.exists():
            raise SystemExit(
                f"{f} 不存在。沒有協定檔就無從得知這個 run 動了哪個設定軸，"
                "而把設定不同的 run 併成一格會靜默抹掉要量的效應")
        protos[rd] = json.load(open(f, encoding="utf-8"))

    rows = []
    for rd in run_dirs:
        d, pert = load(ROOT / rd)
        # 變因軸只在**同一個 site 的 run 之間**比。跨 site 的差異（lr、
        # warp_max_disp）是 site 本身的性質，不是被操作的變因。
        same_site = {r: p for r, p in protos.items()
                     if p.get("sites") == protos[rd].get("sites")}
        variant = variant_axes(same_site)[rd] if len(same_site) > 1 else ""
        for (img, site), cell in sorted(d.items()):
            adv, n = paired(cell)
            m, sd = st.mean(adv), st.stdev(adv)
            need = ((1.96 + 0.84) ** 2 * (sd / abs(m)) ** 2) if m else float("inf")
            rows.append({
                "run": rd, "image": img, "site": site, "variant": variant,
                "n_paired": n,
                "pert_lpips_opt": pert.get((img, site, "opt"), ""),
                "pert_lpips_rand": pert.get((img, site, "rand"), ""),
                "dsiglip_opt": round(st.mean(cell["opt"].values()), 5),
                "dsiglip_rand": round(st.mean(cell["rand"].values()), 5),
                "advantage_mean": round(m, 5),
                "advantage_sd": round(sd, 5),
                "cohen_d": round(m / sd, 3) if sd else "",
                "same_sign": sum(1 for v in adv if v < 0),
                "n_for_power80": round(need, 1),
            })

    if not rows:
        raise SystemExit("沒有任何可比較的格")

    w_var = max([len(r["variant"]) for r in rows] + [len("設定")])
    print(f"{'site':<6}{'影像':<10}{'設定':<{w_var}}{'n':>4}{'擾動LPIPS':>11}"
          f"{'Δsig(opt)':>11}{'Δsig(rand)':>12}{'優勢':>10}{'d':>9}{'同向':>9}")
    for r in rows:
        pl = r["pert_lpips_opt"]
        print(f"{r['site']:<6}{r['image']:<10}{r['variant']:<{w_var}}"
              f"{r['n_paired']:>4}"
              f"{(f'{pl:.4f}' if pl != '' else '—'):>11}"
              f"{r['dsiglip_opt']:>+11.5f}{r['dsiglip_rand']:>+12.5f}"
              f"{r['advantage_mean']:>+10.5f}"
              f"{(r['cohen_d'] if r['cohen_d'] != '' else '—'):>9}"
              f"{r['same_sign']:>5}/{r['n_paired']}")

    # 分組鍵是 (site, 設定) 而不是 site。只用 site 的話，同一個 site 的不同
    # 設定會被平均成一個數字——那正是掃容量軸時要看的差異。
    groups = sorted({(r["site"], r["variant"]) for r in rows})
    if len(groups) >= 2:
        print("\n各組的優勢（負值越大代表最佳化相對同失真隨機多取得越多）：")
        for s, v in groups:
            d = [r["advantage_mean"] for r in rows
                 if r["site"] == s and r["variant"] == v]
            print(f"  site {s:<4}{('［' + v + '］') if v else '':<{w_var + 2}}"
                  f"平均優勢 {st.mean(d):+.5f}（{len(d)} 格）")
        print("\n**各組的擾動 LPIPS 不必相同**：優勢的定義已對各自達到的"
              "失真正規化，那正是它取代直接比 Δsiglip 的理由。")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[gate] 寫出 {out}")


if __name__ == "__main__":
    main()
