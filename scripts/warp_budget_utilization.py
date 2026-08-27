"""位移場的預算使用率診斷：`opt_r*` 那一批是**預算不夠**還是**沒有訓練成功**。

**不跑 GPU，不載編輯模型。** 只用 `WarpParam` 的前向與本機的指標套件。

要分辨的兩件事
────────────────────────────────────────────────────────────────────
`runs/ip2p_warp/opt_r{4,8,16,24}/results.csv` 上，半徑放到 24 px、1000 步，
`fid_dists` 只有 0.0495，而失真帶是 0.1286–0.1447。兩個互斥的解釋：

  (甲) 預算不夠 —— L∞ 球 ‖c‖∞ ≤ radius 裡根本沒有落在失真帶內的點。
  (乙) 沒有訓練成功 —— 球裡有帶內的點，最佳化沒有走到那裡。

分辨的方法照 AdvDrop 那一格的先例（`PENDING.md`「機制進得了失真帶，是最佳化
到不了」）：先量**可行集的天花板**，再量**實際用掉多少預算**。

三條參考曲線，都在同一組 13 張、同一個 16×16 粗網格上
────────────────────────────────────────────────────────────────────
1. `corner`   ── `c` 的每一格都貼在 ±amp（隨機正負號）。這是 L∞ 球的**頂點**，
                 是該半徑下最極端的可行解，給的是**天花板**。
2. `gauss`    ── `c ~ N(0, σ)`，不夾。這是 **sign PGD 在梯度符號無偏時的終點
                 分布**：每一格做步長 α 的 ±1 隨機遊走，N 步之後
                 `c ≈ N(0, α√N)`。故這條曲線把「觀測到的失真」翻譯成「等效
                 的 σ」，再與 `α√N` 對照就知道最佳化到底有沒有方向。
3. `coherent` ── `c` 全格同值（＝整張平移）。同一個 `|c|` 下**最平滑**的場，
                 失真最低。真實的最佳化場若比隨機場平滑，等效 `|c|` 會落在
                 `gauss` 與 `coherent` 兩條反推值之間，故這一條給的是反推的
                 另一側界線。

事前寫下的判讀規則（在看到數字之前）
────────────────────────────────────────────────────────────────────
- 若 `corner` 在 `opt_r*` 用過的半徑上 DISTS **已經跨過 0.1286**，則 (甲) 被
  否證：預算是夠的，帶內點就在可行集裡。
- 把觀測 DISTS 反推成等效 σ，與**無偏隨機遊走**的預測
  `α√N = radius/(steps × 0.25) × √steps` 對照。比值落在 0.5–2 之間，代表
  1000 步 PGD 產生的東西與擲硬幣沒有差別 → (乙) 成立。
- 比值 > 3 代表梯度確實有一致方向、只是走不完預算，那是步長／步數問題而不是
  方向問題，結論要改寫成「訓練有效但未收斂」。
- 比值 < 0.5 時症狀是**振盪**（雙線性折點造成的週期 2 來回），仍屬 (乙)，
  但救法是換更新規則而不是加步數。

用法：
    python scripts/warp_budget_utilization.py --out runs/ip2p_warp
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.defense.param_pgd import WarpParam  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
# `run_param_pgd` 的步長：α = radius / (steps × saturate_at)，saturate_at 預設
# 0.25，`opt_r*` 那一批 steps = 1000，故 α = radius / 250。
SATURATE_AT = 0.25


def make_field(kind: str, shape, amp: float,
               gen: torch.Generator) -> torch.Tensor:
    if kind == "corner":
        s = torch.randint(0, 2, shape, generator=gen).to(torch.float32) * 2 - 1
        return s * amp
    if kind == "gauss":
        return torch.randn(shape, generator=gen) * amp
    if kind == "coherent":
        # 全格同值 = 整張平移；兩個通道各分到 amp/√2，合成量級恰為 amp。
        d = torch.empty(shape)
        d[:, 0] = amp * (2 ** -0.5)
        d[:, 1] = amp * (2 ** -0.5)
        return d
    raise ValueError("未知的場型 " + kind)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("runs/ip2p_warp"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images-file", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images13.txt"))
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amps", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             8.0, 12.0, 16.0, 24.0])
    ap.add_argument("--kinds", nargs="+",
                    default=["gauss", "corner", "coherent"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = [n for n in args.images_file.read_text(encoding="utf-8").split()
             if n]
    suite = MetricSuite(device=torch.device(args.device))
    imgs = []
    for n in names:
        hits = sorted((args.data / n).glob("*.png")) + \
            sorted((args.data / n).glob("*.jpg"))
        if not hits:
            raise SystemExit(str(args.data / n) + " 底下沒有影像")
        imgs.append((n, load_image_tensor(hits[0], torch.device(args.device),
                                          size=RESOLUTION)))
    print("%d 張、%d 個量級、%d 種場型"
          % (len(imgs), len(args.amps), len(args.kinds)), flush=True)

    rows = []
    for kind in args.kinds:
        for amp in args.amps:
            per = {"dists": [], "psnr": [], "lpips": [], "disp": [],
                   "c_absmean": []}
            for i, (name, x) in enumerate(imgs):
                gen = torch.Generator().manual_seed(args.seed * 1000 + i)
                # 半徑設得遠大於量級，投影不介入——本支量的是場本身的代價，
                # 不是投影之後的場。
                p = WarpParam(radius=1e6, grid=args.grid)
                p.reset(x, args.seed)
                c = make_field(kind, tuple(p.c.shape), amp, gen)
                with torch.no_grad():
                    p.c.copy_(c.to(device=x.device, dtype=x.dtype))
                    y = p.render(x)
                m = suite.pairwise(x, y)
                eff = p.effective_displacement(x)
                per["dists"].append(float(m["dists"]))
                per["psnr"].append(float(m["psnr"]))
                per["lpips"].append(float(m["lpips"]))
                per["disp"].append(float(eff.pow(2).sum(1).sqrt().mean()))
                per["c_absmean"].append(float(p.c.abs().mean()))
            rows.append({
                "field": kind, "amp_px": amp, "warp_grid": args.grid,
                "n_images": len(imgs), "seed": args.seed,
                "fid_dists": round(statistics.fmean(per["dists"]), 5),
                "fid_psnr": round(statistics.fmean(per["psnr"]), 3),
                "fid_lpips": round(statistics.fmean(per["lpips"]), 5),
                "effective_disp_px": round(statistics.fmean(per["disp"]), 4),
                "c_absmean_px": round(statistics.fmean(per["c_absmean"]), 4),
            })
            print(rows[-1], flush=True)
            write_csv(args.out / "budget_utilization.csv", rows)

    # ── 反推：把 opt_r* 觀測到的失真翻譯成等效量級，與隨機遊走的預測對照 ──
    obs = []
    for tag in ("opt_r4", "opt_r8", "opt_r16", "opt_r24"):
        p = args.out / tag / "results.csv"
        if not p.exists():
            raise SystemExit("找不到 " + str(p) + "；本支要拿它的觀測值反推")
        r = list(csv.DictReader(p.open(encoding="utf-8")))
        obs.append({
            "tag": tag,
            "radius": float(r[0]["radius"]),
            "steps": int(r[0]["defense_steps"]),
            "n_images": len(r),
            "fid_dists": statistics.fmean(float(x["fid_dists"]) for x in r),
            "fid_psnr": statistics.fmean(float(x["fid_psnr"]) for x in r),
        })

    def invert(kind: str, d: float) -> float:
        """在該場型的「量級 → DISTS」曲線上線性內插出等效量級。

        落在掃描範圍外一律回傳 NaN，**不外插**（與
        `matched_distortion_table.py` 同一條規則）。
        """
        pts = sorted((r["fid_dists"], r["amp_px"]) for r in rows
                     if r["field"] == kind)
        for k in range(1, len(pts)):
            if pts[k][0] >= d:
                d0, a0 = pts[k - 1]
                d1, a1 = pts[k]
                return a0 + (a1 - a0) * (d - d0) / (d1 - d0)
        return float("nan")

    inv_rows = []
    for o in obs:
        alpha = o["radius"] / (o["steps"] * SATURATE_AT)
        sigma_rw = alpha * (o["steps"] ** 0.5)
        s_g = invert("gauss", o["fid_dists"])
        s_c = invert("coherent", o["fid_dists"])
        inv_rows.append({
            "tag": o["tag"], "radius_px": o["radius"], "steps": o["steps"],
            "n_images": o["n_images"],
            "observed_fid_dists": round(o["fid_dists"], 5),
            "observed_fid_psnr": round(o["fid_psnr"], 3),
            "alpha_px_per_step": round(alpha, 5),
            "path_length_px": round(alpha * o["steps"], 3),
            "random_walk_sigma_px": round(sigma_rw, 4),
            "equiv_sigma_gauss_px": round(s_g, 4),
            "equiv_amp_coherent_px": round(s_c, 4),
            "equiv_over_random_walk": round(s_g / sigma_rw, 3),
            "budget_used_frac": round(s_g / o["radius"], 4),
        })
        print(inv_rows[-1], flush=True)
    write_csv(args.out / "budget_utilization_inversion.csv", inv_rows)
    print("\n表：" + str(args.out / "budget_utilization.csv") + "、"
          + str(args.out / "budget_utilization_inversion.csv"))


if __name__ == "__main__":
    main()
