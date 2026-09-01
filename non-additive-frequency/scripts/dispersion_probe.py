"""色散度探針：每單位失真換到多少 latent 位移，對「色散度」的曲線。

要回答什麼
────────────────────────────────────────────────────────────────────
位移場（FND-004）死了兩次：`ip2p_warp/DIAGNOSIS.md` 證明預算夠、是最佳化沒有
走；`ip2p_warp_hard` 補上兩個端點（grid 16 到不了失真帶、grid 64 到得了但付
1.92 倍失真只換 65% 的位移）。

本檔問的是**歸因**：病灶是「位移」還是「單值」？單值位移場是微分同胚，把一張
自然影像映成另一張自然影像；`matched_geometry.csv` 已量到 `warp_roundtrip`
（幾何抵銷、只留內插 artifact）與 `warp_rand` 在等失真上只差 8.6%——**幾何本身
不值錢，走樣才值錢**。若病灶是「單值」，那麼讓每個頻帶各自平移（色散變形）
就應該在同一個失真上換到更多 latent 位移。

色散度是一條連續的軸（見 `src/defense/dispersion.py`）：

    K = 1               古典位移場，**已知失敗，是內建對照**
    K = 2..8            色散變形
    K = 每個頻格獨立     現行的紋理重相位

量什麼
────────────────────────────────────────────────────────────────────
`latent_move = ‖E(x') − E(x)‖`，對 DISTS 的比值。**不做最佳化、不跑編輯**，
只有 VAE 前向，故成本是分鐘量級而不是 GPU-小時量級。

**這個排序不作結論。** 探針量的是隨機方向，最佳化會自己挑方向，本專案兩次
踩過這件事——隨機平面把天花板低估 18.6 倍，定價取向的天花板又高估 1.4 倍而
最佳化之後打平。這裡的讀數只用來決定**要不要上機**。

事前判準（設計文件 §2.4）
────────────────────────────────────────────────────────────────────
    D1  disp_k4 的效率 >= disp_kfull 的 90%  → 值得做成完整參數化
    D2  disp_k4 的效率 <= disp_kfull 的 50%  → 色散假說成立，位移場族結案
    D3  warp_fold 的效率 >= warp_smooth 的 1.5 倍 → 非微分同胚是獨立的軸
    D4  效率隨 K 單調 → 否則「色散度」不是好的描述量，歸因要重寫
    50%–90% 之間記為「不決定」，不強行解讀。

輸出兩份 CSV：逐列的 `results.csv`，與在共同 DISTS 錨點上內插的 `summary.csv`。
**內插落在該條件掃描範圍外一律標 out_of_range，不外插**（`DECISIONS.md`）。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.dispersion import (  # noqa: E402
    apply_theta, band_index, displacement_theta, fold_fraction, make_operator,
    random_displacements, random_phase_theta,
)
from src.defense.param_pgd import WarpRandomParam  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512

# 三族的強度單位不同（像素／弧度／像素），**故不可跨族比同一個 amp**，
# 一律在等失真的錨點上比。梯子取幾何級數以便內插。
DEFAULT_AMPS = {
    "disp": (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),   # 每帶位移，像素
    "phase": (0.1, 0.2, 0.4, 0.8, 1.6, math.pi),     # 逐頻格相位，弧度
    "warp": (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),         # 單值位移場，像素
}
# 梯子兩端都刻意延伸得比「看起來夠用」更遠：等失真內插**拒絕外插**
# （`DECISIONS.md`），任何一條曲線構不到錨點那一格就只能標 out_of_range。
# 逐頻帶位移那一族尤其需要往上延伸——2 px 只走到 DISTS 0.034–0.079，而失真帶
# 的下界是 0.1286，窄梯子會讓帶內那幾個錨點全部落空，而那正是要判的地方。
# 多跑幾個強度的代價只是幾次 VAE 前向。

# 粗網格邊長：16 是 `WarpParam` 的定案值（本機量過的失真對照表就是在它上面
# 做的）；64 是 `ip2p_warp_hard` 走到失真帶的那一格，也是折疊會出現的那一格。
WARP_GRIDS = {"warp_smooth": 16, "warp_fold": 64}


def dispersive_conditions(ks: Sequence[int]) -> List[str]:
    return [f"disp_k{k}" for k in ks]


def condition_family(cond: str) -> str:
    if cond.startswith("disp_k") and cond != "disp_kfull":
        return "disp"
    if cond == "disp_kfull":
        return "phase"
    return "warp"


def render(cond: str, x01: torch.Tensor, amp: float, seed: int,
           block: int, hop: int, r_min: float, field_grid: int = 16):
    """回傳 `(x_def, extras)`。`extras` 逐列寫進 CSV。"""
    if cond in WARP_GRIDS:
        p = WarpRandomParam(radius=amp, grid=WARP_GRIDS[cond])
        p.reset(x01, seed)
        x_def = p.render(x01)
        # **折疊比例量的是實際位移場**，不是粗網格係數：雙三次上採樣會過衝，
        # 兩者的梯度不同。
        disp = p.effective_displacement(x01)
        return x_def, {"fold_frac": round(fold_fraction(disp), 5),
                       "grid": WARP_GRIDS[cond], "n_bands": ""}

    op = make_operator(x01, block, hop, r_min)
    if cond == "disp_kfull":
        bands = band_index(block, 1, r_min, x01.device, dtype=x01.dtype)
        theta = random_phase_theta(op.side, block, amp, seed, bands,
                                   x01.device, grid=field_grid, dtype=x01.dtype)
        n_bands = ""
    else:
        k = int(cond[len("disp_k"):])
        bands = band_index(block, k, r_min, x01.device, dtype=x01.dtype)
        u = random_displacements(op.side, k, amp, seed, x01.device,
                                 grid=field_grid, dtype=x01.dtype)
        theta = displacement_theta(block, bands, u)
        n_bands = k
    # 位移場那兩格有折疊比例可量，頻域這幾格沒有**單一**位移場可言——那正是
    # 這個構造的重點，故欄位留空而不是填 0。
    return apply_theta(op, x01, theta), {"fold_frac": "", "grid": "",
                                         "n_bands": n_bands}


def interpolate(points: Sequence[tuple], anchor: float):
    """在 `(dists, move)` 的折線上取 `anchor` 處的值。範圍外回傳 None。"""
    pts = sorted(points)
    if not pts or anchor < pts[0][0] or anchor > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x0 <= anchor <= x1:
            if x1 == x0:
                return y0
            t = (anchor - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


def build_summary(rows: Sequence[dict], anchors: Sequence[float]) -> List[dict]:
    """逐條件先對影像取平均，再在共同 DISTS 錨點上內插。"""
    curves: Dict[str, Dict[float, List[tuple]]] = {}
    for r in rows:
        curves.setdefault(r["condition"], {}).setdefault(
            float(r["amp"]), []).append((float(r["dists"]), float(r["latent_move"])))
    out = []
    for cond, by_amp in curves.items():
        pts = []
        for amp, vals in sorted(by_amp.items()):
            d = sum(v[0] for v in vals) / len(vals)
            m = sum(v[1] for v in vals) / len(vals)
            pts.append((d, m))
        for a in anchors:
            v = interpolate(pts, a)
            out.append({
                "condition": cond,
                "anchor_dists": a,
                "latent_move": round(v, 4) if v is not None else "",
                "move_per_dists": round(v / a, 2) if v is not None else "",
                "out_of_range": int(v is None),
                "dists_lo": round(pts[0][0], 5),
                "dists_hi": round(pts[-1][0], 5),
                "n_amps": len(pts),
            })
    return out


def parse_pairs(items: Sequence[str]) -> List[tuple]:
    """把 `cond:amp` 解析成 (條件, 振幅)。格式錯了當場拋錯，不猜。"""
    out = []
    for it in items:
        if ":" not in it:
            raise SystemExit(f"--edit-pairs 的格式是 COND:AMP，收到 {it!r}")
        c, a = it.rsplit(":", 1)
        out.append((c, float(a)))
    return out


def run_edit_readout(args, ip2p, suite, dataset) -> None:
    """真讀數：每一格跑一次編輯，量 `edit_lpips`。

    **原圖的編輯每張只跑一次**，所有條件共用——它與條件無關，重跑只是把
    46.8 秒乘上條件數。
    """
    pairs = parse_pairs(args.edit_pairs)
    rows: List[dict] = []
    for item in dataset:
        x = load_image_tensor(item["path"], ip2p.device, size=RESOLUTION)
        e_orig = ip2p.edit(x, item["prompt"])
        for cond, amp in pairs:
            with torch.no_grad():
                x_def, extras = render(cond, x, amp, args.seed, args.block,
                                       args.hop, args.r_min, args.field_grid)
                m = suite.pairwise(x, x_def)
            e_def = ip2p.edit(x_def, item["prompt"])
            prot = suite.pairwise(e_orig, e_def)
            # **兩張編輯輸出之間**的相似度，與 `ip2p_run.py` 的 `edit_clip_sim`
            # 逐字同義（該處是 `suite.image_similarity(e_orig, e_def)`）。
            # 影像對文字的 `semantic()` 是另一個量，代理門檻 0.8445 配的不是它。
            sim = suite.image_similarity(e_orig, e_def)
            rows.append({
                "image": item["name"],
                "condition": cond,
                "amp": amp,
                "block": args.block, "hop": args.hop, "r_min": args.r_min,
                "field_grid": args.field_grid, "seed": args.seed,
                "dists": round(float(m["dists"]), 6),
                "lpips": round(float(m["lpips"]), 6),
                "psnr": round(float(m["psnr"]), 3),
                "edit_lpips": round(float(prot["lpips"]), 5),
                "edit_clip_sim": round(float(sim["clip"]), 5),
                "edit_siglip_sim": round(float(sim["siglip"]), 5),
                **extras,
            })
            write_csv(args.out / "edit_readout.csv", rows)
            print(f"{item['name']:32s} {cond:12s} amp={amp:<8g} "
                  f"dists={rows[-1]['dists']:.5f} "
                  f"effect={rows[-1]['edit_lpips']:.4f} "
                  f"clip={rows[-1]['edit_clip_sim']:.4f}", flush=True)
    print("")
    print("真讀數：{}（{} 列）".format(args.out / "edit_readout.csv", len(rows)))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("runs/dispersion_probe"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", nargs="+", default=None, help="分片用")
    ap.add_argument("--bands", type=int, nargs="+", default=[1, 2, 3, 4, 8],
                    help="要掃的 K。**1 一定要在裡面**：它是古典位移場那一格，"
                         "整條軸的內建對照組")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--hop", type=int, default=8,
                    help="與主線定案值相同（DEC：hop 8）")
    ap.add_argument("--r-min", type=float, default=0.12)
    ap.add_argument("--field-grid", type=int, default=16,
                    help="視窗格點上那個場的空間粗糙度：先在 grid×grid 上抽"
                         "再雙三次上採樣。**固定它，K 才是唯一的變因**——"
                         "逐視窗獨立抽（0）會讓 K=1 在空間上也最粗，色散度"
                         "的效果就與空間粗糙度混在一起。16 對齊 WarpParam 的"
                         "粗網格，本專案量過的位移—失真對照表就在那個構造上")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--edit-pairs", nargs="+", default=None, metavar="COND:AMP",
                    help="給定時改跑**真讀數**：只渲染這些 (條件, 振幅) 組合，"
                         "每一格跑一次編輯並量 edit_lpips。存在的理由是 "
                         "latent_move 這個代理**與被測的那條軸混淆**——平滑的"
                         "幾何搬移把 latent 推得很遠，但推到的是「一張平移過的"
                         "自然影像」的 latent，編輯照樣成功。同一個失真錨點上"
                         "`warp_rand` 的 latent 效率是全表最高的 660，真讀數卻"
                         "只有 0.285（`runs/ip2p_warp/matched_geometry.csv`）。"
                         "振幅由既有的 results.csv 挑成括住失真帶")
    ap.add_argument("--anchors", type=float, nargs="+",
                    default=[0.05, 0.10, 0.1286, 0.15],
                    help="等失真錨點。0.1286 是失真帶的下界")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from src.models.ip2p import IP2PWrapper
    from scripts.ip2p_run import load_dataset  # noqa

    ip2p = IP2PWrapper(dtype=torch.float32)
    suite = MetricSuite(device=ip2p.device)
    dataset = load_dataset(args.data, prompt_index=0)
    if args.images:
        keep = set(args.images)
        dataset = [d for d in dataset if d["name"] in keep]
    if not dataset:
        raise SystemExit(f"{args.data} 底下沒有符合 --images 的影像")
    if 1 not in args.bands:
        raise SystemExit(
            "--bands 必須含 1：K=1 是古典位移場，整條色散度軸的對照組。"
            "少了它，「K 越大越好」就沒有起點可比。")

    if args.edit_pairs:
        run_edit_readout(args, ip2p, suite, dataset)
        return

    conds = dispersive_conditions(args.bands) + [
        "disp_kfull", "warp_smooth", "warp_fold"]
    rows: List[dict] = []
    for item in dataset:
        x = load_image_tensor(item["path"], ip2p.device, size=RESOLUTION)
        with torch.no_grad():
            z0 = ip2p.encode_image(x)
        for cond in conds:
            for amp in DEFAULT_AMPS[condition_family(cond)]:
                with torch.no_grad():
                    x_def, extras = render(cond, x, amp, args.seed,
                                           args.block, args.hop, args.r_min,
                                           args.field_grid)
                    m = suite.pairwise(x, x_def)
                    move = float((ip2p.encode_image(x_def) - z0).flatten().norm())
                d = float(m["dists"])
                rows.append({
                    "image": item["name"],
                    "condition": cond,
                    "amp": amp,
                    "block": args.block,
                    "hop": args.hop,
                    "r_min": args.r_min,
                    "field_grid": args.field_grid,
                    "seed": args.seed,
                    "dists": round(d, 6),
                    "lpips": round(float(m["lpips"]), 6),
                    "psnr": round(float(m["psnr"]), 3),
                    "ssim": round(float(m["ssim"]), 5),
                    "rms": round(float((x_def - x).pow(2).mean().sqrt()), 6),
                    "linf": round(float((x_def - x).abs().max()), 5),
                    "latent_move": round(move, 4),
                    "move_per_dists": round(move / d, 2) if d > 0 else "",
                    **extras,
                })
                write_csv(args.out / "results.csv", rows)
                print(f"{item['name']:32s} {cond:12s} amp={amp:<6g} "
                      f"dists={d:.5f} move={move:.2f} "
                      f"ratio={move / d if d > 0 else 0:.1f}", flush=True)

    write_csv(args.out / "summary.csv", build_summary(rows, args.anchors))
    print(f"\n逐列：{args.out / 'results.csv'}（{len(rows)} 列）")
    print(f"等失真錨點：{args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
