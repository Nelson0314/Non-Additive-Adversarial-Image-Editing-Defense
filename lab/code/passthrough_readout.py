"""穿透分離讀數：把位移拆成「防禦圖的色調直接穿過編輯」與其餘部分。

讀數（`docs/NEXT_PLAN.md` 2.3 節）
────────────────────────────────────────────────────────────────────
    D    = LPIPS( edit(x),       edit(x_def) )     現行主讀數
    P    = LPIPS( edit(x),       T̂(edit(x)) )      編輯器完全等變時的位移預測
    D_T  = LPIPS( T̂(edit(x)),    edit(x_def) )     扣除穿透後的位移

`T̂` 是由 `(x, x_def)` 回推的全域映射：以該臂所屬的參數族（`AbWarpParam`、
`AbPrismParam`、`ColorCurveParam`）最小化 Lab 均方差。三個量都分全圖／主體／背景，
另報 `T̂(edit(x))` 對 `edit(x_def)` 的 SigLIP 與 `blocked_T`。

`T̂` 在 `edit(x)` 中落在 `x` 色彩支撐之外的像素上是外插；逐格報該比例
（RGB 33³ 格的佔用判定），與 `D_T` 並列。

子命令
────────────────────────────────────────────────────────────────────
    fit        回推 T̂，存 `{out}/fit/<臂>/<影像>.pt` 與 `{out}/fit.csv`
    readout    逐格算 D／P／D_T，寫 `{out}/passthrough[_<tag>].csv`
    baseline   非防禦性改動（主表的淨化後未防禦編輯）的輸入 LPIPS 與編輯端 LPIPS

用法（遠端，需要一張卡）
    python lab/code/passthrough_readout.py fit --arms ab_warp ab_warp_ch ...
    python lab/code/passthrough_readout.py readout --arms ... [--seed 20260813]
    python lab/code/passthrough_readout.py baseline
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.defense.ncf_param import rgb_to_lab  # noqa: E402
from src.metrics.regional import RegionalLPIPS, split_displacement  # noqa: E402
from src.metrics.standard import SIGLIP_BLOCKED_THRESHOLD  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

R = Path(__file__).resolve().parents[2]
RES = 512
IMAGES = [f"{g}_{i:02d}" for g in ("man", "woman") for i in range(4)]
PURIFIERS = ["jpeg30", "jpeg50", "jpeg80", "blur1", "blur2"]

#: 臂 → (參數族, 建構參數, 防禦圖路徑樣板, 主種子編輯圖樣板)。參數與 defence_cmd.sh 一致。
WARP = dict(grid=7, extent=90.0, pieces=16, l_radius=0.6)
ARMS = {
    "ab_warp": ("warp", dict(WARP, warp_radius=30.0)),
    "style_warp": ("warp", dict(WARP, warp_radius=30.0)),
    "ab_warp_s12": ("warp", dict(WARP, warp_radius=80.0)),
    "ab_warp_s16": ("warp", dict(WARP, warp_radius=80.0)),
    "ab_warp_ch": ("warp", dict(WARP, warp_radius=80.0)),
    **{a: ("warp", dict(WARP, warp_radius=80.0)) for a in (
        "ab_warp_ch_comm", "ab_warp_ch_free",
        *[f"ab_warp_ch_{o}_random_r{k}" for o in ("comm", "free") for k in (1, 2, 3)])},
    "ab_prism": ("prism", {}),
    "curve_dual_chroma": ("curve", dict(radius=5.0, pieces=64, bound_mode="advcf")),
    "colour_curve_ours": ("curve", dict(radius=5.0, pieces=64, bound_mode="advcf")),
}


def defence_png(arm, name):
    if arm == "colour_curve_ours":          # 主表那一批
        return R / "runs/defence_portraits/colour_curve_ours" / f"{name}__{arm}__def.png"
    return R / "lab/runs/defence" / arm / f"{name}__{arm}__def.png"


def edit_png(arm, name, k, seed=None):
    """未防禦與防禦後的 ip2p 編輯圖。主種子沿用既有檔；其餘種子在 edit_seeds/。"""
    if seed is None:
        if arm == "undefended":
            return R / "runs/edit_preflight/ip2p_si18" / f"{name}__p{k}.png"
        if arm == "colour_curve_ours":
            return R / "runs/edit_defended/colour_curve_ours/ip2p_colour_curve_ours" / f"{name}__p{k}.png"
        return R / "lab/runs/edit_defended" / arm / f"ip2p_{arm}" / f"{name}__p{k}.png"
    return R / "lab/runs/edit_seeds" / arm / f"seed{seed}" / f"ip2p_{arm}" / f"{name}__p{k}.png"


def original(name):
    return R / "lab/data/portraits" / name.split("_")[0] / f"{name}.png"


def carrier_for(arm):
    fam, kw = ARMS[arm]
    if fam == "warp":
        from ab_warp_defence import AbWarpParam
        return AbWarpParam(**kw)
    if fam == "prism":
        from ab_prism_defence import AbPrismParam
        return AbPrismParam(**kw)
    from src.defense.color_param import ColorCurveParam
    return ColorCurveParam(**kw)


def load(path, device):
    if not Path(path).is_file():
        raise SystemExit(f"找不到 {path}")
    return load_image_tensor(Path(path), device, size=RES)


def quantise(y):
    return (y.detach().clamp(0, 1) * 255).round() / 255


def deltae_map(a, b):
    from skimage.color import deltaE_ciede2000, rgb2lab
    f = lambda t: rgb2lab(t[0].permute(1, 2, 0).double().cpu().numpy())
    return deltaE_ciede2000(f(a), f(b))


# ---- fit ----

def fit_one(arm, x, xd, steps, starts, lr):
    """多起點 Adam，Lab 均方差。擬合在 256²（映射與位置無關，降採樣不改變問題）。"""
    xs, target = x[..., ::2, ::2], rgb_to_lab(xd[..., ::2, ::2].float())
    best = None
    for s in range(starts):
        c = carrier_for(arm)
        c.reset(x, s)
        if s > 0:
            g = torch.Generator(device="cpu").manual_seed(1000 + s)
            with torch.no_grad():
                for p in c.params():
                    p.add_(0.3 * torch.randn(p.shape, generator=g).to(p.device))
            c.project()
        opt = torch.optim.Adam(c.params(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = (rgb_to_lab(c.render(xs).float()) - target).pow(2).sum(1).mean()
            loss.backward()
            opt.step()
            c.project()
            sched.step()
        with torch.no_grad():
            final = float((rgb_to_lab(c.render(xs).float()) - target).pow(2).sum(1).mean())
        if best is None or final < best[0]:
            best = (final, s, c.state_dict())
    c = carrier_for(arm)
    c.reset(x, 0)
    c.load_state_dict(best[2])
    return c, best


def keep_other_arms(path, arms):
    """只重算指定的臂時，保留檔裡其他臂的列（整份重寫會把它們丟掉）。"""
    if not path.is_file():
        return []
    return [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["arm"] not in arms]


def cmd_fit(args, device):
    rows = keep_other_arms(args.out / "fit.csv", set(args.arms))
    out = args.out / "fit"
    for arm in args.arms:
        for name in IMAGES:
            if not defence_png(arm, name).is_file():
                continue                       # 隨機對照在該影像沒有合格候選
            x, xd = load(original(name), device), load(defence_png(arm, name), device)
            c, (mse, start, state) = fit_one(arm, x, xd, args.steps, args.starts, args.lr)
            with torch.no_grad():
                de = deltae_map(quantise(c.render(x)), xd)
            (out / arm).mkdir(parents=True, exist_ok=True)
            torch.save(state, out / arm / f"{name}.pt")
            row = {"arm": arm, "image": name, "family": ARMS[arm][0], "best_start": start,
                   "fit_lab_mse": round(mse, 5), "fit_deltaE00_mean": round(float(de.mean()), 4),
                   "fit_deltaE00_p99": round(float(np.percentile(de, 99)), 4),
                   "fit_deltaE00_max": round(float(de.max()), 4)}
            rows.append(row)
            write_csv(args.out / "fit.csv", rows)
            print(row, flush=True)


# ---- readout ----

def occupancy(x):
    q = (x[0].permute(1, 2, 0).reshape(-1, 3) * 32).round().long()
    grid = torch.zeros(33, 33, 33, dtype=torch.bool, device=x.device)
    grid[q[:, 0], q[:, 1], q[:, 2]] = True
    return grid


def outside_fraction(grid, e):
    q = (e[0].permute(1, 2, 0).reshape(-1, 3) * 32).round().long()
    return float((~grid[q[:, 0], q[:, 1], q[:, 2]]).float().mean())


def subject_mask(name, device):
    m = load(R / "lab/data/portraits/masks" / f"{name}.png", device)[:, :1]
    return 1.0 - (m >= 0.5).float()


def cmd_readout(args, device):
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)
    fits = {(r["arm"], r["image"]): r for r in csv.DictReader(open(args.out / "fit.csv", encoding="utf-8"))}
    tag = f"_seed{args.seed}" if args.seed else ""
    rows = keep_other_arms(args.out / f"passthrough{tag}.csv", set(args.arms))
    for arm in args.arms:
        for name in IMAGES:
            dpng = defence_png(arm, name)
            if not dpng.is_file():
                continue                       # 隨機對照在該影像沒有合格候選
            x = load(original(name), device)
            c = carrier_for(arm)
            c.reset(x, 0)
            c.load_state_dict(torch.load(args.out / "fit" / arm / f"{name}.pt", map_location=device))
            grid = occupancy(x)
            mask = subject_mask(name, device)
            for k in range(4):
                a = load(edit_png("undefended", name, k, args.seed), device)
                b = load(edit_png(arm, name, k, args.seed), device)
                with torch.no_grad():
                    ta = quantise(c.render(a))
                    d = split_displacement(regional, a, b, mask)
                    p = split_displacement(regional, a, ta, mask)
                    dt = split_displacement(regional, ta, b, mask)
                    sig = float(suite.image_similarity(ta, b)["siglip"])
                f = fits[(arm, name)]
                rows.append({
                    "arm": arm, "image": name, "prompt_index": k,
                    "seed": args.seed or 20260812,
                    **{f"D_{r}": round(float(v), 5) for r, v in d.items()},
                    **{f"P_{r}": round(float(v), 5) for r, v in p.items()},
                    **{f"DT_{r}": round(float(v), 5) for r, v in dt.items()},
                    "siglip_pair_T": round(sig, 5),
                    "blocked_T": sig < SIGLIP_BLOCKED_THRESHOLD,
                    "siglip_blocked_threshold": SIGLIP_BLOCKED_THRESHOLD,
                    "extrapolated_frac": round(outside_fraction(grid, a), 5),
                    "fit_deltaE00_mean": f["fit_deltaE00_mean"],
                    "fit_deltaE00_p99": f["fit_deltaE00_p99"],
                    "other_batch": arm == "colour_curve_ours",
                })
            write_csv(args.out / f"passthrough{tag}.csv", rows)
            print(f"[{arm}] {name} done", flush=True)


# ---- baseline ----

def cmd_baseline(args, device):
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)
    rows = []
    for pur in PURIFIERS:
        for name in IMAGES:
            x = load(original(name), device)
            xp = load(R / "runs/purified/undefended" / pur / f"{name}__def.png", device)
            mask = subject_mask(name, device)
            with torch.no_grad():
                inp = float(suite.lpips_module(xp, x).mean())
            for k in range(4):
                a = load(edit_png("undefended", name, k), device)
                b = load(R / "runs/edit_purified/undefended" / pur / f"ip2p_undefended_{pur}" / f"{name}__p{k}.png", device)
                with torch.no_grad():
                    d = split_displacement(regional, a, b, mask)
                rows.append({"purifier": pur, "image": name, "prompt_index": k,
                             "input_lpips": round(inp, 5),
                             **{f"D_{r}": round(float(v), 5) for r, v in d.items()}})
        write_csv(args.out / "passthrough_baseline.csv", rows)
        print(f"[baseline] {pur} done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fit", "readout", "baseline"))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", type=Path, default=R / "lab/results/passthrough")
    ap.add_argument("--seed", type=int, default=None, help="readout：非主種子的評估種子")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.03)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU（LPIPS／SigLIP 與 3,000 步回推）；不要靜默退回 CPU")
    args.out.mkdir(parents=True, exist_ok=True)
    {"fit": cmd_fit, "readout": cmd_readout, "baseline": cmd_baseline}[args.cmd](args, torch.device("cuda"))


if __name__ == "__main__":
    main()
