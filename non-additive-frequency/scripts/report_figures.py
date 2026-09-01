"""報告頁要的所有靜態圖：相位方法的閘與學到的參數、位移場的流場、五張圖表。

**不跑 GPU、不需要擴散模型。** 只讀 `_rep/` 底下抓回來的 `__w.pt`／PNG 與
`runs/` 底下的 CSV，全部在 CPU 上重算。

閘與價目表是由**原圖**算出來的，構造上不參與最佳化（`docs/METHOD.md`），
所以在本機重建 `PhaseParam` 再 `reset()` 一次得到的閘，與訓練時用的是同一個。
學到的 `theta`／`gain`／`a` 由 `__w.pt` 載入，不是重跑。

用法：python scripts/report_figures.py --out _rep/fig
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402

# 缺中文字型時 matplotlib 只印 warning 然後把每個字畫成豆腐方塊——圖產得出來、
# 字全部看不懂。一個候選都沒有時明確拋錯。
_CJK = ("Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC",
        "Noto Sans TC", "PingFang TC", "Source Han Sans TW", "SimHei")
_have = {f.name for f in _fm.fontManager.ttflist}
_pick = [n for n in _CJK if n in _have]
if not _pick:
    raise SystemExit(f"找不到中文字型，試過 {_CJK}")
plt.rcParams["font.sans-serif"] = _pick + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np                       # noqa: E402
import torch                             # noqa: E402
from PIL import Image                    # noqa: E402

from src.defense.param_pgd import PhaseParam   # noqa: E402

REP = Path("_rep/runs")
IMAGES = ["task_attr_mod_color_11699", "task_attr_mod_color_6205"]
NAME = {"task_attr_mod_color_11699": "盆栽人",
        "task_attr_mod_color_6205": "瑪利歐"}


def load01(path: Path) -> torch.Tensor:
    a = np.asarray(Image.open(path).convert("RGB")).copy()
    return torch.from_numpy(a).permute(2, 0, 1)[None].float() / 255.0


def param_from_csv(run: Path) -> PhaseParam:
    """由該次執行自己的 `results.csv` 逐欄重建 `PhaseParam`。

    **不硬寫任何一個值。** 閘與價目表是由原圖算一次就固定的（`docs/METHOD.md`），
    所以只要設定欄對得上，本機重建出來的閘與訓練時用的就是同一個；設定寫錯
    的話重建出來的是另一個閘，而圖仍然畫得出來、不會有任何症狀。
    """
    import csv as _csv
    r = list(_csv.DictReader((run / "results.csv").open(encoding="utf-8")))[0]

    def f(k, d=0.0):
        v = r.get(k, "")
        return d if v in ("", None) else float(v)

    def s_(k, d=""):
        v = r.get(k, "")
        return d if v in ("", None) else v

    return PhaseParam(
        size=512, block=int(f("block", 32)), hop=int(f("hop", 8)),
        r_min=f("r_min", 0.12),
        r_max=float("inf") if s_("r_max", "inf") == "inf" else f("r_max"),
        radius=f("radius"), energy_quantile=f("quantile"),
        gain_ratio=f("gain_ratio"), gate_edge_power=f("gate_edge_power", 1.0),
        freq_weight=s_("freq_weight", "binary"),
        freq_weight_power=f("freq_weight_power", 1.0),
        gain_weight=s_("gain_weight", "shared"),
        channels=s_("phase_channels", "rgb"),
        spectral_floor=f("spectral_floor"),
        floor_gate=s_("floor_gate", "uniform"),
        floor_survival=s_("floor_survival", "none") or "none")


def shift_half(half: np.ndarray) -> np.ndarray:
    """rfft2 半平面 (32,17) → fftshift 後的完整 32×32 平面（共軛對稱鏡射）。"""
    n = half.shape[0]
    full = np.zeros((n, n), dtype=half.dtype)
    full[:, : n // 2 + 1] = half
    for u in range(n):
        for v in range(n // 2 + 1, n):
            full[u, v] = half[(-u) % n, n - v]
    return np.fft.fftshift(full)


def save(fig, out: Path, dpi=150):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out}  {out.stat().st_size/1e3:.0f} kB")


# ── 相位方法：閘、價目表、學到的參數 ────────────────────────────────────
def load_data(p="report_data.json") -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _flat_phase(D):
    out = {}
    for b, v in D["phase"].items():
        for t, r in v["defence"].items():
            out[t] = dict(r, batch=b, purify=v["purify"].get(t, {}))
    return out


def _flat_warp(D):
    out = {}
    for b, v in D["warp"].items():
        for t, r in v["defence"].items():
            out[t] = dict(r, batch=b, purify=v["purify"].get(t, {}))
    return out


def _lines(ax, series, ops, labels, title, ylab="淨增益 = effect(p) - 空白地板"):
    """`series` 是 `(標籤, 值序列, 顏色, 標記)`。逐算子畫點並連線。

    不用長條圖：多個條件並排時同一個算子的幾格隔著別人的長條，要用眼睛跨
    過去比。折線把同一個條件連起來，跨算子的走勢與條件之間的高低同時讀得到。
    """
    xs = np.arange(len(ops))
    for lab, v, col, mk in series:
        v = [np.nan if x is None else x for x in v]
        ax.plot(xs, v, mk + "-", color=col, ms=8, lw=2.0, mec="k", mew=0.5,
                label=lab)
        for x, y in zip(xs, v):
            if not np.isnan(y):
                ax.annotate(f"{y:.3f}", (x, y), fontsize=8, xytext=(0, 8),
                            textcoords="offset points", ha="center", color=col)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(ylab, fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=12)


def fig_baseline_tradeoff(D, out: Path):
    """`runs/ip2p_mainline`：DCT-Shield 的 ε 掃描與本方法在同一批裡。"""
    B = D["baseline"]["ip2p_mainline"]["defence"]
    pur = D["baseline"]["ip2p_mainline"]["purify"]
    groups = {"dct_shield_y": ("DCT-Shield（Y-only）", "#d62728", "s"),
              "dct_shield": ("DCT-Shield（base）", "#ff9896", "D"),
              "phase_gain": ("本方法　相位＋幅度增益", "#1f77b4", "o"),
              "phase": ("本方法　純相位", "#9edae5", "^")}
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    ax = axes[0]
    for cond, (lab, col, mk) in groups.items():
        pts = sorted(((r["fid_dists"], r["edit_lpips"], t)
                      for t, r in B.items() if r.get("condition") == cond))
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-",
                color=col, ms=8, lw=1.3, mec="k", mew=0.5, label=lab)
    ax.axhline(D["lpips_ceiling"], color="#00b8d4", ls=":", lw=1.2)
    ax.set_xlabel("失真　DISTS", fontsize=10)
    ax.set_ylabel("未淨化位移　LPIPS", fontsize=10)
    ax.set_title("未淨化", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for cond, (lab, col, mk) in groups.items():
        pts = []
        for t, r in B.items():
            if r.get("condition") != cond or t not in pur:
                continue
            v = pur[t].get("jpeg30", {}).get("net")
            if v is not None:
                pts.append((r["fid_dists"], v))
        if not pts:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-",
                color=col, ms=8, lw=1.3, mec="k", mew=0.5, label=lab)
    ax.set_xlabel("失真　DISTS", fontsize=10)
    ax.set_ylabel("JPEG 30 上的淨增益", fontsize=10)
    ax.set_title("JPEG 30 淨化之後", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    fig.suptitle("對照組 DCT-Shield：同一批（runs/ip2p_mainline）、"
                 "同兩張影像、JPEG 軸五個算子", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out)


def fig_warp_field(img: str, out: Path):
    """學到的稠密流場本身：位移長度圖、方向箭頭、與長度的分佈。

    張量是 `(1, 2, 512, 512)` 的**像素位移**（`sa_r04` 的絕對值上界恰好等於
    它的 radius 4.0，可據此確認單位）。隨機對照取同一批的 `warp_rand`，
    它是粗網格 16 上採樣的，不是稠密場——格點數不同，**併排看的是形態不是
    同一個構造**。
    """
    tags = ["sa_t000", "sa_r04", "sa_t001"]
    fig, axes = plt.subplots(3, len(tags) + 1,
                             figsize=(4.3 * (len(tags) + 1), 12.2))
    for c, t in enumerate(tags):
        f = torch.load(REP / WARP_SRC[t] / f"{img}__warp__w.pt",
                       map_location="cpu")[0][0].float().numpy()
        mag = np.sqrt(f[0] ** 2 + f[1] ** 2)
        im = axes[0][c].imshow(mag, cmap="inferno")
        fig.colorbar(im, ax=axes[0][c], fraction=0.046)
        axes[0][c].set_title(f"{t}　位移長度（像素）　"
                             f"最大 {mag.max():.2f}　均值 {mag.mean():.2f}",
                             fontsize=9)
        s = 16
        yy, xx = np.mgrid[0:512:s, 0:512:s]
        axes[1][c].quiver(xx, yy, f[0][::s, ::s], -f[1][::s, ::s],
                          mag[::s, ::s], cmap="inferno", scale=None,
                          width=0.0035)
        axes[1][c].set_xlim(0, 512); axes[1][c].set_ylim(512, 0)
        axes[1][c].set_aspect("equal")
        axes[1][c].set_title(f"{t}　方向（每 {s} 像素取一個）", fontsize=9)
        d = REP / WARP_SRC[t]
        axes[2][c].imshow(np.clip(
            (load01(d / f"{img}__warp__def.png")
             - load01(d / f"{img}__orig.png"))[0]
            .permute(1, 2, 0).numpy() * 4 + 0.5, 0, 1))
        axes[2][c].set_title(f"{t}　δ×4＋0.5", fontsize=9)
    # 最後一欄：同一張圖上三個條件的位移長度分佈，加上隨機對照。
    ax = axes[0][len(tags)]
    for t in tags:
        f = torch.load(REP / WARP_SRC[t] / f"{img}__warp__w.pt",
                       map_location="cpu")[0][0].float().numpy()
        mag = np.sqrt(f[0] ** 2 + f[1] ** 2).ravel()
        ax.hist(mag, bins=120, histtype="step", lw=1.5, label=t, density=True)
    ax.set_xlabel("位移長度（像素）", fontsize=8.5)
    ax.set_ylabel("機率密度", fontsize=8.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
    ax.set_title("位移長度的分佈", fontsize=9)
    for r, (d, lab) in enumerate([("ip2p_warp/rand_r4", "warp_rand radius 4"),
                                  ("ip2p_warp/opt_r24", "warp opt radius 24")]):
        a = axes[r + 1][len(tags)]
        a.imshow(np.clip(
            (load01(REP / d / f"{img}__{'warp_rand' if 'rand' in d else 'warp'}__def.png")
             - load01(REP / d / f"{img}__orig.png"))[0]
            .permute(1, 2, 0).numpy() * 4 + 0.5, 0, 1))
        a.set_title(f"{lab}（粗網格 16）　δ×4＋0.5", fontsize=9)
    for a in axes.ravel():
        if a is not axes[0][len(tags)]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"位移場：學到的稠密流場　{NAME[img]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, out)


def fig_warp_tradeoff(D, out: Path):
    """最佳化過的位移場 對 同失真的全隨機位移場。"""
    W = _flat_warp(D)
    P = _flat_phase(D)
    fig, ax = plt.subplots(figsize=(11.5, 7))

    rand = sorted((r["fid_dists"], r["edit_lpips"], t) for t, r in W.items()
                  if r.get("condition") == "warp_rand")
    ax.plot([p[0] for p in rand], [p[1] for p in rand], "o-", color="#7f7f7f",
            ms=8, mec="k", mew=0.5, lw=1.6, label="全隨機位移場　warp_rand（粗網格 16）")
    lo, hi = rand[0][0], rand[-1][0]
    ax.axvspan(lo, hi, color="#7f7f7f", alpha=0.08)
    ax.text((lo + hi) / 2, 0.02, f"隨機掃描的範圍 DISTS {lo:.4f}–{hi:.4f}\n"
            "（範圍外不可外插）", fontsize=8, color="#555", ha="center")

    opt = sorted((r["fid_dists"], r["edit_lpips"], t) for t, r in W.items()
                 if r.get("condition") == "warp" and r.get("warp_grid") == "16"
                 and t.startswith("opt_"))
    ax.plot([p[0] for p in opt], [p[1] for p in opt], "s-", color="#2ca02c",
            ms=8, mec="k", mew=0.5, lw=1.6, label="最佳化位移場　warp（粗網格 16）")

    for t, r in sorted(W.items()):
        if not t.startswith("sa_"):
            continue
        ax.scatter(r["fid_dists"], r["edit_lpips"], s=130, marker="^",
                   color="#d62728", edgecolor="k", linewidth=0.6, zorder=4)
        ax.annotate(t, (r["fid_dists"], r["edit_lpips"]), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    for t in ("w_adam_g64", "w_adam_rand"):
        if t in W:
            ax.scatter(W[t]["fid_dists"], W[t]["edit_lpips"], s=110,
                       marker="P", color="#9467bd", edgecolor="k",
                       linewidth=0.6, zorder=4)
            ax.annotate(t, (W[t]["fid_dists"], W[t]["edit_lpips"]), fontsize=8,
                        xytext=(6, 4), textcoords="offset points")

    for t in ("ig_f08_eot",):
        if t in P:
            ax.scatter(P[t]["fid_dists"], P[t]["edit_lpips"], s=150,
                       marker="*", color="#1f77b4", edgecolor="k",
                       linewidth=0.6, zorder=5)
            ax.annotate(t, (P[t]["fid_dists"], P[t]["edit_lpips"]), fontsize=8,
                        xytext=(6, -12), textcoords="offset points")

    ax.axhline(D["lpips_ceiling"], color="#00b8d4", ls=":", lw=1.2)
    ax.text(0.30, D["lpips_ceiling"], f"LPIPS 飽和值 {D['lpips_ceiling']}",
            fontsize=8.5, color="#0097a7", va="bottom")
    hs, _ = ax.get_legend_handles_labels()
    hs += [plt.Line2D([], [], marker="^", ls="", color="#d62728", mec="k",
                      label="stAdv 稠密流場（grid 512、L-BFGS）"),
           plt.Line2D([], [], marker="P", ls="", color="#9467bd", mec="k",
                      label="位移場硬訓練（Adam、4000 步）"),
           plt.Line2D([], [], marker="*", ls="", color="#1f77b4", mec="k",
                      label="相位族（參照）")]
    ax.legend(handles=hs, fontsize=8.5, frameon=False, loc="upper left")
    ax.set_xlabel("失真　DISTS", fontsize=10)
    ax.set_ylabel("未淨化位移　LPIPS", fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_title("位移場：最佳化 對 全隨機（兩張影像平均）", fontsize=12)
    save(fig, out)


def fig_phase_shift_applied(out: Path, run: str = "ip2p_ig_lowdist/r12_f04",
                            tag: str = "r12_f04"):
    """**實際施加的相位變動**：|θ · g_b · m_ω|，逐頻格與逐區塊各看一次。

    只畫這一個量。閘、價目表、幅度增益、加性項各自是另外的東西，不在這張圖裡
    ——它們會把「相位到底被轉了多少」這件事淹掉。

    θ 由 `__w.pt` 載入，閘由該次執行自己的 `results.csv` 重建（閘是由原圖算
    一次就固定的，不參與最佳化，故重建出來的與訓練時用的是同一個）。
    """
    fig, axes = plt.subplots(len(IMAGES), 2, figsize=(11.6, 5.4 * len(IMAGES)))
    axes = np.atleast_2d(axes)
    for r, img in enumerate(IMAGES):
        d = REP / run
        p = param_from_csv(Path("runs") / run)
        p.reset(load01(d / f"{img}__orig.png"), 0)
        m = p.module
        theta = torch.load(d / f"{img}__phase_gain__w.pt",
                           map_location="cpu")[0].float()
        eff = (theta * m.gate()).abs()[0]          # (L, 32, 17) 弧度
        side = int(round(math.sqrt(m.n_blocks)))

        im = axes[r][0].imshow(shift_half(eff.mean(0).numpy()), cmap="magma")
        cb = fig.colorbar(im, ax=axes[r][0], fraction=0.046, pad=0.03)
        cb.set_label("弧度", fontsize=8.5)
        axes[r][0].set_title("頻率平面（直流在中心、外圈 Nyquist）", fontsize=10.5)

        im = axes[r][1].imshow(eff.mean((1, 2)).reshape(side, side).numpy(),
                               cmap="magma")
        cb = fig.colorbar(im, ax=axes[r][1], fraction=0.046, pad=0.03)
        cb.set_label("弧度", fontsize=8.5)
        axes[r][1].set_title(f"空間分佈（{side}×{side} 個區塊）", fontsize=10.5)

        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
        axes[r][0].set_ylabel(NAME[img], fontsize=12)
    fig.suptitle(f"實際施加的相位變動　|θ · g_b · m_ω|　條件 {tag}"
                 f"（radius {p.radius}）", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, out)


# ── 算子集合與標籤 ──────────────────────────────────────────────────────
#
# 三套協定的算子集合不同，**不可互相並列**：
#   OPS6  現行六算子（runs/ip2p_split_band、ip2p_ig_loss、ip2p_matched_headtohead）
#   OPSJ  JPEG 軸五算子（runs/ip2p_mainline_purify）
#   OPS8  八算子（runs/ip2p_purify_headtohead）
OPS6 = ["identity", "jpeg75", "jpeg30", "blur1", "blur2", "crop_resize0.1"]
OPL6 = ["未淨化", "JPEG 75", "JPEG 30", "模糊 σ=1", "模糊 σ=2", "裁切縮放 10%"]
OPSJ = ["identity", "jpeg90", "jpeg75", "jpeg50", "jpeg30"]
OPLJ = ["未淨化", "JPEG 90", "JPEG 75", "JPEG 50", "JPEG 30"]
OPS8 = ["identity", "jpeg75", "jpeg30", "blur1", "crop_resize0.1",
        "jpeg_then_resize75", "gridpure", "adverse_cleaner"]
OPL8 = ["未淨化", "JPEG 75", "JPEG 30", "模糊 σ=1", "裁切縮放 10%",
        "JPEG75→重取樣", "GrIDPure", "AdverseCleaner"]

# 三組等失真配對，取自 `runs/ip2p_mainline`：本方法與 DCT-Shield 在**同一批**
# 裡跑，同兩張影像、同一組算子、同一個空白地板，所以可以直接並列。
PAIRS_JPEG = [("低失真", "ours_pg_n", "dct_native"),
              ("中失真", "ours_ph_q", "dct_aj50_eps0.22"),
              ("中高失真", "ours_pg_q", "dct_aj85")]
# 八算子那一批的失真在各自的來源批次裡，限縮到兩張影像後的平均。
H2H_DIST = {"ours_nonadd": 0.1197, "ours_add": 0.1383, "dct_e18": 0.0978,
            "dct_e14": 0.0794, "dct_y_e14": 0.0558}
OURS = "#1f77b4"
DCT = "#d62728"


def fig_h2h_curve(D, out: Path):
    """三組等失真配對，橫軸是越來越重的 JPEG 淨化。"""
    M = D["baseline"]["ip2p_mainline"]
    B, pur = M["defence"], M["purify"]
    fig, axes = plt.subplots(1, len(PAIRS_JPEG),
                             figsize=(5.8 * len(PAIRS_JPEG), 5.4), sharey=True)
    for ax, (lab, a, b) in zip(np.atleast_1d(axes), PAIRS_JPEG):
        series = [(f"{t}　DISTS {B[t]['fid_dists']:.4f}　"
                   f"PSNR {B[t]['fid_psnr']:.1f}",
                   [pur[t].get(o, {}).get("net") for o in OPSJ], col, mk)
                  for t, col, mk in ((a, OURS, "o"), (b, DCT, "s"))]
        _lines(ax, series, OPSJ, OPLJ, lab, ylab="")
        ax.set_ylim(-0.04, 0.88)
    np.atleast_1d(axes)[0].set_ylabel("淨增益 = effect(p) - 空白地板",
                                      fontsize=10.5)
    fig.suptitle("等失真頭對頭：本方法（藍）對 DCT-Shield（紅）　"
                 "runs/ip2p_mainline、同一批、兩張影像", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out)


def fig_h2h_ops8(D, out: Path):
    """八個算子（含模糊、裁切、擴散淨化）的頭對頭。"""
    H = D["baseline"]["ip2p_purify_headtohead"]["purify"]
    spec = [("ours_add", OURS, "o"), ("ours_nonadd", "#7fb3e0", "v"),
            ("dct_e18", DCT, "s")]
    series = [(f"{t}　DISTS {H2H_DIST.get(t, float('nan')):.4f}",
               [H[t].get(o, {}).get("net") for o in OPS8], col, mk)
              for t, col, mk in spec if t in H]
    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    _lines(ax, series, OPS8, OPL8,
           "八個算子的頭對頭　runs/ip2p_purify_headtohead、兩張影像")
    ax.text(0.5, -0.14, "本方法的兩點失真比 dct_e18 高 22% / 41%，"
            "不是嚴格等失真；裁切與 JPEG→重取樣兩欄為舊參照",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#666")
    save(fig, out)


def fig_matched_ops6(D, out: Path):
    """新批次：三組等失真配對在**現行六算子協定**上（含模糊與裁切）。"""
    M = D["baseline"]["ip2p_mainline"]
    B = M["defence"]
    pur = D["baseline"].get("ip2p_matched_headtohead", {}).get("purify", {})
    if not pur:
        print("  [跳過] 還沒有 runs/ip2p_matched_headtohead 的抗淨化資料")
        return
    fig, axes = plt.subplots(1, len(PAIRS_JPEG),
                             figsize=(6.3 * len(PAIRS_JPEG), 5.6), sharey=True)
    for ax, (lab, a, b) in zip(np.atleast_1d(axes), PAIRS_JPEG):
        series = []
        for t, col, mk in ((a, OURS, "o"), (b, DCT, "s")):
            if t not in pur:
                continue
            series.append((f"{t}　DISTS {B[t]['fid_dists']:.4f}",
                           [pur[t].get(o, {}).get("net") for o in OPS6],
                           col, mk))
        if not series:
            continue
        _lines(ax, series, OPS6, OPL6, lab, ylab="")
        ax.set_ylim(-0.04, 0.88)
        ax.tick_params(axis="x", labelrotation=20)
    np.atleast_1d(axes)[0].set_ylabel("淨增益 = effect(p) - 空白地板",
                                      fontsize=10.5)
    fig.suptitle("等失真頭對頭，現行六算子協定　runs/ip2p_matched_headtohead",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out)


def fig_warp_purify(D, out: Path):
    W = _flat_warp(D)
    P = _flat_phase(D)
    both = dict(W); both.update(P)
    spec = [("sa_t000", DCT, "^"), ("sa_r04", "#ff9896", "v"),
            ("sa_r08", "#c49c94", "D"), ("ig_f08_eot", OURS, "o")]
    series = [(f"{t}　DISTS {both[t]['fid_dists']:.4f}",
               [both[t]["purify"].get(o, {}).get("net") for o in OPS6], col, mk)
              for t, col, mk in spec if t in both and both[t]["purify"]]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    _lines(ax, series, OPS6, OPL6,
           "位移場（stAdv）對相位族：六個淨化算子上的淨增益")
    ax.text(0.5, -0.13, "藍色是相位族，放在這裡當同協定的參照；"
            "裁切欄為舊參照，與其餘欄不同基準",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#666")
    save(fig, out)
