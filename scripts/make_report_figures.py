"""產生實驗報告用的分析圖。圖內文字一律英文（matplotlib 預設字型無 CJK）。

輸出到 docs/figures/。每張圖對應報告裡的一個結論，不畫沒有結論的圖。
"""

import csv
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

C = {"P": "#2b6cb0", "PF": "#c53030", "L": "#2f855a", "LA": "#975a16",
     "W": "#6b46c1", "E": "#b83280"}


def rows(path, heldout=True):
    if not os.path.exists(path):
        return []
    r = list(csv.DictReader(open(path, encoding="utf-8")))
    return [x for x in r if x.get("noise_split") == "heldout"] if heldout else r


def agg(path, site, rank=None, col="net_lpips"):
    """{(purify, strength): (mean, stdev, n)}，對影像取平均。"""
    acc = defaultdict(list)
    for r in rows(path):
        if r["site"] != site or (rank and r["rank"] != rank):
            continue
        v = r.get(col, "")
        if v not in ("", "nan"):
            acc[(r["purify"], float(r["strength"]))].append(float(v))

    return {k: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))
            for k, v in acc.items()}


# ---------------------------------------------------------------- 圖 1
def fig_purify_survival():
    """淨化強度掃描：唯一在運作的位置也在中等淨化下失效。"""
    fams = [("blur", "Gaussian blur sigma"), ("noise", "Gaussian noise sigma"),
            ("jpeg", "JPEG quality"), ("quantize", "Quantization levels")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    arms = [("P", "1", "pixel r=1"), ("P", "4", "pixel r=4"),
            ("P", "16", "pixel r=16"), ("L", "16", "latent r=16 (phi has no effect)")]
    for ax, (fam, xlabel) in zip(axes, fams):
        for site, rank, label in arms:
            a = agg("runs/e2/results.csv", site, rank)
            pts = sorted((k[1], v[0]) for k in a if k[0] == fam for v in [a[k]])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ls = "--" if site == "L" else "-"
            ax.plot(xs, ys, "o" + ls, ms=3.5, lw=1.3, color=C[site],
                    alpha=0.55 if rank != "16" else 1.0,
                    label=label if fam == "blur" else None)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel(xlabel)
        if fam in ("jpeg", "quantize"):
            ax.invert_xaxis()
        ax.set_title(fam)
    axes[0].set_ylabel("net edit shift (LPIPS)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Purification sweep, 6 images, held-out noise seed "
                 "(net = defended minus undefended control)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_purify_survival.png", dpi=150, bbox_inches="tight")
    print("fig1")


# ---------------------------------------------------------------- 圖 2
def fig_rank_is_a_budget():
    """秩只是能量預算：偏移與失真同步上升，兩者比值幾乎不變。"""
    s = list(csv.DictReader(open("runs/e2/summary.csv", encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for site in ("P", "L"):
        pts = []
        for rk in ("1", "4", "16"):
            sel = [r for r in s if r["site"] == site and r["rank"] == rk]
            if not sel:
                continue
            a = agg("runs/e2/results.csv", site, rk)
            shift = a.get(("blur", 0.0), (float("nan"),))[0]
            pts.append((int(rk),
                        st.mean([float(r["final_lpips"]) for r in sel]),
                        st.mean([float(r["final_psnr_total"]) for r in sel]),
                        shift))
        if not pts:
            continue
        r_, lp, ps, sh = zip(*pts)
        axes[0].plot(lp, sh, "o-", color=C[site], label=f"site {site}")
        for i, rr in enumerate(r_):
            axes[0].annotate(f"r={rr}", (lp[i], sh[i]), fontsize=7,
                             xytext=(4, -8), textcoords="offset points")
        axes[1].plot(r_, [s_ / l for s_, l in zip(sh, lp)], "o-", color=C[site],
                     label=f"site {site}")
    axes[0].set_xlabel("perceptual distortion of x_def (LPIPS)")
    axes[0].set_ylabel("net edit shift, no purification")
    axes[0].set_title("more rank buys shift by spending fidelity")
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("rank r")
    axes[1].set_ylabel("shift per unit distortion")
    axes[1].set_title("the exchange rate barely moves")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_rank_is_a_budget.png", dpi=150, bbox_inches="tight")
    print("fig2")


# ---------------------------------------------------------------- 圖 3
def fig_fullrank_unconstrained():
    """E8 失敗的原因：全秩那組從未被 tau 拘束住。"""
    taus = [0.02, 0.05, 0.10]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for site, label in (("P", "low rank r=16"), ("PF", "full rank")):
        ach, linf = [], []
        for t in taus:
            p = f"runs/e8_rank_tau{t:.2f}/summary.csv"
            if not os.path.exists(p):
                ach.append(float("nan")); linf.append(float("nan")); continue
            sel = [r for r in csv.DictReader(open(p, encoding="utf-8"))
                   if r["site"] == site]
            ach.append(st.mean([float(r["final_lpips"]) for r in sel]))
            linf.append(st.mean([float(r["final_linf_total"]) for r in sel]))
        axes[0].plot(taus, ach, "o-", color=C[site], label=label)
        axes[1].plot(taus, linf, "o-", color=C[site], label=label)
    axes[0].plot(taus, taus, "k:", lw=1, label="budget (target)")
    axes[0].set_xlabel("LPIPS budget tau"); axes[0].set_ylabel("achieved LPIPS")
    axes[0].set_title("low rank tracks the budget; full rank ignores it")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("LPIPS budget tau"); axes[1].set_ylabel("achieved L-inf")
    axes[1].set_title("full-rank L-inf is flat: step-budget limited,\n"
                      "not constraint limited")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_fullrank_unconstrained.png", dpi=150,
                bbox_inches="tight")
    print("fig3")


# ---------------------------------------------------------------- 圖 4
def fig_align_capacity():
    """階段一容量測試：發散主導，且劣化幅度隨參數量遞增。"""
    runs = [("runs/e9_align_probe", "L", "r16", 163840,
             "latent eps, lr .008 (no PSNR term)", C["L"], ":"),
            ("runs/e12_align_L_fixed", "L", "r16", 163840,
             "latent eps, lr .008", C["L"], "-"),
            ("runs/e12_align_W_r4", "W", "r4", 397824,
             "weight LoRA r=4, lr .001", C["W"], "--"),
            ("runs/e12_align_W_r16", "W", "r16", 1591296,
             "weight LoRA r=16, lr .001", C["W"], "-")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    degr = []
    for d, site, rk, npar, label, col, ls in runs:
        f = f"{d}/car_00__{site}__{rk}/align_history.json"
        if not os.path.exists(f):
            continue
        lp = [x["fid_lpips"] for x in json.load(open(f, encoding="utf-8"))]
        axes[0].plot(lp, ls, color=col, lw=1.4, label=label)
        b = min(lp)
        axes[0].plot([lp.index(b)], [b], "o", color=col, ms=5,
                     markerfacecolor="white", markeredgewidth=1.4)
        degr.append((npar, lp[-1] - b, label, col))
    axes[0].axhline(0.063, color="#c53030", lw=1.2, ls="-.",
                    label="pixel site operating point (0.063)")
    axes[0].set_xlabel("alignment step")
    axes[0].set_ylabel("LPIPS(G(x;phi), x)")
    axes[0].set_title("car_00: every run ends worse than its own best\n"
                      "(circles mark the best iterate)")
    axes[0].legend(fontsize=6.5, loc="upper right")

    for npar, dg, label, col in degr:
        axes[1].plot([npar], [dg], "o", color=col, ms=9)
        axes[1].annotate(label.split(",")[0], (npar, dg), fontsize=6.5,
                         xytext=(6, 3), textcoords="offset points")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("trainable parameters")
    axes[1].set_ylabel("LPIPS lost after the best iterate")
    axes[1].set_title("degradation grows with parameter count:\n"
                      "an optimization signature, not a capacity limit")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_align_capacity.png", dpi=150, bbox_inches="tight")
    print("fig4")


# ---------------------------------------------------------------- 圖 5
def fig_eot_and_overfit():
    """左：EOT 改梯度平均的效果。右：對訓練噪聲的過擬合幅度。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    conds = [("blur", 0.5), ("blur", 1.0), ("noise", 0.02), ("jpeg", 75.0)]
    labels = ["blur 0.5", "blur 1.0*", "noise 0.02", "jpeg 75*"]
    for path, label, col in (("runs/e8_rank_tau0.05/results.csv",
                              "rotate one operator/step", "#718096"),
                             ("runs/e10_eot_all/results.csv",
                              "average over all operators", C["P"])):
        a = agg(path, "P", "16")
        base = a.get(("blur", 0.0), (1.0,))[0]
        ys = [a[c][0] / base * 100 if c in a else float("nan") for c in conds]
        axes[0].plot(range(len(conds)), ys, "o-", color=col, label=label)
    axes[0].set_xticks(range(len(conds))); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("surviving fraction of the\nunpurified shift (%)")
    axes[0].set_title("EOT: * marks operators in the training set")
    axes[0].legend(fontsize=8)

    tr, ho, names = [], [], []
    for site in ("P", "L"):
        for rk in ("1", "4", "16"):
            a_ho = agg("runs/e2/results.csv", site, rk)
            a_tr = agg("runs/e2/results.csv", site, rk)
            trn = [float(r["net_lpips"]) for r in rows("runs/e2/results.csv", False)
                   if r["site"] == site and r["rank"] == rk
                   and r["noise_split"] == "train"]
            if not trn or ("blur", 0.0) not in a_ho:
                continue
            tr.append(st.mean(trn)); ho.append(a_ho[("blur", 0.0)][0])
            names.append(f"{site} r={rk}")
    axes[1].plot([0, max(tr) * 1.05], [0, max(tr) * 1.05], "k:", lw=1,
                 label="no overfitting")
    for t, h, n in zip(tr, ho, names):
        axes[1].plot([t], [h], "o", ms=8,
                     color=C["P"] if n.startswith("P") else C["L"])
        axes[1].annotate(f"{n}  {t/h:.1f}x", (t, h), fontsize=7,
                         xytext=(5, -3), textcoords="offset points")
    axes[1].set_xlabel("shift on the training noise seed")
    axes[1].set_ylabel("shift on a held-out seed")
    axes[1].set_title("overfitting to one noise sample")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_eot_and_overfit.png", dpi=150, bbox_inches="tight")
    print("fig5")


if __name__ == "__main__":
    for fn in (fig_purify_survival, fig_rank_is_a_budget,
               fig_fullrank_unconstrained, fig_align_capacity,
               fig_eot_and_overfit):
        try:
            fn()
        except Exception as e:      # 缺某個 run 不該讓整批圖產不出來
            print(f"{fn.__name__} 失敗: {type(e).__name__}: {e}")
