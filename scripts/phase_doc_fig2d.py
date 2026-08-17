"""二維示範圖：全域 RPN、區塊網格與 Hann 窗、rfft2 半平面與徑向閘、
紋理閘、theta 掃描與殘差、amp_dev 曲線。全部用專案自己的算子跑，CPU 即可。"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.residual.texture_rephase import (  # noqa: E402
    PhaseResidual, hann2d, radial_gate, texture_gate, block_mean,
)

import os
FIGDIR = Path(os.environ.get(
    "PHASE_DOC_FIGDIR",
    Path(__file__).resolve().parent.parent / "runs" / "_phase_doc_fig"))
OUT = FIGDIR
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
DATA = Path(__file__).resolve().parent.parent / "data" / "set0817"
torch.manual_seed(0)


def load(name, size=512):
    im = Image.open(DATA / name.rsplit("_", 1)[0] / f"{name}.png").convert("RGB")
    im = im.resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)[None]


def to_np(t):
    return t[0].permute(1, 2, 0).clamp(0, 1).numpy()


def psnr(a, b):
    return float(10 * torch.log10(1.0 / ((a - b) ** 2).mean()))


x = load("raccoon_00")
xa = load("person_b_00")

# ---------------------------------------------------------------- F5 全域 RPN
def global_rpn(img, seed=0):
    """Galerne 的原生 RPN：整張圖一次 DFT，加上共軛對稱的隨機相位，三通道共用。"""
    g = torch.Generator().manual_seed(seed)
    H, W = img.shape[-2:]
    # 在完整複數平面上造對稱相位，保證輸出為實數
    ph = torch.rand(H, W, generator=g) * 2 * math.pi - math.pi
    ph = 0.5 * (ph - torch.flip(ph, dims=[0, 1]).roll((1, 1), dims=(0, 1)))
    F_ = torch.fft.fft2(img, norm="ortho")
    out = torch.fft.ifft2(F_ * torch.exp(1j * ph)[None, None], norm="ortho").real
    return out.clamp(0, 1)


# 一塊純紋理（草地）與一塊有結構的（浣熊臉）
grass = x[..., 380:512, 0:132]
face = x[..., 60:192, 150:282]
fig, ax = plt.subplots(2, 3, figsize=(7.6, 5.4))
for r, (patch, tag) in enumerate([(grass, "micro-texture (grass)"),
                                  (face, "structured (raccoon face)")]):
    ax[r, 0].imshow(to_np(patch)); ax[r, 0].set_title(f"{tag}\noriginal", fontsize=8)
    ax[r, 1].imshow(to_np(global_rpn(patch, 1)))
    ax[r, 1].set_title("global RPN, seed 1", fontsize=8)
    ax[r, 2].imshow(to_np(global_rpn(patch, 2)))
    ax[r, 2].set_title("global RPN, seed 2", fontsize=8)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Galerne et al. RPN: same Fourier modulus, random phase", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "f5_global_rpn.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- F6 窗與網格
B = 32
w = hann2d(B, "cpu", torch.float32).numpy()
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
im0 = ax[0].imshow(w, cmap="viridis"); ax[0].set_title("2-D periodic Hann  w  (32x32)")
plt.colorbar(im0, ax=ax[0], fraction=0.046)
ax[1].plot(w[B // 2], color="#c2410c", lw=1.8)
for off in (-16, 16):
    ax[1].plot(np.arange(B) + off, w[B // 2], lw=1.0, ls="--", color="#2563a8")
ax[1].set_title("a slice, and the two neighbours at hop=16")
ax[1].set_xlim(-16, 47); ax[1].grid(alpha=.25)
# sum of w^2 over the overlap-add grid
acc = np.zeros(B * 4)
for s in range(0, B * 3 + 1, 16):
    acc[s:s + B] += w[B // 2] ** 2
ax[2].plot(acc, color="#0d7377", lw=1.8)
ax[2].set_title(r"$\sum w^2$ on the OLA grid  (NOLA: > 0 everywhere)")
ax[2].set_ylim(0, 1.4); ax[2].grid(alpha=.25)
fig.tight_layout()
fig.savefig(OUT / "f6_window.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- F7 rfft2 與徑向閘
for rmin, fname in ((0.12, "f7_radial_gate.png"),):
    m = radial_gate(B, rmin, "cpu", torch.float32).numpy()
    fy = np.fft.fftfreq(B) * 2
    fx = np.fft.rfftfreq(B) * 2
    R = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    i0 = ax[0].imshow(np.fft.fftshift(R, axes=0), cmap="magma", aspect="auto")
    ax[0].set_title("normalised radius  r  on the rfft2 half-plane\n(32 x 17)", fontsize=8)
    plt.colorbar(i0, ax=ax[0], fraction=0.046)
    i1 = ax[1].imshow(np.fft.fftshift(m, axes=0), cmap="gray", aspect="auto")
    ax[1].set_title(fr"radial gate $m_\omega$  ($r_{{min}}$={rmin})" "\n"
                    "black = frozen", fontsize=8)
    plt.colorbar(i1, ax=ax[1], fraction=0.046)
    for a in ax[:2]:
        a.set_xlabel("fx index 0..16"); a.set_ylabel("fy index (shifted)")
    ax[2].axis("off")
    ax[2].text(0.0, 0.95, "why the two end columns are frozen", fontsize=9, va="top")
    ax[2].text(0.0, 0.80,
               "rfft2 stores only fx = 0 .. N/2.\n"
               "The columns fx = 0 and fx = N/2 must be\n"
               "conjugate-symmetric in fy by themselves,\n"
               "because their mirror is inside the stored\n"
               "half-plane. Rotating each bin there\n"
               "independently breaks that relation: the\n"
               "output is still real, but the magnitude of\n"
               "those two columns is no longer preserved.\n\n"
               "Cost: 2 of 17 columns, ~12% of the bins.",
               fontsize=7.5, va="top", family="monospace")
    frac = 1 - m.mean()
    ax[2].text(0.0, 0.16, f"frozen fraction of the {B}x{B//2+1} grid: {frac*100:.1f}%",
               fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- F8 紋理閘
def gate_maps(img, block=32, hop=16):
    lum = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2])[:, None]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8
    ky = kx.transpose(-1, -2)
    import torch.nn.functional as Fn
    gx = Fn.conv2d(Fn.pad(lum, (1, 1, 1, 1), mode="reflect"), kx)
    gy = Fn.conv2d(Fn.pad(lum, (1, 1, 1, 1), mode="reflect"), ky)
    jxx, jxy, jyy = (block_mean(t, block, hop) for t in (gx * gx, gx * gy, gy * gy))
    tr = jxx + jyy
    disc = torch.sqrt(torch.clamp(((jxx - jyy) * .5) ** 2 + jxy ** 2, min=0))
    coh = (2 * disc) / (tr + 1e-8)
    ref = torch.quantile(tr, 0.5, dim=1, keepdim=True).clamp_min(1e-8)
    g = (1 - coh ** 2) * torch.clamp(tr / ref, 0, 1)
    side = int(math.sqrt(tr.shape[1]))
    rs = lambda t: t.view(side, side).numpy()
    return rs(coh), rs(torch.clamp(tr / ref, 0, 1)), rs(g)


coh, en, g = gate_maps(x)
fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.1))
ax[0].imshow(to_np(x)); ax[0].set_title("original", fontsize=8)
for a, (m, t, cm) in zip(ax[1:], [(coh, "coherence  (edge = 1)", "magma"),
                                  (en, "gradient energy, clipped", "magma"),
                                  (g, r"texture gate  $g_b$", "viridis")]):
    i = a.imshow(m, cmap=cm); a.set_title(t, fontsize=8)
    plt.colorbar(i, ax=a, fraction=0.046)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(r"$g_b=(1-\mathrm{coh}^2)\cdot\mathrm{clip}(E/E_{ref})$"
             "   — edges and flat areas are frozen", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "f8_texture_gate.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- F9/F10 theta 掃描
def run(img, theta, seed=0, gl=0):
    mod = PhaseResidual(size=img.shape[-1], block=32, r_min=0.12,
                        theta_max=math.pi, init_std=1.0, seed=seed, gl_iters=gl)
    with torch.no_grad():
        mod.theta.copy_(torch.sign(mod.theta) * theta)   # 全部拉到 |theta| 相同
        mod.prepare_gates(img)
        y = mod.pixel_residual(img)
        dev = mod.amplitude_deviation(img)
    return y.detach(), dev, float(mod.active_fraction())


rows = []
sweep = [0.3, 0.65, 1.30, 2.20, math.pi]
outs = {}
for t in sweep:
    y, dev, af = run(x, t)
    outs[t] = y
    rows.append((t, psnr(x, y), float((y - x).abs().max()), dev))
    print(f"theta={t:.2f}  PSNR={rows[-1][1]:.2f}  Linf={rows[-1][2]:.4f}  amp_dev={dev:.4f}")

fig, ax = plt.subplots(2, len(sweep) + 1, figsize=(2.05 * (len(sweep) + 1), 4.5))
ax[0, 0].imshow(to_np(x)); ax[0, 0].set_title("original", fontsize=8)
ax[1, 0].axis("off")
for j, t in enumerate(sweep):
    y = outs[t]
    ax[0, j + 1].imshow(to_np(y))
    ax[0, j + 1].set_title(fr"$\theta$={t:.2f}" "\n"
                           f"PSNR {rows[j][1]:.1f}", fontsize=8)
    r = (y - x)[0].abs().mean(0).numpy()
    ax[1, j + 1].imshow(r * 8, cmap="inferno", vmin=0, vmax=1)
    ax[1, j + 1].set_title(fr"|residual| $\times$8", fontsize=7)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("texture rephasing on raccoon_00 — top: output, bottom: residual", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "f9_theta_sweep.png", bbox_inches="tight")
plt.close(fig)

# 放大一塊看紋理被重排
crop = (slice(380, 476), slice(20, 116))
fig, ax = plt.subplots(1, 4, figsize=(10.0, 2.7))
ax[0].imshow(to_np(x)[crop]); ax[0].set_title("original crop 96x96", fontsize=8)
for a, t in zip(ax[1:], [0.65, 1.30, math.pi]):
    a.imshow(to_np(outs[t])[crop]); a.set_title(fr"$\theta$={t:.2f}", fontsize=8)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "f9b_crop.png", bbox_inches="tight")
plt.close(fig)

th = np.array([r[0] for r in rows])
fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.7))
ax[0].plot(th, [r[1] for r in rows], "-o", color="#2563a8"); ax[0].set_title("PSNR (dB)")
ax[1].plot(th, [r[2] for r in rows], "-o", color="#c2410c"); ax[1].set_title(r"$L_\infty$ in [0,1]")
ax[2].plot(th, [r[3] for r in rows], "-o", color="#8b3a8b"); ax[2].set_title(r"amplitude deviation")
for a in ax:
    a.set_xlabel(r"$\theta$"); a.grid(alpha=.25)
fig.tight_layout()
fig.savefig(OUT / "f10_theta_curves.png", bbox_inches="tight")
plt.close(fig)

# theta = 0 的恆等
y0, dev0, af0 = run(x, 0.0)
print(f"theta=0 identity: max|y-x| = {float((y0-x).abs().max()):.3e}   amp_dev={dev0:.3e}")
print(f"active_fraction (texture gate mean) = {af0:.4f}")

# Griffin-Lim 投影
for gl in (0, 1, 4):
    y, dev, _ = run(x, 1.30, gl=gl)
    print(f"gl_iters={gl}: amp_dev={dev:.4f}  PSNR={psnr(x,y):.2f}")

(OUT / "f2d_numbers.txt").write_text(
    "theta\tPSNR\tLinf\tamp_dev\n" +
    "\n".join(f"{r[0]:.4f}\t{r[1]:.3f}\t{r[2]:.5f}\t{r[3]:.5f}" for r in rows) +
    f"\nidentity_max_abs\t{float((y0-x).abs().max()):.3e}\n"
    f"active_fraction\t{af0:.4f}\n", encoding="utf-8")
print("2-D figures done")

# ---------------------------------------------------------------- F11/F12 真實資料
# runs/phaseA_human：24 張、三條件、同一個損失與步數，唯一變因是參數化。
R = Path(__file__).resolve().parent.parent / "runs" / "phaseA_human"
IM = "cat_00"


def rd(p):
    return np.asarray(Image.open(R / p).convert("RGB"), np.float32) / 255


orig = rd(f"{IM}__orig.png")
conds = [("add__human", r"additive $\delta$,  $\epsilon$=1.2/255"),
         ("phase__human", r"texture rephasing,  $\theta$=1.30"),
         ("phase_rand__human", r"random phase,  $\theta$=1.30")]

fig, ax = plt.subplots(3, len(conds) + 1, figsize=(2.35 * (len(conds) + 1), 7.0))
ax[0, 0].imshow(orig); ax[0, 0].set_title("original", fontsize=8)
ax[1, 0].axis("off")
ax[2, 0].imshow(rd(f"{IM}__{conds[0][0]}__edit_orig.png"))
ax[2, 0].set_title("undefended edit", fontsize=8)
for j, (tag, lab) in enumerate(conds):
    dfd = rd(f"{IM}__{tag}__def.png")
    ax[0, j + 1].imshow(dfd)
    ax[0, j + 1].set_title(f"{lab}\ndefended image", fontsize=8)
    ax[1, j + 1].imshow(np.abs(dfd - orig).mean(-1) * 10, cmap="inferno", vmin=0, vmax=1)
    ax[1, j + 1].set_title(r"|residual| $\times$10", fontsize=8)
    ax[2, j + 1].imshow(rd(f"{IM}__{tag}__edit_def.png"))
    ax[2, j + 1].set_title("edit of the defended image", fontsize=8)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"runs/phaseA_human · {IM} · same PGD loss, same steps, "
             "only the parameterisation differs", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "f11_real_pipeline.png", bbox_inches="tight")
plt.close(fig)

# 殘差的徑向功率譜：加性住高頻，相位分布較寬
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
for tag, lab, c in [("add__human", "additive", "#2f6b3a"),
                    ("phase__human", "rephasing", "#c2410c")]:
    r = (rd(f"{IM}__{tag}__def.png") - orig).mean(-1)
    Fm = np.abs(np.fft.fftshift(np.fft.fft2(r)))
    nn = Fm.shape[0]
    yy, xx = np.mgrid[0:nn, 0:nn]
    rad = np.sqrt((yy - nn / 2) ** 2 + (xx - nn / 2) ** 2).astype(int)
    prof = np.bincount(rad.ravel(), Fm.ravel()) / np.maximum(np.bincount(rad.ravel()), 1)
    f = np.arange(len(prof)) / (nn / 2)
    m = f <= 1.0
    ax[0].semilogy(f[m], prof[m] / prof[m].max(), color=c, lw=1.8, label=lab)
    ax[1].plot(f[m], np.cumsum(prof[m] * f[m]) / np.sum(prof[m] * f[m]),
               color=c, lw=1.8, label=lab)
ax[0].set_title("radial power profile of the residual (normalised)")
ax[1].set_title("cumulative energy vs normalised frequency")
for a in ax:
    a.set_xlabel("normalised radial frequency"); a.grid(alpha=.25); a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "f12_residual_spectrum.png", bbox_inches="tight")
plt.close(fig)
print("real-data figures done")

# ---------------------------------------------------------------- F7b 三個 r_min
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
for a, rmin in zip(ax, (0.12, 0.25, 0.40)):
    m = radial_gate(B, rmin, "cpu", torch.float32).numpy()
    a.imshow(np.fft.fftshift(m, axes=0), cmap="gray", aspect="auto")
    kmin = int(np.ceil(rmin * (B // 2)))
    per = 1.0 / (rmin * 0.5)
    a.set_title(fr"$r_{{min}}$={rmin}   live {m.mean()*100:.0f}%"
                "\n" f"lowest rotated bin k={kmin}, period {per:.1f} px",
                fontsize=8)
    a.set_xlabel("fx index 0..16")
    a.set_ylabel("fy index (shifted)")
fig.suptitle("radial gate at three settings — white is rotated, black is frozen",
             fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "f7b_rmin_compare.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- F7c 亮度
rng = np.random.default_rng(0)
blk = rng.random((32, 32)) * 0.5 + 0.25
S = np.fft.rfft2(blk, norm="ortho")
ths = np.linspace(0, np.pi, 60)
mean_dc, mean_rest = [], []
for t in ths:
    S2 = S.copy()
    S2[0, 0] *= np.exp(1j * t)
    mean_dc.append(np.fft.irfft2(S2, s=(32, 32), norm="ortho").mean())
    S3 = S.copy()
    S3[1:, 1:] *= np.exp(1j * t)
    mean_rest.append(np.fft.irfft2(S3, s=(32, 32), norm="ortho").mean())

NOTE = "\n".join([
    "DC is the only bin with no oscillation.",
    "For a real image it must stay real, so",
    "irfft keeps only the real part after the",
    "rotation:   DC -> DC * cos(theta).",
    "",
    "theta = 1.30   ->  brightness x 0.267",
    "theta = pi     ->  brightness x -1",
    "",
    "The radial gate freezes it twice over:",
    "  r(0,0) = 0 < r_min       -> m = 0",
    "  the whole fx = 0 column  -> m = 0",
    "",
    "With DC frozen the block mean is",
    "preserved to 2.2e-16.",
])
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
ax[0].plot(ths, np.array(mean_dc) / blk.mean(), lw=2, color="#b91c1c",
           label="rotate the DC bin too")
ax[0].plot(ths, np.cos(ths), "--", lw=1.2, color="#111",
           label=r"$\cos\theta$")
ax[0].plot(ths, np.array(mean_rest) / blk.mean(), lw=2, color="#0d7377",
           label="DC frozen (what the gate does)")
ax[0].set_xlabel(r"$\theta$")
ax[0].set_ylabel("block mean / original mean")
ax[0].legend(fontsize=7.5)
ax[0].grid(alpha=.25)
ax[0].set_title("rotating DC scales brightness by exactly cos(theta)", fontsize=8.5)
ax[1].axis("off")
ax[1].text(0.0, 0.98, NOTE, fontsize=8, va="top", family="monospace")
fig.tight_layout()
fig.savefig(OUT / "f7c_brightness.png", bbox_inches="tight")
plt.close(fig)
print("gate/brightness figures done")
