"""紋理重相位方法說明的一維示範圖：DFT 相位旋轉、幅度不變、
低頻相位＝位移、加窗重疊相加。由 `scripts/phase_doc_build.py` 內嵌。

    python scripts/phase_doc_fig1d.py
圖存到 $PHASE_DOC_FIGDIR，預設 runs/_phase_doc_fig。
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
FIGDIR = Path(os.environ.get(
    "PHASE_DOC_FIGDIR",
    Path(__file__).resolve().parent.parent / "runs" / "_phase_doc_fig"))
OUT = FIGDIR
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

N = 32
n = np.arange(N)
# 一個「有結構＋有紋理」的一維訊號：一個低頻大包絡加上兩個高頻分量
rng = np.random.default_rng(7)
x = (1.0 * np.sin(2 * np.pi * 1 * n / N)
     + 0.45 * np.sin(2 * np.pi * 6 * n / N + 0.7)
     + 0.30 * np.cos(2 * np.pi * 11 * n / N - 1.2))

X = np.fft.rfft(x, norm="ortho")
mag, pha = np.abs(X), np.angle(X)
K = len(X)


def rotate(theta_vec):
    return np.fft.irfft(X * np.exp(1j * theta_vec), n=N, norm="ortho")


# ---------------------------------------------------------------- F1
fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.6))
ax[0].plot(n, x, "-o", ms=3, color="#c2410c")
ax[0].set_title("signal  x[n]   (N=32)")
ax[0].set_xlabel("n")
ax[1].stem(np.arange(K), mag, basefmt=" ", linefmt="#2563a8", markerfmt="o")
ax[1].set_title(r"magnitude  |X[k]|")
ax[1].set_xlabel("k")
ax[2].stem(np.arange(K), pha, basefmt=" ", linefmt="#8b3a8b", markerfmt="o")
ax[2].set_title(r"phase  $\angle X[k]$  (rad)")
ax[2].set_xlabel("k")
ax[2].set_ylim(-np.pi, np.pi)
fig.tight_layout()
fig.savefig(OUT / "f1_signal_spectrum.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- F2
thetas = [0.0, 0.5, 1.3, np.pi]
gate = np.zeros(K)
gate[4:] = 1.0                      # 只轉 k>=4，模擬徑向頻率閘
fig, ax = plt.subplots(1, 2, figsize=(10.5, 2.9))
colors = ["#111111", "#0d7377", "#c2410c", "#8b3a8b"]
for t, c in zip(thetas, colors):
    y = rotate(gate * t)
    ax[0].plot(n, y, "-", lw=1.6, color=c, label=fr"$\theta$={t:.2f}")
ax[0].plot(n, x, "--", lw=1.0, color="#999999", label="original")
ax[0].set_title(r"waveform after rotating phase of $k\geq4$ by $\theta$")
ax[0].set_xlabel("n"); ax[0].legend(fontsize=7, ncol=2)

w = 0.16
for i, (t, c) in enumerate(zip(thetas, colors)):
    y = rotate(gate * t)
    ax[1].bar(np.arange(K) + (i - 1.5) * w, np.abs(np.fft.rfft(y, norm="ortho")),
              width=w, color=c, label=fr"$\theta$={t:.2f}")
ax[1].set_title(r"|X[k]| of each waveform — identical")
ax[1].set_xlabel("k"); ax[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "f2_rotate_keeps_magnitude.png", bbox_inches="tight")
plt.close(fig)

err = max(np.abs(np.abs(np.fft.rfft(rotate(gate * t), norm="ortho")) - mag).max()
          for t in thetas)
ident = np.abs(rotate(np.zeros(K)) - x).max()

# ---------------------------------------------------------------- F3
fig, ax = plt.subplots(1, 2, figsize=(10.5, 2.7))
shift = 4
lin = -2 * np.pi * np.arange(K) * shift / N          # 線性相位 = 平移
ax[0].plot(n, x, "--", lw=1.0, color="#999999", label="original")
ax[0].plot(n, rotate(lin), "-", lw=1.8, color="#2563a8",
           label=fr"linear phase $\theta_k=-2\pi k\cdot{shift}/N$")
ax[0].plot(n, np.roll(x, shift), ":", lw=2.2, color="#c2410c",
           label=f"np.roll(x, {shift})")
ax[0].set_title("linear phase across ALL k  =  pure translation")
ax[0].set_xlabel("n"); ax[0].legend(fontsize=7)

only_low = np.zeros(K); only_low[1] = 1.3
only_high = np.zeros(K); only_high[6:] = 1.3
ax[1].plot(n, x, "--", lw=1.0, color="#999999", label="original")
ax[1].plot(n, rotate(only_low), "-", lw=1.8, color="#b91c1c",
           label=r"rotate ONLY k=1 (low) by 1.3")
ax[1].plot(n, rotate(only_high), "-", lw=1.8, color="#0d7377",
           label=r"rotate ONLY k$\geq$6 (high) by 1.3")
ax[1].set_title("low-frequency phase moves the whole shape")
ax[1].set_xlabel("n"); ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "f3_low_freq_is_position.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- F4  加窗重疊相加
L, B, HOP = 64, 16, 8
rng2 = np.random.default_rng(3)
sig = (np.sin(2 * np.pi * np.arange(L) / 21)
       + 0.4 * rng2.standard_normal(L))
win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(B) / B)      # periodic Hann
pad = B // 2
sp = np.pad(sig, pad, mode="reflect")
starts = np.arange(0, len(sp) - B + 1, HOP)


def ola(theta_scale):
    num = np.zeros(len(sp)); den = np.zeros(len(sp))
    for s in starts:
        seg = sp[s:s + B] * win
        S = np.fft.rfft(seg, norm="ortho")
        k = np.arange(len(S))
        g = (k >= 3).astype(float)                # 頻率閘
        S = S * np.exp(1j * g * theta_scale)
        rec = np.fft.irfft(S, n=B, norm="ortho") * win
        num[s:s + B] += rec
        den[s:s + B] += win ** 2
    return (num / np.maximum(den, 1e-8))[pad:pad + L], den


rec0, den = ola(0.0)
rec13, _ = ola(1.3)

fig, ax = plt.subplots(1, 3, figsize=(11.5, 2.7))
for s in starts[:6]:
    ax[0].plot(np.arange(s, s + B) - pad, win, lw=1.2)
ax[0].plot(np.arange(len(sp)) - pad, den, "k-", lw=2.0, label=r"$\sum w^2$")
ax[0].axhline(1.0, color="#999", ls=":", lw=1)
ax[0].set_title(f"Hann windows, block={B}, hop={HOP}")
ax[0].set_xlabel("n"); ax[0].legend(fontsize=7)

ax[1].plot(sig, "--", lw=1.0, color="#999999", label="original")
ax[1].plot(rec0, "-", lw=1.6, color="#0d7377", label=r"OLA, $\theta$=0")
ax[1].set_title(fr"$\theta$=0 reconstruction, max err {np.abs(rec0-sig).max():.2e}")
ax[1].set_xlabel("n"); ax[1].legend(fontsize=7)

ax[2].plot(sig, "--", lw=1.0, color="#999999", label="original")
ax[2].plot(rec13, "-", lw=1.6, color="#c2410c", label=r"OLA, $\theta$=1.3 on k$\geq$3")
ax[2].set_title("phase rotated: texture reshuffled, envelope kept")
ax[2].set_xlabel("n"); ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "f4_windowed_ola.png", bbox_inches="tight")
plt.close(fig)

# 一致性偏差：逐塊轉相位之後，再分析一次，幅度已經不一樣了
def block_mags(s):
    sp2 = np.pad(s, pad, mode="reflect")
    return np.array([np.abs(np.fft.rfft(sp2[t:t + B] * win, norm="ortho"))
                     for t in starts])


m0, m13 = block_mags(sig), block_mags(rec13)
amp_dev = np.linalg.norm(m0 - m13) / np.linalg.norm(m0)

print(f"F1-F4 done")
print(f"  θ=0 恆等誤差（無窗、單塊）: {ident:.3e}")
print(f"  轉相位後 |X[k]| 最大變化（單塊）: {err:.3e}")
print(f"  加窗重疊 θ=0 重建最大誤差: {np.abs(rec0-sig).max():.3e}")
print(f"  加窗重疊 θ=1.3 的 amp_dev: {amp_dev:.4f}")

# 給文中引用的數值表
np.set_printoptions(precision=3, suppress=True)
tbl = OUT / "f1d_numbers.txt"
tbl.write_text(
    "k\t|X[k]|\tangle\n" +
    "\n".join(f"{k}\t{mag[k]:.4f}\t{pha[k]:+.4f}" for k in range(K)) +
    f"\n\nidentity_err\t{ident:.3e}\nmag_change\t{err:.3e}\n"
    f"ola_identity_err\t{np.abs(rec0-sig).max():.3e}\namp_dev_1d\t{amp_dev:.4f}\n",
    encoding="utf-8")
