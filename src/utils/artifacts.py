"""產出留存 — spec §8.3。

spec 要求每次執行留存下列全部影像，不得只存最終結果：原圖、防禦生成的每步
去噪中間圖 x̂₀、防禦圖、殘差視覺化（含放大倍率標註）與其奇異值譜圖、各淨化
強度的淨化圖、兩側編輯結果。

**殘差視覺化一律標註放大倍率。** 殘差的量級通常在 1e-2 以下，不放大則整張
近乎全灰、看不出結構；放大而不標註倍率，讀者無從判斷失真的真實強度，兩張
不同倍率的圖會被誤讀為同一尺度。故 `save_residual` 把倍率寫進檔名與圖上。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch


def _to_uint8(x: torch.Tensor):
    import numpy as np

    a = x.detach().clamp(0, 1).cpu()
    if a.dim() == 4:
        a = a[0]
    return (a.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def save_image(x: torch.Tensor, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(x)).save(path)


def save_residual(delta: torch.Tensor, path: Path, gain: Optional[float] = None) -> float:
    """把殘差居中到 0.5 後放大存檔，回傳實際使用的放大倍率。

    `gain=None` 時自動取使最大絕對值恰好填滿 ±0.5 的倍率，即
    gain = 0.5 / max|Δ|，讓結構在任何量級下都看得見。
    """
    from PIL import Image, ImageDraw

    d = delta.detach()
    if d.dim() == 4:
        d = d[0]
    peak = float(d.abs().max())
    if gain is None:
        gain = 0.5 / peak if peak > 0 else 1.0

    vis = (d * gain + 0.5).clamp(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(_to_uint8(vis))
    # 倍率標在圖上，避免圖被單獨取用時失去尺度資訊
    ImageDraw.Draw(img).text(
        (4, 4), f"x{gain:.1f}  max|d|={peak:.5f}", fill=(255, 255, 0)
    )
    img.save(path)
    return gain


def save_spectrum_plot(analysis: Dict, path: Path, title: str = "") -> None:
    """奇異值、相對變化率、累積能量比三合一圖 — spec §8.2，形式取自 LRDM Fig.3。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ch, spec in enumerate(analysis["per_channel"]):
        sv = spec["singular_values"]
        idx = range(1, len(sv) + 1)
        axes[0].semilogy(idx, [max(v, 1e-300) for v in sv], label=f"ch{ch}")
        axes[1].plot(idx, spec["decay_ratio"], label=f"ch{ch}")
        axes[2].plot(idx, spec["cumulative_energy"], label=f"ch{ch}")

    axes[0].set_title("singular values"); axes[0].set_xlabel("index")
    axes[1].set_title(r"decay ratio $\sigma_{i+1}/\sigma_i$"); axes[1].set_xlabel("index")
    axes[2].set_title("cumulative energy"); axes[2].set_xlabel("index")
    axes[2].axhline(0.99, ls="--", lw=0.8, color="gray")
    axes[2].axhline(0.90, ls=":", lw=0.8, color="gray")
    for a in axes:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_history_plot(history: List[Dict], path: Path, title: str = "") -> None:
    """優化曲線。L_def 與 edit_shift 分開畫：hinge 飽和後 L_def 恆為 0，
    只看 L_def 會誤以為優化停滯，實際上偏移可能仍在增加。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, keys, name in (
        (axes[0], ["loss", "L_def", "L_fid"], "loss"),
        (axes[1], ["edit_shift"], "edit shift (LPIPS)"),
        (axes[2], ["fid_psnr"], "PSNR(x_def, x) dB"),
        (axes[3], ["fid_linf"], r"$\|\Delta\|_\infty$"),
    ):
        for k in keys:
            ax.plot(steps, [h[k] for h in history], label=k, lw=1.2)
        ax.set_title(name); ax.set_xlabel("step"); ax.grid(alpha=0.3)
        if len(keys) > 1:
            ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_x0_trace(trace: List[torch.Tensor], sd, out_dir: Path) -> None:
    """去噪每步的 x̂₀ 估計解碼成影像 — spec §8.3 要求的中間圖。

    x̂₀ 是 latent，須經 VAE 解碼才是影像；解碼成本不低，故只在最後一步收集。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for j, z in enumerate(trace):
            save_image(sd.decode_latent(z), out_dir / f"x0_step{j:03d}.png")
