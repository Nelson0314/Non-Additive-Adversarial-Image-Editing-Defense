"""產生 紋理重相位架構圖需要的每一張中間影像，供 `hb5_report.py` 使用。

每一張都由 `src/residual/texture_rephase.py` 的真實程式碼算出，**不是示意圖**：
窗、幅度譜、相位譜、兩個閘、旋轉前後的頻譜，都是同一條前向路徑上的實際
張量。一併輸出 `facts.json`，內含三個由構造保證、可被逐次複驗的數值：

    identity_max_abs_err     θ=0 時 max |x_def − x|
    block_mag_max_abs_diff   單區塊 max ‖X·e^{iθ}| − |X‖
    amp_dev_theta130         整圖層級的幅度譜相對偏差（重疊相加後）

用法：
    python scripts/hb5_arch_assets.py --out runs/hb5/arch
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.residual.texture_rephase import (  # noqa: E402
    PhaseResidual, hann2d, radial_gate, rephase_blocks)

# 追蹤用的區塊左上角。取在狗毛上——閘在平坦區與邊緣皆為 0，圖要能看出作用。
TRACK_Y, TRACK_X = 300, 240


def _lut(stops):
    xs = np.linspace(0, 1, len(stops))

    def f(a):
        out = np.zeros(a.shape + (3,), dtype=np.float32)
        for c in range(3):
            out[..., c] = np.interp(a, xs, [s[c] for s in stops])
        return out
    return f


VIRIDIS = _lut([(0.267, 0.005, 0.329), (0.229, 0.322, 0.545),
                (0.128, 0.567, 0.551), (0.369, 0.789, 0.383),
                (0.993, 0.906, 0.144)])
# 相位是週期量，色表兩端必須相同，否則 −π 與 +π 會看起來天差地遠。
TWILIGHT = _lut([(0.89, 0.88, 0.90), (0.32, 0.47, 0.72), (0.13, 0.10, 0.24),
                 (0.70, 0.35, 0.40), (0.89, 0.88, 0.90)])


def load(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def save(out: Path, t: torch.Tensor, name: str, upscale=None, cmap=None) -> None:
    """`t` 為 (1,C,H,W) 或 (H,W)；後者用 `cmap` 上色並各自歸一化。"""
    if t.dim() == 2:
        a = t.detach().cpu().numpy()
        a = (a - a.min()) / (a.max() - a.min() + 1e-12)
        rgb = {"viridis": VIRIDIS, "twilight": TWILIGHT}.get(
            cmap, lambda z: np.repeat(z[..., None], 3, axis=2))(a)
        im = Image.fromarray((rgb * 255).round().astype(np.uint8))
    else:
        a = t.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
        im = Image.fromarray((a * 255).round().astype(np.uint8))
    if upscale:
        im = im.resize((im.width * upscale, im.height * upscale), Image.NEAREST)
    im.save(out / name)


def outline(x: torch.Tensor, y: int, xx: int, n: int, w: int = 2) -> torch.Tensor:
    """在原圖上畫一個紅框標出被追蹤的區塊。"""
    m = x.clone()
    for sl in (np.s_[:, :, y:y + n, xx:xx + w], np.s_[:, :, y:y + n, xx + n - w:xx + n],
               np.s_[:, :, y:y + w, xx:xx + n], np.s_[:, :, y + n - w:y + n, xx:xx + n]):
        m[sl] = torch.tensor([1.0, 0.0, 0.0])[None, :, None, None]
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("runs/hb5/arch"))
    ap.add_argument("--run", type=Path, default=Path("runs/hb5"))
    ap.add_argument("--image", default="dog_03")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--r-min", type=float, default=0.12)
    ap.add_argument("--theta", type=float, default=1.30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    n, out = args.block, args.out
    facts = {}

    x = load(args.run / f"{args.image}__orig.png")
    save(out, x, "01_input.png")
    save(out, outline(x, TRACK_Y, TRACK_X, n), "01b_input_marked.png")

    # ---- 1 · 取區塊，加窗 ----
    P = x[:, :, TRACK_Y:TRACK_Y + n, TRACK_X:TRACK_X + n]
    save(out, P, "02_block.png", upscale=6)
    w = hann2d(n, torch.device("cpu"), torch.float32)
    save(out, w, "03_window.png", upscale=6, cmap="viridis")
    save(out, P * w, "04_windowed.png", upscale=6)

    # ---- 2 · 頻譜 ----
    spec = torch.fft.rfft2(P * w, norm="ortho")
    save(out, torch.log1p(spec.abs().mean(1)[0] * 40), "05_magnitude.png",
         upscale=10, cmap="viridis")
    save(out, torch.angle(spec).mean(1)[0], "06_phase.png", upscale=10,
         cmap="twilight")

    # ---- 3 · 兩個閘 ----
    fg = radial_gate(n, args.r_min, torch.device("cpu"), torch.float32)
    save(out, fg, "07_freq_gate.png", upscale=10, cmap="viridis")
    facts["freq_gate_zeroed_bins"] = int((fg == 0).sum())
    facts["freq_gate_total_bins"] = int(fg.numel())

    mod = PhaseResidual(size=x.shape[-1], block=n, r_min=args.r_min, seed=args.seed)
    mod.prepare_gates(x)
    side = int(math.sqrt(mod.n_blocks))
    save(out, mod.tex_gate.reshape(side, side), "08_tex_gate.png", upscale=6,
         cmap="viridis")
    facts["active_fraction"] = round(float(mod.active_fraction()), 4)
    facts["n_blocks"] = int(mod.n_blocks)
    facts["grid_side"] = side

    # ---- 4 · θ=0 的恆等性 ----
    with torch.no_grad():
        facts["identity_max_abs_err"] = float((mod.pixel_residual(x) - x).abs().max())

    # ---- 5 · θ=θ_max 的示範輸出 ----
    with torch.no_grad():
        mod.theta.fill_(0.0)
        mod.theta.add_(torch.randn_like(mod.theta) * args.theta)
        mod.theta.clamp_(-args.theta, args.theta)
        xr = mod.pixel_residual(x).clamp(0, 1)
        facts["amp_dev_theta130"] = round(float(mod.amplitude_deviation(x)), 5)
    save(out, xr, "09_phase_out_demo.png")
    save(out, ((xr - x) * 8 + 0.5).clamp(0, 1), "10_phase_res_demo.png")
    facts["demo_res_linf"] = round(float((xr - x).abs().max()), 4)

    # ---- 6 · 單區塊的幅度守恆 ----
    rot = rephase_blocks(P * w, torch.full_like(spec.real, args.theta) * fg)
    spec2 = torch.fft.rfft2(rot, norm="ortho")
    facts["block_mag_max_abs_diff"] = float((spec2.abs() - spec.abs()).abs().max())
    save(out, torch.log1p(spec2.abs().mean(1)[0] * 40), "05b_magnitude_after.png",
         upscale=10, cmap="viridis")
    save(out, torch.angle(spec2).mean(1)[0], "06b_phase_after.png", upscale=10,
         cmap="twilight")
    save(out, rot.clamp(0, 1), "04b_windowed_after.png", upscale=6)

    # ---- 7 · 兩個條件的實際訓練結果 ----
    for cond, tag in (("phase", "phase__human"), ("add", "add__human")):
        xd = load(args.run / f"{args.image}__{tag}__def.png")
        r = xd - x
        save(out, xd, f"11_{cond}_def.png")
        save(out, (r * 8 + 0.5).clamp(0, 1), f"12_{cond}_res.png")
        facts[f"{cond}_res_linf"] = round(float(r.abs().max()), 4)
        facts[f"{cond}_res_rms"] = round(float(r.pow(2).mean().sqrt()), 5)

    (out / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
