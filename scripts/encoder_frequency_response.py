"""每個頻帶上「模型看得見多少」對「人眼付多少」的比值。

要回答什麼
────────────────────────────────────────────────────────────────────
知覺加權閘（`freq_weight=jpeg_luma`）把預算按 JPEG 量化表定價，效率因此由
3.3–4.3 拉到 18.1（位移／DISTS，13 張人眼確認服從的影像）。但它的**位移封頂
在 0.50**：半徑由 1.5 推到 12，位移只從 0.2228 走到 0.5034，而效率同時掉回
3.4。純相位那條線在同一批上要 0.63 才換得到 12/13 的擋下率。

假說：VAE 編碼器把 512² 降採樣到 64²，接近 Nyquist 的內容在進 latent 之前
就被丟掉。「人眼看不見」的頻帶**模型也看不見**，於是按 JND 定價等於把預算
推到雙方都看不見的地方——便宜，但買不到東西。

若成立，正確的權重是

    w(omega)  正比於  模型敏感度(omega) / 人眼代價(omega)

而不是人眼代價的倒數本身。本檔把這兩個量分別量出來。

量什麼
────────────────────────────────────────────────────────────────────
逐個徑向頻帶（歸一化半徑等分，角落到 sqrt(2)）：

    grad_energy     防禦損失對輸入的梯度在該帶的 STFT 能量。
                    一階近似下，把單位預算放在該帶能推動損失多少。
    dists_cost      把 theta 固定的隨機相位旋轉**限制在該帶**之後的 DISTS。
                    即該帶的人眼代價。
    latent_move     同一個擾動造成的 ‖E(x') − E(x)‖。實測而非一階近似，
                    故它同時吃到重疊相加的抵銷。
    move_per_dists  latent_move / dists_cost，權重要用的那個比值。

隨機相位而非最佳化的相位：這裡要的是**通道的性質**，不是某個解的性質。
最佳化會把答案繞回「損失喜歡什麼」，那正是 grad_energy 那一欄已經回答的。

**不改任何既有行為。** 這是診斷，輸出一份 CSV。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.suite import MetricSuite  # noqa: E402
from src.residual.texture_rephase import PhaseResidual  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
# 角落頻格的歸一化半徑是 sqrt(2)（fy 與 fx 各到 1）。上界取到那裡，
# 否則最外圈的格子會落在所有帶之外而靜默消失。
R_MAX = math.sqrt(2.0)


def band_edges(n: int) -> List[float]:
    """`n + 1` 個首尾相接的邊界，由 0 到 sqrt(2)。"""
    return [R_MAX * i / n for i in range(n + 1)]


def bands(n: int) -> List[Tuple[float, float]]:
    e = band_edges(n)
    return list(zip(e[:-1], e[1:]))


def radius_grid(block: int, device, dtype) -> torch.Tensor:
    """`(block, block//2+1)` 的歸一化半徑，與 `radial_gate` 同一套座標。"""
    fy = torch.fft.fftfreq(block, device=device, dtype=dtype) * 2.0
    fx = torch.fft.rfftfreq(block, device=device, dtype=dtype) * 2.0
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def band_energy(spec: torch.Tensor, block: int) -> List[float]:
    """逐帶的平方和。帶數由 `spec` 的最後兩維推不出來，故固定用 8 帶的
    邊界；呼叫端要別的帶數就自己傳 `n`。"""
    return band_energy_n(spec, block, 8)


def band_energy_n(spec: torch.Tensor, block: int, n: int) -> List[float]:
    r = radius_grid(block, spec.device, spec.dtype if spec.is_floating_point()
                    else torch.float32)
    out = []
    for lo, hi in bands(n):
        m = (r >= lo) & (r < hi) if hi < R_MAX else (r >= lo) & (r <= hi)
        out.append(float(spec.pow(2)[..., m].sum()))
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--theta", type=float, default=1.0,
                    help="探針的相位半徑。**同一個 theta 用在每一帶**，"
                         "故各帶的 dists_cost 直接可比")
    ap.add_argument("--loss", choices=("latent_norm", "encoder_target"),
                    default="latent_norm")
    ap.add_argument("--target", type=Path, default=Path("data/targets/gray.png"))
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

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

    if args.loss == "encoder_target":
        from src.baselines.encoder_target import make_encoder_target_loss
        y = load_image_tensor(args.target, ip2p.device, size=RESOLUTION)
        loss_fn = make_encoder_target_loss(ip2p, y)
    else:
        def loss_fn(x):
            return ip2p.encode_image(x).flatten(1).norm(dim=1).mean()

    rows = []
    for item in dataset:
        x = load_image_tensor(item["path"], ip2p.device, size=RESOLUTION)

        # 一、損失對輸入的梯度，切塊加窗做 STFT，逐帶算能量。
        xg = x.clone().requires_grad_(True)
        g = torch.autograd.grad(loss_fn(xg), xg)[0].detach()
        probe = PhaseResidual(size=RESOLUTION, block=args.block, r_min=0.0,
                              theta_max=math.pi).to(device=x.device, dtype=x.dtype)
        probe.prepare_gates(x)
        gspec = probe.analyze(g)
        gband = band_energy_n(gspec.abs(), args.block, args.bands)

        with torch.no_grad():
            z0 = ip2p.encode_image(x)

        # 二、把同一個 theta 的隨機相位限制在單一帶上，量代價與 latent 位移。
        for i, (lo, hi) in enumerate(bands(args.bands)):
            m = PhaseResidual(size=RESOLUTION, block=args.block, r_min=lo,
                              r_max=hi if hi < R_MAX else float("inf"),
                              theta_max=math.pi,
                              energy_quantile=0.0).to(device=x.device, dtype=x.dtype)
            m.prepare_gates(x)
            gen = torch.Generator(device="cpu").manual_seed(args.seed + i)
            with torch.no_grad():
                m.theta.copy_((torch.randn(m.theta.shape, generator=gen)
                               * args.theta).clamp(-math.pi, math.pi)
                              .to(device=x.device, dtype=x.dtype))
                x_def = m.pixel_residual(x).clamp(0, 1)
                d = float(suite.pairwise(x, x_def)["dists"])
                move = float((ip2p.encode_image(x_def) - z0).flatten().norm())
            rows.append({
                "image": item["name"],
                "block": args.block,
                "theta": args.theta,
                "loss": args.loss,
                "r_lo": round(lo, 4),
                "r_hi": round(hi, 4),
                "grad_energy": round(gband[i], 8),
                "grad_energy_frac": round(gband[i] / sum(gband), 6),
                "dists_cost": round(d, 6),
                "latent_move": round(move, 4),
                "move_per_dists": round(move / d, 3) if d > 0 else "",
                "active_fraction": round(m.active_fraction(), 4),
            })
            write_csv(args.out, rows)
            print(f"{item['name']:32s} r[{lo:.3f},{hi:.3f})  "
                  f"grad {gband[i]/sum(gband):.3f}  dists {d:.5f}  "
                  f"move {move:.2f}  ratio {move/d if d>0 else 0:.1f}", flush=True)
    print(f"\n表：{args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
