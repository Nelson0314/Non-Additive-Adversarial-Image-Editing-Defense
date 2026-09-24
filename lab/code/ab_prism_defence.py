"""Global, triangular Lab colour map with hue-conditioned monotone lightness.

The carrier and constraints import only tensor utilities. Model construction is
confined to main(). See lab/docs/AB_WARP_NEXT.md for the experimental protocol.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import paths

paths.add_source_to_syspath()

import torch

from colour_support import chroma_p95, skin_colour_support
from src.defense.delta_e_torch import delta_e00_torch
from src.defense.immunise import Cap, optimise_carrier, quantise
from src.defense.ncf_param import lab_to_rgb, rgb_to_lab


def bounded(raw: torch.Tensor) -> torch.Tensor:
    """Smooth (-1, 1) coordinates; polynomial, rather than tanh, saturation."""
    return raw / torch.sqrt(1.0 + raw.square())


def bernstein(t: torch.Tensor, degree: int) -> torch.Tensor:
    return torch.stack([math.comb(degree, j) * t.pow(j)
                        * (1.0 - t).pow(degree - j)
                        for j in range(degree + 1)], dim=-1)


class AbPrismParam:
    """112 scalars by default; no spatial inputs, neighbourhoods or masks.

    Stage 1 is monotone in L for fixed (a,b). Stage 2 is an orientation-
    preserving similarity of (a,b) for fixed output L. Their composition is
    invertible in Lab before RGB gamut clipping and PNG quantisation.
    """

    name = "ab_prism"

    def __init__(self, grid=5, extent=90.0, slope_span=1.5,
                 angle_degrees=12.0, chroma_span=1.25, bias_radius=20.0,
                 init_std=0.03):
        if not all(math.isfinite(v) for v in (
                extent, slope_span, angle_degrees, chroma_span, bias_radius, init_std)):
            raise ValueError("Carrier scales must be finite")
        if (grid < 2 or extent <= 0 or slope_span <= 1 or angle_degrees <= 0
                or chroma_span <= 1 or bias_radius <= 0 or init_std <= 0):
            raise ValueError("Invalid carrier scales or identity-only initialisation")
        self.grid, self.extent = int(grid), float(extent)
        self.slope_span, self.chroma_span = float(slope_span), float(chroma_span)
        self.angle = math.radians(angle_degrees)
        self.bias_radius, self.init_std = float(bias_radius), float(init_std)
        axis = torch.linspace(-extent, extent, grid)
        aa, bb = torch.meshgrid(axis, axis, indexing="ij")
        self.anchors = torch.stack((aa.flatten(), bb.flatten()), dim=-1)
        self.sigma = 2.0 * extent / (grid - 1)
        self.amplitude = 1.0
        self.checkpoint_loaded = False

    def reset(self, x01, seed=0):
        self.anchors = self.anchors.to(device=x01.device, dtype=torch.float32)
        gen = torch.Generator(device=x01.device).manual_seed(seed)
        self.tone_raw = (self.init_std * torch.randn(
            (self.grid ** 2, 4), generator=gen, device=x01.device)).requires_grad_()
        self.grade_raw = (self.init_std * torch.randn(
            (3, 4), generator=gen, device=x01.device)).requires_grad_()
        self.checkpoint_loaded = False

    def params(self):
        return [self.tone_raw, self.grade_raw]

    def project(self):
        """Solver interface only: validate; never project or shrink parameters."""
        if not all(torch.isfinite(p).all() for p in self.params()):
            raise FloatingPointError("Non-finite ab_prism parameters")

    def state_dict(self):
        return {"tone_raw": self.tone_raw.detach().clone(),
                "grade_raw": self.grade_raw.detach().clone()}

    def load_state_dict(self, state):
        self.tone_raw = state["tone_raw"].detach().clone().requires_grad_()
        self.grade_raw = state["grade_raw"].detach().clone().requires_grad_()
        self.checkpoint_loaded = True

    @property
    def stages(self):
        return [self]

    def set_amplitude(self, amplitude):
        # optimise_carrier restores amplitude after load_state_dict(best). Its
        # no-checkpoint fit_caps path reaches here BEFORE any checkpoint load.
        if not self.checkpoint_loaded:
            raise RuntimeError("No feasible checkpoint; post-hoc fit_caps is forbidden")
        values = amplitude if isinstance(amplitude, (list, tuple)) else [amplitude]
        if len(values) != 1 or float(values[0]) != 1.0:
            raise ValueError("ab_prism has no post-hoc amplitude scaling")

    def tone_weights(self, ab):
        # Explicit 2-D distances also keep a pixel's computation independent of
        # image shape / cdist's matrix-multiplication size heuristic.
        d2 = (ab[:, None, :] - self.anchors[None, :, :]).square().sum(-1)
        return torch.softmax(-d2 / (2.0 * self.sigma ** 2), dim=-1)

    def lightness(self, l01, ab):
        slopes = torch.exp(math.log(self.slope_span) * bounded(self.tone_raw))
        slopes = torch.cat((slopes, torch.ones_like(slopes[:, :1])), dim=-1)
        density = self.tone_weights(ab) @ slopes
        # Integral_0^L B_{j,4} = sum_{k=j+1}^5 B_{k,5}(L) / 5.
        # The same /5 in the normalising integral cancels exactly.
        basis = bernstein(l01, 5)
        integrals = torch.stack([basis[:, j + 1:].sum(-1) for j in range(5)], -1)
        return (density * integrals).sum(-1) / density.sum(-1)

    def map_lab(self, lab):
        shape = lab.shape
        flat = lab.permute(0, 2, 3, 1).reshape(-1, 3)
        ab = flat[:, 1:]
        light = self.lightness((flat[:, 0] / 100.0).clamp(0, 1), ab)
        basis = bernstein(light, 2)
        angle = self.angle * (basis @ bounded(self.grade_raw[:, 0]))
        scale = torch.exp(math.log(self.chroma_span)
                          * (basis @ bounded(self.grade_raw[:, 1])))
        raw_bias = self.grade_raw[:, 2:]
        bias = self.bias_radius * raw_bias / torch.sqrt(
            1.0 + raw_bias.square().sum(-1, keepdim=True))
        ca, sa = angle.cos(), angle.sin()
        rotated = torch.stack((ca * ab[:, 0] - sa * ab[:, 1],
                               sa * ab[:, 0] + ca * ab[:, 1]), dim=-1)
        moved = scale[:, None] * rotated + basis @ bias
        out = torch.cat((100.0 * light[:, None], moved), dim=-1)
        return out.reshape(shape[0], shape[2], shape[3], 3).permute(0, 3, 1, 2)

    def unclipped_rgb(self, x01):
        return lab_to_rgb(self.map_lab(rgb_to_lab(x01.clamp(0, 1).float())))

    def render(self, x01):
        return self.unclipped_rgb(x01).clamp(0, 1).to(x01.dtype)


def make_caps(x, face, distance, target, tolerance=0.0025,
              frame_cap=16.0, face_cap=8.0, skin_radius=12.0,
              chroma_gain=1.15, lower=True):
    """Six AL inequalities; distance(y) is the frozen input LPIPS in production.

    Face support is used only for measurement / the fixed skin colour centre.
    It never enters the carrier. Hard colour checks use float64 tensor CIEDE2000
    on quantised RGB; main() additionally audits the exported image with skimage.
    """
    if not all(math.isfinite(v) for v in (
            target, tolerance, frame_cap, face_cap, skin_radius, chroma_gain)):
        raise ValueError("Constraint bounds must be finite")
    if (not 0 < tolerance < target or frame_cap <= 0 or face_cap <= 0
            or skin_radius <= 0 or chroma_gain <= 0):
        raise ValueError("All constraints must be active with positive bounds")
    if face.shape != x[:, :1].shape or not torch.isfinite(face).all():
        raise ValueError("Invalid measurement support")
    if not ((face >= 0).all() and (face <= 1).all() and face.sum() >= 4):
        raise ValueError("Empty or invalid face measurement support")
    skin = skin_colour_support(x, face, skin_radius)
    supports = {"frame": torch.ones_like(face), "face_box": face, "skin_colour": skin}
    caps = []
    for name, support in supports.items():
        if float(support.sum()) <= 0:
            raise ValueError(f"Empty constraint support: {name}")
        value = frame_cap if name == "frame" else face_cap
        caps.append(Cap(
            name,
            lambda y, s=support: delta_e00_torch(x, y, s),
            lambda y, s=support: float(delta_e00_torch(x.double(), y.double(), s.double())),
            value))
    c95 = float(chroma_p95(x)) * chroma_gain
    if not math.isfinite(c95) or c95 <= 0:
        raise ValueError("Non-positive chroma constraint")
    caps.extend([
        Cap("chroma_p95", chroma_p95, lambda y: float(chroma_p95(y)), c95),
        Cap("input_lpips_upper", distance, lambda y: float(distance(y)), target + tolerance),
    ])
    # lower=False：只要求輸入 LPIPS 不超過參照臂。實測（2026-09-25 隨機候選
    # 32,768 個）在四道色彩上限內，本族能達到的 LPIPS 在 7/8 張低於 ab_warp
    # 的逐張值，雙邊帶因此不可行。
    if lower:
        caps.append(Cap("input_lpips_lower", lambda y: 2.0 * target - distance(y),
                        lambda y: 2.0 * target - float(distance(y)), target + tolerance))
    return caps, supports


def read_targets(path, arm):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["arm"] == arm]
    targets = {r["image"]: float(r["lpips"]) for r in rows}
    if not rows or len(targets) != len(rows):
        raise ValueError("LPIPS target rows are empty or contain duplicate images")
    if not all(math.isfinite(v) and v > 0 for v in targets.values()):
        raise ValueError("Invalid LPIPS targets")
    return targets


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="ab_prism", choices=["ab_prism"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("lab/data/portraits"))
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--extent", type=float, default=90.0)
    ap.add_argument("--slope-span", type=float, default=1.5)
    ap.add_argument("--angle-degrees", type=float, default=12.0)
    ap.add_argument("--chroma-span", type=float, default=1.25)
    ap.add_argument("--bias-radius", type=float, default=20.0)
    ap.add_argument("--init-std", type=float, default=0.03)
    ap.add_argument("--frame-cap", type=float, default=16.0)
    ap.add_argument("--face-cap", type=float, default=8.0)
    ap.add_argument("--skin-radius", type=float, default=12.0)
    ap.add_argument("--chroma-gain", type=float, default=1.15)
    ap.add_argument("--lpips-targets", type=Path, default=Path("lab/results/fidelity.csv"))
    ap.add_argument("--lpips-reference", default="ab_warp")
    ap.add_argument("--lpips-tolerance", type=float, default=0.0025)
    ap.add_argument("--no-lpips-lower", dest="lpips_lower", action="store_false",
                    help="只保留 LPIPS 上限（不超過參照臂），不要求達到它")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lr-final-ratio", type=float, default=0.2)
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--lam-every", type=int, default=5)
    ap.add_argument("--check-every", type=int, default=10)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--noise-seed", type=int, default=0)
    return ap


def main():
    args = parser().parse_args()
    if min(args.steps, args.lam_every, args.check_every) < 1 or args.lr <= 0:
        raise ValueError("Solver steps, intervals and learning rate must be positive")
    if (not all(math.isfinite(v) for v in (args.lr, args.lr_final_ratio, args.rho))
            or not 0 < args.lr_final_ratio <= 1 or args.rho <= 0
            or min(args.probe_every, args.log_every) < 0):
        raise ValueError("Invalid solver schedule or penalty")
    targets = read_targets(args.lpips_targets, args.lpips_reference)
    # No diffusion / learned metric is imported or constructed by importing the
    # carrier, parsing --help, or running its CPU smoke test.
    import piq
    from curve_budget_defence import box_support, expanded_box, load_images, write_rows
    from src.defense.color_amplitude import delta_e00
    from src.defense.instruction_free import FreeObjective
    from src.metrics.identity import face_boxes
    from src.models.ip2p import IP2PWrapper
    from src.utils.artifacts import save_image
    from src.utils.io import load_image_tensor

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Production optimisation requires an explicitly scheduled GPU run")
    ip2p = IP2PWrapper(dtype=torch.float32)
    lpips = piq.LPIPS().to(device).eval()
    lpips.requires_grad_(False)
    items = load_images(args.data, set(args.images) if args.images else None)
    if not all(item["name"] in targets for item in items):
        raise ValueError("Missing per-image LPIPS target")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        started = time.time()
        x = load_image_tensor(item["path"], device, size=512)
        boxes = face_boxes(x, device)
        if not boxes:
            raise ValueError(f"No face for skin-colour calibration: {item['name']}")
        box = expanded_box(x, max(boxes, key=lambda q: (q[2]-q[0]) * (q[3]-q[1])))
        face = box_support(x, box)
        carrier = AbPrismParam(**{k: getattr(args, k) for k in (
            "grid", "extent", "slope_span", "angle_degrees", "chroma_span",
            "bias_radius", "init_std")})
        carrier.reset(x, args.noise_seed)
        distance = lambda y: lpips(y, x).mean()
        target = targets[item["name"]]
        caps, supports = make_caps(
            x, face, distance, target, args.lpips_tolerance, args.frame_cap,
            args.face_cap, args.skin_radius, args.chroma_gain, lower=args.lpips_lower)
        objective = FreeObjective(
            ip2p, x, box=box, k=4, steps=50, seed=args.noise_seed,
            weights={"id": 1.0, "enc": 0.5, "cond": 1.0}, chain_steps=6,
            grad_steps=1, s_i=1.5, resample=True, face_weight=0.0)
        stats = optimise_carrier(
            carrier, x, objective, caps=caps, restarts=1,
            **{k: getattr(args, k) for k in ("steps", "lr", "lr_final_ratio",
               "rho", "lam_every", "check_every", "probe_every", "log_every")})
        if stats["free_feasible_step"] < 1 or stats["free_amplitude_shrink"] != 1.0:
            raise RuntimeError("No unscaled feasible checkpoint")
        with torch.no_grad():
            y = quantise(carrier.render(x))
            measured = {c.name: c.hard(y) for c in caps}
            if any(not math.isfinite(measured[c.name]) or measured[c.name] > c.value for c in caps):
                raise RuntimeError(f"Quantised constraint violation: {measured}")
            audit = {name: delta_e00(x, y, s) for name, s in supports.items()}
            for name, value in audit.items():
                cap = args.frame_cap if name == "frame" else args.face_cap
                if not math.isfinite(value) or value > cap:
                    raise RuntimeError(f"Independent skimage audit exceeded {name}: {value}")
            unclipped = carrier.unclipped_rgb(x)
            mse = float((y-x).square().mean())
            row = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "image": item["name"], "class": item["class"], "carrier": carrier.name,
                   "lpips_target": target, "lpips": float(distance(y)),
                   "lpips_residual": float(distance(y)) - target,
                   "dof": sum(p.numel() for p in carrier.params()),
                   "raw_abs_max": max(float(p.abs().max()) for p in carrier.params()),
                   "enc_tanh_slope": 1.0-math.tanh(stats["free_term_enc"])**2,
                   "cond_tanh_slope": 1.0-math.tanh(stats["free_term_cond"])**2,
                   "chroma_p95_out": measured["chroma_p95"],
                   "clip_fraction": float(((unclipped < 0) | (unclipped > 1)).float().mean()),
                   "psnr": -10.0 * math.log10(max(mse, 1e-30)),
                   "rms": math.sqrt(mse), "linf": float((y-x).abs().max()),
                   "seconds": time.time()-started,
                   **{f"deltaE00_{k}": v for k, v in audit.items()}, **stats}
        save_image(x, args.out / f"{item['name']}__orig.png")
        save_image(y, args.out / f"{item['name']}__{args.arm}__def.png")
        torch.save({"state": carrier.state_dict(), "settings": vars(args)},
                   args.out / f"{item['name']}__carrier.pt")
        rows.append(row)
        write_rows(args.out / "results.csv", rows)
        print(f"[{args.arm}] {item['name']} LPIPS={row['lpips']:.6f} "
              f"target={target:.6f} feasible_step={stats['free_feasible_step']}", flush=True)


if __name__ == "__main__":
    main()
