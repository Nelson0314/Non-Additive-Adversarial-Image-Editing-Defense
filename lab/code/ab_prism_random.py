"""Same-function-class rejection control from AB_WARP_NEXT.md section 6.3.

One invocation creates r1..r5 under --out (default lab/runs/defence). A canonical
ab_prism_random_rN directory is also accepted as --out and resolves to its parent.
Reruns reuse completed image searches, including exhausted searches. Use a new
output root when changing settings. Run invocations sharing a root serially.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from ab_prism_defence import AbPrismParam, make_caps, read_targets
from src.defense.color_amplitude import delta_e00
from src.defense.immunise import quantise
from src.utils.artifacts import save_image


IMAGE_ORDER = tuple(f"{group}_{i:02d}" for group in ("man", "woman") for i in range(4))
STDS = (0.03, 0.1, 0.3, 1.0, 3.0)
COLOUR_CAPS = ("frame", "face_box", "skin_colour", "chroma_p95")
CAP_NAMES = COLOUR_CAPS + ("input_lpips_upper", "input_lpips_lower")
CAP_ARGS = ("frame_cap", "face_cap", "skin_radius", "chroma_gain")
FIELDS = ("image", "image_index", "j", "seed", "std", "raw_params", "dof",
          "raw_abs_max", "lpips_target", "lpips_tolerance", "lpips", "lpips_residual",
          *CAP_NAMES, *(f"{name}_limit" for name in CAP_NAMES),
          *(f"audit_{name}" for name in COLOUR_CAPS[:3]),
          "feasible", "replicate", "rejection_reason")


def sample_raw(image, j):
    """CPU RNG: one uniform mixture draw, then 112 independent normal draws.

    j is zero based. The fixed full-dataset index never depends on --images.
    Sampling on CPU also makes candidate parameters independent of CUDA RNGs.
    """
    if image not in IMAGE_ORDER or not 0 <= j < 4096:
        raise ValueError("Expected a protocol image and candidate index 0..4095")
    seed = 100000 + 10000 * IMAGE_ORDER.index(image) + j
    gen = torch.Generator(device="cpu").manual_seed(seed)
    std = STDS[int(torch.randint(len(STDS), (1,), generator=gen))]
    raw = std * torch.randn(112, generator=gen, dtype=torch.float32)
    return seed, std, raw


def sample_carrier(x, image, j):
    seed, std, raw = sample_raw(image, j)
    carrier = AbPrismParam(init_std=std)
    carrier.anchors = carrier.anchors.to(x.device)
    # Do not call reset(): it would restart the RNG before the mixture draw.
    carrier.tone_raw = raw[:100].reshape(25, 4).to(x.device)
    carrier.grade_raw = raw[100:].reshape(3, 4).to(x.device)
    return carrier, {"image": image, "image_index": IMAGE_ORDER.index(image),
                     "j": j, "seed": seed, "std": std, "dof": raw.numel(),
                     "raw_abs_max": float(raw.abs().max()),
                     "raw_params": json.dumps(raw.tolist(), separators=(",", ":"))}


@torch.no_grad()
def screen_candidate(x, rendered, caps, supports, distance, target, tolerance):
    """Measure EVERY candidate after PNG quantisation, including rejected ones.

    One distance evaluation supplies both band inequalities. All four colour
    caps and the three independent skimage audits are logged without early exit.
    There is no objective, editor, optimisation, projection or amplitude search.
    """
    if tuple(c.name for c in caps) != CAP_NAMES:
        raise ValueError("Expected the six ab_prism constraints in protocol order")
    y = quantise(rendered)
    lpips = float(distance(y))
    measured = {c.name: c.hard(y) for c in caps[:4]}
    measured.update(input_lpips_upper=lpips, input_lpips_lower=2 * target - lpips)
    reasons = [c.name for c in caps
               if not math.isfinite(measured[c.name]) or measured[c.name] > c.value]
    # skimage 複檢在 CPU 上約 0.3 s／張；只對通過六道約束的候選做，
    # 已被拒絕的候選複檢欄記 NaN（拒絕原因已由上面的約束值完整記錄）。
    if reasons:
        audit = {f"audit_{name}": float("nan") for name in COLOUR_CAPS[:3]}
    else:
        audit = {f"audit_{name}": delta_e00(x, y, supports[name])
                 for name in COLOUR_CAPS[:3]}
        reasons += [f"audit_{c.name}" for c in caps[:3]
                    if not math.isfinite(audit[f"audit_{c.name}"])
                    or audit[f"audit_{c.name}"] > c.value]
    if not torch.isfinite(rendered).all():
        reasons.append("nonfinite_render")
    return y, {"lpips_target": target, "lpips_tolerance": tolerance, "lpips": lpips,
               "lpips_residual": lpips - target, **measured, **audit,
               **{f"{c.name}_limit": c.value for c in caps},
               "feasible": int(not reasons), "rejection_reason": ";".join(reasons)}


def candidates(x, face, image, distance, target, *, replicates=5,
               max_candidates=4096, tolerance=0.0025, **colour_bounds):
    if not 1 <= replicates <= 5 or not 1 <= max_candidates <= 4096:
        raise ValueError("replicates must be 1..5 and max-candidates 1..4096")
    caps, supports = make_caps(x, face, distance, target, tolerance, **colour_bounds)
    accepted = 0
    for j in range(max_candidates):
        with torch.no_grad():
            carrier, row = sample_carrier(x, image, j)
            y, screened = screen_candidate(
                x, carrier.render(x), caps, supports, distance, target, tolerance)
        accepted += screened["feasible"]
        row.update(screened, replicate=accepted if screened["feasible"] else 0)
        yield y, row
        if accepted == replicates:
            break


def face_support(x, boxes):
    """Same largest-face expansion/rounding as curve_budget_defence helpers.

    Kept local to avoid importing that module's diffusion/objective front end.
    """
    from src.metrics.identity import DETECTOR_IMAGE_SIZE, DETECTOR_MARGIN

    if not boxes:
        raise ValueError("No face for skin-colour calibration")
    box = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
    x0, y0, x1, y1 = (float(v) for v in box)
    mx = DETECTOR_MARGIN * (x1 - x0) / (DETECTOR_IMAGE_SIZE - DETECTOR_MARGIN)
    my = DETECTOR_MARGIN * (y1 - y0) / (DETECTOR_IMAGE_SIZE - DETECTOR_MARGIN)
    h, w = x.shape[-2:]
    box = (max(0.0, x0 - mx / 2), max(0.0, y0 - my / 2),
           min(float(w), x1 + mx / 2), min(float(h), y1 + my / 2))
    a, b, c, d = (int(round(v)) for v in box)
    face = torch.zeros_like(x[:, :1])
    face[..., b:d, a:c] = 1.0
    if float(face.sum()) < 4:
        raise ValueError(f"Degenerate face support: {box}")
    return face


def output_root(path):
    return path.parent if path.name in {f"ab_prism_random_r{r}" for r in range(1, 6)} else path


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path, data):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def export_candidate(root, x, y, row):
    arm = f"ab_prism_random_r{row['replicate']}"
    folder = root / arm
    save_image(x, folder / f"{row['image']}__orig.png")
    save_image(y, folder / f"{row['image']}__{arm}__def.png")
    rows = [r for r in read_csv(folder / "results.csv") if r["image"] != row["image"]]
    rows.append({"arm": arm, "class": row["image"].split("_")[0], "carrier": "ab_prism", **row})
    write_csv(folder / "results.csv", sorted(rows, key=lambda r: r["image"]),
              ("arm", "class", "carrier", *FIELDS))


def cached_image(root, image, status, source_hash, replicates):
    if not status:
        return False
    if status["source_sha256"] != source_hash:
        raise ValueError(f"Source changed for {image}; use a new --out root")
    for r in range(1, replicates + 1):
        arm = f"ab_prism_random_r{r}"
        folder = root / arm
        rows = [row for row in read_csv(folder / "results.csv") if row["image"] == image]
        if r <= status["accepted"]:
            if (len(rows) != 1 or not (folder / f"{image}__orig.png").is_file()
                    or not (folder / f"{image}__{arm}__def.png").is_file()):
                return False
        elif rows:
            return False
    return (root / "ab_prism_random_log/candidates.csv").is_file()


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("lab/runs/defence"),
                    help="Output root, or its canonical ab_prism_random_rN child")
    ap.add_argument("--data", type=Path, default=Path("lab/data/portraits"))
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--images", nargs="+", choices=IMAGE_ORDER)
    ap.add_argument("--max-candidates", type=int, default=4096)
    ap.add_argument("--frame-cap", type=float, default=16.0)
    ap.add_argument("--face-cap", type=float, default=8.0)
    ap.add_argument("--skin-radius", type=float, default=12.0)
    ap.add_argument("--chroma-gain", type=float, default=1.15)
    ap.add_argument("--lpips-targets", type=Path, default=Path("lab/results/fidelity.csv"))
    ap.add_argument("--lpips-reference", default="ab_warp")
    ap.add_argument("--lpips-tolerance", type=float, default=0.0025)
    return ap


def main():
    args = parser().parse_args()
    if not 1 <= args.replicates <= 5 or not 1 <= args.max_candidates <= 4096:
        raise ValueError("replicates must be 1..5 and max-candidates 1..4096")
    if not torch.cuda.is_available():
        raise RuntimeError("ab_prism_random requires CUDA for piq.LPIPS (VGG)")
    targets = read_targets(args.lpips_targets, args.lpips_reference)
    images = [image for image in IMAGE_ORDER if not args.images or image in args.images]
    missing = [image for image in images if image not in targets]
    if missing:
        raise ValueError(f"Missing per-image LPIPS targets: {missing}")
    root = output_root(args.out)
    logdir = root / "ab_prism_random_log"
    logdir.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items()
                if k not in ("out", "data", "images", "lpips_targets")}
    settings.update(targets=targets, sampler="torch-cpu-randint-then-randn112-v1",
                    torch_version=torch.__version__, image_order=list(IMAGE_ORDER),
                    stds=list(STDS), carrier="AbPrismParam-default-112",
                    source_sha256={p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                                   for p in ("ab_prism_random.py", "ab_prism_defence.py")})
    config = logdir / "settings.json"
    if config.exists() and json.loads(config.read_text(encoding="utf-8")) != settings:
        raise ValueError("Random-control settings changed; use a new --out root")
    write_json(config, settings)
    summary_path = logdir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    log_path = logdir / "candidates.csv"
    lpips = None
    shortages = []
    for image in images:
        source = args.data / image.split("_")[0] / f"{image}.png"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if cached_image(root, image, summary.get(image), source_hash, args.replicates):
            count = summary[image]["accepted"]
            print(f"[SKIP] {image}: accepted={count}/{args.replicates}", flush=True)
        else:
            # Invalidate before replacing partial outputs, so interruption is safe.
            summary.pop(image, None)
            write_json(summary_path, summary)
            for r in range(1, args.replicates + 1):
                arm = f"ab_prism_random_r{r}"
                folder = root / arm
                rows = [row for row in read_csv(folder / "results.csv") if row["image"] != image]
                write_csv(folder / "results.csv", rows, ("arm", "class", "carrier", *FIELDS))
                for filename in (f"{image}__orig.png", f"{image}__{arm}__def.png"):
                    (folder / filename).unlink(missing_ok=True)
            # Stream a filtered copy; do not hold the potentially large log in RAM.
            if log_path.exists():
                with log_path.open(encoding="utf-8", newline="") as fh:
                    old = csv.DictReader(fh)
                    temp = log_path.with_name("candidates.filtered.csv")
                    write_csv(temp, (row for row in old if row["image"] != image), FIELDS)
                temp.replace(log_path)
            else:
                write_csv(log_path, [], FIELDS)
            if lpips is None:
                import piq

                lpips = piq.LPIPS().to("cuda").eval()  # piq's fixed VGG LPIPS
                lpips.requires_grad_(False)
            from src.metrics.identity import face_boxes
            from src.utils.io import load_image_tensor

            x = load_image_tensor(source, torch.device("cuda"), size=512)
            face = face_support(x, face_boxes(x, x.device))
            distance = lambda y: lpips(y, x).mean()
            count = attempted = 0
            with log_path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
                for y, row in candidates(
                        x, face, image, distance, targets[image], replicates=args.replicates,
                        max_candidates=args.max_candidates, tolerance=args.lpips_tolerance,
                        **{k: getattr(args, k) for k in CAP_ARGS}):
                    writer.writerow(row)
                    fh.flush()
                    attempted += 1
                    if row["feasible"]:
                        count += 1
                        export_candidate(root, x, y, row)
                    if row["feasible"] or attempted % 16 == 0:
                        print(f"[ab_prism_random] {image}: candidates={attempted} "
                              f"accepted={count}/{args.replicates}", flush=True)
            summary[image] = {"accepted": count, "requested": args.replicates,
                              "candidates": attempted, "feasible_rate": count / attempted,
                              "exhausted": count < args.replicates, "source_sha256": source_hash}
            write_json(summary_path, summary)
            print(f"[DONE] {image}: accepted={count}/{args.replicates}; "
                  f"candidates={attempted}; feasible_rate={count / attempted:.6f}", flush=True)
        if count < args.replicates:
            shortages.append(f"{image}={count}/{args.replicates}")
    if shortages:
        raise RuntimeError("Candidate budget exhausted; constraints unchanged: " + ", ".join(shortages))


if __name__ == "__main__":
    main()
