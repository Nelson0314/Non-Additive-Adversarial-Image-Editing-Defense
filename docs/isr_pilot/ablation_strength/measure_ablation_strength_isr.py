"""Conditional ISR and aesthetics of the reduced-strength ablation (Full, B2 and B3 at three strengths).

Inputs: judgments.csv (sheet,row,cell,purification,accessory,color_cast,mismatch,degraded,note) in this directory,
the blinding key color/results/ablation_strength/key.csv (sheet,row,condition,status) and the per-image readouts
color/results/ablation_strength/per_image.csv and color/results/ablation_isr_aesthetics/inputs/aesthetic_immulut.csv,
passed as arguments. Protocol: 30 portraits of crosseditor100 (default_rng(20260812)) x 4 InstructPix2Pix
instructions x purifications none / JPEG 50 / blur sigma 2 / VAE-Trans, main seed; rubric.md in this directory.

An edit succeeds when accessory=1, mismatch=0 and degraded=0. A defended cell counts toward the denominator only
when the Undefended edit of the same sheet and purification succeeds; it is an immunization success when the
defended edit does not succeed. LAION and MUSIQ are scores of the immunized image before purification, averaged
over the denominator cells. CIs are image-level bootstrap (1000 resamples, seed 20260812); paired differences
use the same resamples.
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PURIFS = ["none", "jpeg50", "blur2", "vae_trans"]
CONDS = ["Full", "B2_lpips0p85", "B2_lpips0p70", "B2_lpips0p55", "B3_lpips0p85", "B3_lpips0p70", "B3_lpips0p55"]


def main(key_path, per_image_path, immulut_aesthetic_path):
    key = {}
    for r in csv.DictReader(open(key_path)):
        assert r["status"] == "ok", r
        key[(r["sheet"].rsplit(".", 1)[0], r["row"])] = r["condition"]
    cell = {}
    for r in csv.DictReader(open(HERE / "judgments.csv")):
        k = (r["sheet"], r["purification"], key[(r["sheet"], r["row"])])
        assert k not in cell, k
        cell[k] = (r["accessory"] == "1" and r["mismatch"] == "0" and r["degraded"] == "0",
                   r["accessory"] == "0", r["mismatch"] == "1", r["degraded"] == "1",
                   r["accessory"] == "1" and r["color_cast"] == "1")
    sheets = sorted({s for s, _, _ in cell})
    assert len(sheets) == 120, len(sheets)
    assert len(cell) == 120 * 4 * 8, len(cell)

    aes = {}
    for r in csv.DictReader(open(per_image_path)):
        assert r["status"] == "delivered", r
        c = f"{r['row']}_lpips{float(r['ratio']):.2f}".replace(".", "p")
        assert (c, r["image"]) not in aes
        aes[(c, r["image"])] = (float(r["laion"]), float(r["musiq"]), float(r["defense_lpips"]))
    images30 = {s.rsplit("__", 1)[0] for s in sheets}
    full_lpips = {}
    for r in csv.DictReader(open(per_image_path)):
        full_lpips[r["image"]] = float(r["full_lpips"])
    for r in csv.DictReader(open(immulut_aesthetic_path)):
        if r["image"] in images30:
            aes[("Full", r["image"])] = (float(r["aes_laion"]), float(r["aes_musiq_ava"]), full_lpips[r["image"]])
    for c in CONDS:
        for img in images30:
            assert (c, img) in aes, (c, img)

    by_image = defaultdict(lambda: defaultdict(list))
    elig = defaultdict(int)
    for s in sheets:
        img = s.rsplit("__", 1)[0]
        for pu in PURIFS:
            if not cell[(s, pu, "Undefended")][0]:
                continue
            elig[pu] += 1
            elig["all"] += 1
            for c in CONDS:
                good, noacc, mism, degr, cast = cell[(s, pu, c)]
                laion, musiq, lp = aes[(c, img)]
                v = {"isr": not good, "noacc": noacc, "mismatch": mism, "degraded": degr, "cast": cast,
                     "laion": laion, "musiq": musiq, "lpips": lp}
                for g in (pu, "all"):
                    by_image[img][(c, g)].append(v)
    images = sorted(by_image)
    rng = random.Random(20260812)
    boots = [[rng.choice(images) for _ in images] for _ in range(1000)]

    def mean(imgs, c, g, m):
        vals = [v[m] for img in imgs for v in by_image[img].get((c, g), [])]
        return sum(vals) / len(vals), len(vals)

    def stat(c, g, m, scale):
        mu, n = mean(images, c, g, m)
        b = sorted(scale * mean(bi, c, g, m)[0] for bi in boots)
        return scale * mu, b[25], b[974], n

    def paired(c, g, m, scale):
        d = lambda imgs: scale * (mean(imgs, c, g, m)[0] - mean(imgs, "Full", g, m)[0])
        b = sorted(d(bi) for bi in boots)
        return d(images), b[25], b[974]

    print("Denominator cells (undefended edit successful): " +
          ", ".join(f"{g} {elig[g]}" for g in PURIFS + ["all"]) + f"; images {len(images)}")
    print("\nConditional ISR % [95% CI] (n cells)")
    print("cond          " + " | ".join(f"{g:>20s}" for g in PURIFS + ["all"]))
    for c in CONDS:
        print(f"{c:13s} " + " | ".join("{:5.1f} [{:4.1f},{:4.1f}] ({})".format(*stat(c, g, "isr", 100))
                                         for g in PURIFS + ["all"]))
    print("\nAll purifications pooled; % of denominator cells; LAION, MUSIQ, defense LPIPS averaged over the same cells")
    print("cond          ISR                  no_acc  mismatch degraded cast_done  LAION                 MUSIQ                 defense_LPIPS")
    for c in CONDS:
        isr = stat(c, "all", "isr", 100)
        la = stat(c, "all", "laion", 1)
        mu = stat(c, "all", "musiq", 1)
        print(f"{c:13s} {isr[0]:5.1f} [{isr[1]:4.1f},{isr[2]:4.1f}]  "
              f"{stat(c, 'all', 'noacc', 100)[0]:6.1f} {stat(c, 'all', 'mismatch', 100)[0]:8.1f} "
              f"{stat(c, 'all', 'degraded', 100)[0]:8.1f} {stat(c, 'all', 'cast', 100)[0]:9.1f}  "
              f"{la[0]:.3f} [{la[1]:.3f},{la[2]:.3f}]  {mu[0]:.3f} [{mu[1]:.3f},{mu[2]:.3f}]  "
              f"{stat(c, 'all', 'lpips', 1)[0]:.4f}")
    print("\nPaired difference to Full (condition - Full), all purifications pooled [95% CI]")
    print("cond          ISR pp                  LAION                     MUSIQ")
    for c in CONDS[1:]:
        print(f"{c:13s} " + "  ".join("{:+6.2f} [{:+6.2f},{:+6.2f}]".format(*paired(c, "all", m, s))
                                       for m, s in (("isr", 100), ("laion", 1), ("musiq", 1))))


if __name__ == "__main__":
    main(*sys.argv[1:4])
