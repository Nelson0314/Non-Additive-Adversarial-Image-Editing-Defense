"""Teaser candidates and figure on the 306 Pexels portraits (run on the remote host, CPU).

    cd ~/paper_figures && ~/venvs/wacv/bin/python in/generate_pexels_teaser_figure.py --rank 40
    cd ~/paper_figures && ~/venvs/wacv/bin/python in/generate_pexels_teaser_figure.py --image man_12 --prompt 1 \
        --out-dir out_teaser --name teaser_man_12_p1

Layout as generate_teaser_figure.py (--layout selects the rows; --no-lpips drops the LPIPS labels): rows are the protected image, the unpurified edit and the edit after
Gaussian blur (sigma 2); columns are no defense, the matched-distortion arms of PhotoGuard (encoder attack),
Mist, DIA-PT, DCT-Shield, and ours (color3d_pexels). The chips give the edit LPIPS against the undefended edit
under the same purification.

--rank lists (image, instruction) cells by the gap between our blur-2 edit LPIPS and the largest blur-2 edit
LPIPS of the four additive arms, keeping cells where the undefended edit follows the instruction (CLIP-S of the
undefended edit at least its median) and our protected image has a defense LPIPS at most the median of ours.
Sources: baseline runtime/pexels_portraits/aligned/<set>/results/retention_<arm>.csv, followup/<set>/results/
{clip_score.csv,color3d/fidelity.csv}, color trial results/color3d_pexels/retention.csv; image files are
resolved from the png column of clip_score.csv and from the defense directories.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

REPO = Path.home() / "image-immunization"
BASE = REPO / "baseline"
RUN = BASE / "runtime/pexels_portraits"
ART = BASE / "artifacts/pexels_portraits"
TRIAL = REPO / "color/trials/lut3d_defense_and_naturalness"
OURS_C = "color3d_pexels"
OURS_LABEL = "CAPGuard"
ARMS = [("PhotoGuard", "photoguard_encoder"), ("Mist", "mist"), ("DIA-PT", "dia_pt"), ("DCT-Shield", "dct_shield")]
LABELS = {"photoguard_encoder": "PhotoGuard", "mist": "Mist", "dia_pt": "DIA-PT", "dct_shield": "DCT-Shield",
          "dct_shield_y": "DCT-Shield-Y", "blurguard": "BlurGuard"}
CITES = {"photoguard_encoder": "Salman et al., 2023", "mist": "Liang & Wu, 2023", "dia_pt": "Hong et al., 2025",
         "dct_shield": "Bala et al., 2025", "dct_shield_y": "Bala et al., 2025", "blurguard": "Kim et al., 2025"}
INSTRUCTIONS = ["Let the person wear sunglasses", "Let the person wear a police suit", "Let the person wear a helmet",
                "Let the person wear a bowtie"]
PURIFIER = "blur2"
FID_QUANTILE = 0.4   # --rank keeps our protected images up to this quantile of defense LPIPS
LAION_DROP = -0.05   # and with LAION no lower than the original by more than this
INK, ADDITIVE, OURS = "#111111", "#A94438", "#1F6F7C"
WIDTH = 6.875


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_roots():
    sets = json.loads((BASE / "data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
    return [Path(s["root"]) if s["root"] else Path() for s in sets]


def load_tables():
    lp = {}       # (condition, image, prompt, purifier) -> edit LPIPS ("none" uses disp_plain)
    png = {}      # (condition, image, prompt, purifier) -> edit png path
    clip_s = {}   # (image, prompt) -> CLIP-S of the undefended edit
    fid = {}      # image -> defense LPIPS of ours
    laion = {}    # (condition, image) -> LAION score of the protected (or original) image
    for root in set_roots():
        res = RUN / "aligned" / root / "results"
        for _, c in ARMS:
            for r in read(res / f"retention_{c}.csv"):
                if r["scenario"] == "ip2p":
                    lp[(c, r["image"], r["prompt_index"], r["purifier"])] = float(r["disp_purified"])
                    lp[(c, r["image"], r["prompt_index"], "none")] = float(r["disp_plain"])
        fol = RUN / "followup" / root / "results"
        for r in read(fol / "clip_score.csv"):
            key = (r["condition"], r["image"], r["prompt_index"], r["purifier"])
            png[key] = BASE / r["png"]
            if r["condition"] == "undefended" and r["purifier"] == "none":
                clip_s[(r["image"], r["prompt_index"])] = float(r["clip_s"])
        for r in read(fol / "color3d/aesthetic.csv"):
            laion[(r["condition"], r["image"])] = float(r["aes_laion"])
        for r in read(fol / "color3d/fidelity.csv"):
            if r["condition"] == OURS_C:
                fid[r["image"]] = float(r["fid_lpips"])
    for r in read(TRIAL / "results/color3d_pexels/retention.csv"):
        lp[(OURS_C, r["image"], r["prompt_index"], r["purifier"])] = float(r["disp_purified"])
        lp[(OURS_C, r["image"], r["prompt_index"], "none")] = float(r["disp_plain"])
    return lp, png, clip_s, fid, laion


def rank(n):
    lp, _, clip_s, fid, laion = load_tables()
    med_clip = statistics.median(clip_s.values())
    fid_cut = sorted(fid.values())[int(len(fid) * FID_QUANTILE)]
    rows = []
    for (img, p), cs in clip_s.items():
        try:
            ours = lp[(OURS_C, img, p, PURIFIER)]
            add = [lp[(c, img, p, PURIFIER)] for _, c in ARMS]
        except KeyError:
            continue
        dl = laion.get((OURS_C, img), -9) - laion.get(("original", img), 9)
        if cs < med_clip or fid.get(img, 1.0) > fid_cut or dl < LAION_DROP:
            continue
        rows.append((ours - max(add), img, p, ours, max(add), cs, fid[img], dl))
    rows.sort(reverse=True)
    print("gap  image  prompt  ours_blur2  max_additive_blur2  clip_s_undefended  ours_defense_lpips  laion_change")
    for r in rows[:n]:
        print(f"{r[0]:.3f} {r[1]} {r[2]} {r[3]:.3f} {r[4]:.3f} {r[5]:.3f} {r[6]:.3f} {r[7]:+.3f}")


def find_one(pattern):
    hits = sorted(ART.glob(pattern)) + sorted(TRIAL.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def figure(img_id, prompt, out_dir, name, layout="full", chips=True, arms=None, cite=False):
    arms = arms or [c for _, c in ARMS]
    purified_only = layout == "purified"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    lp, png, _, _, _ = load_tables()
    p = str(prompt)

    def load(path):
        return np.asarray(Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS))

    columns = []
    for c in arms:
        label = LABELS[c]
        if c == "blurguard":  # matched-distortion BlurGuard on the 100-portrait subset (native-evaluation artifact root)
            files = (find_one(f"**/defenses/blurguard/{img_id}__blurguard__def.png"),
                     find_one(f"**/defended_edits/blurguard/ip2p_blurguard/{img_id}__p{p}.png"),
                     find_one(f"**/purified_edits/blurguard/{PURIFIER}/ip2p_blurguard_{PURIFIER}/{img_id}__p{p}.png"))
        else:
            files = (find_one(f"**/aligned/**/defenses/{c}/{img_id}__{c}__def.png"), png[(c, img_id, p, "none")],
                     png[(c, img_id, p, PURIFIER)])
        columns.append((label, ADDITIVE, *(load(f) for f in files),
                        (lp.get((c, img_id, p, "none"), float("nan")), lp.get((c, img_id, p, PURIFIER), float("nan"))),
                        CITES[c] if cite else None))
    columns.append((OURS_LABEL, OURS, load(TRIAL / f"artifacts_pexels/defenses/{OURS_C}/{img_id}__{OURS_C}__def.png"),
                    load(png[(OURS_C, img_id, p, "none")]), load(png[(OURS_C, img_id, p, PURIFIER)]),
                    (lp[(OURS_C, img_id, p, "none")], lp[(OURS_C, img_id, p, PURIFIER)]), None))
    source = load(find_one(f"**/defenses/**/{img_id}__orig.png"))
    ref_plain = load(png[("undefended", img_id, p, "none")])
    ref_purified = load(png[("undefended", img_id, p, PURIFIER)])

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5, "text.color": INK})
    s, head, foot = 1.0, (0.30 if cite else 0.17), 0.27
    label_w, gap0, gap_ours = 0.36, 0.14, 0.07
    nrows = {"full": 3, "purified": 1, "protected-purified": 2}[layout]
    if purified_only:
        label_w = 0.0
    H = head + nrows * s + foot
    fig = plt.figure(figsize=(WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH), ax.set_ylim(0, H)
    ax.axis("off")
    x_ref = (WIDTH - (6 * s + label_w + gap0 + gap_ours)) / 2
    x0 = x_ref + s + gap0 + label_w
    xs = [x0 + i * s for i in range(4)] + [x0 + 4 * s + gap_ours]
    ys = [H - head - (i + 1) * s for i in range(nrows)]

    def show(im, x, y, edge=INK, lw=0.5, chip=None):
        ax.imshow(im, extent=(x, x + s, y, y + s), interpolation="bilinear", zorder=2)
        inset = lw / 72 / 2
        ax.add_patch(plt.Rectangle((x + inset, y + inset), s - 2 * inset, s - 2 * inset, fill=False, ec=edge, lw=lw,
                                   zorder=3))
        if chip is not None and chips:
            ax.text(x + s / 2, y + 0.075, f"LPIPS {chip:.2f}", ha="center", va="center", fontsize=7.5, color="white",
                    zorder=4, bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.62))

    ax.text(x_ref + s / 2, H - 0.03, "No defense", ha="center", va="top")
    if purified_only:
        show(ref_purified, x_ref, ys[0])
        for x, (label, color, _, _, purified, (_, lp1), src) in zip(xs, columns):
            ax.text(x + s / 2, H - 0.03, label, ha="center", va="top", color=OURS if label == OURS_LABEL else INK)
            if src:
                ax.text(x + s / 2, H - 0.17, src, ha="center", va="top", fontsize=6.5, color="#555555")
            show(purified, x, ys[0], edge=color, lw=1.8, chip=lp1)
    elif layout == "protected-purified":
        show(source, x_ref, ys[0])
        show(ref_purified, x_ref, ys[1])
        for x, (label, color, protected, _, purified, (_, lp1), src) in zip(xs, columns):
            ax.text(x + s / 2, H - 0.03, label, ha="center", va="top", color=OURS if label == OURS_LABEL else INK)
            if src:
                ax.text(x + s / 2, H - 0.17, src, ha="center", va="top", fontsize=6.5, color="#555555")
            show(protected, x, ys[0])
            show(purified, x, ys[1], edge=color, lw=1.8, chip=lp1)
        for label, y in zip(["Protection", "Edited\n(purified)"], ys):
            ax.text(x0 - 0.05, y + s / 2, label, rotation=90, ha="right", va="center", fontsize=8.5, linespacing=1.05)
    else:
        for im, y in zip((source, ref_plain, ref_purified), ys):
            show(im, x_ref, y)
        for x, (label, color, protected, edited, purified, (lp0, lp1), src) in zip(xs, columns):
            ax.text(x + s / 2, H - 0.03, label, ha="center", va="top", color=OURS if label == OURS_LABEL else INK)
            if src:
                ax.text(x + s / 2, H - 0.17, src, ha="center", va="top", fontsize=6.5, color="#555555")
            show(protected, x, ys[0])
            show(edited, x, ys[1], chip=lp0)
            show(purified, x, ys[2], edge=color, lw=1.8, chip=lp1)
        for label, y in zip(["Protection", "Edited\n(unpurified)", "Edited\n(purified)"], ys):
            ax.text(x0 - 0.05, y + s / 2, label, rotation=90, ha="right", va="center", fontsize=8.5, linespacing=1.05)
    ax.text(x_ref + 0.03, 0.12, f"Edit prompt: “{INSTRUCTIONS[prompt]}”", ha="left", va="center", fontsize=8.5,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=INK, lw=0.5))
    ax.text(xs[-1] + s, 0.12, r"Purification: Gaussian blur, $\sigma = 2$", ha="right", va="center", fontsize=8.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", dpi=300)
    fig.savefig(out_dir / f"{name}.png", dpi=300, facecolor="white")
    print(f"wrote {out_dir / name}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, help="list the top cells instead of drawing")
    ap.add_argument("--image")
    ap.add_argument("--prompt", type=int)
    ap.add_argument("--out-dir", type=Path, default=Path("out_teaser"))
    ap.add_argument("--name", default="teaser")
    ap.add_argument("--layout", choices=["full", "purified", "protected-purified"], default="full",
                    help="rows: protected/unpurified/purified, purified only, or protected and purified")
    ap.add_argument("--no-lpips", action="store_true", help="omit the LPIPS labels on the edits")
    ap.add_argument("--arms", help="comma-separated additive conditions, e.g. photoguard_encoder,mist,dia_pt,blurguard")
    ap.add_argument("--cite", action="store_true", help="add the source (author, year) under each method name")
    args = ap.parse_args()
    if args.rank:
        rank(args.rank)
    else:
        figure(args.image, args.prompt, args.out_dir, args.name, args.layout, not args.no_lpips,
               args.arms.split(",") if args.arms else None, args.cite)


if __name__ == "__main__":
    main()
