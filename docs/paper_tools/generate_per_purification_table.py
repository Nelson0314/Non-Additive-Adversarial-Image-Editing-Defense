"""Write the per-purification tables of the supplementary material (docs/paper/sec/X_suppl_purification.tex).

    python generate_per_purification_table.py \
        --per-purifier summary/per_purifier.csv blurguard_306/per_purifier.csv summary/color3d_scale2p5/per_purifier.csv \
        --main-table summary/main_table.csv blurguard_306/summary.csv summary/color3d_scale2p5/main_table.csv

Inputs are baseline's Pexels summary files (remote baseline/runtime/pexels_portraits/): each per_purifier.csv holds,
for each condition and each of the seven purifications, the mean over the 306 images (each image first averaged over
the four instructions); the VAE-Trans column comes from the *_vae_trans columns of the matching main_table.csv
(summary.csv for BlurGuard). Rows of the conditions in the tables are merged by condition; a condition found in two
files is an error.
"""

import argparse
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "paper" / "sec" / "X_suppl_purification.tex"
METHODS = [("dayn", "DAYN"), ("dia_r", "DIA-R"), ("dia_pt", "DIA-PT"), ("sifm", "SIFM"), ("danp", "DANP"),
           ("mist", "Mist"), ("photoguard_encoder", "PhotoGuard"), ("dct_shield", "DCT-Shield"),
           ("dct_shield_y", "DCT-Shield-Y"), ("blurguard", "BlurGuard"),
           ("color3d_scale2p5_pexels", r"\textbf{\capg{}}")]
OURS = "color3d_scale2p5_pexels"
PURIFIERS = [("jpeg80", "JPEG 80"), ("jpeg50", "JPEG 50"), ("jpeg30", "JPEG 30"), ("blur1", r"Blur $\sigma{=}1$"),
             ("blur2", r"Blur $\sigma{=}2$"), ("crop_resize0.1", "Crop"), ("rotate15", "Rotation")]
# (per_purifier column, main_table VAE-Trans column, label, decimals)
METRICS = [
    [("disp", "disp_vae_trans", r"Edit LPIPS $\uparrow$", 3), ("disp_ssim", "disp_ssim_vae_trans", r"SSIM between the two edits $\downarrow$", 3),
     ("disp_psnr", "disp_psnr_vae_trans", r"PSNR between the two edits (dB) $\downarrow$", 1), ("siglip", "siglip_vae_trans", r"SigLIP $\downarrow$", 3)],
    [("clip_i", "clip_i_vae_trans", r"CLIP-I $\downarrow$", 3), ("clip_s", "clip_s_vae_trans", r"CLIP-S $\downarrow$", 3),
     ("arcface_pair", "arcface_pair_vae_trans", r"ArcFace $\downarrow$", 3), ("fid", "fid_vae_trans", r"FID $\uparrow$", 1)],
]


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge(paths, key):
    rows = {}
    for path in paths:
        for r in read(path):
            if r["condition"] not in dict(METHODS):  # e.g. the "original" row present in every main_table.csv
                continue
            k = tuple(r[f] for f in key)
            if k in rows:
                raise SystemExit(f"{k} appears in more than one input ({path})")
            rows[k] = r
    return rows


def fmt(x, d):
    t = f"{x:.{d}f}"
    return t[1:] if t.startswith("0.") else t


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-purifier", type=Path, nargs="+", required=True)
    parser.add_argument("--main-table", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    per = merge(args.per_purifier, ("condition", "purifier"))
    main_rows = {k[0]: r for k, r in merge(args.main_table, ("condition",)).items()}

    out = []
    for k, group in enumerate(METRICS):
        body = ""
        for j, (col, vae_col, label, d) in enumerate(group):
            if j:
                body += "    \\midrule\n"
            body += f"    \\multicolumn{{9}}{{l}}{{\\textit{{{label}}}}} \\\\\n"
            for cond, name in METHODS:
                vals = [fmt(float(per[(cond, p)][col]), d) for p, _ in PURIFIERS]
                vals.append(fmt(float(main_rows[cond][vae_col]), d))
                prefix = "    \\rowcolor{oursrow}\n" if cond == OURS else ""
                body += prefix + "    " + name + " & " + " & ".join(vals) + " \\\\\n"
        part = "first" if k == 0 else "second"
        out.append(r"""\begin{table*}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \caption{Edit displacement for each purification (""" + part + r""" part; 306 Pexels portraits $\times$ 4 instructions, InstructPix2Pix, matched-distortion arms). Each entry is the mean over the 306 images, each image first averaged over the four instructions; \cref{tab:main} reports the mean over the seven purifications. SSIM, PSNR, SigLIP, CLIP-I, and ArcFace compare the immunized and the undefended edit under the same purification; CLIP-S compares the immunized edit with the instruction text; FID is computed between the sets of immunized and undefended edits.}
  \label{tab:supp_purification_""" + part + r"""}
  \begin{tabular}{l ccccccc c}
    \toprule
    Method & """ + " & ".join(label for _, label in PURIFIERS) + r""" & VAE-Trans \\
    \midrule
""" + body + r"""    \bottomrule
  \end{tabular}
\end{table*}
""")
    args.out.write_text("% Generated by docs/paper_tools/generate_per_purification_table.py; do not edit by hand.\n" + "\n".join(out),
                   encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
