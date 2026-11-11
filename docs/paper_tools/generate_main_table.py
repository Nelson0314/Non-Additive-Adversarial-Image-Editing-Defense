"""Write Table 1 (tab:main) of the paper into docs/paper/sec/3_method.tex.

    python generate_main_table.py --main-table main_table.csv --capguard capguard_main_table.csv --blurguard blurguard_summary.csv

Upper block: InstructPix2Pix, 306 Pexels portraits, all methods; protected-image LPIPS, LAION, MUSIQ and the edit LPIPS,
SigLIP, ArcFace of the edit of the protected image against the edit of the original under the same purification.
Inputs: baseline remote runtime/pexels_portraits/summary/main_table.csv (with <col>_ci95) and
baseline remote runtime/pexels_portraits/summary/color3d_scale2p5/main_table.csv (ImmuLUT, 2.5x angle bounds, 306
portraits) and baseline remote runtime/pexels_portraits/blurguard_306/summary.csv (BlurGuard, 306 portraits). Small numbers are 95% interval half-widths
(Purified*: across the seven purifications; other columns: across images).
Lower blocks: the same edit columns on VIBE-DistilledCFG (306 portraits; baseline results/pexels_portraits/vibe_distilled/
summary.csv, --vibe) and FLUX.2 [klein] 4B (306 portraits; --klein, red placeholders when omitted), for PhotoGuard,
DCT-Shield, BlurGuard, Mist and ImmuLUT; the protected images of the upper block are used without re-optimization.
"""
import argparse
import csv
from pathlib import Path

TEX = Path(__file__).resolve().parents[1] / "paper" / "sec" / "3_method.tex"
ORDER = [("dayn", "DAYN"), ("dia_r", "DIA-R"), ("dia_pt", "DIA-PT"), ("sifm", "SIFM"), ("danp", "DANP"),
         ("mist", "Mist"), ("photoguard_encoder", "PhotoGuard"), ("dct_shield", "DCT-Shield"),
         ("dct_shield_y", "DCT-Shield-Y"), ("blurguard", "BlurGuard"), ("color3d_scale2p5_pexels", r"\textbf{\capg{} (Ours)}")]
CROSS = [("mist", "Mist"), ("photoguard_encoder", "PhotoGuard"), ("dct_shield", "DCT-Shield"), ("blurguard", "BlurGuard"),
         ("color3d_scale2p5_pexels", r"\textbf{\capg{} (Ours)}")]
COLS = [("defense_lpips", 3, min), ("aes_laion", 2, max), ("aes_musiq_ava", 2, max), ("disp_none", 3, max),
        ("disp_seven", 3, max), ("disp_vae_trans", 3, max), ("siglip_seven", 3, min), ("siglip_vae_trans", 3, min),
        ("arcface_pair_seven", 3, min), ("arcface_pair_vae_trans", 3, min)]
PRELIM_ROWS = set()


def f(x, d=3):
    t = f"{x:.{d}f}"
    return t[1:] if t.startswith("0.") else t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-table", type=Path, required=True)
    ap.add_argument("--capguard", type=Path, required=True)
    ap.add_argument("--blurguard", type=Path, required=True)
    ap.add_argument("--vibe", type=Path, required=True)
    ap.add_argument("--klein", type=Path)
    args = ap.parse_args()
    rows = {r["condition"]: r for r in csv.DictReader(open(args.main_table, encoding="utf-8"))}
    rows.update({r["condition"]: r for r in csv.DictReader(open(args.capguard, encoding="utf-8")) if r["condition"] != "original"})
    rows["blurguard"] = next(csv.DictReader(open(args.blurguard, encoding="utf-8")))
    best = {c: fn(float(rows[k][c]) for k, _ in ORDER) for c, _, fn in COLS}

    def cell(k, c, d):
        v = float(rows[k][c])
        t = f(v, d)
        if v == best[c]:
            t = r"\textbf{" + t + "}"
        t += r"{\scriptsize$\pm$" + f(float(rows[k][c + "_ci95"]), d) + "}"
        return r"\prelim{" + t + "}{}" if k in PRELIM_ROWS else t

    def editor_block(path):
        out = ""
        table = {r["condition"]: r for r in csv.DictReader(open(path, encoding="utf-8"))} if path else None
        if table:
            bests = {c: fn(float(table[k][c]) for k, _ in CROSS) for c, _, fn in COLS[3:]}
        for k, name in CROSS:
            if k == "color3d_scale2p5_pexels":
                out += "    \\midrule\n    \\rowcolor{oursrow}\n"
            if table:
                vals = []
                for c, d, _ in COLS[3:]:
                    v = float(table[k][c])
                    t = r"\textbf{" + f(v, d) + "}" if v == bests[c] else f(v, d)
                    vals.append(t + r"{\scriptsize$\pm$" + f(float(table[k][c + "_ci95"]), d) + "}")
            else:
                vals = [r"\prelim{--}{}"] * len(COLS[3:])
            out += "    " + name + " & & & & " + " & ".join(vals) + " \\\\\n"
        return out

    body = ""
    for k, name in ORDER:
        if k == "color3d_scale2p5_pexels":
            body += "    \\midrule\n    \\rowcolor{oursrow}\n"
        body += "    " + name + " & " + " & ".join(cell(k, c, d) for c, d, _ in COLS) + " \\\\\n"
    U, D = r"$\uparrow$", r"$\downarrow$"
    table = r"""\begin{table*}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{2.4pt}
  \begin{tabular}{l ccc ccc cc cc}
    \toprule
    & \multicolumn{3}{c}{\textbf{Protected image}} & \multicolumn{7}{c}{\textbf{Edit of the protected image} vs.\ edit of the original} \\
    \cmidrule(lr){2-4} \cmidrule(l){5-11}
    & & & & \multicolumn{3}{c}{LPIPS """ + U + r"""} & \multicolumn{2}{c}{SigLIP """ + D + r"""} & \multicolumn{2}{c}{ArcFace """ + D + r"""} \\
    \cmidrule(lr){5-7} \cmidrule(lr){8-9} \cmidrule(l){10-11}
    Method & LPIPS """ + D + " & LAION " + U + " & MUSIQ " + U + r""" & None & Purified$^{*}$ & VAE-Trans & Purified$^{*}$ & VAE-Trans & Purified$^{*}$ & VAE-Trans \\
    \midrule
    \multicolumn{11}{l}{\emph{InstructPix2Pix, 306 portraits}} \\
""" + body + r"""    \midrule[0.8pt]
    \multicolumn{11}{l}{\emph{VIBE-DistilledCFG, 306 portraits; protected images of the upper block, without re-optimization}} \\
""" + editor_block(args.vibe) + r"""    \midrule[0.8pt]
    \multicolumn{11}{l}{\emph{FLUX.2 [klein] 4B, 306 portraits; protected images of the upper block, without re-optimization}} \\
""" + editor_block(args.klein) + r"""    \bottomrule
  \end{tabular}
  \caption{Main comparison (306 Pexels portraits $\times$ 4 instructions; additive methods at the settings of their matched-distortion arm). Each image is first averaged over the four instructions. Small numbers: half-width of the 95\% confidence interval, across the seven purifications in the Purified$^{*}$ columns and across images (images with a detected face for ArcFace) in the other columns. Edits are compared under the same purification; SigLIP and ArcFace are the similarities of the two edits. Best mean in bold within each block. $^{*}$Mean over the seven purifications (JPEG 80, 50, 30; Gaussian blur $\sigma = 1, 2$; crop-and-resize; rotation). LAION and MUSIQ of the original images: """ + f"{float(rows['original']['aes_laion']):.2f} and {float(rows['original']['aes_musiq_ava']):.2f}" + r""".}
  \label{tab:main}
\end{table*}
""" + ("" if args.klein else r"""\prelimnote{Tab.~1, FLUX.2 [klein] block: 306-portrait edits in progress.}
""")
    s = TEX.read_text(encoding="utf-8")
    a = s.index(r"\begin{table*}[t]", s.index(r"\label{fig:plane}"))
    b = s.index(r"\end{table*}", s.index(r"\label{tab:main}")) + len(r"\end{table*}")
    while s[b:].lstrip().startswith(r"\prelimnote"):
        b = s.index("\n", s.index(r"\prelimnote", b)) + 1
    TEX.write_text(s[:a] + table + s[b:].lstrip("\n"), encoding="utf-8", newline="\n")
    print(f"wrote {TEX}")


if __name__ == "__main__":
    main()
