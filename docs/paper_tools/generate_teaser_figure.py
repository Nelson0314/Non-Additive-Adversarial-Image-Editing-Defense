"""Fig. 1（fig:teaser）：各防禦在未淨化與 Gaussian blur σ=2 後的編輯結果。

列：防禦圖、未淨化的編輯、淨化後的編輯。欄：未防禦、四個加性方法（等失真臂）、本方法（color3d）。
編輯 LPIPS 對同一淨化下未防禦的編輯：加性方法讀 baseline results/aligned/retention.csv，
本方法讀 color trial results/comparison/cuda/metrics_color3d.csv。

遠端 CPU 執行（basic-1）：
  cd ~/paper_figures && ~/venvs/wacv/bin/python in/generate_teaser_figure.py --out-dir out_teaser
輸出 teaser.{pdf,png}；scp 回 docs/paper/fig/。
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path.home() / "image-immunization"
TRIAL = REPO / "color/trials/lut3d_defense_and_naturalness"
BASE = REPO / "baseline/artifacts/aligned"
IMG, PROMPT, PURIFIER = "woman_00", 3, "blur2"
INSTRUCTIONS = ["Let the person wear sunglasses", "Let the person wear a police suit", "Let the person wear a helmet",
                "Let the person wear a bowtie"]  # color/data/portraits/prompts.yaml 的 edits.ip2p
INSTRUCTION = INSTRUCTIONS[PROMPT]
ARMS = [("PhotoGuard", "photoguard_encoder"), ("Mist", "mist"), ("DIA-PT", "dia_pt"), ("DCT-Shield", "dct_shield")]

INK = "#111111"
ADDITIVE = "#A94438"
OURS = "#1F6F7C"
WIDTH = 6.875


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS))


def arm_lpips(cond: str) -> tuple[float, float]:
    path = REPO / "baseline/results/aligned/retention.csv"
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if (r["condition"], r["purifier"], r["scenario"], r["image"], r["prompt_index"]) == (
                cond, PURIFIER, "ip2p", IMG, str(PROMPT)):
            return float(r["disp_plain"]), float(r["disp_purified"])
    raise KeyError(cond)


def ours_lpips() -> tuple[float, float]:
    path = TRIAL / "results/comparison/cuda/metrics_color3d.csv"
    rows = {r["purifier"]: float(r["lpips"]) for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["image"] == IMG and r["prompt_index"] == str(PROMPT)}
    return rows["none"], rows[PURIFIER]


def main():
    global IMG, PROMPT, INSTRUCTION
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--image", default="woman_00", help="portraits 影像 ID；預設為論文 Fig. 1 的例子")
    ap.add_argument("--prompt", type=int, default=3, help="指令編號（color/data/portraits/prompts.yaml 的 ip2p 清單）")
    ap.add_argument("--name", default="teaser", help="輸出檔名（不含副檔名）")
    args = ap.parse_args()
    IMG, PROMPT = args.image, args.prompt
    INSTRUCTION = INSTRUCTIONS[PROMPT]
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5, "text.color": INK})

    p = f"{IMG}__p{PROMPT}.png"
    columns = [(name, ADDITIVE,
                load(BASE / f"defenses/{c}/{IMG}__{c}__def.png"),
                load(BASE / f"defended_edits/{c}/ip2p_{c}_aligned" / p),
                load(BASE / f"purified_edits/{c}/{PURIFIER}/ip2p_{c}_{PURIFIER}" / p),
                arm_lpips(c)) for name, c in ARMS]
    columns.append(("Ours", OURS,
                    load(TRIAL / f"artifacts/defenses/color3d/{IMG}__color3d__def.png"),
                    load(TRIAL / "artifacts/defended_edits/color3d/ip2p_color3d" / p),
                    load(TRIAL / f"artifacts/purified_edits/color3d/{PURIFIER}/ip2p_color3d_{PURIFIER}" / p),
                    ours_lpips()))
    source = load(BASE / f"defenses/dct_shield/{IMG}__orig.png")
    ref_plain = load(REPO / "color/artifacts/undefended_edits/ip2p_si18" / p)
    ref_purified = load(TRIAL / f"artifacts/purified_edits/undefended/{PURIFIER}/ip2p_undefended_{PURIFIER}" / p)

    s, head, foot = 1.0, 0.17, 0.27
    label_w, gap0, gap_ours = 0.36, 0.14, 0.07
    H = head + 3 * s + foot
    fig = plt.figure(figsize=(WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH), ax.set_ylim(0, H)
    ax.axis("off")

    x_ref = (WIDTH - (6 * s + label_w + gap0 + gap_ours)) / 2
    x0 = x_ref + s + gap0 + label_w
    xs = [x0 + i * s for i in range(4)] + [x0 + 4 * s + gap_ours]
    ys = [H - head - (i + 1) * s for i in range(3)]

    def show(img, x, y, edge=INK, lw=0.5, chip=None):
        ax.imshow(img, extent=(x, x + s, y, y + s), interpolation="bilinear", zorder=2)
        inset = lw / 72 / 2
        ax.add_patch(plt.Rectangle((x + inset, y + inset), s - 2 * inset, s - 2 * inset, fill=False, ec=edge, lw=lw,
                                   zorder=3))
        if chip is not None:
            ax.text(x + s / 2, y + 0.075, f"LPIPS {chip:.2f}", ha="center", va="center", fontsize=7.5, color="white",
                    zorder=4, bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.62))

    ax.text(x_ref + s / 2, H - 0.03, "No defense", ha="center", va="top")
    for img, y in zip((source, ref_plain, ref_purified), ys):
        show(img, x_ref, y)

    for x, (name, color, protected, edited, purified, (lp0, lp1)) in zip(xs, columns):
        ax.text(x + s / 2, H - 0.03, name, ha="center", va="top", color=OURS if name == "Ours" else INK)
        show(protected, x, ys[0])
        show(edited, x, ys[1], chip=lp0)
        show(purified, x, ys[2], edge=color, lw=1.8, chip=lp1)

    labels = ["Protection", "Edited\n(unpurified)", "Edited\n(purified)"]
    for label, y in zip(labels, ys):
        ax.text(x0 - 0.05, y + s / 2, label, rotation=90, ha="right", va="center", fontsize=8.5, linespacing=1.05)

    ax.text(x_ref + 0.03, 0.12, f"Edit prompt: “{INSTRUCTION}”", ha="left", va="center", fontsize=8.5,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=INK, lw=0.5))
    ax.text(xs[-1] + s, 0.12, r"Purification: Gaussian blur, $\sigma = 2$", ha="right", va="center", fontsize=8.5)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / f"{args.name}.pdf", dpi=300)
    fig.savefig(args.out_dir / f"{args.name}.png", dpi=300, facecolor="white")


if __name__ == "__main__":
    main()
