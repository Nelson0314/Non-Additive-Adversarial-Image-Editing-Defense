"""論文質性圖（Fig. 4，purification_example）：一張影像、一句 ip2p 指令，取主表 306 張的既有產物，不重跑編輯。

欄：Undefended（原圖與其編輯）、PhotoGuard、Mist、DIA-PT、DCT-Shield（等失真臂）、BlurGuard（306 張版）、ImmuLUT
（color3d_scale2p5_pexels）；列：Immunized（防禦圖）、Edited (unpurified)、Edited (purified)（--purification，blur2 或 jpeg<q>）。
編輯協定為主表 ip2p（text 7.5、image 1.8、50 步、主種子 20260812）。配色、邊框與方法名取自 color trial
tools/generate_teaser_paper_figure.py 的常數（只讀取、不修改）；版面在本腳本繪製：不列出處，所有文字的字級
＝ --min-print-pt × 圖寬 6.875 in ／ --print-width（論文置入寬度），使印刷字級不小於 --min-print-pt；高寬比固定為 0.4696
（原版面 6.875 × 3.228 in），圖格邊長取寬、高兩個限制的較小者。
所用產物的來源路徑寫入 <輸出根>/<影像>/edits.csv。產物路徑規則同 scripts/run_inpaint_edits.py；缺檔即失敗。
輸出 <輸出根>/<影像>_p<k>[_<淨化>].{pdf,png}（預設輸出根 artifacts/pexels_portraits/purification_example/）。
用法（遠端 baseline 專案根，CPU）：$PY scripts/generate_purification_example_figure.py --image woman_111 --prompt-index 1 [--purification blur2]
"""
import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import yaml

WIDTH, ASPECT = 6.875, 0.4696
ROW_LABELS = ("Immunized", "Edited\n(unpurified)", "Edited\n(purified)")
ARMS = ("photoguard_encoder", "mist", "dia_pt", "dct_shield")
SEED, TEXT_GUIDANCE, IMAGE_GUIDANCE = 20260812, 7.5, 1.8
DATA = Path("data/pexels_portraits")
COLOR_TRIAL = Path("../color/trials/lut3d_defense_and_naturalness")
FIGURE_TOOL = COLOR_TRIAL / "tools/generate_teaser_paper_figure.py"


def sources(img, k, purifier):
    """{條件: (防禦圖, 不淨化編輯, 淨化後編輯)}；條件 undefended 的防禦圖為原圖。"""
    sets = json.loads((DATA / "image_sets.json").read_text(encoding="utf-8"))["sets"]
    root = next(s for s in sets if img in s["images"])["root"] or ""
    na = Path("artifacts/pexels_portraits") / root
    aa = Path("artifacts/pexels_portraits/aligned") / root
    edit = f"{img}__p{k}.png"
    out = {"undefended": (DATA / img.split("_")[0] / f"{img}.png", na / "undefended_edits/ip2p_si18" / edit,
                          na / "purified_edits/undefended" / purifier / f"ip2p_undefended_{purifier}" / edit)}
    for c in ARMS:
        out[c] = (aa / "defenses" / c / f"{img}__{c}__def.png", aa / "defended_edits" / c / f"ip2p_{c}" / edit,
                  aa / "purified_edits" / c / purifier / f"ip2p_{c}_{purifier}" / edit)
    first = set(Path("runtime/pexels_portraits/blurguard/images.txt").read_text(encoding="utf-8").split())
    if img in first:
        b = na
    else:
        unit = next(p.stem for p in sorted(Path("runtime/pexels_portraits/blurguard_rest/units").glob("*.txt"))
                    if img in p.read_text(encoding="utf-8").split())
        b = Path("runtime/pexels_portraits/blurguard_rest/artifacts") / unit
    out["blurguard"] = (b / "defenses/blurguard" / f"{img}__blurguard__def.png",
                        b / "defended_edits/blurguard/ip2p_blurguard" / edit,
                        b / "purified_edits/blurguard" / purifier / f"ip2p_blurguard_{purifier}" / edit)
    c = "color3d_scale2p5_pexels"
    e = COLOR_TRIAL / "artifacts_pexels/edits_by_image" / c / img
    out["immulut"] = (COLOR_TRIAL / "artifacts_pexels/defenses" / c / f"{img}__{c}__def.png",
                      e / "defended" / f"ip2p_{c}" / edit, e / "purified" / purifier / f"ip2p_{c}_{purifier}" / edit)
    missing = [p.as_posix() for files in out.values() for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"缺 {len(missing)} 個檔：{missing}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt-index", type=int, required=True)
    ap.add_argument("--purification", default="blur2")
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/pexels_portraits/purification_example"))
    ap.add_argument("--print-width", type=float, default=0.62 * WIDTH, help="論文置入寬度（英吋），預設為論文正文寬度的 0.62 倍")
    ap.add_argument("--min-print-pt", type=float, default=7.0, help="印刷字級下限（pt）")
    a = ap.parse_args()
    img, k, purifier = a.image, a.prompt_index, a.purification
    prompts = yaml.safe_load((DATA / "prompts.yaml").read_text(encoding="utf-8"))["edits"]["ip2p"]
    run = a.output_root / img
    if run.exists():
        shutil.rmtree(run)
    files = sources(img, k, purifier)
    run.mkdir(parents=True)
    rows = [{"condition": c, "row": row, "image": img, "prompt_index": k, "source_png": src.as_posix(),
             "text_guidance": TEXT_GUIDANCE, "image_guidance": IMAGE_GUIDANCE, "seed": SEED}
            for c, triple in files.items() for row, src in zip(("immunized", "none", purifier), triple)]
    with open(run / "edits.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    name = f"{img}_p{k}" + ("" if purifier == "blur2" else f"_{purifier}")
    draw(files, prompts[k], purifier, a.min_print_pt * WIDTH / a.print_width, a.output_root / name)


def draw(files, prompt, purifier, fs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    spec = importlib.util.spec_from_file_location("teaser_paper_figure", FIGURE_TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    ink, additive, ours = tool.INK, tool.ADDITIVE, tool.OURS
    columns = [(tool.LABELS[c], ink, additive, files[c]) for c in (*ARMS, "blurguard")]
    columns.append((tool.OURS_LABEL.replace(" (", "\n("), ours, ours, files["immulut"]))

    def load(path):
        return np.asarray(Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS))

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": fs, "text.color": ink})
    line = fs * 1.15 / 72                      # 一行文字的高度（英吋）
    H = ASPECT * WIDTH
    head, foot = 2 * line + 0.06, line + 0.12  # ImmuLUT (Ours) 分兩行
    label_w, gap0, gap_ours = 2 * line + 0.04, 0.10, 0.07
    n = len(columns) + 1
    s = min((WIDTH - 0.1 - label_w - gap0 - gap_ours) / n, (H - head - foot) / 3)
    x_ref = (WIDTH - (n * s + label_w + gap0 + gap_ours)) / 2
    x0 = x_ref + s + gap0 + label_w
    xs = [x0 + i * s for i in range(len(columns) - 1)] + [x0 + (len(columns) - 1) * s + gap_ours]
    ys = [H - head - (i + 1) * s for i in range(3)]
    fig = plt.figure(figsize=(WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH), ax.set_ylim(0, H)
    ax.axis("off")

    def show(path, x, y, edge=ink, lw=0.5):
        ax.imshow(load(path), extent=(x, x + s, y, y + s), interpolation="bilinear", zorder=2)
        inset = lw / 72 / 2
        ax.add_patch(plt.Rectangle((x + inset, y + inset), s - 2 * inset, s - 2 * inset, fill=False, ec=edge, lw=lw,
                                   zorder=3))

    y_head = ys[0] + s + 0.04
    ax.text(x_ref + s / 2, y_head, "Undefended", ha="center", va="bottom")
    for path, y in zip(files["undefended"], ys):
        show(path, x_ref, y)
    for x, (label, color, edge, (protected, edited, purified)) in zip(xs, columns):
        ax.text(x + s / 2, y_head, label, ha="center", va="bottom", color=color, linespacing=1.0)
        show(protected, x, ys[0])
        show(edited, x, ys[1])
        show(purified, x, ys[2], edge=edge, lw=1.8)
    for label, y in zip(ROW_LABELS, ys):
        ax.text(x0 - 0.05, y + s / 2, label, rotation=90, ha="right", va="center", linespacing=1.0)
    y_foot = ys[2] - 0.06 - line / 2
    ax.text(x_ref + 0.03, y_foot, f"Edit prompt: “{prompt}”", ha="left", va="center",
            bbox=dict(boxstyle="square,pad=0.25", fc="white", ec=ink, lw=0.5))
    if purifier == "blur2":
        note = r"Purification: Gaussian blur, $\sigma = 2$"
    elif purifier.startswith("jpeg") and purifier[4:].isdigit():
        note = f"Purification: JPEG, quality {int(purifier[4:])}"
    else:
        raise SystemExit(f"未定義的淨化標示：{purifier}")
    ax.text(xs[-1] + s, y_foot, note, ha="right", va="center")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[WROTE] {out}.{{pdf,png}}  {WIDTH} x {H:.3f} in，字級 {fs:.2f} pt，圖格 {s:.3f} in")


if __name__ == "__main__":
    main()
