"""Stack teaser candidates (PNG files from generate_pexels_teaser_figure.py) into one overview, one candidate per band,
with the image ID and instruction written on the left (run on the remote host).

    python generate_teaser_overview.py --out overview.png candidate_a.png:"woman_93, bowtie" candidate_b.png:"..."
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("items", nargs="+", help="PNG:label")
    args = ap.parse_args()
    items = [(Path(i.split(":", 1)[0]), i.split(":", 1)[1]) for i in args.items]
    images = [Image.open(path).convert("RGB") for path, _ in items]
    label_w, gap = 420, 40
    width = label_w + max(im.width for im in images)
    height = sum(im.height for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSerif.ttf", 44)
    except OSError:
        font = ImageFont.load_default()
    y = 0
    for im, (_, label) in zip(images, items):
        canvas.paste(im, (label_w, y))
        draw.multiline_text((20, y + im.height // 2), label.replace(", ", "\n"), fill="black", font=font, anchor="lm",
                            spacing=12)
        y += im.height
        if y < height:
            draw.line([(0, y + gap // 2), (width, y + gap // 2)], fill="#999999", width=3)
        y += gap
    canvas.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
