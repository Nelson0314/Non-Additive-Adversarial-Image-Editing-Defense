"""原尺寸（512）的逐張對比頁，**拆成多頁**以繞開內嵌上限。

為什麼要拆
────────────────────────────────────────────────────────────────────
Artifact 的內嵌總量上限是 16 MB，而全部條件的原尺寸圖遠超過：
512×512 的 JPEG 在 q82 約 70 KB，10 個條件 × 10 張 × 5 欄 = 500 張，
base64 之後約 **47 MB**。這個帳號沒有開 `assets` 能力（可用的只有
`artifact`／`downloads`／`mcp`／`self`），所以無法把影像放到頁面之外。

**縮圖不是選項**：判「擋下與否」看的是語意內容，縮放會把細節抹掉，邊緣案例
會判錯——這正是使用者要求原尺寸的理由。

於是改成拆頁，並去掉一份冗餘：`原圖` 與 `原圖的編輯` 對同一張影像的所有條件
都相同，**每張只放一次**，之後每個條件只放三欄（防禦圖／淨化後／淨化後的編輯）。
每頁 C 個條件的影像數是 `10 × (2 + 3C)`，C = 3 時約 10.5 MB，留得下餘裕。

用法：
    python scripts/mainline_figure_pages.py --purifier jpeg75 \\
        --sheets runs/ip2p_mainline/sheets --out runs/ip2p_mainline/figpages
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TILE, PAD, TITLE_BAND, HEAD_BAND, ROWLAB = 512, 16, 46, 34, 30
JPEG_Q = 82          # 實測 512×512 在此品質約 70 KB
PER_PAGE = 3         # 每頁幾個條件。10 × (2 + 3×3) = 110 張 ≈ 10.5 MB
LIMIT_MB = 15.0

STYLE = """<style>
:root{--ink:#14101A;--ink-2:#4A4353;--ink-3:#736B7E;--paper:#F1ECEF;
 --surface:#FCFAFB;--line:#DCD3D9;--accent:#B01455;--alt:#0E6E78;--good:#15803D}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ink:#EDE7EC;--ink-2:#B3A9B6;--ink-3:#8A8090;--paper:#131017;--surface:#1B1620;
 --line:#2E2733;--accent:#F2679F;--alt:#5FC7D0;--good:#5CC98A}}
:root[data-theme="dark"]{--ink:#EDE7EC;--ink-2:#B3A9B6;--ink-3:#8A8090;
 --paper:#131017;--surface:#1B1620;--line:#2E2733;--accent:#F2679F;--alt:#5FC7D0;
 --good:#5CC98A}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
 font-family:"IBM Plex Serif",Georgia,"Songti TC",serif;font-size:16px;line-height:1.7}
.mono{font-family:ui-monospace,Menlo,"IBM Plex Mono",monospace}
h1,h2,h3{font-family:Archivo,"Helvetica Neue",Arial,"PingFang TC",sans-serif;
 letter-spacing:-.015em;text-wrap:balance}
.wrap{max-width:1760px;margin:0 auto;padding:0 20px}
header{border-bottom:1px solid var(--line);background:var(--surface);
 background-image:radial-gradient(var(--line) 1px,transparent 1px);background-size:12px 12px}
header .wrap{padding:34px 20px 26px}
.eyebrow{font-family:ui-monospace,monospace;font-size:11.5px;letter-spacing:.16em;
 text-transform:uppercase;color:var(--accent);margin:0 0 10px}
h1{font-size:clamp(26px,3.6vw,38px);margin:0 0 10px;font-weight:700}
.lede{font-size:17px;color:var(--ink-2);margin:0;max-width:70ch}
.img-block{padding:30px 0;border-bottom:1px solid var(--line)}
.imhead{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin:0 0 4px}
.imhead .id{font-family:ui-monospace,monospace;font-size:13.5px;font-weight:600}
.imhead .prompt{font-size:15px;color:var(--accent);font-style:italic}
.ref{display:flex;gap:10px;margin:12px 0 20px;flex-wrap:wrap}
.cond-row{margin:0 0 18px}
.cond-name{font-family:Archivo,sans-serif;font-size:13.5px;font-weight:600;margin:0 0 6px}
.cond-name.ours{color:var(--accent)} .cond-name.rival{color:var(--alt)}
.strip{display:flex;gap:10px;flex-wrap:wrap}
figure{margin:0}
figure img{display:block;width:512px;height:512px;border:1px solid var(--line);
 border-radius:5px;background:var(--surface)}
figure.key img{border-color:var(--accent);border-width:2px}
figcaption{font-family:ui-monospace,monospace;font-size:11px;color:var(--ink-3);
 margin-top:5px;max-width:512px}
@media (max-width:1200px){figure img{width:100%;height:auto;max-width:512px}}
footer{padding:30px 0 50px;color:var(--ink-3);font-size:13px}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
</style>"""


def crop_full(png: Path, n_rows: int):
    """把五欄大圖逐列裁出來，**保持 512 原尺寸**。"""
    from PIL import Image

    im = Image.open(png)
    top = TITLE_BAND + HEAD_BAND
    out = []
    for i in range(n_rows):
        y = top + i * (ROWLAB + TILE + PAD) + ROWLAB
        row = []
        for c in range(5):
            x = PAD + c * (TILE + PAD)
            b = io.BytesIO()
            im.crop((x, y, x + TILE, y + TILE)).convert("RGB").save(
                b, format="JPEG", quality=JPEG_Q, optimize=True)
            row.append("data:image/jpeg;base64,"
                       + base64.b64encode(b.getvalue()).decode())
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--purifier", default="jpeg75")
    ap.add_argument("--tags", nargs="+", required=True,
                    help="條件的順序；每 --per-page 個一頁")
    ap.add_argument("--per-page", type=int, default=PER_PAGE)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from apa_baseline import load_dataset

    T = json.loads((args.tables / "tables.json").read_text(encoding="utf-8"))
    label = T.get("label", {})
    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    ds = {d["name"]: d for d in load_dataset(args.data)}
    args.out.mkdir(parents=True, exist_ok=True)

    have = {}
    for tag in args.tags:
        p = args.sheets / f"{tag}__{args.purifier}.png"
        if p.exists():
            have[tag] = crop_full(p, len(names))
        else:
            print(f"  跳過 {tag}：{p} 不存在")
    tags = [t for t in args.tags if t in have]
    if not tags:
        raise SystemExit("沒有任何條件有對應的大圖")

    pages = [tags[i:i + args.per_page] for i in range(0, len(tags), args.per_page)]
    written = []
    for pi, group in enumerate(pages, 1):
        parts = [f"<title>對比圖 {args.purifier} 第{pi}頁</title>", STYLE,
                 "<header><div class='wrap'>",
                 "<p class='eyebrow'>原尺寸 512 · 不縮放</p>",
                 f"<h1>逐張對比 · {html.escape(args.purifier)} · 第 {pi}/{len(pages)} 頁</h1>",
                 "<p class='lede'>每張影像先給<b>原圖</b>與<b>原圖的編輯</b>（判準的參照點），"
                 "接著每個條件三欄：防禦圖 → 被淨化 → 淨化後的編輯。"
                 "<b>最後一欄才是判「擋下與否」的那一格。</b>本頁的條件："
                 + "、".join(html.escape(label.get(t, t)) for t in group)
                 + "。</p></div></header>"]
        for i, n in enumerate(names):
            pr = ds[n]["prompt"]
            ref = "".join(
                f"<figure><img src='{have[group[0]][i][c]}' alt='{html.escape(n)}'"
                f" loading='lazy'><figcaption>{lab}</figcaption></figure>"
                for c, lab in ((0, "原圖"), (1, "原圖的編輯（未防禦）")))
            rows = ""
            for tag in group:
                cls = "ours" if tag.startswith("ours") else "rival"
                cells = "".join(
                    f"<figure class='{'key' if c == 4 else ''}'>"
                    f"<img src='{have[tag][i][c]}' alt='{html.escape(n)} {lab}'"
                    f" loading='lazy'><figcaption>{html.escape(lab)}</figcaption></figure>"
                    for c, lab in ((2, "防禦圖"),
                                   (3, f"防禦圖被 {args.purifier} 淨化"),
                                   (4, f"淨化後的編輯 ← 判這一格")))
                rows += (f"<div class='cond-row'><p class='cond-name {cls}'>"
                         f"{html.escape(label.get(tag, tag))}</p>"
                         f"<div class='strip'>{cells}</div></div>")
            parts.append(
                f"<div class='img-block'><div class='wrap'>"
                f"<div class='imhead'><span class='id'>#{i + 1:02d} {html.escape(n)}</span>"
                f"<span class='prompt'>指令：{html.escape(pr)}</span></div>"
                f"<div class='ref'>{ref}</div>{rows}</div></div>")
        parts.append("<footer><div class='wrap'>runs/ip2p_mainline/sheets ·"
                     " 原生 512 不縮放 · 尚未進入 RESULTS.md，等裁定</div></footer>")
        f = args.out / f"figures_{args.purifier}_p{pi}.html"
        f.write_text("".join(parts), encoding="utf-8")
        mb = f.stat().st_size / 1048576
        written.append((f, mb, group))
        print(f"寫出 {f}（{mb:.1f} MB，條件 {group}）"
              + ("  **超過上限**" if mb > LIMIT_MB else ""))
    over = [w for w in written if w[1] > LIMIT_MB]
    if over:
        raise SystemExit(f"{len(over)} 頁超過 {LIMIT_MB} MB，調小 --per-page")


if __name__ == "__main__":
    main()
