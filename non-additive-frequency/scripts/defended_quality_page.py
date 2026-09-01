"""防禦圖的品質檢視頁：**原圖與防禦圖並排，附各品質的 JPEG 檔案大小。**

用途只有一個：用眼睛看防禦圖能不能接受，並看它在各個 JPEG 品質下壓得多小。
**不跑 GPU、不重算任何指標**，只讀已存的 PNG。

壓縮比要怎麼讀
────────────────────────────────────────────────────────────────────
擾動是高頻的，而 JPEG 的位元組數大致反映高頻內容的量，所以**防禦圖一定比
原圖大**——大多少就是擾動塞了多少進去。量化交付（`--deliver-jpeg QD`）的圖
已經被壓過一次，理論上在同一個品質上重壓會**接近它自己的檔案大小**，那正是
「它已經在量化格點上」的直接證據。

**頁面內嵌的預覽是 JPEG**（為了讓頁面開得動），所以**不要用預覽判斷擾動的
細節**；要看細節請開原生 512 的 PNG。表格裡的位元組數是對**原始 PNG** 量的，
不受預覽影響。

用法：
    python scripts/defended_quality_page.py \\
        --run 現行主線=runs/ip2p_axis_necessity/b_pg_r20 \\
              量化交付=runs/ip2p_deliver_jpeg/qd85 \\
        --out runs/ip2p_deliver_jpeg/quality.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUALITIES = (95, 85, 75, 50, 30)
PREVIEW = 384          # 頁面預覽的邊長。原圖是 512，看細節要開 PNG


def jpeg_bytes(im, q: int) -> int:
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=q, subsampling=0)
    return b.tell()


def preview(im) -> str:
    from PIL import Image
    t = im.resize((PREVIEW, PREVIEW), Image.LANCZOS)
    b = io.BytesIO()
    t.save(b, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="標籤=目錄")
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images13.txt"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run 要寫成 標籤=目錄，收到 {spec!r}")
        k, v = spec.split("=", 1)
        runs.append((k, Path(v)))
    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]

    def find(d: Path, name: str, kind: str):
        if kind == "orig":
            hits = sorted(d.glob(f"{name}__orig.png"))
        else:
            hits = [p for p in sorted(d.glob(f"{name}__*__def.png"))
                    if not p.name.endswith("__edit_def.png")]
        return hits[0] if hits else None

    parts = ["<meta charset='utf-8'><title>防禦圖品質</title><style>"
             "body{background:#808080;color:#111;font:14px/1.5 monospace;margin:0;padding:20px}"
             "table{border-collapse:collapse;background:#eee;margin:6px 0 22px}"
             "td,th{border:1px solid #999;padding:4px 9px;text-align:right}"
             "th{background:#ddd}td:first-child,th:first-child{text-align:left}"
             ".row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start}"
             "figure{margin:0}figcaption{font-size:12px;padding:3px 0}"
             "img{display:block;border:1px solid #555}"
             "h2{font-size:15px;margin:26px 0 6px;background:#eee;padding:5px 9px}"
             "</style>"]

    for i, name in enumerate(names, 1):
        op = find(runs[0][1], name, "orig")
        if op is None:
            parts.append(f"<h2>#{i:02d} {html.escape(name)} — 缺原圖</h2>")
            continue
        orig = Image.open(op).convert("RGB")
        cells = [f"<figure><img src='{preview(orig)}' width='{PREVIEW}'>"
                 f"<figcaption>原圖</figcaption></figure>"]
        rows = [("原圖", orig)]
        for label, d in runs:
            dp = find(d, name, "def")
            if dp is None:
                cells.append("<figure><div style='width:%dpx;height:%dpx;background:#444;"
                             "color:#ccc;display:flex;align-items:center;justify-content:center'>"
                             "缺</div><figcaption>%s</figcaption></figure>"
                             % (PREVIEW, PREVIEW, html.escape(label)))
                continue
            im = Image.open(dp).convert("RGB")
            cells.append(f"<figure><img src='{preview(im)}' width='{PREVIEW}'>"
                         f"<figcaption>{html.escape(label)}</figcaption></figure>")
            rows.append((label, im))

        head = "".join(f"<th>Q{q}</th>" for q in QUALITIES)
        body = []
        base = {q: jpeg_bytes(rows[0][1], q) for q in QUALITIES}
        for label, im in rows:
            tds = []
            for q in QUALITIES:
                n = jpeg_bytes(im, q)
                ratio = n / base[q]
                extra = "" if label == "原圖" else f"<br><span style='color:#a00'>{ratio:.2f}×</span>"
                tds.append(f"<td>{n / 1024:.0f} KB{extra}</td>")
            body.append(f"<tr><td>{html.escape(label)}</td>{''.join(tds)}</tr>")
        parts.append(f"<h2>#{i:02d} {html.escape(name)}</h2>"
                     f"<div class='row'>{''.join(cells)}</div>"
                     f"<table><tr><th>存成 JPEG 的大小</th>{head}</tr>"
                     f"{''.join(body)}</table>")
        print(f"{name} 完成", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts), encoding="utf-8")
    print(f"\n寫出 {args.out}（{args.out.stat().st_size / 1048576:.1f} MB）")


if __name__ == "__main__":
    main()
