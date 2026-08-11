#!/usr/bin/env python
"""注意力並排頁：把各條件的聚合注意力圖放在同一列，並附 `attn_stats.csv` 的量。

    python scripts/attn_compare.py --out runs/attn_compare.html \
        --col "未防禦=runs/s3t20_merged/control" \
        --col "apa+A=runs/s3t20_r_merged/apa" ...

## 這一頁要回答的問題

`apa`（`suppress_attn_ca`）最小化 `‖Att ⊙ M‖₁`，而那個量**有兩種降法**：
把質量移出遮罩，或把整張圖的注意力一起壓平。兩者在損失上不可區分，但對防禦
的意義完全不同——攤平是去噪鏈後續步數會自行補回來的擾動。`apa_rd`
（`redirect_attn_ca`）改取比例正是為了排除後者。

**哪一種真的發生了，只有看圖與看 `content_mass` 的分佈才判得出來。**
故本頁並排聚合圖，並列出 `attn_stats.csv` 的 `content_mass_mean`（該詞在全圖
拿到的平均質量）與 `entropy`（分佈的集中程度）：

- 質量下降、熵**不變** → 移走了（分佈仍集中，只是集中在別處）
- 質量下降、熵**上升** → 攤平了（分佈變均勻）

兩者的差別就是這個 arm 的全部假設。
"""

import argparse
import base64
import csv
import io
import statistics as st
from pathlib import Path

from PIL import Image

IMAGES = ["horse_00", "horse_03", "woman_03"]


def b64(path: Path, quality: int = 92, side: int = 256) -> str:
    """注意力圖是 64×64 的格點，放大用 NEAREST。

    雙線性會在格點之間內插出實際不存在的中間值，而這一頁要判的正是「質量
    落在哪幾格」——平滑掉格線等於把判斷依據抹掉。
    """
    with Image.open(path) as im:
        im = im.convert("RGB").resize((side, side), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def stats(d: Path):
    """`attn_stats.csv` 的兩個量，對全部層與 timestep 取平均。"""
    f = d / "attn_stats.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(f.open(encoding="utf-8")))
    if not rows:
        return None
    return (st.fmean(float(r["content_mass_mean"]) for r in rows),
            st.fmean(float(r["entropy"]) for r in rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--col", action="append", required=True,
                    help="`名稱=條件目錄`，可給多次；順序即欄序。"
                        "**名稱裡不可有 `=`**——切分取第一個等號，"
                        "「未防禦 φ=0」會被切成名稱「未防禦 φ」與路徑「0=…」，"
                        "而症狀只是那一欄整排顯示缺圖")
    ap.add_argument("--purify", default="identity_0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    cols = []
    for c in a.col:
        name, _, root = c.partition("=")
        if not root or not Path(root).is_dir():
            raise SystemExit(
                f"--col {c!r} 解析出的目錄 {root!r} 不存在。名稱裡若有 `=` "
                "（例如「φ=0」）會被切在錯的位置，請改寫名稱")
        cols.append((name, root))
    h = ["<title>注意力並排 · 移走還是攤平</title>", """<style>
body{font:14px/1.5 system-ui,sans-serif;margin:24px}
table{border-collapse:collapse;margin:8px 0 26px}
td,th{border:1px solid #ddd;padding:5px;text-align:center;vertical-align:top}
img{width:230px;height:230px;object-fit:contain;display:block}
.n{font-size:11px;color:#666;font-family:ui-monospace,monospace}
.miss{width:230px;height:230px;display:flex;align-items:center;
      justify-content:center;color:#b00;background:#faf0f0;font-size:12px}
th{background:#f5f5f5;font-size:12px}h2{margin:28px 0 4px}
p.note{color:#555;max-width:1050px}
</style>"""]
    h.append("<h1>注意力聚合圖：質量是被移走還是被攤平</h1>")
    h.append("""<p class="note">每一格是 Lo et al. 式 (3) 的跨層聚合注意力圖
（c_a 那幾個 token 的質量分佈，亮處代表模型認為那個詞在這裡）。
<code>mass</code> 是全圖平均質量、<code>H</code> 是分佈的熵。<br>
<b>質量降而熵不變 → 移走了；質量降而熵升 → 攤平了。</b>
攤平是去噪鏈後續步數會自行補回來的擾動，移走才是有方向的改動。</p>""")

    for img in IMAGES:
        h.append(f"<h2>{img}</h2><table><tr>"
                 + "".join(f"<th>{n}</th>" for n, _ in cols) + "</tr><tr>")
        base = None
        for name, root in cols:
            d = Path(root) / img / "purify" / a.purify / "attn"
            png = next((p for p in (d / f"seed{a.seed}_agg.png",
                                    d / f"tau0.04_seed{a.seed}_agg.png")
                        if p.exists()), None)
            s = stats(d)
            if s and base is None:
                base = s
            note = ""
            if s:
                dm = f"{(s[0] / base[0] - 1) * 100:+.0f}%" if base and base[0] else ""
                dh = f"{(s[1] / base[1] - 1) * 100:+.0f}%" if base and base[1] else ""
                note = f"mass {s[0]:.4f} {dm}<br>H {s[1]:.3f} {dh}"
            if png is None:
                h.append(f'<td><div class="miss">缺 {a.purify}</div></td>')
            else:
                h.append(f'<td><img src="data:image/jpeg;base64,{b64(png)}">'
                         f'<div class="n">{note}</div></td>')
        h.append("</tr></table>")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(h), encoding="utf-8")
    print(f"寫入 {a.out}（{a.out.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
