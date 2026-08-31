"""把 `runs/ip2p_ig_loss` 的整條鏈做成一頁可以逐格對照的 HTML。

**在遠端跑**（影像在那裡），產出單一檔案 `report_ig_loss.html`，圖以 data URI
內嵌，不入版控（`CLAUDE.md` 的資料保全）。

一條鏈上有六個影像節點，缺任何一個都會讓讀者只能猜：

    原圖 ──防禦──▶ 防禦圖 ──淨化 T──▶ 淨化後的防禦圖 ──編輯──▶ 交出去的結果
      │                                                              ▲
      └──淨化 T──▶ 淨化後的原圖 ──編輯──▶ **空白地板**（要扣掉的那一項）─┘

主讀數是**扣掉地板的淨增益**：淨化算子自己就會把編輯推開（裁切 10% 的地板
實測 0.55），不扣掉它，「淨化後位移較大」無法排除「該算子本來就把編輯推得
比較開」這個平庸解釋。所以地板那一欄一定要並排放在旁邊，不是放在附錄。

**擋下與否用眼睛判「重畫」對「劣化」**：模型重畫成無關的場景、或整張變成
噪紋都算擋下；原圖仍認得出來、只是變糊變髒的單純劣化不算——攻擊方還是拿得到
可用的東西。SigLIP 代理已實測會把「人還在、只是被蓋上紋理」標成 blocked，
故本頁不放自動判定，只把圖擺出來。

殘差是有號小量，顯示時取 `|Δ|` 的通道平均再開根號提亮，**只為了看得見**，
不是量測；每一格的 RMS 與 L∞ 照實寫在標籤上。

用法（遠端）：
    python scripts/build_ig_loss_report.py --out report_ig_loss.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

SRC = Path("runs/ip2p_ig_loss")
GAL = SRC / "purify"
IMAGES = ["task_attr_mod_color_11699", "task_attr_mod_color_6205"]
IMAGE_LABEL = {
    "task_attr_mod_color_11699": "盆栽人（指令：turn the color of potted plant to pink）",
    "task_attr_mod_color_6205": "瑪利歐（指令：turn the color of toolbox to red）",
}
OPS = ["identity", "jpeg75", "jpeg30", "blur1", "blur2", "crop_resize0.1"]
OP_LABEL = {
    "identity": "未淨化", "jpeg75": "JPEG 75", "jpeg30": "JPEG 30",
    "blur1": "高斯模糊 σ=1", "blur2": "高斯模糊 σ=2",
    "crop_resize0.1": "裁切縮放 10%",
}
# 每一格的順序固定：先看防禦強度（全部條件、未淨化），再逐算子看整條鏈。
CONDS = ["ig_f08_eot", "ln_eot", "ig_r40_eot", "ig_blend_eot",
         "ig_eot", "ig_noeot", "ig_lot_eot", "ig_hit_eot"]
COND_NOTE = {
    "ig_f08_eot": "UNet 影像引導損失 ＋ 頻譜加性下限 0.08 ＋ 寬 EOT",
    "ln_eot": "舊損失 latent_norm ＋ 寬 EOT（等失真對照）",
    "ig_r40_eot": "UNet 損失，radius 4.0 ＋ 寬 EOT",
    "ig_blend_eot": "兩個損失等權相加 ＋ 寬 EOT",
    "ig_eot": "UNet 損失，radius 2.5 ＋ 寬 EOT",
    "ig_noeot": "UNet 損失，radius 2.5，**不放 EOT**",
    "ig_lot_eot": "UNet 損失，時間步窗 t∈[1,300] ＋ 寬 EOT",
    "ig_hit_eot": "UNet 損失，時間步窗 t∈[800,1000] ＋ 寬 EOT",
}
# 整條鏈只對這兩個條件展開——八個條件 × 六個算子會讓頁面沒有辦法讀。
CHAIN_CONDS = ["ig_f08_eot", "ln_eot"]

LPIPS_CEILING = 0.772          # runs/readout_ceiling/：45 對不相干影像的中位數
THUMB = 384


def uri(img: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality,
                            optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def thumb(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    if im.width > THUMB:
        im = im.resize((THUMB, THUMB), Image.LANCZOS)
    return uri(im)


def residual(orig: Path, defended: Path):
    """回傳 `(data-uri, rms, linf)`；顯示用開根號提亮，量測值照實回傳。"""
    if not (orig.exists() and defended.exists()):
        return None, None, None
    a = np.asarray(Image.open(orig).convert("RGB"), dtype=np.float64) / 255.0
    b = np.asarray(Image.open(defended).convert("RGB"), dtype=np.float64) / 255.0
    d = b - a
    rms = float(np.sqrt((d ** 2).mean()))
    linf = float(np.abs(d).max())
    m = np.abs(d).mean(axis=2)
    m = np.sqrt(m / max(m.max(), 1e-12))          # 只為了看得見，不是量測
    im = Image.fromarray((m * 255).astype("uint8"))
    if im.width > THUMB:
        im = im.resize((THUMB, THUMB), Image.LANCZOS)
    return uri(im), rms, linf


def read_defence_metrics() -> Dict[str, Dict[str, dict]]:
    """`{cond: {image: row}}`，來自各條件的 `results.csv`。"""
    out: Dict[str, Dict[str, dict]] = {}
    for cond in CONDS:
        p = SRC / cond / "results.csv"
        if not p.exists():
            continue
        out[cond] = {r["image"]: r for r in csv.DictReader(p.open(encoding="utf-8"))}
    return out


def read_effects() -> Dict[str, Dict[tuple, float]]:
    """`{cond_or_floor: {(image, op): effect_mean}}`。"""
    out: Dict[str, Dict[tuple, float]] = defaultdict(dict)
    for p in sorted(GAL.glob("*_all.csv")):
        tag = p.name[: -len("_all.csv")]
        for r in csv.DictReader(p.open(encoding="utf-8")):
            try:
                out[tag][(r["image"], r["purifier"])] = float(r["effect_mean"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("report_ig_loss.html"))
    args = ap.parse_args()

    dm = read_defence_metrics()
    eff = read_effects()
    floor = eff.get("floor", {})
    if not floor:
        raise SystemExit(f"{GAL} 沒有 floor_all.csv——扣地板不可省略")

    P: List[str] = []
    A = P.append

    A("<title>UNet 影像引導損失：防禦、淨化、編輯的逐格對照</title>")
    A("""<style>
:root{--bg:#fff;--fg:#16181d;--mut:#5b6472;--line:#e3e6ea;--card:#f7f8fa;
      --accent:#8a1c1c;--good:#0f5132}
:root:not([data-theme="light"]) {}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#111317;--fg:#e8eaed;--mut:#9aa4b2;--line:#2a2f38;--card:#181b21;
  --accent:#ff8a8a;--good:#7ee2b8}}
:root[data-theme="dark"]{--bg:#111317;--fg:#e8eaed;--mut:#9aa4b2;--line:#2a2f38;
  --card:#181b21;--accent:#ff8a8a;--good:#7ee2b8}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2rem 1.25rem 6rem;
  font:15px/1.65 -apple-system,"Segoe UI","Noto Sans TC",system-ui,sans-serif;
  max-width:1500px;margin-inline:auto}
h1{font-size:1.65rem;margin:0 0 .4rem;letter-spacing:-.01em}
h2{font-size:1.2rem;margin:3rem 0 .5rem;padding-top:1rem;
  border-top:1px solid var(--line)}
h3{font-size:1rem;margin:1.8rem 0 .5rem;color:var(--mut);font-weight:600}
p{max-width:78ch;color:var(--fg)}
.lede{color:var(--mut);max-width:78ch}
code{background:var(--card);padding:.1em .35em;border-radius:3px;font-size:.9em}
table{border-collapse:collapse;font-size:.86rem;margin:.8rem 0 1.2rem;
  display:block;overflow-x:auto;max-width:100%}
th,td{border:1px solid var(--line);padding:.35rem .6rem;text-align:right;
  white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--card);position:sticky;top:0}
tr.hl td{font-weight:700;color:var(--accent)}
.row{display:flex;gap:.6rem;overflow-x:auto;padding:.2rem 0 .6rem}
figure{margin:0;flex:0 0 auto;width:200px}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
  border-radius:4px;background:var(--card)}
figcaption{font-size:.72rem;color:var(--mut);margin-top:.3rem;line-height:1.35}
figcaption b{color:var(--fg)}
.chain{border:1px solid var(--line);border-radius:6px;padding:.7rem .8rem;
  margin:.8rem 0;background:var(--card)}
.chain h4{margin:0 0 .3rem;font-size:.92rem}
.meta{font-size:.78rem;color:var(--mut);margin:0 0 .5rem}
.gain{font-weight:700;color:var(--good)}
.note{border-left:3px solid var(--accent);padding:.15rem 0 .15rem .8rem;
  color:var(--mut);margin:1rem 0;max-width:78ch}
</style>""")

    A("<h1>UNet 影像引導損失：防禦、淨化、編輯的逐格對照</h1>")
    A('<p class="lede">兩張影像、八個防禦條件、六個淨化算子。'
      '主讀數是<b>扣掉空白地板</b>的位移淨增益——淨化算子自己就會把編輯推開，'
      '不扣掉它，「淨化後位移較大」無法排除「該算子本來就把編輯推得比較開」'
      '這個平庸解釋。地板因此與每一格並排，不放附錄。</p>')
    A('<div class="note"><b>擋下與否請用眼睛判。</b>'
      '模型重畫成無關的場景、或整張變成噪紋都算擋下；原圖仍認得出來、只是'
      '變糊變髒的<b>單純劣化不算</b>——攻擊方還是拿得到可用的東西。'
      'SigLIP 代理已實測會把「人還在、只是被蓋上紋理」標成 blocked，'
      '故本頁不放自動判定。</div>')

    # ---------------------------------------------------------- 數值總表
    A("<h2>一、扣地板的淨增益</h2>")
    A(f'<p class="lede">下表第二段是「佔可達範圍的比例」。主讀數 LPIPS 在兩張'
      f'不相干的自然影像之間會飽和，十張兩兩配對（45 對）的中位數是 '
      f'<b>{LPIPS_CEILING}</b>，所以每一欄的可達範圍是 '
      f'<code>{LPIPS_CEILING} − 地板</code>，不是 {LPIPS_CEILING}。'
      f'同一個絕對淨增益放在地板 0.11 的欄與地板 0.55 的欄，意義差很多。</p>')

    fm = {op: st.mean([v for (i, o), v in floor.items() if o == op] or [0.0])
          for op in OPS}
    for mode in ("abs", "frac"):
        A("<h3>" + ("絕對淨增益" if mode == "abs" else "佔可達範圍的比例")
          + "</h3><table><thead><tr><th>條件</th><th>DISTS</th>"
          + "".join(f"<th>{OP_LABEL[o]}</th>" for o in OPS)
          + "</tr></thead><tbody>")
        for cond in CONDS:
            if cond not in eff:
                continue
            d = dm.get(cond, {})
            dists = st.mean([float(r["fid_dists"]) for r in d.values()]) if d else None
            cells = []
            for op in OPS:
                diffs = [v - floor[(i, op)] for (i, o), v in eff[cond].items()
                         if o == op and (i, op) in floor]
                if not diffs:
                    cells.append("<td>—</td>")
                    continue
                g = st.mean(diffs)
                room = LPIPS_CEILING - fm[op]
                cells.append(f"<td>{g:.4f}</td>" if mode == "abs"
                             else f"<td>{100 * g / room:.1f}%</td>")
            klass = ' class="hl"' if cond == "ig_f08_eot" else ""
            A(f"<tr{klass}><td>{cond}</td><td>{fmt(dists)}</td>"
              + "".join(cells) + "</tr>")
        A("<tr><td>空白地板（絕對位移）</td><td>—</td>"
          + "".join(f"<td>{fm[o]:.4f}</td>" for o in OPS) + "</tr>")
        A("<tr><td>可達範圍</td><td>—</td>"
          + "".join(f"<td>{LPIPS_CEILING - fm[o]:.4f}</td>" for o in OPS)
          + "</tr></tbody></table>")

    # ------------------------------------------------- 防禦圖本身（未淨化）
    A("<h2>二、防禦圖本身：原圖、交付圖、殘差</h2>")
    A('<p class="lede">殘差取 <code>|Δ|</code> 的通道平均再開根號提亮，'
      '<b>只為了看得見</b>，不是量測；每一格的 RMS 與 L∞ 是實際值。</p>')
    for img in IMAGES:
        A(f"<h3>{html.escape(IMAGE_LABEL[img])}</h3>")
        A('<div class="row">')
        o = SRC / CONDS[0] / f"{img}__orig.png"
        t = thumb(o)
        if t:
            A(f'<figure><img src="{t}" alt="原圖"><figcaption><b>原圖</b>'
              "</figcaption></figure>")
        for cond in CONDS:
            dp = SRC / cond / f"{img}__phase_gain__def.png"
            t = thumb(dp)
            if not t:
                continue
            r = dm.get(cond, {}).get(img, {})
            A(f'<figure><img src="{t}" alt="{cond}"><figcaption>'
              f'<b>{cond}</b><br>{html.escape(COND_NOTE.get(cond, ""))}<br>'
              f'DISTS {fmt(float(r["fid_dists"]) if r else None)}・'
              f'PSNR {fmt(float(r["fid_psnr"]) if r else None, 2)}'
              "</figcaption></figure>")
        A("</div>")
        A('<div class="row">')
        for cond in CONDS:
            u, rms, linf = residual(SRC / cond / f"{img}__orig.png",
                                    SRC / cond / f"{img}__phase_gain__def.png")
            if not u:
                continue
            A(f'<figure><img src="{u}" alt="殘差"><figcaption>'
              f'<b>殘差 {cond}</b><br>RMS {fmt(rms)}・L∞ {fmt(linf, 3)}'
              "</figcaption></figure>")
        A("</div>")

    # --------------------------------------------------------- 整條鏈
    A("<h2>三、整條鏈：淨化前後與編輯前後</h2>")
    A('<p class="lede">每一列六格：淨化後的原圖、淨化後的防禦圖、'
      '編輯(原圖)、<b>編輯(淨化後的原圖)＝空白地板</b>、'
      '編輯(淨化後的防禦圖)＝交出去的結果。'
      '要判的是最後一格相對於<b>地板那一格</b>有沒有多擋下東西。</p>')
    for img in IMAGES:
        A(f"<h3>{html.escape(IMAGE_LABEL[img])}</h3>")
        for cond in CHAIN_CONDS:
            A('<div class="chain">')
            A(f"<h4>{cond}　<span style='font-weight:400'>"
              f"{html.escape(COND_NOTE.get(cond, ''))}</span></h4>")
            for op in OPS:
                g = None
                if (img, op) in eff.get(cond, {}) and (img, op) in floor:
                    g = eff[cond][(img, op)] - floor[(img, op)]
                room = LPIPS_CEILING - fm[op]
                pct = f"（佔可達 {100 * g / room:.1f}%）" if g is not None else ""
                A(f'<p class="meta">{OP_LABEL[op]}　'
                  f'淨增益 <span class="gain">{fmt(g)}</span> {pct}　'
                  f'地板 {fmt(fm[op])}</p>')
                A('<div class="row">')
                panels = [
                    (GAL / "gallery_floor" / f"{img}__None__{op}__pur.png",
                     "淨化後的原圖"),
                    (GAL / f"gallery_{cond}" / f"{img}__phase_gain__{op}__pur.png",
                     "淨化後的防禦圖"),
                    (SRC / cond / f"{img}__phase_gain__edit_orig.png",
                     "編輯（原圖，未淨化）"),
                    (GAL / "gallery_floor" / f"{img}__None__{op}__edit_def.png",
                     "編輯（淨化後的原圖）＝<b>空白地板</b>"),
                    (GAL / f"gallery_{cond}" / f"{img}__phase_gain__{op}__edit_def.png",
                     "編輯（淨化後的防禦圖）＝<b>交出去的結果</b>"),
                ]
                for path, cap in panels:
                    t = thumb(path)
                    if t:
                        A(f'<figure><img src="{t}" alt="{cap}">'
                          f"<figcaption>{cap}</figcaption></figure>")
                A("</div>")
            A("</div>")

    A('<p class="lede" style="margin-top:3rem">'
      '兩張影像的規模只夠當探針。數值來源：'
      '<code>runs/ip2p_ig_loss/*/results.csv</code>（防禦端）與 '
      '<code>runs/ip2p_ig_loss/purify/*_all.csv</code>（抗淨化，3 seed）。'
      'LPIPS 天花板 0.772 出自 <code>runs/readout_ceiling/</code>。</p>')

    args.out.write_text("\n".join(P), encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    print(f"寫出 {args.out}（{mb:.1f} MB）")


if __name__ == "__main__":
    main()
