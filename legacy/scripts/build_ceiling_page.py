import base64
import io
import json
import os
from pathlib import Path

from PIL import Image

R = Path(r"C:\WACV-s3\runs")
SW = R / "suppress_render"
LO = R / "lo_semantic_s3"
sweep = json.load(open(SW / "sweep.json", encoding="utf-8"))

# 完整掃描的數字（sweep.log），逐影像 k → (抑制, lpips, 銳利度)
FULL = {
    "horse_00": [(1, 9.9, .1175, .956), (2, 16.1, .1780, 1.029),
                 (4, 24.5, .3149, 1.341), (8, 29.6, .5034, 2.567),
                 (16, 38.0, .6236, 5.969), (32, 31.3, .6858, 10.693),
                 (64, 23.9, .7555, 13.700)],
    "horse_03": [(1, 11.1, .1629, .930), (2, 19.3, .2521, 1.007),
                 (4, 31.1, .4036, 1.377), (8, 33.8, .5648, 2.463),
                 (16, 33.1, .6688, 3.928), (32, 42.6, .7268, 4.694),
                 (64, 43.8, .7778, 4.805)],
    "woman_03": [(1, 7.3, .1188, .929), (2, 12.0, .2464, .983),
                 (4, 12.1, .3931, 1.257), (8, 12.9, .5029, 2.539),
                 (16, 13.3, .5682, 6.034), (32, 10.8, .6198, 11.603),
                 (64, 4.5, .6813, 15.812)],
}
DAYN = {"horse_00": (90, .5944, .5599), "horse_03": (89, .5189, .5603),
        "woman_03": (70, .5287, .5051)}


def b64(p, side=210):
    with Image.open(p) as im:
        im = im.convert("RGB").resize((side, side), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def cell(p, title, note=""):
    if not Path(p).exists():
        return f'<td><div class=miss>缺</div><div class=t>{title}</div></td>'
    return (f'<td><img src="{b64(p)}"><div class=t>{title}</div>'
            f'<div class=n>{note}</div></td>')


h = ["<title>注意力抑制的天花板 · 參數化而非預算</title>", """<style>
body{font:15px/1.65 system-ui,-apple-system,"Noto Sans TC",sans-serif;
     margin:26px;max-width:1500px;background:#fff;color:#14171C}
table{border-collapse:collapse;margin:8px 0 10px}
td,th{border:1px solid #D8DCE2;padding:6px;text-align:center;vertical-align:top}
th{background:#F2F3F5;font-size:12px}
img{display:block}.t{font-size:12px;margin-top:5px}
.n{font-size:11px;color:#5B6472;font-family:ui-monospace,monospace}
.miss{width:210px;height:210px;background:#F6E0DF;color:#A8231F}
h2{margin:34px 0 4px;font-size:18px}h1{font-size:22px}
p.d{color:#3a4250;margin:4px 0 10px;max-width:1250px}
.bad{color:#A8231F;font-weight:600}.good{color:#1F7A4C;font-weight:600}
.curve{border:1px solid #D8DCE2;padding:10px;margin:10px 0;border-radius:6px}
</style>"""]
h.append("<h1>把 apa_pj 的 φ 放大到 DAYN 的抑制水準會怎樣</h1>")
h.append("""<p class="d"><b>不需要重新訓練</b>——射線縮放只是把方向參數乘上 k。
掃 k = 1…64，逐點量遮罩內的 c_a 注意力 L1（與訓練期同一個遮罩、同一個 c_a
嵌入、同一組 timestep），再渲染防禦圖與攻擊方的編輯。
<b>結論：抑制到不了 70–90%，而且對 k 不是單調的——先升後降。</b></p>""")

# ---- 曲線表 ----
h.append("<h2>① 抑制隨 k 的變化（DAYN 在同一組影像上是 70–90%）</h2>")
h.append('<div class="curve"><table><tr><th>影像</th>'
         + "".join(f"<th>k={k}</th>" for k in (1, 2, 4, 8, 16, 32, 64))
         + "<th>峰值</th><th>DAYN（像素+L∞）</th></tr>")
for img, rows in FULL.items():
    peak = max(rows, key=lambda r: r[1])
    h.append(f"<tr><td><b>{img}</b></td>")
    for k, sup, lp, ac in rows:
        cls = ' class="good"' if (k, sup) == (peak[0], peak[1]) else ""
        h.append(f'<td{cls}>{sup:.1f}%<div class=n>lpips {lp:.2f}<br>'
                 f'銳利 {ac:.1f}</div></td>')
    d = DAYN[img]
    h.append(f'<td class="bad">{peak[1]:.1f}%<div class=n>k={peak[0]}</div></td>'
             f'<td class="good">{d[0]}%<div class=n>pert_lpips {d[1]:.2f}<br>'
             f'edit_lpips {d[2]:.2f}</div></td></tr>')
h.append("</table></div>")
h.append("""<p class="d">woman_03 最極端：<b>峰值只有 13.3%</b>，而且 k=64 時退回
4.5%。機制是：k 大到一定程度之後解碼出來的是噪聲，注意力隨之攤成均勻分佈，
遮罩區反而<b>拿回它按面積該得的那一份</b>——所以壓不下去還會回升。</p>""")

# ---- 影像對照 ----
h.append("<h2>② 同一張圖，三個 k 的防禦圖與編輯結果</h2>")
h.append('<p class="d">最右兩欄是 DAYN 的忠實重現（像素 δ + L∞ ≤ 0.06 + 同一條式 (5)）。'
         '<b>注意它與 k=16 的 LPIPS 相當（0.59 對 0.62），但圖完全不同</b>。</p>')
for img in ("horse_00", "horse_03", "woman_03"):
    rows = {k: r for k, *r in [(r[0], r[1], r[2], r[3]) for r in FULL[img]]}
    h.append(f"<h3>{img}</h3><table><tr>"
             "<th>原圖</th><th>k=1（工作點 Δ=0.04）</th><th>k=4</th>"
             "<th>k=16（峰值附近）</th><th>DAYN 的防禦圖</th></tr><tr>")
    h.append(cell(SW / f"{img}__orig.png", "x", ""))
    for k in (1, 4, 16):
        sup, lp, ac = rows[k]
        h.append(cell(SW / f"{img}__xdef_k{k}.png", f"x_def · k={k}",
                      f"抑制 {sup:.1f}% · lpips {lp:.2f} · 銳利 {ac:.1f}"))
    d = DAYN[img]
    h.append(cell(LO / f"{img}__semantic__adv.png", "x_adv · DAYN",
                  f"抑制 {d[0]}% · lpips {d[1]:.2f}"))
    h.append("</tr><tr>")
    h.append(cell(SW / f"{img}__edit_undefended.png", "未防禦的編輯", "判準基線"))
    for k in (1, 4, 16):
        h.append(cell(SW / f"{img}__edit_k{k}.png", f"編輯 · k={k}", ""))
    h.append(cell(LO / f"{img}__semantic_edit_def.png", "編輯 · DAYN",
                  f"edit_lpips {d[2]:.2f}"))
    h.append("</tr></table>")

h.append("""<h2>③ 這回答了「為什麼 attn loss 對我們沒效」</h2>
<p class="d">三層原因，最後一層是今天才確立的：</p>
<ol>
<li><b>不是損失的問題。</b>同一條式 (5)，在像素參數化上壓掉 70–90%，
<code>edit_lpips</code> 0.51–0.56。</li>
<li><b>不只是預算的問題。</b>先前的解釋是「DAYN 花 pert_lpips 0.53、我們花
0.13」。但把我們的 φ 放大到 <b>LPIPS 0.62</b>（與 DAYN 相當）之後，抑制仍然
只有 13–44%。</li>
<li class="bad"><b>是參數化的天花板。</b>latent 上的低秩方向<b>在任何量級上都
達不到</b>那個抑制水準，而且過了峰值還會回落。</li>
</ol>
<p class="d">同時它揭露一件對整個專案更要緊的事：
<b>同一個 LPIPS 在兩個參數化上是兩種東西</b>。DAYN 在 lpips 0.59 是一層細網紋
覆蓋、馬清晰可辨；我們在 lpips 0.62 是結構被打散成色塊。
用 LPIPS／DISTS 當跨參數化的共同預算，本身就低估了非加性的可見破壞——
這與先前記錄的「LPIPS 對非加性是寬鬆而非嚴苛」是同一件事的極端形式。</p>""")

out = R / "suppression_ceiling.html"
out.write_text("\n".join(h), encoding="utf-8")
print(f"寫入 {out}（{os.path.getsize(out)/1e6:.1f} MB）")
