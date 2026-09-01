"""三節報告頁：相位方法、位移場、淨化對相位偏移做了什麼。

**不跑 GPU。** 讀 `report_data.json`（`collect_report_data.py` 產生）、
`_rep/fig/` 底下的圖（`report_figures.py`／`phase_shift_heatmap.py` 產生）、
以及 `_rep/runs/` 底下抓回來的 PNG。輸出單一 HTML，圖一律以 data URI 內嵌，
不入版控（`CLAUDE.md` 資料保全）。

**本頁只擺數據與圖，不下判斷。** 沒有「成立／不成立」「值得／不值得」的結論
（`CLAUDE.md`：訓練方法的效果由使用者判斷）。

讀數的規定
────────────────────────────────────────────────────────────────────
一律報兩個**絕對值**：

    總增益 = effect(p)                   位移本身
    淨增益 = effect(p) − 空白地板         算子自己造成的位移已扣掉

比例讀數（保留率、佔可達範圍）在分母塌陷或參照改變時不可解讀，不出現在本頁。
LPIPS 在兩張不相干的自然影像之間飽和於 0.772（`runs/readout_ceiling/`），
只作為飽和參考，不進任何算式。

**幾何欄（裁切縮放、JPEG→重取樣）用的是舊參照**：`LPIPS(編輯(原圖),
編輯(p(防禦圖)))`、地板非 0。現行協定對幾何類改取「同一個算子淨化過的原圖」
為參照、地板由構造為 0。兩者基準不同，頁面上逐處標註，不代為換算。

用法：python scripts/build_full_report.py --out report_full.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

REP = Path("_rep/runs")
FIG = Path("_rep/fig")
IMAGES = [("task_attr_mod_color_11699", "盆栽人",
           "turn the color of potted plant to pink"),
          ("task_attr_mod_color_6205", "瑪利歐",
           "turn the color of toolbox to red")]
OPS6 = ["identity", "jpeg75", "jpeg30", "blur1", "blur2", "crop_resize0.1"]
OPL6 = {"identity": "未淨化", "jpeg75": "JPEG 75", "jpeg30": "JPEG 30",
        "blur1": "模糊 σ=1", "blur2": "模糊 σ=2",
        "crop_resize0.1": "裁切縮放 10%"}
OPSJ = ["identity", "jpeg90", "jpeg75", "jpeg50", "jpeg30"]
OPLJ = {"identity": "未淨化", "jpeg90": "JPEG 90", "jpeg75": "JPEG 75",
        "jpeg50": "JPEG 50", "jpeg30": "JPEG 30"}
OPS8 = ["identity", "jpeg75", "jpeg30", "blur1", "crop_resize0.1",
        "jpeg_then_resize75", "gridpure", "adverse_cleaner"]
OPL8 = {"identity": "未淨化", "jpeg75": "JPEG 75", "jpeg30": "JPEG 30",
        "blur1": "模糊 σ=1", "crop_resize0.1": "裁切縮放 10%",
        "jpeg_then_resize75": "JPEG75→重取樣", "gridpure": "GrIDPure",
        "adverse_cleaner": "AdverseCleaner"}
GEOM = {"crop_resize0.1", "jpeg_then_resize75"}


def uri(path: Path, width: int | None = None, quality: int = 88) -> str:
    """PNG → JPEG data URI。原圖 512² 一張約 400 kB，88 張會把頁面撐到百 MB。"""
    im = Image.open(path).convert("RGB")
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def png_uri(path: Path) -> str:
    """圖表照原樣內嵌成 PNG——重壓成 JPEG 會讓細字與格線糊掉。"""
    return ("data:image/png;base64,"
            + base64.b64encode(path.read_bytes()).decode())


def fig_block(path: Path, cap: str) -> str:
    return (f'<figure class="chart"><img src="{png_uri(path)}" alt="">'
            f'<figcaption>{cap}</figcaption></figure>')


def cell(path: Path, cap: str, w: int = 300) -> str:
    if not path.exists():
        return f'<figure class="miss"><div>缺 {html.escape(path.name)}</div></figure>'
    return (f'<figure><img src="{uri(path, w)}" alt="">'
            f'<figcaption>{html.escape(cap)}</figcaption></figure>')


def fmt(v, n=4):
    return "—" if v is None else f"{v:.{n}f}"


def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                for r in rows)
    return (f'<table class="{cls}"><thead><tr>{h}</tr></thead>'
            f"<tbody>{b}</tbody></table>")


def geom_cell(op: str, v) -> str:
    """幾何欄用不同的參照量的，上色標出來，不與其餘欄並列閱讀。"""
    return f'<span class="geom">{fmt(v)}</span>' if op in GEOM else fmt(v)


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#dcdcdc;--head:#f4f4f6;
      --warn:#8a5a00;--warnbg:#fff8e6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.65 "Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif}
.wrap{max-width:1680px;margin:0 auto;padding:28px 26px 80px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:.4px}
h2{font-size:21px;margin:52px 0 6px;padding-bottom:7px;
   border-bottom:2px solid var(--fg)}
h3{font-size:16.5px;margin:30px 0 8px;color:#333}
h4{font-size:14.5px;margin:20px 0 6px;color:#444;font-weight:600}
.sub{color:var(--mut);font-size:13.5px;margin:0 0 16px}
.note{background:var(--warnbg);border-left:3px solid var(--warn);
      color:var(--warn);padding:9px 13px;margin:12px 0;font-size:13px;
      border-radius:0 3px 3px 0}
table{border-collapse:collapse;margin:12px 0 6px;font-size:13px;
      font-variant-numeric:tabular-nums;width:100%}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:right}
th:first-child,td:first-child{text-align:left;white-space:nowrap}
thead th{background:var(--head);font-weight:600;text-align:center}
tbody tr:nth-child(even){background:#fafafa}
tbody tr.hi{background:#e8f2fd}
tbody tr.hi td{font-weight:600}
.geom{color:#8a5a00}
.scroll{overflow-x:auto}
figure{margin:0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
           border-radius:3px}
figcaption{font-size:11.5px;color:var(--mut);margin-top:4px;line-height:1.4}
.chart{margin:16px 0 26px}
.grid{display:grid;gap:11px;margin:10px 0 20px}
.g4{grid-template-columns:repeat(4,1fr)}
.g5{grid-template-columns:repeat(5,1fr)}
.g6{grid-template-columns:repeat(6,1fr)}
.miss{border:1px dashed var(--line);border-radius:3px;padding:26px 8px;
      color:var(--mut);font-size:11px;text-align:center;background:#fbfbfb;
      display:flex;align-items:center;justify-content:center}
.toc{background:#fafafa;border:1px solid var(--line);border-radius:4px;
     padding:12px 18px;margin:18px 0 4px;font-size:14px}
.toc a{color:#0b57d0;text-decoration:none;margin-right:20px}
.toc a:hover{text-decoration:underline}
code{background:#f2f2f4;padding:1px 5px;border-radius:3px;font-size:12.5px}
.dl{font-size:12.5px;color:var(--mut);margin:4px 0 14px}
"""


def flatten(D, fam) -> dict:
    out = {}
    for b, v in D[fam].items():
        for t, r in v["defence"].items():
            out[t] = dict(r, batch=b, purify=v["purify"].get(t, {}))
    return out


def defence_table(P, order, mark=()) -> str:
    rows = []
    for t in order:
        r = P[t]
        rows.append([
            t, r.get("batch", "—"), fmt(r["fid_dists"]), fmt(r.get("fid_lpips")),
            fmt(r.get("fid_psnr"), 2), fmt(r.get("fid_ssim")),
            fmt(r.get("fid_rms")), fmt(r.get("fid_linf"), 3),
            fmt(r["edit_lpips"]), r.get("stop_reason", "—"),
            str(r.get("stopped_at", "—"))])
    h = ["條件", "批次", "DISTS↓", "LPIPS↓", "PSNR↑", "SSIM↑", "RMS↓", "L∞↓",
         "未淨化位移↑", "停止原因", "停在第幾步"]
    body = ""
    for t, r in zip(order, rows):
        c = ' class="hi"' if t in mark else ""
        body += f"<tr{c}>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>"
    head = "".join(f"<th>{c}</th>" for c in h)
    return ('<div class="scroll"><table><thead><tr>' + head
            + "</tr></thead><tbody>" + body + "</tbody></table></div>")


def purify_table(P, tags, ops, oplab, key, mark=()) -> str:
    body = ""
    for t in tags:
        c = ' class="hi"' if t in mark else ""
        cs = [geom_cell(op, P[t]["purify"].get(op, {}).get(key)) for op in ops]
        body += (f"<tr{c}><td>{t}</td><td>{fmt(P[t]['fid_dists'])}</td>"
                 + "".join(f"<td>{x}</td>" for x in cs) + "</tr>")
    fl = P[tags[0]]["purify"]
    body += ("<tr><td><i>空白地板</i></td><td>—</td>"
             + "".join(f"<td><i>{fmt(fl.get(op, {}).get('floor'))}</i></td>"
                       for op in ops) + "</tr>")
    head = "".join(f"<th>{c}</th>" for c in ["條件", "DISTS"]
                   + [oplab[op] for op in ops])
    return ('<div class="scroll"><table><thead><tr>' + head
            + "</tr></thead><tbody>" + body + "</tbody></table></div>")


GEOM_NOTE = ('<div class="note">帶顏色的那幾欄（裁切縮放、JPEG→重取樣）'
             '用的是舊參照：<code>LPIPS(編輯(原圖), 編輯(p(防禦圖)))</code>，'
             '空白地板非 0。現行協定對幾何類算子改取「同一個算子淨化過的原圖」'
             '為參照、地板由構造為 0。兩者基準不同，該欄不可與其餘欄並列。</div>')


# 三組等失真配對取自 `runs/ip2p_mainline`：本方法與 DCT-Shield 在**同一批**
# 裡跑，同兩張影像、同一組算子、同一個空白地板，所以可以直接並列。
PAIRS_JPEG = [("低失真", "ours_pg_n", "dct_native"),
              ("中失真", "ours_ph_q", "dct_aj50_eps0.22"),
              ("中高失真", "ours_pg_q", "dct_aj85")]
COND = {"ours_pg_n": "phase_gain", "ours_ph_q": "phase",
        "ours_pg_q": "phase_gain", "dct_native": "dct_shield",
        "dct_aj50_eps0.22": "dct_shield_y", "dct_aj85": "dct_shield_y"}
H2H_DIST = {"ours_nonadd": 0.1197, "ours_add": 0.1383, "dct_e18": 0.0978}

KNOBS = [
    ("損失", "<code>image_guidance</code>（UNet 影像引導）",
     "換掉 <code>latent_norm</code> 之後，JPEG 75 的淨增益 +0.164、"
     "JPEG 30 +0.110"),
    ("頻率閘的定價", "<code>jpeg_luma</code>，指數 0.25",
     "二值帶通對 r=0.15 與 r=0.9 開同一個價；人眼對前者的敏感度高一個數量級"),
    ("重疊步長 hop", "8",
     "決定殘差的紋理有多粗。等 radius 下失真更低、PSNR 更高、位移更高；"
     "hop 4 沒有再進步"),
    ("可學幅度增益", "開啟（<code>gain_ratio = 1.0</code>）",
     "相位是週期量，θ 加到 π 就繞回去；增益是唯一能拆掉這個天花板的旋鈕"),
    ("加性下限", "<code>spectral_floor</code> 0.04",
     "純乘性在平坦區動不了（|spec| ≈ 0 乘什麼都接近零）。"
     "加性項是唯一能自由選擇放在哪裡的那一半"),
    ("寬 EOT", "identity / JPEG / blur / crop",
     "把可微分的良性算子放進最佳化迴圈"),
]


def pair_table(B, pur, ops, oplab) -> str:
    """一組配對兩列，只放淨增益。"""
    body = ""
    for lab, a, b in PAIRS_JPEG:
        if a not in pur or b not in pur:
            continue
        for t, cls in ((a, ' class="hi"'), (b, "")):
            cs = [geom_cell(op, pur[t].get(op, {}).get("net")) for op in ops]
            body += (f"<tr{cls}><td>{lab if t == a else ''}</td><td>{t}</td>"
                     f"<td>{fmt(B[t]['fid_dists'])}</td>"
                     f"<td>{fmt(B[t]['fid_psnr'], 2)}</td>"
                     + "".join(f"<td>{x}</td>" for x in cs) + "</tr>")
    first = next(t for _, t, _ in PAIRS_JPEG if t in pur)
    body += ('<tr><td></td><td><i>空白地板</i></td><td>—</td><td>—</td>'
             + "".join(f"<td><i>{fmt(pur[first].get(op, {}).get('floor'))}</i></td>"
                       for op in ops) + "</tr>")
    head = "".join(f"<th>{c}</th>" for c in ["配對", "條件", "DISTS↓", "PSNR↑"]
                   + [oplab[op] for op in ops])
    return ('<div class="scroll"><table><thead><tr>' + head
            + "</tr></thead><tbody>" + body + "</tbody></table></div>")


def sec_method(D) -> str:
    M = D["baseline"]["ip2p_mainline"]
    B, pur = M["defence"], M["purify"]
    H = D["baseline"].get("ip2p_purify_headtohead", {}).get("purify", {})
    MP = D["baseline"].get("ip2p_matched_headtohead", {}).get("purify", {})

    o = ['<h2 id="s1">一、相位方法（紋理重相位）</h2>',
         '<p class="sub">程式：<code>src/residual/texture_rephase.py</code>（算子）、'
         '<code>src/defense/param_pgd.py</code>（參數化與 PGD 迴圈）。'
         '兩張影像：盆栽人 <code>task_attr_mod_color_11699</code>、'
         '瑪利歐 <code>task_attr_mod_color_6205</code>。</p>']

    o.append("<h3>1.1　構造</h3>")
    o.append(svg_phase_pipeline())
    o.append('<p class="sub">純相位（不含加性項）時整條路徑收斂成一式：'
             '　x_def = OLA( irfft2( rfft2(w·P_b) · exp(i·g_b·m_ω·θ_b) ) · w ) '
             '/ OLA(w²)。512² 上：32×32 區塊、hop 8、Hann 窗，'
             '4225 個區塊 × 32×17 個頻格，約 59 萬個參數'
             '（同尺寸的加性 δ 是 78.6 萬）。'
             '<code>÷ OLA(w²)</code> 是 Griffin &amp; Lim (1984) 的最小平方解，'
             '它保證 <b>θ = 0 時輸出逐位元等於原圖</b>（float64 誤差 5.6e−16）。'
             '</p>')
    f = FIG / "phase_shift_applied.png"
    if f.exists():
        o.append(fig_block(f, "實際施加的相位變動 |θ · g_b · m_ω|，"
                              "逐頻格與逐區塊各看一次。條件 r12_f04。"))

    o.append("<h3>1.2　定案的變因</h3>")
    o.append('<div class="scroll">' + table(
        ["旋鈕", "定案值", "理由"], [[a, b, c] for a, b, c in KNOBS]) + "</div>")

    o.append("<h3>1.3　等失真頭對頭：本方法 對 DCT-Shield</h3>")
    o.append('<p class="sub">三組配對全部取自 <code>runs/ip2p_mainline</code>'
             '——本方法與 DCT-Shield 在<b>同一批</b>裡跑，同兩張影像、'
             '同一組算子、同一個空白地板。每組的 DISTS 差距在 3% 以內。</p>')

    if MP:
        o.append("<h4>現行六算子協定（含模糊與裁切）</h4>")
        f = FIG / "matched_ops6.png"
        if f.exists():
            o.append(fig_block(f, ""))
        o.append(pair_table(B, MP, OPS6, OPL6))
        o.append("<h4>JPEG 軸五個算子</h4>")
    f = FIG / "h2h_curve.png"
    if f.exists():
        o.append(fig_block(f, "橫軸由左到右是越來越重的 JPEG 淨化。"))
    o.append(pair_table(B, pur, OPSJ, OPLJ))
    o.append('<p class="dl">淨增益 ＝ effect(p) − 空白地板，兩者都是絕對值；'
             '不報保留率或「佔可達範圍」這類比例讀數。'
             f'LPIPS 在兩張不相干的自然影像之間飽和於 {D["lpips_ceiling"]}。'
             '藍底列是本方法。</p>')

    if H:
        o.append("<h3>1.4　另一個批次：八個算子（含擴散淨化）</h3>")
        o.append('<div class="note">這一組<b>不是嚴格等失真</b>：'
                 '<code>ours_nonadd</code> 的 DISTS 0.1197 與 '
                 '<code>ours_add</code> 的 0.1383 比 <code>dct_e18</code> 的 '
                 '0.0978 高出 22% 與 41%。帶顏色的兩欄（裁切縮放、'
                 'JPEG→重取樣）用的是舊參照，空白地板非 0。</div>')
        f = FIG / "h2h_ops8.png"
        if f.exists():
            o.append(fig_block(f, ""))

    o.append("<h3>1.5　對比圖</h3>")
    o.append('<p class="sub">每一組配對四列一組：交出去的圖、它的編輯結果、'
             '再來是逐個淨化算子的編輯結果。<b>純淨化圖不放</b>——要看的是'
             '攻擊方最後拿到什麼，不是中間那張。</p>')
    gal = "ip2p_matched_headtohead"
    # identity 單獨拉出來當「編輯（防禦圖）」那一列，不混在淨化算子裡：
    # identity 淨化就是不淨化，放在算子清單裡讀者會以為它也是一種淨化。
    PUR = [o_ for o_ in OPS6 if o_ != "identity"]
    for img, name, prompt in IMAGES:
        o.append(f"<h4>{name}　<code>{html.escape(prompt)}</code></h4>")
        for lab, a, b in PAIRS_JPEG:
            o.append(f'<p class="dl"><b>{lab}</b>　'
                     f'{a} DISTS {fmt(B[a]["fid_dists"])} / '
                     f'PSNR {fmt(B[a]["fid_psnr"], 2)}　　'
                     f'{b} DISTS {fmt(B[b]["fid_dists"])} / '
                     f'PSNR {fmt(B[b]["fid_psnr"], 2)}</p>')
            da, db = REP / f"ip2p_mainline/{a}", REP / f"ip2p_mainline/{b}"
            # 第一列：交出去的圖。
            o.append('<div class="grid g4">')
            o.append(cell(da / f"{img}__orig.png", "原圖"))
            o.append('<figure class="miss"><div>交出去的圖</div></figure>')
            o.append(cell(da / f"{img}__{COND[a]}__def.png", f"{a}　防禦圖"))
            o.append(cell(db / f"{img}__{COND[b]}__def.png", f"{b}　防禦圖"))
            o.append("</div>")
            # 第二列：編輯結果（未淨化）。
            o.append('<div class="grid g4">')
            o.append(cell(da / f"{img}__{COND[a]}__edit_orig.png",
                          "編輯（原圖）── 沒有防禦時攻擊方拿到的"))
            o.append('<figure class="miss"><div>編輯（防禦圖）<br>未淨化'
                     '</div></figure>')
            o.append(cell(da / f"{img}__{COND[a]}__edit_def.png",
                          f"{a}　編輯（防禦圖）"))
            o.append(cell(db / f"{img}__{COND[b]}__edit_def.png",
                          f"{b}　編輯（防禦圖）"))
            o.append("</div>")
            if not MP:
                continue
            # 之後每一列一個淨化算子：地板、我方、DCT。
            for op in PUR:
                o.append('<div class="grid g4">')
                o.append(cell(REP / gal / "gallery_floor"
                              / f"{img}__None__{op}__edit_def.png",
                              f"{OPL6[op]}　空白地板：無防禦、同算子", 300))
                o.append(f'<figure class="miss"><div>{OPL6[op]}<br>'
                         f'編輯（淨化後的防禦圖）</div></figure>')
                o.append(cell(REP / gal / f"gallery_{a}"
                              / f"{img}__{COND[a]}__{op}__edit_def.png",
                              f"{a}　{OPL6[op]}", 300))
                o.append(cell(REP / gal / f"gallery_{b}"
                              / f"{img}__{COND[b]}__{op}__edit_def.png",
                              f"{b}　{OPL6[op]}", 300))
                o.append("</div>")
    return "\n".join(o)


def sec_warp(D) -> str:
    W = flatten(D, "warp")
    P = flatten(D, "phase")
    both = dict(W)
    both.update(P)
    o = ['<h2 id="s2">二、位移場</h2>',
         '<p class="sub">程式：<code>src/defense/param_pgd.py</code> 的 '
         '<code>WarpParam</code>／<code>WarpRandomParam</code>／'
         '<code>WarpRoundTripParam</code>，流場正則在 '
         '<code>src/defense/stadv_flow.py</code>。'
         '<code>sa_*</code> 是 stAdv（Xiao et al., ICLR 2018, arXiv:1801.02612）'
         '的逐行移植：稠密流場（<code>--warp-grid 512</code>）、'
         '雙線性四鄰居取樣、流場正則 L_flow（<b>根號在鄰居和的裡面</b>）、'
         'L-BFGS with strong Wolfe。<code>opt_*</code>／<code>rand_*</code> 是'
         '較早的粗網格 16 批次。</p>']
    o.append('<div class="note">τ <b>不可沿用原文的 0.05</b>：原文的 0.05 是在'
             '分類器 logits 的 Carlini–Wagner 損失上網格搜尋得到的，本專案的 '
             'L_adv 是擴散模型上的損失，尺度不同。L_flow 的根號內加了一個 '
             '<code>eps</code>（原文沒有），否則 f ≡ 0 的起點會拿到 NaN；'
             '該偏移是常數、不影響梯度方向，但報表上的 L_flow 讀數含它。</div>')

    o.append("<h3>2.1　構造</h3>")
    o.append(svg_warp_pipeline())
    o.append(svg_rand_control())
    o.append('<p class="dl">與相位方法的差別在於：位移場<b>不新增像素值</b>，'
             '它只把既有的像素搬走，所以擾動的振幅完全由原圖自己的梯度決定。'
             '全隨機對照走的是逐行相同的取樣路徑，唯一的差別是 f 不進 PGD。</p>')

    o.append("<h3>2.2　學到的流場</h3>")
    for img, name, _ in IMAGES:
        f = FIG / f"warp_field_{img[-5:]}.png"
        if f.exists():
            o.append(fig_block(
                f, f"{name}：上列是位移長度（像素），中列是方向，"
                   "下列是像素域的殘差。右欄最上是位移長度的分佈，"
                   "下面兩張是粗網格 16 的隨機與最佳化對照。"))

    o.append("<h3>2.3　最佳化 對 全隨機</h3>")
    rand = sorted((W[t]["fid_dists"], W[t]["edit_lpips"], t) for t in W
                  if W[t].get("condition") == "warp_rand")
    rows = [[t, fmt(d), fmt(l)] for d, l, t in rand]
    o.append('<div class="scroll">' + table(
        ["全隨機位移場", "DISTS", "未淨化位移"], rows) + "</div>")
    # 退化的兩格不列：`sa_t010`／`sa_t050` 的位移是 0.004 與 0.0007、
    # PSNR 55 與 67，也就是 TV 正則把攻擊整個壓成零。那兩格說明的是
    # 「τ 不可沿用原文的 0.05」，已寫在上面的說明框裡。
    opt = sorted((W[t]["fid_dists"], W[t]["edit_lpips"], t) for t in W
                 if t.startswith("sa_") and W[t]["edit_lpips"] > 0.02)
    rows = [[t, fmt(d), fmt(l), W[t].get("stop_reason", "—")]
            for d, l, t in opt]
    o.append('<div class="scroll">' + table(
        ["stAdv 稠密流場（最佳化過）", "DISTS", "未淨化位移", "停止原因"],
        rows) + "</div>")
    o.append('<p class="dl">兩張表要並排讀：全隨機那一張是同一個參數化、'
             '同一個構造、只是不最佳化。落在隨機掃描的失真範圍'
             f'（DISTS {rand[0][0]:.4f}–{rand[-1][0]:.4f}）之外的點不可外插比較。</p>')

    o.append("<h3>2.4　抗淨化</h3>")
    o.append(GEOM_NOTE)
    have = [t for t in ["sa_t000", "sa_r04", "sa_r08", "ig_f08_eot"]
            if t in both and both[t]["purify"]]
    o.append("<h4>總增益 ＝ effect(p)</h4>")
    o.append(purify_table(both, have, OPS6, OPL6, "total", ("ig_f08_eot",)))
    o.append("<h4>淨增益 ＝ effect(p) − 空白地板</h4>")
    o.append(purify_table(both, have, OPS6, OPL6, "net", ("ig_f08_eot",)))
    o.append('<p class="dl">藍底那一列是相位族，放在這裡當同協定的參照。</p>')
    f = FIG / "warp_purify.png"
    if f.exists():
        o.append(fig_block(f, ""))

    o.append("<h3>2.5　防禦圖與編輯結果</h3>")
    pairs = [("sa_t000", "ip2p_stadv/sa_t000", "warp"),
             ("sa_r04", "ip2p_stadv_radius/sa_r04", "warp"),
             ("rand_r4", "ip2p_warp/rand_r4", "warp_rand"),
             ("rand_r8", "ip2p_warp/rand_r8", "warp_rand")]
    for img, name, _ in IMAGES:
        o.append(f"<h4>{name}</h4>")
        for kind, lab in (("def", "防禦圖"), ("edit_def", "編輯（防禦圖）")):
            o.append('<div class="grid g5">')
            if kind == "def":
                o.append(cell(REP / pairs[0][1] / f"{img}__orig.png", "原圖"))
            else:
                o.append(cell(REP / pairs[0][1] / f"{img}__warp__edit_orig.png",
                              "編輯（原圖）"))
            for tag, d, c in pairs:
                dd = both.get(tag, {})
                dt = (f"　DISTS {fmt(dd['fid_dists'])}"
                      if dd.get("fid_dists") is not None else "")
                o.append(cell(REP / d / f"{img}__{c}__{kind}.png",
                              f"{tag}{dt}　{lab}"))
            o.append("</div>")

    o.append("<h4>六個淨化算子逐一（stAdv）</h4>")
    gal = [("sa_t000", "ip2p_stadv/purify/gallery_sa_t000", "warp"),
           ("sa_r04", "ip2p_stadv_radius/purify/gallery_sa_r04", "warp")]
    for img, name, _ in IMAGES:
        for tag, g, c in gal:
            o.append(f"<h4>{name}　{tag}</h4>")
            for kind, lab in (("pur", "淨化後的防禦圖"),
                              ("edit_def", "它的編輯結果")):
                o.append('<div class="grid g6">')
                for op in OPS6:
                    o.append(cell(REP / g / f"{img}__{c}__{op}__{kind}.png",
                                  f"{OPL6[op]}　{lab}", 260))
                o.append("</div>")
    return "\n".join(o)


def sec_heatmap() -> str:
    o = ['<h2 id="s3">三、淨化把「我們施加的相位偏移」轉掉了多少</h2>',
         '<p class="sub">'
         '注入的擾動 δ = 防禦圖 − 原圖；存活的擾動 δ&#39; = T(防禦圖) − T(原圖)，'
         'T 是淨化算子。<b>兩側都扣掉各自的乾淨底</b>——只扣一側會把'
         '「淨化對原圖做的事」算進擾動裡。<br>'
         'R(ω)、R&#39;(ω) 是 δ 與 δ&#39; 的加窗區塊頻譜（32×32、hop 8、Hann、'
         'rfft2，與方法本身同一組基底）。<br>'
         '　相位熱圖(ω) = Σ_b |R_b(ω)|·|Δφ_b(ω)| / Σ_b |R_b(ω)|　（弧度）<br>'
         '　能量存活(ω) = Σ_b |R&#39;_b(ω)| / Σ_b |R_b(ω)|　（無單位）<br>'
         '權數取<b>注入的擾動自己的幅度</b>：|R| ≈ 0 的頻格上 ∠R 由數值噪聲決定，'
         '加權等於只在真的放了擾動的地方取平均。'
         '色階上限固定取 π/2 = 1.571，那是兩個獨立均勻角度之差的絕對值的期望，'
         '即無資訊水平。程式：<code>scripts/phase_shift_heatmap.py</code>。</p>']
    o.append('<div class="note"><code>identity</code> 不畫成一欄：T = 恆等時 '
             'δ&#39; ≡ δ，相位恆為 0、能量存活恆為 1。程式仍然算它一次當守門'
             '——不成立代表讀到的檔案不是防禦圖／原圖本身，那種錯沒有症狀。</div>')
    for f, cap in ((FIG / "phase_shift_sbsurv.png", "條件 sb_surv"),
                   (FIG / "phase_shift.png", "條件 ig_f08_eot")):
        if f.exists():
            o.append(fig_block(f, cap))
    o.append('<div class="note">裁切縮放是繞中心的放大 1.2488×，'
             '區塊格點與原圖對不上，該欄量到的是<b>對位破壞</b>不是相位破壞，'
             '不可與其餘欄並列解讀。</div>')
    return "\n".join(o)


# ── Pipeline 圖（inline SVG）────────────────────────────────────────────
#
# **樣式一律寫成 presentation attribute，不用 class ＋ 外部 CSS。** 外部 CSS
# 只有瀏覽器吃得到；SVG 一旦被複製出頁面、或送進不支援 CSS 選擇器的光柵化器
# （cairosvg 就是），`fill` 會退回預設的黑色——整張圖變成一片黑塊，而且不會
# 報錯。實測踩過一次。
#
# **圖上不放中文說明。** 方塊只寫算子名與形狀，說明留給頁面的正文。
#
# 顏色分四類：
#     藍  主資料路徑（方塊最大）
#     綠  由原圖算一次就固定的閘
#     橘  PGD 學到的張量
#     灰  輔助節點

FONT = '"Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif'
MONO = 'ui-monospace,Consolas,"Courier New",monospace'
BOX = {
    "bx": ("#e7f0fb", "#6d9fd6"),
    "gt": ("#e9f4e8", "#7cae76"),
    "ln": ("#fdeee2", "#d9975f"),
    "nt": ("#f6f6f8", "#c2c2c9"),
    "io": ("#ffffff", "#4a5560"),
}
TXT = {
    "xl": f'font-size="21" font-weight="600" fill="#12141a" font-family={MONO!r}',
    "lg": f'font-size="18" font-weight="600" fill="#12141a" font-family={FONT!r}',
    "tb": f'font-size="15" font-weight="600" fill="#16181c" font-family={FONT!r}',
    "m": f'font-size="14" fill="#16181c" font-family={MONO!r}',
    "ms": f'font-size="12.5" fill="#5c6672" font-family={MONO!r}',
    "s": f'font-size="12" fill="#5c6672" font-family={FONT!r}',
    "lab": f'font-size="12.5" fill="#98a0aa" letter-spacing="0.6" font-family={MONO!r}',
}
STROKE = "#6b7683"
DASH = "#d9975f"


def _box(x, y, w, h, kind, lines, rx=8, sw=1.5):
    fill, stroke = BOX[kind]
    top = y + h / 2 - (len(lines) - 1) * 11
    t = "".join(
        f'<text {TXT[c]} x="{x + w / 2:.0f}" y="{top + i * 22:.0f}" '
        f'text-anchor="middle" dominant-baseline="middle">{html.escape(s)}</text>'
        for i, (s, c) in enumerate(lines))
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>' + t)


def _band(x, y, w, h, cap=""):
    r = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" '
         f'stroke="#dcdce2" stroke-width="1.2" stroke-dasharray="5 5"/>')
    return r + (_label(x + 16, y + 23, cap) if cap else "")


def _line(pts, dashed=False, head=True):
    p = " ".join(f"{a:.0f},{b:.0f}" for a, b in pts)
    col = DASH if dashed else STROKE
    d = ' stroke-dasharray="6 4"' if dashed else ""
    m = f' marker-end="url(#{"ahd" if dashed else "ah"})"' if head else ""
    return (f'<polyline points="{p}" fill="none" stroke="{col}" '
            f'stroke-width="1.8"{d}{m}/>')


def _label(x, y, s, anchor="start"):
    return (f'<text {TXT["lab"]} x="{x}" y="{y}" text-anchor="{anchor}">'
            f"{html.escape(s)}</text>")


def _defs():
    def mk(i, c):
        return (f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    return "<defs>" + mk("ah", STROKE) + mk("ahd", DASH) + "</defs>"


def _svg(w, h, parts, alt):
    """`width`／`height` 給真實數字，再用 style 覆寫成滿版。

    只寫 `width="100%"` 的話瀏覽器沒問題（有 viewBox 就能推出內在比例），
    但不吃 CSS 的光柵化器解析不出尺寸，會**安靜地**吐出一張全白的圖。
    """
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'style="width:100%;height:auto;display:block;margin:10px 0 4px" '
            f'role="img" aria-label="{html.escape(alt)}">'
            + _defs() + "".join(parts) + "</svg>")


def svg_phase_pipeline() -> str:
    W, H = 1660, 520
    p = []
    C1, C2, C3 = 340, 830, 1320

    # 上兩帶：閘（固定）與可學張量。方塊刻意小，主路徑才是視覺重心。
    p.append(_band(24, 14, W - 48, 106, "fixed  ·  from the source image"))
    p.append(_band(24, 132, W - 48, 88, "learned  ·  PGD"))
    for cx, lines in [
        (C1, [("g_b", "xl"), ("texture gate   1 x 4225", "ms")]),
        (C2, [("m_w", "xl"), ("radial gate   32 x 17", "ms")]),
        (C3, [("q(w)", "xl"), ("floor price   32 x 17", "ms")]),
    ]:
        p.append(_box(cx - 190, 42, 380, 66, "gt", lines))
    for cx, lines in [
        (C1, [("theta", "lg"), ("phase rotation", "ms")]),
        (C2, [("g", "lg"), ("magnitude gain", "ms")]),
        (C3, [("a", "lg"), ("additive floor", "ms")]),
    ]:
        p.append(_box(cx - 190, 150, 380, 60, "ln", lines))
    for cx in (C1, C2, C3):
        p.append(_line([(cx, 108), (cx, 150)]))
        p.append(_line([(cx, 210), (cx, 252)]))

    # 匯流排：三者合到主路徑的同一格。
    p.append(_line([(C1, 252), (C1, 276), (C3, 276), (C3, 252)], head=False))
    p.append(_line([(C2, 276), (C2, 306)]))

    # 主路徑：方塊明顯大於上下兩帶。
    MY, MH = 306, 96
    boxes = [
        (150, 108, "io", [("x", "xl")]),
        (300, 214, "bx", [("unfold + Hann", "lg"), ("32 x 32, hop 8", "ms")]),
        (556, 118, "bx", [("rfft2", "lg")]),
        (716, 228, "bx", [("reparameterize", "lg"),
                          ("(|S|e^g + a q s) e^i(phi+theta)", "ms")]),
        (986, 118, "bx", [("irfft2", "lg")]),
        (1146, 200, "bx", [("window + OLA", "lg")]),
        (1388, 122, "io", [("x_def", "xl")]),
    ]
    for x, w, kind, lines in boxes:
        p.append(_box(x, MY, w, MH, kind, lines, rx=10, sw=1.8))
    for i in range(len(boxes) - 1):
        p.append(_line([(boxes[i][0] + boxes[i][1], MY + MH / 2),
                        (boxes[i + 1][0], MY + MH / 2)]))
    # `÷ OLA(w²)` 是 window+OLA 那一格的一部分，寫在框下不另立方塊。
    p.append(f'<text {TXT["ms"]} x="{1146 + 100}" y="{MY + MH + 18}" '
             f'text-anchor="middle">/ OLA(w^2)</text>')

    # PGD 迴圈。
    LY = 448
    p.append(_band(24, LY - 26, W - 48, 62, "PGD"))
    loop = [
        (300, 300, [("IP2P UNet loss   EOT", "tb")]),
        (660, 160, [("backprop", "tb")]),
        (860, 260, [("sign PGD + clamp", "tb")]),
        (1180, 300, [("fixed-sample eval", "tb")]),
    ]
    for x, w, lines in loop:
        p.append(_box(x, LY - 8, w, 40, "nt", lines))
    for i in range(len(loop) - 1):
        p.append(_line([(loop[i][0] + loop[i][1], LY + 12),
                        (loop[i + 1][0], LY + 12)]))
    p.append(_line([(1442, MY + MH), (1442, LY - 8)]))
    p.append(_line([(300, LY + 12), (120, LY + 12), (120, 180), (C1 - 190, 180)],
                   dashed=True))
    return _svg(W, H, p, "texture rephasing pipeline")


def svg_warp_pipeline() -> str:
    W, H = 1660, 380
    p = []
    # 上帶：學到的流場，經上採樣接進主路徑。夾取屬於 PGD 的投影步驟，
    # 畫在下面的迴圈裡而不是這裡——它作用在 f 上，不是作用在輸出上。
    p.append(_band(24, 14, W - 48, 96, "learned  ·  PGD"))
    p.append(_box(150, 38, 470, 60, "ln",
                  [("f = (du, dv)   1 x 2 x 512 x 512", "lg")]))
    p.append(_box(760, 38, 420, 60, "nt",
                  [("bicubic upsample", "tb"), ("identity at grid 512", "ms")]))
    p.append(_line([(620, 68), (760, 68)]))

    MY, MH = 160, 96
    boxes = [
        (150, 108, "io", [("x", "xl")]),
        (330, 330, "bx", [("grid = base + f", "lg")]),
        (760, 420, "bx", [("grid_sample", "lg"), ("bilinear, border", "ms")]),
        (1300, 200, "io", [("x_def", "xl")]),
    ]
    for x, w, kind, lines in boxes:
        p.append(_box(x, MY, w, MH, kind, lines, rx=10, sw=1.8))
    for i2 in range(len(boxes) - 1):
        p.append(_line([(boxes[i2][0] + boxes[i2][1], MY + MH / 2),
                        (boxes[i2 + 1][0], MY + MH / 2)]))
    # 上採樣後的 f 是拿去組成取樣網格的，不是直接餵給 grid_sample。
    p.append(_line([(830, 98), (830, 130), (495, 130), (495, 160)]))

    LY = 310
    p.append(_band(24, LY - 26, W - 48, 66, "objective  ·  update"))
    p.append(_box(150, LY - 8, 260, 44, "nt", [("L_adv", "lg")]))
    p.append(_box(470, LY - 8, 600, 44, "nt",
                  [("tau * L_flow = sum_p sum_q sqrt(|df|^2 + eps)", "m")]))
    p.append(_box(1130, LY - 8, 200, 44, "nt", [("L-BFGS", "lg")]))
    p.append(_box(1380, LY - 8, 230, 44, "nt", [("clamp |f|", "m")]))
    for a, b in ((410, 470), (1070, 1130), (1330, 1380)):
        p.append(_line([(a, LY + 14), (b, LY + 14)]))
    p.append(_line([(1400, MY + MH), (1400, LY - 26)]))
    p.append(_line([(1495, LY + 36), (1495, 358), (60, 358), (60, 68), (150, 68)],
                   dashed=True))
    return _svg(W, H, p, "displacement field pipeline")


def svg_rand_control() -> str:
    W, H = 1660, 130
    p = [_label(150, 28, "random control  ·  same construction, no optimization")]
    p.append(_box(150, 40, 460, 62, "nt",
                  [("f ~ random, frozen", "lg"), ("params() is empty", "ms")]))
    p.append(_box(720, 40, 380, 62, "bx", [("same grid_sample path", "lg")]))
    p.append(_box(1210, 48, 340, 46, "io", [("x_def", "xl")]))
    p.append(_line([(610, 71), (720, 71)]))
    p.append(_line([(1100, 71), (1210, 71)]))
    return _svg(W, H, p, "random displacement field control")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("report_data.json"))
    ap.add_argument("--out", type=Path, default=Path("report_full.html"))
    args = ap.parse_args()
    D = json.loads(args.data.read_text(encoding="utf-8"))

    body = [
        '<div class="wrap">',
        "<h1>相位方法、位移場、與淨化對相位偏移做了什麼</h1>",
        '<p class="sub">兩張影像：盆栽人 <code>task_attr_mod_color_11699</code>'
        '（turn the color of potted plant to pink）、'
        '瑪利歐 <code>task_attr_mod_color_6205</code>'
        '（turn the color of toolbox to red）。'
        '攻擊模型 InstructPix2Pix，100 步、s_T 7.5、s_I 1.5、seed 20260812。'
        '抗淨化每格三個種子。</p>',
        '<div class="toc"><a href="#s1">一、相位方法（含 DCT-Shield 頭對頭）</a>'
        '<a href="#s2">二、位移場</a>'
        '<a href="#s3">三、淨化對相位偏移做了什麼</a></div>',
        '<div class="note">位移一律報兩個<b>絕對值</b>：'
        '總增益 ＝ effect(p)，淨增益 ＝ effect(p) − 空白地板。'
        '不報保留率或「佔可達範圍」這類比例讀數——分母塌陷或參照改變時不可解讀。'
        f'LPIPS 在兩張不相干的自然影像之間飽和於 {D["lpips_ceiling"]}'
        '，該值只作為飽和參考，不進任何算式。</div>',
        sec_method(D), sec_warp(D), sec_heatmap(),
        "</div>"]
    doc = ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           "<title>相位方法與位移場</title><style>" + CSS
           + "</style></head><body>"
           + "\n".join(body) + "</body></html>")
    args.out.write_text(doc, encoding="utf-8")
    print(f"寫出 {args.out}（{args.out.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
