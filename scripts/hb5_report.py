"""hb5 批次的圖形化報告：架構、對比圖、淨化結果、數據表。

刻意不含敘述文字與結論（使用者 2026-08-14 指定）。每一格都必須有影像可看。

架構圖的每一張中間影像都由 `src/residual/texture_rephase.py` 的真實程式碼算出，
不是示意圖；`arch/facts.json` 存的是同一次計算印出的可驗證數值。

前置：
    python scripts/hb5_arch_assets.py --out runs/hb5/arch
    python scripts/hb5_purify_gallery.py --out runs/hb5/purified ...
用法：
    python scripts/hb5_report.py --out hb5_report.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import io
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGES = ["man_02", "woman_02", "dog_03", "horse_03", "cat_01"]
CONDS = ["phase", "add", "phase_rand", "apa_weak", "mist", "dia_r", "photoguard_c"]
LABEL = {"phase": "紋理重相位", "add": "加性 δ", "phase_rand": "隨機相位 RPN",
         "apa_weak": "弱 baseline APA", "mist": "Mist", "dia_r": "DIA-R",
         "photoguard_c": "PhotoGuard-c"}
SHORT = {"phase": "紋理重相位", "add": "加性 δ", "phase_rand": "隨機相位",
         "apa_weak": "APA", "mist": "Mist", "dia_r": "DIA-R",
         "photoguard_c": "PhotoGuard-c"}
BUDGET = {"phase": "θ = 1.30", "add": "ε∞ = 1.2/255", "phase_rand": "θ = 1.30",
          "apa_weak": "ε_a = 0.4 latent", "mist": "ε∞ = 16/255",
          "dia_r": "ε∞ = 0.025", "photoguard_c": "‖δ‖₂ = 16"}
ORIGIN = {"phase": "人眼門檻", "add": "人眼門檻", "phase_rand": "同失真",
          "apa_weak": "原生", "mist": "原生", "dia_r": "原生",
          "photoguard_c": "原生 L2"}
PURIF = ["identity", "blur1", "noise0.05", "quantize16", "jpeg75", "jpeg30",
         "crop_resize0.1", "jpeg_then_resize75", "adverse_cleaner", "impress"]
PNAME = {"identity": "未淨化", "blur1": "模糊 σ1", "noise0.05": "雜訊 .05",
         "quantize16": "量化 16", "jpeg75": "JPEG 75", "jpeg30": "JPEG 30",
         "crop_resize0.1": "裁切重縮", "jpeg_then_resize75": "C&R 串接",
         "adverse_cleaner": "AdvCleaner", "impress": "IMPRESS"}
PROMPT = {"man_02": "an old woman", "woman_02": "a man", "dog_03": "a cat",
          "horse_03": "a zebra", "cat_01": "a dog"}
PUR_IMG = "dog_03"
PUR_CONDS = ["phase", "add", "phase_rand", "photoguard_c"]


def defpath(img, cond, kind):
    if cond == "photoguard_c":
        return "runs/hb5_pgc/%s__photoguard_c__%s.png" % (img, kind)
    seg = cond + "__human" if cond in ("phase", "add", "phase_rand") else cond
    return "runs/hb5/%s__%s__%s.png" % (img, seg, kind)


def b64(p, size=None, q=80):
    im = Image.open(p).convert("RGB")
    if size and im.size[0] != size:
        im = im.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, subsampling=1)
    return base64.b64encode(buf.getvalue()).decode()


def tile(src, title, sub="", data="", cls="", size=None, q=80):
    return ('<figure class="%s"><img src="data:image/jpeg;base64,%s" alt="%s">'
            '<figcaption><b>%s</b>%s%s</figcaption></figure>'
            % (cls, b64(src, size, q), title, title,
               '<span class="m">%s</span>' % sub if sub else "",
               '<span class="d">%s</span>' % data if data else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", type=Path, default=Path("runs/hb5/arch"))
    ap.add_argument("--purified", type=Path, default=Path("runs/hb5/purified"))
    args = ap.parse_args()
    A = args.arch
    F = json.loads((A / "facts.json").read_text(encoding="utf-8"))

    # ---------- 階段一數據 ----------
    res = {}
    for p in ("runs/phaseA_human/results.csv", "runs/hb5/g2/results.csv",
              "runs/hb5_pgc/results.csv"):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["image"] in IMAGES:
                res[(r["image"], r["condition"])] = r
    S = {c: {k: st.fmean(float(res[(i, c)][f]) for i in IMAGES) for k, f in
             (("edit", "edit_lpips"), ("dists", "fid_dists"),
              ("lpips", "fid_lpips"), ("psnr", "fid_psnr"))} for c in CONDS}

    # ---------- 階段二數據 ----------
    rows = []
    for p in sorted(glob.glob("runs/hb5/retention_*.csv")):
        rows += list(csv.DictReader(open(p, encoding="utf-8")))
    ok = {(r["image"], r["condition"]): r["usable"] == "True" for r in rows}
    common = [i for i in IMAGES if all(ok.get((i, c), False) for c in CONDS)]
    eff = defaultdict(lambda: defaultdict(list))
    ret = defaultdict(lambda: defaultdict(list))
    per = {}
    for r in rows:
        per[(r["image"], r["condition"], r["purifier"])] = r
        if r["image"] in common:
            eff[r["purifier"]][r["condition"]].append(float(r["effect_mean"]))
            if r["retention"]:
                ret[r["purifier"]][r["condition"]].append(float(r["retention"]))
    E = {k: {c: st.fmean(v) for c, v in d.items()} for k, d in eff.items()}
    R = {k: {c: st.fmean(v) for c, v in d.items()} for k, d in ret.items()}
    ident = {c: E["identity"][c] for c in CONDS}
    mret = {c: st.fmean([R[k][c] for k in PURIF if k != "identity"]) for c in CONDS}
    X, Y = [ident[c] for c in CONDS], [mret[c] for c in CONDS]
    mx, my = st.fmean(X), st.fmean(Y)
    PEAR = (sum((a - mx) * (b - my) for a, b in zip(X, Y)) /
            math.sqrt(sum((a - mx) ** 2 for a in X) * sum((b - my) ** 2 for b in Y)))

    # ---------- 架構：紋理重相位 ----------
    step = lambda n, t: '<div class="op"><span class="no">%s</span>%s</div>' % (n, t)
    flow = "".join([
        tile(A / "01b_input_marked.png", "x", "512² 輸入", "紅框＝下方追蹤的區塊",
             cls="fx", size=340),
        step("1", "切重疊區塊<br>32² · hop 16 · %d 塊" % F["n_blocks"]),
        tile(A / "02_block.png", "P_b", "單一區塊 32²", cls="fs", size=176),
        step("2", "加 Hann 窗<br>w ⊙ P_b"),
        tile(A / "03_window.png", "w", "週期 Hann 窗", cls="fs", size=176),
        tile(A / "04_windowed.png", "w ⊙ P_b", "加窗後", cls="fs", size=176),
        step("3", "rfft2<br>正交歸一"),
        tile(A / "05_magnitude.png", "|X|", "幅度譜 32×17", cls="fs", size=176),
        tile(A / "06_phase.png", "∠X", "相位譜 32×17", cls="fs", size=176),
    ])
    flow2 = "".join([
        step("4", "乘上單位模長複數<br>X · e^(i·g_b·m_ω·θ_b)"),
        tile(A / "07_freq_gate.png", "m_ω", "徑向閘 r_min=0.12",
             "%d/%d 格歸零" % (F["freq_gate_zeroed_bins"], F["freq_gate_total_bins"]),
             cls="fs", size=176),
        tile(A / "08_tex_gate.png", "g_b", "紋理閘 33×33",
             "有效面積 %.3f" % F["active_fraction"], cls="fs", size=176),
        tile(A / "05b_magnitude_after.png", "|X·e^iθ|", "幅度譜（旋轉後）",
             "與 |X| 最大差 %.1e" % F["block_mag_max_abs_diff"], cls="fs b", size=176),
        tile(A / "06b_phase_after.png", "∠(X·e^iθ)", "相位譜（旋轉後）",
             "已改變", cls="fs", size=176),
        step("5", "irfft2 · ⊙w<br>重疊相加"),
        tile(A / "04b_windowed_after.png", "區塊輸出", "θ=1.30 示範", cls="fs", size=176),
        step("6", "OLA ÷ OLA(w²)<br>正規化"),
        tile(A / "11_phase_def.png", "x_def", "實際訓練結果",
             "L∞ %.4f · RMS %.5f" % (F["phase_res_linf"], F["phase_res_rms"]),
             cls="fx", size=340),
    ])
    checks = "".join([
        '<div class="chk"><div class="ck">θ = 0 的輸出</div>'
        '<div class="cv">%.2e</div><div class="cs">max |x_def − x|，float32 機器精度</div></div>' % F["identity_max_abs_err"],
        '<div class="chk"><div class="ck">區塊幅度譜</div>'
        '<div class="cv">%.2e</div><div class="cs">max ‖X·e^iθ| − |X‖</div></div>' % F["block_mag_max_abs_diff"],
        '<div class="chk"><div class="ck">整圖幅度偏差</div>'
        '<div class="cv">%.4f</div><div class="cs">θ=1.30 示範，重疊相加後</div></div>' % F["amp_dev_theta130"],
        '<div class="chk"><div class="ck">參數量</div>'
        '<div class="cv">%s</div><div class="cs">%d 塊 × 32×17 格</div></div>'
        % ("{:,}".format(F["n_blocks"] * 32 * 17), F["n_blocks"]),
    ])

    # ---------- 架構：加性 ----------
    addflow = "".join([
        tile(A / "01_input.png", "x", "512² 輸入", cls="fx", size=340),
        step("1", "加上 δ<br>φ = δ，逐像素自由"),
        tile(A / "12_add_res.png", "δ", "殘差 ×8 ＋0.5",
             "L∞ %.4f · RMS %.5f" % (F["add_res_linf"], F["add_res_rms"]),
             cls="fx", size=340),
        step("2", "clamp(x + δ, 0, 1)<br>δ ← clamp(δ, −ε, +ε)"),
        tile(A / "11_add_def.png", "x_def", "實際訓練結果", "ε = 1.2/255",
             cls="fx", size=340),
    ])
    phres = tile(A / "12_phase_res.png", "紋理重相位的殘差", "×8 ＋0.5",
                 "L∞ %.4f · RMS %.5f" % (F["phase_res_linf"], F["phase_res_rms"]),
                 cls="fx", size=340)
    adres = tile(A / "12_add_res.png", "加性 δ 的殘差", "×8 ＋0.5",
                 "L∞ %.4f · RMS %.5f" % (F["add_res_linf"], F["add_res_rms"]),
                 cls="fx", size=340)

    # ---------- 表 ----------
    be = max(S[c]["edit"] for c in CONDS)
    t1 = "".join(
        '<tr class="%s"><td>%s</td><td class="g">%s</td><td class="mono">%s</td>'
        '<td class="n%s">%.4f</td><td class="n">%.4f</td><td class="n">%.4f</td>'
        '<td class="n">%.2f</td></tr>'
        % ("hi" if c == "phase" else "", LABEL[c], ORIGIN[c], BUDGET[c],
           " b" if S[c]["edit"] == be else "", S[c]["edit"], S[c]["dists"],
           S[c]["lpips"], S[c]["psnr"]) for c in CONDS)

    def ptab(D):
        out = []
        for k in PURIF:
            v = {c: D[k][c] for c in CONDS}
            bst = max(v.values())
            rk = sorted(v, key=lambda c: -v[c]).index("phase") + 1
            out.append('<tr><td>%s</td>%s<td class="n rk">%d/7</td></tr>' % (
                PNAME[k], "".join('<td class="n%s%s">%.4f</td>' % (
                    " b" if v[c] == bst else "", " hi" if c == "phase" else "",
                    v[c]) for c in CONDS), rk))
        return "".join(out)

    corr = "".join(
        '<tr class="%s"><td>%s</td><td class="n">%.4f</td><td class="n">%.4f</td></tr>'
        % ("hi" if c == "phase" else "", LABEL[c], ident[c], mret[c])
        for c in sorted(CONDS, key=lambda c: -ident[c]))
    head = "".join("<th>%s</th>" % SHORT[c] for c in CONDS)

    # ---------- 影像板 ----------
    plates = []
    for img in IMAGES:
        t = [tile("runs/hb5/%s__orig.png" % img, "原圖", "未防禦", cls="pl", size=512)]
        for c in CONDS:
            r = res[(img, c)]
            t.append(tile(defpath(img, c, "def"), LABEL[c],
                          "%s · %s" % (BUDGET[c], ORIGIN[c]),
                          "DISTS %.4f · LPIPS %.4f · PSNR %.1f"
                          % (float(r["fid_dists"]), float(r["fid_lpips"]),
                             float(r["fid_psnr"])),
                          cls="pl" + (" hi" if c == "phase" else ""), size=512))
        e = [tile("runs/hb5/%s__phase__human__edit_orig.png" % img,
                  "未防禦的編輯", "", "", cls="ed", size=384)]
        for c in CONDS:
            e.append(tile(defpath(img, c, "edit_def"), LABEL[c], "",
                          "位移 %.3f" % float(res[(img, c)]["edit_lpips"]),
                          cls="ed" + (" hi" if c == "phase" else ""), size=384))
        tag = "" if img in common else '<span class="tag">retention 部分不可用</span>'
        plates.append(
            '<h3>%s <span class="pr">“%s”</span>%s</h3>'
            '<h4>防禦後（512² 原生解析度）</h4><div class="gp">%s</div>'
            '<h4>編輯結果</h4><div class="ge">%s</div>'
            % (img, PROMPT[img], tag, "".join(t), "".join(e)))

    # ---------- 淨化區段 ----------
    purs = []
    for c in PUR_CONDS:
        cells = []
        for k in PURIF:
            r = per.get((PUR_IMG, c, k))
            d = ("effect %.3f · ret %.2f" % (float(r["effect_mean"]),
                                             float(r["retention"]))) if r else ""
            cells.append(
                '<div class="pu">'
                '<img src="data:image/jpeg;base64,%s" alt="%s 淨化後">'
                '<img src="data:image/jpeg;base64,%s" alt="%s 編輯">'
                '<div class="pc"><b>%s</b><span class="d">%s</span></div></div>'
                % (b64(args.purified / ("%s__%s__%s__pur.png" % (PUR_IMG, c, k)), 248),
                   k,
                   b64(args.purified / ("%s__%s__%s__edit.png" % (PUR_IMG, c, k)), 248),
                   k, PNAME[k], d))
        purs.append('<h4>%s · %s</h4><div class="gu">%s</div>'
                    % (LABEL[c], BUDGET[c], "".join(cells)))

    html = TEMPLATE % dict(
        flow=flow, flow2=flow2, checks=checks, addflow=addflow,
        phres=phres, adres=adres, t1=t1, plates="".join(plates),
        purs="".join(purs), purimg=PUR_IMG, head=head,
        eff=ptab(E), ret=ptab(R), corr=corr, pear=PEAR,
        ncommon=len(common), common="、".join(common), css=CSS)
    args.out.write_text(html, encoding="utf-8")
    print("bytes", len(html.encode()), "->", args.out)


CSS = """
:root{--bg:#f6f8f9;--surf:#fff;--surf2:#eef2f4;--ink:#131a1f;--mut:#5b6b76;
--line:#dbe3e8;--acc:#0d6d8a;--acc-w:#e3f1f6;--warn:#9c4f18;--warn-w:#f7ece3;--ok:#1d6b45}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#0d1113;--surf:#151b1f;--surf2:#1d252a;--ink:#e4ebef;--mut:#8ea0ac;
--line:#232d33;--acc:#54c4e2;--acc-w:#0f2c36;--warn:#e0904f;--warn-w:#2e2013;--ok:#5cc48f}}
:root[data-theme="dark"]{--bg:#0d1113;--surf:#151b1f;--surf2:#1d252a;--ink:#e4ebef;
--mut:#8ea0ac;--line:#232d33;--acc:#54c4e2;--acc-w:#0f2c36;--warn:#e0904f;
--warn-w:#2e2013;--ok:#5cc48f}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
font:15px/1.6 -apple-system,"Segoe UI","Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:1460px;margin:0 auto;padding:40px 24px 90px}
.mono,.n,.d,.cv,code{font-family:ui-monospace,"Cascadia Code",Consolas,monospace;
font-variant-numeric:tabular-nums}
header{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:30px}
h1{font-size:29px;line-height:1.2;margin:0;letter-spacing:-.02em}
.meta{display:flex;flex-wrap:wrap;gap:6px 20px;margin-top:14px;font-size:12.5px;color:var(--mut)}
.meta b{color:var(--ink);font-weight:600}
h2{font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--acc);
margin:56px 0 14px;font-weight:700;border-bottom:1px solid var(--line);padding-bottom:7px}
h3{font-size:17px;margin:36px 0 4px;letter-spacing:-.01em}
h4{font-size:11px;letter-spacing:.11em;text-transform:uppercase;color:var(--mut);
margin:22px 0 9px;font-weight:700}
.pr{font-weight:400;font-size:13px;color:var(--mut)}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:10px;margin:14px 0}
.op{flex:0 0 auto;align-self:center;background:var(--surf2);border:1px solid var(--line);
border-radius:6px;padding:10px 13px;font-size:12px;line-height:1.45;position:relative;
min-width:118px;color:var(--mut)}
.op .no{display:inline-flex;width:17px;height:17px;border-radius:50%;background:var(--acc);
color:var(--bg);align-items:center;justify-content:center;font-size:10.5px;font-weight:700;
margin-right:6px;vertical-align:1px}
figure{margin:0;background:var(--surf);border:1px solid var(--line);border-radius:6px;
overflow:hidden}
figure.hi{border-color:var(--acc);border-width:2px}
figure.b{border-color:var(--ok);border-width:2px}
figure img{width:100%;display:block;aspect-ratio:1;object-fit:cover}
figure.fs{flex:0 0 132px} figure.fx{flex:0 0 232px}
figcaption{padding:6px 8px;font-size:11.5px;line-height:1.4}
figcaption b{display:block;font-weight:650}
figcaption .m{display:block;color:var(--mut);font-size:10.5px}
figcaption .d{display:block;color:var(--mut);font-size:10px;margin-top:2px;
font-family:ui-monospace,Consolas,monospace}
.checks{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:11px;margin:18px 0}
.chk{background:var(--surf);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
.ck{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);font-weight:700}
.cv{font-size:22px;font-weight:700;letter-spacing:-.02em;margin:5px 0 2px;color:var(--ok)}
.cs{font-size:11px;color:var(--mut)}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surf);margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:700px}
th,td{padding:7px 11px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--surf2);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;
color:var(--mut);font-weight:700}
tr:last-child td{border-bottom:none}
td.n{text-align:right} td.g{color:var(--mut);font-size:12px}
td.mono{font-size:11.5px;color:var(--mut)}
td.b{font-weight:700;color:var(--ok)}
tr.hi td,td.hi{background:var(--acc-w)}
td.rk{color:var(--mut);font-size:11.5px}
.gp{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
.ge{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:12px}
.gu{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:10px}
.pu{background:var(--surf);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.pu img{width:100%;display:block;aspect-ratio:1;object-fit:cover}
.pu img+img{border-top:2px solid var(--acc)}
.pc{padding:6px 8px;font-size:11px;line-height:1.35}
.pc b{display:block;font-weight:650}
.pc .d{display:block;color:var(--mut);font-size:9.5px;margin-top:2px;
font-family:ui-monospace,Consolas,monospace}
.tag{font-size:10.5px;padding:2px 7px;border-radius:99px;background:var(--warn-w);
color:var(--warn);font-weight:700;vertical-align:2px;margin-left:6px}
.key{display:flex;flex-wrap:wrap;gap:14px;font-size:11.5px;color:var(--mut);margin:8px 0 0}
.key span::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;
margin-right:5px;vertical-align:0}
.key .a::before{background:var(--acc)} .key .o::before{background:var(--ok)}
"""

TEMPLATE = """<title>紋理重相位防禦圖譜</title>
<style>%(css)s</style>
<div class="wrap">
<header>
<h1>紋理重相位防禦圖譜 · hb5</h1>
<div class="meta">
<span><b>批次</b> runs/hb5 · runs/hb5_pgc</span>
<span><b>影像</b> 5 類別各 1 張</span>
<span><b>條件</b> 7</span>
<span><b>淨化算子</b> 10</span>
<span><b>攻擊者</b> stock SD v1.4 · SDEdit 0.55</span>
<span><b>硬體</b> 4 × RTX 3090 · 4.05 h</span>
</div>
<div class="key"><span class="a">紋理重相位</span><span class="o">最佳值 / 構造保證</span></div>
</header>

<h2>架構 · 紋理重相位</h2>
<div class="flow">%(flow)s</div>
<div class="flow">%(flow2)s</div>
<div class="checks">%(checks)s</div>

<h2>架構 · 加性 δ 對照</h2>
<div class="flow">%(addflow)s</div>

<h2>兩者的殘差</h2>
<div class="flow">%(phres)s%(adres)s</div>

<h2>抗編輯 · 五張平均</h2>
<div class="tw"><table><thead><tr><th>條件</th><th>預算來源</th><th>半徑</th>
<th>位移量 edit_lpips</th><th>DISTS</th><th>LPIPS</th><th>PSNR</th></tr></thead>
<tbody>%(t1)s</tbody></table></div>

<h2>對比圖</h2>
%(plates)s

<h2>淨化結果 · %(purimg)s</h2>
<h4>每格上＝淨化後的防禦圖，下＝該圖再經編輯</h4>
%(purs)s

<h2>抗淨化 · 淨化後的絕對位移量</h2>
<div class="tw"><table><thead><tr><th>淨化算子</th>%(head)s<th>紋理重相位</th></tr></thead>
<tbody>%(eff)s</tbody></table></div>

<h2>抗淨化 · 衰減率 retention</h2>
<div class="tw"><table><thead><tr><th>淨化算子</th>%(head)s<th>紋理重相位</th></tr></thead>
<tbody>%(ret)s</tbody></table></div>

<h2>分母與衰減率 · Pearson r = %(pear)+.2f</h2>
<div class="tw"><table><thead><tr><th>條件</th><th>分母 effect(identity)</th>
<th>平均 retention</th></tr></thead><tbody>%(corr)s</tbody></table></div>
<div class="key"><span>兩張抗淨化表取七條件全部通過 3σ 分母閘的 %(ncommon)d 張交集：%(common)s</span></div>
</div>"""


if __name__ == "__main__":
    main()
