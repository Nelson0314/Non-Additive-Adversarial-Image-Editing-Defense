"""inpainting 批次的圖形化報告：威脅模型、對比圖、淨化結果、數據表。

刻意不含敘述文字與結論（使用者 2026-08-14 指定）。每一格都必須有影像可看。

威脅模型那一段的每一張圖都是實際張量：遮罩由 `data/lo_masks` 讀入，
`x ⊙ (1 − mask)` 就是 `SDWrapper.mask_latents` 編碼的那一張，重畫結果取自
批次的 `*__gen_orig.png`。

用法：
    python scripts/inpaint_report.py --out inpaint_report.html \
        --runs runs/ip5/man_02 runs/ip5/woman_02 ... --retention runs/ip5
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import io
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

CONDS = ["phase", "phase_out", "add", "add_out", "phase_rand",
         "photoguard_c", "advpaint", "promptflare"]
LABEL = {"phase": "site F 紋理重相位", "phase_out": "site F ＋遮罩感知閘",
         "add": "加性 δ", "add_out": "加性 δ ＋遮罩限制",
         "phase_rand": "隨機相位 RPN", "photoguard_c": "PhotoGuard-c",
         "advpaint": "AdvPaint", "promptflare": "PromptFlare"}
SHORT = {"phase": "site F", "phase_out": "site F+M", "add": "加性 δ",
         "add_out": "加性+M", "phase_rand": "隨機相位",
         "photoguard_c": "PhotoGuard-c", "advpaint": "AdvPaint",
         "promptflare": "PromptFlare"}
BUDGET = {"phase": "θ = 1.30", "phase_out": "θ = 1.30", "add": "ε∞ = 1.2/255",
          "add_out": "ε∞ = 1.2/255", "phase_rand": "θ = 1.30",
          "photoguard_c": "‖δ‖₂ = 16", "advpaint": "ε∞ = 0.03",
          "promptflare": "ε∞ = 0.0235"}
ORIGIN = {"phase": "人眼門檻", "phase_out": "人眼門檻", "add": "人眼門檻",
          "add_out": "人眼門檻", "phase_rand": "同失真",
          "photoguard_c": "原生", "advpaint": "原生", "promptflare": "原生"}
PURIF = ["identity", "blur1", "noise0.05", "quantize16", "jpeg75", "jpeg30",
         "crop_resize0.1", "jpeg_then_resize75", "adverse_cleaner", "impress"]
PNAME = {"identity": "未淨化", "blur1": "模糊 σ1", "noise0.05": "雜訊 .05",
         "quantize16": "量化 16", "jpeg75": "JPEG 75", "jpeg30": "JPEG 30",
         "crop_resize0.1": "裁切重縮", "jpeg_then_resize75": "C&R 串接",
         "adverse_cleaner": "AdvCleaner", "impress": "IMPRESS"}
PROMPT = {"man_02": "an old woman", "woman_02": "a man", "dog_03": "a cat",
          "horse_03": "a zebra", "cat_01": "a dog"}


def b64(p, size=None, q=80):
    im = Image.open(p).convert("RGB")
    if size and im.size[0] != size:
        im = im.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, subsampling=1)
    return base64.b64encode(buf.getvalue()).decode()


def b64_arr(a, size=None, q=80):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))
    if size and im.size[0] != size:
        im = im.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, subsampling=1)
    return base64.b64encode(buf.getvalue()).decode()


def tile(data, title, sub="", meta="", cls="", size=None, q=80, arr=False):
    src = b64_arr(data, size, q) if arr else b64(data, size, q)
    return ('<figure class="%s"><img src="data:image/jpeg;base64,%s" alt="%s">'
            '<figcaption><b>%s</b>%s%s</figcaption></figure>'
            % (cls, src, title, title,
               '<span class="m">%s</span>' % sub if sub else "",
               '<span class="d">%s</span>' % meta if meta else ""))


def find(runs, name):
    for r in runs:
        p = Path(r) / name
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--retention", nargs="+", default=[])
    ap.add_argument("--gallery", type=Path, default=Path("runs/ip5/gallery"))
    ap.add_argument("--masks", type=Path, default=Path("data/lo_masks"))
    ap.add_argument("--pur-image", default="dog_03")
    args = ap.parse_args()

    res = {}
    for r in args.runs:
        p = Path(r) / "results.csv"
        if not p.exists():
            continue
        for row in csv.DictReader(open(p, encoding="utf-8")):
            res[(row["image"], row["condition"])] = row
    images = sorted({i for i, _ in res})
    conds = [c for c in CONDS if any((i, c) in res for i in images)]

    def col(c, key):
        v = [float(res[(i, c)][key]) for i in images if (i, c) in res]
        return st.fmean(v) if v else float("nan")

    S = {c: {k: col(c, k) for k in
             ("gen_lpips", "gen_lpips_full", "fid_dists", "fid_lpips",
              "fid_psnr", "attack_strength")} for c in conds}

    # ---------- 抗淨化 ----------
    rows = []
    for r in args.retention:
        for p in sorted(glob.glob(str(Path(r) / "retention*.csv"))):
            rows += list(csv.DictReader(open(p, encoding="utf-8")))
    ok = {(x["image"], x["condition"]): x["usable"] == "True" for x in rows}
    rconds = [c for c in conds if any(k[1] == c for k in ok)]
    common = [i for i in images if all(ok.get((i, c), False) for c in rconds)]
    eff = defaultdict(lambda: defaultdict(list))
    ret = defaultdict(lambda: defaultdict(list))
    per = {}
    for x in rows:
        per[(x["image"], x["condition"], x["purifier"])] = x
        if x["image"] in common:
            eff[x["purifier"]][x["condition"]].append(float(x["effect_mean"]))
            if x["retention"]:
                ret[x["purifier"]][x["condition"]].append(float(x["retention"]))
    E = {k: {c: st.fmean(v) for c, v in d.items()} for k, d in eff.items()}
    R = {k: {c: st.fmean(v) for c, v in d.items()} for k, d in ret.items()}

    PEAR = float("nan")
    corr = ""
    if R and rconds:
        ident = {c: E["identity"][c] for c in rconds if c in E.get("identity", {})}
        mret = {c: st.fmean([R[k][c] for k in PURIF if k != "identity" and c in R.get(k, {})])
                for c in ident}
        X, Y = [ident[c] for c in ident], [mret[c] for c in ident]
        if len(X) > 2:
            mx, my = st.fmean(X), st.fmean(Y)
            den = math.sqrt(sum((a - mx) ** 2 for a in X) * sum((b - my) ** 2 for b in Y))
            if den > 0:
                PEAR = sum((a - mx) * (b - my) for a, b in zip(X, Y)) / den
        corr = "".join(
            '<tr class="%s"><td>%s</td><td class="n">%.4f</td><td class="n">%.4f</td></tr>'
            % ("hi" if c == "phase" else "", LABEL[c], ident[c], mret[c])
            for c in sorted(ident, key=lambda c: -ident[c]))

    # ---------- 威脅模型 ----------
    ref = args.pur_image if args.pur_image in images else images[0]
    x = np.asarray(Image.open(find(args.runs, f"{ref}__orig.png")).convert("RGB"),
                   dtype=np.float32) / 255.0
    m = np.asarray(Image.open(args.masks / f"{ref}.png").convert("L"),
                   dtype=np.float32)[..., None] / 255.0
    step = lambda n, t: '<div class="op"><span class="no">%s</span>%s</div>' % (n, t)
    flow = "".join([
        tile(find(args.runs, f"{ref}__orig.png"), "x", "原圖 512²", cls="fx", size=300),
        tile(m.repeat(3, axis=2), "mask", "人工繪製，1 = 重畫區",
             "涵蓋 %.3f" % float(m.mean()), cls="fx", size=300, arr=True),
        step("1", "x ⊙ (1 − mask)<br>遮罩後影像"),
        tile(x * (1 - m), "x ⊙ (1−mask)", "UNet 後 4 通道編碼的就是這張",
             "重畫區的擾動在此歸零", cls="fx b", size=300, arr=True),
        step("2", "9 通道 UNet<br>[z, m, E(x⊙(1−m))]"),
        step("3", "每一步把遮罩外<br>貼回原圖"),
        tile(find(args.runs, f"{ref}__phase__gen_orig.png"), "重畫結果",
             "未防禦", "攻擊強度 %.3f" % S.get("phase", {}).get("attack_strength", 0),
             cls="fx", size=300),
    ])

    # ---------- 表 ----------
    be = max(S[c]["gen_lpips"] for c in conds)
    t1 = "".join(
        '<tr class="%s"><td>%s</td><td class="g">%s</td><td class="mono">%s</td>'
        '<td class="n%s">%.4f</td><td class="n">%.4f</td><td class="n">%.4f</td>'
        '<td class="n">%.4f</td><td class="n">%.2f</td></tr>'
        % ("hi" if c == "phase" else "", LABEL[c], ORIGIN[c], BUDGET[c],
           " b" if S[c]["gen_lpips"] == be else "", S[c]["gen_lpips"],
           S[c]["gen_lpips_full"], S[c]["fid_dists"], S[c]["fid_lpips"],
           S[c]["fid_psnr"]) for c in conds)

    def ratio_row(a, b, lab):
        va = [float(res[(i, a)]["gen_lpips"]) for i in images if (i, a) in res and (i, b) in res]
        vb = [float(res[(i, b)]["gen_lpips"]) for i in images if (i, a) in res and (i, b) in res]
        if not va:
            return ""
        return ('<tr><td>%s</td><td class="n b">%.3f</td><td class="n">%d/%d</td></tr>'
                % (lab, st.fmean(va) / st.fmean(vb),
                   sum(p > q for p, q in zip(va, vb)), len(va)))
    ratios = "".join([
        ratio_row("phase", "add", "site F ÷ 加性 δ"),
        ratio_row("phase", "phase_rand", "site F ÷ 隨機相位"),
        ratio_row("phase_out", "phase", "遮罩感知閘 ÷ 原閘"),
        ratio_row("add_out", "add", "遮罩限制 ÷ 原加性"),
    ] + [ratio_row("phase", c, "site F ÷ " + LABEL[c])
         for c in ("photoguard_c", "advpaint", "promptflare") if c in conds])

    def ptab(D):
        out = []
        for k in PURIF:
            if k not in D:
                continue
            v = {c: D[k][c] for c in rconds if c in D[k]}
            if not v:
                continue
            bst = max(v.values())
            rk = sorted(v, key=lambda c: -v[c]).index("phase") + 1 if "phase" in v else 0
            out.append('<tr><td>%s</td>%s<td class="n rk">%s</td></tr>' % (
                PNAME[k],
                "".join('<td class="n%s%s">%s</td>' % (
                    " b" if v.get(c) == bst else "", " hi" if c == "phase" else "",
                    "%.4f" % v[c] if c in v else "—") for c in rconds),
                "%d/%d" % (rk, len(v)) if rk else "—"))
        return "".join(out)

    # ---------- 影像板 ----------
    plates = []
    for img in images:
        mk = np.asarray(Image.open(args.masks / f"{img}.png").convert("L"),
                        dtype=np.float32)[..., None] / 255.0
        t = [tile(find(args.runs, f"{img}__orig.png"), "原圖", "未防禦", cls="pl", size=512),
             tile(mk.repeat(3, axis=2), "遮罩", "1 = 攻擊方重畫",
                  "涵蓋 %.3f" % float(mk.mean()), cls="pl", size=512, arr=True)]
        g = []
        gen0 = find(args.runs, f"{img}__{conds[0]}__gen_orig.png")
        if gen0:
            g.append(tile(gen0, "未防禦的重畫", "", "", cls="ed", size=384))
        for c in conds:
            r = res.get((img, c))
            if not r:
                continue
            t.append(tile(find(args.runs, f"{img}__{c}__def.png"), LABEL[c],
                          "%s · %s" % (BUDGET[c], ORIGIN[c]),
                          "DISTS %.4f · LPIPS %.4f · PSNR %.1f"
                          % (float(r["fid_dists"]), float(r["fid_lpips"]),
                             float(r["fid_psnr"])),
                          cls="pl" + (" hi" if c == "phase" else ""), size=512))
            g.append(tile(find(args.runs, f"{img}__{c}__gen_def.png"), LABEL[c], "",
                          "效果 %.3f" % float(r["gen_lpips"]),
                          cls="ed" + (" hi" if c == "phase" else ""), size=384))
        tag = "" if not common or img in common else '<span class="tag">retention 部分不可用</span>'
        plates.append(
            '<h3>%s <span class="pr">“%s”</span>%s</h3>'
            '<h4>防禦後（512² 原生解析度）</h4><div class="gp">%s</div>'
            '<h4>重畫結果</h4><div class="ge">%s</div>'
            % (img, PROMPT.get(img, ""), tag, "".join(t), "".join(g)))

    # ---------- 淨化區段 ----------
    purs = []
    for c in [x for x in ("phase", "add", "phase_rand", "photoguard_c") if x in rconds]:
        cells = []
        for k in PURIF:
            pu = args.gallery / f"{args.pur_image}__{c}__{k}__pur.png"
            gn = args.gallery / f"{args.pur_image}__{c}__{k}__gen.png"
            if not (pu.exists() and gn.exists()):
                continue
            r = per.get((args.pur_image, c, k))
            d = ("效果 %.3f · ret %.2f" % (float(r["effect_mean"]),
                                           float(r["retention"]))) if r else ""
            cells.append(
                '<div class="pu"><img src="data:image/jpeg;base64,%s" alt="淨化">'
                '<img src="data:image/jpeg;base64,%s" alt="重畫">'
                '<div class="pc"><b>%s</b><span class="d">%s</span></div></div>'
                % (b64(pu, 248), b64(gn, 248), PNAME[k], d))
        if cells:
            purs.append('<h4>%s · %s</h4><div class="gu">%s</div>'
                        % (LABEL[c], BUDGET[c], "".join(cells)))

    head = "".join("<th>%s</th>" % SHORT[c] for c in rconds)
    html = TEMPLATE % dict(
        css=CSS, flow=flow, t1=t1, ratios=ratios, plates="".join(plates),
        purs="".join(purs) or "<p class=\"na\">尚未產出</p>",
        head=head, eff=ptab(E) or "", ret=ptab(R) or "",
        corr=corr, pear=PEAR, nimg=len(images), ncond=len(conds),
        ncommon=len(common), common="、".join(common) or "—",
        purimg=args.pur_image)
    args.out.write_text(html, encoding="utf-8")
    print("bytes", len(html.encode()), "->", args.out)
    print("影像", images)
    print("條件", conds, "| retention 條件", rconds, "| 共同可用", common)


CSS = """
:root{--bg:#f7f6f4;--surf:#fff;--surf2:#efedea;--ink:#191714;--mut:#6b665e;
--line:#e0dcd6;--acc:#8a4b1f;--acc-w:#f7ece2;--warn:#7a5a12;--warn-w:#f6f0dd;--ok:#2c6047}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#12100e;--surf:#1a1815;--surf2:#232019;--ink:#ece8e2;--mut:#a09a90;
--line:#2c2822;--acc:#d99257;--acc-w:#2e1e12;--warn:#d4b25c;--warn-w:#2a2413;--ok:#6bbd96}}
:root[data-theme="dark"]{--bg:#12100e;--surf:#1a1815;--surf2:#232019;--ink:#ece8e2;
--mut:#a09a90;--line:#2c2822;--acc:#d99257;--acc-w:#2e1e12;--warn:#d4b25c;
--warn-w:#2a2413;--ok:#6bbd96}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
font:15px/1.6 -apple-system,"Segoe UI","Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:1460px;margin:0 auto;padding:40px 24px 90px}
.mono,.n,.d,code{font-family:ui-monospace,"Cascadia Code",Consolas,monospace;
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
.na{color:var(--mut);font-size:13px}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:10px;margin:14px 0}
.op{flex:0 0 auto;align-self:center;background:var(--surf2);border:1px solid var(--line);
border-radius:6px;padding:10px 13px;font-size:12px;line-height:1.45;min-width:126px;color:var(--mut)}
.op .no{display:inline-flex;width:17px;height:17px;border-radius:50%;background:var(--acc);
color:var(--bg);align-items:center;justify-content:center;font-size:10.5px;font-weight:700;
margin-right:6px;vertical-align:1px}
figure{margin:0;background:var(--surf);border:1px solid var(--line);border-radius:6px;overflow:hidden}
figure.hi{border-color:var(--acc);border-width:2px}
figure.b{border-color:var(--ok);border-width:2px}
figure img{width:100%;display:block;aspect-ratio:1;object-fit:cover}
figure.fx{flex:0 0 210px}
figcaption{padding:6px 8px;font-size:11.5px;line-height:1.4}
figcaption b{display:block;font-weight:650}
figcaption .m{display:block;color:var(--mut);font-size:10.5px}
figcaption .d{display:block;color:var(--mut);font-size:10px;margin-top:2px;
font-family:ui-monospace,Consolas,monospace}
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
"""

TEMPLATE = """<title>遮罩重繪防禦圖譜</title>
<style>%(css)s</style>
<div class="wrap">
<header>
<h1>遮罩重繪防禦圖譜 · inpainting</h1>
<div class="meta">
<span><b>批次</b> runs/ip5</span>
<span><b>權重</b> runwayml/stable-diffusion-inpainting（9 通道）</span>
<span><b>影像</b> %(nimg)d</span>
<span><b>條件</b> %(ncond)d</span>
<span><b>重畫</b> 50 步 · 無 strength</span>
<span><b>遮罩</b> 人工繪製，主體在遮罩外</span>
</div>
</header>

<h2>威脅模型</h2>
<div class="flow">%(flow)s</div>

<h2>抗編輯 · %(nimg)d 張平均</h2>
<div class="tw"><table><thead><tr><th>條件</th><th>預算來源</th><th>半徑</th>
<th>效果（遮罩內）</th><th>整張 LPIPS</th><th>DISTS</th><th>LPIPS</th><th>PSNR</th></tr></thead>
<tbody>%(t1)s</tbody></table></div>

<h4>比值</h4>
<div class="tw"><table><thead><tr><th>比較</th><th>比值</th><th>逐圖勝出</th></tr></thead>
<tbody>%(ratios)s</tbody></table></div>

<h2>對比圖</h2>
%(plates)s

<h2>淨化結果 · %(purimg)s</h2>
<h4>每格上＝淨化後的防禦圖，下＝該圖再經重畫</h4>
%(purs)s

<h2>抗淨化 · 淨化後的絕對效果</h2>
<div class="tw"><table><thead><tr><th>淨化算子</th>%(head)s<th>site F</th></tr></thead>
<tbody>%(eff)s</tbody></table></div>

<h2>抗淨化 · 衰減率 retention</h2>
<div class="tw"><table><thead><tr><th>淨化算子</th>%(head)s<th>site F</th></tr></thead>
<tbody>%(ret)s</tbody></table></div>

<h2>分母與衰減率 · Pearson r = %(pear)+.2f</h2>
<div class="tw"><table><thead><tr><th>條件</th><th>分母 effect(identity)</th>
<th>平均 retention</th></tr></thead><tbody>%(corr)s</tbody></table></div>
<div class="key"><span>抗淨化兩表取全部條件通過 3σ 分母閘的 %(ncommon)d 張交集：%(common)s</span></div>
</div>"""


if __name__ == "__main__":
    main()
