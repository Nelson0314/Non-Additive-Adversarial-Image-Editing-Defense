"""inpainting 批次的圖形化報告：威脅模型、兩種場景的對比圖、淨化結果、數據表。

刻意不含敘述文字與結論（使用者 2026-08-14 指定）。每一格都必須有影像可看。

威脅模型那一段的每一張圖都是實際張量，包含 **UNet 後 4 通道真正編碼的那一張**
（遮罩區填 `[0,1]` 的 0.5，換算後為模型值域的 0）——填黑會讓模型把洞畫黑，
那正是 2026-08-14 修掉的缺陷。

用法：
    python scripts/inpaint_report.py --out inpaint_report.html \
        --runs runs/ip2/background runs/ip2/subject --retention runs/ip2
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

CONDS = ["phase", "add", "phase_rand", "apa_weak", "mist", "dia_r", "photoguard_c"]
LABEL = {"phase": "site F 紋理重相位", "add": "加性 δ", "phase_rand": "隨機相位 RPN",
         "apa_weak": "弱 baseline APA", "mist": "Mist", "dia_r": "DIA-R",
         "photoguard_c": "PhotoGuard-c"}
SHORT = {"phase": "site F", "add": "加性 δ", "phase_rand": "隨機相位",
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


def img_of(run: Path, name: str):
    p = run / name
    return p if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--retention", nargs="+", default=[])
    ap.add_argument("--gallery", type=Path, default=Path("runs/ip2/gallery"))
    ap.add_argument("--masks", type=Path, default=Path("data/lo_masks_auto"))
    ap.add_argument("--pur-image", default="dog_03")
    args = ap.parse_args()

    # 每個影像一個目錄（`inpaint_edit.py` 逐影像分片）。
    res, where = {}, {}
    for r in args.runs:
        r = Path(r)
        for row in csv.DictReader(open(r / "results.csv", encoding="utf-8")):
            res[(row["image"], row["condition"])] = row
            where[row["image"]] = r
    images = sorted({i for i, _ in res})
    conds = [c for c in CONDS if any((i, c) in res for i in images)]

    def agg(c, key):
        v = [float(res[(i, c)][key]) for i in images
             if (i, c) in res and res[(i, c)].get(key)]
        return st.fmean(v) if v else float("nan")

    # ---------- 抗淨化 ----------
    rows = []
    for r in args.retention:
        for p in sorted(glob.glob(str(Path(r) / "ret_*.csv"))):
            rows += list(csv.DictReader(open(p, encoding="utf-8")))
    ok = {(x["image"], x["condition"]): x["usable"] == "True" for x in rows}
    rconds = [c for c in conds if any(k[1] == c for k in ok)]
    common = [i for i in images if rconds and all(ok.get((i, c), False) for c in rconds)]
    eff = defaultdict(lambda: defaultdict(list))
    per = {}
    # **池化估計。** 逐影像的 3σ 分母閘會排除掉效果最小的條件（`add` 與
    # `phase_rand`），而那兩個正是 site F 最需要比的對照組，取交集等於把
    # 比較對象刪掉。故跨影像與 seed 加總後再取比值；閘的結果照報。
    for x in rows:
        per[(x["image"], x["condition"], x["purifier"])] = x
        eff[x["purifier"]][x["condition"]].append(float(x["effect_mean"]))
    E = {k: {c: st.fmean(v) for c, v in d.items()} for k, d in eff.items()}
    base = E.get("identity", {})
    R = {k: {c: v / base[c] for c, v in d.items() if base.get(c)}
         for k, d in E.items()}

    PEAR, corr = float("nan"), ""
    if len(base) > 2:
        mret = {c: st.fmean([R[k][c] for k in PURIF
                             if k != "identity" and c in R.get(k, {})])
                for c in base}
        X, Y = list(base.values()), [mret[c] for c in base]
        mx, my = st.fmean(X), st.fmean(Y)
        den = math.sqrt(sum((a - mx) ** 2 for a in X) * sum((b - my) ** 2 for b in Y))
        if den > 0:
            PEAR = sum((a - mx) * (b - my) for a, b in zip(X, Y)) / den
        corr = "".join(
            '<tr class="%s"><td>%s</td><td class="n">%.4f</td><td class="n">%.4f</td></tr>'
            % ("hi" if c == "phase" else "", LABEL[c], base[c], mret[c])
            for c in sorted(base, key=lambda c: -base[c]))

    # ---------- 威脅模型 ----------
    ref = args.pur_image if args.pur_image in images else images[0]
    r0 = where[ref]
    x = np.asarray(Image.open(r0 / f"{ref}__orig.png").convert("RGB"),
                   dtype=np.float32) / 255.0
    m = np.asarray(Image.open(r0 / f"{ref}__mask.png").convert("L"),
                   dtype=np.float32)[..., None] / 255.0
    step = lambda n, t: '<div class="op"><span class="no">%s</span>%s</div>' % (n, t)
    ovl = img_of(args.masks, f"{ref}__overlay.png")
    flow = "".join(
        [tile(r0 / f"{ref}__orig.png", "x", "原圖 512²，防禦已加在整張圖上",
              cls="fx", size=300)]
        + ([tile(ovl, "分割", "紅＝主體、藍＝ρ=1.2 保護帶",
                 "Mask R-CNN 自動產生", cls="fx", size=300)] if ovl else [])
        + [step("1", "取遮罩<br>1 = 攻擊方重畫"),
           tile(m.repeat(3, axis=2), "mask",
                "%s ｜ %s" % (res[(ref, conds[0])]["region"],
                             res[(ref, conds[0])]["prompt"]),
                "涵蓋 %.3f" % float(m.mean()), cls="fx", size=300, arr=True),
           step("2", "遮罩區填中灰<br>再編碼成後 4 通道"),
           tile(x * (1 - m) + 0.5 * m, "UNet 收到的條件影像",
                "填 0.5（模型值域的 0）", "填黑會讓模型把洞畫黑",
                cls="fx b", size=300, arr=True),
           step("3", "9 通道 UNet<br>50 步，無 strength"),
           tile(r0 / f"{ref}__gen_orig.png", "未防禦的重畫", "",
                "攻擊強度 %.3f" % agg(conds[0], "attack_strength"),
                cls="fx", size=300)])

    # ---------- 表 ----------
    be = max(agg(c, "gen_lpips") for c in conds)
    body = "".join(
        '<tr class="%s"><td>%s</td><td class="g">%s</td><td class="mono">%s</td>'
        '<td class="n%s">%.4f</td><td class="n">%.4f</td><td class="n">%.4f</td>'
        '<td class="n">%.4f</td><td class="n">%.2f</td></tr>'
        % ("hi" if c == "phase" else "", LABEL[c], ORIGIN[c], BUDGET[c],
           " b" if agg(c, "gen_lpips") == be else "",
           agg(c, "gen_lpips"), agg(c, "gen_lpips_sd"), agg(c, "fid_dists"),
           agg(c, "fid_lpips"), agg(c, "fid_psnr"))
        for c in sorted(conds, key=lambda c: -agg(c, "gen_lpips")))
    tabs = ('<h4>攻擊強度 %.3f · 重畫區 %.3f</h4>'
            '<div class="tw"><table><thead><tr><th>條件</th><th>預算來源</th>'
            '<th>半徑</th><th>效果（遮罩內）</th><th>效果 sd</th><th>DISTS</th>'
            '<th>LPIPS</th><th>PSNR</th></tr></thead><tbody>%s</tbody></table></div>'
            % (agg(conds[0], "attack_strength"),
               agg(conds[0], "mask_coverage"), body))

    rr = []
    for b in ("add", "phase_rand", "dia_r", "apa_weak", "photoguard_c", "mist"):
        if b not in conds:
            continue
        va = [float(res[(i, "phase")]["gen_lpips"]) for i in images]
        vb = [float(res[(i, b)]["gen_lpips"]) for i in images]
        r = st.fmean(va) / st.fmean(vb)
        rr.append('<tr><td>site F ÷ %s</td><td class="n %s">%.3f</td>'
                  '<td class="n">%d/%d</td></tr>'
                  % (LABEL[b], "b" if r > 1 else "", r,
                     sum(p > q for p, q in zip(va, vb)), len(va)))
    ratios = ('<div class="tw"><table><thead><tr><th>比較</th><th>比值</th>'
              '<th>逐圖勝出</th></tr></thead><tbody>%s</tbody></table></div>'
              % "".join(rr))

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
        r = where[img]
        r0w = res[(img, conds[0])]
        g = [tile(r / f"{img}__orig.png", "原圖", "未防禦", cls="ed", size=384),
             tile(r / f"{img}__mask.png", "遮罩", r0w["region"],
                  "涵蓋 %.3f" % float(r0w["mask_coverage"]), cls="ed", size=384),
             tile(r / f"{img}__gen_orig.png", "未防禦的重畫", "",
                  "攻擊強度 %.3f" % float(r0w["attack_strength"]),
                  cls="ed", size=384)]
        for c in conds:
            row = res.get((img, c))
            if not row:
                continue
            g.append(tile(r / f"{img}__{c}__gen_def.png", LABEL[c],
                          "%s · %s" % (BUDGET[c], ORIGIN[c]),
                          "效果 %.3f · DISTS %.4f"
                          % (float(row["gen_lpips"]), float(row["fid_dists"])),
                          cls="ed" + (" hi" if c == "phase" else ""), size=384))
        plates.append('<h3>%s <span class="pr">%s ｜ “%s”</span></h3>'
                      '<div class="ge">%s</div>'
                      % (img, r0w["region"], r0w["prompt"], "".join(g)))

    # ---------- 淨化區段 ----------
    purs = []
    for c in [z for z in ("phase", "add", "phase_rand", "photoguard_c")
              if z in rconds]:
        cells = []
        for k in PURIF:
            pu = args.gallery / f"{args.pur_image}__{c}__{k}__pur.png"
            gn = args.gallery / f"{args.pur_image}__{c}__{k}__gen.png"
            if not (pu.exists() and gn.exists()):
                continue
            rr = per.get((args.pur_image, c, k))
            d = ("效果 %.3f · ret %.2f" % (float(rr["effect_mean"]),
                                           float(rr["retention"]))) if rr else ""
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
        css=CSS, flow=flow, tabs=tabs, ratios=ratios,
        plates="".join(plates),
        purs="".join(purs) or '<p class="na">尚未產出</p>',
        head=head or "<th>—</th>", eff=ptab(E), ret=ptab(R), corr=corr, pear=PEAR,
        nimg=len(images), ncond=len(conds),
        nusable=sum(1 for v in ok.values() if v), ncell=len(ok),
        ncommon=len(common), common="、".join(common) or "—",
        purimg=args.pur_image)
    args.out.write_text(html, encoding="utf-8")
    print("bytes", len(html.encode()), "->", args.out)
    print("影像", images, "| 條件", conds)
    print("retention 可用 %d/%d，共同可用影像 %s"
          % (sum(1 for v in ok.values() if v), len(ok), common))


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
h3{font-size:16px;margin:30px 0 8px;letter-spacing:-.01em}
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
.ge{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
.gu{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:10px}
.pu{background:var(--surf);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.pu img{width:100%;display:block;aspect-ratio:1;object-fit:cover}
.pu img+img{border-top:2px solid var(--acc)}
.pc{padding:6px 8px;font-size:11px;line-height:1.35}
.pc b{display:block;font-weight:650}
.pc .d{display:block;color:var(--mut);font-size:9.5px;margin-top:2px;
font-family:ui-monospace,Consolas,monospace}
.key{display:flex;flex-wrap:wrap;gap:14px;font-size:11.5px;color:var(--mut);margin:8px 0 0}
"""

TEMPLATE = """<title>遮罩重繪防禦圖譜</title>
<style>%(css)s</style>
<div class="wrap">
<header>
<h1>遮罩重繪防禦圖譜 · inpainting</h1>
<div class="meta">
<span><b>批次</b> runs/ip2</span>
<span><b>權重</b> runwayml/stable-diffusion-inpainting（9 通道）</span>
<span><b>場景</b> 主體加配件</span>
<span><b>影像</b> %(nimg)d</span>
<span><b>條件</b> %(ncond)d</span>
<span><b>遮罩</b> Mask R-CNN 自動產生 · ρ = 1.2</span>
<span><b>防禦圖</b> 取自 img2img 批次，加在整張圖上</span>
</div>
</header>

<h2>威脅模型</h2>
<div class="flow">%(flow)s</div>

<h2>抗編輯</h2>
%(tabs)s

<h2>比值</h2>
%(ratios)s

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
<div class="key"><span>抗淨化兩表為<b>池化估計</b>：跨 %(nimg)d 張影像與 3 個 seed
加總效果後再取比值。逐影像的 3σ 分母閘 %(nusable)d/%(ncell)d 格通過，全部條件同時可用的
影像 %(ncommon)d 張（%(common)s）——被閘擋下的正是效果最小的 add 與隨機相位，
取交集會刪掉比較對象，故不使用。</span></div>
</div>"""


if __name__ == "__main__":
    main()
