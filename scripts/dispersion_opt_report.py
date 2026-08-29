"""可學位移場那批的報告頁。**純資料呈現，頁上沒有文字說明。**

色票取自 dataviz 技能參考色盤的前三槽，明暗兩式都跑過
`validate_palette.js --pairs all` 全數通過（亮色 aqua 的對比 WARN 由直接標籤
與下方表格補償，即該技能的 relief 規則）。

三張圖各自的形式依「資料的工作」選：

    收斂        隨步數變化 → 折線
    抗淨化      淨化算子是一條有序的軸 → 折線
    失真對效果  兩個連續量的關係 → 散點（帶直接標籤）

用法：
    python scripts/dispersion_opt_report.py \\
        --tables runs/ip2p_dispersion_opt/tables \\
        --convergence runs/ip2p_dispersion_opt/convergence \\
        --gallery runs/gallery_dispersion_opt \\
        --defense runs/ip2p_dispersion_opt --out report_dispersion_opt.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TAGS = ["d1_opt", "d4_opt", "d8_opt"]
COND = {"d1_opt": "disp_k1_opt", "d4_opt": "disp_k4_opt",
        "d8_opt": "disp_k8_opt"}
SHORT = {"d1_opt": "K=1", "d4_opt": "K=4", "d8_opt": "K=8"}
PUR_LABEL = {"identity": "未淨化", "jpeg90": "JPEG 90", "jpeg75": "JPEG 75",
             "jpeg50": "JPEG 50", "jpeg30": "JPEG 30", "blur1": "模糊 σ1",
             "blur2": "模糊 σ2", "crop_resize0.1": "裁切 10%",
             "crop_resize0.15": "裁切 15%"}
FID = [("fid_dists", "DISTS"), ("fid_lpips", "LPIPS"), ("fid_psnr", "PSNR"),
       ("fid_ssim", "SSIM"), ("fid_vif_p", "VIFp"), ("fid_linf", "L∞"),
       ("fid_rms", "RMS")]
EDIT = [("edit_lpips", "LPIPS"), ("edit_dists", "DISTS"),
        ("edit_psnr", "PSNR"), ("edit_ssim", "SSIM"),
        ("edit_clip_sim", "CLIP"), ("edit_siglip_sim", "SigLIP")]


def rd(p: Path):
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def thumb(path: Path, size: int, q: int = 74) -> str:
    """縮到 `size` 再編成 JPEG 的 data URI。找不到檔就回空字串。"""
    from PIL import Image
    if not path.exists():
        return ""
    im = Image.open(path).convert("RGB")
    if im.width != size:
        im = im.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


INT_COLS = {"median_stopped_at", "early_stopped", "ran_full", "n_images"}


def fmt(v, d=4, key="") -> str:
    """整數欄不加小數：步數與張數寫成 2801 而不是 2801.0000。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{int(round(f))}" if key in INT_COLS else f"{f:.{d}f}"


# ---------------------------------------------------------------- 圖表

def legend(tags) -> str:
    """圖例。兩條以上的序列一律要有，identity 不能只靠顏色。"""
    o = ["<div class='legend'>"]
    for k, t in enumerate(tags, start=1):
        o.append(f"<span class='lg s{k}'><span class='swatch s{k}'></span>"
                 f"{SHORT[t]}</span>")
    o.append("</div>")
    return "".join(o)


def _stagger(items, gap=15.0):
    """把 y 座標相近的直接標籤上下錯開。

    三條線在右端收斂到幾乎同一個高度，標籤會疊在一起；錯開之後仍然是直接
    標籤（不必回頭數圖例），只是位置讓了一點。
    """
    out = []
    for x, y, txt, k in sorted(items, key=lambda v: v[1]):
        while out and y - out[-1][1] < gap:
            y = out[-1][1] + gap
        out.append((x, y, txt, k))
    return out


def _axes(x0, y0, x1, y1) -> str:
    return (f"<line class='ax' x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}'/>"
            f"<line class='ax' x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}'/>")


def convergence_chart(curve) -> str:
    """收斂曲線：固定抽樣的評估值對步數。三條線 ＋ 直接標籤。"""
    W, H = 940, 380
    L, R, T, B = 78, 150, 26, 52
    by = {}
    for t in TAGS:
        pts = sorted((int(r["step"]), float(r["eval_median"]))
                     for r in curve if r["tag"] == t)
        if pts:
            by[t] = pts
    if not by:
        return ""
    xs = [p[0] for v in by.values() for p in v]
    ys = [p[1] for v in by.values() for p in v]
    x1 = max(xs)
    y0, y1 = min(ys) * 0.9995, max(ys) * 1.0005
    sx = lambda x: L + x / max(1, x1) * (W - L - R)
    sy = lambda y: T + (y1 - y) / (y1 - y0) * (H - T - B)
    out = [f"<svg viewBox='0 0 {W} {H}' class='chart' role='img' "
           "aria-label='收斂曲線：評估損失對步數'>"]
    for i in range(5):
        yv = y0 + (y1 - y0) * i / 4
        out.append(f"<line class='grid' x1='{L}' y1='{sy(yv):.1f}' "
                   f"x2='{W - R}' y2='{sy(yv):.1f}'/>")
        out.append(f"<text class='tick end' x='{L - 10}' "
                   f"y='{sy(yv) + 4:.1f}'>{yv:.1f}</text>")
    out.append(_axes(L, T, W - R, H - B))
    for i in range(6):
        xv = x1 * i / 5
        out.append(f"<text class='tick mid' x='{sx(xv):.1f}' "
                   f"y='{H - B + 20}'>{int(xv)}</text>")
    for k, t in enumerate(by, start=1):
        pts = by[t]
        d = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in pts)
        out.append(f"<polyline class='ln s{k}' points='{d}'/>")
        ax, ay = sx(pts[-1][0]), sy(pts[-1][1])
        out.append(f"<circle class='dot s{k}' cx='{ax:.1f}' cy='{ay:.1f}' r='4'/>")
        out.append(f"<text class='lbl s{k}' x='{ax + 11:.1f}' "
                   f"y='{ay + 4:.1f}'>{SHORT[t]} · 停 {pts[-1][0]}</text>")
    out.append(f"<text class='axlab mid' x='{(L + W - R) / 2:.0f}' "
               f"y='{H - 12}'>PGD 步數</text>")
    out.append(f"<text class='axlab' x='{L - 62}' y='{T - 8}'>"
               "評估損失（固定抽樣，中位）</text>")
    out.append("</svg>")
    return "".join(out)


def gain_chart(gain, purs) -> str:
    """抗淨化淨增益：算子是一條有序的軸，折線。"""
    W, H = 940, 372
    L, R, T, B = 78, 132, 26, 78
    ys = [gain[t][p] for t in TAGS if t in gain for p in purs
          if gain[t].get(p) is not None]
    if not ys:
        return ""
    y0, y1 = min(0.0, min(ys)), max(ys) * 1.08
    sx = lambda i: L + i / max(1, len(purs) - 1) * (W - L - R)
    sy = lambda y: T + (y1 - y) / (y1 - y0) * (H - T - B)
    out = [f"<svg viewBox='0 0 {W} {H}' class='chart' role='img' "
           "aria-label='抗淨化淨增益對淨化算子'>"]
    for i in range(5):
        yv = y0 + (y1 - y0) * i / 4
        out.append(f"<line class='grid' x1='{L}' y1='{sy(yv):.1f}' "
                   f"x2='{W - R}' y2='{sy(yv):.1f}'/>")
        out.append(f"<text class='tick end' x='{L - 10}' "
                   f"y='{sy(yv) + 4:.1f}'>{yv:.2f}</text>")
    out.append(_axes(L, T, W - R, H - B))
    for i, p in enumerate(purs):
        x = sx(i)
        out.append(f"<text class='tick rot' x='{x:.1f}' y='{H - B + 18}' "
                   f"transform='rotate(-34 {x:.1f} {H - B + 18})'>"
                   f"{html.escape(PUR_LABEL.get(p, p))}</text>")
    labels = []
    for k, t in enumerate([t for t in TAGS if t in gain], start=1):
        pts = [(i, gain[t][p]) for i, p in enumerate(purs)
               if gain[t].get(p) is not None]
        if not pts:
            continue
        d = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in pts)
        out.append(f"<polyline class='ln s{k}' points='{d}'/>")
        for i, v in pts:
            ttl = f"{SHORT[t]} · {PUR_LABEL.get(purs[i], purs[i])} · {v:.4f}"
            out.append(f"<circle class='dot s{k}' cx='{sx(i):.1f}' "
                       f"cy='{sy(v):.1f}' r='4'><title>{html.escape(ttl)}"
                       "</title></circle>")
        labels.append((sx(pts[-1][0]), sy(pts[-1][1]), SHORT[t], k))
    for x, y, txt, k in _stagger(labels):
        out.append(f"<text class='lbl s{k}' x='{x + 11:.1f}' "
                   f"y='{y + 4:.1f}'>{txt}</text>")
    out.append(f"<text class='axlab' x='{L - 62}' y='{T - 8}'>"
               "淨增益（扣空白地板）</text>")
    out.append("</svg>")
    return "".join(out)


def scatter_chart(fid, edit) -> str:
    """失真對效果：兩個連續量，散點＋直接標籤。"""
    W, H = 460, 372
    L, R, T, B = 70, 56, 26, 78
    pts = [(t, float(fid[t]["fid_dists"]), float(edit[t]["edit_lpips"]))
           for t in TAGS if t in fid and t in edit]
    if not pts:
        return ""
    x1 = max(p[1] for p in pts) * 1.22
    y1 = max(p[2] for p in pts) * 1.22
    sx = lambda x: L + x / x1 * (W - L - R)
    sy = lambda y: T + (y1 - y) / y1 * (H - T - B)
    out = [f"<svg viewBox='0 0 {W} {H}' class='chart' role='img' "
           "aria-label='防禦圖失真對編輯位移'>"]
    for i in range(5):
        yv = y1 * i / 4
        out.append(f"<line class='grid' x1='{L}' y1='{sy(yv):.1f}' "
                   f"x2='{W - R}' y2='{sy(yv):.1f}'/>")
        out.append(f"<text class='tick end' x='{L - 10}' "
                   f"y='{sy(yv) + 4:.1f}'>{yv:.2f}</text>")
    out.append(_axes(L, T, W - R, H - B))
    for i in range(5):
        xv = x1 * i / 4
        out.append(f"<text class='tick mid' x='{sx(xv):.1f}' "
                   f"y='{H - B + 20}'>{xv:.2f}</text>")
    for k, (t, x, y) in enumerate(pts, start=1):
        out.append(f"<circle class='dot s{k}' cx='{sx(x):.1f}' "
                   f"cy='{sy(y):.1f}' r='7'/>")
        out.append(f"<text class='lbl s{k}' x='{sx(x) + 13:.1f}' "
                   f"y='{sy(y) + 4:.1f}'>{SHORT[t]}</text>")
    out.append(f"<text class='axlab mid' x='{(L + W - R) / 2:.0f}' "
               f"y='{H - 30}'>防禦圖 DISTS</text>")
    out.append(f"<text class='axlab' x='{L - 54}' y='{T - 8}'>"
               "編輯位移 LPIPS</text>")
    out.append("</svg>")
    return "".join(out)


def _box(x, y, w, h, t, s="", cls="n") -> str:
    o = [f"<rect class='{cls}' x='{x}' y='{y}' width='{w}' height='{h}' rx='7'/>",
         f"<text class='nt mid' x='{x + w / 2}' "
         f"y='{y + (20 if s else h / 2 + 4)}'>{t}</text>"]
    if s:
        o.append(f"<text class='ns mid' x='{x + w / 2}' y='{y + 37}'>{s}</text>")
    return "".join(o)


def _arrow(pts) -> str:
    d = " ".join(f"{a},{b}" for a, b in pts)
    (px, py), (ex, ey) = pts[-2], pts[-1]
    if px == ex:
        back = ey - 7 if ey > py else ey + 7
        head = f"{ex - 4.5},{back} {ex + 4.5},{back} {ex},{ey}"
    else:
        back = ex - 7 if ex > px else ex + 7
        head = f"{back},{ey - 4.5} {back},{ey + 4.5} {ex},{ey}"
    return f"<polyline class='e' points='{d}'/><polygon class='eh' points='{head}'/>"


def pipeline() -> str:
    """這一批實際跑的每一步運算。"""
    n = [_box(16, 44, 96, 48, "原圖 x", "3×512×512"),
         _arrow([(112, 68), (146, 68)]),
         _box(146, 44, 150, 48, "反射填補 ＋ unfold", "32×32 · hop 8 · 61²"),
         _arrow([(296, 68), (330, 68)]),
         _box(330, 44, 132, 48, "× Hann ＋ rfft2", "3×3721×32×17"),
         _box(146, 150, 150, 48, "頻帶編號 k(ω)", "log₂ 半徑等分 · r ≥ 0.12"),
         _box(146, 226, 150, 48, "紋理閘 ＋ 帶價", "結構張量 · q̄ₖ^0.25"),
         _arrow([(64, 92), (64, 174), (146, 174)]),
         _arrow([(64, 174), (64, 250), (146, 250)]),
         _box(330, 188, 132, 48, "θ = −2π⟨f, uₖ⟩", "平移定理 · 無近似", "n key"),
         _arrow([(296, 174), (313, 174), (313, 212), (330, 212)]),
         _arrow([(296, 250), (313, 250), (313, 212), (330, 212)]),
         "<polygon class='pm' points='512,188 598,188 586,236 500,236'/>",
         "<text class='pt mid' x='549' y='217'>uₖ(b) 可學</text>",
         _arrow([(462, 212), (500, 212)]),
         "<circle class='op' cx='549' cy='68' r='15'/>",
         "<text class='opt mid' x='549' y='74'>⊗</text>",
         _arrow([(462, 68), (534, 68)]),
         _arrow([(549, 188), (549, 83)]),
         "<text class='el mid' x='590' y='140'>exp(iθ)</text>",
         _box(616, 44, 138, 48, "irfft2 × Hann", "重疊相加 ÷ OLA(w²)"),
         _arrow([(564, 68), (616, 68)]),
         _box(788, 44, 118, 48, "防禦圖 x′", "3×512×512", "n out"),
         _arrow([(754, 68), (788, 68)]),
         _box(788, 150, 118, 56, "淨化算子 T", "JPEG · 模糊 · 裁切", "n alt"),
         _arrow([(847, 92), (847, 150)]),
         _box(788, 250, 118, 48, "IP2P 編輯", "s_T 7.5 · s_I 1.5", "n alt"),
         _arrow([(847, 206), (847, 250)])]
    return ("<svg viewBox='0 0 940 316' class='arch' role='img' "
            "aria-label='本批的運算流程'>" + "".join(n) + "</svg>")


def table(rows, cols) -> str:
    o = ["<table><thead><tr><th>條件</th>"]
    o += [f"<th>{html.escape(c[1])}</th>" for c in cols]
    o.append("</tr></thead><tbody>")
    for i, r in enumerate(rows, start=1):
        o.append(f"<tr><td class='name s{i}'><span class='swatch s{i}'></span>"
                 f"{html.escape(str(r['label']))}</td>")
        o += [f"<td>{fmt(r.get(c[0]), key=c[0])}</td>" for c in cols]
        o.append("</tr>")
    o.append("</tbody></table>")
    return "".join(o)


def build_panels(args, gain, purs, names, ds):
    tabs, panels = [], []
    for i, nm in enumerate(names):
        active = " active" if i == 0 else ""
        tabs.append(f"<button class='tab{active}' data-i='{i}'>"
                    f"#{i + 1:02d}</button>")
        orig = thumb(args.defense / TAGS[0] / f"{nm}__orig.png", args.embed)
        eo = thumb(args.defense / TAGS[0] /
                   f"{nm}__{COND[TAGS[0]]}__edit_orig.png", args.embed)
        cells = ["<div class='rowlab hdr'></div>"]
        cells += [f"<div class='colhdr'>{html.escape(PUR_LABEL.get(p, p))}</div>"
                  for p in purs]
        for k, t in enumerate(TAGS, start=1):
            c = COND[t]
            cells.append(f"<div class='rowlab s{k}'>"
                         f"<span class='swatch s{k}'></span>{SHORT[t]}</div>")
            for p in purs:
                g = gain.get(t, {}).get(p)
                ttl = f"{SHORT[t]} · {PUR_LABEL.get(p, p)}"
                if g is not None:
                    ttl += f" · 淨增益 {g:.4f}"
                if p == "identity":
                    e = thumb(args.defense / t / f"{nm}__{c}__edit_def.png",
                              args.embed)
                    pu = thumb(args.defense / t / f"{nm}__{c}__def.png",
                               args.embed)
                else:
                    e = thumb(args.gallery / t /
                              f"{nm}__{c}__{p}__edit_def.png", args.embed)
                    pu = thumb(args.gallery / t / f"{nm}__{c}__{p}__pur.png",
                               args.embed)
                if not e:
                    cells.append("<div class='cell miss'><span>缺</span></div>")
                    continue
                img_p = (f"<img class='p' src='{pu}' alt='' loading='lazy'>"
                         if pu else "")
                cells.append(
                    f"<div class='cell' title='{html.escape(ttl)}'>"
                    f"<img class='e' src='{e}' alt='{html.escape(ttl)}' "
                    f"loading='lazy'>{img_p}</div>")
        on = " on" if i == 0 else ""
        prompt = html.escape(ds.get(nm, {}).get("prompt", ""))
        panels.append(
            f"<section class='panel{on}' data-i='{i}'>"
            f"<div class='imhead'><span class='id'>#{i + 1:02d} "
            f"{html.escape(nm)}</span><span class='prompt'>{prompt}</span></div>"
            f"<div class='body'><div class='ref'>"
            f"<div class='cell'><img src='{orig}' alt='原圖' loading='lazy'></div>"
            f"<div class='reflab'>原圖</div>"
            f"<div class='cell'><img src='{eo}' alt='未防禦的編輯' "
            f"loading='lazy'></div>"
            f"<div class='reflab'>未防禦的編輯</div></div>"
            f"<div class='matrix' style='--cols:{len(purs)}'>"
            + "".join(cells) + "</div></div></section>")
    return tabs, panels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--convergence", type=Path, required=True)
    ap.add_argument("--gallery", type=Path, required=True)
    ap.add_argument("--defense", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--embed", type=int, default=384,
                    help="影像**內嵌**的畫素數")
    ap.add_argument("--thumb", type=int, default=132,
                    help="頁上顯示的起始邊長，只是 CSS 的初值")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from apa_baseline import load_dataset

    T = json.loads((args.tables / "tables.json").read_text(encoding="utf-8"))
    fid = {r["tag"]: r for r in T["fidelity"] if r["tag"] in TAGS}
    edit = {r["tag"]: r for r in T["edit"] if r["tag"] in TAGS}
    gain = {t: v for t, v in T["gain"].items() if t in TAGS}
    purs, label = T["purifiers"], T["label"]
    conv = rd(args.convergence / "curve.csv")
    csum = {r["tag"]: r for r in rd(args.convergence / "summary.csv")}

    names = [ln.strip() for ln in
             args.images.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ds = {d["name"]: d for d in load_dataset(args.data, prompt_index=0)}
    tabs, panels = build_panels(args, gain, purs, names, ds)

    conv_rows = [{"label": label.get(t, t), **csum[t]}
                 for t in TAGS if t in csum]
    gain_rows = [{"label": label.get(t, t),
                  **{p: gain[t].get(p) for p in purs}}
                 for t in TAGS if t in gain]

    out = TEMPLATE.format(
        pipeline=pipeline(),
        conv_chart=convergence_chart(conv),
        gain_chart=gain_chart(gain, purs),
        scatter=scatter_chart(fid, edit),
        legend_conv=legend([t for t in TAGS if t in csum]),
        legend_gain=legend([t for t in TAGS if t in gain]),
        legend_sc=legend([t for t in TAGS if t in fid]),
        t_conv=table(conv_rows,
                     [("median_stopped_at", "停止步數（中位）"),
                      ("early_stopped", "早停"), ("ran_full", "跑滿"),
                      ("tail_change_median", "尾段相對變化")]),
        t_fid=table([{"label": label.get(t, t), **fid[t]}
                     for t in TAGS if t in fid], FID),
        t_edit=table([{"label": label.get(t, t), **edit[t]}
                      for t in TAGS if t in edit], EDIT),
        t_gain=table(gain_rows, [(p, PUR_LABEL.get(p, p)) for p in purs]),
        tabs="".join(tabs), panels="".join(panels),
        thumb=args.thumb, embed=args.embed, nimg=len(names), npur=len(purs))
    args.out.write_text(out, encoding="utf-8")
    mb = args.out.stat().st_size / 1048576
    print(f"寫出 {args.out}（{mb:.1f} MB，{len(names)} 張 × "
          f"{len(TAGS)} 條件 × {len(purs)} 算子）")


TEMPLATE = """<title>可學位移場：三個色散度</title>
<style>
:root{{
 --bg:#f7f7f5; --surface:#fcfcfb; --line:#e5e4df; --line-2:#cfcec7;
 --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#84837c;
 --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
 --thumb:{thumb}px;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
 --bg:#131312; --surface:#1a1a19; --line:#302f2c; --line-2:#46453f;
 --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#918f86;
 --s1:#3987e5; --s2:#d95926; --s3:#199e70;
}}}}
:root[data-theme="dark"]{{
 --bg:#131312; --surface:#1a1a19; --line:#302f2c; --line-2:#46453f;
 --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#918f86;
 --s1:#3987e5; --s2:#d95926; --s3:#199e70;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-size:15px;
 font-family:"IBM Plex Sans","Helvetica Neue","PingFang TC",
 "Microsoft JhengHei",sans-serif;line-height:1.6;-webkit-font-smoothing:antialiased}}
h1,h2{{font-family:Archivo,"Helvetica Neue","PingFang TC",sans-serif;
 letter-spacing:-.018em;margin:0;font-weight:640;text-wrap:balance}}
h1{{font-size:29px}} h2{{font-size:17px}}
.mono{{font-family:ui-monospace,Menlo,Consolas,monospace;
 font-variant-numeric:tabular-nums}}
header{{border-bottom:1px solid var(--line);background:var(--surface);
 padding:26px 30px 20px}}
.hwrap,.sec,.tables{{max-width:1180px;margin:0 auto}}
.eyebrow{{font:600 11px/1 ui-monospace,monospace;letter-spacing:.15em;
 color:var(--ink-3);text-transform:uppercase;margin:0 0 10px}}
.meta{{color:var(--ink-2);font-size:12.5px;margin:10px 0 0}}
.sec{{padding:30px}}
.sec>h2{{margin-bottom:16px;padding-bottom:9px;border-bottom:1px solid var(--line)}}
.card{{background:var(--surface);border:1px solid var(--line);
 border-radius:11px;padding:18px}}
.card.narrow{{max-width:560px}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:0 0 10px 4px;
 font-size:12.5px;color:var(--ink-2)}}
.legend .lg{{display:inline-flex;align-items:center;gap:0}}
.chart,.arch{{width:100%;height:auto;display:block;overflow:visible}}
.chart .grid{{stroke:var(--line);stroke-width:1}}
.chart .ax{{stroke:var(--line-2);stroke-width:1.2}}
.chart .tick{{fill:var(--ink-3);font:11px ui-monospace,monospace}}
.chart .axlab{{fill:var(--ink-3);font:11.5px sans-serif}}
.chart .mid{{text-anchor:middle}}
.chart .end,.chart .rot{{text-anchor:end}}
.chart .ln{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.chart .dot{{stroke:var(--surface);stroke-width:2}}
.chart .lbl{{font:600 12px sans-serif}}
.s1{{--c:var(--s1)}} .s2{{--c:var(--s2)}} .s3{{--c:var(--s3)}}
.chart .ln.s1,.chart .ln.s2,.chart .ln.s3{{stroke:var(--c)}}
.chart .dot.s1,.chart .dot.s2,.chart .dot.s3{{fill:var(--c)}}
.chart .lbl.s1,.chart .lbl.s2,.chart .lbl.s3{{fill:var(--c)}}
.arch .n{{fill:var(--surface);stroke:var(--line-2);stroke-width:1.3}}
.arch .n.key{{fill:color-mix(in srgb,var(--s1) 12%,var(--surface));
 stroke:var(--s1);stroke-width:2}}
.arch .n.alt{{fill:color-mix(in srgb,var(--s2) 10%,var(--surface));
 stroke:var(--s2);stroke-width:1.6}}
.arch .n.out{{fill:color-mix(in srgb,var(--ink) 6%,var(--surface));
 stroke:var(--ink-2);stroke-width:1.6}}
.arch .nt{{fill:var(--ink);font:600 12px sans-serif}}
.arch .ns{{fill:var(--ink-3);font:10px ui-monospace,monospace}}
.arch .e{{fill:none;stroke:var(--ink-3);stroke-width:1.2}}
.arch .eh{{fill:var(--ink-3)}}
.arch .el{{fill:var(--ink-3);font:10.5px ui-monospace,monospace}}
.arch .pm{{fill:color-mix(in srgb,var(--s1) 20%,var(--surface));
 stroke:var(--s1);stroke-width:1.3}}
.arch .pt{{fill:var(--ink);font:600 11.5px sans-serif}}
.arch .op{{fill:var(--surface);stroke:var(--ink-2);stroke-width:1.6}}
.arch .opt{{fill:var(--ink);font:600 15px sans-serif}}
.arch .mid{{text-anchor:middle}}
.scroll{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px;
 font-variant-numeric:tabular-nums}}
th,td{{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);
 font-family:ui-monospace,Menlo,monospace;white-space:nowrap}}
th{{color:var(--ink-3);font:600 11px sans-serif;letter-spacing:.04em;
 text-transform:uppercase;border-bottom:1px solid var(--line-2)}}
th:first-child,td:first-child{{text-align:left;font-family:inherit}}
td.name{{color:var(--ink);font-weight:600}}
.swatch{{display:inline-block;width:9px;height:9px;border-radius:2px;
 background:var(--c);margin-right:8px}}
.tables{{padding:0 30px 44px}}
.tables h2{{margin:30px 0 14px;padding-bottom:9px;
 border-bottom:1px solid var(--line)}}
.bar{{display:flex;gap:16px;align-items:center;flex-wrap:wrap;
 background:var(--surface);border:1px solid var(--line);border-radius:11px;
 padding:11px 14px;margin-bottom:14px;position:sticky;top:0;z-index:5}}
.tabs{{display:flex;gap:4px;flex-wrap:wrap}}
.tab{{font:600 11.5px ui-monospace,monospace;padding:5px 9px;border-radius:6px;
 border:1px solid var(--line-2);background:transparent;color:var(--ink-2);
 cursor:pointer}}
.tab.active{{background:var(--ink);color:var(--bg);border-color:var(--ink)}}
.ctl{{display:flex;gap:9px;align-items:center;font-size:12.5px;color:var(--ink-2)}}
input[type=range]{{width:130px;accent-color:var(--s1)}}
.seg button{{font:600 11.5px sans-serif;padding:5px 11px;
 border:1px solid var(--line-2);background:transparent;color:var(--ink-2);
 cursor:pointer}}
.seg button:first-child{{border-radius:6px 0 0 6px}}
.seg button:last-child{{border-radius:0 6px 6px 0;border-left:0}}
.seg button.on{{background:var(--ink);color:var(--bg);border-color:var(--ink)}}
.panel{{display:none}} .panel.on{{display:block}}
.imhead{{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;
 margin:6px 0 12px}}
.imhead .id{{font:600 13px ui-monospace,monospace}}
.imhead .prompt{{color:var(--ink-2);font-size:13px;font-style:italic}}
.body{{display:flex;gap:20px;align-items:flex-start;overflow-x:auto;
 padding-bottom:8px}}
.ref{{display:grid;grid-template-columns:var(--thumb);gap:5px;flex:0 0 auto}}
.reflab{{font-size:10.5px;color:var(--ink-3);text-align:center;margin-bottom:7px}}
.matrix{{display:grid;grid-template-columns:auto repeat(var(--cols),var(--thumb));
 gap:5px;align-items:center}}
.colhdr{{font-size:10.5px;color:var(--ink-3);text-align:center}}
.rowlab{{font-size:11.5px;color:var(--ink-2);padding-right:10px;
 white-space:nowrap;font-weight:600}}
.rowlab.hdr{{height:0}}
.cell{{width:var(--thumb);height:var(--thumb);border-radius:5px;overflow:hidden;
 background:#7a7a76;border:1px solid var(--line)}}
.cell img{{width:100%;height:100%;object-fit:cover;display:block}}
.cell img.p{{display:none}}
body.showpur .cell img.e{{display:none}}
body.showpur .cell img.p{{display:block}}
.cell.miss{{display:flex;align-items:center;justify-content:center;
 color:var(--ink-3);font-size:11px}}
:focus-visible{{outline:2px solid var(--s1);outline-offset:2px}}
</style>
<header><div class="hwrap">
<p class="eyebrow">白盒頻域／相位抗文字編輯防禦 · InstructPix2Pix</p>
<h1>可學位移場：三個色散度</h1>
<p class="meta mono">{nimg} 張 · 3 條件 · {npur} 算子 · 1 種子 · latent_norm ·
半徑 8 · 上限 5000 步 · early stop</p>
</div></header>

<section class="sec"><h2>Pipeline</h2>
<div class="card scroll">{pipeline}</div></section>

<section class="sec"><h2>收斂</h2>
<div class="card">{legend_conv}{conv_chart}</div>
<div class="card scroll" style="margin-top:14px">{t_conv}</div></section>

<section class="sec"><h2>抗淨化的淨增益（扣空白地板）</h2>
<div class="card">{legend_gain}{gain_chart}</div></section>

<section class="sec"><h2>防禦圖失真對編輯位移</h2>
<div class="card narrow">{legend_sc}{scatter}</div></section>

<section class="sec">
<h2>逐張影像</h2>
<div class="bar">
<div class="tabs">{tabs}</div>
<div class="ctl"><label for="sz">縮圖</label>
<input id="sz" type="range" min="56" max="{embed}" step="4" value="{thumb}">
<span class="mono" id="szv">{thumb}</span>px</div>
<div class="seg"><button id="be" class="on">編輯輸出</button>
<button id="bp">淨化後的防禦圖</button></div>
</div>
{panels}
</section>

<section class="tables">
<h2>防禦圖的失真</h2><div class="card scroll">{t_fid}</div>
<h2>編輯輸出的位移與語意</h2><div class="card scroll">{t_edit}</div>
<h2>抗淨化的淨增益</h2><div class="card scroll">{t_gain}</div>
</section>

<script>
const tabs=[...document.querySelectorAll('.tab')];
const panels=[...document.querySelectorAll('.panel')];
tabs.forEach(t=>t.onclick=()=>{{
  tabs.forEach(x=>x.classList.remove('active'));
  panels.forEach(p=>p.classList.remove('on'));
  t.classList.add('active');
  panels[+t.dataset.i].classList.add('on');
}});
const sz=document.getElementById('sz'), szv=document.getElementById('szv');
const set=v=>{{document.documentElement.style.setProperty('--thumb',v+'px');
  szv.textContent=v; try{{localStorage.setItem('dispthumb',v);}}catch(e){{}}}};
sz.oninput=()=>set(sz.value);
try{{const v=localStorage.getItem('dispthumb'); if(v){{sz.value=v; set(v);}}}}catch(e){{}}
const be=document.getElementById('be'), bp=document.getElementById('bp');
be.onclick=()=>{{document.body.classList.remove('showpur');
  be.classList.add('on'); bp.classList.remove('on');}};
bp.onclick=()=>{{document.body.classList.add('showpur');
  bp.classList.add('on'); be.classList.remove('on');}};
</script>
"""


if __name__ == "__main__":
    main()
