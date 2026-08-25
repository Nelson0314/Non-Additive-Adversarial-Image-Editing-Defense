"""主線報告頁：**條件 × 淨化算子的矩陣**，一個畫面看完一張影像的全部結果。

為什麼換掉五欄大圖
────────────────────────────────────────────────────────────────────
先前的版面是「每個條件一張五欄大圖、原生 512 不縮放」。它答得了「這一格擋下
了沒有」，但答不了實際要做的比較——**同一張影像上，十個條件在九個淨化算子
底下誰活得下來**。那個比較橫跨 10 個條件 × 9 個算子 = 90 格，攤成 512 的大圖
要捲十幾個畫面，等於沒辦法比。

本版把它壓成一張矩陣：**列是條件、欄是淨化算子**，格子裡是**淨化之後的編輯
輸出**——也就是判「擋下與否」看的那一張。影像大小由頁內滑桿控制（預設
128 px，記在 localStorage），影像之間用頁籤切換，不靠捲動。

三個相對於前一版的補充：

1. **`identity` 那一欄就是「防禦圖的編輯、未經任何淨化」**，先前的版面沒有它，
   於是看不出淨化到底還原了多少。它排在最左，是同一列其餘八欄的參照點。
2. **九個淨化算子全部在頁上**，不是只有 jpeg75。
3. 切換鈕可以把整張矩陣從「編輯輸出」換成「淨化後的防禦圖」，用來分辨
   「編輯被擋下」與「圖被淨化算子本身毀掉」。

判準與資料來源不變：淨增益一律扣空白地板（`DECISIONS.md`），語意讀數是
**編輯輸出對編輯輸出的餘弦相似度，不含任何文字**。

用法：
    python scripts/mainline_matrix.py --tables runs/ip2p_mainline/tables \\
        --gallery runs/gallery_mainline --defense runs/ip2p_mainline \\
        --out runs/ip2p_mainline/report_matrix.html
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

import mainline_architecture as MA  # noqa: E402
import mainline_charts as MC  # noqa: E402

THUMB = 128          # 頁上顯示的起始邊長（CSS，可由滑桿改）
EMBED = 512          # 編輯輸出內嵌的畫素數：原生，放大不糊
EMBED_PUR = 256      # 淨化後防禦圖內嵌的畫素數：脈絡用，壓小換檔案大小
QUALITY = 72
PUR_LABEL = {
    "identity": "未淨化",
    "jpeg90": "JPEG 90", "jpeg75": "JPEG 75",
    "jpeg50": "JPEG 50", "jpeg30": "JPEG 30",
    "blur1": "模糊 σ1", "blur2": "模糊 σ2",
    "crop_resize0.1": "裁切 10%", "crop_resize0.15": "裁切 15%",
}
FID_LABEL = {"fid_dists": "DISTS", "fid_lpips": "LPIPS", "fid_psnr": "PSNR",
             "fid_ssim": "SSIM", "fid_vif_p": "VIFp", "fid_linf": "L∞",
             "fid_rms": "RMS"}
EDIT_LABEL = {"edit_lpips": "LPIPS", "edit_dists": "DISTS", "edit_psnr": "PSNR",
              "edit_ssim": "SSIM", "edit_vif_p": "VIFp",
              "edit_clip_sim": "CLIP", "edit_siglip_sim": "SigLIP"}
# 「越高越好」的欄位；其餘一律越低越好。失真表上 PSNR／SSIM／VIFp 是高的好，
# 位移表上只有 LPIPS／DISTS 是高的好（位移越大代表編輯被推得越遠）。
FID_HIGHER = {"fid_psnr", "fid_ssim", "fid_vif_p"}
EDIT_HIGHER = {"edit_lpips", "edit_dists"}


def thumb(path: Path, size: int, quality: int) -> str:
    """讀一張圖、縮到 `size`、回傳 data URI。缺檔回傳空字串由呼叫端標記。

    **`size` 是內嵌的畫素數，與頁上的顯示尺寸無關。** 顯示尺寸由 CSS 的
    `--thumb` 控制，滑桿與「符合畫面」只動它。兩者脫鉤的理由：先前把內嵌解析度
    綁在顯示尺寸上（96 px），滑桿一拉大就是放大 96 px 的圖，糊到判不出東西。
    現在編輯輸出一律內嵌**原生 512**，怎麼放大都是原始畫素。

    `size = 0` 表示原樣不縮。
    """
    from PIL import Image

    if not path.exists():
        return ""
    im = Image.open(path).convert("RGB")
    if size and im.width != size:
        im = im.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def one(paths, size, quality) -> str:
    """在候選路徑裡取第一個存在的。命名有三種來源，必須逐一試。"""
    for p in paths:
        if p.exists():
            return thumb(p, size, quality)
    return ""


def cell(src: str, alt: str) -> str:
    if not src:
        return "<div class='cell miss' title='缺圖'><span>缺</span></div>"
    return (f"<div class='cell'><img src='{src}' alt='{html.escape(alt)}'"
            " loading='lazy'></div>")


def fmt(v, digits=4) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def build_panels(args, T, names, ds, tags):
    label, purs, gain = T["label"], purs_of(args, T), T["gain"]
    S, Q = args.embed, args.quality
    SP = args.embed_purified
    tabs, panels = [], []
    for i, n in enumerate(names):
        d = ds[n]
        act = " active" if i == 0 else ""
        on = " on" if i == 0 else ""
        orig = one([args.defense / tags[0] / f"{n}__orig.png"], S, Q)
        eorig = one(sorted((args.defense / tags[0]).glob(
            f"{n}__*__edit_orig.png")), S, Q)
        tabs.append(f"<button class='tab{act}' data-i='{i}'>#{i + 1:02d}</button>")

        head = ("<div class='ref'>"
                + cell(orig, "原圖") + "<div class='reflab'>原圖</div>"
                + cell(eorig, "原圖的編輯")
                + "<div class='reflab'>原圖的編輯<br><small>未防禦</small></div>"
                + "</div>")

        rows = []
        for t in tags:
            cells = []
            for p in purs:
                e = one(sorted((args.gallery / t).glob(
                    f"{n}__*__{p}__edit_def.png")), S, Q)
                # 淨化後的防禦圖內嵌得小一些：它是脈絡（看算子把圖弄成什麼樣），
                # 判「擋下與否」看的是編輯那一張。兩組都用 512 會到 40 MB。
                pu = "" if args.no_purified else one(sorted(
                    (args.gallery / t).glob(f"{n}__*__{p}__pur.png")), SP, Q)
                g = gain.get(t, {}).get(p)
                title = f"{label.get(t, t)} · {PUR_LABEL.get(p, p)}"
                if g is not None:
                    title += f" · 淨增益 {g:.4f}"
                if not e:
                    cells.append("<div class='cell miss' title='尚未跑完'>"
                                 "<span>缺</span></div>")
                    continue
                img = (f"<img class='e' src='{e}' alt='{html.escape(title)}'"
                       " loading='lazy'>")
                if pu:
                    img += f"<img class='p' src='{pu}' alt='' loading='lazy'>"
                cells.append(f"<div class='cell' title='{html.escape(title)}'>"
                             f"{img}</div>")
            cls = "ours" if t.startswith("ours") else "rival"
            rows.append(f"<div class='rowlab {cls}'>"
                        f"{html.escape(label.get(t, t))}</div>" + "".join(cells))

        headrow = "<div class='rowlab hdr'></div>" + "".join(
            f"<div class='colhdr'>{html.escape(PUR_LABEL.get(p, p))}</div>"
            for p in purs)
        grid = (f"<div class='matrix' style='--cols:{len(purs)}'>"
                f"{headrow}{''.join(rows)}</div>")
        panels.append(
            f"<section class='panel{on}' data-i='{i}'>"
            f"<div class='imhead'><span class='id'>#{i + 1:02d} "
            f"{html.escape(n)}</span>"
            f"<span class='prompt'>編輯指令：{html.escape(d['prompt'])}</span>"
            f"</div><div class='body'>{head}{grid}</div></section>")
    return tabs, panels


def purs_of(args, T):
    """要顯示的淨化算子。`--purifiers` 未給時用 CSV 裡的全部。"""
    if not args.purifiers:
        return T["purifiers"]
    unknown = [p for p in args.purifiers if p not in T["purifiers"]]
    if unknown:
        raise SystemExit(f"未知的淨化算子：{unknown}；可用：{T['purifiers']}")
    return list(args.purifiers)


def rank_ours(T, purs, n):
    """本方法的條件依**保留欄位的淨增益均值**排名，取前 n。

    為什麼用均值而不是單看 jpeg75：頁上顯示哪幾欄，排名就該由哪幾欄決定，
    否則會出現「排進來的條件在頁上看起來不如被排掉的」。兩種排法在第三名
    上不一致（均值選 r2.0 無量化、jpeg75 選 r0.9 量化），均值那一組同時保住
    「同半徑量化與否」與「純相位對相位＋增益」兩個對照，資訊量較大。
    """
    import statistics as _st
    gain = T["gain"]
    cand = [t for t in T["order"] if t.startswith("ours") and t in gain]
    if n <= 0 or n >= len(cand):
        return cand, []
    scored = sorted(cand, key=lambda t: -_st.fmean(gain[t][p] for p in purs))
    keep = set(scored[:n])
    return ([t for t in T["order"] if t in keep],
            [t for t in cand if t not in keep])


def render_table(rows_src, cols, labels, higher, extra_col=None):
    """綠色標該欄最佳，欄名旁的箭頭標「對防禦有利的方向」。

    **箭頭與綠色由同一個 `higher` 集合推出**，所以兩者不可能各說各話——
    先前的版本只有顏色沒有箭頭，讀者得自己記得哪一欄是越低越好，而位移表上
    PSNR 越低越好、失真表上 PSNR 越高越好，記錯就把結論讀反。
    """
    best = {}
    for k in cols:
        vals = [r[k] for r in rows_src if r.get(k) is not None]
        best[k] = (max(vals) if k in higher else min(vals)) if vals else None
    arrow = lambda k: ("<span class='up'>↑</span>" if k in higher
                       else "<span class='down'>↓</span>")
    out = ["<table><thead><tr><th>條件</th>"]
    out += [f"<th>{labels[k]} {arrow(k)}</th>" for k in cols]
    if extra_col:
        out.append(f"<th>{extra_col} <span class='up'>↑</span></th>")
    out.append("</tr></thead><tbody>")
    for r in rows_src:
        cls = "ours" if r["tag"].startswith("ours") else "rival"
        out.append(f"<tr class='{cls}'><td class='name'>"
                   f"{html.escape(r['label'])}</td>")
        for k in cols:
            v = r.get(k)
            mark = " class='best'" if v is not None and v == best[k] else ""
            out.append(f"<td{mark}>{fmt(v)}</td>")
        if extra_col:
            out.append(f"<td>{html.escape(str(r.get('siglip_blocked', '—')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--gallery", type=Path, default=Path("runs/gallery_mainline"))
    ap.add_argument("--defense", type=Path, default=Path("runs/ip2p_mainline"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--thumb", type=int, default=THUMB,
                    help="頁上**顯示**的邊長，只是 CSS 的起始值")
    ap.add_argument("--embed", type=int, default=EMBED,
                    help="編輯輸出**內嵌**的畫素數。512 = 原生，放大不糊")
    ap.add_argument("--embed-purified", type=int, default=EMBED_PUR,
                    help="淨化後防禦圖內嵌的畫素數。它是脈絡不是判準，"
                         "壓小一半換檔案大小")
    ap.add_argument("--purifiers", nargs="*", default=None,
                    help="只顯示這幾個算子。未給時全上")
    ap.add_argument("--top-ours", type=int, default=0,
                    help="本方法只留前 N 名（依保留欄位的淨增益均值）。"
                         "0 = 全留。對照組不受影響")
    ap.add_argument("--quality", type=int, default=QUALITY)
    ap.add_argument("--no-purified", action="store_true",
                    help="不嵌入「淨化後的防禦圖」那一組，檔案小一半")
    ap.add_argument("--chart-drop", nargs="*", default=[],
                    help="**只**從圖表裡拿掉的條件。與 `--drop` 分開的理由："
                         "逐張矩陣是被影像撐爆的，圖表不是——把回答得了"
                         "「贏是不是因為失真高」的低失真條件從圖表上剪掉，"
                         "等於把答案剪掉。抗淨化沒跑完的條件要放進這裡，"
                         "六張與十張不可畫在同一組軸上。")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="不放進報告頁的條件（**CSV 仍保留**）。用途是把族內"
                         "買不到東西的校準點拿掉：判準是「多付的失真沒有換到"
                         "可辨識的位移」，不是印象。")
    ap.add_argument("--drop-note", default="",
                    help="上面那些條件為什麼被拿掉，逐字寫在頁尾")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="抗淨化影像覆蓋率低於此值的條件不進矩陣——整列都是"
                         "「缺」只會擠掉版面。**它們仍然留在下方三張表上**，"
                         "而且表三會標出缺的是哪幾列")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from apa_baseline import load_dataset

    T = json.loads((args.tables / "tables.json").read_text(encoding="utf-8"))
    label, gain = T["label"], T["gain"]
    purs = purs_of(args, T)
    keep_ours, cut_ours = rank_ours(T, purs, args.top_ours)
    if cut_ours:
        args.drop = list(args.drop) + cut_ours
    fid = {r["tag"]: r for r in T["fidelity"]}
    edit = {r["tag"]: r for r in T["edit"]}
    need = len(purs) * len([ln for ln in
                            args.images.read_text(encoding="utf-8").splitlines()
                            if ln.strip()])
    order = [t for t in T["order"] if t not in set(args.drop)]
    tags, thin = [], []
    for t in order:
        d = args.gallery / t
        if not d.is_dir():
            continue
        cov = len(list(d.glob("*__edit_def.png"))) / need
        (tags if cov >= args.min_coverage else thin).append((t, cov))
    thin_note = "".join(
        f"<li>{html.escape(label.get(t, t))}——抗淨化只跑到 {c:.0%}，"
        "仍在跑</li>" for t, c in thin)
    tags = [t for t, _ in tags]

    names = [ln.strip() for ln in
             args.images.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ds = {d["name"]: d for d in load_dataset(args.data)}

    tabs, panels = build_panels(args, T, names, ds, tags)

    # 表上要含**全部**條件，包括還沒跑抗淨化、因此不在矩陣裡的那幾個。
    fid_rows = [fid[t] for t in order if t in fid]
    edit_rows = [edit[t] for t in order if t in edit]
    t1 = render_table(fid_rows, list(FID_LABEL), FID_LABEL, FID_HIGHER)
    t2 = render_table(edit_rows, list(EDIT_LABEL), EDIT_LABEL, EDIT_HIGHER,
                      extra_col="擋下")
    g_rows = [{"tag": t, "label": label.get(t, t),
               **{p: gain[t].get(p) for p in purs}}
              for t in order if t in gain]
    t3 = render_table(g_rows, purs, {p: PUR_LABEL.get(p, p) for p in purs},
                      set(purs))

    missing = [label.get(t, t) for t in order if t in fid and t not in gain]
    warn = (f"<p class='warn'>抗淨化尚未跑完，表三缺：{'、'.join(missing)}。"
            "這幾列<b>不是 0，是沒有資料</b>，不可當成失敗讀。</p>"
            if missing else "")
    if thin_note:
        warn += ("<p class='warn'>下列條件因抗淨化尚未跑到一半而不進矩陣"
                 f"（表上仍在）：<ul class='warn'>{thin_note}</ul></p>")

    dropped = ""
    if args.drop:
        names_d = "、".join(html.escape(label.get(t, t)) for t in args.drop)
        why = (html.escape(args.drop_note) if args.drop_note
               else "族內買不到東西的校準點")
        dropped = (f"<p class='note'><b>本頁不列</b>：{names_d}。理由：{why}。"
                   "數值仍在 <span class='mono'>runs/ip2p_mainline/tables/"
                   "</span> 的 CSV 裡，沒有刪除。</p>")
    # **圖表用的是全部條件，不受 `--drop` 影響。** `--drop` 管的是逐張矩陣
    # 與表格；圖表被剪掉低失真的那幾組時，「等失真下誰比較強」就看不出來了。
    chart_order = [t for t in T["order"]
                   if t in gain and t not in set(args.chart_drop)]
    ch_tags = chart_order
    charts = {
        "curve": MC.quality_curve(ch_tags, label, gain),
        "bar": MC.distortion_bar(chart_order, label, fid, gain, "jpeg30"),
        "scatter": MC.tradeoff(ch_tags, label, fid, gain, "jpeg30"),
        "arch_ours": MA.ours(),
        "arch_dct": MA.dct(),
    }
    out = TEMPLATE.format(
        tabs="".join(tabs), panels="".join(panels),
        t1=t1, t2=t2, t3=t3, warn=warn, dropped=dropped, thumb=args.thumb,
        embed=args.embed, chart_css=MC.CHART_CSS + MA.ARCH_CSS, **charts,
        ncond=len(tags), npur=len(purs), nimg=len(names))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out, encoding="utf-8")
    mb = args.out.stat().st_size / 1048576
    print(f"寫出 {args.out}（{mb:.1f} MB，{len(names)} 張 × {len(tags)} 條件 "
          f"× {len(purs)} 算子，縮圖 {args.thumb} px）")


TEMPLATE = """<title>量化交付對 DCT-Shield</title>
<style>
{chart_css}
:root{{--ink:#16121B;--ink-2:#4C4555;--ink-3:#7A7182;--paper:#F2EEF1;
 --surface:#FCFAFB;--line:#DDD4DB;--accent:#A9124F;--alt:#0B6B76;--good:#146B34;
 --thumb:{thumb}px}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
 --ink:#EEE8ED;--ink-2:#B5ABB8;--ink-3:#8B8191;--paper:#121016;--surface:#1C1721;
 --line:#302935;--accent:#F26AA1;--alt:#5FC9D2;--good:#5FCB8C}}}}
:root[data-theme="dark"]{{--ink:#EEE8ED;--ink-2:#B5ABB8;--ink-3:#8B8191;
 --paper:#121016;--surface:#1C1721;--line:#302935;--accent:#F26AA1;
 --alt:#5FC9D2;--good:#5FCB8C}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-size:15px;
 font-family:"IBM Plex Sans","Helvetica Neue","PingFang TC",sans-serif;
 line-height:1.6}}
h1,h2{{font-family:Archivo,"Helvetica Neue","PingFang TC",sans-serif;
 letter-spacing:-.015em;text-wrap:balance;margin:0}}
.mono{{font-family:ui-monospace,Menlo,monospace}}
header{{border-bottom:1px solid var(--line);background:var(--surface);
 padding:20px 24px 14px}}
h1{{font-size:23px;margin-bottom:6px}}
.lede{{color:var(--ink-2);max-width:84ch;font-size:14px;margin:0}}
.bar{{position:sticky;top:0;z-index:9;background:var(--surface);
 border:1px solid var(--line);border-radius:8px;padding:9px 12px;display:flex;
 gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:6px}}
.tabs{{display:flex;gap:4px;flex-wrap:wrap}}
.tab{{font:600 12px/1 ui-monospace,monospace;padding:7px 9px;border-radius:6px;
 border:1px solid var(--line);background:transparent;color:var(--ink-2);
 cursor:pointer}}
.tab.active{{background:var(--accent);border-color:var(--accent);color:#fff}}
.ctl{{display:flex;gap:7px;align-items:center;font-size:12.5px;color:var(--ink-2)}}
.ctl input[type=range]{{width:120px;accent-color:var(--accent)}}
.mini{{font:600 11.5px/1 inherit;padding:6px 9px;border-radius:6px;
 border:1px solid var(--line);background:transparent;color:var(--ink-2);
 cursor:pointer}}
.mini:hover{{border-color:var(--accent);color:var(--accent)}}
.seg{{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}}
.seg button{{font:600 12px/1 inherit;padding:7px 10px;border:0;cursor:pointer;
 background:transparent;color:var(--ink-2)}}
.seg button.on{{background:var(--accent);color:#fff}}
.panel{{display:none;padding:14px 0 20px}}
.panel.on{{display:block}}
.imhead{{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;
 margin-bottom:10px}}
.imhead .id{{font:600 13px/1.4 ui-monospace,monospace}}
.imhead .prompt{{color:var(--accent);font-style:italic;font-size:14px}}
.body{{display:flex;gap:18px;align-items:flex-start}}
.ref{{display:grid;grid-template-columns:var(--thumb);gap:3px;flex:0 0 auto}}
.reflab{{font-size:11px;color:var(--ink-3);text-align:center;margin-bottom:6px;
 line-height:1.3}}
.matrix{{display:grid;
 grid-template-columns:minmax(140px,auto) repeat(var(--cols),var(--thumb));
 gap:3px;overflow-x:auto}}
.colhdr{{font:600 11px/1.3 inherit;color:var(--ink-2);text-align:center;
 padding:2px 0 5px;align-self:end}}
.rowlab{{font-size:11.5px;color:var(--ink-2);padding-right:9px;display:flex;
 align-items:center;justify-content:flex-end;text-align:right;line-height:1.25}}
.rowlab.ours{{color:var(--accent);font-weight:600}}
.rowlab.rival{{color:var(--alt)}}
.cell{{width:var(--thumb);height:var(--thumb);border:1px solid var(--line);
 border-radius:4px;overflow:hidden;background:var(--surface)}}
.cell img{{display:block;width:100%;height:100%;object-fit:cover}}
.cell img.p{{display:none}}
body.showpur .cell img.e{{display:none}}
body.showpur .cell img.p{{display:block}}
.cell.miss{{display:flex;align-items:center;justify-content:center;
 color:var(--ink-3);font-size:11px;background:repeating-linear-gradient(
 45deg,transparent,transparent 5px,var(--line) 5px,var(--line) 6px)}}
.hwrap{{max-width:80ch}}
.eyebrow{{font:600 11px/1 ui-monospace,monospace;letter-spacing:.16em;
 text-transform:uppercase;color:var(--accent);margin:0 0 10px}}
.sec{{padding:26px 24px 8px;border-top:1px solid var(--line)}}
.sec>h2{{margin:0 0 14px}}
.sec>.chart{{max-width:860px}}
.sec>.arch{{max-width:1060px;margin-bottom:26px}}
section.tables{{padding:8px 24px 60px}}
h2{{font-size:17px;margin:26px 0 4px}}
.note{{color:var(--ink-3);font-size:12.5px;margin:0 0 10px;max-width:88ch}}
.note .up{{color:var(--good);font-weight:700}}
.note .down{{color:var(--alt);font-weight:700}}
.warn{{color:var(--accent);font-size:12.5px;max-width:88ch}}
ul.warn{{margin:4px 0 0;padding-left:20px}}
table{{border-collapse:collapse;font-size:12.5px;
 font-variant-numeric:tabular-nums;display:block;overflow-x:auto;max-width:100%}}
th,td{{border-bottom:1px solid var(--line);padding:5px 9px;text-align:right;
 white-space:nowrap}}
th{{color:var(--ink-3);font-weight:600;font-size:11.5px}}
td.name,th:first-child{{text-align:left}}
tr.ours td.name{{color:var(--accent);font-weight:600}}
tr.rival td.name{{color:var(--alt)}}
td.best{{color:var(--good);font-weight:700}}
th .up,th .down{{font-size:11px;opacity:.75}}
th .up{{color:var(--good)}}
th .down{{color:var(--alt)}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
</style>
<header><div class="hwrap">
<p class="eyebrow">白盒頻域／相位抗文字編輯防禦 · InstructPix2Pix</p>
<h1>量化交付對 DCT-Shield</h1>
<p class="lede">十張影像、{ncond} 個防禦條件、{npur} 個淨化算子、每格三顆種子。
效果一律以<b>扣掉空白地板的淨增益</b>表示。</p>
</div></header>

<section class="sec">
<h2>兩個方法的網路圖</h2>
{arch_ours}
{arch_dct}
</section>

<section class="sec">
<h2>淨增益對壓縮強度</h2>
{curve}
</section>

<section class="sec">
<h2>各條件的失真高度</h2>
{bar}
</section>

<section class="sec">
<h2>失真與低品質存活的關係</h2>
{scatter}
</section>

<section class="sec">
<h2>逐張影像</h2>
<div class="bar">
<div class="tabs">{tabs}</div>
<div class="ctl"><label for="sz">縮圖</label>
<input id="sz" type="range" min="48" max="{embed}" step="4" value="{thumb}">
<span class="mono" id="szv">{thumb}</span>px
<button id="fit" class="mini">符合畫面</button></div>
<div class="seg"><button id="be" class="on">編輯輸出</button>
<button id="bp">淨化後的防禦圖</button></div>
</div>
{panels}
</section>
<section class="tables">
<h2>防禦圖的失真</h2>
<p class="note">十張平均。箭頭是<b>對防禦有利的方向</b>
（<span class="up">↑</span> 越高越好、<span class="down">↓</span> 越低越好），
綠色是該欄最佳。</p>
{t1}
<h2>編輯輸出的位移與語意</h2>
<p class="note">量的是<b>防禦圖的編輯</b>相對<b>原圖的編輯</b>差多少。
同一個指標在這裡的箭頭與上表相反——上表要防禦圖像原圖，這裡要編輯結果不像
原本的編輯。CLIP 與 SigLIP 是兩張<b>編輯輸出之間的影像對影像餘弦相似度，
不含任何文字</b>（<span class="mono">openai/clip-vit-base-patch32</span>、
<span class="mono">google/siglip-base-patch16-224</span>）。「擋下」是 SigLIP
過門檻 0.8445 的張數，是代理讀數，與人眼判定不一致。</p>
{t2}
<h2>抗淨化的淨增益</h2>
<p class="note">扣掉同影像同算子的空白地板。淨化算子自己就會把編輯推開，
不扣掉它，「淨化後位移較大」無法排除「該算子本來就把編輯推得比較開」。
全部用差值不用比例：分母塌陷時比值不可解讀。</p>
{warn}
{t3}
{dropped}
</section>
<script>
const tabs=[...document.querySelectorAll('.tab')],
      panels=[...document.querySelectorAll('.panel')];
tabs.forEach(t=>t.onclick=()=>{{
  tabs.forEach(x=>x.classList.toggle('active',x===t));
  panels.forEach(p=>p.classList.toggle('on',p.dataset.i===t.dataset.i));
}});
const sz=document.getElementById('sz'),szv=document.getElementById('szv');
function setSize(v){{v=Math.max(48,Math.min({embed},Math.round(v)));sz.value=v;
  document.documentElement.style.setProperty('--thumb',v+'px');szv.textContent=v;}}
sz.oninput=()=>{{setSize(+sz.value);
  try{{localStorage.setItem('thumb',sz.value)}}catch(e){{}}}};
// 「一次看完」是這一版存在的理由，所以縮圖大小由畫面反推，不是給一個猜的預設。
// 高度看列數（表頭那一列只有字，扣掉固定 22px），寬度看欄數與左欄標籤。
function fitScreen(){{
  const m=document.querySelector('.panel.on .matrix');if(!m)return;
  const r=m.getBoundingClientRect();
  const rows=m.querySelectorAll('.rowlab:not(.hdr)').length;
  const cols=+getComputedStyle(m).getPropertyValue('--cols').trim();
  if(!rows||!cols)return;
  const byH=(window.innerHeight-r.top-22-26)/rows-3;
  const byW=(window.innerWidth-r.left-26)/cols-3;
  setSize(Math.min(byH,byW));
  try{{localStorage.setItem('thumb',sz.value)}}catch(e){{}}
}}
document.getElementById('fit').onclick=fitScreen;
let stored=null;try{{stored=localStorage.getItem('thumb')}}catch(e){{}}
if(stored)setSize(+stored);else requestAnimationFrame(fitScreen);
const be=document.getElementById('be'),bp=document.getElementById('bp');
be.onclick=()=>{{document.body.classList.remove('showpur');
  be.classList.add('on');bp.classList.remove('on')}};
bp.onclick=()=>{{document.body.classList.add('showpur');
  bp.classList.add('on');be.classList.remove('on')}};
document.addEventListener('keydown',e=>{{
  const i=tabs.findIndex(t=>t.classList.contains('active'));
  if(e.key==='ArrowRight'&&i<tabs.length-1)tabs[i+1].click();
  if(e.key==='ArrowLeft'&&i>0)tabs[i-1].click();
}});
</script>
"""


if __name__ == "__main__":
    main()
