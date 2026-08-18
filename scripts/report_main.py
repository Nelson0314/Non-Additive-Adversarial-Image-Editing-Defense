"""主線實驗的報告頁。純資料呈現：影像、折線圖、語意指標，無文字解釋。

    python scripts/report_main.py --out reports/main

排版原則（2026-08-18 使用者裁定）：
  * 原始數據一律收進 `<details>`，預設不展開
  * 結果用折線圖呈現——淨化算子是一條有序的軸，折線比長條好讀
  * 只留平均，逐圖的表與圖已移除
  * 每一格都必須有影像可看；缺的標「—」，不靜默略過

圖表是自己畫的 inline SVG，沒有外部相依。
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import io
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Sequence

RUN = Path("runs/s0817/merged")
PURIFIED = Path("runs/s0817/purified")
SEMANTIC = Path("runs/s0817/semantic.csv")
DATA = Path("data/set0817")

COND = ["phase", "phase_rand", "add", "photoguard_c", "mist", "dia_r", "apa_weak"]
LABEL = {
    "phase": "紋理重相位", "phase_rand": "隨機相位", "add": "加性 δ",
    "photoguard_c": "PhotoGuard-c", "mist": "Mist", "dia_r": "DIA-R",
    "apa_weak": "APA",
}
COLOR = {
    "phase": "#c2410c", "phase_rand": "#ea9a5a", "add": "#2f6b3a",
    "photoguard_c": "#2563a8", "mist": "#8b3a8b", "dia_r": "#0d7377",
    "apa_weak": "#8a6d1f", "none": "#94a3b8",
}
PURIF = ["identity", "blur1", "noise0.05", "quantize16", "jpeg75", "jpeg30",
         "crop_resize0.1", "jpeg_then_resize75", "adverse_cleaner", "impress"]


# ---------------------------------------------------------------- 讀取

def rd(p: Path) -> List[dict]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def edit_settings() -> Dict[str, float]:
    """從 `scripts/apa_baseline.py` 讀 SDEdit 的常數（用 ast，不 import torch）。"""
    src = (Path(__file__).resolve().parent / "apa_baseline.py").read_text(encoding="utf-8")
    want = {"EDIT_STRENGTH", "EDIT_GUIDANCE", "EDIT_STEPS", "EDIT_SEED"}
    out: Dict[str, float] = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in want:
                out[t.id] = ast.literal_eval(node.value)
    missing = want - out.keys()
    if missing:
        raise ValueError(f"apa_baseline.py 找不到常數：{sorted(missing)}")
    return out


def tag_of(r: dict) -> str:
    c = r["condition"]
    if c in ("add", "phase", "phase_rand"):
        b = r.get("budget_mode") or ""
        return f"{c}__human" if b == "human" else f"{c}__d{float(r['budget_target']):g}"
    return c


def num(r: Optional[dict], k: str) -> Optional[float]:
    if not r or r.get(k) in (None, ""):
        return None
    return float(r[k])


def fmt(v: Optional[float], nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


# ---------------------------------------------------------------- 影像

def img_tag(path: Optional[Path], side: int = 500, quality: int = 84) -> str:
    if path is None or not path.exists():
        return '<div class="miss">—</div>'
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return ('<img loading="lazy" src="data:image/jpeg;base64,'
            f'{base64.b64encode(buf.getvalue()).decode()}">')


def png(name: str) -> Optional[Path]:
    p = RUN / f"{name}.png"
    return p if p.exists() else None


def pur_png(stem: str, kind: str) -> Optional[Path]:
    p = PURIFIED / f"{stem}__{kind}.png"
    return p if p.exists() else None


# ---------------------------------------------------------------- 圖表

def lines(xs: Sequence[str], series: Sequence[tuple], *, height: int = 300,
          nd: int = 3, ymin: Optional[float] = None,
          ylabel: str = "") -> str:
    """折線圖。`series` 是 (名稱, 顏色, {x: 值}, 虛線?) 的序列。

    淨化算子是一條有序的軸，折線讓「誰在哪個算子上掉得多」一眼可讀；
    長條圖在九個算子七個條件下會變成 63 根柱子，讀不出趨勢。
    """
    vals = [v for s in series for v in s[2].values() if v is not None]
    if not vals:
        return '<p class="miss">—</p>'
    lo = min(vals) if ymin is None else ymin
    hi = max(vals)
    pad = (hi - lo) * 0.12 or 0.05
    lo, hi = lo - pad, hi + pad
    left, right, top = 66, 18, 16
    w = left + right + max(560, len(xs) * 96)
    plot_w = w - left - right
    h = height + 78

    def X(i):
        return left + (plot_w * i / max(1, len(xs) - 1))

    def Y(v):
        return top + height - height * (v - lo) / (hi - lo)

    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" preserveAspectRatio="xMinYMin meet">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        p.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{left-8}" y="{y+4:.1f}" class="ax" text-anchor="end">{v:.{nd}f}</text>')
    for i, x in enumerate(xs):
        p.append(f'<text x="{X(i):.1f}" y="{top+height+18}" class="ax" '
                 f'text-anchor="end" transform="rotate(-28 {X(i):.1f} {top+height+18})">{x}</text>')
    if ylabel:
        p.append(f'<text x="4" y="{top+height/2:.0f}" class="ax" '
                 f'transform="rotate(-90 12 {top+height/2:.0f})" text-anchor="middle">{ylabel}</text>')
    for name, colr, m, *rest in series:
        dash = ' stroke-dasharray="6,4"' if (rest and rest[0]) else ''
        pts = [(X(i), Y(m[x])) for i, x in enumerate(xs) if m.get(x) is not None]
        if not pts:
            continue
        d = " ".join(f"{'M' if k == 0 else 'L'}{a:.1f} {b:.1f}" for k, (a, b) in enumerate(pts))
        p.append(f'<path d="{d}" fill="none" stroke="{colr}" stroke-width="2.2"{dash}/>')
        for a, b in pts:
            p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.2" fill="{colr}"/>')
    p.append("</svg>")
    leg = "".join(f'<span class="lg"><i style="background:{c}"></i>{n}</span>'
                  for n, c, *_ in series)
    return f'<div class="chartbox">{"".join(p)}<div class="legend">{leg}</div></div>'


def hbars(rows: Sequence[tuple], *, nd: int = 4, unit: str = "") -> str:
    """橫向長條：(標籤, 顏色, 值)。給「各條件的一個平均值」這種類別型比較用。"""
    vals = [v for _, _, v in rows if v is not None]
    if not vals:
        return '<p class="miss">—</p>'
    vmax = max(vals) * 1.08 or 1.0
    bh, gap = 24, 10
    w, left = 720, 150
    h = len(rows) * (bh + gap) + 12
    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" preserveAspectRatio="xMinYMin meet">']
    for i, (lab, colr, v) in enumerate(rows):
        y = i * (bh + gap) + 6
        p.append(f'<text x="{left-10}" y="{y+bh*0.68:.0f}" class="ax" text-anchor="end">{lab}</text>')
        if v is None:
            continue
        bw = (w - left - 90) * v / vmax
        p.append(f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bh}" rx="3" fill="{colr}"/>')
        p.append(f'<text x="{left+bw+8:.1f}" y="{y+bh*0.68:.0f}" class="ax">{v:.{nd}f}{unit}</text>')
    p.append("</svg>")
    return f'<div class="chartbox">{"".join(p)}</div>'


# ---------------------------------------------------------------- 版面

def table(headers, rows, right_from: int = 1) -> str:
    h = "".join(f'<th{" class=n" if i >= right_from else ""}>{x}</th>'
                for i, x in enumerate(headers))
    body = "".join("<tr>" + "".join(
        f'<td{" class=n" if i >= right_from else ""}>{x}</td>'
        for i, x in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><tr>{h}</tr>{body}</table></div>'


def raw(title: str, inner: str) -> str:
    """原始數據收合。預設不展開——頁面上先看圖，要數字再點開。"""
    return f'<details><summary>原始數據：{title}</summary>{inner}</details>'


CSS = """
:root{--bg:#fbfaf8;--fg:#1d1c1a;--mut:#6b665e;--line:#e2ded6;--card:#fff;--code:#f4f2ee}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#171614;--fg:#e8e4dc;--mut:#9d968a;--line:#33302b;--card:#1f1e1b;--code:#232019}}
:root[data-theme=dark]{--bg:#171614;--fg:#e8e4dc;--mut:#9d968a;--line:#33302b;
--card:#1f1e1b;--code:#232019}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);line-height:1.7;font-size:15px;
font-family:"Noto Sans TC","PingFang TC","Microsoft JhengHei",system-ui,sans-serif}
main{max-width:1900px;margin:0 auto;padding:34px 24px 100px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:48px 0 14px;padding-top:16px;border-top:2px solid var(--line)}
h3{font-size:16px;margin:26px 0 10px;color:var(--mut)}
.tw{overflow-x:auto;margin:0 0 16px}
table{border-collapse:collapse;font-size:13.5px;width:100%}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:left}
th{background:var(--code);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
table.imgs{table-layout:fixed;width:max-content;min-width:100%}
table.imgs col{width:470px}
col.rowcol{width:190px}
table.imgs.pur col{width:212px}
col.prow{width:150px}
table.imgs.pur img{margin-bottom:3px}
table.imgs.pur th.rowh{font-size:11.5px;vertical-align:middle;text-align:left}
table.imgs td,table.imgs th{padding:3px;text-align:center;vertical-align:top}
table.imgs th{font-size:14px;line-height:1.35}
table.imgs th.rowh{text-align:left;font-size:13px;color:var(--mut)}
table.imgs th.rowh .pr{display:block;font-weight:400;font-style:italic;
font-size:12.5px;margin-top:2px}
table.imgs tr.after td{padding-bottom:16px;border-bottom:2px solid var(--mut)}
table.imgs img{display:block;width:100%;aspect-ratio:1;object-fit:cover;border-radius:3px}
.cap{display:block;font-size:13px;font-variant-numeric:tabular-nums;
color:var(--mut);line-height:1.45;margin-top:3px}
.miss{display:grid;place-items:center;aspect-ratio:1;background:var(--code);
color:var(--mut);border-radius:3px}
.chartbox{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:14px 12px 10px;margin:0 0 16px;overflow-x:auto}
svg.chart{display:block;width:100%;height:auto;min-width:620px}
.grid{stroke:var(--line);stroke-width:1}
text.ax{font-size:11px;fill:var(--mut);font-family:inherit}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:8px;font-size:12.5px}
.lg{display:inline-flex;align-items:center;gap:5px;color:var(--mut)}
.lg i{width:11px;height:11px;border-radius:2px;display:inline-block}
.meta{color:var(--mut);font-size:13px;margin:0 0 22px}
details{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:8px 14px;margin:0 0 18px}
details[open]{padding-bottom:2px}
summary{cursor:pointer;font-size:13px;color:var(--mut);font-weight:600}
details .tw{margin-top:12px}
"""


# ---------------------------------------------------------------- 主體

def stamp_radius(res: Dict[str, Dict[str, dict]]) -> None:
    """把實際跑出來的半徑寫進標籤，避免標籤與設定漂開。"""
    for cond, unit in (("phase", "θ"), ("phase_rand", "θ"), ("add", "ε")):
        rows = res.get(cond)
        if not rows:
            continue
        rad = {r["radius"] for r in rows.values() if r.get("radius")}
        if len(rad) != 1:
            raise ValueError(f"{cond} 的 radius 在各影像間不一致：{sorted(rad)}")
        v = float(rad.pop())
        shown = f"{v * 255:.1f}/255" if cond == "add" else f"{v:.2f}"
        LABEL[cond] = f"{LABEL[cond]} {unit}={shown}"


def build() -> str:
    import yaml
    spec = yaml.safe_load((DATA / "prompts.yaml").read_text(encoding="utf-8"))
    prompts = {f"{c}_00": spec[c]["prompts"][0] for c in spec}

    rows = rd(RUN / "results.csv")
    res: Dict[str, Dict[str, dict]] = {}
    for r in rows:
        res.setdefault(r["condition"], {})[r["image"]] = r
    imgs = sorted({r["image"] for r in rows})
    stamp_radius(res)
    conds = [c for c in COND if c in res]

    ret: List[dict] = []
    for f in sorted(RUN.glob("retention_*.csv")):
        ret += rd(f)
    floor: List[dict] = []
    for f in sorted(RUN.glob("floor_*.csv")):
        floor += rd(f)
    sem = rd(SEMANTIC)

    P: List[str] = []
    A = P.append
    es = edit_settings()
    st_used = rows[0].get("edit_strength") or es["EDIT_STRENGTH"]
    A(f'<p class="meta">{DATA} · {len(imgs)} 張 · prompt 0 · '
      f'SDEdit strength {float(st_used):g} / guidance {es["EDIT_GUIDANCE"]:g} / '
      f'{es["EDIT_STEPS"]} 步 / seed {es["EDIT_SEED"]}　|　'
      f'淨化 {len({r["purifier"] for r in ret})} 個算子 × 3 個編輯 seed'
      f'{"　|　空白地板已量" if floor else ""}</p>')

    # ---------- 1 編輯前後 ----------
    A('<h2>1　編輯前後</h2>')
    A('<p class="meta">每張影像兩列：上列是送進 SDEdit 之前的圖，下列是編輯之後。'
      '第一欄的編輯後即未防禦的編輯，是所有位移量的參照。</p>')
    cols = [("原圖 / 未防禦編輯", None)] + [(LABEL[c], c) for c in conds]
    colgroup = f'<colgroup><col class="rowcol"><col span="{len(cols)}"></colgroup>'
    head = "".join(f"<th>{t}</th>" for t, _ in cols)
    trs = []
    for im in imgs:
        before, after = [], []
        for _, c in cols:
            if c is None:
                before.append(f'<td>{img_tag(png(f"{im}__orig"))}'
                              f'<span class="cap">原圖</span></td>')
                r0 = res[conds[0]][im]
                after.append(f'<td>{img_tag(png(f"{im}__{tag_of(r0)}__edit_orig"))}'
                             f'<span class="cap">未防禦編輯</span></td>')
                continue
            r = res.get(c, {}).get(im)
            tg = tag_of(r) if r else None
            before.append(
                f'<td>{img_tag(png(f"{im}__{tg}__def") if tg else None)}'
                f'<span class="cap">LPIPS {fmt(num(r, "fid_lpips"))}<br>'
                f'DISTS {fmt(num(r, "fid_dists"))}</span></td>')
            after.append(
                f'<td>{img_tag(png(f"{im}__{tg}__edit_def") if tg else None)}'
                f'<span class="cap">位移 {fmt(num(r, "edit_lpips"))}</span></td>')
        trs.append(f'<tr><th class="rowh" rowspan="2">{im}'
                   f'<span class="pr">「{prompts.get(im, "")}」</span>'
                   f'<span class="pr">上：編輯前　下：編輯後</span></th>'
                   f'{"".join(before)}</tr><tr class="after">{"".join(after)}</tr>')
    A(f'<div class="tw"><table class="imgs">{colgroup}'
      f'<tr><th></th>{head}</tr>{"".join(trs)}</table></div>')

    # ---------- 2 保真度 ----------
    def avg(c: str, k: str) -> Optional[float]:
        v = [num(res.get(c, {}).get(i), k) for i in imgs]
        v = [x for x in v if x is not None]
        return st.fmean(v) if v else None

    A(f'<h2>2　保真度（{len(imgs)} 張平均）</h2>')
    A("<h3>2.1 LPIPS</h3>")
    A(hbars([(LABEL[c], COLOR[c], avg(c, "fid_lpips")) for c in conds]))
    A("<h3>2.2 DISTS</h3>")
    A(hbars([(LABEL[c], COLOR[c], avg(c, "fid_dists")) for c in conds]))
    A("<h3>2.3 PSNR（dB，越高越好）</h3>")
    A(hbars([(LABEL[c], COLOR[c], avg(c, "fid_psnr")) for c in conds], nd=2))
    A(raw("六項保真度指標",
          table(["方法", "n", "LPIPS↓", "DISTS↓", "PSNR↑", "SSIM↑", "NIMA↑", "CNNIQA↑"],
                [[LABEL[c], str(len(res.get(c, {}))), fmt(avg(c, "fid_lpips")),
                  fmt(avg(c, "fid_dists")), fmt(avg(c, "fid_psnr"), 2),
                  fmt(avg(c, "fid_ssim")), fmt(avg(c, "fid_nima"), 3),
                  fmt(avg(c, "fid_cnniqa"))] for c in conds])))

    # ---------- 3 位移量 ----------
    A('<h2>3　編輯位移量</h2>')
    A("<h3>3.1 未淨化</h3>")
    A(hbars([(LABEL[c], COLOR[c], avg(c, "edit_lpips")) for c in conds]))

    if ret:
        per: Dict[str, Dict[str, List[float]]] = {}
        for r in ret:
            per.setdefault(r["purifier"], {}).setdefault(
                r["condition"], []).append(float(r["effect_mean"]))
        pur = [p for p in PURIF if p in per]

        fl: Dict[str, List[float]] = {}
        for r in floor:
            fl.setdefault(r["purifier"], []).append(float(r["effect_mean"]))

        A("<h3>3.2 過九個淨化算子之後</h3>")
        series = [(LABEL[c], COLOR[c],
                   {p: (st.fmean(per[p][c]) if per[p].get(c) else None) for p in pur})
                  for c in conds]
        if fl:
            series.append(("空白地板（原圖直接淨化）", COLOR["none"],
                           {p: (st.fmean(fl[p]) if p in fl else None) for p in pur},
                           True))
        A(lines(pur, series, ylabel="位移量 (LPIPS)"))
        A(raw("各淨化算子 × 各方法",
              table(["淨化算子"] + [LABEL[c] for c in conds] + (["空白地板"] if fl else []),
                    [[p] + [fmt(st.fmean(per[p][c]) if per[p].get(c) else None)
                            for c in conds]
                     + ([fmt(st.fmean(fl[p]) if p in fl else None)] if fl else [])
                     for p in pur])))

        if fl:
            A("<h3>3.3 扣掉空白地板之後的淨增益</h3>")
            A('<p class="meta">位移量減去「把原圖直接淨化再編輯」造成的位移。'
              '這一段是防禦真正貢獻的部分。</p>')
            A(lines(pur, [(LABEL[c], COLOR[c],
                           {p: ((st.fmean(per[p][c]) - st.fmean(fl[p]))
                                if (per[p].get(c) and p in fl) else None) for p in pur})
                          for c in conds], ylabel="淨增益"))

        # ---------- 4 逐格勝負 ----------
        A('<h2>4　紋理重相位 vs 各方法的逐格勝負</h2>')
        grid: Dict[tuple, Dict[str, float]] = {}
        for r in ret:
            grid.setdefault((r["image"], r["purifier"]), {})[
                r["condition"]] = float(r["effect_mean"])

        def duel(base: str, only_identity: bool) -> List[str]:
            w = l = 0
            rr: List[float] = []
            for (im, px), d in grid.items():
                if (px == "identity") != only_identity:
                    continue
                if "phase" not in d or base not in d:
                    continue
                if d["phase"] > d[base]:
                    w += 1
                elif d["phase"] < d[base]:
                    l += 1
                if d[base]:
                    rr.append(d["phase"] / d[base])
            return ["—", "—"] if not rr else [f"{w} / {l}", f"{st.fmean(rr):.2f}"]

        others = [c for c in conds if c != "phase"]
        npur = len([p for p in pur if p != "identity"])
        A(table([f"對照方法", f"未淨化 勝/敗（{len(imgs)} 格）", "平均比",
                 f"淨化後 勝/敗（{npur}×{len(imgs)}={npur*len(imgs)} 格）", "平均比"],
                [[LABEL[c]] + duel(c, True) + duel(c, False) for c in others]))

    # ---------- 5 語意 ----------
    A('<h2>5　語意指標</h2>')
    A('<p class="meta">CLIP-T 與 SigLIP-T 量的是編輯結果與 prompt 的對齊程度。'
      '正規化成「達成度」＝(C_防禦後 − C_原圖) / (C_未防禦 − C_原圖)，'
      '1 = 編輯完全達成、0 = 完全沒往 prompt 走；防禦成功度 = 1 − 達成度。</p>')

    base_clip = {}
    for r in sem:
        base_clip[r["image"]] = (float(r["clip_orig_image"]),
                                 float(r["siglip_orig_image"]))
    if not base_clip:
        # 沒有 semantic.csv 時退回 results.csv 的未淨化欄位
        A('<p class="meta">尚未產出 <code>runs/s0817/semantic.csv</code>，'
          '以下只有未淨化的一格。</p>')

    def succ(c: str, key: str) -> Optional[float]:
        v = []
        for im in imgs:
            r = res.get(c, {}).get(im)
            if not r or im not in base_clip:
                continue
            c0 = base_clip[im][0 if key == "clip" else 1]
            c1 = float(r[f"edit_{key}_orig"])
            c2 = float(r[f"edit_{key}_def"])
            if abs(c1 - c0) > 1e-6:
                v.append(1.0 - (c2 - c0) / (c1 - c0))
        return st.fmean(v) if v else None

    A("<h3>5.1 未淨化的防禦成功度</h3>")
    A(hbars([(LABEL[c], COLOR[c], succ(c, "clip")) for c in conds], nd=3))
    A(raw("CLIP / SigLIP 的原始值",
          table(["方法", "CLIP 成功度", "SigLIP 成功度", "CLIP 掉幅", "SigLIP 掉幅"],
                [[LABEL[c], fmt(succ(c, "clip"), 3), fmt(succ(c, "siglip"), 3),
                  fmt(avg(c, "edit_clip_drop")), fmt(avg(c, "edit_siglip_drop"))]
                 for c in conds])))

    if sem:
        by: Dict[str, Dict[str, List[float]]] = {}
        bys: Dict[str, Dict[str, List[float]]] = {}
        for r in sem:
            by.setdefault(r["purifier"], {}).setdefault(
                r["condition"], []).append(float(r["clip"]))
            bys.setdefault(r["purifier"], {}).setdefault(
                r["condition"], []).append(float(r["siglip"]))
        spur = [p for p in PURIF if p in by]
        A("<h3>5.2 淨化之後的 CLIP-T</h3>")
        A(lines(spur, [(LABEL[c], COLOR[c],
                        {p: (st.fmean(by[p][c]) if by[p].get(c) else None)
                         for p in spur}) for c in conds], ylabel="CLIP-T"))
        A("<h3>5.3 淨化之後的 SigLIP-T</h3>")
        A(lines(spur, [(LABEL[c], COLOR[c],
                        {p: (st.fmean(bys[p][c]) if bys[p].get(c) else None)
                         for p in spur}) for c in conds], ylabel="SigLIP-T"))
        A(raw("淨化後的 CLIP-T 逐算子",
              table(["淨化算子"] + [LABEL[c] for c in conds],
                    [[p] + [fmt(st.fmean(by[p][c]) if by[p].get(c) else None)
                            for c in conds] for p in spur])))

    # ---------- 6 淨化後的影像 ----------
    A('<h2>6　淨化後的防禦圖與編輯圖</h2>')
    A('<p class="meta">上排是淨化後的防禦圖、下排是它再被編輯的結果。'
      '只跑一個編輯 seed，<b>不參與任何數字</b>，純供人眼。'
      '產生器 <code>scripts/hb5_purify_gallery.py</code>。</p>')
    if not PURIFIED.exists():
        A(f'<p class="miss" style="aspect-ratio:auto;padding:26px">{PURIFIED} 不存在</p>')
    else:
        gpur = [p for p in PURIF if (PURIFIED / f"{imgs[0]}__{conds[0]}__{p}__pur.png").exists()]
        gcols = f'<colgroup><col class="prow"><col span="{len(conds)}"></colgroup>'
        ghead = "".join(f"<th>{LABEL[c]}</th>" for c in conds)
        for im in imgs:
            A(f'<h3>{im}</h3>')
            grows = []
            for pf in gpur:
                cells = []
                for c in conds:
                    stem = f"{im}__{c}__{pf}"
                    cells.append(
                        f'<td>{img_tag(pur_png(stem, "pur"), side=200, quality=78)}'
                        f'{img_tag(pur_png(stem, "edit"), side=200, quality=78)}</td>')
                grows.append(f'<tr><th class="rowh">{pf}</th>{"".join(cells)}</tr>')
            A(f'<div class="tw"><table class="imgs pur">{gcols}'
              f'<tr><th></th>{ghead}</tr>{"".join(grows)}</table></div>')

    return "".join(P)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("reports/main"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    html = ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>主線實驗</title><style>{CSS}</style></head><body><main>'
            '<h1>主線實驗</h1>' + build() + "</main></body></html>")
    p = args.out / "index.html"
    p.write_text(html, encoding="utf-8")
    print(f"  {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
