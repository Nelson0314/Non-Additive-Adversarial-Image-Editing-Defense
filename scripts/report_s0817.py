"""2026-08-17 批次的對比頁。純資料呈現：影像、LPIPS、DISTS、圖表，無文字解釋。

    python scripts/report_s0817.py --out reports/2026-08-17-s0817

圖表是自己畫的 inline SVG，沒有外部相依。缺的資料標成「—」，不靜默略過。
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import ast
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Sequence

RUN = Path("runs/s0817/merged")
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
    "apa_weak": "#8a6d1f",
}
PURIF = ["identity", "blur1", "noise0.05", "quantize16", "jpeg75", "jpeg30",
         "crop_resize0.1", "jpeg_then_resize75", "adverse_cleaner", "impress"]


# ---------------------------------------------------------------- 讀取


def edit_settings() -> Dict[str, float]:
    """從 `scripts/apa_baseline.py` 讀 SDEdit 的四個常數。

    用 ast 讀而不是 import，是為了不把 torch 與 diffusers 拉進報告流程；用讀的
    而不是抄一份，是因為抄過的那份曾經停在 strength 0.55 而實際跑的是 0.8。
    """
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



def rd(p: Path) -> List[dict]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_results() -> Dict[str, Dict[str, dict]]:
    """condition -> image -> 該列。只讀 `RUN` 這一個目錄。

    先前是掃 `runs/s0817/*/results.csv` 把所有分片並起來。2026-08-17 之後
    `merged/` 與各分片目錄不再同步——分片留著上一批 prompt 的 `edit_*` 與
    編輯圖，`merged/` 才是重算過的——而排序後 `pg*`／`px` 排在 `merged` 之後，
    會把新值覆寫回舊值，且不留任何症狀。改成只認一個目錄。
    """
    out: Dict[str, Dict[str, dict]] = {}
    for r in rd(RUN / "results.csv"):
        out.setdefault(r["condition"], {})[r["image"]] = r
    return out


def load_retention() -> List[dict]:
    rows: List[dict] = []
    for p in sorted(RUN.glob("retention_*.csv")):
        rows += rd(p)
    return rows


def tag_of(r: dict) -> str:
    """由列還原防禦圖檔名中間那一段。與 phase_retention 的規則相同。"""
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


def find_png(image: str, suffix: str) -> Optional[Path]:
    p = RUN / f"{image}__{suffix}.png"
    return p if p.exists() else None


def find_png_any(image: str, suffix: str) -> Optional[Path]:
    """`__edit_orig` 是逐條件各存一份的，取先找到的那份。

    同一張影像的各份未防禦編輯只差 8 位元捨入（實測最大 1/255、0.05% 的像素），
    不是不同的編輯結果。
    """
    for p in sorted(RUN.glob(f"{image}__*__{suffix}.png")):
        return p
    return None


# ---------------------------------------------------------------- 圖表


def bars(groups: Sequence[str], series: Sequence[tuple], *, height: int = 240,
         gap: int = 26, bw: int = 15, nd: int = 3, unit: str = "") -> str:
    """分組長條圖。`series` 是 (名稱, 顏色, {group: 值}) 的序列。"""
    vals = [v for _, _, m in series for v in m.values() if v is not None]
    if not vals:
        return '<p class="miss">—</p>'
    vmax = max(vals) * 1.15 or 1.0
    gw = len(series) * bw + gap
    w = 62 + len(groups) * gw
    h = height + 74
    p = [f'<svg viewBox="0 0 {w} {h}" class="chart" '
         f'preserveAspectRatio="xMinYMin meet">']
    # 網格與刻度
    for i in range(5):
        y = 18 + height - height * i / 4
        val = vmax * i / 4
        p.append(f'<line x1="58" y1="{y:.1f}" x2="{w-6}" y2="{y:.1f}" '
                 f'class="grid"/>')
        p.append(f'<text x="52" y="{y+4:.1f}" class="ax" '
                 f'text-anchor="end">{val:.{nd}f}</text>')
    for gi, g in enumerate(groups):
        x0 = 62 + gi * gw
        for si, (_, colr, m) in enumerate(series):
            v = m.get(g)
            if v is None:
                continue
            bh = height * v / vmax
            x = x0 + si * bw
            p.append(f'<rect x="{x}" y="{18+height-bh:.1f}" width="{bw-2}" '
                     f'height="{bh:.1f}" fill="{colr}"><title>{g} · '
                     f'{series[si][0]} · {v:.{nd}f}{unit}</title></rect>')
        cx = x0 + len(series) * bw / 2
        p.append(f'<text x="{cx:.1f}" y="{18+height+16}" class="ax" '
                 f'text-anchor="middle">{g}</text>')
    p.append("</svg>")
    leg = "".join(f'<span class="lg"><i style="background:{c}"></i>{n}</span>'
                  for n, c, _ in series)
    return f'<div class="chartbox">{"".join(p)}<div class="legend">{leg}</div></div>'


# ---------------------------------------------------------------- 版面


def table(headers, rows, right_from: int = 1) -> str:
    h = "".join(f'<th{" class=n" if i >= right_from else ""}>{x}</th>'
                for i, x in enumerate(headers))
    body = "".join("<tr>" + "".join(
        f'<td{" class=n" if i >= right_from else ""}>{x}</td>'
        for i, x in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><tr>{h}</tr>{body}</table></div>'


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
h2{font-size:20px;margin:44px 0 14px;padding-top:16px;border-top:2px solid var(--line)}
h3{font-size:16px;margin:26px 0 10px;color:var(--mut)}
.tw{overflow-x:auto;margin:0 0 16px}
table{border-collapse:collapse;font-size:13.5px;width:100%}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:left}
th{background:var(--code);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
table.imgs{table-layout:fixed;width:max-content;min-width:100%}
table.imgs col{width:470px}
col.rowcol{width:190px}
table.imgs td,table.imgs th{padding:3px;text-align:center;vertical-align:top}
table.imgs tr.after td{padding-bottom:16px;border-bottom:2px solid var(--mut)}
table.imgs th{font-size:14px;line-height:1.35}
table.imgs th.rowh{text-align:left;font-size:13px;color:var(--mut)}
table.imgs th.rowh .pr{display:block;font-weight:400;font-style:italic;
font-size:12.5px;margin-top:2px}
table.imgs img{display:block;width:100%;aspect-ratio:1;object-fit:cover;border-radius:3px}
.cap{display:block;font-size:13px;font-variant-numeric:tabular-nums;
color:var(--mut);line-height:1.45;margin-top:3px}
.miss{display:grid;place-items:center;aspect-ratio:1;background:var(--code);
color:var(--mut);border-radius:3px}
.chartbox{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:14px 12px 10px;margin:0 0 16px;overflow-x:auto}
svg.chart{display:block;width:100%;height:auto;min-width:640px}
.grid{stroke:var(--line);stroke-width:1}
text.ax{font-size:11px;fill:var(--mut);font-family:inherit}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:8px;font-size:12.5px}
.lg{display:inline-flex;align-items:center;gap:5px;color:var(--mut)}
.lg i{width:11px;height:11px;border-radius:2px;display:inline-block}
.meta{color:var(--mut);font-size:13px;margin:0 0 22px}
"""


# ---------------------------------------------------------------- 主體

def stamp_radius(res: Dict[str, Dict[str, dict]]) -> None:
    """把實際跑出來的半徑寫進標籤。

    半徑寫死在標籤裡曾經與實際設定不一致（2026-08-17：標成 θ=π 而實際是
    人眼門檻，或反之）。標籤由資料決定就不會再漂。同一條件若各影像半徑不同，
    直接報錯而不是取一個代表值——那代表分片設定不一致。
    """
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

    res = load_results()
    imgs = sorted({i for m in res.values() for i in m})
    stamp_radius(res)
    conds = [c for c in COND if c in res]
    P: List[str] = []
    A = P.append

    es = edit_settings()
    A(f'<p class="meta">{DATA} · {len(imgs)} 張 · prompt 0 · '
      f'SDEdit strength {es["EDIT_STRENGTH"]:g} / '
      f'guidance {es["EDIT_GUIDANCE"]:g} / '
      f'{es["EDIT_STEPS"]} 步 / seed {es["EDIT_SEED"]}</p>')

    # ---------- 1 編輯前後對比圖 ----------
    A('<h2>1　編輯前後</h2>')
    A('<p class="meta">每張影像兩列：上列是送進 SDEdit 之前的圖（原圖與各方法的'
      '防禦圖），下列是同一張圖被 SDEdit 編輯之後的結果。第一欄的編輯後即未防禦'
      '的編輯，是各方法位移量的參照。</p>')
    cols = [("原圖 / 未防禦編輯", None)] + [(LABEL[c], c) for c in conds]
    colgroup = ('<colgroup><col class="rowcol">'
                f'<col span="{len(cols)}"></colgroup>')
    head = "".join(f"<th>{t}</th>" for t, _ in cols)
    rows = []
    for im in imgs:
        before, after = [], []
        for _, c in cols:
            if c is None:
                before.append(f'<td>{img_tag(find_png(im, "orig"))}'
                              f'<span class="cap">原圖</span></td>')
                after.append(f'<td>{img_tag(find_png_any(im, "edit_orig"))}'
                             f'<span class="cap">未防禦編輯</span></td>')
                continue
            r = res.get(c, {}).get(im)
            tag = tag_of(r) if r else None
            before.append(
                f'<td>{img_tag(find_png(im, f"{tag}__def") if tag else None)}'
                f'<span class="cap">'
                f'LPIPS {fmt(num(r, "fid_lpips"))}<br>'
                f'DISTS {fmt(num(r, "fid_dists"))}</span></td>')
            after.append(
                f'<td>{img_tag(find_png(im, f"{tag}__edit_def") if tag else None)}'
                f'<span class="cap">編輯 LPIPS '
                f'{fmt(num(r, "edit_lpips"))}</span></td>')
        rows.append(f'<tr><th class="rowh" rowspan="2">{im}'
                    f'<span class="pr">「{prompts.get(im, "")}」</span>'
                    f'<span class="pr">上：編輯前　下：編輯後</span></th>'
                    f'{"".join(before)}</tr><tr class="after">{"".join(after)}</tr>')
    A(f'<div class="tw"><table class="imgs">{colgroup}'
      f'<tr><th></th>{head}</tr>{"".join(rows)}</table></div>')

    # ---------- 2 LPIPS / DISTS ----------
    A('<h2>2　LPIPS 與 DISTS</h2>')
    A("<h3>2.1 逐影像</h3>")
    A(table(["影像"] + [f"{LABEL[c]}<br>LPIPS" for c in conds]
            + [f"{LABEL[c]}<br>DISTS" for c in conds],
            [[im] + [fmt(num(res.get(c, {}).get(im), "fid_lpips")) for c in conds]
             + [fmt(num(res.get(c, {}).get(im), "fid_dists")) for c in conds]
             for im in imgs]))

    A("<h3>2.2 LPIPS</h3>")
    A(bars(imgs, [(LABEL[c], COLOR[c],
                   {im: num(res.get(c, {}).get(im), "fid_lpips") for im in imgs})
                  for c in conds]))
    A("<h3>2.3 DISTS</h3>")
    A(bars(imgs, [(LABEL[c], COLOR[c],
                   {im: num(res.get(c, {}).get(im), "fid_dists") for im in imgs})
                  for c in conds]))

    A("<h3>2.4 平均</h3>")
    def avg(c: str, k: str) -> Optional[float]:
        v = [num(res.get(c, {}).get(i), k) for i in imgs]
        v = [x for x in v if x is not None]
        return st.fmean(v) if v else None
    A(table(["方法", "n", "LPIPS", "DISTS", "PSNR", "SSIM"],
            [[LABEL[c], str(len(res.get(c, {}))), fmt(avg(c, "fid_lpips")),
              fmt(avg(c, "fid_dists")), fmt(avg(c, "fid_psnr"), 2),
              fmt(avg(c, "fid_ssim"))] for c in conds]))
    A(bars(["LPIPS", "DISTS"],
           [(LABEL[c], COLOR[c],
             {"LPIPS": avg(c, "fid_lpips"), "DISTS": avg(c, "fid_dists")})
            for c in conds], bw=22, gap=60))

    # ---------- 3 淨化 ----------
    A('<h2>3　淨化後的編輯位移量</h2>')
    ret = load_retention()
    if not ret:
        A('<p class="miss" style="aspect-ratio:auto;padding:26px">'
          '尚未產出</p>')
    else:
        per: Dict[str, Dict[str, List[float]]] = {}
        for r in ret:
            per.setdefault(r["purifier"], {}).setdefault(
                r["condition"], []).append(float(r["effect_mean"]))
        pur = [p for p in PURIF if p in per]
        A("<h3>3.1 各淨化算子 × 各方法（9 張平均）</h3>")
        A(table(["淨化算子"] + [LABEL[c] for c in conds],
                [[p] + [fmt(st.fmean(per[p][c]) if per[p].get(c) else None)
                        for c in conds] for p in pur]))
        A("<h3>3.2 圖表</h3>")
        A(bars(pur, [(LABEL[c], COLOR[c],
                      {p: (st.fmean(per[p][c]) if per[p].get(c) else None)
                       for p in pur}) for c in conds], bw=13, gap=22))

        A("<h3>3.3 逐影像逐算子</h3>")
        byi: Dict[str, Dict[str, Dict[str, float]]] = {}
        for r in ret:
            byi.setdefault(r["image"], {}).setdefault(
                r["purifier"], {})[r["condition"]] = float(r["effect_mean"])
        A(table(["影像", "淨化算子"] + [LABEL[c] for c in conds],
                [[im, p] + [fmt(byi[im].get(p, {}).get(c)) for c in conds]
                 for im in sorted(byi) for p in pur if p in byi[im]],
                right_from=2))

        # ---------- 4 retention ----------
        A('<h2>4　retention 比值</h2>')
        rt: Dict[str, Dict[str, List[float]]] = {}
        bad: Dict[str, set] = {}
        for r in ret:
            if r["purifier"] != "identity":
                rt.setdefault(r["purifier"], {}).setdefault(
                    r["condition"], []).append(float(r["retention"]))
            if r["usable"].strip().lower() in ("false", "0"):
                bad.setdefault(r["condition"], set()).add(r["image"])
        rpur = [x for x in pur if x != "identity"]
        A("<h3>4.1 各淨化算子 × 各方法（9 張平均）</h3>")
        A(table(["淨化算子"] + [LABEL[c] for c in conds],
                [[x] + [fmt(st.fmean(rt[x][c]) if rt[x].get(c) else None, 3)
                        for c in conds] for x in rpur]))
        A("<h3>4.2 圖表</h3>")
        A(bars(rpur, [(LABEL[c], COLOR[c],
                       {x: (st.fmean(rt[x][c]) if rt[x].get(c) else None)
                        for x in rpur}) for c in conds], bw=13, gap=22, nd=2))
        A("<h3>4.3 分母塌陷（usable=False）的影像數</h3>")
        A(table(["方法", "影像數", "全部"],
                [[LABEL[c], str(len(bad.get(c, set()))), str(len(imgs))]
                 for c in conds]))

        # ---------- 5 逐格勝負 ----------
        A('<h2>5　紋理重相位 vs 各方法的逐格勝負</h2>')
        grid: Dict[tuple, Dict[str, float]] = {}
        for r in ret:
            grid.setdefault((r["image"], r["purifier"]), {})[
                r["condition"]] = float(r["effect_mean"])

        def duel(base: str, only_identity: bool) -> List[str]:
            w = l = 0
            ratios: List[float] = []
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
                    ratios.append(d["phase"] / d[base])
            if not ratios:
                return ["—", "—"]
            return [f"{w} / {l}", f"{st.fmean(ratios):.2f}"]

        others = [c for c in conds if c != "phase"]
        A("<h3>5.1 未淨化（identity，9 格）與淨化後（8 算子 × 9 影像 = 72 格）</h3>")
        A(table(["對照方法", "未淨化 勝/敗", "未淨化 平均比",
                 "淨化後 勝/敗", "淨化後 平均比"],
                [[LABEL[c]] + duel(c, True) + duel(c, False) for c in others]))
    return "".join(P)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("reports/2026-08-17-s0817"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    html = ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>s0817 對比</title><style>{CSS}</style></head><body><main>'
            '<h1>s0817 對比</h1>' + build() + "</main></body></html>")
    p = args.out / "index.html"
    p.write_text(html, encoding="utf-8")
    print(f"  {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
