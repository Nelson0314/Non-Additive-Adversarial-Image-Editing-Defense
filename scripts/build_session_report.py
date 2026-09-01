"""相位重參數化 對 位移場：一頁把兩族的數據、圖表與逐張對比擺在一起。

**不跑 GPU。** 讀 `session_summary.json`（`collect_session_summary.py` 產生）與
已存的影像；輸出單一 HTML，圖以 data URI 內嵌，不入版控（`CLAUDE.md` 資料保全）。

讀數的規定
────────────────────────────────────────────────────────────────────
一律報兩個**絕對值**，不報比例：

    總增益 = effect(p)                 位移本身
    淨增益 = effect(p) − 空白地板       算子自己造成的位移已扣掉

比例讀數（保留率、佔可達範圍）在分母塌陷或參照改變時不可解讀，故不出現在本頁。
位移是 LPIPS，在兩張不相干的自然影像之間飽和於 **0.772**
（`runs/readout_ceiling/`，45 對的中位數）——這個值只作為「還剩多少空間」的
參考，不進任何算式。

**裁切那一欄用的是舊參照。** 現行協定下幾何類算子的參照是「同一個算子淨化過的
原圖」，其空白地板由構造為 0；本頁引用的 CSV 是在舊參照（未淨化的原圖）下量的，
故裁切欄的地板是 0.5506。該欄不可與新協定的數字並列，頁面上逐處標註。

用法：python scripts/build_session_report.py --out report_phase_vs_warp.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
from typing import Optional

from PIL import Image

REP = Path("_rep/runs")
IMAGES = [("task_attr_mod_color_11699", "盆栽人", "turn the color of potted plant to pink"),
          ("task_attr_mod_color_6205", "瑪利歐", "turn the color of toolbox to red")]
OPS = ["identity", "jpeg75", "jpeg30", "blur1", "blur2", "crop_resize0.1"]
OP_LABEL = {"identity": "未淨化", "jpeg75": "JPEG 75", "jpeg30": "JPEG 30",
            "blur1": "模糊 σ=1", "blur2": "模糊 σ=2",
            "crop_resize0.1": "裁切縮放 10%"}
CEIL = 0.772
THUMB = 360

# 條件 → (顯示名, 族, 所在批次, 說明)
COND = {
    "sb_surv":     ("sb_surv", "phase", "ip2p_split_band",
                    "UNet 影像引導損失 ＋ 加性下限 0.08 ＋ 加性項存活加權 ＋ 寬 EOT"),
    "ig_f08_eot":  ("ig_f08_eot", "phase", "ip2p_ig_loss",
                    "UNet 影像引導損失 ＋ 加性下限 0.08 ＋ 寬 EOT"),
    "ln_eot":      ("ln_eot", "phase", "ip2p_ig_loss",
                    "舊損失 latent_norm ＋ 寬 EOT"),
    "r12_f04":     ("r12_f04", "phase", "ip2p_ig_lowdist",
                    "同配方的低失真端：radius 1.2、加性下限 0.04"),
    "sa_t000":     ("sa_t000", "warp", "ip2p_stadv",
                    "stAdv 稠密流場，τ=0（無 TV 正則），L-BFGS，radius 64"),
    "sa_r04":      ("sa_r04", "warp", "ip2p_stadv_radius",
                    "同上，radius 4——失真最接近相位族工作點的一格"),
    "sa_r08":      ("sa_r08", "warp", "ip2p_stadv_radius", "同上，radius 8"),
}
CHAIN = ["ig_f08_eot", "sb_surv", "sa_t000", "sa_r04"]
DEF_DIR = {"ig_f08_eot": "ip2p_ig_loss/ig_f08_eot",
           "sb_surv": "ip2p_split_band/sb_surv",
           "r12_f04": "ip2p_ig_lowdist/r12_f04",
           "sa_t000": "ip2p_stadv/sa_t000",
           "sa_r04": "ip2p_stadv_radius/sa_r04"}
GAL_DIR = {"ig_f08_eot": "ip2p_ig_loss/purify/gallery_ig_f08_eot",
           "sb_surv": "ip2p_split_band/purify/gallery_sb_surv",
           "sa_t000": "ip2p_stadv/purify/gallery_sa_t000",
           "sa_r04": "ip2p_stadv_radius/purify/gallery_sa_r04"}
CONDTAG = {"ig_f08_eot": "phase_gain", "sb_surv": "phase_gain",
           "r12_f04": "phase_gain", "sa_t000": "warp", "sa_r04": "warp"}


def uri(path: Path, q: int = 88) -> Optional[str]:
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    if im.width > THUMB:
        im = im.resize((THUMB, THUMB), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fig(path: Path, cap: str) -> str:
    u = uri(path)
    if u is None:
        return ""
    return (f'<figure><img src="{u}" alt="{html.escape(cap)}">'
            f"<figcaption>{cap}</figcaption></figure>")


def scatter(rows) -> str:
    """失真—位移散布圖。x = DISTS，y = 未淨化位移。"""
    W, H, L, B = 660, 360, 62, 46
    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    xmax = max(xs) * 1.12
    ymax = max(max(ys) * 1.12, CEIL * 1.02)
    def px(v): return L + v / xmax * (W - L - 18)
    def py(v): return H - B - v / ymax * (H - B - 20)
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'aria-label="失真對未淨化位移的散布圖">']
    p.append(f'<line x1="{L}" y1="{H-B}" x2="{W-8}" y2="{H-B}" '
             'stroke="var(--line)"/>')
    p.append(f'<line x1="{L}" y1="16" x2="{L}" y2="{H-B}" stroke="var(--line)"/>')
    p.append(f'<line x1="{L}" y1="{py(CEIL):.1f}" x2="{W-8}" y2="{py(CEIL):.1f}" '
             'stroke="var(--accent)" stroke-dasharray="5 4" stroke-width="1.2"/>')
    p.append(f'<text x="{W-12}" y="{py(CEIL)-6:.1f}" text-anchor="end" '
             f'font-size="11" fill="var(--accent)">LPIPS 飽和 {CEIL}</text>')
    for i in range(5):
        v = xmax * i / 4
        p.append(f'<text x="{px(v):.1f}" y="{H-B+16}" text-anchor="middle" '
                 f'font-size="10" fill="var(--mut)">{v:.2f}</text>')
        w = ymax * i / 4
        p.append(f'<text x="{L-8}" y="{py(w)+4:.1f}" text-anchor="end" '
                 f'font-size="10" fill="var(--mut)">{w:.2f}</text>')
    for name, d, l, fam in rows:
        c = "var(--phase)" if fam == "phase" else "var(--warp)"
        p.append(f'<circle cx="{px(d):.1f}" cy="{py(l):.1f}" r="6" fill="{c}" '
                 'fill-opacity="0.85"/>')
        p.append(f'<text x="{px(d)+10:.1f}" y="{py(l)+4:.1f}" font-size="10.5" '
                 f'fill="var(--fg)">{name}</text>')
    p.append(f'<text x="{(W+L)/2:.0f}" y="{H-8}" text-anchor="middle" '
             'font-size="11.5" fill="var(--mut)">DISTS（失真，越小越好）</text>')
    p.append(f'<text transform="translate(15,{H/2:.0f}) rotate(-90)" '
             'text-anchor="middle" font-size="11.5" fill="var(--mut)">'
             '未淨化位移（越大越好）</text>')
    p.append("</svg>")
    return "".join(p)


def bars(series) -> str:
    """逐算子的淨增益長條圖。series = [(條件, 族, {算子: 淨增益})]。"""
    W, H, L, B = 780, 380, 62, 74
    n_op, n_s = len(OPS), len(series)
    gw = (W - L - 20) / n_op
    bw = gw * 0.78 / n_s
    ymax = max(max(v for v in s[2].values()) for s in series) * 1.15
    def py(v): return H - B - v / ymax * (H - B - 26)
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
         'aria-label="逐算子的淨增益長條圖">']
    p.append(f'<line x1="{L}" y1="{H-B}" x2="{W-8}" y2="{H-B}" stroke="var(--line)"/>')
    for i in range(5):
        v = ymax * i / 4
        p.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{W-8}" y2="{py(v):.1f}" '
                 'stroke="var(--line)" stroke-dasharray="2 4"/>')
        p.append(f'<text x="{L-8}" y="{py(v)+4:.1f}" text-anchor="end" '
                 f'font-size="10" fill="var(--mut)">{v:.2f}</text>')
    for oi, op in enumerate(OPS):
        x0 = L + oi * gw + gw * 0.11
        for si, (name, fam, vals) in enumerate(series):
            v = vals.get(op, 0.0)
            c = "var(--phase)" if fam == "phase" else "var(--warp)"
            o = 1.0 - si * 0.22
            p.append(f'<rect x="{x0+si*bw:.1f}" y="{py(v):.1f}" '
                     f'width="{bw-2:.1f}" height="{H-B-py(v):.1f}" fill="{c}" '
                     f'fill-opacity="{o:.2f}"><title>{name} {op} {v:.4f}</title></rect>')
        lab = OP_LABEL[op]
        p.append(f'<text x="{L+oi*gw+gw/2:.1f}" y="{H-B+16}" text-anchor="middle" '
                 f'font-size="10.5" fill="var(--mut)">{lab}</text>')
    lx = L
    for si, (name, fam, _) in enumerate(series):
        c = "var(--phase)" if fam == "phase" else "var(--warp)"
        p.append(f'<rect x="{lx}" y="{H-30}" width="11" height="11" fill="{c}" '
                 f'fill-opacity="{1.0-si*0.22:.2f}"/>')
        p.append(f'<text x="{lx+16}" y="{H-20}" font-size="10.5" '
                 f'fill="var(--fg)">{name}</text>')
        lx += 26 + len(name) * 7.2
    p.append(f'<text transform="translate(15,{(H-B)/2+10:.0f}) rotate(-90)" '
             'text-anchor="middle" font-size="11.5" fill="var(--mut)">'
             '淨增益（扣掉空白地板）</text>')
    p.append("</svg>")
    return "".join(p)


CSS = """<style>
:root{--bg:#fff;--fg:#15181d;--mut:#5c6572;--line:#e2e6ea;--card:#f7f8fa;
 --accent:#a3341f;--phase:#2563a8;--warp:#b8562a;--good:#0f5132}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --bg:#101216;--fg:#e7e9ec;--mut:#98a2b0;--line:#282d36;--card:#171a20;
 --accent:#ff9478;--phase:#6aa9e8;--warp:#f0996a;--good:#7ee2b8}}
:root[data-theme="dark"]{--bg:#101216;--fg:#e7e9ec;--mut:#98a2b0;--line:#282d36;
 --card:#171a20;--accent:#ff9478;--phase:#6aa9e8;--warp:#f0996a;--good:#7ee2b8}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2rem 1.2rem 5rem;
 font:15px/1.68 -apple-system,"Segoe UI","Noto Sans TC",system-ui,sans-serif;
 max-width:1480px;margin-inline:auto}
h1{font-size:1.7rem;margin:0 0 .35rem;letter-spacing:-.01em}
h2{font-size:1.22rem;margin:2.8rem 0 .5rem;padding-top:1rem;
 border-top:1px solid var(--line)}
h3{font-size:1rem;margin:1.7rem 0 .45rem;color:var(--mut);font-weight:600}
p{max-width:80ch}
.lede{color:var(--mut);max-width:80ch}
code{background:var(--card);padding:.1em .35em;border-radius:3px;font-size:.9em}
table{border-collapse:collapse;font-size:.85rem;margin:.7rem 0 1.1rem;
 display:block;overflow-x:auto;max-width:100%}
th,td{border:1px solid var(--line);padding:.34rem .55rem;text-align:right;
 white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--card)}
tr.hl td{font-weight:700;color:var(--accent)}
tr.warp td:first-child{color:var(--warp)}
tr.phase td:first-child{color:var(--phase)}
.row{display:flex;gap:.55rem;overflow-x:auto;padding:.2rem 0 .5rem}
figure{margin:0;flex:0 0 auto;width:184px}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
 border-radius:4px;background:var(--card)}
figcaption{font-size:.72rem;color:var(--mut);margin-top:.28rem;line-height:1.35}
.card{border:1px solid var(--line);border-radius:6px;padding:.7rem .8rem;
 margin:.7rem 0;background:var(--card)}
.card h4{margin:0 0 .25rem;font-size:.93rem}
.meta{font-size:.78rem;color:var(--mut);margin:0 0 .4rem}
.note{border-left:3px solid var(--accent);padding:.15rem 0 .15rem .8rem;
 color:var(--mut);margin:1rem 0;max-width:80ch}
.chart{background:var(--card);border:1px solid var(--line);border-radius:6px;
 padding:.7rem;margin:.8rem 0;overflow-x:auto}
</style>"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=Path("session_summary.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("report_phase_vs_warp.html"))
    args = ap.parse_args()

    S = json.loads(args.summary.read_text(encoding="utf-8"))
    B = S["batches"]

    def dfn(tag):
        return B[COND[tag][2]]["defence"][tag]

    def pur(tag):
        return B[COND[tag][2]]["purify"].get(tag, {})

    P, A = [], None
    A = P.append
    A("<title>相位重參數化對位移場</title>")
    A(CSS)
    A("<h1>相位重參數化 對 位移場：兩族方法的完整對照</h1>")
    A('<p class="lede">兩張影像（盆栽人、瑪利歐）、InstructPix2Pix、'
      '六個淨化算子、每格 3 個種子。規模只夠當篩選，不足以支撐結論。</p>')

    A('<div class="note"><b>讀數一律報兩個絕對值，不報比例。</b> '
      '<code>總增益 = effect(p)</code> 是位移本身；'
      '<code>淨增益 = 總增益 − 空白地板</code>，空白地板是淨化算子自己把編輯'
      '推開的量。保留率與「佔可達範圍」這類比例讀數在分母塌陷時不可解讀，'
      '本頁不使用。位移是 LPIPS，在兩張不相干的自然影像之間飽和於 '
      f'<b>{CEIL}</b>（45 對的中位數，<code>runs/readout_ceiling/</code>）——'
      '這個值只作為「還剩多少空間」的參考，不進任何算式。</div>')
    A('<div class="note"><b>裁切那一欄用的是舊參照。</b> 現行協定下幾何類算子的'
      '參照是「同一個算子淨化過的原圖」，空白地板由構造為 0。本頁引用的 CSV 是'
      '在舊參照（未淨化的原圖）下量的，所以裁切欄的地板是 0.5506，而 LPIPS 的'
      f'飽和值只有 {CEIL}——那一欄的淨增益因此被壓在很小的範圍裡。'
      '<b>該欄不可與新協定的數字並列</b>，要新數字必須重跑。</div>')

    # ---------------------------------------------------------- 防禦端
    A("<h2>一、防禦端：失真與未淨化位移</h2>")
    A('<p class="lede">「停止」欄是收斂判定的結果：<code>early_stop</code> 表示'
      '固定抽樣的評估連續多次沒有改善而停下，<code>max_steps</code> 表示撞到'
      '步數上限——<b>撞上限的一律讀成未收斂</b>。</p>')
    A("<table><thead><tr><th>條件</th><th>族</th><th>DISTS↓</th><th>LPIPS↓</th>"
      "<th>PSNR↑</th><th>殘差 RMS↓</th><th>L∞↓</th><th>未淨化位移↑</th>"
      "<th>位移/DISTS</th><th>停止</th></tr></thead><tbody>")
    order = ["sb_surv", "ig_f08_eot", "ln_eot", "r12_f04",
             "sa_t000", "sa_r04", "sa_r08"]
    for t in order:
        r = dfn(t)
        fam = COND[t][1]
        eff = r["edit_lpips"] / r["fid_dists"] if r.get("fid_dists") else 0
        cls = "hl" if t == "sb_surv" else fam
        A(f'<tr class="{cls}"><td>{t}</td>'
          f'<td>{"相位" if fam == "phase" else "位移場"}</td>'
          f'<td>{r["fid_dists"]:.4f}</td><td>{r["fid_lpips"]:.4f}</td>'
          f'<td>{r["fid_psnr"]:.2f}</td><td>{r["fid_rms"]:.4f}</td>'
          f'<td>{r["fid_linf"]:.3f}</td><td>{r["edit_lpips"]:.4f}</td>'
          f'<td>{eff:.2f}</td>'
          f'<td>{r.get("stop_reason", "—")}@{r.get("stopped_at", "—")}</td></tr>')
    A("</tbody></table>")
    for t in order:
        A(f'<p class="meta"><b>{t}</b>　{html.escape(COND[t][3])}</p>')

    A('<div class="chart">'
      + scatter([(t, dfn(t)["fid_dists"], dfn(t)["edit_lpips"], COND[t][1])
                 for t in order]) + "</div>")
    A('<p class="lede">位移場那一族（橘）整條落在相位族（藍）的右下方：'
      '**同樣的位移要付更多失真**。`sa_t000` 的位移 0.647 與 `ig_f08_eot` 的'
      '0.644 打平，但 DISTS 是 0.389 對 0.229。</p>')

    # ---------------------------------------------------------- 抗淨化
    A("<h2>二、抗淨化：總增益與淨增益</h2>")
    for kind, key in (("總增益（位移本身）", "total"), ("淨增益（扣掉空白地板）", "net")):
        A(f"<h3>{kind}</h3><table><thead><tr><th>條件</th><th>DISTS</th>"
          + "".join(f"<th>{OP_LABEL[o]}</th>" for o in OPS)
          + "</tr></thead><tbody>")
        for t in order:
            p = pur(t)
            if not p:
                continue
            cls = "hl" if t == "sb_surv" else COND[t][1]
            cells = "".join(
                f'<td>{p[o][key]:.4f}</td>' if o in p and p[o][key] is not None
                else "<td>—</td>" for o in OPS)
            A(f'<tr class="{cls}"><td>{t}</td>'
              f'<td>{dfn(t)["fid_dists"]:.4f}</td>{cells}</tr>')
        if key == "net":
            fl = pur("sb_surv")
            A("<tr><td>空白地板</td><td>—</td>"
              + "".join(f'<td>{fl[o]["floor"]:.4f}</td>' if o in fl else "<td>—</td>"
                        for o in OPS) + "</tr>")
        A("</tbody></table>")

    series = [(t, COND[t][1],
               {o: (pur(t)[o]["net"] or 0) for o in OPS if o in pur(t)})
              for t in ("sb_surv", "ig_f08_eot", "sa_t000", "sa_r04")]
    A('<div class="chart">' + bars(series) + "</div>")

    A('<p class="lede"><b>等失真下位移場輸掉全部六欄。</b> '
      '<code>sa_r04</code>（DISTS 0.2556）比 <code>ig_f08_eot</code>（0.2289）'
      '多付 11.7% 失真，六欄的淨增益全部較低。'
      '<code>sa_t000</code> 在模糊 σ=2 上的 0.2606 確實遠高於相位族，但它的'
      'DISTS 是 0.3893——<b>那是 1.70 倍失真換來的，不是機制差異</b>。</p>')
    A('<p class="lede"><code>sb_surv</code> 是相位族的最佳點：把期望存活振幅'
      '<code>(1 + Σ_σ exp(−2π²σ²f²))/3</code> 乘到<b>加性項</b>的價目表上。'
      '對 <code>ig_f08_eot</code> 多付 8% 失真，模糊 σ=1 的淨增益由 0.2639 升到'
      '0.3148、σ=2 由 0.0980 升到 0.1178。<b>這是所有嘗試過的路徑中唯一動得了'
      '模糊 σ=2 的。</b></p>')

    # ---------------------------------------------------------- 影像
    A("<h2>三、逐張對比</h2>")
    A('<p class="lede">每一列：原圖、防禦圖、編輯(原圖)、編輯(防禦圖)。'
      '<b>擋下與否要用眼睛判「重畫」對「劣化」</b>——模型重畫成無關的場景、或'
      '整張變成噪紋都算擋下；原圖仍認得出來、只是變糊變髒的單純劣化不算。</p>')
    for img, name, prompt in IMAGES:
        A(f"<h3>{name}　<span style='font-weight:400;color:var(--mut)'>"
          f"指令：{html.escape(prompt)}</span></h3>")
        for t in ("ig_f08_eot", "sb_surv", "r12_f04", "sa_t000", "sa_r04"):
            d = REP / DEF_DIR[t]
            c = CONDTAG[t]
            r = dfn(t)
            A('<div class="card">')
            A(f"<h4>{t}　<span style='font-weight:400'>"
              f"{html.escape(COND[t][3])}</span></h4>")
            A(f'<p class="meta">DISTS {r["fid_dists"]:.4f}・'
              f'PSNR {r["fid_psnr"]:.2f}・L∞ {r["fid_linf"]:.3f}・'
              f'未淨化位移 {r["edit_lpips"]:.4f}</p>')
            A('<div class="row">')
            A(fig(REP / "ip2p_ig_loss/ig_f08_eot" / f"{img}__orig.png", "原圖"))
            A(fig(d / f"{img}__{c}__def.png", "防禦圖（交出去的）"))
            A(fig(REP / "ip2p_ig_loss/ig_f08_eot" /
                  f"{img}__phase_gain__edit_orig.png", "編輯（原圖）"))
            A(fig(d / f"{img}__{c}__edit_def.png", "編輯（防禦圖）"))
            A("</div></div>")

    # ---------------------------------------------------------- 淨化鏈
    A("<h2>四、淨化之後的編輯結果</h2>")
    A('<p class="lede">左起：編輯(淨化後的原圖)＝空白地板那一格，'
      '其餘各條件的編輯(淨化後的防禦圖)。要判的是後者相對於第一格有沒有多擋下'
      '東西。</p>')
    for img, name, _ in IMAGES:
        A(f"<h3>{name}</h3>")
        for op in OPS:
            A(f'<p class="meta"><b>{OP_LABEL[op]}</b>　空白地板 '
              f'{pur("sb_surv")[op]["floor"]:.4f}'
              + ("　<b>（此欄為舊參照）</b>" if op.startswith("crop") else "")
              + "</p>")
            A('<div class="row">')
            A(fig(REP / "ip2p_ig_loss/purify/gallery_floor" /
                  f"{img}__None__{op}__edit_def.png", "編輯（淨化後的原圖）"))
            for t in CHAIN:
                g = REP / GAL_DIR[t]
                A(fig(g / f"{img}__{CONDTAG[t]}__{op}__edit_def.png",
                      f"{t}　淨增益 {pur(t)[op]['net']:.4f}"
                      if op in pur(t) else t))
            A("</div>")

    A('<p class="lede" style="margin-top:2.6rem">數值來源：'
      '<code>runs/{ip2p_ig_loss,ip2p_ig_lowdist,ip2p_split_band,ip2p_stadv,'
      'ip2p_stadv_radius}/</code> 的 <code>results.csv</code> 與 '
      '<code>purify/*_all.csv</code>，經 '
      '<code>scripts/collect_session_summary.py</code> 彙整。'
      '本頁由 <code>scripts/build_session_report.py</code> 產生。</p>')

    args.out.write_text("\n".join(P), encoding="utf-8")
    print(f"寫出 {args.out}（{args.out.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
