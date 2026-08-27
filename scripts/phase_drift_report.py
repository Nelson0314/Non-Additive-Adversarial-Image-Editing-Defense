"""blur／crop／JPEG 對「裝上去的相位偏移」做了什麼——量測與視覺化的單頁報告。

資料來源
────────────────────────────────────────────────────────────────────
數值：`runs/phase_drift_diagnosis/`（`summary.csv`、`phase_retention_by_band.csv`、
`residual_survival.csv`），由 `scripts/phase_drift_diagnosis.py` 產生。
影像面板：由 `scripts/phase_drift_figure.py` 先產生 PNG，本檔以 `--figures`
指向那個目錄。**兩步驟是刻意的**：面板需要防禦圖，而防禦圖不入版控、只在遠端，
所以報告不能宣稱自己從 CSV 就能單獨重建。

圖一律**以 data URI 內嵌**（照片面板轉成 JPEG 以壓縮體積，曲線圖維持 PNG），
所以產出的 HTML 是自給自足的單一檔案，放在專案根目錄不會斷圖。

用法：
    python scripts/phase_drift_report.py \
        --figures <phase_drift_figure 的輸出目錄> --out report_phase_drift.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                          "PMingLiU", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COND_LABEL = {
    "ours_ph_q": "本方法 純相位＋量化交付 r0.9",
    "ours_ph_n": "本方法 純相位 未量化 r0.9",
    "ours_pg_q20": "本方法 相位+增益＋量化 r2.0",
    "dct_aj85": "DCT-Shield 抗JPEG q0.85",
}
PUR_LABEL = {
    "identity": "未淨化", "jpeg75": "JPEG 75", "jpeg50": "JPEG 50", "jpeg30": "JPEG 30",
    "blur1": "模糊 σ1", "blur2": "模糊 σ2",
    "crop_resize0.1": "裁切 10%", "crop_resize0.15": "裁切 15%",
}
PUR_ORDER = ["identity", "blur1", "blur2", "jpeg75", "jpeg50", "jpeg30",
             "crop_resize0.1", "crop_resize0.15"]
IMG_LABEL = {"task_attr_mod_color_11699": "盆栽與人物",
             "task_obj_remove_380621": "冰山湖景"}


def read(p: Path) -> List[Dict[str, str]]:
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fig_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def photo_uri(path: Path, max_w: int, quality: int) -> str:
    """照片面板轉 JPEG 再內嵌。原 PNG 每張約 5 MB，八張就 40 MB。"""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def table(headers: Sequence[str], rows: Sequence[Sequence[str]],
          hi: Sequence[int] = ()) -> str:
    h = "".join(f"<th>{html.escape(x)}</th>" for x in headers)
    body = "".join(
        f"<tr{' class=hi' if i in hi else ''}>"
        + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
        for i, r in enumerate(rows))
    return f"<div class='wrap'><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>"


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#5f5f5f;--line:#dcdcdc;--hi:#fff4d6;
      --card:#fafafa;--acc:#1b6e3c;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e8e8e8;--mut:#a2a2a2;
      --line:#333;--hi:#3a3320;--card:#1c1c1c;--acc:#7ddba0;--bad:#f2b8b5}}
*{box-sizing:border-box}
body{margin:0;padding:2.2rem 1.2rem 5rem;background:var(--bg);color:var(--fg);
     font:15px/1.75 "Microsoft JhengHei","Segoe UI",system-ui,sans-serif}
main{max-width:1120px;margin:0 auto}
h1{font-size:1.6rem;line-height:1.35;margin:0 0 .4rem}
h2{font-size:1.18rem;margin:2.6rem 0 .7rem;padding-top:.9rem;border-top:1px solid var(--line)}
h3{font-size:1rem;margin:1.6rem 0 .4rem;color:var(--mut)}
.sub{color:var(--mut);margin:0 0 1.6rem}
.key{padding:.9rem 1.1rem;border-left:4px solid var(--acc);background:var(--card);
     border-radius:0 6px 6px 0;margin:1rem 0}
.key b{display:block;margin-bottom:.2rem}
.key.warn{border-left-color:var(--bad)}
table{border-collapse:collapse;width:100%;margin:.7rem 0;font-size:13.5px}
th,td{border-bottom:1px solid var(--line);padding:.4rem .5rem;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{border-bottom:2px solid var(--line);font-weight:600;color:var(--mut)}
tr.hi{background:var(--hi)}
.wrap{overflow-x:auto}
figure{margin:1.2rem 0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;
           background:#fff}
figcaption{color:var(--mut);font-size:13px;margin-top:.4rem}
code{background:var(--card);padding:.1rem .3rem;border-radius:3px;font-size:.92em}
nav{display:flex;flex-wrap:wrap;gap:.4rem;margin:1rem 0 .2rem}
nav button{font:inherit;font-size:13px;padding:.32rem .7rem;border:1px solid var(--line);
           background:var(--card);color:var(--fg);border-radius:999px;cursor:pointer}
nav button[aria-selected=true]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.panel{display:none}.panel.on{display:block}
ul{padding-left:1.2rem}
"""

JS = """
document.querySelectorAll('[data-tabs]').forEach(function(group){
  var btns=group.querySelectorAll('nav button');
  btns.forEach(function(b){
    b.addEventListener('click',function(){
      btns.forEach(function(x){x.setAttribute('aria-selected',x===b)});
      group.querySelectorAll('.panel').forEach(function(p){
        p.classList.toggle('on',p.dataset.key===b.dataset.key)});
    });
  });
});
"""


def build(args) -> str:
    root = Path(args.root)
    diag = root / "runs" / "phase_drift_diagnosis"
    summary = read(diag / "summary.csv")
    bands = read(diag / "phase_retention_by_band.csv")

    def s(cond, pur, key):
        for r in summary:
            if r["condition"] == cond and r["purifier"] == pur:
                return float(r[key])
        return float("nan")

    # ---- 主表 ----
    rows = []
    for p in PUR_ORDER:
        rows.append([PUR_LABEL[p],
                     f"{s('ours_ph_q', p, 'rho_mean'):.3f}",
                     f"{s('ours_ph_q', p, 'inst_mag'):.3f}",
                     f"{s('ours_ph_q', p, 'surv_mag'):.3f}",
                     f"{s('ours_ph_q', p, 'energy_ratio'):.3f}",
                     f"{s('ours_ph_q', p, 'cosine'):.3f}",
                     (f"{s('ours_ph_q', p, 'cos_vs_warped'):.4f}"
                      if p.startswith("crop") else "—")])
    main_tbl = table(["淨化算子", "相位保留 ρ", "裝上去的 |Δφ|", "存活的 |Δφ|",
                      "殘差能量存活", "對原格點 cos", "對「搬過的同一份」cos"],
                     rows, hi=[1, 2, 6, 7])

    # ---- 跨條件對照 ----
    cross = []
    for c in ("ours_ph_q", "ours_ph_n", "ours_pg_q20", "dct_aj85"):
        cross.append([COND_LABEL[c],
                      f"{s(c, 'blur1', 'energy_ratio'):.3f}",
                      f"{s(c, 'blur2', 'energy_ratio'):.3f}",
                      f"{s(c, 'jpeg30', 'energy_ratio'):.3f}",
                      f"{s(c, 'jpeg30', 'cosine'):.3f}",
                      f"{s(c, 'crop_resize0.1', 'cos_vs_warped'):.4f}"])
    cross_tbl = table(["條件", "blur σ1 能量", "blur σ2 能量", "jpeg30 能量",
                       "jpeg30 對原格點 cos", "crop 對「搬過的」cos"], cross)

    # ---- 逐頻帶曲線 ----
    blist = sorted({(float(r["band_lo"]), float(r["band_hi"])) for r in bands})
    centres = [(lo + hi) / 2 for lo, hi in blist]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    for ax, cond in zip(axes, ("ours_ph_q", "dct_aj85")):
        for p in ("blur1", "blur2", "jpeg75", "jpeg30", "crop_resize0.1"):
            ys = [st.fmean(float(r["rho"]) for r in bands
                           if r["condition"] == cond and r["purifier"] == p
                           and float(r["band_lo"]) == lo) for lo, _ in blist]
            ax.plot(centres, ys, "o-", ms=4, label=PUR_LABEL[p])
        ax.set_ylim(0, 1.05); ax.grid(alpha=.25)
        ax.set_xlabel("歸一化頻率半徑（1 = Nyquist）")
        ax.set_ylabel("相位保留 ρ（1 = 原封不動）")
        ax.set_title(COND_LABEL[cond], fontsize=10.5)
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("裝上去的相位偏移，在每一個頻帶上還剩多少（十張平均）", fontsize=12)
    curves = fig_uri(fig)

    # ---- 相位保留 vs 能量存活 ----
    fig2, ax = plt.subplots(figsize=(6.8, 5.0))
    mk = {"ours_ph_q": "o", "ours_ph_n": "s", "ours_pg_q20": "^", "dct_aj85": "D"}
    for c, m in mk.items():
        xs = [s(c, p, "rho_mean") for p in PUR_ORDER[1:]]
        ys = [s(c, p, "energy_ratio") for p in PUR_ORDER[1:]]
        ax.scatter(xs, ys, marker=m, s=52, alpha=.85, label=COND_LABEL[c])
        if c == "ours_ph_q":
            for x, y, p in zip(xs, ys, PUR_ORDER[1:]):
                ax.annotate(PUR_LABEL[p], (x, y), fontsize=8,
                            textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("相位保留 ρ"); ax.set_ylabel("殘差能量存活率")
    ax.set_yscale("log"); ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    ax.set_title("相位還在，不代表它還推得動任何東西", fontsize=11)
    scatter = fig_uri(fig2)

    # ---- 影像面板（頁籤） ----
    figs = Path(args.figures)
    panels, btns, first = [], [], True
    for cond in ("ours_ph_q", "ours_ph_n", "ours_pg_q20", "dct_aj85"):
        for img in ("task_attr_mod_color_11699", "task_obj_remove_380621"):
            hits = sorted(figs.glob(f"phase_drift_{cond}_{img}_*rows.png"))
            if not hits:
                continue
            key = f"{cond}__{img}"
            lab = f"{COND_LABEL[cond].split('（')[0]} · {IMG_LABEL[img]}"
            btns.append(f"<button data-key='{key}' aria-selected='"
                        f"{'true' if first else 'false'}'>{html.escape(lab)}</button>")
            uri = photo_uri(hits[-1], args.panel_width, args.quality)
            panels.append(
                f"<div class='panel{' on' if first else ''}' data-key='{key}'>"
                f"<figure><img src='{uri}' alt='{html.escape(lab)}'></figure></div>")
            first = False
    gallery = ("<div data-tabs><nav>" + "".join(btns) + "</nav>"
               + "".join(panels) + "</div>")

    body = f"""
<div class='key'><b>結論：淨化沒有讓相位「跑掉」。三個算子壞在完全不同的地方。</b>
模糊是零相位濾波器，<b>動不了相位</b>——它抽走的是相位所乘的那個係數幅度；
JPEG 是在原偏移之上<b>再疊一層</b>自己的相位擾動，不是把它磨平；
裁切則是把整份擾動<b>原封不動地搬走</b>，對「搬過的同一份」餘弦是 0.9997。
<b>「裁切讓相位跑掉」這個說法是錯的，不可以寫進論文。</b></div>

<h2>一、量在哪裡，量的是什麼</h2>
<p>量測做在<b>本方法自己的分析域</b>上——32×32 區塊、hop 8、Hann 窗的加窗
STFT，與 <code>texture_rephase.analyze</code> 同一支程式。<b>不是全域 FFT</b>：
本方法旋轉的是逐區塊相位，在全域 FFT 上量會把它與區塊之間的相對位置混在一起。</p>
<p>設 <code>S = STFT(x)</code>、<code>S' = STFT(x_def)</code>，<b>裝上去的偏移</b>
是 <code>Δφ_inst = angle(S') − angle(S)</code>。算子 <code>p</code> 之後，用
<b>同一個算子也過一遍的原圖</b>當參照（不是原圖本身，否則量到的是算子自己造成
的相位變化）：<code>Δφ_surv = angle(STFT(p(x_def))) − angle(STFT(p(x)))</code>。</p>
<p>保留度取圓統計的平均合成長度
<code>ρ = |Σ w·exp(i(Δφ_surv − Δφ_inst))| / Σ w</code>，權重 <code>w = |S|</code>
（相位在幅度近零處由捨入雜訊決定）。<b>直接把角度相減再平均是錯的</b>，會被
±π 的折返騙到。</p>

<h2>二、十張平均（<code>ours_ph_q</code>）</h2>
{main_tbl}

<div class='key'><b>模糊：相位原封不動，是載體被抽掉。</b>
σ1 下 ρ = 0.967、存活的 |Δφ| 是 0.366 對裝上去的 0.375（差 2.4%），而殘差能量
只剩 <b>8.9%</b>（σ2 只剩 3.3%）。這不是巧合，是從構造推得出來的：高斯模糊
頻域上乘的是一個<b>實正數</b> <code>H(ω) ≥ 0</code>，而乘實正數<b>不改變相位</b>。
加窗 STFT 下模糊與加窗只近似交換，ρ 因此是 0.967 而不是 1。
<br><br><b>後果</b>：先前把模糊那一格說成「擾動被破壞」是錯的歸因，正確的是
「擾動被靜音」。兩者的下一步不同——後者只要把同一份相位資訊放到 <code>H(ω)</code>
還沒壓掉的頻率上，而 <code>m_ω</code> 與 <code>H(ω)</code> 都是已知曲線，
乘積的極大值可以直接解，不必掃參數。</div>

<div class='key'><b>JPEG：相位被打散，而且是往上打散。</b>
存活的 |Δφ| 由 0.375 漲到 0.486（q75）／0.558（q30），ρ 卻掉到 0.813／0.687，
而 q75 的能量存活率是 <b>2.16</b>——比裝上去的還多。量化是在原有的偏移之上
<b>再疊一層</b>，不是把它磨平。</div>

<div class='key warn'><b>裁切：整份原封不動地通過，只是換了座標。</b>
對「把殘差單獨送進同一個幾何變換」的餘弦是 <b>0.9997</b>，對原格點是 <b>0.006</b>。
逐區塊 ρ 掉到 0.599 是<b>索引對不上</b>的結果——<code>crop_resize(0.1)</code> 是
以中心為不動點的 1.2488 倍放大，<code>p(x)</code> 的區塊格點與 <code>x</code>
的不是同一組區塊。<br><br>
<b>一個踩到並修掉的實作陷阱</b>：<code>purify.ops.crop_resize</code> 末行的
<code>.clamp(0,1)</code> 是給影像用的值域維護，套在有號殘差上會把每一個負值
推成 0，控制組只剩一半訊號，餘弦被架高到 0.72，看起來像「只搬走七成」。
拿掉夾取（其餘逐行照抄）之後才是 0.9997。</div>

<h2>三、逐頻帶的形狀</h2>
<figure><img src='{curves}' alt='逐頻帶相位保留'><figcaption>
模糊 σ1 那條在 r &lt; 0.55 的四個帶上是 1.000／0.999／0.994／0.983，幾乎水平；
JPEG 與裁切則隨頻率單調下滑，形狀跟量化階隨頻率變大一致。
</figcaption></figure>

<h2>四、相位保留與能量存活是兩件事</h2>
<figure><img src='{scatter}' alt='相位保留對能量存活'><figcaption>
模糊 σ1 在最右（ρ ≈ 0.96，相位幾乎完好）卻在最下（能量剩 0.09）。
這張圖是本頁的一句話：<b>相位還在，不代表它還推得動任何東西。</b>
</figcaption></figure>

<h2>五、四個條件的橫向對照</h2>
{cross_tbl}
<ul>
<li><b>裁切對四個條件一視同仁</b>（對「搬過的同一份」餘弦全部 0.9996–0.9997）。
這一格不是參數化的問題，是評測算子把座標搬走的問題，<b>任何非幾何不變的擾動
都會落在同一格</b>。</li>
<li><b>模糊那一格本方法留下的能量是 DCT-Shield 的 2.6 倍</b>（0.089 對 0.034），
<code>ours_pg_q20</code> 更是 5.3 倍——但三者都擋不下來。<b>留得比較多不等於
留得夠多。</b></li>
<li><b>一個可能有用的副產品</b>：q30 上本方法的能量存活是 DCT-Shield 的 2.6 倍、
對原格點方向 2.7 倍，而該格的淨增益比是 2.39 倍。相位保留這個<b>純 CPU</b> 的量
在 JPEG 軸上排得出淨增益的順序，可當設計的預篩代理。<b>保留</b>：四個條件上的
一致，不是回歸，不可寫成定量預測。</li>
</ul>

<h2>六、逐張看</h2>
<p>每一列一個算子，五欄依序是：原圖經算子、防禦圖經算子、裝上去的殘差 ×12、
淨化後存活的殘差 ×12、逐區塊相位一致度 ρ（亮 = 相位原封不動）。
殘差一律用<b>同一個放大倍率</b>顯示，否則列與列之間的深淺不可比。</p>
{gallery}

<h2>七、這一頁不改變什麼</h2>
<p><code>docs/RESULTS.md</code> 與 <code>PENDING.md</code> 上「blur 與 crop
三個方法都沒有防禦」<b>維持不變</b>——本頁量的是擾動的存活，不是編輯位移，
沒有跑任何 GPU 上的編輯。它改變的是<b>失效的歸因</b>，不是失效與否。</p>
<p class='sub'>數值：<code>runs/phase_drift_diagnosis/</code>（README ＋ 三張 CSV）。
產生器：<code>scripts/phase_drift_diagnosis.py</code>（量測）、
<code>phase_drift_figure.py</code>（面板）、<code>phase_drift_report.py</code>（本頁）。</p>
"""
    return (f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>淨化之後，相位還在不在</title><style>{CSS}</style></head><body>"
            f"<main><h1>淨化之後，相位還在不在</h1>"
            f"<p class='sub'>十張主線影像 · 四個條件 · 八個淨化算子 · "
            f"量測在本方法自己的 32×32／hop 8 STFT 域上，純 CPU</p>"
            f"{body}</main><script>{JS}</script></body></html>")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", required=True,
                    help="phase_drift_figure.py 的輸出目錄（含 *_7rows.png）")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel-width", type=int, default=1500)
    ap.add_argument("--quality", type=int, default=86)
    args = ap.parse_args(argv)

    args.out.write_text(build(args), encoding="utf-8")
    print(f"寫出 {args.out}（{args.out.stat().st_size / 1e6:.2f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
