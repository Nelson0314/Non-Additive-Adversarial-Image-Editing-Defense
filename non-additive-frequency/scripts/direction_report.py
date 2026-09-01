"""把一個研究方向的判定做成一頁自帶圖表的 HTML。**不跑 GPU。**

兩個方向共用同一個版面，因為要回答的是同一種問題：**事前寫下的判準過了沒有、
以及沒過的話是哪一種死法。**

    --direction dct_rotate   DCT 域的保長配對旋轉（`runs/ip2p_dct_rotate`）
    --direction warp         位移場的預算使用率（`runs/ip2p_warp`）

圖一律以 **data URI 內嵌**，所以產出的 HTML 是自給自足的單一檔案，搬到任何
目錄都不會斷圖。這是刻意的：報告頁放在專案根目錄，而數值留在 `runs/`，
兩者的相對路徑會隨搬動而變。

**HTML 不入版控**（`docs/DECISIONS.md`）：它由已記錄的 CSV 重新產生得出來，
而 CSV 不可重現。根目錄的 `report_*.html` 已加進 `.gitignore`。

用法：
    python scripts/direction_report.py --direction dct_rotate --out report_dct_rotate.html
    python scripts/direction_report.py --direction warp --out report_warp.html
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

BAND = (0.1286, 0.1447)          # 失真帶，`docs/EVALUATION.md` 的工作點對齊


def fig_uri(fig) -> str:
    """把 matplotlib 圖轉成 data URI。內嵌而不是另存檔，見模組 docstring。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def read(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean_of(rows: Sequence[Dict[str, str]], key: str) -> float:
    return st.fmean(float(r[key]) for r in rows)


def table(headers: Sequence[str], rows: Sequence[Sequence[str]],
          highlight: Sequence[int] = ()) -> str:
    h = "".join(f"<th>{html.escape(x)}</th>" for x in headers)
    body = []
    for i, r in enumerate(rows):
        cls = " class='hi'" if i in highlight else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table>"


CSS = """
:root{--bg:#ffffff;--fg:#1a1a1a;--mut:#606060;--line:#d8d8d8;--hi:#fff4d6;
      --bad:#b3261e;--good:#1b6e3c;--card:#fafafa}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e8e8e8;--mut:#a0a0a0;
      --line:#333;--hi:#3a3320;--bad:#f2b8b5;--good:#7ddba0;--card:#1c1c1c}}
*{box-sizing:border-box}
body{margin:0;padding:2.2rem 1.2rem 5rem;background:var(--bg);color:var(--fg);
     font:15px/1.75 "Microsoft JhengHei","Segoe UI",system-ui,sans-serif}
main{max-width:1000px;margin:0 auto}
h1{font-size:1.6rem;line-height:1.35;margin:0 0 .4rem}
h2{font-size:1.15rem;margin:2.4rem 0 .7rem;padding-top:.9rem;
   border-top:1px solid var(--line)}
h3{font-size:1rem;margin:1.5rem 0 .4rem}
.sub{color:var(--mut);margin:0 0 1.6rem}
.verdict{padding:.9rem 1.1rem;border-left:4px solid var(--bad);
         background:var(--card);border-radius:0 6px 6px 0;margin:1rem 0}
.verdict.pass{border-left-color:var(--good)}
.verdict b{display:block;margin-bottom:.2rem}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:13.5px}
th,td{border-bottom:1px solid var(--line);padding:.42rem .5rem;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{border-bottom:2px solid var(--line);font-weight:600;color:var(--mut)}
tr.hi{background:var(--hi)}
figure{margin:1.2rem 0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
figcaption{color:var(--mut);font-size:13px;margin-top:.4rem}
code{background:var(--card);padding:.1rem .3rem;border-radius:3px;font-size:.92em}
.wrap{overflow-x:auto}
ul{padding-left:1.2rem}
"""


def page(title: str, subtitle: str, body: str) -> str:
    return (f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"<main><h1>{html.escape(title)}</h1>"
            f"<p class='sub'>{subtitle}</p>{body}</main></body></html>")


# ------------------------------------------------------------------ DCT 旋轉

def build_dct(root: Path) -> Tuple[str, str, str]:
    rot = root / "runs" / "ip2p_dct_rotate"
    main = root / "runs" / "ip2p_mainline"

    def defense(tag: str, base: Path) -> Tuple[float, float, int]:
        r = read(base / tag / "results.csv")
        return (mean_of(r, "fid_dists"), mean_of(r, "edit_lpips"),
                sum(x["blocked"] == "True" for x in r))

    pts = [("dct_rot_t08", "transpose θ0.8", rot), ("dct_rot_t11", "transpose θ1.1", rot),
           ("dct_rot_t15", "transpose θ1.5", rot), ("dct_rot_zz11", "zigzag θ1.1（對照）", rot),
           ("dct_rot_rand11", "隨機 θ1.1（不最佳化）", rot),
           ("ours_ph_q", "現行量化交付 r0.9", main),
           ("ours_ph_n", "同半徑未量化", main)]
    data = {}
    rows = []
    for tag, label, base in pts:
        p = base / tag / "results.csv"
        if not p.exists():
            continue
        d, e, b = defense(tag, base)
        data[tag] = (d, e)
        rows.append([label, f"{d:.4f}", f"{e:.4f}", f"{b}/10"])

    # 等失真內插到 ours_ph_q 的錨點
    anchor = data["ours_ph_q"][0]
    a, bb = data["dct_rot_t11"], data["dct_rot_t15"]
    interp = a[1] + (bb[1] - a[1]) * (anchor - a[0]) / (bb[0] - a[0])
    ratio = interp / data["ours_ph_q"][1]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = [data[t][0] for t in ("dct_rot_t08", "dct_rot_t11", "dct_rot_t15")]
    ys = [data[t][1] for t in ("dct_rot_t08", "dct_rot_t11", "dct_rot_t15")]
    ax.plot(xs, ys, "o-", label="DCT 旋轉（transpose）")
    ax.plot(*data["dct_rot_zz11"], "s", ms=8, label="zigzag 對照")
    ax.plot(*data["dct_rot_rand11"], "^", ms=8, label="隨機（不最佳化）")
    ax.plot(*data["ours_ph_q"], "*", ms=16, label="現行量化交付")
    ax.plot(*data["ours_ph_n"], "*", ms=16, label="同半徑未量化")
    ax.plot([anchor], [interp], "x", ms=11, mew=2.5, color="k",
            label=f"內插到等失真：{interp:.4f}")
    ax.axvspan(*BAND, alpha=.10, color="gray")
    ax.set_xlabel("防禦圖失真 DISTS（越左越不明顯）")
    ax.set_ylabel("未淨化的編輯位移（越高越有效）")
    ax.grid(alpha=.25); ax.legend(fontsize=8.5)
    ax.set_title("等失真下，DCT 旋轉買到的位移只有現行做法的六成", fontsize=11)
    curve = fig_uri(fig)

    # 天花板
    ce = read(root / "runs" / "dct_phase_design" / "ceiling.csv")
    ths = sorted({float(r["theta"]) for r in ce})
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.0))
    for gate, lab in (("texture", "紋理閘（主線）"), ("band", "帶通閘（全開）")):
        sub = [r for r in ce if r["variant"] == "post_int" and r["gate"] == gate
               and r["sign"] == "random"]
        ax2.plot(ths, [st.fmean(float(r["dists"]) for r in sub
                                if float(r["theta"]) == t) for t in ths],
                 "o-", label="天花板 " + lab)
    # `xs` 是三個 transpose 點實際達成的 DISTS，橫軸換成它們各自的 θ 上界。
    ax2.plot([0.8, 1.1, 1.5], xs, "s--", color="k", label="最佳化實際達成")
    ax2.axhspan(*BAND, alpha=.10, color="gray")
    ax2.set_xlabel("旋轉角上界 θ"); ax2.set_ylabel("防禦圖失真 DISTS")
    ax2.grid(alpha=.25); ax2.legend(fontsize=8.5)
    ax2.set_title("可行集夠大（天花板遠在帶上），最佳化用掉約六成", fontsize=11)
    ceil = fig_uri(fig2)

    zero_pair = st.fmean(float(r["zero_pair_frac"]) for r in ce
                         if r["variant"] == "post_int" and r["zero_pair_frac"])
    dw = st.fmean(float(r["delta_within_1"]) for r in ce
                  if r["variant"] == "post_int" and float(r["theta"]) == 1.2
                  and r["delta_within_1"])

    body = f"""
<div class='verdict'><b>判定：兩條事前判準都不通過。</b>
在等失真上，DCT 域的保長配對旋轉只買到現行量化交付 <b>{ratio:.0%}</b> 的位移，
四個工作點的人眼代理擋下數全部是 0/10。配對規則的對照組與主設定分不開，
「保長在感知上有意義」這個唯一的設計依據沒有實證支持。</div>

<h2>一、要問的是什麼</h2>
<p>現行做法是在<b>像素域</b>做完切窗 STFT 的相位旋轉、得到連續值影像、才壓成
JPEG 交付——最佳化在一個空間裡找解，交付把解投影到另一個空間，代價已經量到
（等失真下未淨化位移掉 21%）。本方向把參數化直接搬進 JPEG 自己的 8×8 DCT
係數：同一個區塊內的兩格係數當成平面向量做旋轉，<code>θ = π</code> 恰好就是
聯合翻號，也就是文獻上「DCT 的相位即係數的正負號」（Ito &amp; Kiya 2007）的
連續化。可行集<b>就是</b>交付集，所以那 21% 不該存在——這是一條可證偽的預測。</p>

<h2>二、防禦讀數（十張、1000 步、交付品質 0.85）</h2>
<div class='wrap'>{table(["條件", "DISTS↓", "未淨化位移↑", "擋下"], rows, highlight=[5, 6])}</div>
<figure><img src='{curve}' alt='等失真曲線'><figcaption>
灰帶是本專案的失真帶。黑色叉號是把 DCT 旋轉那條曲線內插到現行做法的失真高度
（{anchor:.4f}）之後的位移 {interp:.4f}，對照組是 {data['ours_ph_q'][1]:.4f}。
</figcaption></figure>

<div class='verdict'><b>D1 不通過</b>
判準逐字寫著「等失真下的未淨化位移沒有高於同失真的量化交付點，那條預測就是
錯的」。它不但沒有高於，還低了四成。<b>要撤回的是</b>：消掉交付投影並沒有換來
位移，換到的是一個更小的可行集。</div>

<div class='verdict'><b>D2 不通過</b>
<code>zigzag</code>（{data['dct_rot_zz11'][0]:.4f} / {data['dct_rot_zz11'][1]:.4f}）
與 <code>transpose</code>（{data['dct_rot_t11'][0]:.4f} / {data['dct_rot_t11'][1]:.4f}）
在幾乎同一個失真上差 0.9%，分不開。這是<b>事前排好</b>的對照組，它否證的是
配對規則的唯一依據——「保長只有在兩軸價錢相同時才有感知意義」<b>不可以寫進
論文</b>。</div>

<h2>三、為什麼弱：容量，不是最佳化沒收斂</h2>
<figure><img src='{ceil}' alt='天花板與實際達成'><figcaption>
天花板是「把每一對合格係數都轉到角度上界」，不最佳化、純 CPU。
</figcaption></figure>
<ul>
<li><b>{zero_pair:.1%} 的合格配對是零向量。</b> 交付品質 0.85 的量化把大部分
高頻係數清成 0，而旋轉零向量還是零向量。28 對 × 64×64 區塊名目上有 11.4 萬個
自由度，實際帶得動的約四分之一；現行方法在 32×32 STFT 上是 59 萬個。</li>
<li><b>L∞ 反而更高</b>（θ1.5 是 0.7047，而現行做法在更高的 DISTS 上只有
0.3608）。少數還活著的大係數被轉了很大的角度，失真集中成稀疏尖峰——正是
「可用位置太少」的症狀。</li>
<li><b>最佳化本身沒有壞。</b> 隨機對照（同角度上界、不最佳化）在等失真上是
最佳化解的 1/1.42；也就是最佳化買到了 42%。<b>這與位移場是不同的死法</b>
——那邊最佳化過的比同半徑隨機的還弱。<b>兩者的下一步不同</b>：位移場要修
更新規則，這裡修更新規則不會有幫助。</li>
</ul>

<h2>四、新穎性的上限（必須照實寫）</h2>
<p>「DCT 的相位是正負號」是 Ito &amp; Kiya (2007) 的既有結論，不是本專案的發現。
帶內工作點上 <b>{dw:.1%}</b> 的整數位移落在 DCT-Shield 的 ε=1 球內，所以
「我們做得到它做不到的事」站不住。可主張的只有<b>約束不同</b>（保長的受約束
子集），不是動作不同——而 D2 又否證了那個約束的感知依據。</p>

<h2>五、還開著的路（都要新理由才值得花機時）</h2>
<ul>
<li><b>提高交付品質</b>（0.95 而非 0.85）會讓零向量配對變少。但那把交付推向
「攻擊方壓得比我們狠」的那一側，而本方法的優勢正好在低品質壓縮——<b>方向相反</b>。</li>
<li><b>放棄保長</b>，改成在 DCT 域做一般的受約束擾動。那就是 DCT-Shield，不是新方法。</li>
<li><b>跨區塊配對</b>。沒有量過，但 8×8 區塊不重疊正是本設計「失真可精確預測」
的來源，跨區塊會把那個唯一的硬性質拿掉。</li>
</ul>
<p class='sub'>抗淨化<b>沒有跑，也不建議跑</b>：D1 逐字寫著不通過就不必再走，
位移只有對照組六成、擋下 0/10，淨增益的分母在那個高度不具鑑別力。<br>
數值：<code>runs/ip2p_dct_rotate/</code>、<code>runs/dct_phase_design/ceiling.csv</code>。</p>
"""
    return ("DCT 域保長旋轉：判定與證據",
            "十張主線影像 · InstructPix2Pix · 判準在看到數字之前就寫定", body)


# ------------------------------------------------------------------ 位移場

def build_warp(root: Path) -> Tuple[str, str, str]:
    warp = root / "runs" / "ip2p_warp"
    bu = read(warp / "budget_utilization.csv")

    curves: Dict[str, List[Tuple[float, float]]] = {}
    for r in bu:
        curves.setdefault(r["field"], []).append(
            (float(r["amp_px"]), float(r["fid_dists"])))
    for k in curves:
        curves[k].sort()

    opt = {}
    for rad in (4, 8, 16, 24):
        p = warp / f"opt_r{rad}" / "results.csv"
        if p.exists():
            rows = read(p)
            opt[rad] = (mean_of(rows, "fid_dists"), mean_of(rows, "edit_lpips"))

    def invert(curve, y):
        for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
            if y0 <= y <= y1:
                return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
        return float("nan")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    lab = {"corner": "corner（L∞ 球的頂點＝天花板）",
           "gauss": "gauss（隨機場）", "coherent": "coherent（整張平移＝最平滑）"}
    for k in ("corner", "gauss", "coherent"):
        ax.plot([a for a, _ in curves[k]], [d for _, d in curves[k]], "o-",
                label=lab[k], ms=4)
    ax.plot(list(opt), [opt[r][0] for r in opt], "s--", ms=8, color="k",
            label="最佳化實際達成（1000 步）")
    ax.axhspan(*BAND, alpha=.12, color="gray")
    ax.set_xscale("log"); ax.set_xlabel("位移幅度 amp（px，對數軸）")
    ax.set_ylabel("防禦圖失真 DISTS")
    ax.grid(alpha=.25); ax.legend(fontsize=8.5)
    ax.set_title("預算從頭到尾就夠：4 px 的天花板已經在失真帶內", fontsize=11)
    budget = fig_uri(fig)

    corner = dict(curves["corner"])
    rows = []
    for rad in sorted(opt):
        d = opt[rad][0]
        rows.append([f"opt_r{rad}", f"{rad} px", f"{d:.4f}",
                     f"{invert(curves['gauss'], d):.2f} px",
                     f"{corner.get(float(rad), float('nan')):.4f}",
                     f"{d / corner.get(float(rad), float('nan')):.1%}"])

    alpha_rows = []
    for rad in sorted(opt):
        a = rad / (1000 * 0.25)
        pred = a * (1000 ** .5)
        eff = invert(curves["gauss"], opt[rad][0])
        alpha_rows.append([f"opt_r{rad}", f"{a:.3f}", f"{pred:.2f} px",
                           f"{eff:.2f} px", f"{eff / pred:.2f}"])

    body = f"""
<div class='verdict'><b>判定：不是預算不夠，是最佳化沒有走。</b>
L∞ 球在 amp = 4 px 的天花板已經是 DISTS {corner[4.0]:.4f}，落在失真帶內——
而已跑過的最小半徑正是 4 px。四個半徑的可行集裡全都有帶內的點，最佳化卻只
用掉天花板的 7–15%，而且走出來的東西與擲硬幣無法區分。</div>

<h2>一、可行集的天花板</h2>
<figure><img src='{budget}' alt='預算曲線'><figcaption>
三條曲線都在同一組十三張、16×16 粗網格上，<b>不跑 GPU、不載編輯模型</b>。
灰帶是失真帶。
</figcaption></figure>
<div class='wrap'>{table(
    ["跑次", "半徑上界", "觀測 DISTS", "等效有效位移", "同半徑天花板", "預算使用率"], rows)}</div>
<p><b>半徑放大六倍，等效位移只從 0.43 px 走到 1.52 px。</b> 上界從頭到尾沒有
咬到。「等效有效位移」是用 <code>gauss</code>（粗糙那一端）反推；用
<code>coherent</code> 反推是約 7 px，故使用率讀成區間 <b>6–29%</b> 較誠實，
但結論不變。</p>

<h2>二、走出來的東西與擲硬幣沒有差別</h2>
<p>sign-PGD 的步長是 <code>α = radius / (steps · saturate_at)</code>。若梯度符號
<b>無偏</b>，每一格就是步長 α 的 ±1 隨機遊走，1000 步之後 <code>|c| ≈ α√N</code>。</p>
<div class='wrap'>{table(
    ["跑次", "α (px/step)", "無偏隨機遊走的預測", "觀測等效位移", "比值"], alpha_rows)}</div>
<p>事前寫下的規則是：比值落在 0.5–2 之間代表「與擲硬幣沒有差別」；小於 0.5
代表症狀是<b>振盪</b>，救法是換更新規則而不是加步數。四個跑次全部落在
0.40–0.84，<code>opt_r16</code> 的 0.40 已經掉到振盪那一側。<b>加步數或加預算
都不會有幫助。</b></p>

<h2>三、兩個可以指名的原因</h2>
<ul>
<li><b>初始化恰好落在病態點上。</b> <code>WarpParam.reset</code> 把位移場設成
全零，而逐步探針量到 <code>latent_norm</code> 在零位移處有一個<b>帶折點的局部
極小</b>：梯度完全正常（absmean 3.4e−2、零元素比例 0.0000）但每走一步損失都
上升（105.95 → 110.07），sign-PGD 於是在 0 與 ±α 之間形成週期 2 的振盪。
<code>reset</code> 收下 <code>seed</code> 卻沒有使用，所以現行程式沒有隨機起點可用。</li>
<li><b>步長綁在半徑上。</b>「放寬預算」同時「放大步長」，所以放寬預算不等於
走得更遠，它同時讓振盪的幅度變大——比值隨半徑下降（0.84 → 0.50 → 0.40）正是
這個形狀。</li>
</ul>

<h2>四、就算最佳化修好，天花板在哪</h2>
<p><b>平滑的位移場在任何幅度上都到不了失真帶</b>：<code>coherent</code>（整張
平移）放到 24 px 的天花板只有 {dict(curves['coherent'])[24.0]:.4f}，仍在
0.1286 之外。<b>失真來自位移場的粗糙度，不是幅度</b>——若最佳化收斂到平滑解，
加多少預算都到不了帶內。</p>
<p>另外，等失真錨點 DISTS 0.1286 上，隨機位移場的編輯位移是 0.2853、擋下率
0.087，而「同一個隨機場先施加 f 再施加 −f」（只留內插 artifact）是 0.2609 /
0.091——<b>相差 8.6%，落在該批事前寫下的 &lt;10% 門檻內</b>。依那條規則，
這一族的作用機制是<b>重取樣內插的低通，不是幾何</b>。整族在等失真上的隨機
基準只到本方法工作點（0.7011）的 41%。</p>

<h2>五、要繼續的話，下一批該長什麼樣</h2>
<p>兩個混淆必須先拆掉，否則量到的是 sign-PGD 的性質而不是損失的性質：
<b>(1)</b> 把步長與半徑解耦（<code>ip2p_run.py</code> 加 <code>--saturate-at</code>，
預設不變）；<b>(2)</b> 讓起點離開折點（<code>WarpParam.reset</code> 加
<code>init_std</code>，預設 0 時逐位元等於現在）。兩者都是加參數、不改既有行為，
但都動到程式。</p>
<p>六個條件（兩個損失 × 步長 × 起點）、13 張、4000 步、實質不設限的半徑，
合計約 <b>2.6 GPU 小時</b>，三卡六槽<b>一次送完約 26 分鐘</b>。判準要加一條
<b>W0</b>：每一格都要一併報場的粗糙度，否則「走不到帶內」會被誤讀成預算或
步數問題。</p>
<p class='sub'>數值：<code>runs/ip2p_warp/budget_utilization.csv</code>、
<code>DIAGNOSIS.md</code>、<code>matched_geometry.csv</code>、
<code>step_probe_*.csv</code>。</p>
"""
    return ("位移場：預算夠，是最佳化沒有走",
            "十三張 · 16×16 粗網格 · 天花板與使用率皆為純 CPU 量測", body)


BUILDERS = {"dct_rotate": build_dct, "warp": build_warp}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    title, sub, body = BUILDERS[args.direction](args.root)
    args.out.write_text(page(title, sub, body), encoding="utf-8")
    size = args.out.stat().st_size / 1e6
    print(f"寫出 {args.out}（{size:.2f} MB，圖以 data URI 內嵌，可任意搬移）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
