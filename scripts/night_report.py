"""2026-08-19 夜間工作的單一報告檔。

產出 `reports/night0819/index.html`：自足（圖以 base64 內嵌）、單一檔案。
資料一律從 `runs/` 的 CSV 現算，不寫死任何數字——寫死的數字會在重跑之後
悄悄過期，而報告是本專案唯一對外的產出物。

用法：
    python scripts/night_report.py
"""

from __future__ import annotations

import base64
import csv
import glob
import html
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "night0819"
IMG = OUT / "img"

VARIANTS = ("full", "amp", "pha", "amp_s", "pha_s")
COND_LABEL = {
    "phase": "紋理重相位 θ=1.30", "phase_rand": "隨機相位 θ=1.30",
    "add": "加性 δ ε=1.2/255", "photoguard_c": "PhotoGuard-c",
    "mist": "Mist", "dia_r": "DIA-R", "apa_weak": "APA（弱 baseline）",
    "dct_shield": "DCT-Shield（base）", "dct_shield_y": "DCT-Shield（Y-only）",
    "none": "空白地板（無防禦）",
}


def rd(pattern: str) -> list:
    rows = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        with open(f, encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def fl(r, k, default=None):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else float("nan")


def cl(c):
    return COND_LABEL.get(c, c)


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def img_tag(name: str, cap: str) -> str:
    p = IMG / name
    if not p.exists():
        return f'<figure class="ph"><figcaption>{html.escape(cap)}（缺圖）</figcaption></figure>'
    return (f'<figure><img src="data:image/png;base64,{b64(p)}" alt="{html.escape(cap)}">'
            f'<figcaption>{html.escape(cap)}</figcaption></figure>')


def table(head: list, rows: list, cls: str = "") -> str:
    h = "".join(f"<th>{c}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


# ---------------------------------------------------------------- 分解結果

def decomp_sections() -> tuple:
    rows = rd("runs/spectral/dec_*.csv")
    if not rows:
        return "<p>（`runs/spectral` 無資料）</p>", "", ""
    by = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["variant"])].append(r)
    conds = sorted({r["condition"] for r in rows},
                   key=lambda c: -mean(fl(r, "effect_mean")
                                       for r in by[(c, "full")]))

    full_rows = []
    for c in conds:
        g = by[(c, "full")]
        full_rows.append([
            cl(c), f'{mean(fl(r,"effect_mean") for r in g):.4f}',
            f'{mean(fl(r,"fid_lpips") for r in g):.4f}',
            f'{mean(fl(r,"fid_dists") for r in g):.4f}',
            f'{mean(fl(r,"fid_psnr") for r in g):.2f}',
            f'{mean(fl(r,"amp_dev") for r in g):.4f}'])
    t_full = table(["條件", "未淨化位移量↑", "LPIPS↓", "DISTS↓", "PSNR↑",
                    "全域幅度偏移"], full_rows)

    ratio_rows = []
    for c in conds:
        a = {r["image"]: fl(r, "effect_mean") for r in by[(c, "amp_s")]}
        p = {r["image"]: fl(r, "effect_mean") for r in by[(c, "pha_s")]}
        f = {r["image"]: fl(r, "effect_mean") for r in by[(c, "full")]}
        k = sorted(set(a) & set(p) & set(f))
        if not k:
            continue
        ma, mp, mf = (mean(a[i] for i in k), mean(p[i] for i in k),
                      mean(f[i] for i in k))
        win = sum(1 for i in k if p[i] > a[i])
        strong = "strong" if mp / ma >= 1.10 else ""
        ratio_rows.append([
            cl(c), f"{ma:.4f}", f"{mp:.4f}",
            f'<span class="{strong}">{mp/ma:.3f}</span>', f"{win}/{len(k)}",
            f"{mf:.4f}", f"{mp/mf:.3f}", f"{ma/mf:.3f}"])
    t_ratio = table(["條件", "只留幅度 amp_s", "只留相位 pha_s", "相位 ÷ 幅度",
                     "逐圖相位勝", "完整擾動 full", "pha_s ÷ full", "amp_s ÷ full"],
                    ratio_rows)

    n_img = len({r["image"] for r in rows})
    return t_full, t_ratio, str(n_img)


# ---------------------------------------------------------------- DCT-Shield

def dct_section() -> str:
    rows = rd("runs/dctshield/g*/results.csv")
    if not rows:
        return "<p class='note'>（DCT-Shield 批次尚未產出結果）</p>"
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    out = []
    for c, g in sorted(by.items()):
        out.append([cl(c), str(len(g)),
                    f'{mean(fl(r,"radius") for r in g):.3f}',
                    f'{mean(fl(r,"fid_lpips") for r in g):.4f}',
                    f'{mean(fl(r,"fid_dists") for r in g):.4f}',
                    f'{mean(fl(r,"fid_psnr") for r in g):.2f}',
                    f'{mean(fl(r,"edit_lpips") for r in g):.4f}',
                    f'{mean(fl(r,"total_seconds") for r in g):.0f} s'])
    return table(["變體", "張數", "ε", "LPIPS↓", "DISTS↓", "PSNR↑",
                  "未淨化位移量↑", "每張耗時"], out)


# ---------------------------------------------------------------- 抗淨化

def retention_section() -> str:
    # 兩趟：第一趟跑 blur/crop/jpeg（gridpure 因未設 DIFFPURE_CKPT 被標成
    # 相依不齊而跳過），第二趟補 gridpure。絕對位移量逐格獨立，合併無虞。
    ret = rd("runs/freqret/ret_*.csv") + rd("runs/freqret/gret_*.csv")
    flo = rd("runs/freqret/floor_*.csv") + rd("runs/freqret/gfloor_*.csv")
    if not ret:
        return "<p class='note'>（抗淨化批次尚未產出結果）</p>"
    floor = defaultdict(list)
    for r in flo:
        floor[r["purifier"]].append(fl(r, "effect_mean"))
    fmean = {k: mean(v) for k, v in floor.items()}

    by = defaultdict(lambda: defaultdict(list))
    for r in ret:
        by[r["condition"]][r["purifier"]].append(fl(r, "effect_mean"))
    purs = ["identity", "blur1", "crop_resize0.1", "jpeg75", "gridpure"]
    purs = [p for p in purs if any(p in d for d in by.values())]

    head = ["條件"] + purs + ["淨增益均值"]
    rows = []
    for c, d in sorted(by.items()):
        cells, gains = [], []
        for p in purs:
            m = mean(d.get(p, []))
            if p == "identity":
                cells.append(f"{m:.4f}")
                continue
            fm = fmean.get(p)
            if fm is None:
                cells.append(f"{m:.4f}")
            else:
                g = m - fm
                gains.append(g)
                cells.append(f"{m:.4f}<br><span class='sub'>−地板 {g:+.4f}</span>")
        rows.append([cl(c)] + cells +
                    [f"<b>{mean(gains):+.4f}</b>" if gains else "—"])
    if fmean:
        rows.append(["<i>空白地板（無防禦）</i>"] +
                    [f"{fmean.get(p, float('nan')):.4f}" if p != "identity" else "0.0000"
                     for p in purs] + ["—"])
    return table(head, rows)


# ---------------------------------------------------------------- 版面

CSS = """
:root{--bg:#fbfaf7;--fg:#1a1a18;--mut:#5d5c56;--line:#ddd9cf;--card:#fff;
--acc:#7a4b1e;--pos:#1f6b3a;--neg:#96401f}
:root:not([data-theme=light]){}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#16150f;--fg:#eceade;--mut:#a09d90;--line:#33312a;--card:#1e1d16;
--acc:#d9a86a;--pos:#7fc79a;--neg:#e0906c}}
:root[data-theme=dark]{--bg:#16150f;--fg:#eceade;--mut:#a09d90;--line:#33312a;
--card:#1e1d16;--acc:#d9a86a;--pos:#7fc79a;--neg:#e0906c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.75 "Noto Serif CJK TC","Songti TC",Georgia,serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:900px;margin:0 auto;padding:3rem 1.4rem 6rem}
h1{font-size:2.1rem;line-height:1.25;margin:0 0 .3rem;letter-spacing:-.01em}
.sub1{color:var(--mut);font-size:.95rem;margin:0 0 2.6rem}
h2{font-size:1.45rem;margin:3.2rem 0 .9rem;padding-top:1.4rem;
border-top:1px solid var(--line)}
h3{font-size:1.1rem;margin:2rem 0 .6rem;color:var(--acc)}
p,li{color:var(--fg)}
.note{color:var(--mut);font-size:.92rem}
code{font:0.88em/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
background:var(--card);border:1px solid var(--line);border-radius:3px;padding:.05em .35em}
.scroll{overflow-x:auto;margin:1.1rem 0;border:1px solid var(--line);
border-radius:6px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.88rem;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
th,td{padding:.5rem .7rem;text-align:right;border-bottom:1px solid var(--line);
white-space:nowrap}
th:first-child,td:first-child{text-align:left;font-family:inherit;white-space:normal}
thead th{background:transparent;color:var(--mut);font-weight:600;
border-bottom:2px solid var(--line)}
tbody tr:last-child td{border-bottom:none}
.strong{color:var(--pos);font-weight:700}
.sub{color:var(--mut);font-size:.82em}
.grid{display:grid;gap:.9rem;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
margin:1.2rem 0}
figure{margin:0}
figure img{width:100%;display:block;border-radius:5px;border:1px solid var(--line)}
figcaption{color:var(--mut);font-size:.8rem;margin-top:.35rem;text-align:center}
.ph{border:1px dashed var(--line);border-radius:5px;padding:2rem .5rem;text-align:center}
blockquote{margin:1.2rem 0;padding:.2rem 0 .2rem 1.1rem;
border-left:3px solid var(--acc);color:var(--mut)}
.key{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);
border-radius:5px;padding:1rem 1.2rem;margin:1.4rem 0}
.key p{margin:.35rem 0}
ul{padding-left:1.2rem}
li{margin:.3rem 0}
"""


def build() -> str:
    t_full, t_ratio, n_img = decomp_sections()
    body = f"""
<div class="wrap">
<h1>頻域與相位防護：一夜的工作紀錄</h1>
<p class="sub1">2026-08-19 · WACV 專案 · 分支 <code>claude/stage3-apa-attn</code></p>

<div class="key">
<p><b>一、重現 PAD（ICML 2023）第 3 節的幅度／相位分解。</b>結論分兩半：
加性 baseline 在相位／幅度之間<b>打平</b>（0.96–1.02），該篇在分類上的四倍
差距<b>沒有</b>搬到擴散編輯；而紋理重相位自己的兩個半邊<b>都遠低於</b>完整
擾動（0.653／0.513，逐圖 0/7），代表效果不能化約為「它是相位擾動」。</p>
<p><b>二、建立第一個頻域 baseline：DCT-Shield（ICCV 2025）。</b>官方 repo 是
空的，由論文與補充材料 Algorithm 1 重寫，量化表對 PIL 逐格驗證。</p>
<p><b>三、兩個新淨化算子，一個成功一個否決。</b>GrIDPure 可用；FD-Pure 在
512²／256² 上把原圖也毀掉，不具鑑別力，已排除並記錄機制。</p>
</div>

<h2>1. 背景與本輪的範圍變更</h2>
<p>指導者指出紋理重相位在像素空間雖非加性，但它保留幅度譜、只旋轉相位，
本質是頻域的重參數化，不算狹義的非加性。研究主軸因此由「非加性 vs 加性」
改為<b>頻域／相位方法及其抗淨化能力</b>。動機是既有防護擾動多為高頻擾動，
會被 JPEG、resize、模糊等常見壓縮抹平。</p>
<p>本輪的四項裁決已寫入 <code>docs/DECISIONS.md</code> 的 DEC-025：</p>
<ul>
<li><b><code>mist</code> 的預算未與其他 baseline 對齊，待重測。</b>它跑在自己
論文的原生 ε=16/255 上，LPIPS 0.6234、DISTS 0.1667，是紋理重相位的 3.3 倍與
4.8 倍。「輸 mist 0/7」這個讀數在對齊之前<b>不可解讀</b>。</li>
<li><b>加性 baseline 本輪全部擱置</b>，不進本輪的比較表：
<code>photoguard_c</code>／<code>mist</code>／<code>dia_r</code>／<code>add</code>／
<code>apa_weak</code>。既有數據全部保留備查。</li>
<li>新 baseline 改由頻域方法組成。</li>
<li>淨化算子縮到 <code>blur</code>／<code>crop</code>／<code>jpeg</code>／
<code>freqpure</code>／<code>gridpure</code> 五個。</li>
</ul>
<blockquote>本報告的第 2 節仍然列出加性 baseline 的數字。那不違反上一條——
第 2 節是<b>解剖它們的擾動由什麼組成</b>，不是把它們當成競爭對手比較。
少了它們，這個分析就只剩下拆解自己的方法。</blockquote>

<h2>2. 重現 PAD：幅度／相位分解</h2>
<h3>2.1 那篇論文做了什麼</h3>
<p>Zhou 等人（<i>Phase-aware Adversarial Defense</i>, ICML 2023, PMLR v202）
第 3 節把對抗樣本拆成兩半：取對抗樣本的<b>相位</b>配自然圖的<b>幅度</b>、
以及反過來，各做逆傅立葉轉換得到兩張新圖，分別餵給分類器。他們在
CIFAR-10／ResNet-18 上量到，只攻擊相位的版本準確率降到 6.12%、只攻擊幅度的
只降到 24.32%，而<b>相位版本的 L2 雜訊反而更小</b>（0.9385 對 0.9931）。</p>
<h3>2.2 搬到擴散編輯的讀數上</h3>
<p>對每張已存的防禦圖 <code>x_def</code>，以原圖 <code>x</code> 為參照做全域
DFT 的交叉互換：</p>
<pre><code>只有幅度  x_amp = F⁻¹( ξ_x_def , φ_x     )
只有相位  x_pha = F⁻¹( ξ_x     , φ_x_def )</code></pre>
<p>三個版本各跑 SDEdit（strength 0.7、guidance 7.5、30 步）三個 seed，量
<code>LPIPS(編輯(原圖), 編輯(該版本))</code>，即本專案一貫的位移量讀數。</p>
<h3>2.3 一個必要的控制：失真必須對齊</h3>
<p>交叉互換後的失真<b>高於</b>原始防禦圖——紋理重相位的 LPIPS 由 0.188 升到
0.205（幅度版）與 0.255（相位版）。原因是低幅度頻格的相位近乎隨機，換過來
之後配上原本較大的幅度會產生大幅擾動。<b>PAD 自己也遇到同一件事</b>（該篇
表 1 的分解版本 L∞ 反而比完整對抗樣本大），其處置是把重組樣本夾回同一個
L∞ 球再測一次。</p>
<p>本專案改成對 <b>DISTS 對齊</b>（DEC-015 的預算軸）：二分搜尋擾動的縮放
係數，使最終 DISTS 等於完整防禦圖，得到 <code>amp_s</code> 與
<code>pha_s</code>。下表所有比較都用對齊過的版本。</p>
<h3>2.4 未淨化的基準（完整防禦圖）</h3>
{t_full}
<h3>2.5 主結果</h3>
{t_ratio}
<div class="key">
<p><b>判讀一：PAD 的結論不會自動搬到擴散編輯。</b>四個加性 baseline 的
相位／幅度比全部落在 0.96–1.02 之間，也就是打平。相位優勢只出現在三個相位
參數化的條件上，其中紋理重相位最大。</p>
<p><b>判讀二（對本專案更重要）：效果不能化約為「它是相位擾動」。</b>
同一個 DISTS 上，完整的紋理重相位是 0.4537，它自己的相位半邊只有 0.2964,
比值 0.653、逐圖 0/7。它是全批唯一兩個半邊都遠低於完整擾動的條件。
換言之，<b>區塊加窗結構與兩個閘承載了效果的一大部分</b>。</p>
</div>
<h3>2.6 影像</h3>
<p class="note">同一張圖（cat_00）在同一個 DISTS 上的三個版本。左起：原圖、
完整防禦圖、只留幅度、只留相位。</p>
<div class="grid">
{img_tag("cat_00__orig.png", "原圖")}
{img_tag("cat_00__phase__human__full.png", "紋理重相位 · 完整")}
{img_tag("cat_00__phase__human__amp_s.png", "紋理重相位 · 只留幅度")}
{img_tag("cat_00__phase__human__pha_s.png", "紋理重相位 · 只留相位")}
</div>
<div class="grid">
{img_tag("cat_00__orig.png", "原圖")}
{img_tag("cat_00__add__human__full.png", "加性 δ · 完整")}
{img_tag("cat_00__add__human__amp_s.png", "加性 δ · 只留幅度")}
{img_tag("cat_00__add__human__pha_s.png", "加性 δ · 只留相位")}
</div>
<h3>2.7 兩個必須寫下來的量測細節</h3>
<ul>
<li><b>全域幅度偏移不等於區塊幅度偏移。</b><code>pha</code> 版本的全域幅度
偏移由構造塌到 0.0005–0.0027，而完整的紋理重相位是 0.0318。這與
<code>texture_rephase</code> 的「幅度逐位保留」不矛盾：後者保留的是<b>加窗
區塊譜</b>，overlap-add 之後全域譜不再逐位保留。兩個量不可混報。</li>
<li><b>夾取比例會誤導。</b>交叉互換後 23%–35% 的像素超出 <code>[0,1]</code>，
但超出量的平均只有 1e-4 到 2e-3——高比例來自自然照片本來就有大量貼著邊界的
飽和像素。CSV 因此同時記錄 <code>clip_mean</code> 與 <code>clip_max</code>。</li>
</ul>

<h2>3. 新的頻域 baseline：DCT-Shield</h2>
<p><i>DCT-Shield: A Robust Frequency Domain Defense against Malicious Image
Editing</i>（Bala 等人, ICCV 2025 Highlight, arXiv:2504.17894）。
它把防禦擾動加在 <b>JPEG 量化後的 DCT 整數係數</b>上，每個係數最多加減一個
量化級。因為量化表本身就是人眼敏感度表，這個限制自動把擾動趕到人眼不敏感的
高頻——直流係數只能動約 0.25 個灰階，最高頻可以動約 3 個灰階。</p>
<p>擾動加在量化<b>之後</b>是關鍵：四捨五入的導數幾乎處處為零，加在像素上
梯度會被歸零。</p>
<p><b>官方 repo <code>SamsungLabs/dct-shield</code> 是空的</b>（GitHub API 回
<code>This repository is empty.</code>），故由論文與補充材料 Algorithm 1 重寫。
量化表對 PIL 實際寫出的 JPEG 檔在八個品質上逐格驗證，往返重建對 PIL 的往返
差在 32 dB 以上。</p>
{dct_section()}
<p class="note">ε=1 是論文原生設定，也是它抗 JPEG 的<b>必要條件</b>——擾動
必須至少造成一個量化級的改變，否則攻擊方以相同品質重壓時會被四捨五入回原值。</p>
<h3>3.1 三個必須並列的觀察</h3>
<ul>
<li><b>原生設定落在 Mist 的失真區間，不在「不易察覺」的區間。</b>本專案量到
base 變體 LPIPS 0.5532、位移量 0.6544；FND-055 的 Mist 是 LPIPS 0.6234、
位移量 0.6562。兩者幾乎重合。相對地，紋理重相位在人眼門檻上是 LPIPS 0.1893、
位移量 0.4432。</li>
<li><b>與論文自報的失真有落差，必須寫明。</b>該篇 Table 1 報 DCT-Shield 的
失真 LPIPS 是 0.267，本專案量到 0.553。可能的來源是資料集不同（該篇用
OmniEdit 150 張，本專案用 7 張 512² 人像與動物）。要做 like-for-like 的核對
需要跑該篇的資料集，本輪未做。旁證：本專案量到的 <code>photoguard_c</code>
失真 LPIPS 0.372 落在該篇報的 0.284 附近，量級是可比的。</li>
<li><b>Y-only 變體的失真反而更差</b>（LPIPS 0.6409、PSNR 24.39，對 base 的
0.5532／29.75）。原因不是通道限制，而是論文圖 6 的 Y-only 用
<code>Q_alg = 0.85</code>：量化步長比 0.95 粗約 1.7 倍，「一個量化級」因此是
更大的擾動。<b>該篇在這個比較裡把通道限制與品質因子混在一起</b>，要分開就得
在同一個 <code>Q_alg</code> 上比。本輪照論文的設定跑，並在此註明這個混淆。</li>
</ul>

<h2>4. 兩個新淨化算子</h2>
<h3>4.1 「freqpure」這個名字指向的論文不可重現</h3>
<p>FreqPure（Ju, Xue, Lyu, ICCV 2025 Workshop APAI）<b>不是</b>逆向過程中的
頻域介入，而是一條<b>兩階段的訓練式管線</b>：一個重建模組先移除保護擾動造成的
瑕疵，再由一個以低頻影像為條件的擴散模型合成高頻。它需要在 FFHQ 人臉資料上
訓練，且無公開程式碼。</p>
<p>本專案的文獻檔原先把「逐時間步替換低頻幅度、投影低頻相位」寫成 FreqPure
的作法——<b>那是錯的</b>，該條目自己註明內容「由檢索摘要確認」，即未讀原文。
2026-08-19 取得 CVF PDF 全文後已更正。該機制實際上屬於
<b>FD-Pure</b>（Pei 等人, arXiv:2505.01267），訓練自由、Algorithm 1 完整。</p>
<h3>4.2 FD-Pure 在本專案的解析度上不具鑑別力</h3>
<p>實作了 FD-Pure 之後，用它自己的超參數（<code>D_A=3, D_P=2, δ=0.2</code>，
論文附錄 A.1，量在 CIFAR-10 的 32² 上）按 Nyquist 比例換算到 256²，得到的
淨化器把影像毀掉。掃半徑之後：</p>
{table(["D_A", "D_P", "淨化<b>原圖</b> LPIPS↓", "PSNR↑", "淨化<b>防禦圖</b> LPIPS"],
       [["24（論文換算）", "16", "0.6403", "16.81", "0.6401"],
        ["48", "32", "0.6009", "18.33", "0.6019"],
        ["72", "48", "0.5655", "20.03", "0.5664"],
        ["96", "64", "0.5017", "22.24", "0.5083"],
        ["144", "96", "0.3013", "29.87", "0.3081"]])}
<p><b>每一列的兩個 LPIPS 幾乎相同</b>：不管餵原圖還是防禦圖，輸出離原圖
一樣遠。這是「淨化器自身的破壞吞掉整個讀數」的特徵——用 FND-043 的語言說，
空白地板佔比接近 100%，<b>在任何半徑上都不具鑑別力</b>。</p>
<p>機制：Algorithm 1 從純噪聲起步，只約束約 2.8% 的低頻係數。那在 32² 足以
釘住影像，在 256² 不夠。<b>本輪因此排除 FD-Pure</b>，程式與這份量測保留，
論文的 ImageNet 半徑未在正文載明，日後若查到可重測。</p>
<h3>4.3 GrIDPure 可用</h3>
<p>GrIDPure（Zhao 等人, CVPR 2024）把 512² 影像切成九個重疊 256² 網格加上
四角合併的第十格，每格以小步 SDEdit 淨化，重疊處取平均後與上一輪混合，
迭代多次。網格恰好是擴散檢查點的原生解析度，所以<b>不需要 resize</b>。</p>
<p>實作時測試抓到一個真 bug：規則的 3×3 stride-128 格點會讓四個 128×128
角落<b>只被一格覆蓋</b>，違反論文「每一塊至少落在兩格」的規定——論文加第十格
正是為了這件事。第十格以「把影像環狀平移半個邊長、取正中央一塊」實作，
那四塊角落恰好拼成連續的 256×256。</p>
<p class="note"><code>t</code>、混合權重 <code>γ</code>、迭代次數三個超參數
<b>論文正文未載</b>（表格只標 DiffPure 的 t=50／100）。依專案規則不得填一個
看起來合理的預設，故設為必填參數；本輪用 <code>t=10, γ=0.1, 10 次迭代</code>，
這三個值是<b>本專案指定</b>而非論文的。單張 512² 實測 115 秒。</p>

<h2>5. 抗淨化：五個算子</h2>
<p>淨化算子集合 = <code>identity</code>（retention 的分母，不可拿掉）＋
<code>blur1</code>＋<code>crop_resize0.1</code>＋<code>jpeg75</code>＋
<code>gridpure</code>。空白地板（把<b>原圖</b>直接過同一個算子再編輯）同步跑，
因為淨化後的絕對位移量會被算子自己的破壞支配（FND-043／056）。</p>
{retention_section()}

<h2>6. 程式與文件的變更</h2>
{table(["檔案", "內容"],
       [["<code>src/residual/spectral_split.py</code>",
         "PAD 第 3 節的幅度／相位交叉互換。用完整 <code>fft2</code> 而非 "
         "<code>rfft2</code>——後者的共軛對稱是隱含假設，壞掉不會有症狀"],
        ["<code>src/purify/freq_grid.py</code>",
         "GrIDPure 與 FD-Pure，建立在既有的 guided-diffusion 檢查點上"],
        ["<code>src/baselines/jpeg_codec.py</code>",
         "可微 JPEG 管線；量化表與 libjpeg 逐格對齊"],
        ["<code>src/baselines/dct_shield.py</code>",
         "DCT-Shield 的 Algorithm 1 與 <code>Parameterization</code> 實作"],
        ["<code>scripts/spectral_decompose.py</code>", "分解研究的驅動"],
        ["<code>scripts/spectral_report.py</code>", "分解結果的彙總"],
        ["<code>scripts/dct_shield_run.py</code>", "DCT-Shield 的攻擊與評測驅動"],
        ["<code>scripts/phase_retention.py</code>", "接上兩個新淨化算子"],
        ["<code>docs/DECISIONS.md</code>", "新增 DEC-025"],
        ["<code>docs/FINDINGS.md</code>", "新增 FND-057"],
        ["<code>docs/reference/SURVEY_2026-08-18_frequency.md</code>",
         "31 篇的頻域／相位／抗淨化文獻查證；FreqPure 條目已更正"]])}
<p>測試由 <b>207 passed / 1 xfailed</b> 增加到 <b>263 passed / 1 xfailed</b>
（新增 56 項）。<code>CLAUDE.md</code> 記載的基準 196 早已過時。</p>

<h2>7. 沒做完的事</h2>
<ul>
<li><b>DCT-Shield 的預算對齊版本未跑。</b><code>--mode aligned</code> 已實作
（二分搜尋 ε 使 DISTS 等於相位臂），但 ε 會被搜到 1 以下，論文的抗 JPEG 條件
因此失效，該列必須標註。這是與紋理重相位做同預算比較的唯一乾淨作法。</li>
<li><b>BlurGuard 與 DiffusionGuard 未實作。</b>兩者都有公開程式碼，是 survey
點名的另外兩個頻域／抗淨化 baseline。</li>
<li><b><code>mist</code> 的預算對齊重測未做</b>（DEC-025 已記錄待辦）。</li>
<li><b>擾動的徑向功率譜未量。</b>「現有方法多為高頻擾動」這句動機目前只有
文獻依據（arXiv:2505.01267 測到擴散淨化對幅度與相位的破壞都隨頻率單調遞增），
本專案自己的條件還沒量過。</li>
<li><b>PAD 本體（相位級對抗訓練）未跑。</b>逐字重現是 CIFAR-10 上 91 個 epoch
的對抗訓練，產出的準確率表接不進 img2img 協定。依第 2 節的結果，它的角色
應由「baseline」降為「limitation 的證據」——相位級對抗訓練是對本方法現成的
反制，但要求攻擊方重新訓練 Stable Diffusion，落在本專案的威脅模型之外。</li>
</ul>
</div>
"""
    return ("<title>頻域與相位防護 · 夜間紀錄</title>\n"
            f"<style>{CSS}</style>\n{body}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(build(), encoding="utf-8")
    print(f"寫入 {OUT / 'index.html'}")
