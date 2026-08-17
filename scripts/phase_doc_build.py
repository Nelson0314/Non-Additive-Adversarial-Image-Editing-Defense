"""把圖與說明組成一份自足的 HTML 教學文件：紋理重相位的方法說明。

先跑 `phase_doc_fig1d.py` 與 `phase_doc_fig2d.py` 產圖，再跑本檔內嵌。
圖以 base64 內嵌，產出物只有一個 HTML、可單獨寄出。

    python scripts/phase_doc_fig1d.py
    python scripts/phase_doc_fig2d.py
    python scripts/phase_doc_build.py
"""
import base64
import io
from pathlib import Path

from PIL import Image

import os
FIGDIR = Path(os.environ.get(
    "PHASE_DOC_FIGDIR",
    Path(__file__).resolve().parent.parent / "runs" / "_phase_doc_fig"))
FIG = FIGDIR
OUT = Path(r"C:\WACV-s3\reports\2026-08-18-phase-method\index.html")
OUT.parent.mkdir(parents=True, exist_ok=True)


def img(name, cap, width="100%"):
    p = FIG / name
    im = Image.open(p).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (f'<figure><img style="width:{width}" '
            f'src="data:image/jpeg;base64,{b64}" alt="{cap}">'
            f'<figcaption>{cap}</figcaption></figure>')


CSS = """
:root{--bg:#fbfaf8;--fg:#1d1c1a;--mut:#6b665e;--line:#e2ded6;--card:#fff;
--code:#f4f2ee;--accent:#c2410c;--accent2:#2563a8}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#171614;--fg:#e8e4dc;--mut:#9d968a;--line:#33302b;--card:#1f1e1b;
--code:#232019;--accent:#e07a4a;--accent2:#6fa8dc}}
:root[data-theme=dark]{--bg:#171614;--fg:#e8e4dc;--mut:#9d968a;--line:#33302b;
--card:#1f1e1b;--code:#232019;--accent:#e07a4a;--accent2:#6fa8dc}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);line-height:1.85;font-size:16px;
font-family:"Noto Serif TC","Songti TC","PMingLiU",Georgia,serif}
main{max-width:1080px;margin:0 auto;padding:40px 26px 120px}
h1{font-size:30px;line-height:1.35;margin:0 0 6px;font-family:"Noto Sans TC",system-ui,sans-serif}
h2{font-size:23px;margin:56px 0 14px;padding-top:20px;border-top:3px solid var(--accent);
font-family:"Noto Sans TC",system-ui,sans-serif}
h3{font-size:18px;margin:34px 0 10px;font-family:"Noto Sans TC",system-ui,sans-serif}
h4{font-size:16px;margin:24px 0 8px;color:var(--mut);
font-family:"Noto Sans TC",system-ui,sans-serif}
p{margin:0 0 14px}
.lede{color:var(--mut);font-size:15px;margin:0 0 30px}
code,kbd{background:var(--code);padding:1px 5px;border-radius:3px;font-size:0.88em;
font-family:"Cascadia Mono",Consolas,monospace}
pre{background:var(--code);padding:14px 16px;border-radius:6px;overflow-x:auto;
font-size:13.5px;line-height:1.6;font-family:"Cascadia Mono",Consolas,monospace}
figure{margin:22px 0 26px;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:14px}
figure img{display:block;max-width:100%;height:auto;border-radius:4px;margin:0 auto}
figcaption{font-size:13.5px;color:var(--mut);margin-top:10px;line-height:1.65;
font-family:"Noto Sans TC",system-ui,sans-serif}
.tw{overflow-x:auto;margin:0 0 20px}
table{border-collapse:collapse;font-size:14.5px;width:100%;
font-family:"Noto Sans TC",system-ui,sans-serif}
th,td{border:1px solid var(--line);padding:7px 11px;text-align:left;vertical-align:top}
th{background:var(--code);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
blockquote{margin:18px 0;padding:12px 18px;border-left:4px solid var(--accent2);
background:var(--card);color:var(--mut);font-size:15px}
.note{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
border-radius:0 6px 6px 0;padding:14px 18px;margin:20px 0;font-size:15px}
.note b{color:var(--accent)}
details{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:10px 16px;margin:10px 0}
summary{cursor:pointer;font-weight:600;font-family:"Noto Sans TC",system-ui,sans-serif}
.toc{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px 22px}
.toc ol{margin:0;padding-left:22px}
.toc a{color:var(--fg)}
svg.dia{display:block;width:100%;height:auto;margin:0 auto}
.eq{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:14px 18px;margin:18px 0;overflow-x:auto;text-align:center;
font-family:"Cascadia Mono",Consolas,monospace;font-size:14.5px}
"""

# ---------------------------------------------------------------- 流程方塊圖
DIA_OP = """
<svg class="dia" viewBox="0 0 1000 320" role="img"
     aria-label="紋理重相位算子的資料流">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
            markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
    <style>
      .bx{fill:var(--card);stroke:var(--line);stroke-width:1.6}
      .bxa{fill:var(--card);stroke:var(--accent);stroke-width:2.2}
      .tx{font:13px "Noto Sans TC",system-ui,sans-serif;fill:var(--fg)}
      .sm{font:11px "Cascadia Mono",Consolas,monospace;fill:var(--mut)}
      .ln{stroke:var(--mut);stroke-width:1.6;fill:none;color:var(--mut)}
      .lnh{stroke:var(--accent);stroke-width:2;fill:none;color:var(--accent)}
    </style>
  </defs>

  <rect class="bx" x="8" y="118" width="110" height="58" rx="6"/>
  <text class="tx" x="63" y="142" text-anchor="middle">原圖 x</text>
  <text class="sm" x="63" y="160" text-anchor="middle">(1,3,512,512)</text>

  <rect class="bx" x="150" y="118" width="118" height="58" rx="6"/>
  <text class="tx" x="209" y="142" text-anchor="middle">切塊 + 加窗</text>
  <text class="sm" x="209" y="160" text-anchor="middle">32x32, hop 16</text>

  <rect class="bx" x="300" y="118" width="100" height="58" rx="6"/>
  <text class="tx" x="350" y="142" text-anchor="middle">rfft2</text>
  <text class="sm" x="350" y="160" text-anchor="middle">32 x 17</text>

  <rect class="bxa" x="432" y="118" width="150" height="58" rx="6"/>
  <text class="tx" x="507" y="140" text-anchor="middle">乘 e^(i·θ·g·m)</text>
  <text class="sm" x="507" y="159" text-anchor="middle">|·| 逐位不變</text>

  <rect class="bx" x="614" y="118" width="100" height="58" rx="6"/>
  <text class="tx" x="664" y="142" text-anchor="middle">irfft2</text>

  <rect class="bx" x="746" y="118" width="120" height="58" rx="6"/>
  <text class="tx" x="806" y="140" text-anchor="middle">加窗 + 疊加</text>
  <text class="sm" x="806" y="159" text-anchor="middle">OLA</text>

  <rect class="bx" x="898" y="118" width="94" height="58" rx="6"/>
  <text class="tx" x="945" y="140" text-anchor="middle">÷ Σw²</text>
  <text class="sm" x="945" y="159" text-anchor="middle">x_def</text>

  <path class="ln" d="M118 147 H150" marker-end="url(#ar)"/>
  <path class="ln" d="M268 147 H300" marker-end="url(#ar)"/>
  <path class="ln" d="M400 147 H432" marker-end="url(#ar)"/>
  <path class="ln" d="M582 147 H614" marker-end="url(#ar)"/>
  <path class="ln" d="M714 147 H746" marker-end="url(#ar)"/>
  <path class="ln" d="M866 147 H898" marker-end="url(#ar)"/>

  <rect class="bx" x="380" y="18" width="254" height="56" rx="6"/>
  <text class="tx" x="507" y="40" text-anchor="middle">θ　可學參數（唯一）</text>
  <text class="sm" x="507" y="58" text-anchor="middle">(1, 1089, 32, 17)　≈ 5.9e5</text>
  <path class="lnh" d="M507 74 V118" marker-end="url(#ar)"/>

  <rect class="bx" x="150" y="240" width="200" height="58" rx="6"/>
  <text class="tx" x="250" y="262" text-anchor="middle">紋理閘 g_b（逐區塊）</text>
  <text class="sm" x="250" y="280" text-anchor="middle">結構張量 coherence</text>

  <rect class="bx" x="386" y="240" width="200" height="58" rx="6"/>
  <text class="tx" x="486" y="262" text-anchor="middle">頻率閘 m_ω（逐頻格）</text>
  <text class="sm" x="486" y="280" text-anchor="middle">r &lt; 0.12 與兩端行取 0</text>

  <path class="ln" d="M63 176 V269 H150" marker-end="url(#ar)"/>
  <path class="ln" d="M350 269 H386" marker-end="url(#ar)"/>
  <path class="ln" d="M586 269 H640 V176" marker-end="url(#ar)"/>
  <text class="sm" x="70" y="228">由原圖算一次，固定不可學</text>
</svg>
"""

DIA_PGD = """
<svg class="dia" viewBox="0 0 1000 260" role="img" aria-label="外圈的 PGD 最佳化">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
            markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
    <style>
      .bx2{fill:var(--card);stroke:var(--line);stroke-width:1.6}
      .bxa2{fill:var(--card);stroke:var(--accent);stroke-width:2.2}
      .tx2{font:13px "Noto Sans TC",system-ui,sans-serif;fill:var(--fg)}
      .sm2{font:11px "Cascadia Mono",Consolas,monospace;fill:var(--mut)}
      .ln2{stroke:var(--mut);stroke-width:1.6;fill:none;color:var(--mut)}
      .lnb{stroke:var(--accent2);stroke-width:2;fill:none;color:var(--accent2)}
    </style>
  </defs>
  <rect class="bxa2" x="20" y="90" width="120" height="56" rx="6"/>
  <text class="tx2" x="80" y="114" text-anchor="middle">θ</text>
  <text class="sm2" x="80" y="132" text-anchor="middle">clamp(±θ_max)</text>

  <rect class="bx2" x="184" y="90" width="170" height="56" rx="6"/>
  <text class="tx2" x="269" y="114" text-anchor="middle">紋理重相位算子</text>
  <text class="sm2" x="269" y="132" text-anchor="middle">上圖那一條</text>

  <rect class="bx2" x="398" y="90" width="150" height="56" rx="6"/>
  <text class="tx2" x="473" y="114" text-anchor="middle">VAE encoder E</text>
  <text class="sm2" x="473" y="132" text-anchor="middle">不動 UNet</text>

  <rect class="bx2" x="592" y="90" width="196" height="56" rx="6"/>
  <text class="tx2" x="690" y="112" text-anchor="middle">L = ‖E(x_def) − E(y)‖²</text>
  <text class="sm2" x="690" y="130" text-anchor="middle">y = 灰圖 target</text>

  <rect class="bx2" x="832" y="90" width="148" height="56" rx="6"/>
  <text class="tx2" x="906" y="112" text-anchor="middle">θ ← θ − α·sign(∇)</text>
  <text class="sm2" x="906" y="130" text-anchor="middle">200 步</text>

  <path class="ln2" d="M140 118 H184" marker-end="url(#ar2)"/>
  <path class="ln2" d="M354 118 H398" marker-end="url(#ar2)"/>
  <path class="ln2" d="M548 118 H592" marker-end="url(#ar2)"/>
  <path class="ln2" d="M788 118 H832" marker-end="url(#ar2)"/>
  <path class="lnb" d="M906 146 V206 H80 V146" marker-end="url(#ar2)"/>
  <text class="sm2" x="470" y="226" text-anchor="middle">反向傳播經過整條算子（全部可微）</text>
  <text class="sm2" x="26" y="66">預算：θ_max = 1.30（人眼門檻）</text>
</svg>
"""

BODY = f"""
<h1>紋理重相位（Texture Rephasing）</h1>
<p class="lede">白盒非加性抗文字編輯防禦 · 方法說明 · 2026-08-18<br>
本文假設讀者熟悉「以擾動保護影像不被擴散模型編輯」這個框架，不假設熟悉傅立葉相位。
所有圖與數字都是用本專案的程式當場跑出來的，不是示意用的假資料。</p>

<div class="toc">
<ol>
  <li><a href="#s1">來源：Random Phase Noise（Galerne 等人, TIP 2011）</a></li>
  <li><a href="#s2">為什麼不能把 RPN 直接拿來用</a></li>
  <li><a href="#s3">一維完整版：把整個方法縮到 32 個數字</a></li>
  <li><a href="#s4">從一維到影像：窗、網格、兩個閘</a></li>
  <li><a href="#s5">恆等保證與已知的不精確之處</a></li>
  <li><a href="#s6">Pipeline</a></li>
  <li><a href="#s7">真實資料上的樣子</a></li>
  <li><a href="#s8">與加性擾動的差別、限制、術語</a></li>
</ol>
</div>

<h2 id="s1">1　來源：Random Phase Noise（Galerne, Gousseau, Morel, IEEE TIP 2011）</h2>

<h3>1.1 原始論文要解決的問題</h3>

<p>該論文處理的是<b>紋理合成</b>，與對抗攻擊無關。問題設定是：給定一張紋理樣本，
產生任意大小、與樣本<b>知覺外觀相同</b>但不是逐像素複製的新影像。</p>

<p>作者把適用範圍界定在他們稱為 <b>micro-texture（微紋理）</b>的一類影像。原文的定義是
「拍攝遠處由細小物件構成的區域時形成的均勻影像區域；成像模糊使幾何特徵與顏色混在
一起，結果是統計上均勻的區域」，並主張<b>任何影像中的大多數均勻區域都應該是微紋理</b>。
草地、沙、布料、皮毛都屬於這一類。</p>

<h3>1.2 原生做法</h3>

<p>RPN 演算法本身只有四步：</p>

<pre>1. 產生白噪聲相位 θ(ξ₁,ξ₂) ~ U[−π, π]，並強制其滿足共軛對稱
2. 對輸入影像做 DFT，得到 ĥ(ξ₁,ξ₂)
3. ĥ(ξ₁,ξ₂) ← ĥ(ξ₁,ξ₂) · e^{{iθ(ξ₁,ξ₂)}}
4. 做逆 DFT，輸出</pre>

<p>三個實作細節值得記下，因為本專案沿用了其中兩個：</p>

<div class="tw"><table>
<tr><th>細節</th><th>原文做法</th><th>本專案</th></tr>
<tr><td>共軛對稱</td><td>強制 θ 對稱，保證輸出是實數</td><td>改用 <code>rfft2</code>／<code>irfft2</code>，實數性由型別保證；但半平面上
<code>fx=0</code> 與 <code>fx=N/2</code> 兩行需另外處理（§4.3）</td></tr>
<tr><td>彩色</td><td><b>三通道加同一組隨機相位</b>，否則會產生假色</td><td>沿用。θ 在 RGB 三通道共用</td></tr>
<tr><td>非週期性假影</td><td>先用 Moisan 分解取出<b>週期分量</b>（解 Δp = Δᵢh），
避免邊界不連續造成的水平／垂直條紋</td><td>不需要。本方法切成重疊區塊並加 Hann 窗，
窗本身就把區塊邊界壓到零</td></tr>
</table></div>

<h3>1.3 論文保證的性質</h3>

<blockquote>幅度譜完全保留，相位完全隨機化。對微紋理而言，這個操作不改變知覺外觀。</blockquote>

<p>換句話說：<b>紋理由幅度譜刻畫，相位只決定紋素擺在哪裡。</b>這一句是整個方法的地基。</p>

{img("f5_global_rpn.png", "圖 1　用本專案的程式重現 Galerne 等人的 RPN（整張圖一次 DFT、三通道共用隨機相位）。上排是一塊草地（微紋理），下排是浣熊的臉（有結構）。三張的幅度譜逐位相同，只有相位不同。上排的三張看起來都像草地，統計外觀確實保住了；但葉子與腳掌不見了。下排更明顯：毛的質感還在，臉沒了。")}

<h2 id="s2">2　為什麼不能把 RPN 直接拿來用</h2>

<p>圖 1 已經回答了一半。另一半有更早的文獻依據：Oppenheim &amp; Lim（Proc. IEEE, 1981）
的經典結果是<b>相位比幅度更決定影像的可辨識內容</b>——把 A 圖的幅度配上 B 圖的相位，
看到的是 B。</p>

<div class="note">
<b>本方法必須正面處理的矛盾。</b>Galerne 說「相位隨機化不改變外觀」，Oppenheim 說
「相位決定內容」。兩者都對，差別在<b>作用的尺度與位置</b>：相位的低頻分量攜帶結構與
位置，高頻分量攜帶紋素的擺放。RPN 對微紋理成立，是因為微紋理裡「結構」本來就沒有
內容可言。
</div>

<p>本方法的三個修改，逐條對應這個矛盾：</p>

<div class="tw"><table>
<tr><th>修改</th><th>要擋掉什麼</th></tr>
<tr><td><b>切成 32×32 重疊區塊</b>，不是整張圖一次</td><td>把相位的作用範圍限制在區塊內。
整張圖的低頻（＝物件在畫面中的位置）不進入參數</td></tr>
<tr><td><b>徑向頻率閘</b> m<sub>ω</sub>：區塊內歸一化半徑 &lt; 0.12 的頻格凍結</td>
<td>區塊內的低頻仍帶著該區塊的結構，動它會在重疊相加後留下接縫</td></tr>
<tr><td><b>紋理閘</b> g<sub>b</sub>：邊緣與平坦區凍結</td>
<td>邊緣動相位會產生鬼影；平坦區任何改動都直接可見。只留下 Galerne 意義下的微紋理區</td></tr>
</table></div>

<p>第四個修改與矛盾無關，而是與目的有關：<b>把「隨機」換成「最佳化」。</b>
θ 不再抽樣自 U[−π,π]，而是 PGD 的可學參數。隨機那一版仍然保留，作為同失真的
對照組 <code>phase_rand</code>——它就是 RPN 本身。</p>

<h2 id="s3">3　一維完整版：把整個方法縮到 32 個數字</h2>

<p>本節用 N=32 的一維訊號把每一步跑一遍。二維只是同樣的事做兩次。</p>

<h3>3.1 訊號、幅度、相位</h3>

<p>取一個由三個正弦組成的訊號：一個低頻（k=1）代表「結構」，兩個高頻（k=6, 11）
代表「紋理」。</p>

<div class="eq">x[n] = 1.00·sin(2π·1·n/N) + 0.45·sin(2π·6·n/N + 0.7) + 0.30·cos(2π·11·n/N − 1.2)</div>

{img("f1_signal_spectrum.png", "圖 2　訊號與它的離散傅立葉轉換。|X[k]| 只在 k=1,6,11 有值——那是三個成分的強度。∠X[k] 是三個成分各自的相位，也就是它們被擺在哪裡。")}

<h3>3.2 只轉相位：波形變了，幅度譜逐位不變</h3>

<p>對 k ≥ 4 的係數乘上 e<sup>iθ</sup>（模擬頻率閘：低頻凍結），θ 取四個值。</p>

{img("f2_rotate_keeps_magnitude.png", "圖 3　左：θ 越大波形變化越大，但整體包絡（來自被凍結的 k=1）維持不動。右：四條波形的 |X[k]| 疊在一起——完全相同。這不是近似，是代數恆等式 |X·e^{iθ}| = |X|。實測最大差 5.6e-16，即機器精度。")}

<p>兩個由構造保證的性質在這張圖上都看得到：</p>

<div class="tw"><table>
<tr><th>性質</th><th>實測</th></tr>
<tr><td>θ = 0 時輸出逐位等於輸入</td><td>max |y − x| = <b>6.7e-16</b></td></tr>
<tr><td>任何 θ 下幅度譜不變</td><td>max ‖|Y| − |X|‖<sub>∞</sub> = <b>5.6e-16</b></td></tr>
</table></div>

<h3>3.3 為什麼低頻的相位不能動</h3>

<p>相位與位置的關係是精確的：整條頻譜乘上<b>線性</b>相位 θ<sub>k</sub> = −2πk·s/N
等於把訊號整體平移 s 格。</p>

{img("f3_low_freq_is_position.png", "圖 4　左：對所有 k 施加線性相位，結果與 np.roll(x, 4) 逐點重合。相位不是抽象的東西，它就是位置。右：只轉 k=1（低頻）會把整個形狀搬走；只轉 k≥6（高頻）只改細節。徑向頻率閘擋掉的正是左邊那種效果。")}

<h3>3.4 加窗、重疊、以窗平方和正規化</h3>

<p>實際的算子不是對整條訊號做一次 FFT，而是切成重疊的區塊。一維版本：長度 64 的訊號、
區塊 16、hop 8（50% 重疊）、週期 Hann 窗。</p>

<div class="eq">y = Σ<sub>b</sub> w · IDFT( DFT(w · x<sub>b</sub>) · e<sup>iθ</sup> )　÷　Σ<sub>b</sub> w²</div>

<p>分析時乘一次窗、合成時再乘一次窗，最後除以窗平方和。這條式子不是本方法發明的：
它是 <b>Griffin &amp; Lim (1984)</b> 對「由被修改過的短時傅立葉轉換還原訊號」這個問題的
<b>最小平方最佳解</b>。</p>

{img("f4_windowed_ola.png", "圖 5　左：相鄰的 Hann 窗與它們的平方和 Σw²。中：θ=0 時重疊相加的重建誤差是 4.4e-16——恆等成立。右：θ=1.3 施加在 k≥3 上，細節被重排，大包絡維持。")}

<div class="note">
<b>恆等依賴的是 NOLA，不是 COLA。</b>COLA（相鄰窗和為常數）是完美重建的<b>充分</b>條件；
必要的只是更弱的 NOLA：Σw² &gt; 0 處處成立。因為分母就是 Σw²，它只要不為零就能除。
程式用 <code>clamp_min(1e-8)</code> 保證，並用一個<b>不滿足 COLA</b> 的 hop=8 寫成測試
（<code>test_identity_holds_for_non_cola_hop</code>）——換窗型或換 hop 都不會靜默破壞恆等。
</div>

<h3>3.5 一維版本已經看得到的不精確之處</h3>

<p>相鄰區塊有 50% 重疊。各自轉相位之後，兩塊在重疊區給出的值一般<b>不一致</b>；
最小平方解會取一個折衷。折衷改動了係數，所以「局部幅度譜保留」在整條訊號的層級是
<b>近似而非恆等</b>。</p>

<p>把重建結果重新分析一次、比對幅度譜，一維版在 θ=1.3 下的相對偏差是 <b>0.181</b>。
二維實測小得多（§5.2）。這個量在程式裡叫 <code>amplitude_deviation</code>，是必報的診斷值。</p>

<h2 id="s4">4　從一維到影像：窗、網格、兩個閘</h2>

<h3>4.1 區塊網格與二維窗</h3>

<p>512×512 的影像，block = 32、hop = 16、反射填補 16 格，得到 33×33 = <b>1089</b> 個
重疊區塊。二維 Hann 窗由一維外積構成。</p>

{img("f6_window.png", "圖 6　左：32×32 的二維週期 Hann 窗。中：一條切面與 hop=16 的兩個鄰居。右：重疊網格上的 Σw²，處處大於零，NOLA 有充裕餘裕。取週期版而非對稱版，就是為了讓這條曲線不出現接近零的格。")}

<h3>4.2 rfft2 的半平面</h3>

<p>實數輸入的頻譜有共軛對稱性，<code>rfft2</code> 因此只存半平面：32×32 的區塊得到
32×17 的複數格。每個區塊、每個頻格一個 θ，三通道共用，參數量
1089 × 32 × 17 ≈ <b>5.9×10⁵</b>——與加性 δ 的 3×512×512 ≈ 7.9×10⁵ 同數量級。</p>

<h3>4.3 徑向頻率閘 m<sub>ω</sub>：轉的是哪些頻率</h3>

<p>半徑以 Nyquist 為 1 歸一化。主線使用 <b>r<sub>min</sub> = 0.12</b>，
在 32×32 的區塊上換算成：</p>

<div class="tw"><table>
<tr><th class="n">r<sub>min</sub></th><th class="n">最低被轉的頻格 k</th>
<th class="n">cycles/pixel</th><th class="n">對應的紋理週期</th><th class="n">被轉的格數佔比</th></tr>
<tr><td class="n"><b>0.12（主線）</b></td><td class="n">k ≥ 2</td><td class="n">0.060</td>
<td class="n">16.7 px</td><td class="n">87.7%</td></tr>
<tr><td class="n">0.25</td><td class="n">k ≥ 4</td><td class="n">0.125</td>
<td class="n">8.0 px</td><td class="n">84.7%</td></tr>
<tr><td class="n">0.40</td><td class="n">k ≥ 7</td><td class="n">0.200</td>
<td class="n">5.0 px</td><td class="n">77.6%</td></tr>
</table></div>

<p>換句話說，主線凍結的是<b>週期長於約 17 像素</b>的成分（含 DC 與第一圈），
其餘全部可轉。</p>

{img("f7_radial_gate.png", "圖 7　左：半平面上每一格的歸一化半徑。中：主線 r_min = 0.12 的閘，黑色是凍結的頻格。右：兩端行必須凍結的理由。")}

<h4>逐步：m<sub>ω</sub> 的每一格怎麼被定成 1 或 0</h4>

<p>徑向閘是<b>二值</b>的，只有 0 與 1，而且完全由 <code>block</code> 與
<code>r_min</code> 決定——與影像內容無關，整批共用同一張表。程式是
<code>radial_gate()</code> 五行：</p>

<pre>fy = torch.fft.fftfreq(block) * 2.0    # (32,)   [-1, 1)
fx = torch.fft.rfftfreq(block) * 2.0   # (17,)   [0, 1]
r  = torch.sqrt(fy[:, None]**2 + fx[None, :]**2)   # (32, 17)
m  = (r &gt;= r_min).to(dtype)
m[:, 0] = 0.0
m[:, -1] = 0.0</pre>

<p>逐步展開（block = 32、r<sub>min</sub> = 0.12）：</p>

<ol>
<li><b>算兩個頻率軸。</b><code>fftfreq(32)</code> 給的是 cycles/sample，範圍
[−0.5, 0.5)；乘 2 之後變成「以 Nyquist 為 1」的座標。實際數值：
<code>fy = [0, 0.0625, 0.125, …, 0.9375, −1, −0.9375, …, −0.0625]</code>，
<code>fx = [0, 0.0625, …, 1.0]</code>（17 個，只有非負，因為
<code>rfft2</code> 只存半平面）。</li>

<li><b>算每一格到原點的距離。</b>r 是 32×17 的矩陣。左上角 5×5 是：
<pre>0.0000  0.0625  0.1250  0.1875  0.2500
0.0625  0.0884  0.1398  0.1976  0.2577
0.1250  0.1398  0.1768  0.2253  0.2795
0.1875  0.1976  0.2253  0.2652  0.3125
0.2500  0.2577  0.2795  0.3125  0.3536</pre></li>

<li><b>硬門檻。</b><code>r &gt;= 0.12</code> 為 1，否則 0。這一步只擋掉
<b>6 格</b>：(0,0)、(0,1)、(1,0)、(1,1)、(0,−1)、(−1,0) 那一小圈。
r 是連續值但 m 是 0/1，<b>沒有過渡帶</b>——這是硬切，不是漸變。</li>

<li><b>把兩端行整行清零。</b><code>m[:, 0] = 0</code>（fx = 0）與
<code>m[:, -1] = 0</code>（fx = N/2 = Nyquist）。這一步又擋掉 <b>61 格</b>，
比門檻本身擋得多得多。理由是共軛對稱（見下段）。</li>
</ol>

<p>最後：544 格裡凍結 67 格、可轉 <b>477 格（87.7%）</b>。</p>

<div class="note">
<b>值得記一下的數量關係。</b>在 r<sub>min</sub> = 0.12 這個設定下，
<b>頻率門檻只凍結 6 格，共軛對稱那兩行凍結 61 格</b>。也就是說「凍結低頻」在
主線設定下幾乎沒有實際作用，真正被擋掉的絕大多數是為了保住實數性與幅度而必須
放棄的那兩行。要靠頻率門檻擋住低頻，得把 r<sub>min</sub> 拉到 0.25 以上
（凍結 83 格）或 0.40（凍結 122 格）。
</div>

{img("f7d_radial_steps.png", "圖 8　徑向閘的三個步驟。左：每一格的半徑，是連續值。中：套 r ≥ 0.12，只有中心一小圈變黑。右：再把最左與最右兩整行清零——黑色面積主要來自這一步。")}

{img("f7b_rmin_compare.png", "圖 9　三個 r_min 的閘。三者差的格數不多（87.7% / 84.7% / 77.6%），但差的都是最低頻的那幾圈——那正是失真最貴的地方。FND-042 實測：r_min = 0.40 拿到主線 87% 的效果，只付 26% 的 DISTS 與 74% 的 LPIPS，且 PSNR 高 5.9 dB；效率排序在 DISTS、LPIPS、PSNR、SSIM 四個軸上完全一致。主線的 0.12 是照『可達的 DISTS 天花板最高』選的，不是效率上的操作點。")}

<p>兩端行（<code>fx = 0</code> 與 <code>fx = N/2</code>）的處理值得單獨講，因為
Galerne 不切塊、遇不到這個問題。這兩行的鏡像落在<b>被儲存的半平面之內</b>，
因此它們自身必須對 fy 共軛對稱。對它們逐格施加獨立相位會破壞這個關係——
輸出仍然是實數，<b>但那兩行的幅度不再保留</b>，而且不會有任何症狀。
代價是 17 行裡少掉 2 行；相對於一個只在兩行上失效的近似，這個代價是划算的。</p>

<h3>4.4 紋理閘 g<sub>b</sub></h3>

<p>逐區塊的純量，由<b>原圖</b>算一次後固定：</p>

<div class="eq">g<sub>b</sub> = ( 1 − coherence² ) · clip( E / E<sub>ref</sub>, 0, 1 )</div>

<p>coherence = (λ₁−λ₂)/(λ₁+λ₂) 取自結構張量的兩個特徵值（這個量與名稱見 Weickert, IJCV 1999）。
邊緣的梯度方向一致，coherence ≈ 1，第一個因子把它壓成 0。平坦區梯度能量低，
第二個因子把它壓成 0——那裡任何改動都直接可見，而 coherence 在該處是零除的雜訊。</p>

<p>E<sub>ref</sub> 取<b>該影像自己的</b>梯度能量中位數而非固定常數：梯度能量的絕對值
隨影像內容差好幾個數量級，用固定常數會讓不同影像拿到不同的有效閘，而那個差異不會
有症狀。</p>

{img("f8_texture_gate.png", "圖 10　raccoon_00 的兩個因子與合成閘。coherence 在毛的走向一致處與石頭邊緣偏高；梯度能量在背景虛化處偏低。相乘之後留下的（黃色）是草叢與皮毛的雜亂紋理區。這張圖的有效面積佔比 active_fraction = 0.469。")}

<h4>逐步：g<sub>b</sub> 怎麼從影像算出來</h4>

<div class="note">
<b>先更正一個容易誤會的地方：紋理閘不是 0 或 1，是 [0,1] 之間的連續值。</b>
只有徑向閘是二值的。紋理閘算出來的是每個區塊的一個純量權重，
raccoon_00 的 1089 個區塊裡平均值 0.469、最大 0.9999，有 29.7% 落在 0.05 以下。
它作用的方式是<b>把該區塊的旋轉角度按比例縮小</b>，不是「這塊轉／那塊不轉」。
</div>

<ol>
<li><b>轉成亮度。</b>三通道時取
<code>0.299R + 0.587G + 0.114B</code>。結構張量要的是人眼看得到的那個梯度。
（latent 是 4 通道且沒有亮度語意，那裡改取平均——套 RGB 權重會是憑空的假設。）</li>

<li><b>算梯度。</b>3×3 Sobel 除以 8，反射填補：
<code>gx = conv(lum, kx)</code>、<code>gy = conv(lum, ky)</code>。</li>

<li><b>逐區塊平均三個外積項。</b>
<code>J_xx = mean(gx²)</code>、<code>J_xy = mean(gx·gy)</code>、
<code>J_yy = mean(gy²)</code>，平均的範圍就是 32×32 的區塊，走的是
<code>block_mean()</code>——與遮罩閘同一個函式，兩者若用不同的填補或步幅就會落在
不同的網格上而且沒有症狀。得到 1089 個 2×2 的對稱矩陣
<code>J = [[J_xx, J_xy], [J_xy, J_yy]]</code>。</li>

<li><b>取兩個特徵值。</b>2×2 對稱矩陣有封閉解：
<div class="eq">λ<sub>1,2</sub> = ( tr ± √( (J_xx − J_yy)² + 4·J_xy² ) ) / 2，
　tr = J_xx + J_yy</div>
程式存的是 <code>disc = √( ((J_xx−J_yy)/2)² + J_xy² )</code>，
所以 λ₁ − λ₂ = 2·disc、λ₁ + λ₂ = tr。</li>

<li><b>算 coherence。</b>
<code>coh = (λ₁−λ₂)/(λ₁+λ₂) = 2·disc / (tr + 1e-8)</code>。
梯度方向一致（邊緣）時 λ₂ ≈ 0，coh → 1；方向雜亂（紋理）時 λ₁ ≈ λ₂，coh → 0。</li>

<li><b>算能量飽和項。</b><code>E_ref</code> 取<b>該影像自己的</b> tr 中位數
（<code>energy_quantile = 0.5</code>），然後
<code>sat = clip(tr / E_ref, 0, 1)</code>。所以能量達到中位數以上的區塊 sat = 1，
低於中位數的按比例遞減。raccoon_00 的 E_ref = 0.001943。</li>

<li><b>相乘。</b><code>g = (1 − coh²) · sat</code>。
兩個因子分工明確：第一個壓邊緣，第二個壓平坦區。</li>

<li><b>固定住。</b><code>self.tex_gate = tex.detach()</code>。
之後整個最佳化過程都不再重算。</li>
</ol>

<h4>三個實際區塊的算式</h4>

<div class="tw"><table>
<tr><th>區塊</th><th class="n">J_xx</th><th class="n">J_xy</th><th class="n">J_yy</th>
<th class="n">tr</th><th class="n">λ₁</th><th class="n">λ₂</th><th class="n">coh</th>
<th class="n">1−coh²</th><th class="n">sat</th><th class="n">g</th></tr>
<tr><td>#405 邊緣（鬍鬚）</td><td class="n">0.00113</td><td class="n">−0.00201</td>
<td class="n">0.02005</td><td class="n">0.02119</td><td class="n">0.02026</td>
<td class="n">0.00092</td><td class="n">0.913</td><td class="n">0.167</td>
<td class="n">1.000</td><td class="n"><b>0.167</b></td></tr>
<tr><td>#94 平坦（虛化背景）</td><td class="n">0.0000004</td><td class="n">+0.0000001</td>
<td class="n">0.0000006</td><td class="n">0.0000010</td><td class="n">0.0000006</td>
<td class="n">0.0000004</td><td class="n">0.114</td><td class="n">0.987</td>
<td class="n">0.0003</td><td class="n"><b>0.000</b></td></tr>
<tr><td>#1001 紋理（草）</td><td class="n">0.00388</td><td class="n">−0.00004</td>
<td class="n">0.00386</td><td class="n">0.00774</td><td class="n">0.00391</td>
<td class="n">0.00383</td><td class="n">0.010</td><td class="n">0.9999</td>
<td class="n">1.000</td><td class="n"><b>1.000</b></td></tr>
</table></div>

<p>三個區塊各走到 0 的路徑不同：邊緣那塊被<b>第一個因子</b>壓掉（coh 0.913），
平坦那塊被<b>第二個因子</b>壓掉（sat 0.0003），草那塊兩個因子都放行。</p>

{img("f8b_gate_examples.png", "圖 11　三個區塊在原圖上的位置與各自的 32×32 內容。左紅框是鬍鬚——梯度全部同一方向；藍框是虛化背景——幾乎沒有梯度；右下綠框是草——梯度方向雜亂且能量足夠。")}

<h3>4.5 兩個閘怎麼合起來，作用在什麼上</h3>

<p>合成閘是外積：</p>

<pre>def gate(self):
    return self.tex_gate[..., None, None] * self.freq_gate
    #      (1, L, 1, 1)                     (32, 17)
    #   -&gt; (1, L, 32, 17)

shift = torch.clamp(self.theta, -theta_max, theta_max)
shift = (shift * self.gate()).unsqueeze(1)
spec  = self.analyze(x01)
x_def = self.synthesize(rotate_spectrum(spec, shift))</pre>

<p>所以區塊 b、頻格 (u,v) 實際被轉的角度是</p>

<div class="eq">θ<sub>b,u,v</sub> · g<sub>b</sub> · m<sub>u,v</sub></div>

<p>三個要點：</p>

<ul>
<li><b>閘乘在角度上，不是乘在係數上。</b>g<sub>b</sub> = 0.3 的區塊仍然每一格都轉，
只是只轉 0.3θ。這一點很重要——若閘乘在係數上就會直接改幅度，幅度保留的性質立刻失效。</li>
<li><b>m = 0 的頻格轉 0 度，等於完全不動</b>，係數逐位保持原值。</li>
<li><b>θ 是唯一的可學參數。</b>兩個閘都在 <code>prepare_gates()</code> 裡算完就
<code>detach()</code>，梯度不會流回它們。</li>
</ul>

<div class="note">
<b>閘取自原圖而非當前的防禦圖。</b>閘若跟著防禦圖漂移，g<sub>b</sub> 會變成優化目標的
一部分——最佳化會把擾動搬到閘自己放寬的地方，那不是本方法要量的東西。
程式在第一次前向前強制呼叫 <code>prepare_gates()</code>，沒呼叫直接拋
<code>RuntimeError</code>，不給預設值。
</div>

<h3>4.6 為什麼轉相位不會改變亮度</h3>

<p>一個區塊的平均亮度就是它的 <b>DC 係數</b>（頻率 0 那一格）。相位旋轉之所以不動
亮度，理由不是「相位與亮度無關」——恰恰相反，<b>轉 DC 會直接把亮度乘上 cos θ</b>。</p>

<p>DC 是唯一不振盪的成分。對實數影像而言它必須是實數，所以乘上 e<sup>iθ</sup> 之後，
<code>irfft</code> 只留下實部，等於把它乘上 cos θ。實測（32×32 的區塊）：</p>

<div class="tw"><table>
<tr><th class="n">θ</th><th class="n">只轉 DC 之後的區塊平均</th><th class="n">相對原值</th><th class="n">cos θ</th></tr>
<tr><td class="n">0.65</td><td class="n">0.4038</td><td class="n">0.7961</td><td class="n">0.7961</td></tr>
<tr><td class="n">1.30</td><td class="n">0.1357</td><td class="n">0.2675</td><td class="n">0.2675</td></tr>
<tr><td class="n">π</td><td class="n">−0.5073</td><td class="n">−1.0000</td><td class="n">−1.0000</td></tr>
</table></div>

<p>θ = 1.30 會讓該塊只剩 26.75% 的亮度，θ = π 直接反相。<b>這正是 DC 必須凍結的理由。</b>
徑向閘凍結它兩次：它的半徑是 0，小於任何 r<sub>min</sub>；而且它落在
<code>fx = 0</code> 那一行，整行本來就取 0（§4.3）。</p>

<p>凍結 DC、只轉其餘頻格：區塊平均的變化是 <b>2.2e-16</b>，即機器精度。</p>

{img("f7c_brightness.png", "圖 12　左：轉 DC 時區塊平均隨 θ 走 cos 曲線（紅），與 cos θ 逐點重合；凍結 DC 時是一條平線（綠）。右：兩層凍結的說明。")}

<p>整張圖的層級上，亮度不是<b>逐位</b>保留而是<b>近似</b>保留，理由與 §5.2 的
一致性投影相同：重疊區塊各自轉相位後互相不一致，最小平方重建會做折衷。
實測 raccoon_00：</p>

<div class="tw"><table>
<tr><th class="n">θ</th><th class="n">全圖平均（原圖 0.41893014）</th><th class="n">差</th></tr>
<tr><td class="n">0.65</td><td class="n">0.41897276</td><td class="n">4.3e-5</td></tr>
<tr><td class="n">1.30</td><td class="n">0.41899833</td><td class="n">6.8e-5</td></tr>
<tr><td class="n">π</td><td class="n">0.41899475</td><td class="n">6.5e-5</td></tr>
</table></div>

<p>6.8e-5 相當於 8 位元灰階的 <b>0.017 級</b>，遠低於量化步階，肉眼與任何指標都測不到。</p>

<h2 id="s5">5　恆等保證與已知的不精確之處</h2>

<h3>5.1 θ = 0 逐位等於原圖</h3>

<p>在 512×512 的真實影像上實測：<code>max |x_def − x| = 4.8e-7</code>，
<code>amp_dev = 1.3e-7</code>。差異是 float32 的捨入，不是演算法的近似。</p>

<h3>5.2 STFT 一致性投影誤差</h3>

<p>逐區塊各自轉相位之後，那組係數<b>一般不是任何一張實影像的短時傅立葉轉換</b>。
Griffin &amp; Lim 稱這種輸入為 modified STFT，而 §3.4 的重建式正是把它投影回一致集合的
最小平方解。投影會改動係數，所以「局部幅度譜保留」在整張圖的層級是近似。</p>

<div class="tw"><table>
<tr><th class="n">θ</th><th class="n">PSNR (dB)</th><th class="n">L∞</th><th class="n">amp_dev</th></tr>
<tr><td class="n">0</td><td class="n">—（4.8e-7）</td><td class="n">—</td><td class="n">1.3e-7</td></tr>
<tr><td class="n">0.30</td><td class="n">40.15</td><td class="n">0.111</td><td class="n">0.0092</td></tr>
<tr><td class="n">0.65</td><td class="n">33.01</td><td class="n">0.230</td><td class="n">0.0226</td></tr>
<tr><td class="n"><b>1.30</b></td><td class="n"><b>25.95</b></td><td class="n"><b>0.516</b></td><td class="n"><b>0.0550</b></td></tr>
<tr><td class="n">2.20</td><td class="n">20.60</td><td class="n">1.103</td><td class="n">0.0738</td></tr>
<tr><td class="n">3.14</td><td class="n">18.10</td><td class="n">1.538</td><td class="n">0.0726</td></tr>
</table></div>

<p>偏差在 0.009–0.074 之間，遠低於一維示範的 0.18。若這個值變大，代表算子在<b>造新能量</b>
而不是重排相位，那會讓它退化成被紋理遮蔽的加性高頻噪聲——這是設計時列出的第一號風險，
所以它是必報的診斷值而不是可選的。</p>

<h4>可以把它壓下來</h4>

<p>Griffin-Lim 迭代投影：每輪重新分析、把幅度換回原圖的、再合成。實測（raccoon_00, θ=1.30）：</p>

<div class="tw"><table>
<tr><th class="n">gl_iters</th><th class="n">amp_dev</th><th class="n">PSNR (dB)</th></tr>
<tr><td class="n">0（預設）</td><td class="n">0.0550</td><td class="n">25.95</td></tr>
<tr><td class="n">1</td><td class="n">0.0297</td><td class="n">27.01</td></tr>
<tr><td class="n">4</td><td class="n">0.0184</td><td class="n">28.79</td></tr>
</table></div>

<p>偏差降低的同時失真也降低，兩個變因綁在一起——迭代投影不只壓掉不一致，它把整個擾動
一起縮小。這一點在結論上還沒有落地（見 §8 限制）。</p>

<h2 id="s6">6　Pipeline</h2>

<h3>6.1 算子內部</h3>

{DIA_OP}

<p>對應到程式（<code>src/residual/texture_rephase.py</code>）：</p>

<div class="tw"><table>
<tr><th>方塊</th><th>函式</th></tr>
<tr><td>切塊 + 加窗 + rfft2</td><td><code>analyze()</code></td></tr>
<tr><td>乘 e<sup>iθgm</sup></td><td><code>rotate_spectrum()</code>。寫成乘單位模複數而不是拆
<code>abs</code>／<code>angle</code> 再重組——後者在零幅度處梯度未定義</td></tr>
<tr><td>irfft2 + 加窗 + OLA + ÷Σw²</td><td><code>synthesize()</code>。整個模組只有這一條合成路徑，
主前向與 Griffin-Lim 迭代不可能分岔</td></tr>
<tr><td>兩個閘</td><td><code>prepare_gates()</code> → <code>texture_gate()</code>,
<code>radial_gate()</code></td></tr>
</table></div>

<h3>6.2 外圈：用什麼損失把 θ 學出來</h3>

{DIA_PGD}

<p>損失是 encoder-targeted：<b>‖E(x_def) − E(y_target)‖²</b>，y_target 是一張灰圖。
只跑 VAE 編碼器、不碰 UNet，因此一張圖 13 秒（對照：PhotoGuard-c 6429 秒）。
選它的理由有二：成本是分鐘量級；以及它與弱 baseline 的 targeted 形式同源，
所以三個像素臂條件的<b>唯一變因是參數化</b>。</p>

<div class="tw"><table>
<tr><th>條件</th><th>參數 φ</th><th>投影</th><th>角色</th></tr>
<tr><td><code>add</code></td><td>δ，逐像素加性</td><td>L∞ 球，ε = 1.2/255</td><td>同損失的加性對照</td></tr>
<tr><td><code>phase</code></td><td>θ，本方法</td><td>逐元素夾 |θ| ≤ 1.30</td><td>主體</td></tr>
<tr><td><code>phase_rand</code></td><td>同幅度的隨機 θ，<b>不最佳化</b></td><td>同上</td><td>同失真隨機對照，即 RPN 本身</td></tr>
</table></div>

<div class="note">
<code>phase_rand</code> 從第一天就存在，不是事後補的。此前有兩次（FND-004、FND-018）
是被「贏不過同失真隨機對照」擋下來的；事後補對照組等於重蹈覆轍。
</div>

<h3>6.3 θ 的大小換到什麼</h3>

{img("f9_theta_sweep.png", "圖 13　raccoon_00 上的 θ 掃描（此處 θ 是隨機初始化後全部拉到同一絕對值，只為展示幅度效果）。上排是輸出，下排是殘差放大 8 倍。殘差不是均勻的噪聲——它精確地落在草叢與皮毛上，那正是紋理閘留下的區域。")}

{img("f9b_crop.png", "圖 14　同一批的 96×96 放大。即使 θ = π，場景、物件位置、邊緣全部維持；改變的是草與毛的紋素排列。這與圖 1 下排的全域 RPN 形成對比——同樣是只轉相位，切塊加閘之後內容完全保住。")}

{img("f10_theta_curves.png", "圖 15　θ 與三個量的關係。PSNR 單調下降，L∞ 單調上升，amp_dev 在 θ≈2.2 之後反而回落——超過該點之後相位旋轉開始互相抵消。")}

<h2 id="s7">7　真實資料上的樣子</h2>

<p>以下取自 <code>runs/phaseA_human</code>：24 張影像、三個條件、同一個損失、同樣的步數與種子，
<b>唯一變因是參數化</b>。預算是使用者以人眼裁定的門檻：θ = 1.30、ε = 1.2/255。</p>

{img("f11_real_pipeline.png", "圖 16　cat_00 的完整一列。上排：原圖與三個條件的防禦圖，肉眼幾乎分不出。中排：殘差放大 10 倍——加性的幾乎看不見（它住在高頻），相位的則貼著紋理分布。下排：SDEdit 編輯的結果。未防禦的編輯把貓變成狗；加性防禦下編輯幾乎照常成功；紋理重相位下編輯被推得最遠。")}

{img("f12_residual_spectrum.png", "圖 17　同一張圖的殘差徑向功率譜。加性擾動的能量集中在高頻末端，相位擾動的分布明顯較寬。這是「加性把能量放在 VAE 敏感、人眼不敏感的高頻」這個既有觀察（FND-026）的直接呈現。")}

<h3>24 張的匯總</h3>

<div class="tw"><table>
<tr><th>條件</th><th class="n">半徑</th><th class="n">LPIPS↓</th><th class="n">DISTS↓</th><th class="n">PSNR↑</th>
<th class="n">編輯位移量↑</th><th class="n">相對加性</th></tr>
<tr><td><b>紋理重相位</b></td><td class="n">θ = 1.30</td><td class="n">0.2477</td><td class="n">0.0459</td>
<td class="n">30.69</td><td class="n"><b>0.5026</b></td><td class="n"><b>1.55×</b></td></tr>
<tr><td>加性 δ</td><td class="n">ε = 1.2/255</td><td class="n">0.1389</td><td class="n">0.0055</td>
<td class="n">47.76</td><td class="n">0.3251</td><td class="n">1.00×</td></tr>
<tr><td>隨機相位（RPN）</td><td class="n">θ = 1.30</td><td class="n">0.0828</td><td class="n">0.0523</td>
<td class="n">31.50</td><td class="n">0.1889</td><td class="n">0.58×</td></tr>
</table></div>

<p>逐圖比較：勝加性 <b>22/24</b>、勝同失真隨機相位 <b>24/24</b>。</p>

<div class="note">
<b>這張表要小心讀。</b>三者在<b>人眼裁定的門檻</b>上對齊，不是在任何單一數值指標上對齊——
LPIPS 說相位比加性差 1.8 倍，DISTS 說差 8.3 倍，PSNR 說差 17 dB，而人眼在這組半徑上
判定兩者可見失真相當。數值指標與人眼在非加性擾動上系統性不一致（FND-034），
本專案的判準是人眼，比對頁是主要產出物。
</div>

<h2 id="s8">8　與加性擾動的差別、限制、術語</h2>

<h3>8.1 一句話的差別</h3>

<div class="tw"><table>
<tr><th></th><th>加性 δ</th><th>紋理重相位</th></tr>
<tr><td>作用方式</td><td>x + δ，δ 與 x 無關</td><td>把 x 自己的頻譜係數轉相位，係數大小由 x 決定</td></tr>
<tr><td>θ / ε = 0</td><td>恆等</td><td>恆等（由重建式保證，非近似）</td></tr>
<tr><td>能量落在哪</td><td>高頻（圖 17）</td><td>紋理區，跟著影像內容走（圖 16 中排）</td></tr>
<tr><td>幅度譜</td><td>改變</td><td>區塊層級逐位保留，整張圖層級偏差 0.009–0.074</td></tr>
<tr><td>參數量（512²）</td><td>7.9×10⁵</td><td>5.9×10⁵</td></tr>
</table></div>

<h3>8.2 已知限制（照實列）</h3>

<ol>
<li><b>固定 θ 不等於固定失真。</b>同一個 θ = 1.30 在 24 張影像上的 PSNR 標準差是 4.85 dB、
全距 16.4 dB；加性的 ε 對應標準差只有 0.38 dB。而且該漂移<b>預測勝負</b>（r = +0.776），
所以逐圖比值不是乾淨的比較（FND-038）。文獻上有解法：<code>arXiv:2602.06577</code> 的
幅度相依上限 2·arcsin(ε/(2·|X|))。</li>
<li><b>整張圖層級的幅度保留是近似。</b>偏差與效果正相關 r = +0.449（FND-040）；後續分析
顯示 amp_dev 與 DISTS 共線，該相關<b>不是</b>「投影誤差造出能量」的獨立證據（FND-051），
但迭代投影臂尚未跑完，這一點還沒有定論。</li>
<li><b>語意抵抗不成立。</b>CLIP-T 對齊掉幅在雜訊範圍內，四個軸全部否證（FND-024/029/030），
且已有獨立文獻測到同一現象。本方法報的是位移量，不是語意破壞。</li>
<li><b>新穎性主張已收窄。</b>相位擾動用於對抗攻擊有前例（<code>arXiv:2602.06577</code>、
JEI 34(1):013041）。可宣稱的是首次把<b>加窗重疊區塊的頻譜相位旋轉</b>用於<b>擴散編輯防護</b>，
並以兩個由原圖決定的閘限制作用範圍。</li>
</ol>

<h3>8.3 術語對照</h3>

<div class="tw"><table>
<tr><th>中文</th><th>English</th><th>在本文的意思</th></tr>
<tr><td>幅度譜</td><td>Fourier modulus / magnitude spectrum</td><td>|X[k]|，各頻率成分的強度</td></tr>
<tr><td>相位</td><td>phase</td><td>∠X[k]，各頻率成分的位移</td></tr>
<tr><td>微紋理</td><td>micro-texture</td><td>統計上均勻、無個別可辨物件的影像區域</td></tr>
<tr><td>重疊相加</td><td>overlap-add (OLA)</td><td>把加窗處理過的區塊疊回整張圖</td></tr>
<tr><td>短時傅立葉轉換</td><td>short-time Fourier transform (STFT)</td><td>切塊、加窗、逐塊 FFT 的表示</td></tr>
<tr><td>一致性投影</td><td>consistency projection</td><td>把不是任何實訊號 STFT 的係數投影回可實現集合</td></tr>
<tr><td>結構張量</td><td>structure tensor</td><td>梯度外積的區塊平均，用來分辨邊緣與紋理</td></tr>
<tr><td>同調性</td><td>coherence</td><td>(λ₁−λ₂)/(λ₁+λ₂)，梯度方向一致的程度</td></tr>
</table></div>

<h3>8.4 自測</h3>

<details><summary>一、為什麼 θ = 0 時輸出會逐位等於原圖？這是靠什麼保證的？</summary>
<p>分析時乘一次窗、合成時再乘一次窗，於是分子是 Σ<sub>b</sub> w²·x，分母是 Σ<sub>b</sub> w²，
兩者逐位相消。保證來自 Griffin &amp; Lim (1984) 的最小平方重建式，
必要條件是 NOLA（Σw² &gt; 0 處處成立），不是 COLA。這一點由一個不滿足 COLA 的
hop 寫成測試釘住。</p></details>

<details><summary>二、Galerne 說相位隨機化不改變外觀，Oppenheim 說相位決定內容。本方法怎麼同時容納這兩句？</summary>
<p>兩句都對，差別在尺度。相位的低頻分量攜帶結構與位置，高頻分量攜帶紋素擺放。
本方法用三個手段把作用範圍限制在後者：切成 32×32 區塊（整張圖的低頻不入參）、
徑向頻率閘（區塊內的低頻凍結）、紋理閘（邊緣與平坦區凍結）。圖 1 與圖 14 的對比
就是這個限制的效果。</p></details>

<details><summary>三、為什麼 <code>fx = 0</code> 與 <code>fx = N/2</code> 兩行要凍結？不凍結會怎樣？</summary>
<p><code>rfft2</code> 只存半平面。這兩行的鏡像落在被儲存的半平面之內，因此它們自身必須
對 fy 共軛對稱。逐格施加獨立相位會破壞這個關係——輸出仍是實數，但那兩行的幅度不再
保留，而且沒有任何症狀。代價是 17 行少掉 2 行。</p></details>

<details><summary>四、相位旋轉為什麼不會改變整體亮度？</summary>
<p>亮度就是 DC 係數。轉 DC 會把該塊的平均乘上 cos θ——θ=1.30 剩 26.75%，θ=π 反相。
徑向閘凍結 DC 兩次（半徑 0 &lt; r<sub>min</sub>，且它在 <code>fx=0</code> 那一整行）。
凍結之後區塊平均的變化是 2.2e-16。整張圖的層級因為一致性投影會有 6.8e-5 的漂移，
相當於 8 位元灰階的 0.017 級。</p></details>

<details><summary>五、<code>amplitude_deviation</code> 量的是什麼？為什麼它必須報？</summary>
<p>量的是「重建後的影像再分析一次，局部幅度譜相對於原圖的偏差」。逐區塊各自轉相位後
的係數一般不是任何實影像的 STFT，最小平方重建會做折衷，折衷改動了係數。
若這個值變大，代表算子在造新能量而不是重排相位，那會使它退化成被紋理遮蔽的加性高頻
噪聲——也就是整個方法的主張失效。實測 0.009–0.074。</p></details>

<h3>參考文獻</h3>

<div class="tw"><table>
<tr><th>文獻</th><th>在本方法的角色</th></tr>
<tr><td>Galerne, Gousseau, Morel. <i>Random Phase Textures: Theory and Synthesis.</i>
IEEE TIP 20(1):257–267, 2011</td><td>構造來源。相位隨機化保留微紋理外觀；三通道共用相位</td></tr>
<tr><td>Oppenheim, Lim. <i>The Importance of Phase in Signals.</i>
Proc. IEEE 69(5):529–541, 1981</td><td>必須正面處理的矛盾。兩個閘就是答案</td></tr>
<tr><td>Griffin, Lim. <i>Signal Estimation from Modified STFT.</i>
IEEE TASSP 32(2):236–243, 1984</td><td>重建式 OLA(w²x)/OLA(w²) 的出處；一致性投影與 amp_dev</td></tr>
<tr><td>Allen, Rabiner. <i>A Unified Approach to Short-Time Fourier Analysis and Synthesis.</i>
Proc. IEEE 65(11):1558–1564, 1977</td><td>STFT 的分析／合成框架</td></tr>
<tr><td>Weickert. <i>Coherence-Enhancing Diffusion Filtering.</i>
IJCV 31:111–127, 1999</td><td>紋理閘用的 coherence 的來源</td></tr>
<tr><td>Ding, Ma, Wang, Simoncelli. <i>Unifying Structure and Texture Similarity.</i>
TPAMI 2021</td><td>DISTS。「對紋理重取樣寬容」這半個機制假設的依據</td></tr>
<tr><td>Madry et al. <i>Towards Deep Learning Models Resistant to Adversarial Attacks.</i>
ICLR 2018</td><td>PGD 本身</td></tr>
</table></div>
"""

html = ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>紋理重相位方法說明</title><style>{CSS}</style></head>'
        f'<body><main>{BODY}</main></body></html>')
OUT.write_text(html, encoding="utf-8")
print(f"{OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
