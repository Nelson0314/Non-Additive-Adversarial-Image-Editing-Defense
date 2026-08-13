"""hb5 批次的完整對比報告。每一格都必須有影像可看（CLAUDE.md 的判準）。"""
import base64, csv, glob, io, math, statistics as st
from collections import defaultdict
from PIL import Image

OUT = (r"C:\Users\nelso\AppData\Local\Temp\claude\C--WACV-s3"
       r"\53c06f45-714a-4e23-83dd-29764bd8e3a5\scratchpad\hb5_report.html")
IMAGES = ["man_02", "woman_02", "dog_03", "horse_03", "cat_01"]
CONDS = ["phase", "add", "phase_rand", "apa_weak", "mist", "dia_r", "photoguard_c"]
LABEL = {"phase": "site F 紋理重相位", "add": "加性 δ", "phase_rand": "隨機相位 (RPN)",
         "apa_weak": "弱 baseline (APA)", "mist": "Mist", "dia_r": "DIA-R",
         "photoguard_c": "PhotoGuard-c"}
BUDGET = {"phase": "θ = 1.30（人眼門檻）", "add": "ε∞ = 1.2/255（人眼門檻）",
          "phase_rand": "θ = 1.30（同失真隨機）", "apa_weak": "ε_a = 0.4 latent（原生）",
          "mist": "ε∞ = 16/255（原生）", "dia_r": "ε∞ = 0.025（原生）",
          "photoguard_c": "‖δ‖₂ = 16（原生 L2）"}
GROUP = {"phase": "site F", "add": "內部對照", "phase_rand": "內部對照",
         "apa_weak": "弱 baseline", "mist": "強 baseline", "dia_r": "強 baseline",
         "photoguard_c": "強 baseline"}
PURIF = ["identity", "blur1", "noise0.05", "quantize16", "jpeg75", "jpeg30",
         "crop_resize0.1", "jpeg_then_resize75", "adverse_cleaner", "impress"]
PNAME = {"identity": "未淨化", "blur1": "高斯模糊 σ=1", "noise0.05": "高斯雜訊 0.05",
         "quantize16": "量化 16 階", "jpeg75": "JPEG q75", "jpeg30": "JPEG q30",
         "crop_resize0.1": "裁切重縮 0.1", "jpeg_then_resize75": "C&R 串接",
         "adverse_cleaner": "AdverseCleaner", "impress": "IMPRESS"}
PROMPT = {"man_02": "an old woman", "woman_02": "a man", "dog_03": "a cat",
          "horse_03": "a zebra", "cat_01": "a dog"}


def path(img, cond, kind):
    if cond == "photoguard_c":
        return "runs/hb5_pgc/%s__photoguard_c__%s.png" % (img, kind)
    seg = cond + "__human" if cond in ("phase", "add", "phase_rand") else cond
    return "runs/hb5/%s__%s__%s.png" % (img, seg, kind)


def b64(p, size, q=82):
    im = Image.open(p).convert("RGB")
    if im.size[0] != size:
        im = im.resize((size, size), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, subsampling=1)
    return base64.b64encode(b.getvalue()).decode()


# ---------- 階段一 ----------
res = {}
for p in ("runs/phaseA_human/results.csv", "runs/hb5/g2/results.csv",
          "runs/hb5_pgc/results.csv"):
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["image"] in IMAGES:
            res[(r["image"], r["condition"])] = r
S = {}
for c in CONDS:
    v = [res[(i, c)] for i in IMAGES]
    S[c] = {k: st.fmean(float(x[f]) for x in v) for k, f in
            (("edit", "edit_lpips"), ("dists", "fid_dists"),
             ("lpips", "fid_lpips"), ("psnr", "fid_psnr"))}

# ---------- 階段二 ----------
rows = []
for p in sorted(glob.glob("runs/hb5/retention_*.csv")):
    rows += list(csv.DictReader(open(p, encoding="utf-8")))
usable = {(r["image"], r["condition"]): r["usable"] == "True" for r in rows}
common = [i for i in IMAGES if all(usable.get((i, c), False) for c in CONDS)]
eff = defaultdict(lambda: defaultdict(list))
ret = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["image"] in common:
        eff[r["purifier"]][r["condition"]].append(float(r["effect_mean"]))
        if r["retention"]:
            ret[r["purifier"]][r["condition"]].append(float(r["retention"]))
E = {k: {c: st.fmean(v) for c, v in d.items()} for k, d in eff.items()}
R = {k: {c: st.fmean(v) for c, v in d.items()} for k, d in ret.items()}
ident = {c: E["identity"][c] for c in CONDS}
mret = {c: st.fmean([R[k][c] for k in PURIF if k != "identity"]) for c in CONDS}
X = [ident[c] for c in CONDS]
Y = [mret[c] for c in CONDS]
mx, my = st.fmean(X), st.fmean(Y)
PEAR = (sum((a - mx) * (b - my) for a, b in zip(X, Y)) /
        math.sqrt(sum((a - mx) ** 2 for a in X) * sum((b - my) ** 2 for b in Y)))

# ---------- 影像板 ----------
plates = []
for img in IMAGES:
    tiles = ['<figure class="pl"><img src="data:image/jpeg;base64,%s" alt="%s 原圖">'
             '<figcaption><b>原圖</b><span class="m">未防禦</span></figcaption></figure>'
             % (b64("runs/hb5/%s__orig.png" % img, 512), img)]
    for c in CONDS:
        r = res[(img, c)]
        tiles.append(
            '<figure class="pl%s"><img src="data:image/jpeg;base64,%s" alt="%s %s">'
            '<figcaption><b>%s</b><span class="m">%s</span>'
            '<span class="d">DISTS %.4f · LPIPS %.4f · PSNR %.1f</span>'
            '</figcaption></figure>'
            % (" hi" if c == "phase" else "", b64(path(img, c, "def"), 512), img, c,
               LABEL[c], BUDGET[c], float(r["fid_dists"]), float(r["fid_lpips"]),
               float(r["fid_psnr"])))
    ed = ['<figure class="ed"><img src="data:image/jpeg;base64,%s" alt="未防禦編輯">'
          '<figcaption><b>未防禦的編輯</b></figcaption></figure>'
          % b64("runs/hb5/%s__phase__human__edit_orig.png" % img, 288)]
    for c in CONDS:
        ed.append(
            '<figure class="ed%s"><img src="data:image/jpeg;base64,%s" alt="%s 編輯">'
            '<figcaption><b>%s</b><span class="d">位移 %.3f</span></figcaption></figure>'
            % (" hi" if c == "phase" else "", b64(path(img, c, "edit_def"), 288), c,
               LABEL[c], float(res[(img, c)]["edit_lpips"])))
    note = "" if img in common else '<span class="tag">retention 部分不可用</span>'
    plates.append(
        '<section class="plate"><h3>%s <span class="pr">編輯 prompt：“%s”</span> %s</h3>'
        '<h4>防禦後的影像（人眼判定失真用）</h4><div class="grid8">%s</div>'
        '<h4>編輯結果</h4><div class="grid8 sm">%s</div></section>'
        % (img, PROMPT[img], note, "".join(tiles), "".join(ed)))

# ---------- 表 ----------
best_edit = max(S[c]["edit"] for c in CONDS)
t1 = "".join(
    '<tr class="%s"><td>%s</td><td class="g">%s</td><td class="mono">%s</td>'
    '<td class="n%s">%.4f</td><td class="n">%.4f</td><td class="n">%.4f</td>'
    '<td class="n">%.2f</td></tr>'
    % ("hi" if c == "phase" else "", LABEL[c], GROUP[c], BUDGET[c],
       " b" if S[c]["edit"] == best_edit else "", S[c]["edit"], S[c]["dists"],
       S[c]["lpips"], S[c]["psnr"])
    for c in CONDS)


def ptab(D):
    out = []
    for k in PURIF:
        vals = {c: D[k][c] for c in CONDS}
        best = max(vals.values())
        rank = sorted(vals, key=lambda c: -vals[c]).index("phase") + 1
        out.append('<tr><td>%s</td>%s<td class="n rk">%d/7</td></tr>' % (
            PNAME[k],
            "".join('<td class="n%s%s">%.4f</td>'
                    % (" b" if vals[c] == best else "",
                       " hi" if c == "phase" else "", vals[c]) for c in CONDS),
            rank))
    return "".join(out)


corr = "".join(
    '<tr class="%s"><td>%s</td><td class="n">%.4f</td><td class="n">%.4f</td></tr>'
    % ("hi" if c == "phase" else "", LABEL[c], ident[c], mret[c])
    for c in sorted(CONDS, key=lambda c: -ident[c]))
head = "".join("<th>%s</th>" % LABEL[c] for c in CONDS)

CSS = """
:root{
  --bg:#f6f8f9; --surf:#ffffff; --surf2:#eef2f4; --ink:#131a1f; --mut:#5b6b76;
  --line:#dbe3e8; --acc:#0d6d8a; --acc-w:#e3f1f6; --warn:#9c4f18; --warn-w:#f7ece3;
  --ok:#1d6b45;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0d1113; --surf:#151b1f; --surf2:#1d252a; --ink:#e4ebef; --mut:#8ea0ac;
  --line:#232d33; --acc:#54c4e2; --acc-w:#0f2c36; --warn:#e0904f; --warn-w:#2e2013;
  --ok:#5cc48f;
}}
:root[data-theme="dark"]{
  --bg:#0d1113; --surf:#151b1f; --surf2:#1d252a; --ink:#e4ebef; --mut:#8ea0ac;
  --line:#232d33; --acc:#54c4e2; --acc-w:#0f2c36; --warn:#e0904f; --warn-w:#2e2013;
  --ok:#5cc48f;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font:16px/1.65 -apple-system,"Segoe UI","Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:44px 26px 90px}
.mono,.n,.d,code{font-family:ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace;
  font-variant-numeric:tabular-nums}
code{font-size:.9em;background:var(--surf2);padding:1px 5px;border-radius:3px}
header{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:34px}
h1{font-size:31px;line-height:1.2;margin:0 0 10px;letter-spacing:-.02em;text-wrap:balance}
.sub{color:var(--mut);margin:0;max-width:68ch;font-size:15px}
.meta{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:18px;font-size:12.5px;color:var(--mut)}
.meta b{color:var(--ink);font-weight:600}
h2{font-size:13px;letter-spacing:.13em;text-transform:uppercase;color:var(--acc);
  margin:52px 0 6px;font-weight:700}
.lede{font-size:20px;line-height:1.42;margin:0 0 20px;max-width:60ch;letter-spacing:-.01em;
  text-wrap:balance;font-weight:600}
h3{font-size:17px;margin:34px 0 4px;letter-spacing:-.01em}
h4{font-size:11.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--mut);
  margin:22px 0 9px;font-weight:700}
p{max-width:66ch}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:7px;background:var(--surf);margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:640px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--surf2);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--mut);font-weight:700}
tr:last-child td{border-bottom:none}
td.n{text-align:right}
td.g{color:var(--mut);font-size:12px}
td.mono{font-size:12px;color:var(--mut)}
td.b{font-weight:700;color:var(--ok)}
tr.hi td,td.hi{background:var(--acc-w)}
td.rk{color:var(--mut);font-size:12px}
.grid8{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:13px}
.grid8.sm{grid-template-columns:repeat(auto-fill,minmax(155px,1fr))}
figure{margin:0;background:var(--surf);border:1px solid var(--line);border-radius:7px;overflow:hidden}
figure.hi{border-color:var(--acc);border-width:2px}
figure img{width:100%;display:block;aspect-ratio:1;object-fit:cover}
figcaption{padding:7px 9px;font-size:12px;line-height:1.45}
figcaption b{display:block;font-weight:650}
figcaption .m{display:block;color:var(--mut);font-size:11px}
figcaption .d{display:block;color:var(--mut);font-size:10.5px;margin-top:3px;
  font-family:ui-monospace,Consolas,monospace}
.plate{border-top:1px solid var(--line);padding-top:8px;margin-top:38px}
.pr{font-weight:400;font-size:13px;color:var(--mut)}
.tag{font-size:10.5px;padding:2px 7px;border-radius:99px;background:var(--warn-w);
  color:var(--warn);font-weight:700;vertical-align:2px}
.box{border-left:3px solid var(--acc);background:var(--surf);padding:15px 19px;
  border-radius:0 7px 7px 0;margin:20px 0}
.box.w{border-left-color:var(--warn)}
.box h5{margin:0 0 7px;font-size:14px}
.box p{margin:0;font-size:14.5px}
.box p+p{margin-top:9px}
ul{max-width:66ch;padding-left:20px}
li{margin:6px 0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:22px 0}
.stat{background:var(--surf);border:1px solid var(--line);border-radius:7px;padding:14px 16px}
.stat .k{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);
  font-weight:700;margin-bottom:6px}
.stat .v{font-size:26px;font-weight:700;letter-spacing:-.02em;
  font-family:ui-monospace,Consolas,monospace}
.stat .s{font-size:11.5px;color:var(--mut);margin-top:3px}
.stat.up .v{color:var(--ok)}
.stat.dn .v{color:var(--warn)}
a:focus-visible,figure:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
"""

html = """<title>相位重取樣防禦實測</title>
<style>%s</style>
<div class="wrap">
<header>
<h1>人眼門檻下的紋理重相位：抗編輯與抗淨化實測</h1>
<p class="sub">五個類別各一張真實照片、七個防禦條件、十個淨化算子。site F 走使用者裁定的
人眼可接受半徑，四個 baseline 走各自論文的原生預算（未對齊失真）。</p>
<div class="meta">
<span><b>批次</b> runs/hb5 · runs/hb5_pgc</span>
<span><b>日期</b> 2026-08-14</span>
<span><b>硬體</b> 4 × RTX 3090</span>
<span><b>耗時</b> 4.05 小時</span>
<span><b>攻擊者</b> stock SD v1.4 · SDEdit strength 0.55</span>
<span><b>retention</b> 3 個編輯 seed · 3σ 分母閘</span>
</div>
</header>

<h2>總結</h2>
<p class="lede">位移量上 site F 勝過同門檻的加性對照 1.41×、勝過同失真隨機 2.73×，
但與原生預算的 PhotoGuard-c 打成平手；抗淨化的「衰減率」讀數在這組預算下不可解讀。</p>

<div class="stats">
<div class="stat up"><div class="k">phase / add</div><div class="v">1.41×</div>
<div class="s">同為人眼門檻</div></div>
<div class="stat up"><div class="k">phase / 隨機相位</div><div class="v">2.73×</div>
<div class="s">同失真對照</div></div>
<div class="stat"><div class="k">phase / PhotoGuard-c</div><div class="v">0.99×</div>
<div class="s">實質打平</div></div>
<div class="stat up"><div class="k">淨化後勝加性</div><div class="v">10/10</div>
<div class="s">絕對位移量</div></div>
<div class="stat dn"><div class="k">r(分母, 衰減率)</div><div class="v">%+.2f</div>
<div class="s">比值被分母支配</div></div>
</div>

<h2>一 · 抗編輯</h2>
<p class="lede">未淨化時的位移量與保真度，五張影像平均。</p>
<div class="tw"><table>
<thead><tr><th>條件</th><th>組別</th><th>預算</th><th>位移量 edit_lpips</th>
<th>DISTS</th><th>LPIPS</th><th>PSNR</th></tr></thead>
<tbody>%s</tbody></table></div>

<div class="box w"><h5>這張表不是公平比較，必須這樣讀</h5>
<p>只有 <b>site F</b>、<b>加性 δ</b>、<b>隨機相位</b> 三者在同一條人眼門檻上——它們是唯一
可以直接比位移量的三個條件，比值為 <b>1.41×</b> 與 <b>2.73×</b>。</p>
<p>四個 baseline 跑在各自論文的原生預算上，失真高出一到一個數量級。Mist 的位移量最高
（0.630），但它的 LPIPS 是 0.561、DISTS 是 0.134，分別是 site F 的 2.6 倍與 3.3 倍——
那個位移量是用肉眼可見的破壞換來的。</p>
<p>真正的對手是 <b>PhotoGuard-c</b>：位移量 0.478 對 site F 的 0.476，差 0.6%%，實質打平。
它的 DISTS 只有一半、PSNR 高 10.7 dB，但 LPIPS 高 33%%。<b>誰的失真較小，十六項指標
沒有共識</b>，只能由下方的影像板以人眼裁定。</p></div>

<h2>二 · 影像板</h2>
<p class="lede">每一格都有影像。防禦後的圖以 512² 呈現供判定失真，編輯結果以 288²
呈現供判定防禦效果。</p>
<p>判定的問題有兩個。第一，<b>PhotoGuard-c 的失真是否通過你的人眼門檻</b>——它的 L∞ 高達
0.14–0.46（36–116/255），遠超你為加性劃的 1.2/255，但因為 L2 球讓擾動極稀疏，PSNR
仍有 40.9。指標在這裡分不出來。第二，<b>site F 在 θ=1.30 下的塊狀斑駁</b>是否在這五張上
仍可接受——它的 PSNR 逐圖從 23.2 漂到 30.6，固定半徑不等於固定失真。</p>
%s

<h2>三 · 抗淨化</h2>
<p class="lede">兩種讀法給出相反的答案，原因可診斷。</p>
<p>35 格中 30 格通過 3σ 分母閘。<b>七個條件全部可用的影像只有 %d 張</b>（%s）——
<code>cat_01</code> 的三個 A 臂條件與 <code>dia_r</code>、以及 <code>horse_03</code> 的
<code>apa_weak</code> 分母塌陷。下面兩張表一律取這 %d 張的交集，避免偷換樣本。</p>

<h4>讀法甲 · 淨化後的絕對位移量（越高越好）</h4>
<div class="tw"><table><thead><tr><th>淨化算子</th>%s<th>site F 名次</th></tr></thead>
<tbody>%s</tbody></table></div>
<p>site F 在十個算子中的八個排第二，唯一贏它的是 Mist——而 Mist 的失真是它的 3.3 倍。
在同為人眼門檻的三個條件之間，site F 對加性 δ 是 <b>10/10 全勝</b>，含 C&amp;R 串接。</p>

<h4>讀法乙 · 衰減率 retention（越高越好）</h4>
<div class="tw"><table><thead><tr><th>淨化算子</th>%s<th>site F 名次</th></tr></thead>
<tbody>%s</tbody></table></div>

<div class="box w"><h5>比值在這組預算下不可解讀</h5>
<p>注意<code>裁切重縮</code>那一列：七個條件的<b>絕對</b>位移量收斂到 0.495–0.617，幾乎相同。
<code>高斯雜訊</code>也是（0.434–0.564）。<b>強淨化算子自己就會把編輯推開約 0.5，
與有沒有防禦無關。</b></p>
<p>於是比值 =（防禦效果 ＋ 算子地板）/ 防禦效果，分母越小、比值越被地板灌大。
把七個條件的分母對上它們的平均衰減率，Pearson <b>r = %+.2f</b>：</p></div>

<div class="tw"><table>
<thead><tr><th>條件</th><th>分母 effect(identity)</th><th>平均 retention</th></tr></thead>
<tbody>%s</tbody></table></div>
<p>兩個排序幾乎完全相反。<b>衰減率量到的是分母大小，不是防禦的耐受度。</b>
FND-033 當時成立是因為 DISTS 對齊讓兩個分母只差 7.7%%（0.5094 對 0.4728）；
人眼門檻把加性的分母壓到 site F 的 57%%，比值就失去可比性。</p>

<h2>四 · 結論與缺口</h2>
<ul>
<li><b>並列主張成立。</b> 同一條人眼門檻上，site F 的位移量是加性 δ 的 1.41×、
同失真隨機的 2.73×，淨化後對加性 10/10 全勝。</li>
<li><b>主張一（更抗淨化）在此不可判定。</b> 不是被否證，是量測工具在這組預算下失效。
現行的 <code>retention</code> 定義只在分母相近時可用。</li>
<li><b>PhotoGuard-c 是真正的對手。</b> 原生預算下位移量與 site F 打平，且在 DISTS/PSNR
上更優、LPIPS 上更差。必須由人眼裁定，且該裁定要進論文就得是雙盲 2AFC，
不能是單一評分者。</li>
<li><b>缺一個控制組。</b> 需要 <code>effect(purify(原圖))</code>——把未防禦的原圖淨化後
再編輯，量算子自己的位移地板。有了它才能報扣掉地板的淨增益，主張一才有可判定的讀數。
成本約 55 分鐘一張卡。</li>
<li><b>site F 的固定 θ 不等於固定失真。</b> PSNR 在五張上從 23.2 漂到 30.6。
PhotoGuard-c 的 L2 球則穩定在 40.90，因為固定 L2 半徑就是固定 RMS。
這是可改進的工程缺陷。</li>
</ul>
</div>""" % (CSS, PEAR, t1, "".join(plates), len(common), "、".join(common),
             len(common), head, ptab(E), head, ptab(R), PEAR, corr)

open(OUT, "w", encoding="utf-8").write(html)
print("bytes", len(html.encode()), "->", OUT)
print("common:", common, "PEAR=%.4f" % PEAR)
