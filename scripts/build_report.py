"""產生實驗報告，版面對齊根目錄的 report_s3t25.html。

主指標只用位移；三個條件差別只在 L_def；保真約束固定為投影。
"""
import base64
import io
import json
import os
from pathlib import Path

from PIL import Image

D = Path(os.path.expandvars(
    r"$TEMP\claude\C--WACV-s3\f97b0be2-7c2c-4175-8705-a671a63a1017\scratchpad"))
R = Path(r"C:\WACV-s3\runs")
DATA = json.load(open(D / "report_data.json", encoding="utf-8"))
CSS = io.open(D / "report_css.html", encoding="utf-8").read()
ARCH = io.open(D / "arch_vertical.svg", encoding="utf-8").read()

ARM = {"attn": ("A", "注意力抑制", R / "s3t20_pj_merged" / "apa_pj"),
       "target": ("B", "目標輸出", R / "s3t20_tj_merged" / "apa_tj"),
       "random": ("C", "隨機對照", R / "s3t20_r_merged" / "Ra")}
IMGS = ["horse_00", "horse_03", "woman_03"]
PUR = [("identity", 0.0, "不淨化"), ("blur", 0.25, "模糊 0.25"),
       ("blur", 0.5, "模糊 0.5"), ("blur", 0.75, "模糊 0.75"),
       ("noise", 0.005, "雜訊 0.005"), ("noise", 0.01, "雜訊 0.01"),
       ("quantize", 128.0, "量化 128"), ("quantize", 64.0, "量化 64"),
       ("quantize", 32.0, "量化 32"), ("quantize", 16.0, "量化 16"),
       ("jpeg", 75.0, "JPEG 75"), ("jpeg", 30.0, "JPEG 30"),
       ("crop_resize", 0.1, "裁切縮放"), ("adverse_cleaner", 0.0, "去噪器"),
       ("diffpure", 150.0, "DiffPure")]


def b64(p, side=250, q=90):
    with Image.open(p) as im:
        im = im.convert("RGB")
        if im.size[0] != side:
            im = im.resize((side, side), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def fig(p, cap, sub=""):
    if not Path(p).exists():
        return f'<figure><div class="missing">缺 {Path(p).name}</div></figure>'
    s = f'<span class="sub">{sub}</span>' if sub else ""
    return (f'<figure><img alt="{cap}" src="{b64(p)}">'
            f'<figcaption><span class="t">{cap}</span>{s}</figcaption></figure>')


def pdir(kind, strength):
    return f"{kind}_{strength:g}" if kind != "identity" else "identity_0"


H = ["<!DOCTYPE html><html lang=\"zh-Hant\"><head><meta charset=\"utf-8\">",
     '<meta name="viewport" content="width=device-width,initial-scale=1">',
     "<title>抗文字編輯防禦 · 損失函數的比較</title>", CSS, "</head><body>",
     '<div class="wrap">']

# ── masthead ─────────────────────────────────────────────────────────
H.append('''<div class="masthead">
<p class="eyebrow">實驗報告</p>
<h1>非加性抗文字編輯防禦：注意力抑制與目標輸出兩種訓練目標的比較</h1>
<p class="lede measure">在固定的參數化、失真預算與保真約束下，唯一的變因是防禦項
<code>L_def</code>。主指標為<b>位移量</b>——防禦後的編輯輸出離未防禦的編輯輸出多遠。
三個條件為注意力抑制、目標輸出、以及同參數化的隨機對照。</p>
<p class="meta">Stable Diffusion v1.4 · 512² · fp32 · strength 0.4 ·
失真預算 Δ = 0.04（相對 DISTS） · 3 影像 × 5 種子 × 15 個淨化設定 ·
2026-08-11</p>
</div>''')

# ── verdicts ─────────────────────────────────────────────────────────
m = DATA["main"]
H.append(f'''<div class="verdicts">
<div class="card is-bad"><span class="k">否定</span>
<span class="v">兩個訓練目標都沒有產生可用的抗編輯效果</span>
<span class="sub">不淨化時位移量：注意力抑制 {m['attn']['edit_lpips']:.4f}、
目標輸出 {m['target']['edit_lpips']:.4f}、隨機對照 {m['random']['edit_lpips']:.4f}。
最佳化相對隨機的增益是 1.56× 與 1.20×，而外部參照的加性方法在同一評測下是
0.24–0.36。</span></div>
<div class="card is-bad"><span class="k">否定</span>
<span class="v">與評測指標對齊的損失反而更差</span>
<span class="sub">目標輸出直接最小化「離未防禦編輯的距離」，即評測所量的東西，
其位移量卻低於注意力抑制 14.4%（配對 n=225，較優比例 25%）。訓練期的代理編輯鏈
（10 步）與評測期的真實攻擊鏈（50 步）之間的落差，大於損失形式的差別。</span></div>
<div class="card is-good"><span class="k">確立</span>
<span class="v">投影式約束使訓練與評測綁在同一個預算上</span>
<span class="sub">兩個訓練條件的射線縮放係數為 0.938–1.000，即訓練所得的 φ
本身就落在評測的預算球面上；隨機對照為 0.500–0.875。三者實測的
<code>fid_dists</code> 為 0.0756 / 0.0752 / 0.0752。</span></div>
</div>''')

# ── 架構 ─────────────────────────────────────────────────────────────
H.append('<div class="head"><p class="eyebrow">一</p><h2>系統架構</h2></div>')
H.append('<p class="measure">下圖是完整的資料流與訓練設定。'
         '灰色虛線是未防禦的對照支線，它產生的編輯輸出 y₀ 是位移量的基準——'
         '沒有那條支線，「編輯被推得多遠」沒有定義。</p>')
H.append(f'<figure>{ARCH}<figcaption><span class="t">系統架構</span>'
         f'<span class="sub">唯一的架構變因是 L_def；參數化、失真預算、'
         f'保真約束、優化器與停止準則三個條件完全相同</span></figcaption></figure>')

H.append('''<div class="callout"><b>關於保真約束。</b>
每一步梯度更新之後，φ 的方向參數被縮放回
<code>metric(G(x;φ)) − metric(G(x;0)) = Δ</code> 這個球面上，用的度量與 Δ
與評測期逐字相同。銳利度與色偏兩項不隨縮放單調變化，無法以縮放保證，
故以可行性過濾承擔：只有滿足它們的步才有資格成為最佳步。</div>''')

# ── 主結果 ───────────────────────────────────────────────────────────
H.append('<div class="head"><p class="eyebrow">二</p><h2>位移量</h2></div>')
H.append('<p class="measure">位移量比的是<b>未防禦的編輯</b>對<b>防禦後的編輯</b>，'
         '同一張影像、同一個 prompt、同一個噪聲種子、同一個淨化算子。'
         '五個欄位取自文獻共用的那一組。</p>')

rows = []
for k, (tag, name, _) in ARM.items():
    a = m[k]
    rows.append(f'<tr><td style="text-align:left">{tag} · {name}</td>'
                f'<td class="num{" good" if k=="attn" else ""}">'
                f'{a["edit_lpips"]:.4f}</td>'
                f'<td class="num">{a["edit_psnr"]:.2f}</td>'
                f'<td class="num">{a["edit_ssim"]:.4f}</td>'
                f'<td class="num">{a["edit_vif_p"]:.4f}</td>'
                f'<td class="num">{a["edit_fsim"]:.4f}</td></tr>')
rows.append('<tr><td colspan="6" style="text-align:left;padding-top:.6rem;'
            'color:var(--ink-faint);font-size:.8rem">以下為外部參照，'
            '不同參數化、不同保真約束型態，僅供判斷絕對水準</td></tr>')
for k, lab in DATA["ref_labels"].items():
    a = m[k]
    rows.append(f'<tr style="opacity:.65"><td style="text-align:left">{lab}</td>'
                f'<td class="num">{a["edit_lpips"]:.4f}</td>'
                f'<td class="num">{a["edit_psnr"]:.2f}</td>'
                f'<td class="num">{a["edit_ssim"]:.4f}</td>'
                f'<td class="num">{a["edit_vif_p"]:.4f}</td>'
                f'<td class="num">{a["edit_fsim"]:.4f}</td></tr>')
H.append('<div class="tw"><table><caption>不淨化下的位移量（3 影像 × 5 種子）'
         '</caption><thead><tr><th>條件</th><th>LPIPS ↑</th><th>PSNR ↓</th>'
         '<th>SSIM ↓</th><th>VIF_p ↓</th><th>FSIM ↓</th></tr></thead><tbody>'
         + "\n".join(rows) + '</tbody></table></div>')

p = DATA["paired"]
H.append(f'''<div class="tw"><table><caption>配對比較：同影像、同淨化、同種子
（以 A 為基準，15 個淨化設定全部納入）</caption>
<thead><tr><th>條件</th><th>A 的位移量</th><th>本條件</th><th>差</th>
<th>本條件較優的比例</th><th>n</th></tr></thead><tbody>
<tr><td style="text-align:left">B · 目標輸出</td>
<td class="num">{p["target"]["attn"]:.4f}</td>
<td class="num">{p["target"]["other"]:.4f}</td>
<td class="num">{(p["target"]["other"]/p["target"]["attn"]-1)*100:+.1f}%</td>
<td class="num">{p["target"]["win"]*100:.0f}%</td>
<td class="num">{p["target"]["n"]}</td></tr>
<tr><td style="text-align:left">C · 隨機對照</td>
<td class="num">{p["random"]["attn"]:.4f}</td>
<td class="num">{p["random"]["other"]:.4f}</td>
<td class="num">{(p["random"]["other"]/p["random"]["attn"]-1)*100:+.1f}%</td>
<td class="num">{p["random"]["win"]*100:.0f}%</td>
<td class="num">{p["random"]["n"]}</td></tr>
</tbody></table></div>''')

H.append('''<div class="callout bad"><b>兩個訓練目標都只小幅超過隨機方向。</b>
注意力抑制對隨機是 1.56×、目標輸出是 1.20×。同參數化的隨機擾動在同一個失真
預算下已經取得多數的位移，最佳化貢獻的部分有限。</div>''')

# ── 逐淨化曲線（inline SVG）─────────────────────────────────────────
H.append('<div class="head"><p class="eyebrow">三</p><h2>位移量隨淨化強度的變化</h2></div>')
H.append('<p class="measure">橫軸為淨化設定，縱軸為位移量。'
         '本節只納入未防禦的編輯自身尚未被淨化毀掉的設定——'
         '判準是該設定下未防禦編輯的 NIQE 相對不淨化的增幅小於 1.0，'
         '超過者其位移量比的是兩張都已毀掉的圖。</p>')

CW, CH, PADL, PADB = 860, 330, 56, 92
keys = [f"{k}_{s:g}" for k, s, _ in PUR]
labels = [lab for _, _, lab in PUR]
series = [("attn", "var(--ours)", "A 注意力抑制"),
          ("target", "var(--base)", "B 目標輸出"),
          ("random", "var(--ctrl)", "C 隨機對照")]
vals = [DATA["purify"][k][key] for k, _, _ in series for key in keys
        if key in DATA["purify"][k]]
vmax = max(vals) * 1.15
sx = lambda i: PADL + i * (CW - PADL - 20) / (len(keys) - 1)
sy = lambda v: CH - PADB - v / vmax * (CH - PADB - 18)

svg = [f'<svg viewBox="0 0 {CW} {CH}" style="width:100%;height:auto" '
       f'xmlns="http://www.w3.org/2000/svg">']
for gv in [0.05, 0.10, 0.15, 0.20, 0.25]:
    if gv > vmax:
        continue
    svg.append(f'<line x1="{PADL}" y1="{sy(gv):.1f}" x2="{CW-20}" '
               f'y2="{sy(gv):.1f}" stroke="var(--rule)" stroke-width="1"/>')
    svg.append(f'<text x="{PADL-8}" y="{sy(gv)+4:.1f}" fill="var(--ink-faint)" '
               f'font-size="11" text-anchor="end">{gv:.2f}</text>')
for i, lab in enumerate(labels):
    svg.append(f'<text x="{sx(i):.1f}" y="{CH-PADB+16}" fill="var(--ink-faint)" '
               f'font-size="10.5" text-anchor="end" '
               f'transform="rotate(-45 {sx(i):.1f} {CH-PADB+16})">{lab}</text>')
for k, col, name in series:
    pts = [(sx(i), sy(DATA["purify"][k][key]))
           for i, key in enumerate(keys) if key in DATA["purify"][k]]
    d = " ".join(("M" if j == 0 else "L") + f" {x:.1f} {y:.1f}"
                 for j, (x, y) in enumerate(pts))
    svg.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.5"/>')
    for x, y in pts:
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{col}"/>')
for j, (k, col, name) in enumerate(series):
    lx = PADL + j * 210
    svg.append(f'<line x1="{lx}" y1="{CH-18}" x2="{lx+26}" y2="{CH-18}" '
               f'stroke="{col}" stroke-width="2.5"/>')
    svg.append(f'<text x="{lx+32}" y="{CH-14}" fill="var(--ink-soft)" '
               f'font-size="12">{name}</text>')
svg.append('</svg>')
H.append(f'<figure>{"".join(svg)}<figcaption><span class="t">'
         f'位移量對淨化強度</span><span class="sub">三個條件的曲線幾乎平行，'
         f'順序在全部 15 個設定上不變</span></figcaption></figure>')

rows = []
for k, _, name in series:
    cells = "".join(f'<td class="num">{DATA["purify"][k].get(key, float("nan")):.4f}</td>'
                    for key in keys)
    rows.append(f'<tr><td style="text-align:left">{name}</td>{cells}</tr>')
H.append('<div class="tw"><table><caption>逐淨化設定的位移量</caption><thead><tr>'
         '<th>條件</th>' + "".join(f'<th>{l}</th>' for l in labels)
         + '</tr></thead><tbody>' + "\n".join(rows) + '</tbody></table></div>')

# ── 保真 ─────────────────────────────────────────────────────────────
H.append('<div class="head"><p class="eyebrow">四</p><h2>防禦圖的失真</h2></div>')
H.append('<p class="measure">三個條件在同一個失真預算上，故 DISTS 幾乎相同；'
         '差別在失真的<b>樣式</b>。銳利度比為 1 表示與原圖的梯度能量相同，'
         '偏離 1 即代表變鈍或變銳。</p>')
rows = []
for k, (tag, name, _) in ARM.items():
    a = m[k]
    rows.append(f'<tr><td style="text-align:left">{tag} · {name}</td>'
                f'<td class="num">{a["fid_lpips"]:.4f}</td>'
                f'<td class="num">{a["fid_dists"]:.4f}</td>'
                f'<td class="num">{a["fid_psnr"]:.2f}</td>'
                f'<td class="num">{a["fid_acutance_ratio"]:.3f}</td>'
                f'<td class="num">{a["acut_dev"]:.4f}</td>'
                f'<td class="num">{a["dniqe"]:+.3f}</td></tr>')
H.append('<div class="tw"><table><caption>防禦圖對原圖（不淨化）</caption>'
         '<thead><tr><th>條件</th><th>LPIPS</th><th>DISTS</th><th>PSNR</th>'
         '<th>銳利度比</th><th>|1−銳利度比|</th><th>ΔNIQE</th></tr></thead>'
         '<tbody>' + "\n".join(rows) + '</tbody></table></div>')
H.append('<p class="measure">ΔNIQE 為負代表防禦後的編輯輸出品質不比未防禦的差。'
         '三個條件皆為負，即位移量不是靠把輸出弄糟換來的。</p>')

# ── 影像 ─────────────────────────────────────────────────────────────
H.append('<div class="head"><p class="eyebrow">五</p><h2>防禦圖與編輯結果</h2></div>')
H.append('<p class="measure">每一組的第一張是未防禦的編輯，其餘為三個條件。'
         '種子固定為 0。</p>')
for img in IMGS:
    H.append(f'<h3>{img}</h3>')
    H.append('<div class="plate n3">')
    H.append(fig(R / "s3t20_r_merged" / "apa" / img / "orig.png", "原圖"))
    for k, (tag, name, root) in ARM.items():
        a = DATA["per_image"][k][img]
        H.append(fig(root / img / "x_def_tau0.04.png", f"{tag} · 防禦圖",
                     f"銳利度比 {a['fid_acutance_ratio']:.3f}"))
    H.append('</div>')
    H.append('<div class="plate n3">')
    H.append(fig(R / "s3t20_merged" / "control" / img / "purify" /
                 "identity_0" / "edit_seed0.png", "未防禦的編輯", "位移量的基準"))
    for k, (tag, name, root) in ARM.items():
        a = DATA["per_image"][k][img]
        H.append(fig(root / img / "purify" / "identity_0" /
                     "edit_tau0.04_seed0.png", f"{tag} · 編輯結果",
                     f"位移量 {a['edit_lpips']:.4f}"))
    H.append('</div>')

# 淨化後
H.append('<h3>淨化之後（模糊 0.75）</h3>')
H.append('<p class="measure">加性擾動在此強度上失去多數效果，'
         '非加性參數化不受影響——但兩者的絕對水準仍有差距。</p>')
for img in IMGS[:1]:
    H.append('<div class="plate n3">')
    H.append(fig(R / "s3t20_merged" / "control" / img / "purify" / "blur_0.75" /
                 "edit_seed0.png", "未防禦的編輯", img))
    for k, (tag, name, root) in ARM.items():
        v = DATA["purify"][k].get("blur_0.75", float("nan"))
        H.append(fig(root / img / "purify" / "blur_0.75" /
                     "edit_tau0.04_seed0.png", f"{tag} · 編輯結果",
                     f"位移量 {v:.4f}"))
    H.append('</div>')

# ── 注意力 ───────────────────────────────────────────────────────────
H.append('<div class="head"><p class="eyebrow">六</p><h2>注意力分佈</h2></div>')
H.append('<p class="measure">條件 A 的訓練目標作用在此。圖為跨層聚合後的注意力，'
         '亮處代表模型認為該詞位於此。條件 A 的訓練期抑制達 89–94%（'
         '在允許較大失真的設定下量得），而在本報告的失真預算下，'
         '評測期實測的抑制為 5–16%。</p>')
H.append('<div class="plate n3">')
H.append(fig(R / "s3t20_merged" / "control" / "horse_00" / "purify" /
             "identity_0" / "attn" / "seed0_agg.png", "未防禦", "horse_00"))
for k, (tag, name, root) in ARM.items():
    H.append(fig(root / "horse_00" / "purify" / "identity_0" / "attn" /
                 "tau0.04_seed0_agg.png", f"{tag} · {name}", ""))
H.append('</div>')

# ── 併自既有批次報告的章節 ──────────────────────────────────────────
exec(io.open(os.path.join(str(D), "extra_sections.py"),
             encoding="utf-8").read())

# ── 訓練 ─────────────────────────────────────────────────────────────
H.append('<div class="head"><p class="eyebrow">七</p><h2>訓練</h2></div>')
rows = []
for k, (tag, name, _) in ARM.items():
    for img in IMGS:
        e = DATA["train"][k].get(img, {})
        rows.append(
            f'<tr><td style="text-align:left">{tag} · {name}</td>'
            f'<td style="text-align:left">{img}</td>'
            f'<td class="num">{e.get("steps", "—")}</td>'
            f'<td class="num">{e.get("scale_k", float("nan")):.3f}</td>'
            f'<td class="num">{(e.get("seconds") or 0):.0f}</td>'
            f'<td style="text-align:left;white-space:normal;font-size:.8rem">'
            f'{e.get("stop", "—")}</td></tr>')
H.append('<div class="tw"><table><caption>逐格的訓練紀錄</caption><thead><tr>'
         '<th>條件</th><th>影像</th><th>步數</th><th>射線縮放係數</th>'
         '<th>秒</th><th>停止原因</th></tr></thead><tbody>'
         + "\n".join(rows) + '</tbody></table></div>')
H.append('''<p class="measure">射線縮放係數為 1 表示訓練所得的 φ 已落在評測的
預算球面上，縮放為空操作。兩個訓練條件為 0.938–1.000；隨機對照不經訓練，
其方向由縮放拉到同一個預算，係數 0.500–0.875。</p>''')

# ── 限制 ─────────────────────────────────────────────────────────────
H.append('<div class="head"><p class="eyebrow">八</p><h2>適用範圍與限制</h2></div>')
H.append('''<ul class="measure">
<li>樣本為 3 張影像 × 5 個噪聲種子。條件之間的比較是配對的（n = 225），
但跨影像的推論受限於樣本數。</li>
<li>本報告的主指標是位移量。位移量大不等於編輯在語意上失敗——先前的量測顯示
兩者可以背離，故本報告不由位移量推論編輯是否被導離 prompt。</li>
<li>15 個淨化設定之外另有 7 個更強的設定已量測但不納入曲線，理由見第三節：
在那些設定下未防禦的編輯自身已被毀掉。原始資料保留。</li>
<li>條件 C 不經訓練，其射線縮放係數與另兩者不同（0.500–0.875 對
0.938–1.000）。三者最終落在同一個失真預算上（DISTS 0.0752–0.0756），
但抵達的方式不同。</li>
<li>外部參照的三個加性方法使用不同的參數化與不同型態的保真約束，
其數值僅用於判斷絕對水準，不構成受控比較。</li>
</ul>''')

H.append('<div class="foot sub">資料來源：runs/s3t20_pj_merged · '
         'runs/s3t20_tj_merged · runs/s3t20_r_merged · runs/s3t20_merged。'
         '全部指標由同一組實作計算，未經挑選。</div>')
H.append('</div></body></html>')

out = Path(r"C:\WACV-s3\report_s3t20.html")
out.write_text("\n".join(H), encoding="utf-8")
print(f"寫入 {out}（{out.stat().st_size/1e6:.1f} MB）")
