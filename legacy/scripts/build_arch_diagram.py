"""直式架構圖（大字），供報告內嵌。只描述現行設計，不提批次代號。"""
import json
import os

D = os.path.expandvars(
    r"$TEMP\claude\C--WACV-s3\f97b0be2-7c2c-4175-8705-a671a63a1017\scratchpad")
IMG = json.load(open(os.path.join(D, "imgs.json")))

W, H = 900, 2260
S = []

# 報告用的 CSS 變數色（淺底），與 report_s3t25 同一組語意
INK, SOFT, FAINT, RULE = "#14171C", "#5B6472", "#8A93A0", "#D8DCE2"
OURS, BASE, CTRL = "#0E5A61", "#A5680A", "#5B6472"
SURF, SURF2 = "#FFFFFF", "#EAECEF"
GOOD, BAD = "#1F7A4C", "#A8231F"


def box(x, y, w, h, title, sub="", sub2="", stroke=OURS, fill="#F4F8F8",
        ts=19, ss=14):
    o = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="2.5"/>']
    cx = x + w / 2
    o.append(f'<text x="{cx}" y="{y+30}" fill="{INK}" font-size="{ts}" '
             f'font-weight="800" text-anchor="middle">{title}</text>')
    if sub:
        o.append(f'<text x="{cx}" y="{y+55}" fill="{SOFT}" font-size="{ss}" '
                 f'font-weight="600" text-anchor="middle">{sub}</text>')
    if sub2:
        o.append(f'<text x="{cx}" y="{y+77}" fill="{FAINT}" font-size="{ss-1}" '
                 f'font-weight="600" text-anchor="middle">{sub2}</text>')
    return "\n".join(o)


def t(x, y, s, fill=SOFT, size=14, anchor="middle", weight="600"):
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def down(x, y1, y2, color=CTRL, dash=None, w=2.5):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M {x} {y1} L {x} {y2}" stroke="{color}" stroke-width="{w}" '
            f'fill="none" marker-end="url(#d)"{d}/>')


def right(x1, x2, y, color=CTRL, dash=None, w=2.5):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M {x1} {y} L {x2} {y}" stroke="{color}" stroke-width="{w}" '
            f'fill="none" marker-end="url(#r)"{d}/>')


def image(x, y, side, key, cap, sub=""):
    o = [f'<image x="{x}" y="{y}" width="{side}" height="{side}" href="{IMG[key]}"/>',
         f'<rect x="{x}" y="{y}" width="{side}" height="{side}" fill="none" '
         f'stroke="{RULE}" stroke-width="1.5"/>',
         t(x + side / 2, y + side + 20, cap, INK, 13, weight="600")]
    if sub:
        o.append(t(x + side / 2, y + side + 38, sub, FAINT, 12))
    return "\n".join(o)


S.append(f'''<defs>
<marker id="d" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto">
  <polygon points="0 0, 12 4.5, 0 9" fill="{CTRL}"/></marker>
<marker id="r" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto">
  <polygon points="0 0, 12 4.5, 0 9" fill="{CTRL}"/></marker>
</defs>''')
S.append(f'<rect width="{W}" height="{H}" fill="{SURF}"/>')

# ── 標題 ───────────────────────────────────────────────────────────────
S.append(t(W / 2, 42, "白盒非加性抗編輯防禦", INK, 30, weight="800"))
S.append(t(W / 2, 70, "威脅模型：攻擊方使用 stock Stable Diffusion，"
                      "prompt 未知；防禦方為外掛模組", SOFT, 14))

# ── ① 原圖 ────────────────────────────────────────────────────────────
y = 100
S.append(image(390, y, 120, "orig", "原圖 x", "512 × 512"))
S.append(down(450, y + 162, y + 196))

# ── ② 防禦圖生成 G(x; φ) ──────────────────────────────────────────────
gy = 200
S.append(f'<rect x="90" y="{gy}" width="720" height="470" rx="12" fill="none" '
         f'stroke="{OURS}" stroke-width="2" stroke-dasharray="10,5"/>')
S.append(t(110, gy + 26, "防禦圖生成 G(x; φ)　—　生成路徑", OURS, 16,
           anchor="start", weight="700"))

S.append(box(140, gy + 44, 300, 78, "生成起點",
             "由最佳化求得的 latent z*，", "使 decode(z*) ≈ x（4×64×64）"))
S.append(t(590, gy + 74, "取代編碼器輸出", INK, 14, weight="600"))
S.append(t(590, gy + 96, "重建誤差 LPIPS", SOFT, 13))
S.append(t(590, gy + 114, "0.128 → 0.080", GOOD, 14, weight="700"))
S.append(down(290, gy + 128, gy + 156))

S.append(box(140, gy + 160, 300, 68, "精確反演（BDIA）",
             "k = 10 步，模組關閉", stroke=CTRL, fill=SURF2))
S.append(t(590, gy + 196, "latent 來回誤差 1.4e−4", SOFT, 13))
S.append(down(290, gy + 234, gy + 262))

S.append(box(140, gy + 266, 300, 88, "去噪 10 步（UNet）",
             "每步一次噪聲預測 ε_θ", "φ 於此注入"))
S.append(down(290, gy + 360, gy + 388))

S.append(box(140, gy + 392, 300, 68, "解碼器",
             "含逐影像微調的 35,715 個參數", stroke=CTRL, fill=SURF2))

# φ 方塊（右側）
S.append(box(490, gy + 266, 280, 88, "φ ∈ ℝ^163840",
             "10 步 × 4×64×64 的低秩張量", "rank ≤ 32，外積參數化",
             stroke=BASE, fill="#FBF6EC"))
S.append(right(485, 445, gy + 310, BASE))
S.append(t(630, gy + 372, "φ = 0 時殘差精確為零", FAINT, 12))
S.append(t(630, gy + 390, "→ 可取得未防禦的重建基準", FAINT, 12))

S.append(down(450, gy + 470, gy + 508))

# ── ③ 防禦圖 ──────────────────────────────────────────────────────────
dy = 700
S.append(image(390, dy, 120, "xdef_pj", "防禦圖 x_def", "失真預算 Δ = 0.04"))
S.append(down(450, dy + 162, dy + 200))

# 對照支線
S.append(f'<path d="M 390 {dy+60} L 210 {dy+60} L 210 {dy+340}" '
         f'stroke="{CTRL}" stroke-width="2" stroke-dasharray="7,5" fill="none" '
         f'marker-end="url(#d)"/>')
S.append(t(210, dy + 40, "對照支線 φ = 0", CTRL, 13, weight="600"))
S.append(t(210, dy + 22, "（未防禦）", FAINT, 12))

# ── ④ 淨化與攻擊 ──────────────────────────────────────────────────────
ay = 900
S.append(box(300, ay, 300, 62, "淨化算子 𝒫", "16 個設定（模糊／JPEG／量化…）",
             stroke=BAD, fill="#FBF0F0"))
S.append(down(450, ay + 66, ay + 100))
S.append(box(300, ay + 104, 300, 78, "攻擊方 SDEdit",
             "stock SD v1.4，50 步", "prompt 未知，strength 0.4",
             stroke=BAD, fill="#FBF0F0"))
S.append(box(120, ay + 104, 180, 78, "同一條鏈", "同 prompt / seed / 淨化",
             stroke=CTRL, fill=SURF2, ts=14, ss=12))
S.append(down(450, ay + 186, ay + 220))

# ── ⑤ 輸出與判準 ──────────────────────────────────────────────────────
oy = 1130
S.append(image(250, oy, 120, "edit_ctrl", "未防禦的編輯 y₀", "判準基線"))
S.append(image(530, oy, 120, "edit_apa", "防禦後的編輯 y", ""))
S.append(f'<path d="M 380 {oy+60} L 520 {oy+60}" stroke="{OURS}" '
         f'stroke-width="2.5" fill="none"/>')
S.append(t(450, oy + 48, "主指標", OURS, 15, weight="700"))
S.append(t(450, oy + 216, "位移量 = LPIPS(y₀, y)　越大代表編輯被推得越遠",
           INK, 16, weight="700"))
S.append(t(450, oy + 240, "另報 PSNR↓ / SSIM↓ / VIF_p↓ / FSIM↓（同一組文獻欄位）",
           SOFT, 13))

# ── ⑥ 訓練目標與約束 ──────────────────────────────────────────────────
ly = 1420
S.append(f'<path d="M 40 {ly-24} L 860 {ly-24}" stroke="{RULE}" stroke-width="2"/>')
S.append(t(W / 2, ly + 6, "訓練目標與約束", INK, 22, weight="700"))
S.append(t(W / 2, ly + 32, "本研究的唯一架構變因是 L_def；其餘全部固定",
           SOFT, 14))

S.append(box(90, ly + 52, 340, 96, "保真約束（固定）",
             "每步更新後將 φ 投影回球面：", "metric(G(x;φ)) − metric(G(x;0)) = Δ",
             stroke=OURS, fill="#F4F8F8"))
S.append(t(260, ly + 168, "與評測期同一個度量與同一個 Δ", FAINT, 12))
S.append(t(260, ly + 186, "→ 訓練與評測綁在同一個預算上", FAINT, 12))

S.append(box(470, ly + 52, 340, 96, "銳利度與色偏（固定）",
             "不隨縮放單調，無法投影", "→ 以可行性過濾承擔",
             stroke=CTRL, fill=SURF2))
S.append(t(640, ly + 168, "只有滿足約束的步", FAINT, 12))
S.append(t(640, ly + 186, "才有資格成為最佳步", FAINT, 12))

vy = ly + 216
S.append(t(W / 2, vy + 6, "三個條件（L_def 之外逐項相同）", INK, 18, weight="700"))
arms = [
    ("A　注意力抑制", "L_def = ‖Att(x_def, c_a) ⊙ M‖₁",
     "壓低防禦方指名的詞 c_a 在其對應區域的注意力", OURS, "#F4F8F8"),
    ("B　目標輸出", "L_def = ‖SDEdit(x_def; c_∅) − y_target‖²",
     "把代理編輯鏈的輸出推向固定目標影像", BASE, "#FBF6EC"),
    ("C　隨機對照", "不最佳化",
     "同參數化的高斯方向，縮放至同一個 Δ", CTRL, SURF2),
]
ay2 = vy + 26
for i, (name, formula, desc, sk, fl) in enumerate(arms):
    yy = ay2 + i * 106
    S.append(f'<rect x="90" y="{yy}" width="720" height="92" rx="8" fill="{fl}" '
             f'stroke="{sk}" stroke-width="2"/>')
    S.append(t(110, yy + 30, name, sk, 17, anchor="start", weight="700"))
    S.append(t(110, yy + 58, formula, INK, 15, anchor="start", weight="600"))
    S.append(t(110, yy + 80, desc, SOFT, 13, anchor="start"))

S.append(t(W / 2, ay2 + 340, "三者的參數化、失真預算、優化器、停止準則、"
                             "評測設定完全相同", FAINT, 13))

svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
       f'style="width:100%;height:auto;max-width:900px;display:block;margin:0 auto">\n'
       + "\n".join(S) + "\n</svg>")

out = os.path.join(D, "arch_vertical.svg")
open(out, "w", encoding="utf-8").write(svg)
print(f"寫入 {out}（{os.path.getsize(out)/1024:.0f} KB）")
