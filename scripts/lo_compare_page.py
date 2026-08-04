"""L1 與 L4 的人眼比對頁與大圖。

    python scripts/lo_compare_page.py

只讀 `runs/` 底下既有的 PNG 與 CSV，不需要 GPU、不需要 SD 權重。產出：

    runs/figs/compare.html                     人眼比對頁（原尺寸，可放大看）
    runs/figs/2026-08-04_l1_three_attacks.png  L1：三個攻擊的破壞型態
    runs/figs/2026-08-04_l4_ours_vs_baseline.png  L4：兩個位置 vs pg_encoder

為什麼需要這一頁
─────────────────────────────────────────────────────────────────────

LEDGER 1.8：指標之間矛盾時，把影像做成比對頁交人眼判斷。這一輪有兩個
具體的矛盾要判：

1. **Table 1 判準判 `semantic` 最弱**（編輯 LPIPS 0.5593，對 pg_encoder 的
   0.6363），但三個攻擊的破壞**型態**完全不同——`pg_encoder` 整張糊掉褪色、
   `pg_diffusion` 保留結構疊雜訊與重影、`semantic` 色帶撕裂但構圖保留。
   單一個距離純量分不出「編輯失敗」與「編輯成功但輸出品質下降」。
2. **L4 的編輯 LPIPS 0.2490 聽起來像「有部分效果」**，實際看圖是「編輯照樣
   成功，只是細節略有不同」（LEDGER 3.20）。

字型：Windows 的比對頁曾整排標籤變成豆腐字。此處明確指定 msjh.ttc
（微軟正黑體），找不到就報錯而不是靜默退回 Pillow 的點陣預設字型——
標籤看不懂的比對頁等於沒有比對頁。
"""

import argparse
import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 依偏好順序。找不到任何一個就拋出：預設點陣字型畫不出中文，
# 而「標籤是豆腐」與「標籤畫錯」在外部分不出來。
CJK_FONTS = [
    "C:/Windows/Fonts/msjh.ttc",        # 微軟正黑體
    "C:/Windows/Fonts/mingliu.ttc",     # 細明體
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

L1_ATTACKS = ["pg_encoder", "pg_diffusion", "semantic"]
L4_SITES = ["PF", "S"]


def load_font(size: int):
    from PIL import ImageFont

    for p in CJK_FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise FileNotFoundError(
        f"找不到任何中日韓字型（試過 {CJK_FONTS}）。Pillow 的預設點陣字型"
        "畫不出中文，標籤會全部變成豆腐字，比對頁就失去用途"
    )


def summary_rows(run: str) -> dict:
    """回傳 {(影像, 攻擊/site): 該格的 summary 列}。"""
    f = ROOT / "runs" / run / "summary.csv"
    if not f.exists():
        return {}
    out = {}
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["image"], r["attack"])] = r
    return out


def montage(rows, out_path: Path, cell: int = 256):
    """rows = [(列標題, [(欄標題, 影像路徑), ...]), ...]，缺檔的格畫成灰底。"""
    from PIL import Image, ImageDraw

    if not rows:
        raise ValueError("沒有任何列可以畫")
    ncol = max(len(cs) for _, cs in rows)
    pad_top, pad_left = 30, 90
    W = pad_left + ncol * cell
    H = pad_top + len(rows) * (cell + 20)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_head = load_font(15)
    f_side = load_font(13)

    for ci, (title, _) in enumerate(rows[0][1]):
        d.text((pad_left + ci * cell + 4, 8), title, fill="black", font=f_head)

    for ri, (label, cells) in enumerate(rows):
        y = pad_top + ri * (cell + 20)
        d.text((4, y + cell // 2), label, fill="black", font=f_side)
        for ci, (_, path) in enumerate(cells):
            x = pad_left + ci * cell
            if path and Path(path).exists():
                im = Image.open(path).convert("RGB").resize(
                    (cell, cell), Image.LANCZOS)
                canvas.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(230, 230, 230))
                d.text((x + 8, y + cell // 2), "（無此檔）",
                       fill=(120, 120, 120), font=f_side)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def rel(p: Path) -> str:
    """比對頁放在 runs/figs/，圖片以相對路徑引用，整個 repo 可搬動。"""
    return "../../" + p.relative_to(ROOT).as_posix()


def img_cell(path: Path, caption: str) -> str:
    if not path.exists():
        return f'<td><div class="miss">（無此檔）</div><div class="cap">{caption}</div></td>'
    u = html.escape(rel(path))
    return (f'<td><a href="{u}" target="_blank"><img src="{u}"></a>'
            f'<div class="cap">{caption}</div></td>')


def l1_section(images) -> str:
    base = ROOT / "runs/lo_baseline"
    sm = summary_rows("lo_baseline")
    out = []
    for name, cls in images:
        src = ROOT / "data/lo_aligned" / cls / f"{name}.png"
        ref = base / f"{name}__pg_encoder_edit_ref.png"
        cells = [img_cell(src, "原圖"),
                 img_cell(ref, "未防禦的編輯（攻擊成功的樣子）")]
        for a in L1_ATTACKS:
            r = sm.get((name, a), {})
            lp = f"{float(r['edit_lpips']):.4f}" if r else "—"
            cells.append(img_cell(base / f"{name}__{a}__adv.png",
                                  f"{a} 免疫圖"))
            cells.append(img_cell(base / f"{name}__{a}_edit_def.png",
                                  f"{a} 防禦後編輯　編輯 LPIPS {lp}"))
        out.append(f"<tr>{''.join(cells)}</tr>")
    return "\n".join(out)


def l4_section(images) -> str:
    base = ROOT / "runs/ours_lo"
    lo = ROOT / "runs/lo_baseline"
    sm = summary_rows("ours_lo")
    smb = summary_rows("lo_baseline")
    out = []
    for name, cls in images:
        src = ROOT / "data/lo_aligned" / cls / f"{name}.png"
        cells = [img_cell(src, "原圖"),
                 img_cell(base / f"{name}__PF_edit_ref.png", "未防禦的編輯")]
        for s in L4_SITES:
            r = sm.get((name, s))
            if r:
                done, cap = int(r["steps_done"]), 150
                tag = "跑滿上限，不可比" if done >= cap else f"{done} 步收斂"
                cap_txt = (f"site {s} 防禦後編輯　編輯 LPIPS "
                           f"{float(r['edit_lpips']):.4f}　擾動 LPIPS "
                           f"{float(r['pert_lpips']):.4f}　{tag}")
            else:
                cap_txt = f"site {s} 防禦後編輯"
            cells.append(img_cell(base / f"{name}__{s}__def.png",
                                  f"site {s} 防禦圖"))
            cells.append(img_cell(base / f"{name}__{s}_edit_def.png", cap_txt))
        rb = smb.get((name, "pg_encoder"))
        lp = f"　編輯 LPIPS {float(rb['edit_lpips']):.4f}" if rb else ""
        cells.append(img_cell(lo / f"{name}__pg_encoder_edit_def.png",
                              f"pg_encoder 防禦後編輯（基準）{lp}"))
        out.append(f"<tr>{''.join(cells)}</tr>")
    return "\n".join(out)


CSS = """
body{font-family:"Microsoft JhengHei","Noto Sans CJK TC",sans-serif;
     margin:24px;line-height:1.7;color:#111;max-width:none}
h1{font-size:22px} h2{font-size:18px;margin-top:36px;border-top:2px solid #333;
   padding-top:12px}
table{border-collapse:collapse;margin:12px 0}
td{vertical-align:top;padding:4px;border:1px solid #ddd}
img{width:240px;height:240px;object-fit:cover;display:block}
.cap{font-size:12px;width:240px;color:#333;margin-top:4px}
.miss{width:240px;height:240px;background:#eee;color:#888;font-size:12px;
      display:flex;align-items:center;justify-content:center}
.note{background:#f6f6f6;border-left:4px solid #666;padding:10px 14px;
      margin:12px 0;font-size:14px}
"""


def main():
    ap = argparse.ArgumentParser(description="L1／L4 的人眼比對頁與大圖")
    ap.add_argument("--out", default="runs/figs")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # L1 每類取第一張，涵蓋六類；L4 只有 man 類跑到（機器時間用盡）
    l1_imgs = [("man_00", "man"), ("woman_00", "woman"), ("dog_00", "dog"),
               ("cat_00", "cat"), ("horse_00", "horse"), ("bird_00", "bird")]
    l4_imgs = [("man_00", "man"), ("man_01", "man"),
               ("man_02", "man"), ("man_03", "man")]

    base, ours = ROOT / "runs/lo_baseline", ROOT / "runs/ours_lo"
    montage(
        [(n, [("原圖", ROOT / "data/lo_aligned" / c / f"{n}.png"),
              ("未防禦編輯", base / f"{n}__pg_encoder_edit_ref.png")]
          + [(f"{a}\n免疫圖", base / f"{n}__{a}__adv.png") for a in L1_ATTACKS]
          + [(f"{a}\n防禦後編輯", base / f"{n}__{a}_edit_def.png")
             for a in L1_ATTACKS])
         for n, c in l1_imgs],
        out / "2026-08-04_l1_three_attacks.png")

    montage(
        [(n, [("原圖", ROOT / "data/lo_aligned" / c / f"{n}.png"),
              ("未防禦編輯", ours / f"{n}__PF_edit_ref.png"),
              ("PF 防禦圖", ours / f"{n}__PF__def.png"),
              ("PF 防禦後編輯", ours / f"{n}__PF_edit_def.png"),
              ("S 防禦圖", ours / f"{n}__S__def.png"),
              ("S 防禦後編輯", ours / f"{n}__S_edit_def.png"),
              ("pg_encoder\n防禦後編輯", base / f"{n}__pg_encoder_edit_def.png")])
         for n, c in l4_imgs],
        out / "2026-08-04_l4_ours_vs_baseline.png")

    page = f"""<!doctype html>
<meta charset="utf-8"><title>L1／L4 人眼比對頁</title><style>{CSS}</style>
<h1>L1／L4 人眼比對頁</h1>
<div class="note">
本頁只引用 <code>runs/</code> 底下真實產生的 PNG，沒有示意圖。點圖可開原尺寸。
每一格的數字取自該 run 自己的 <code>summary.csv</code>。
</div>

<h2>L1：三個攻擊在 κ = 0.06 上的破壞型態</h2>
<div class="note">
三個基準方法的編輯 LPIPS 是 pg_encoder 0.6363、pg_diffusion 0.6070、
semantic 0.5593，即該判準把 semantic 排最後。<b>看圖不是這樣排的</b>：
三者的破壞是三種不同的東西，單一個距離純量分不出「編輯失敗」與
「編輯成功但輸出品質下降」。這正是 LEDGER 1.10 量到的
ρ(距離, 劣化) = −0.207 的直接展示。
</div>
<table>{l1_section(l1_imgs)}</table>

<h2>L4：本專案的兩個位置，走 L1 的同一條評測路徑</h2>
<div class="note">
τ_lpips = 0.10。<b>site S 的三格全部跑滿 150 步上限</b>，依 LEDGER 6.4
不可用於跨 site 比較；site PF 只有 man_00（48 步）與 man_03（122 步）收斂。
最右欄是基準的 pg_encoder，放在同一列供對照——注意它的失真預算是本條件的
5.4 倍（擾動 LPIPS 0.55 對 0.10），<b>兩者不在同一條軸上</b>。
</div>
<table>{l4_section(l4_imgs)}</table>
"""
    (out / "compare.html").write_text(page, encoding="utf-8")
    print(f"寫出 {out/'compare.html'}")
    print(f"寫出 {out/'2026-08-04_l1_three_attacks.png'}")
    print(f"寫出 {out/'2026-08-04_l4_ours_vs_baseline.png'}")


if __name__ == "__main__":
    main()
