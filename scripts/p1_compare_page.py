"""P1 的比對頁 —— 讓人眼判定「等 LPIPS 的模糊與雜訊，哪個比較明顯」。

指標之間出現矛盾時由人眼定調，這是本專案既有的作法（E18 即靠此發現銳利度
盲區）。頁面的設計依同一組要求：

- 原地切換，不並排。並排會讓注意力花在移動視線上，敏感度差很多。
- 附 4× 放大裁切，位置由原圖的梯度能量自動選（差異在高頻處最明顯），
  三個版本取同一座標，放大用最近鄰以免插值本身造成模糊。
- 影像的環境底色在明暗主題下都固定為同一階中性灰。環境亮度會改變感知對比，
  隨主題變動會使不同時間的判斷不可比。
- 指標預設收起。先看圖再展開數字，否則就不是在測眼睛。

輸出 `runs/p1_iso_lpips_probe/compare.html`，以相對路徑引用同目錄的 PNG。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.acutance import _KX, _KY, _luma

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "p1_iso_lpips_probe"

CROP = 128          # 裁切邊長（原圖像素）
ZOOM = 4            # 放大倍率，最近鄰
BASE_VARIANTS = [("orig", "原圖"), ("blur", "模糊"), ("noise", "雜訊")]
# 兩個空間變形位置只在 τ=0.05 產生（見 scripts/p1b_warp_arm.py 的範圍說明）
WARP_VARIANTS = [("warp_bilinear", "變形-雙線性"), ("warp_bicubic", "變形-雙三次")]
WARP_TAU = 0.05

# 展開後要顯示的指標，順序即閱讀順序：先是被固定住的 LPIPS，再是候選項。
SHOW = ["lpips", "acutance_ratio", "local_acutance_dev", "local_acutance_signed",
        "stlpips", "musiq", "gmsd", "nlpd", "vif_p", "haarpsi", "dists",
        "ms_ssim", "ssim", "psnr", "niqe"]


def to_tensor(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def pick_crop(orig: Path) -> tuple[int, int]:
    """選梯度能量最高的 CROP×CROP 視窗，回傳左上角座標。

    以積分影像做，避免對每個候選視窗重算。步長取 CROP//2，足以定位到
    高頻區而不必逐像素搜尋。
    """
    import torch.nn.functional as F

    # `acutance.gradient_energy` 回傳整張圖的純量，此處需要逐像素的梯度平方
    # 才能挑視窗，故直接用同一組 Sobel 核重算，確保與銳利度指標同一定義。
    x = to_tensor(orig)
    y = _luma(x)
    y = F.pad(y, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(y, _KX.view(1, 1, 3, 3))
    gy = F.conv2d(y, _KY.view(1, 1, 3, 3))
    g = (gx.pow(2) + gy.pow(2))[0, 0]

    H, W = g.shape
    step = CROP // 2
    best, best_xy = -1.0, (0, 0)
    for i in range(0, H - CROP + 1, step):
        for j in range(0, W - CROP + 1, step):
            s = float(g[i:i + CROP, j:j + CROP].sum())
            if s > best:
                best, best_xy = s, (i, j)
    return best_xy


def make_crop(src: Path, dst: Path, top: int, left: int) -> None:
    im = Image.open(src).convert("RGB").crop((left, top, left + CROP, top + CROP))
    im.resize((CROP * ZOOM, CROP * ZOOM), Image.NEAREST).save(dst)


CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, "Noto Sans TC", sans-serif; margin: 0; padding: 24px;
       line-height: 1.6; }
h1 { font-size: 1.4rem; margin: 0 0 4px; }
.note { opacity: .75; font-size: .9rem; max-width: 62ch; margin: 0 0 24px; }
.panel { margin: 0 0 40px; border-top: 1px solid rgba(128,128,128,.35); padding-top: 16px; }
.hdr { display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap;
       margin-bottom: 10px; }
.hdr b { font-size: 1.05rem; }
.hdr span { opacity: .7; font-size: .85rem; }
.btns { display: flex; gap: 6px; margin-bottom: 10px; }
button { font: inherit; padding: 4px 14px; border: 1px solid rgba(128,128,128,.5);
         background: transparent; color: inherit; border-radius: 4px; cursor: pointer; }
button[aria-pressed="true"] { background: rgba(128,128,128,.35); font-weight: 600; }
/* 環境底色固定為同一階中性灰，不隨明暗主題變動：環境亮度會改變感知對比 */
.stage { display: flex; gap: 12px; flex-wrap: wrap; background: #808080;
         padding: 12px; border-radius: 6px; width: max-content; max-width: 100%; }
.stage img { display: block; max-width: 100%; height: auto; image-rendering: pixelated; }
.frame { position: relative; }
.frame img { position: absolute; inset: 0; opacity: 0; }
.frame img.on { position: static; opacity: 1; }
details { margin-top: 12px; }
summary { cursor: pointer; font-size: .9rem; opacity: .8; }
table { border-collapse: collapse; font-size: .85rem; margin-top: 8px; }
th, td { border: 1px solid rgba(128,128,128,.35); padding: 3px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
"""

JS = """
document.querySelectorAll('.panel').forEach(p => {
  const show = v => {
    p.querySelectorAll('.frame img').forEach(i =>
      i.classList.toggle('on', i.dataset.v === v));
    p.querySelectorAll('button').forEach(b =>
      b.setAttribute('aria-pressed', b.dataset.v === v));
  };
  p.querySelectorAll('button').forEach(b =>
    b.addEventListener('click', () => show(b.dataset.v)));
  p.addEventListener('mouseenter', () => p.dataset.hot = '1');
  p.addEventListener('mouseleave', () => delete p.dataset.hot);
});
// 數字鍵切換滑鼠所在面板的第 n 個版本：切換要快，移到按鈕上再點會打斷比對
document.addEventListener('keydown', e => {
  const i = parseInt(e.key, 10);
  if (!(i >= 1 && i <= 9)) return;
  const p = document.querySelector('.panel[data-hot="1"]');
  if (!p) return;
  const b = p.querySelectorAll('button')[i - 1];
  if (b) b.click();
});
"""


def collect_values() -> dict:
    """把三個來源的指標值收成 (影像, τ, 條件, 指標) → 值。

    - `probe.csv`：模糊與雜訊兩個條件的全部指標（三個 τ）
    - `warp.csv`：兩個空間變形位置的全部指標（僅 τ=0.05）
    - `p3_local_acutance/probe_arms.csv`：四個條件的局部銳利度偏差（僅 τ=0.05）
    """
    vals: dict = {}
    for r in csv.DictReader((OUT / "probe.csv").open(encoding="utf-8")):
        tgt = float(r["target_lpips"])
        for arm in ("blur", "noise"):
            for k, v in r.items():
                if k.startswith(f"{arm}_") and not k.endswith("_orig"):
                    vals[(r["image"], tgt, arm, k[len(arm) + 1:])] = float(v)

    wp = OUT / "warp.csv"
    if wp.exists():
        for r in csv.DictReader(wp.open(encoding="utf-8")):
            for arm in ("bilinear", "bicubic"):
                for k, v in r.items():
                    if k.startswith(f"{arm}_") and not k.endswith("_orig"):
                        vals[(r["image"], WARP_TAU, f"warp_{arm}",
                              k[len(arm) + 1:])] = float(v)

    lp = ROOT / "runs" / "p3_local_acutance" / "probe_arms.csv"
    if lp.exists():
        for r in csv.DictReader(lp.open(encoding="utf-8")):
            for k in ("local_acutance_dev", "local_acutance_signed",
                      "local_acutance_worst"):
                vals[(r["image"], WARP_TAU, r["arm"], k)] = float(r[k])
    return vals


def main() -> None:
    rows = list(csv.DictReader((OUT / "probe.csv").open(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError("probe.csv 是空的，請先跑 scripts/p1_iso_lpips_probe.py")
    values = collect_values()

    parts = []
    crops: dict[str, tuple[int, int]] = {}
    for r in rows:
        name, tgt = r["image"], float(r["target_lpips"])
        tag = f"{tgt:.2f}".replace(".", "p")
        if name not in crops:
            crops[name] = pick_crop(OUT / f"{name}__orig.png")
        top, left = crops[name]

        variants = list(BASE_VARIANTS)
        if abs(tgt - WARP_TAU) < 1e-9:
            variants += WARP_VARIANTS

        files = {"orig": f"{name}__orig.png"}
        for v, _ in variants[1:]:
            files[v] = f"{name}__lpips{tag}__{v}.png"
        missing = [f for f in files.values() if not (OUT / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"缺少 {missing}。τ={tgt} 的全部條件必須齊備；變形兩個條件由 "
                "scripts/p1b_warp_arm.py 產生")

        zooms = {}
        for v, f in files.items():
            z = f"zoom__{name}__lpips{tag}__{v}.png"
            make_crop(OUT / f, OUT / z, top, left)
            zooms[v] = z

        full = "".join(
            f'<img data-v="{v}" class="{"on" if v == "orig" else ""}" '
            f'src="{files[v]}" width="512" alt="{lab}">' for v, lab in variants)
        zoom = "".join(
            f'<img data-v="{v}" class="{"on" if v == "orig" else ""}" '
            f'src="{zooms[v]}" width="{CROP * ZOOM}" alt="{lab} 放大">'
            for v, lab in variants)
        btns = "".join(
            f'<button data-v="{v}" aria-pressed="{str(v == "orig").lower()}">'
            f'{lab}</button>' for v, lab in variants)

        cols = [v for v, _ in variants[1:]]
        head = "".join(f"<th>{lab}</th>" for _, lab in variants[1:])
        tbl = [f"<table><tr><th>指標</th>{head}</tr>"]
        for k in SHOW:
            cells = [values.get((name, tgt, c, k)) for c in cols]
            if any(v is None for v in cells):
                continue
            tbl.append(f"<tr><td>{k}</td>"
                       + "".join(f"<td>{v:.4f}</td>" for v in cells) + "</tr>")
        tbl.append("</table>")

        parts.append(f"""
<div class="panel">
  <div class="hdr"><b>{name}</b>
    <span>目標 LPIPS {tgt:.2f}　·　實測 模糊 {float(r['lpips_blur']):.4f} /
    雜訊 {float(r['lpips_noise']):.4f}</span>
    <span>σ = {float(r['sigma']):.3f}　·　雜訊振幅 = {float(r['noise_amp']):.4f}</span>
  </div>
  <div class="btns">{btns}</div>
  <div class="stage">
    <div class="frame">{full}</div>
    <div class="frame">{zoom}</div>
  </div>
  <details><summary>展開指標數值（先看圖）</summary>{''.join(tbl)}</details>
</div>""")

    html = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>P1 比對頁：等 LPIPS 的模糊 vs 雜訊</title>
<style>{CSS}</style></head><body>
<h1>等 LPIPS 的模糊 vs 雜訊</h1>
<p class="note">同一組裡的每個版本都已校準到<b>相同的 LPIPS</b>。若 LPIPS 真的
代表人眼可辨失真，它們應該一樣明顯。請按<b>數字鍵</b>（或點按鈕）在滑鼠所在
的面板上原地切換，判斷是否同樣明顯。τ=0.05 的組另有兩個<b>空間變形</b>版本，
差別只在重取樣是雙線性還是雙三次——量測說兩者的銳利度差 15 個百分點
（85.0% vs 99.9%），請確認人眼是否同意，以及 0.4–0.6 px 的位移本身是否可見。
右側為 4× 最近鄰放大，位置取原圖梯度能量最高處，各版本同一座標。
指標數值預設收起。</p>
{''.join(parts)}
<script>{JS}</script></body></html>"""
    (OUT / "compare.html").write_text(html, encoding="utf-8")
    print(f"已寫出 {OUT / 'compare.html'}（{len(rows)} 組）")


if __name__ == "__main__":
    main()
