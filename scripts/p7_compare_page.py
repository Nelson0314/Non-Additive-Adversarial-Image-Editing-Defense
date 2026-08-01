"""E26 比對頁 —— guidance scale 對「編輯有沒有發生」的影響。

沿用 `scripts/p1_compare_page.py` 的設計要求（原地切換、4× 最近鄰放大裁切、
環境底色固定中性灰、指標預設收起）。

切換的問題只有一個：**在哪一個 guidance 下，prompt 描述的東西真的出現了。**
w = 1.0 是本專案 E2–E23 全部實驗所用的設定；w = 7.5 是攻擊方實際會用的值。

輸出 `runs/p7_attack_sanity/compare.html`。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from p1_compare_page import CSS, JS  # noqa: E402
from src.metrics.acutance import _KX, _KY, _luma  # noqa: E402

OUT = ROOT / "runs" / "p7_attack_sanity"
CROP = 128
ZOOM = 4


def pick_crop(path: Path) -> tuple:
    import torch.nn.functional as F

    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    y = _luma(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0))
    y = F.pad(y, (1, 1, 1, 1), mode="replicate")
    g = (F.conv2d(y, _KX.view(1, 1, 3, 3)).pow(2)
         + F.conv2d(y, _KY.view(1, 1, 3, 3)).pow(2))[0, 0]
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


def main() -> None:
    rows = list(csv.DictReader((OUT / "probe.csv").open(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError("probe.csv 是空的，請先跑 scripts/p7_attack_sanity.py")

    by_image = {}
    for r in rows:
        by_image.setdefault(r["image"], []).append(r)

    panels = []
    for image, rs in by_image.items():
        orig = OUT / f"{image}__orig.png"
        top, left = pick_crop(orig)

        variants = [("orig", "原圖")]
        for r in rs:
            tag = f"w{r['guidance_scale']}_s{r['strength']}"
            variants.append((tag, r["label"]))

        files = {"orig": orig.name}
        for r in rs:
            tag = f"w{r['guidance_scale']}_s{r['strength']}"
            files[tag] = f"{image}__{tag}.png"
        missing = [f for f in files.values() if not (OUT / f).exists()]
        if missing:
            raise FileNotFoundError(f"缺少 {missing}；請重跑 p7_attack_sanity.py")

        zooms = {}
        for k, f in files.items():
            z = f"zoom__{image}__{k}.png"
            make_crop(OUT / f, OUT / z, top, left)
            zooms[k] = z

        full = "".join(
            f'<img data-v="{k}" class="{"on" if k == "orig" else ""}" '
            f'src="{files[k]}" width="512" alt="{lab}">' for k, lab in variants)
        zoom = "".join(
            f'<img data-v="{k}" class="{"on" if k == "orig" else ""}" '
            f'src="{zooms[k]}" width="{CROP * ZOOM}" alt="{lab} 放大">'
            for k, lab in variants)
        btns = "".join(
            f'<button data-v="{k}" aria-pressed="{str(k == "orig").lower()}">'
            f'{lab}</button>' for k, lab in variants)

        head = "".join(f"<th>{r['label']}</th>" for r in rs)
        tbl = [f"<table><tr><th>量</th>{head}</tr>"]
        for name, key, fmt in (
            ("Δclip（對 prompt 的對齊變化）", "d_clip", "{:+.4f}"),
            ("Δsiglip", "d_siglip", "{:+.4f}"),
            ("LPIPS→原圖", "lpips_to_orig", "{:.4f}"),
            ("PSNR→原圖 (dB)", "psnr_to_orig", "{:.2f}"),
        ):
            tbl.append(f"<tr><td>{name}</td>"
                       + "".join(f"<td>{fmt.format(float(r[key]))}</td>" for r in rs)
                       + "</tr>")
        tbl.append("</table>")

        panels.append(f"""
<div class="panel">
  <div class="hdr"><b>{image}</b>
    <span>prompt：{rs[0]['prompt']}</span>
    <span>原圖 clip {float(rs[0]['clip_orig']):.4f}</span>
  </div>
  <div class="btns">{btns}</div>
  <div class="stage">
    <div class="frame">{full}</div>
    <div class="frame">{zoom}</div>
  </div>
  <details><summary>展開數值（先看圖）</summary>{''.join(tbl)}</details>
</div>""")

    html = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E26：guidance scale 決定編輯有沒有發生</title>
<style>{CSS}</style></head><body>
<h1>guidance scale 決定編輯有沒有發生</h1>
<p class="note">全部是<b>未防禦</b>的原圖經 SDEdit 的結果，只差在
classifier-free guidance 的權重 w。<b>w = 1.0 是本專案 E2–E23 全部實驗所用的
設定</b>——`src/models/sd.py` 的 <code>_eps</code> 只以條件嵌入呼叫一次 UNet，
等同 w = 1；<b>w = 7.5 是攻擊方實際會用的值</b>。請按<b>數字鍵</b>（或點按鈕）
原地切換，判斷<b>在哪一個 w 下，prompt 描述的東西真的出現了</b>。
若只有 w ≥ 3 才看得出編輯，則 E2–E23 全部是在防禦一個不存在的攻擊，
那些實驗量到的 net_lpips 是兩次隨機去噪之間的漂移。
右側為 4× 最近鄰放大，位置取原圖梯度能量最高處，各版本同一座標。</p>
{''.join(panels)}
<script>{JS}</script></body></html>"""
    (OUT / "compare.html").write_text(html, encoding="utf-8")
    print(f"已寫出 {OUT / 'compare.html'}（{len(panels)} 組）")


if __name__ == "__main__":
    main()
