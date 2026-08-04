"""E27 校準的人眼比對頁 —— 新的可行域裡，防禦圖看起來還可以接受嗎？

為什麼一定要這一頁。E27 把 `alpha_lpips` 設為 0 之後，τ 以內的失真對
最佳化完全免費，於是它會把預算用滿。問題是「用滿 LPIPS 0.05 的預算」在
site C 上長什麼樣子沒有人看過：實測 `L_fid = 0.0000`（三道 hinge 全部未
啟動）的同時 `PSNR = 24.75 dB`、`L∞ = 0.62`——像素層級的改變非常大，而
LPIPS 與鈍化都不收費。

這正是 `src/residual/site_color.py` 的 §5.2(1) 事先寫下的風險：現行約束集
（LPIPS ∩ 鈍化）對 site C 的特徵失真——色偏、色度串音、飽和區假色——是否
收費從未量測。鈍化約束在構造上就對它盲目（它只看亮度），若 LPIPS 也
盲目，那就是重蹈 site S 用 LPIPS 換到的是模糊的覆轍，只是換成用由色度換來。

沿用 `scripts/p1_compare_page.py` 的設計要求：原地切換、4× 最近鄰放大裁切、
環境底色固定中性灰、指標預設收起。

切換的問題只有一個：這張防禦圖，你能接受它跟原圖是「同一張照片」嗎。

輸出 `runs/<第一個 run>/compare.html`。
"""

import argparse
import csv
import json
import shutil
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
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None, help="輸出目錄，預設為第一個 run")
    args = ap.parse_args()

    run_dirs = [ROOT / r if not Path(r).is_absolute() else Path(r) for r in args.runs]
    out = Path(args.out) if args.out else run_dirs[0]
    out.mkdir(parents=True, exist_ok=True)

    panels, crops = [], {}
    for rd in run_dirs:
        env = json.loads((rd / "env.json").read_text(encoding="utf-8"))
        loss = env.get("loss", {})
        summary = {r["image"]: r
                   for r in csv.DictReader((rd / "summary.csv").open(encoding="utf-8"))}
        for cell in sorted(p for p in rd.iterdir() if p.is_dir()):
            image = cell.name.split("__")[0]
            files = {"orig": cell / "orig.png", "defended": cell / "defended.png"}
            if not all(f.exists() for f in files.values()):
                continue
            if image not in crops:
                crops[image] = pick_crop(files["orig"])
            top, left = crops[image]

            tag = f"{rd.name}__{cell.name}"
            names, zooms = {}, {}
            for k, src in files.items():
                dst = out / f"{tag}__{k}.png"
                shutil.copyfile(src, dst)
                names[k] = dst.name
                z = out / f"zoom__{tag}__{k}.png"
                make_crop(src, z, top, left)
                zooms[k] = z.name

            variants = [("orig", "原圖"), ("defended", "防禦圖")]
            full = "".join(
                f'<img data-v="{k}" class="{"on" if k == "orig" else ""}" '
                f'src="{names[k]}" width="512" alt="{lab}">' for k, lab in variants)
            zoom = "".join(
                f'<img data-v="{k}" class="{"on" if k == "orig" else ""}" '
                f'src="{zooms[k]}" width="{CROP * ZOOM}" alt="{lab} 放大">'
                for k, lab in variants)
            btns = "".join(
                f'<button data-v="{k}" aria-pressed="{str(k == "orig").lower()}">'
                f'{lab}</button>' for k, lab in variants)

            s = summary.get(image, {})
            rows = [
                ("LPIPS（受 τ 約束）", s.get("final_lpips"), loss.get("tau_lpips")),
                ("鈍化偏差（受 τ_acut 約束）", s.get("final_ssim"), loss.get("tau_acut")),
                ("PSNR (dB)（不在梯度裡）", s.get("final_psnr_total"), None),
                ("L∞（beta_linf=0，不在梯度裡）", s.get("final_linf_total"), None),
                ("編輯偏移 edit_shift", s.get("final_shift"), loss.get("margin")),
                ("實際步數", s.get("steps_done"), s.get("steps")),
            ]
            tbl = ["<table><tr><th>量</th><th>實測</th><th>門檻／上限</th></tr>"]
            for name, v, lim in rows:
                if v in (None, ""):
                    continue
                try:
                    vs = f"{float(v):.4f}"
                except (TypeError, ValueError):
                    vs = str(v)
                tbl.append(f"<tr><td>{name}</td><td>{vs}</td>"
                           f"<td>{'' if lim is None else lim}</td></tr>")
            tbl.append("</table>")

            panels.append(f"""
<div class="panel">
  <div class="hdr"><b>{image}</b>
    <span>{rd.name}　·　lr={env.get('lr')}　·　α_lpips={loss.get('alpha_lpips')}</span>
    <span>site {s.get('site', '?')}</span>
  </div>
  <div class="btns">{btns}</div>
  <div class="stage">
    <div class="frame">{full}</div>
    <div class="frame">{zoom}</div>
  </div>
  <details><summary>展開數值（先看圖）</summary>{''.join(tbl)}</details>
</div>""")

    if not panels:
        raise SystemExit("沒有找到任何 orig.png / defended.png")

    html = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E27 校準：把失真預算用滿之後長什麼樣</title>
<style>{CSS}</style></head><body>
<h1>把失真預算用滿之後，防禦圖長什麼樣</h1>
<p class="note">E27 把 <code>alpha_lpips</code> 設為 0，於是 τ 以內的失真對
最佳化完全免費，它會把預算用滿。<b>要判斷的只有一件事：這張防禦圖，你能接受
它跟原圖是「同一張照片」嗎。</b>請按<b>數字鍵</b>（或點按鈕）原地切換。
<br><br>
需要特別注意<b>色偏</b>：鈍化約束只看亮度，對色度變化在構造上就是盲的；
若 LPIPS 也對它盲目，那就是重蹈 site S 用 LPIPS 換到的是模糊的覆轍，只是換成用
由色度換來。實測有格子在三道 hinge 全部未啟動（L_fid = 0）的同時
PSNR 只有 24.75 dB、L∞ 達 0.62。
右側為 4× 最近鄰放大，位置取原圖梯度能量最高處，兩版本同一座標。</p>
{''.join(panels)}
<script>{JS}</script></body></html>"""
    (out / "compare.html").write_text(html, encoding="utf-8")
    print(f"已寫出 {out / 'compare.html'}（{len(panels)} 組）")


if __name__ == "__main__":
    sys.exit(main())
