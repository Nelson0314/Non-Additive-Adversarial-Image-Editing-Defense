"""E28 —— 色度偏壓的階梯：τ_chroma 要定在哪裡？

為什麼需要人眼定調。P9 選出了 `local_chroma_bias` 這把尺，但沒有定出
刻度。目前只有兩個由人眼判讀的點：

| 來源 | `local_chroma_bias` | 人眼判讀 |
|---|---|---|
| site P（E27 校準，n=2） | 0.30 – 0.33 | 看不出來 |
| site C（E27 校準，n=2） | 3.94 – 6.00 | 有色調偏移 |

中間是空的，而資料本身無法再收窄。更麻煩的是 `e23_P_s100`（100 步的加性
基準）在 car_01 上量到 1.01——加性擾動訓練久了也會產生連貫色偏，
所以 τ 若隨手設在 0.5 會把基準也擋掉，比較就不成立。

本腳本產生一組只有色度偏壓在變的階梯，由使用者指出從哪一級開始看得出
差別，τ 就定在該級之下。這是 E20 定 τ_acut 的同一套作法（那次由四個實測值
決定為 0.04），差別只在這次的夾點需要新的人眼判讀。

階梯用的是 site C 自己的參數化（`ColorResidual` 的粗網格 ΔM），而不是
任意的色偏，否則量到的可見度不對應真實的解。每一級以二分搜尋命中目標偏壓。

輸出 `runs/p10_chroma_ladder/{ladder.csv, compare.html}`。
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
from src.metrics.chroma import local_chroma_bias  # noqa: E402
from src.residual.site_color import ColorResidual  # noqa: E402
from src.utils.device import get_device  # noqa: E402

OUT = ROOT / "runs" / "p10_chroma_ladder"
CROP, ZOOM, GRID, SEED = 128, 4, 32, 20260728

# 級距涵蓋兩個已判讀定出的點（site P 0.30、site C 3.94）並在中間補滿。
LEVELS = [0.3, 0.6, 1.0, 1.5, 2.5, 4.0]

# 取三張色度含量不同的圖：p8 量到平均色度量值 person_00 0.0362、
# person_01 0.0752，相差兩倍，可見度很可能隨之不同。
IMAGES = ["car/car_00", "dog/dog_00", "person/person_01"]


def to_tensor(p: Path, device) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def make(x: torch.Tensor, scale: float) -> torch.Tensor:
    """以 site C 自己的參數化產生一個平滑的色度偏壓場。"""
    mod = ColorResidual(size=x.shape[-1], grid_size=GRID, init_std=0.0,
                        max_dev=10.0).to(x.device)
    g = torch.Generator().manual_seed(SEED)
    mod.delta.data = torch.randn(mod.delta.shape, generator=g).to(x.device) * scale
    with torch.no_grad():
        return mod.pixel_residual(x).clamp(0, 1)


def calibrate(x: torch.Tensor, target: float, hi: float = 4.0,
              iters: int = 26) -> float:
    """二分搜尋 ΔM 的尺度，使 `local_chroma_bias` 命中目標。

    先確認上界真的越過目標——上界不足時二分會回傳上界，產生一級
    「標示 4.0 但其實只有 1.2」的階梯，整組判讀即失效。
    """
    with torch.no_grad():
        top = float(local_chroma_bias(x, make(x, hi)))
    if top < target:
        raise ValueError(
            f"尺度上界 {hi} 只能達到偏壓 {top:.3f}，未及目標 {target}。"
            "請提高上界，不可回傳該值"
        )
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        with torch.no_grad():
            v = float(local_chroma_bias(x, make(x, mid)))
        if v < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def pick_crop(x: torch.Tensor) -> tuple:
    import torch.nn.functional as F

    y = _luma(x.cpu())
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


def save(x: torch.Tensor, path: Path) -> None:
    a = (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(a).save(path)


def main() -> None:
    import piq

    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    lpips = piq.LPIPS().to(device)
    print(f"[p10] device={device}", flush=True)

    rows, panels = [], []
    for rel in IMAGES:
        path = ROOT / "data/dayn_testset" / f"{rel}.png"
        name = path.stem
        x = to_tensor(path, device)
        top, left = pick_crop(x)
        save(x, OUT / f"{name}__orig.png")

        variants = [("orig", "原圖")]
        print(f"\n[{name}]", flush=True)
        for lv in LEVELS:
            scale = calibrate(x, lv)
            y = make(x, scale)
            with torch.no_grad():
                bias = float(local_chroma_bias(x, y))
                lp = float(lpips(x, y))
            key = f"b{lv}".replace(".", "p")
            save(y, OUT / f"{name}__{key}.png")
            rows.append({"image": name, "level": lv, "scale": scale,
                         "bias": bias, "lpips": lp})
            variants.append((key, f"偏壓 {lv}"))
            print(f"  目標 {lv:4.1f}  實得 {bias:6.3f}  LPIPS {lp:.4f}", flush=True)

        for key, _ in variants:
            src = OUT / f"{name}__{key}.png"
            im = Image.open(src).convert("RGB").crop(
                (left, top, left + CROP, top + CROP))
            im.resize((CROP * ZOOM, CROP * ZOOM), Image.NEAREST).save(
                OUT / f"zoom__{name}__{key}.png")

        full = "".join(
            f'<img data-v="{k}" class="{"on" if k == "orig" else ""}" '
            f'src="{name}__{k}.png" width="512" alt="{lab}">'
            for k, lab in variants)
        zoom = "".join(
            f'<img data-v="{k}" class="{"on" if k == "orig" else ""}" '
            f'src="zoom__{name}__{k}.png" width="{CROP * ZOOM}" alt="{lab}">'
            for k, lab in variants)
        btns = "".join(
            f'<button data-v="{k}" aria-pressed="{str(k == "orig").lower()}">'
            f'{lab}</button>' for k, lab in variants)

        sub = [r for r in rows if r["image"] == name]
        tbl = ["<table><tr><th>級</th><th>實得偏壓</th><th>LPIPS</th></tr>"]
        for r in sub:
            tbl.append(f"<tr><td>{r['level']}</td><td>{r['bias']:.3f}</td>"
                       f"<td>{r['lpips']:.4f}</td></tr>")
        tbl.append("</table>")

        panels.append(f"""
<div class="panel">
  <div class="hdr"><b>{name}</b>
    <span>只有色度偏壓在變，亮度與結構不動</span>
  </div>
  <div class="btns">{btns}</div>
  <div class="stage">
    <div class="frame">{full}</div>
    <div class="frame">{zoom}</div>
  </div>
  <details><summary>展開數值（先看圖）</summary>{''.join(tbl)}</details>
</div>""")

    with (OUT / "ladder.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    html = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>色度偏壓的階梯：τ 要定在哪裡</title>
<style>{CSS}</style></head><body>
<h1>色度偏壓從哪一級開始看得出來</h1>
<p class="note">每一級<b>只有色度偏壓在變</b>，亮度與結構完全不動，用的是
site C 自己的參數化。請按<b>數字鍵</b>（或點按鈕）與<b>原圖</b>來回切換，
指出<b>從哪一級開始你看得出差別</b>。
<br><br>
兩個已知的錨點：site P 的防禦圖在偏壓 <b>0.30–0.33</b>（你判讀為看不出來），
site C 的在 <b>3.94–6.00</b>（你判讀為有色調偏移）。中間這幾級沒有人看過。
另外量到 100 步的加性基準在一張圖上達 <b>1.01</b>，所以門檻若定得比 1.0 低，
會連加性基準都擋掉、比較就不成立——這一級請特別看。
右側為 4× 最近鄰放大，位置取原圖梯度能量最高處，各級同一座標。</p>
{''.join(panels)}
<script>{JS}</script></body></html>"""
    (OUT / "compare.html").write_text(html, encoding="utf-8")
    print(f"\n已寫出 {OUT / 'compare.html'}（{len(panels)} 組 × {len(LEVELS)} 級）")


if __name__ == "__main__":
    main()
