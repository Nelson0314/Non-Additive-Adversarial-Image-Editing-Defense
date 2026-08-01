"""E25-3 比對頁 —— 讓人眼判定「編輯到底有沒有被擋下來」。

E25-1 的量測與既有結論互相矛盾：`net_lpips` 說防禦有效（0.08–0.12），
SigLIP 說編輯對 prompt 的服從程度沒有下降（726 格語意失敗 0 格）。
本專案的既定作法是指標矛盾時把影像做成比對頁由人眼定調，不用數字自行
定調（E18 的銳利度盲區、E20 的四臂判別都是這樣定案的）。

頁面沿用 `scripts/p1_compare_page.py` 的設計要求：原地切換而非並排、4× 最近鄰
放大裁切、環境底色固定中性灰、指標預設收起。

五個版本的用意各不相同，切換的方式也不同：

| 切換 | 要判斷的事 |
|---|---|
| 原圖 ↔ 防禦圖 | 防禦本身看不看得出來（保真度） |
| 未防禦編輯 ↔ 防禦後編輯 | 編輯有沒有被擋下來（本頁的主要問題） |
| 防禦後編輯 ↔ 防禦後編輯（JPEG 30） | 淨化把防禦洗掉多少 |

判斷「編輯有沒有被擋下來」時要看的是編輯的內容有沒有照 prompt 發生，
不是兩張圖長不長得一樣。兩張擴散輸出即使 prompt 都達成了也本來就會不一樣
（噪聲不同、路徑不同），那個差異會被 `net_lpips` 全部記成防禦效果。

輸出 `runs/p5_semantic_axis/compare.html`。
"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.acutance import _KX, _KY, _luma

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "p5_semantic_axis"

CROP = 128
ZOOM = 4

# 只收入對等比較用得上的 run：E23 是唯一兩臂都被同一道約束綁住且步數足夠的
# 配對（見 docs/RESULTS_E21-E22.md §5），E21 τ=0.05 是它 25 步的對照。
PAIRS = [
    ("E23 τ=0.05、100 步", "e23_Sbic_s100_tau0.05", "e23_P_s100_tau0.05"),
    ("E21 τ=0.05、25 步", "e21_Sbic_tau0.05", "e21_P_tau0.05"),
]

VARIANTS = [
    ("orig", "原圖", "orig.png"),
    ("defended", "防禦圖", "defended.png"),
    ("edit_orig", "未防禦編輯", "edit_orig.png"),
    ("edit_def", "防禦後編輯", "edit_def_blur_0.0.png"),
    ("edit_def_jpeg", "防禦後編輯 + JPEG30", "edit_def_jpeg_30.png"),
]

# 版面與互動與 P1 完全共用。抄一份會讓兩頁在往後的調整中漂移，而兩頁的
# 判讀結果必須可比（同樣的底色、同樣的放大方式、同樣的鍵盤切換）。
sys.path.insert(0, str(ROOT / "scripts"))
from p1_compare_page import CSS, JS  # noqa: E402


def to_tensor(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def pick_crop(orig: Path) -> tuple:
    """選原圖梯度能量最高的 CROP×CROP 視窗，與 p1 的作法一致。"""
    import torch.nn.functional as F

    y = _luma(to_tensor(orig))
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


def cell_rows(run: str):
    """{cell 目錄名: {(算子, 強度): 該列}}，只取未見種子那一批。"""
    out = {}
    path = ROOT / "runs" / run / "results.csv"
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["noise_split"] != "heldout":
            continue
        cell = f"{r['image']}__{r['site']}__r{r['rank']}"
        out.setdefault(cell, {})[(r["purify"], float(r["strength"]))] = r
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels, crops = [], {}

    for label, run_s, run_p in PAIRS:
        for run in (run_s, run_p):
            rows = cell_rows(run)
            for cell in sorted(rows):
                src = ROOT / "runs" / run / cell
                base = rows[cell][("blur", 0.0)]
                jp = rows[cell][("jpeg", 30.0)]
                image = base["image"]

                missing = [f for _, _, f in VARIANTS if not (src / f).exists()]
                if missing:
                    raise FileNotFoundError(
                        f"{run}/{cell} 缺少 {missing}。五個版本必須齊備，"
                        "缺一個就無法在同一個面板內原地切換")

                if image not in crops:
                    crops[image] = pick_crop(src / "orig.png")
                top, left = crops[image]

                tag = f"{run}__{cell}"
                files, zooms = {}, {}
                for key, _, fname in VARIANTS:
                    dst = OUT / f"{tag}__{key}.png"
                    shutil.copyfile(src / fname, dst)
                    files[key] = dst.name
                    z = OUT / f"zoom__{tag}__{key}.png"
                    make_crop(src / fname, z, top, left)
                    zooms[key] = z.name

                full = "".join(
                    f'<img data-v="{k}" class="{"on" if k == "orig" else ""}" '
                    f'src="{files[k]}" width="512" alt="{lab}">'
                    for k, lab, _ in VARIANTS)
                zoom = "".join(
                    f'<img data-v="{k}" class="{"on" if k == "orig" else ""}" '
                    f'src="{zooms[k]}" width="{CROP * ZOOM}" alt="{lab} 放大">'
                    for k, lab, _ in VARIANTS)
                btns = "".join(
                    f'<button data-v="{k}" aria-pressed="{str(k == "orig").lower()}">'
                    f'{lab}</button>' for k, lab, _ in VARIANTS)

                d_sig = float(base["edit_siglip_b"]) - float(base["edit_siglip_a"])
                d_clip = float(base["edit_clip_b"]) - float(base["edit_clip_a"])
                tbl = [
                    "<table><tr><th>量</th><th>無淨化</th><th>JPEG 30</th></tr>",
                    _tr("防禦圖對原圖 LPIPS（保真）", base["defimg_lpips"], jp["defimg_lpips"]),
                    _tr("防禦圖對原圖 銳利度比", base["defimg_acutance_ratio"],
                        jp["defimg_acutance_ratio"]),
                    _tr("編輯結果的 LPIPS 位移", base["edit_lpips"], jp["edit_lpips"]),
                    _tr("net_lpips（扣掉對照）", base["net_lpips"], jp["net_lpips"]),
                    _tr("ΔSigLIP（負才是防禦）", d_sig,
                        float(jp["edit_siglip_b"]) - float(jp["edit_siglip_a"])),
                    _tr("ΔCLIP（未通過對照，僅列）", d_clip,
                        float(jp["edit_clip_b"]) - float(jp["edit_clip_a"])),
                    "</table>",
                ]

                panels.append(f"""
<div class="panel">
  <div class="hdr"><b>{image} · {base['site']}</b>
    <span>{label}　·　{run}</span>
    <span>prompt：{base['prompt']}</span>
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
<title>E25 比對頁：編輯有沒有被擋下來</title>
<style>{CSS}</style></head><body>
<h1>編輯有沒有被擋下來</h1>
<p class="note">量測互相矛盾：<b>net_lpips</b> 說防禦有效（0.08–0.12），
<b>SigLIP</b> 說編輯對 prompt 的服從程度沒有下降（726 格中語意失敗 0 格）。
請按<b>數字鍵</b>（或點按鈕）在滑鼠所在的面板上原地切換。三種切換各問一件事：
<b>1↔2</b> 防禦本身看不看得出來；<b>3↔4</b> 編輯有沒有被擋下來（本頁的主要
問題）；<b>4↔5</b> JPEG 把防禦洗掉多少。判斷第二項時請看<b>編輯的內容有沒有
照 prompt 發生</b>，而不是兩張圖長不長得一樣——兩次擴散輸出即使都達成了
prompt 也本來就會不同，那個差異會被 net_lpips 全部記成防禦效果。
右側為 4× 最近鄰放大，位置取原圖梯度能量最高處，各版本同一座標。</p>
{''.join(panels)}
<script>{JS}</script></body></html>"""
    (OUT / "compare.html").write_text(html, encoding="utf-8")
    print(f"已寫出 {OUT / 'compare.html'}（{len(panels)} 組）")


def _tr(name: str, a, b) -> str:
    return (f"<tr><td>{name}</td><td>{float(a):.4f}</td>"
            f"<td>{float(b):.4f}</td></tr>")


if __name__ == "__main__":
    main()
