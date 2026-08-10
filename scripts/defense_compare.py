#!/usr/bin/env python
"""防禦是否成功的人眼判定頁：把**未防禦的編輯**放在每一列的第一格。

    python scripts/defense_compare.py --out runs/defense_compare.html \
        --batch apa+A=runs/s3t20_r_merged --batch 其餘=runs/s3t20_merged

## 為什麼要這一頁

`compare.html` 是逐格的報告頁，它答的是「這一格產生了什麼」。使用者要判的
是另一個問題：**編輯有沒有失敗**。那個判定需要一個對照——未防禦的同一張圖、
同一個 prompt、同一個 seed 編出來是什麼樣子。沒有它，「背景扭曲」「出現不自然
物體」都無從說起，因為 stock SD 本來就會改動背景。

故本頁的版面是固定的：每一列最左邊是 `control`（φ=0，未防禦），其右依序是
各條件在**同一個淨化算子、同一個 seed** 下的編輯結果。跨列變的只有淨化算子。

## 兩個刻意的選擇

**條件可以來自不同批次。** `apa+A` 在 `s3t20_r_merged`、三個加性 baseline 在
`s3t20_merged`，兩批的影像、prompt、seed、τ 與淨化設定逐字相同，差別只有
A 段。硬要在同一個批次目錄裡找齊，等於為了排版重跑兩小時的 baseline。

**影像以 JPEG q=92 內嵌。** 全部用 PNG 會讓這一頁超過 30 MB 而送不出去。
q=92 在 512² 上不會產生可與「防禦造成的失真」混淆的塊狀假影，但**銳利度的
細微差別不要在這一頁上判**——那要看 `compare.html` 的原始 PNG。
"""

import argparse
import base64
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src.experiment.executors import purify_dir_name  # noqa: E402

# 淨化算子：identity 是主表，其餘四個是使用者關心的「淨化過後還剩多少」。
# 取兩端與中間各一，而不是全部 18 個——那會是 400 張圖。
PURIFIERS = [("identity", 0.0), ("blur", 3.0), ("jpeg", 30.0),
             ("quantize", 64.0), ("diffpure", 150.0)]
IMAGES = ["horse_00", "horse_03", "woman_03"]
TAU = "0.04"


def b64(path: Path, quality: int = 92) -> str:
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def rows_of(batch: Path):
    """grid.csv 的 eval 列，鍵為 (條件, 影像, 淨化目錄名, seed)。"""
    out = {}
    with (batch / "grid.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["stage"] != "eval":
                continue
            key = (r["condition"], r["image_id"],
                   purify_dir_name(r["purify_kind"], float(r["purify_strength"])),
                   int(r["seed"]))
            out[key] = r
    return out


def cell(title: str, path: Path, note: str) -> str:
    if not path.exists():
        return f'<td><div class="miss">缺 {path.name}</div><div class="t">{title}</div></td>'
    return (f'<td><img src="data:image/jpeg;base64,{b64(path)}">'
            f'<div class="t">{title}</div><div class="n">{note}</div></td>')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, required=True,
                    help="帶 A 段的批次目錄（apa+A／Ra+A 由此取）")
    ap.add_argument("--old", type=Path, required=True,
                    help="不帶 A 段的批次目錄（control 與三個 baseline 由此取）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    grid_new, grid_old = rows_of(a.new), rows_of(a.old)
    # (顯示名, 批次目錄, grid, 條件名)。順序即欄序。
    cols = [
        ("apa（無 A 段）", a.old, grid_old, "apa"),
        ("apa + A 段", a.new, grid_new, "apa"),
        ("Ra + A 段（隨機對照）", a.new, grid_new, "Ra"),
        ("PhotoGuard-c", a.old, grid_old, "photoguard_c"),
        ("Mist", a.old, grid_old, "mist"),
        ("DIA-R", a.old, grid_old, "dia_r"),
    ]

    h = ["<title>防禦是否成功 · 未防禦編輯為對照</title>", """<style>
body{font:14px/1.5 system-ui,sans-serif;margin:24px;background:#fff;color:#111}
table{border-collapse:collapse;margin:8px 0 28px}
td,th{border:1px solid #ddd;padding:4px;vertical-align:top;text-align:center}
img{width:250px;height:250px;object-fit:contain;display:block}
.t{font-size:11px;color:#333;margin-top:3px}
.n{font-size:11px;color:#777;font-family:ui-monospace,monospace}
.miss{width:250px;height:250px;display:flex;align-items:center;
      justify-content:center;color:#b00;font-size:12px;background:#faf0f0}
th{background:#f5f5f5;font-size:12px;position:sticky;top:0}
h2{margin:32px 0 4px}h3{margin:20px 0 4px;color:#444;font-weight:600}
.ctrl{background:#fffbe6}
p.note{color:#555;max-width:1100px}
</style>"""]
    h.append("<h1>防禦是否成功：以未防禦的編輯為對照</h1>")
    h.append(f"""<p class="note">每一列最左邊的<b>未防禦編輯</b>是同一張圖、同一個
prompt、同一個 seed（seed{a.seed}）在<b>沒有任何防禦</b>下被編輯的結果，它是
「編輯有沒有失敗」唯一的判準基線。往右每一格是各條件在<b>同一個淨化算子</b>下的
編輯結果。跨列變動的只有淨化算子。<br>
數字：<code>edit_lpips</code> 是該格對未防禦編輯的距離（越大代表被推得越遠），
括號內是它對未防禦編輯的 CLIP 對齊差。防禦圖那一段標的是對<b>原圖</b>的失真。<br>
影像以 JPEG q=92 內嵌以控制檔案大小；<b>銳利度的細微差別請回去看
compare.html 的原始 PNG</b>。</p>""")

    for img in IMAGES:
        prompt = next((r["prompt"] for (c, i, *_), r in grid_new.items()
                       if i == img), "?")
        h.append(f'<h2>{img} · 攻擊 prompt「{prompt}」</h2>')

        # ---- 防禦圖本身（保真）----
        h.append("<h3>防禦圖 x_def（對原圖的失真，Δ=0.04 相對 DISTS）</h3><table><tr>")
        h.append(cell("原圖", a.new / "control" / img / "orig.png"
                      if (a.new / "control" / img / "orig.png").exists()
                      else a.new / "apa" / img / "orig.png", ""))
        for name, batch, grid, cond in cols:
            r = grid.get((cond, img, "identity_0", a.seed))
            note = ("lpips {:.3f}  dists {:.3f}  銳利 {:.2f}".format(
                float(r["fid_lpips"]), float(r["fid_dists"]),
                float(r["fid_acutance_ratio"])) if r else "")
            h.append(cell(name, batch / cond / img / f"x_def_tau{TAU}.png", note))
        h.append("</tr></table>")

        # ---- 編輯結果，逐淨化算子一列 ----
        h.append("<h3>編輯結果</h3><table>")
        h.append("<tr><th>淨化</th><th>未防禦（對照）</th>"
                 + "".join(f"<th>{n}</th>" for n, *_ in cols) + "</tr>")
        for kind, strength in PURIFIERS:
            d = purify_dir_name(kind, strength)
            label = "不淨化" if kind == "identity" else f"{kind} {strength:g}"
            h.append(f'<tr><td class="ctrl"><b>{label}</b></td>')
            h.append(cell("未防禦 φ=0",
                          a.old / "control" / img / "purify" / d
                          / f"edit_seed{a.seed}.png", "基線"))
            for name, batch, grid, cond in cols:
                r = grid.get((cond, img, d, a.seed))
                note = ("edit_lpips {:.3f}  (ΔCLIP {:+.3f})".format(
                    float(r["edit_lpips"]), float(r["effect_clip"]))
                    if r else "")
                h.append(cell(name, batch / cond / img / "purify" / d
                              / f"edit_tau{TAU}_seed{a.seed}.png", note))
            h.append("</tr>")
        h.append("</table>")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(h), encoding="utf-8")
    print(f"寫入 {a.out}（{a.out.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
