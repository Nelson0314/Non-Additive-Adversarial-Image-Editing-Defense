"""六格比對頁：原圖／防禦圖／淨化防禦圖／編輯原圖／編輯防禦圖／編輯淨化防禦圖。
**不跑 GPU，只讀已存下的圖與 CSV。**

為什麼要這一頁
────────────────────────────────────────────────────────────────────
淨增益量的是位移，而位移把「單純變糊變髒」與「模型把原圖丟掉重畫」算成同一
件事——人眼只認後者（`docs/GOAL.md` 的防禦成功判準）。實測過一次差距：同一
批上淨增益只差 1.14 倍，人眼判定是 5 擋下對 0（`runs/ip2p_deliver_jpeg`）。
**指標之間矛盾時以人眼裁定**，而人眼要能裁定就得看得到圖。

空白地板為什麼也要畫
────────────────────────────────────────────────────────────────────
淨化算子自己就會把編輯推開（模糊 σ1 的地板是 0.178、裁切 10% 是 0.505），
不畫它就無法排除「這一格看起來被擋下，其實只是算子自己把圖弄壞了」。
每一張影像的第一列固定是地板：沒有防禦，只有淨化。

面板的編碼
────────────────────────────────────────────────────────────────────
原生 512 不縮放，以 JPEG 內嵌（品質由 `--quality` 給，預設 90）。**這是為了
讓單一檔案帶得走而付的代價，要記在頁面上**：防禦擾動的 RMS 在 0.04–0.07、
L∞ 在 0.3–0.7，遠高於 q90 的量化雜訊，所以看得到的東西不會是編碼造成的；
但要逐像素驗證失真請回去看 `--src` 底下的原生 PNG，路徑印在每一列下方。

同一個檔案只編碼一次：面板走 CSS 背景，`.px<n>` 一個類別對應一個檔案，
原圖與編輯原圖在同一張影像的各列之間共用，不會重複膨脹。

用法
────────────────────────────────────────────────────────────────────
    python scripts/compare_sheet.py \\
        --src runs/ip2p_mainline --gallery runs/gallery_mainline \\
        --conditions ours_pg_m:phase_gain ours_pg_q:phase dct_aj85:dct_shield_y \\
        --purifier jpeg30 --out report_compare_jpeg30.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

FLOOR_COND = "None"          # 地板那一批的檔名裡條件欄是字串 None


# ---------------------------------------------------------------------------
# 圖：一個檔案編碼一次
# ---------------------------------------------------------------------------

class PanelPool:
    """把用到的圖編成 data URI，同一個路徑只編一次。"""

    def __init__(self, quality: int):
        self.quality = quality
        self._idx: Dict[str, int] = {}
        self._css: List[str] = []
        self.bytes_total = 0

    def cls(self, path: Optional[Path]) -> Optional[str]:
        if path is None or not path.exists():
            return None
        key = str(path)
        if key not in self._idx:
            with Image.open(path) as im:
                im = im.convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=self.quality,
                        subsampling=0, optimize=True)
            raw = buf.getvalue()
            self.bytes_total += len(raw)
            b64 = base64.b64encode(raw).decode("ascii")
            i = len(self._idx)
            self._idx[key] = i
            self._css.append(
                f".px{i}{{background-image:url(data:image/jpeg;base64,{b64})}}")
        return f"px{self._idx[key]}"

    def css(self) -> str:
        return "\n".join(self._css)


# ---------------------------------------------------------------------------
# 讀數
# ---------------------------------------------------------------------------

def read_defense(src: Path, tag: str) -> Dict[str, dict]:
    """逐圖的失真與未淨化位移。找不到就回空的——頁面照畫，數字欄留白。"""
    p = src / tag / "results.csv"
    if not p.exists():
        return {}
    return {r["image"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def read_purify(purify_dir: Path, tag: str, purifier: str) -> Dict[str, dict]:
    """逐圖的 `effect`（該算子下的位移）。分片檔一律合併。"""
    out: Dict[str, dict] = {}
    for p in sorted(purify_dir.glob(f"{tag}_*.csv")):
        # **不可以用 glob 前綴反推 tag**：`dct_aj85_*.csv` 會吃到
        # `dct_aj85_eps1.5_*.csv`。用 rsplit 去掉分片名再比對。
        if p.stem.rsplit("_", 1)[0] != tag:
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["purifier"] == purifier:
                out[r["image"]] = r
    return out


def fmt(v, spec="{:.4f}") -> str:
    try:
        return spec.format(float(v))
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# 頁面
# ---------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#111110;color:#eceae4;
font:15px/1.7 -apple-system,"Segoe UI","Noto Sans TC","PingFang TC",
"Microsoft JhengHei",sans-serif}
main{max-width:1500px;margin:0 auto;padding:36px 20px 90px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:18px;margin:44px 0 4px;padding-top:22px;
border-top:1px solid #34332f}
.lede,.note{color:#a3a099;font-size:13.5px;margin:6px 0}
.warn{color:#e0a05e}
.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:10px 0 4px}
.head span{font-size:12px;color:#a3a099;letter-spacing:.02em}
.cellwrap{display:flex;flex-direction:column;gap:4px}
.p{aspect-ratio:1/1;background-size:cover;background-position:center;
border-radius:6px;border:1px solid #34332f}
.p.empty{background:repeating-linear-gradient(45deg,#1c1c1a,#1c1c1a 7px,
#232320 7px,#232320 14px);border-style:dashed}
.rowhead{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
margin:18px 0 2px}
.tag{font-weight:600;font-size:14.5px}
.nums{color:#a3a099;font-size:12.5px;font-variant-numeric:tabular-nums}
.nums b{color:#eceae4;font-weight:600}
.floorrow .tag{color:#e0a05e}
.path{color:#6f6d67;font-size:11px;font-family:ui-monospace,Menlo,monospace;
word-break:break-all;margin:2px 0 0}
.blocked{color:#6cc196;font-weight:600}
.kept{color:#e0745e;font-weight:600}
"""

COLS = ["原圖", "防禦圖", "淨化後的防禦圖", "編輯（原圖）",
        "編輯（防禦圖）", "編輯（淨化防禦圖）"]


def cell(pool: PanelPool, path: Optional[Path], caption: str = "") -> str:
    cls = pool.cls(path)
    inner = (f'<div class="p {cls}"></div>' if cls
             else '<div class="p empty"></div>')
    return f'<div class="cellwrap">{inner}</div>'


def build(args) -> str:
    src, gal = Path(args.src), Path(args.gallery)
    purify_dir = Path(args.purify_dir)
    images = [ln.strip() for ln in open(args.images, encoding="utf-8")
              if ln.strip()]
    conds: List[Tuple[str, str]] = [c.split(":", 1) for c in args.conditions]
    pool = PanelPool(args.quality)

    defense = {tag: read_defense(src, tag) for tag, _ in conds}
    purify = {tag: read_purify(purify_dir, tag, args.purifier) for tag, _ in conds}
    floor = read_purify(purify_dir, "floor", args.purifier)

    body: List[str] = []
    for img in images:
        body.append(f"<h2>{img}</h2>")
        head = "".join(f"<span>{c}</span>" for c in COLS)
        body.append(f'<div class="grid head">{head}</div>')

        first_tag = conds[0][0]
        orig = src / first_tag / f"{img}__orig.png"
        edit_orig = src / first_tag / f"{img}__{conds[0][1]}__edit_orig.png"
        f_pur = gal / "floor" / f"{img}__{FLOOR_COND}__{args.purifier}__pur.png"
        f_edit = gal / "floor" / f"{img}__{FLOOR_COND}__{args.purifier}__edit_def.png"

        fr = floor.get(img, {})
        body.append(
            '<div class="rowhead floorrow"><span class="tag">'
            f'空白地板（沒有防禦，只有 {args.purifier}）</span>'
            f'<span class="nums">算子自己造成的位移 <b>'
            f'{fmt(fr.get("effect_mean"))}</b>'
            '　　任何淨增益都要扣掉這一格</span></div>')
        body.append('<div class="grid">'
                    + cell(pool, orig) + cell(pool, None) + cell(pool, f_pur)
                    + cell(pool, edit_orig) + cell(pool, None) + cell(pool, f_edit)
                    + "</div>")

        for tag, cond in conds:
            d = defense.get(tag, {}).get(img, {})
            pr = purify.get(tag, {}).get(img, {})
            eff = pr.get("effect_mean")
            net = None
            if eff not in (None, "") and fr.get("effect_mean") not in (None, ""):
                net = float(eff) - float(fr["effect_mean"])
            blocked = pr.get("siglip_blocked")
            mark = ("" if blocked in (None, "") else
                    f'　代理判定 <span class="{"blocked" if blocked == "True" else "kept"}">'
                    f'{"擋下" if blocked == "True" else "沒擋下"}</span>')
            body.append(
                f'<div class="rowhead"><span class="tag">{tag}</span>'
                f'<span class="nums">失真 DISTS <b>{fmt(d.get("fid_dists"))}</b>'
                f'　PSNR <b>{fmt(d.get("fid_psnr"), "{:.2f}")}</b>'
                f'　未淨化位移 <b>{fmt(d.get("edit_lpips"))}</b>'
                f'　{args.purifier} 位移 <b>{fmt(eff)}</b>'
                f'　扣地板淨增益 <b>{fmt(net)}</b>{mark}</span></div>')
            body.append(
                '<div class="grid">'
                + cell(pool, orig)
                + cell(pool, src / tag / f"{img}__{cond}__def.png")
                + cell(pool, gal / tag / f"{img}__{cond}__{args.purifier}__pur.png")
                + cell(pool, edit_orig)
                + cell(pool, src / tag / f"{img}__{cond}__edit_def.png")
                + cell(pool, gal / tag / f"{img}__{cond}__{args.purifier}__edit_def.png")
                + "</div>")
            body.append(f'<p class="path">{src}/{tag}/　'
                        f'{gal}/{tag}/</p>')

    n_cond = "、".join(t for t, _ in conds)
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>六格比對：{args.purifier}</title>
<style>{CSS}
{pool.css()}</style></head><body><main>
<h1>六格比對　·　淨化算子 {args.purifier}</h1>
<p class="lede">條件：{n_cond}。每一張影像的第一列是<strong>空白地板</strong>
——沒有防禦、只有淨化，用來排除「看起來被擋下其實是算子自己把圖弄壞了」。</p>
<p class="lede">判準是<strong>原圖還認得出來嗎</strong>：模型重畫成無關的場景、
或整張變成噪紋，兩者都算擋下；原圖仍認得出來、只是變糊變髒的單純劣化不算。
列上的「代理判定」是 SigLIP 讀數，<strong>金標準是人眼</strong>，代理只用於
條件內排序。</p>
<p class="note warn">面板是原生 512 不縮放、以品質 {args.quality} 的 JPEG 內嵌
（單一檔案帶得走的代價）。防禦擾動的 RMS 在 0.04–0.07、L∞ 在 0.3–0.7，遠高於
該品質的量化雜訊，所以看得到的差異不是編碼造成的；要逐像素驗證失真請回去看
每一列下方那兩個目錄裡的原生 PNG。</p>
{"".join(body)}
</main></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/ip2p_mainline")
    ap.add_argument("--gallery", default="runs/gallery_mainline")
    ap.add_argument("--purify-dir", default="runs/ip2p_mainline_purify")
    ap.add_argument("--images", default="runs/ip2p_fair_comparison/images10.txt")
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="tag:condition，condition 是檔名裡的那一段")
    ap.add_argument("--purifier", default="jpeg30")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    html = build(args)
    Path(args.out).write_text(html, encoding="utf-8")
    size = Path(args.out).stat().st_size / 1024 / 1024
    print(f"寫出 {args.out}（{size:.1f} MB）")


if __name__ == "__main__":
    main()
