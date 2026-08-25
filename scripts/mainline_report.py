"""主線頭對頭的報告頁。**不跑 GPU、不重算任何數字**，只讀 CSV 與 PNG。

輸入
────────────────────────────────────────────────────────────────────
    tables.json         `mainline_tables.py` 的輸出（三張表 ＋ 淨增益格）
    sheets/*.png        `mainline_sheets.py` 的五欄大圖，逐列裁出來內嵌

頁面上的圖是為了顯示而重新編碼的 JPEG；**判定一律以原生 512 的 PNG 為準**，
本檔只是把它們排進頁面。

三件必須寫在頁面上的事
────────────────────────────────────────────────────────────────────
1. **編輯指令**：攻擊方餵給 IP2P 的那一句話。不同指令的服從率差很多
   （`DECISIONS.md`：改顏色 5/5、加物件 1/5），沒有指令就看不出這一列在測什麼。
2. **語意指標比對的對象**：`edit_clip_sim`／`edit_siglip_sim` 是
   **兩張編輯輸出之間**的影像—影像餘弦，**完全沒有用到文字**。
   影像對文字那條路在本專案上近乎隨機（OmniEdit 給的是指令不是描述），
   已經棄用。不寫清楚會被讀成「拿指令當文字端算的」。
3. **一律用淨增益（差值）不用比例**：比值在分母塌陷時不可解讀
   （`FND-037`／`FND-039`，相關係數 −0.83／−0.900）。

用法：
    python scripts/mainline_report.py --tables runs/ip2p_mainline/tables \\
        --sheets runs/ip2p_mainline/sheets --out report_mainline.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TILE, PAD, TITLE_BAND, HEAD_BAND, ROWLAB = 512, 16, 46, 34, 30
# 頁面上影像的邊長。**512 = 原生尺寸不縮放**，判「擋下與否」看的是語意內容，
# 縮放會把細節抹掉、邊緣案例會判錯。
#
# 代價是體積：512 在 q88 實測每張約 83 KB，base64 之後 ×1.37，十個條件 ×
# 十張 × 五欄約 **57 MB**。**本機檔沒有上限所以放得下**；若要發成 Artifact
# 則有 16 MB 的硬上限（且此帳號未開 `assets`），那時才需要縮圖或拆頁。
# `--limit-mb 0` 關掉守門，供本機輸出使用。
THUMB = 512
THUMB_Q = 88

FID = [("fid_dists", "DISTS"), ("fid_lpips", "LPIPS"), ("fid_psnr", "PSNR"),
       ("fid_ssim", "SSIM"), ("fid_vif_p", "VIFp"), ("fid_linf", "L∞"),
       ("fid_rms", "RMS")]
EDIT = [("edit_lpips", "位移 LPIPS"), ("edit_dists", "位移 DISTS"),
        ("edit_psnr", "PSNR"), ("edit_ssim", "SSIM"), ("edit_vif_p", "VIFp"),
        ("edit_clip_sim", "CLIP 影像對"), ("edit_siglip_sim", "SigLIP 影像對")]
LOWER_BETTER_FID = {"fid_dists", "fid_lpips", "fid_linf", "fid_rms"}
# 位移越大越好；語意相似度越**低**代表內容被換掉，也就是防禦成功。
LOWER_BETTER_EDIT = {"edit_psnr", "edit_ssim", "edit_vif_p",
                     "edit_clip_sim", "edit_siglip_sim"}


def crop_rows(png: Path, n_rows: int, thumb: int = THUMB, quality: int = THUMB_Q):
    """把五欄大圖逐列裁出來，每一列回傳五張縮圖的 data URI。"""
    from PIL import Image

    im = Image.open(png)
    top = TITLE_BAND + HEAD_BAND
    out = []
    for i in range(n_rows):
        y = top + i * (ROWLAB + TILE + PAD) + ROWLAB
        row = []
        for c in range(5):
            x = PAD + c * (TILE + PAD)
            t = im.crop((x, y, x + TILE, y + TILE)).resize((thumb, thumb),
                                                           Image.LANCZOS)
            b = io.BytesIO()
            t.convert("RGB").save(b, format="JPEG", quality=quality, optimize=True)
            row.append("data:image/jpeg;base64,"
                       + base64.b64encode(b.getvalue()).decode())
        out.append(row)
    return out


def num_table(rows, cols, lower_better, extra=None):
    """一張數值表。**最好的一格加粗**，方向由 `lower_better` 決定。"""
    best = {}
    for key, _ in cols:
        vals = [r[key] for r in rows if key in r]
        if vals:
            best[key] = min(vals) if key in lower_better else max(vals)
    head = "".join(f"<th>{html.escape(lab)}</th>" for _, lab in cols)
    if extra:
        head += f"<th>{html.escape(extra[1])}</th>"
    body = []
    for r in rows:
        cls = "ours" if r["tag"].startswith("ours") else "rival"
        tds = []
        for key, _ in cols:
            v = r.get(key)
            if v is None:
                tds.append("<td>—</td>")
                continue
            mark = " class='best'" if v == best.get(key) else ""
            tds.append(f"<td{mark}>{v:.4f}</td>")
        if extra:
            tds.append(f"<td>{html.escape(str(r.get(extra[0], '—')))}</td>")
        body.append(f"<tr class='{cls}'><td>{html.escape(r['label'])}</td>"
                    f"{''.join(tds)}</tr>")
    return (f"<div class='scroll'><table><thead><tr><th>條件</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--sheets", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--purifier", default="jpeg75",
                    help="對比圖用哪一個淨化算子那一版")
    ap.add_argument("--figure-tags", nargs="+",
                    default=["ours_pg_q", "dct_aj85"],
                    help="要出對比圖的條件")
    ap.add_argument("--thumb", type=int, default=THUMB,
                    help="頁面上影像的邊長。512 = 原生不縮放")
    ap.add_argument("--quality", type=int, default=THUMB_Q,
                    help="JPEG 品質。**不改尺寸**，只改壓縮率——傳輸有 30 MiB "
                         "上限而 q88 的完整頁約 47 MB，降到 q75 約 27 MB")
    ap.add_argument("--limit-mb", type=float, default=0.0,
                    help="超過就拋錯。0 = 不限（本機檔）；發 Artifact 時給 15")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from apa_baseline import load_dataset

    T = json.loads((args.tables / "tables.json").read_text(encoding="utf-8"))
    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    ds = {d["name"]: d for d in load_dataset(args.data)}
    prompts = [ds[n]["prompt"] for n in names]

    # ---- 對比圖 ----------------------------------------------------------
    figs = {}
    for tag in args.figure_tags:
        p = args.sheets / f"{tag}__{args.purifier}.png"
        if p.exists():
            figs[tag] = crop_rows(p, len(names), args.thumb, args.quality)

    strips = []
    for i, (n, pr) in enumerate(zip(names, prompts)):
        blocks = []
        for tag in args.figure_tags:
            if tag not in figs:
                continue
            cells = "".join(
                f"<figure class='cell'><img src='{src}' alt='{html.escape(n)} {lab}'"
                f" loading='lazy'><figcaption>{html.escape(lab)}</figcaption></figure>"
                for src, lab in zip(figs[tag][i],
                                    ["原圖", "原圖的編輯", "防禦圖",
                                     f"防禦圖 +{args.purifier}",
                                     f"壓縮後的編輯"]))
            blocks.append(f"<div class='cond'><p class='condname'>"
                          f"{html.escape(T['label'].get(tag, tag))}</p>"
                          f"<div class='grid5'>{cells}</div></div>")
        strips.append(
            f"<div class='strip'><div class='strip-head'>"
            f"<span class='mono id'>#{i + 1:02d} {html.escape(n)}</span>"
            f"<span class='prompt'>指令：{html.escape(pr)}</span></div>"
            f"{''.join(blocks)}</div>")

    # ---- 淨增益表 --------------------------------------------------------
    purs = T["purifiers"]
    grid = T["gain"]
    # 抗淨化的張數可能不齊（有條件還在跑）。**不齊的一律標出來**——
    # 十張與四張的平均放在同一欄裡比是錯的，而報表上看不出來。
    counts = {}
    import csv as _csv
    gp = args.tables / "net_gain.csv"
    if gp.exists():
        for r in _csv.DictReader(gp.open(encoding="utf-8")):
            counts[r["tag"]] = int(r["n"])

    ghead = "".join(f"<th>{html.escape(p.replace('crop_resize', 'crop'))}</th>"
                    for p in purs)
    # **未跑完的條件不參與「最好的一格」**：它的平均是別的樣本集算的。
    full = [t for t in grid if counts.get(t, len(names)) >= len(names)]
    best = {p: max((grid[t][p] for t in full if p in grid[t]), default=None)
            for p in purs}
    gbody = []
    for tag in T["order"]:
        if tag not in grid:
            continue
        cls = "ours" if tag.startswith("ours") else "rival"
        n_img = counts.get(tag)
        partial = n_img is not None and n_img < len(names)
        tds = []
        for p in purs:
            v = grid[tag].get(p)
            if v is None:
                tds.append("<td>—</td>")
                continue
            mark = " class='best'" if v == best[p] else ""
            tds.append(f"<td{mark}>{v:+.4f}</td>")
        lab = html.escape(T["label"].get(tag, tag))
        if partial:
            lab += (f" <span class='partial'>僅 {n_img}/{len(names)} 張，"
                    f"尚未跑完</span>")
        gbody.append(f"<tr class='{cls}{' dim' if partial else ''}'>"
                     f"<td>{lab}</td>{''.join(tds)}</tr>")
    gain_table = (f"<div class='scroll'><table><thead><tr><th>條件</th>{ghead}</tr>"
                  f"</thead><tbody>{''.join(gbody)}</tbody></table></div>")

    prompt_rows = "".join(
        f"<tr><td class='mono'>#{i + 1:02d}</td><td class='mono'>{html.escape(n)}</td>"
        f"<td class='pr'>{html.escape(p)}</td></tr>"
        for i, (n, p) in enumerate(zip(names, prompts)))

    # ---- 人眼判定 --------------------------------------------------------
    vp = args.tables.parent / "human_verdict.json"
    verdict_rows = ""
    if vp.exists():
        V = json.loads(vp.read_text(encoding="utf-8"))
        for q in ("jpeg75", "jpeg50", "jpeg30"):
            if q not in V:
                continue
            for tag, r in V[q].items():
                nb, nbd = len(r["blocked"]), len(r["borderline"])
                cls = "ours" if tag.startswith("ours") else "rival"
                g = grid.get(tag, {}).get(q)
                verdict_rows += (
                    f"<tr class='{cls}'><td>{html.escape(q)}</td>"
                    f"<td>{html.escape(T['label'].get(tag, tag))}</td>"
                    f"<td>{g:+.4f}</td>" if g is not None else
                    f"<tr class='{cls}'><td>{html.escape(q)}</td>"
                    f"<td>{html.escape(T['label'].get(tag, tag))}</td><td>—</td>")
                verdict_rows += (
                    f"<td class='{'best' if nb else ''}'>{nb} / {len(names)}</td>"
                    f"<td>{nbd}</td>"
                    f"<td class='vnote'>{html.escape(r['note'])}</td></tr>")

    tpl = Path(__file__).with_name("mainline_report_template.html").read_text(
        encoding="utf-8")
    out = (tpl.replace("__FIDELITY__", num_table(T["fidelity"], FID,
                                                 LOWER_BETTER_FID))
              .replace("__EDIT__", num_table(T["edit"], EDIT, LOWER_BETTER_EDIT,
                                             extra=("siglip_blocked", "SigLIP 擋下")))
              .replace("__GAIN__", gain_table)
              .replace("__PROMPTS__", prompt_rows)
              .replace("__FIGURES__", "".join(strips))
              .replace("__NIMG__", str(len(names)))
              .replace("__PURIFIER__", html.escape(args.purifier))
              .replace("__VERDICT__", verdict_rows))
    args.out.write_text(out, encoding="utf-8")
    mb = args.out.stat().st_size / 1048576
    print(f"寫出 {args.out}（{mb:.1f} MB，{len(figs)} 個條件的對比圖，"
          f"影像邊長 {args.thumb}）")
    if args.limit_mb and mb > args.limit_mb:
        raise SystemExit(
            f"頁面 {mb:.1f} MB 超過 {args.limit_mb} MB。"
            "減少 --figure-tags 的條件數，或調小 --thumb。")


if __name__ == "__main__":
    main()
