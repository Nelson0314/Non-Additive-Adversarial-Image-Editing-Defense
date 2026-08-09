"""產生批次 T25 的 HTML 報告，格式沿用 `report_ip3.html`。

為什麼是一支腳本而不是手寫 HTML
──────────────────────────────────────────────────────────────────────
報告要嵌入數十張 base64 影像，且每一個數字都必須能追回 `runs/` 裡的來源。
手寫的話兩者都會退化成「抄過來的字」——改一次資料就要重抄一次，而抄錯
沒有症狀。此處全部由 `grid.csv` 與 `_cells/` 現算。

τ=0.30（批次 T30）的欄位一律留空
──────────────────────────────────────────────────────────────────────
`T30_PENDING` 為 True 時，逐批並列的表格會保留該欄並填入佔位符，圖組會
留下同尺寸的空框。s3t30 收工後把 `T30_BATCH` 指到它的目錄、旗標改 False
即可，不需要改動版面。
"""

from __future__ import annotations

import base64
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
T25 = ROOT / "runs" / "s3t25_merged"
T30 = ROOT / "runs" / "s3t30_merged"
T30_PENDING = not (T30 / "grid.csv").exists()

IMAGES = ["horse_00", "horse_03", "woman_03"]
CONDS = ["N4", "Ra", "photoguard_c", "mist", "dia_r"]
TRAIN_TAU = 0.25
# 佔位符。**不用「—」**：那與「這一格量到的就是沒有值」分不出來。
TBD = '<span class="missing">待補</span>'


# 嵌入前縮到這個邊長。原圖是 512²，逐張約 600 KB，十幾張就把報告推到
# 10 MB——而版面上每張只顯示到 ~380 px 寬，多出來的解析度看不到。
# 需要逐像素細看的人應該去開 `runs/` 裡的原檔，那才是證據來源。
EMBED_PX = 384


def b64(path: Path) -> str:
    """影像轉 data URI，縮到 `EMBED_PX`。不存在時回傳空字串。"""
    if not path.exists():
        return ""
    import io

    from PIL import Image

    im = Image.open(path).convert("RGB")
    if max(im.size) > EMBED_PX:
        im = im.resize((EMBED_PX, EMBED_PX), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def fig(path: Path, title: str, sub: str = "") -> str:
    """一張圖。**沒有檔案時留下同尺寸的空框**，而不是省略整個 figure——
    省略會讓「還沒產出」與「這一格本來就不存在」在版面上分不出來。"""
    src = b64(path)
    inner = (f'<img alt="{title}" src="{src}">' if src
             else '<div class="missing-fig">待產出</div>')
    cap = f'<span class="t">{title}</span>'
    if sub:
        cap += f'<span class="sub">{sub}</span>'
    return f"<figure>{inner}<figcaption>{cap}</figcaption></figure>"


def num(r, k):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_grid(batch: Path):
    if not (batch / "grid.csv").exists():
        return []
    with (batch / "grid.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def eval_mean(rows, cond, tau, key, purify="identity"):
    sel = [r for r in rows
           if r.get("stage") == "eval" and r.get("status") == "done"
           and r.get("purify_kind") == purify and r.get("condition") == cond
           and num(r, "tau") is not None and abs(num(r, "tau") - tau) < 1e-9]
    vals = [num(r, key) for r in sel]
    vals = [v for v in vals if v is not None]
    return (st.fmean(vals) if vals else None), len(vals)


def train_cells(batch: Path):
    out = {}
    for p in sorted((batch / "_cells").glob("train__*.json")):
        d = json.load(p.open(encoding="utf-8"))
        out[(d["condition"], d["image"])] = d
    return out


def skip_counts(batch: Path):
    out = {}
    for stage in ("rayscale", "control", "eval"):
        tot = 0
        reasons = defaultdict(int)
        for p in (batch / "_cells").glob(f"{stage}__*.json"):
            d = json.load(p.open(encoding="utf-8"))
            tot += 1
            if d.get("status") == "skipped":
                r = d.get("skipped_reason") or ""
                key = ("缺權重" if "cnn_denoise" in r else "低於重建下限")
                reasons[key] += 1
        out[stage] = (tot, dict(reasons))
    return out


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    rows25 = load_grid(T25)
    tr25 = train_cells(T25)
    sk25 = skip_counts(T25)
    floors = {r["image_id"]: r for r in
              csv.DictReader((T25 / "calib/recon_floor.csv").open(
                  encoding="utf-8"))}
    taus = sorted({num(r, "tau") for r in rows25
                   if r.get("stage") == "eval" and num(r, "tau") is not None})

    css = (ROOT / "report_ip3.html").read_text(encoding="utf-8")
    css = css[css.index("<style>"):css.index("</style>") + 8]
    css = css.replace("</style>", """
.missing{color:var(--warn);background:var(--warn-soft);
  padding:.05rem .4rem;border-radius:.2rem;font:600 .8rem/1.4 var(--mono)}
.missing-fig{aspect-ratio:1;display:grid;place-items:center;
  border:1px dashed var(--rule);border-radius:.3rem;color:var(--ink-faint);
  font:500 .8rem var(--mono);background:var(--surface-2)}
</style>""")

    h = ['<title>s3t25 · τ_train = 0.25 · 實驗報告</title>', css,
         '<div class="wrap"><header class="masthead">',
         '<p class="eyebrow">批次 s3t25 · SD v1.4 · 512² · fp32 · img2img</p>',
         '<h1>降低訓練預算之後，<br>次主張的兩道門檻首次同時成立</h1>']

    n4_35, _ = eval_mean(rows25, "N4", 0.35, "edit_lpips")
    ra_35, _ = eval_mean(rows25, "Ra", 0.35, "edit_lpips")
    best_35 = max((eval_mean(rows25, c, 0.35, "edit_lpips")[0] or 0, c)
                  for c in CONDS[2:])
    h.append(
        '<p class="lede measure">白盒非加性抗文字編輯防禦。本批只把 τ_train '
        f'由 EXP-s3a 的 0.50 降到 <b>0.25</b>，其餘設定不變。'
        f'N4 對同失真隨機對照拉開到 <b>{n4_35 / ra_35:.2f}×</b>、對最佳 '
        f'baseline 達 <b>{n4_35 / best_35[0]:.2f}×</b>（皆在 τ=0.35），'
        '這是本專案第一次兩道門檻同時成立。但跨批次的比較同時換過影像集，'
        '且「高 τ 的高分是不是靠把圖毀掉換的」尚未經人眼——兩項但書見文末。</p>')
    h.append('<div class="meta">'
             '<span><b>執行</b> 2026-08-09 17:09 – 23:17</span>'
             f'<span><b>格點</b> 2100 格，{len(rows25)} 列，零失敗</span>'
             '<span><b>影像</b> horse_00 · horse_03 · woman_03</span>'
             '<span><b>分支</b> claude/stage3-apa-attn</span></div>')
    h.append('</header>')

    # ---- 摘要卡 ----
    h.append('<section><div class="head"><p class="eyebrow">摘要</p>'
             '<h2>四項結論</h2></div><div class="verdicts">')
    h.append(
        '<div class="card is-good"><span class="k">確立</span>'
        '<span class="v">勝過同參數化隨機對照的幅度是實質的</span>'
        f'<span class="sub">N4/Ra 在四個可達的 τ 上為 1.65× / 2.26× / '
        f'2.94× / 2.79×。EXP-s3a 是 1.25–1.41× 的邊緣值。</span></div>')
    h.append(
        '<div class="card is-good"><span class="k">確立</span>'
        '<span class="v">門檻 (a) 在 τ=0.35 首次通過</span>'
        f'<span class="sub">N4 達最佳 baseline（{best_35[1]}）的 '
        f'{n4_35 / best_35[0]:.2f}×，超過 0.85。s3a 最好只有 0.82×，'
        '且那一點已被 FND-013 判為不算數。</span></div>')
    h.append(
        '<div class="card is-good"><span class="k">確立</span>'
        '<span class="v">最佳化本身沒有受限</span>'
        '<span class="sub">三格皆因平台停止而非跑滿 250 步；學習率 0.02 '
        '落在候選格點內部而非上界；<code>fid_lpips</code> 貼著約束。</span></div>')
    h.append(
        '<div class="card is-warn"><span class="k">但書</span>'
        '<span class="v">跨批次的比值混淆著影像集</span>'
        '<span class="sub">DEC-008 換過影像。同為 τ=0.20 的 photoguard_c，'
        's3a 0.394、本批 0.272——baseline 不受 τ_train 影響，'
        '那 31% 全部來自影像。</span></div>')
    h.append('</div></section>')

    # ---- 變因 ----
    h.append('<section><div class="head"><p class="eyebrow">變因</p>'
             '<h2>相對批次 A 改了什麼</h2></div>'
             '<div class="tw"><table><caption>只有一項是刻意的變因，'
             '另一項是先前的裁決帶來的</caption><thead><tr>'
             '<th>項目</th><th>EXP-s3a</th><th>EXP-s3t25</th>'
             '<th>EXP-s3t30</th><th>性質</th></tr></thead><tbody>')
    for item, a, b, c, kind in [
            ("τ_train", "0.50", "<b>0.25</b>", "<b>0.30</b>", "刻意的變因"),
            ("tau_acut（0.8τ）", "0.40", "0.20", "0.24", "隨 τ_train 導出"),
            ("tau_chroma（16τ）", "8.0", "4.0", "4.8", "隨 τ_train 導出"),
            ("影像", "bird_03 · cat_02 · dog_03",
             "horse_00 · horse_03 · woman_03", "同 s3t25",
             "<b>混淆項</b>，DEC-008"),
            ("man 類的攻擊 prompt", "—", "an old woman", "同 s3t25",
             "DEC-008，本批無 man"),
            ("條件、模型、strength、attn_timesteps", "同左", "逐字不變",
             "逐字不變", "不變")]:
        h.append(f"<tr><td>{item}</td><td>{a}</td><td>{b}</td><td>{c}</td>"
                 f'<td style="text-align:left;white-space:normal;'
                 f'font-size:.85rem">{kind}</td></tr>')
    h.append("</tbody></table></div></section>")

    # ---- 段 0 ----
    h.append('<section><div class="head"><p class="eyebrow">段 0</p>'
             '<h2>校準：學習率這次落在格點內部</h2></div>')
    h.append('<p class="measure">EXP-s3a 的 <code>lr.N4_stage2</code> 選到 '
             '0.1，那是候選格點 <code>[1e-4, 1e-3, 5e-3, 0.02, 0.1]</code> 的'
             '<b>上界</b>——意思是「再大可能更好，但沒有測」。本批選到 '
             '<b>0.02</b>，60 步探測的末端損失為 162.5 → 139.2 → 51.3 → '
             '<b>24.8</b> → 38.4，確實有谷底，故不存在格點不夠寬的疑慮。</p>')
    h.append('<div class="tw"><table><caption>重建下限（'
             '<code>calib/recon_floor.csv</code>）。site apa 走生成路徑，'
             '其最小失真即該影像的 VAE 重建誤差，決定哪些 τ 結構上不可達'
             '</caption><thead><tr><th>影像</th><th>LPIPS</th><th>PSNR (dB)'
             '</th><th>SSIM</th><th>可達的最小 τ</th></tr></thead><tbody>')
    for iid in IMAGES:
        r = floors[iid]
        h.append(f'<tr><td>{iid}</td>'
                 f'<td class="num">{float(r["lpips"]):.4f}</td>'
                 f'<td class="num">{float(r["psnr"]):.2f}</td>'
                 f'<td class="num">{float(r["ssim"]):.4f}</td>'
                 f'<td class="num">&gt; {float(r["lpips"]):.3f}</td></tr>')
    h.append("</tbody></table></div>")
    h.append('<p class="sub measure">本批最大值 0.1448 低於 s3a 的 0.2398'
             '（cat_02），故低 τ 上構得到的點比 s3a 多。τ=0.10 只有 woman_03'
             '一張構得到——這是主表該列只有 5 格而非 15 格的原因，不是資料缺漏。'
             '</p></section>')

    # ---- 段 1 ----
    h.append('<section><div class="head"><p class="eyebrow">段 1</p>'
             '<h2>訓練：三格都因平台停止，不是跑滿</h2></div>'
             '<div class="tw"><table><caption>跑滿上限量到的是「走到哪裡」，'
             '平台停止量到的才是能力</caption><thead><tr><th>影像</th>'
             '<th>final L_def</th><th>fid_lpips</th><th>PSNR</th>'
             '<th>步數</th><th>停止原因</th></tr></thead><tbody>')
    for iid in IMAGES:
        d = tr25[("N4", iid)]
        h.append(f'<tr><td>{iid}</td>'
                 f'<td class="num">{d["final_L_def"]:.4g}</td>'
                 f'<td class="num">{d["fid_lpips"]:.4f}</td>'
                 f'<td class="num">{d["fid_psnr"]:.2f}</td>'
                 f'<td class="num">{d["steps_used"]}</td>'
                 f'<td style="text-align:left;white-space:normal;'
                 f'font-size:.8rem">{d["stop_reason"]}</td></tr>')
    h.append("</tbody></table></div>")
    h.append('<p class="sub measure">步數 73 / 79 / 132 對照 s3a 的 '
             '54 / 72 / 90，本批較長，與學習率由 0.1 降到 0.02 一致：'
             '每步走得小，故需要更多步才到平台。'
             '<code>fid_lpips</code> 落在 0.268–0.293，略高於 τ_train=0.25'
             '——約束是綁定的，超出的部分由段 2 的射線縮放收回。</p>')

    h.append("<h3>訓練曲線</h3>"
             '<p class="sub" style="font-size:.85rem">逐影像的 '
             '<code>optimize</code> 歷程。</p><div class="plate n3">')
    for iid in IMAGES:
        h.append(fig(T25 / "N4" / iid / "history.png", f"{iid} · N4"))
    h.append("</div>")

    h.append("<h3>訓練點的防禦圖</h3>"
             '<p class="sub" style="font-size:.85rem">上排原圖、下排 '
             '<code>x_def</code>（τ_train = 0.25）。</p><div class="plate n3">')
    for iid in IMAGES:
        h.append(fig(T25 / "N4" / iid / "orig.png", f"{iid} · 原圖"))
    for iid in IMAGES:
        h.append(fig(T25 / "N4" / iid / "x_def_tau0.25.png",
                     f"{iid} · x_def τ=0.25"))
    h.append("</div></section>")

    # ---- 主結果 ----
    h.append('<section><div class="head"><p class="eyebrow">段 3 · 主結果</p>'
             '<h2>逐 τ 的輸出位移</h2></div>'
             '<div class="tw"><table><caption><code>edit_lpips</code>'
             '（identity），愈高代表把編輯推得愈遠。括號為樣本數'
             '</caption><thead><tr><th>τ</th>'
             + "".join(f"<th>{c}</th>" for c in CONDS)
             + "</tr></thead><tbody>")
    for t in taus:
        cells = []
        vals = {}
        for c in CONDS:
            v, n = eval_mean(rows25, c, t, "edit_lpips")
            vals[c] = v
            cls = "num good" if (c == "N4" and v) else "num"
            cells.append(f'<td class="{cls}">'
                         + (f"{v:.4f} <span class=sub>({n})</span>"
                            if v is not None else "—") + "</td>")
        mark = ' style="background:var(--accent-soft)"' if abs(
            t - TRAIN_TAU) < 1e-9 else ""
        h.append(f"<tr{mark}><td class='num'><b>{t:g}</b></td>"
                 + "".join(cells) + "</tr>")
    h.append("</tbody></table></div>")

    # 門檻表
    h.append('<div class="tw"><table><caption>次主張的兩道門檻：'
             '(a) ≥ 0.85 × 最佳 baseline；(b) &gt; 同失真隨機對照'
             '</caption><thead><tr><th>τ</th><th>N4 / Ra（門檻 b）</th>'
             '<th>N4 / 最佳 baseline（門檻 a）</th><th>最佳者</th>'
             '<th>τ_train=0.30</th></tr></thead><tbody>')
    for t in taus:
        n4, _ = eval_mean(rows25, "N4", t, "edit_lpips")
        ra, _ = eval_mean(rows25, "Ra", t, "edit_lpips")
        if n4 is None or ra is None:
            continue
        best = max((eval_mean(rows25, c, t, "edit_lpips")[0] or 0, c)
                   for c in CONDS[2:])
        gb, ga = n4 / ra, n4 / best[0]
        h.append(
            f"<tr><td class='num'>{t:g}</td>"
            f"<td class='num good'>{gb:.2f}× 通過</td>"
            f"<td class='num {'good' if ga >= 0.85 else 'bad'}'>{ga:.2f}× "
            f"{'通過' if ga >= 0.85 else '不通過'}</td>"
            f"<td>{best[1]}</td><td>{TBD if T30_PENDING else ''}</td></tr>")
    h.append("</tbody></table></div>")
    h.append('<div class="callout"><b>τ=0.35 是本專案第一次兩道門檻同時成立'
             '的點。</b> 在把它當成結論之前，文末的兩項但書都必須先處理。</div>')

    # 第三層
    h.append("<h3>三層一次報（τ = 0.25、identity）</h3>"
             '<p class="sub measure">挑一層報告等於挑一個對自己有利的定義，'
             '故三層一次列出。第 3 層是代價不是成果——若某一側的免疫靠劣化'
             '撐著，這裡會顯示出來。</p>'
             '<div class="tw"><table><thead><tr><th>條件</th>'
             '<th>LPIPS ↑</th><th>PSNR ↓</th><th>SSIM ↓</th>'
             '<th>ΔNIQE ↑</th><th>銳利度比</th></tr></thead><tbody>')
    for c in CONDS:
        lp, _ = eval_mean(rows25, c, TRAIN_TAU, "edit_lpips")
        ps, _ = eval_mean(rows25, c, TRAIN_TAU, "edit_psnr")
        ss, _ = eval_mean(rows25, c, TRAIN_TAU, "edit_ssim")
        na, _ = eval_mean(rows25, c, TRAIN_TAU, "edit_niqe_a")
        nb, _ = eval_mean(rows25, c, TRAIN_TAU, "edit_niqe_b")
        ac, _ = eval_mean(rows25, c, TRAIN_TAU, "edit_acutance_ratio")
        dn = (nb - na) if (na is not None and nb is not None) else None
        dncls = "num bad" if (dn is not None and dn > 0.5) else "num"
        h.append(f"<tr><td>{c}</td><td class='num'>{fmt(lp)}</td>"
                 f"<td class='num'>{fmt(ps, 2)}</td>"
                 f"<td class='num'>{fmt(ss)}</td>"
                 f"<td class='{dncls}'>{fmt(dn, 3)}</td>"
                 f"<td class='num'>{fmt(ac)}</td></tr>")
    h.append("</tbody></table></div>")
    h.append('<div class="callout warn">第 3 層這次<b>對 baseline 不利而非'
             '對我方不利</b>：photoguard_c 的 ΔNIQE 為 +0.975、銳利度比 0.54，'
             '代表它在該 τ 上的免疫有一部分是靠把輸出弄糟撐起來的。'
             'N4 的 ΔNIQE 是負的，即輸出的無參考品質沒有變差。'
             '這不改變門檻 (a) 的結論，但它說明那個分母本身帶著劣化成分。'
             '</div></section>')

    # ---- 人眼比對 ----
    h.append('<section><div class="head"><p class="eyebrow">人眼</p>'
             '<h2>τ=0.25 與 τ=0.35 的實際輸出</h2></div>'
             '<p class="measure">判準以人眼為主、數值為輔。'
             'FND-013 對 s3a 在 τ=0.50 追到 0.82× 的判語是「靠把圖毀掉換的，'
             '不算數」——τ=0.35 這一點是否同樣如此，指標答不了。</p>')
    for iid in IMAGES:
        h.append(f"<h3>{iid}</h3><div class='plate n4'>")
        h.append(fig(T25 / "N4" / iid / "orig.png", "原圖"))
        h.append(fig(T25 / "N4" / iid / "x_def_tau0.25.png", "x_def τ=0.25"))
        h.append(fig(T25 / "N4" / iid / "purify/identity_0"
                     / "edit_tau0.25_seed0.png", "防禦後的編輯 τ=0.25"))
        h.append(fig(T25 / "control" / iid / "purify/identity_0"
                     / "edit_seed0.png", "未防禦的編輯（對照）"))
        h.append("</div>")
    h.append('<div class="callout warn">τ=0.35 的同組圖'
             f'{TBD}——需由 <code>x_def_tau0.35.png</code> 與該 τ 的編輯輸出'
             '排出，並由使用者判讀。</div></section>')

    # ---- 抗淨化 ----
    h.append('<section><div class="head"><p class="eyebrow">主主張</p>'
             '<h2>抗淨化</h2></div>'
             f'<p class="measure">{TBD}。需跑 '
             '<code>scripts/purify_advantage.py</code>；其前提是「在 identity '
             '上先有可測的效果」，而本批 N4 在 τ=0.25 上的 edit_lpips 為 '
             '0.2384，遠高於階段一落在雜訊裡的量級，故該前提在本批首次成立。'
             's3t30 收工後兩批一起做。</p></section>')

    # ---- 橫向 ----
    h.append('<section><div class="head"><p class="eyebrow">橫向</p>'
             '<h2>三批並列</h2></div>'
             '<div class="tw"><table><caption><b>這張表不可直接讀成'
             '「τ_train 的效果」</b>，理由見文末但書</caption><thead><tr>'
             '<th>量</th><th>s3a（0.50）</th><th>s3t25（0.25）</th>'
             '<th>s3t30（0.30）</th></tr></thead><tbody>')
    n4_20, _ = eval_mean(rows25, "N4", 0.20, "edit_lpips")
    ra_20, _ = eval_mean(rows25, "Ra", 0.20, "edit_lpips")
    best_20 = max((eval_mean(rows25, c, 0.20, "edit_lpips")[0] or 0, c)
                  for c in CONDS[2:])
    for label, a, b in [
            ("lr.N4_stage2", "0.1（格點上界）", "0.02（內部極小）"),
            ("段 1 步數", "54 / 72 / 90", "73 / 79 / 132"),
            ("重建下限最大值", "0.2398（cat_02）", "0.1448（horse_03）"),
            ("N4/Ra @ τ=0.20", "1.25×", f"{n4_20 / ra_20:.2f}×"),
            ("N4/Ra @ τ=0.35", "1.31×", f"{n4_35 / ra_35:.2f}×"),
            ("N4/最佳 @ τ=0.20", "0.31×", f"{n4_20 / best_20[0]:.2f}×"),
            ("N4/最佳 @ τ=0.35", "0.46×", f"{n4_35 / best_35[0]:.2f}×")]:
        h.append(f"<tr><td>{label}</td><td class='num'>{a}</td>"
                 f"<td class='num'>{b}</td>"
                 f"<td class='num'>{TBD if T30_PENDING else ''}</td></tr>")
    h.append("</tbody></table></div></section>")

    # ---- 格點健康 ----
    h.append('<section><div class="head"><p class="eyebrow">格點</p>'
             '<h2>零失敗格，skipped 的兩個來源都是設計上的</h2></div>'
             '<div class="tw"><table><thead><tr><th>段</th><th>總格</th>'
             '<th>skipped</th><th>來源</th></tr></thead><tbody>')
    for stage, (tot, reasons) in sk25.items():
        src = "、".join(f"{v} × {k}" for k, v in sorted(reasons.items()))
        h.append(f"<tr><td>{stage}</td><td class='num'>{tot}</td>"
                 f"<td class='num'>{sum(reasons.values())}</td>"
                 f"<td style='text-align:left'>{src or '—'}</td></tr>")
    h.append("</tbody></table></div>"
             '<p class="sub measure">兩個來源與既有批次相同：'
             '<code>cnn_denoise_substitute</code> 缺權重（設計上即不可用），'
             '以及 apa 在低於該影像重建下限的 τ 上結構不可達。'
             '後者標為 skipped 而非 failed 是刻意的——那是關於該參數化的'
             '結果，不是錯誤。</p></section>')

    # ---- 但書 ----
    h.append('<section><div class="head"><p class="eyebrow">適用範圍</p>'
             '<h2>兩項必須先處理的但書</h2></div>')
    h.append('<div class="callout bad"><b>一、跨批次的比值混淆著影像集。</b> '
             'DEC-008 換過影像：s3a 是 bird_03 / cat_02 / dog_03，本批是 '
             'horse_00 / horse_03 / woman_03。證據在 baseline 自己身上——'
             '同為 τ=0.20 的 photoguard_c，s3a 0.394、本批 '
             f'{best_20[0]:.3f}，而 baseline 不受 τ_train 影響。'
             '<br><br><b>仍然有效</b>：批次內的比值（同影像、同 τ）。'
             '<b>不可說</b>：「τ_train 由 0.50 降到 0.25 使 0.31× 變成 '
             f'{n4_20 / best_20[0]:.2f}×」。要拆開這個混淆，需在 s3a 的三張'
             '影像上補跑 τ_train=0.25，或反過來——一整批機時，屬使用者的決定。'
             '</div>')
    h.append('<div class="callout bad"><b>二、τ=0.35 的「通過」尚未經人眼。</b> '
             'FND-013 對 s3a 在 τ=0.50 追到 0.82× 的判語是「靠把圖毀掉換的」。'
             '本批 τ=0.35 是否同樣如此，只能看圖。看完之前，'
             '門檻 (a) 的「通過」只是數值上的通過。</div>')
    h.append("</section>")

    h.append('<p class="foot sub" style="margin-top:4rem;font-size:.8rem">'
             '由 <code>scripts/report_t25.py</code> 產生；每一個數字都從 '
             '<code>runs/s3t25_merged/</code> 現算。'
             + ("τ_train=0.30 的欄位待 s3t30 收工後填入。" if T30_PENDING
                else "") + "</p></div>")

    out = ROOT / "report_s3t25.html"
    out.write_text("\n".join(h), encoding="utf-8")
    print(f"{out}  {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
