"""2026-08-16 批次的報告產生器。

產出四份自足的 HTML，全部落在 `reports/2026-08-16/`：

    01_external.html    與外部方法的比較（本輪的主目標）
    02_budget.html      既有 24 張資料的重新分析：預算漂移與比值的分母支配
    03_ablation.html    目標影像、約束設定、第二個 prompt 三組消融
    04_floor.html       空白地板：淨化算子自己造成的位移

設計上**不吃缺資料**：每個區塊各自檢查來源檔在不在，缺了就在頁面上寫明缺什麼、
該跑哪一支腳本，而不是靜默少一張表。跑到一半的批次可以直接產報告。

影像一律內嵌成 data URI，故 HTML 可以單獨寄出、不依賴 runs/ 目錄。
每一格都要有圖可看是專案既有規則（判準以人眼為主）。

用法：
    python scripts/report_0816.py --out reports/2026-08-16
"""

from __future__ import annotations

import argparse
import base64
import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 條件的顯示名稱。程式內部的字串是 add／phase／phase_rand，讀者看到的是中文。
LABEL = {
    "phase": "紋理重相位",
    "add": "加性 δ（同損失）",
    "phase_rand": "隨機相位 RPN",
    "photoguard_c": "PhotoGuard-c",
    "mist": "Mist",
    "dia_r": "DIA-R",
    "apa_weak": "APA 弱 baseline",
    "none": "無防禦（地板）",
}
ORDER = ["phase", "photoguard_c", "mist", "dia_r", "add", "apa_weak", "phase_rand"]


# ---------------------------------------------------------------- 讀檔


def read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(row: dict, key: str, default=None):
    """CSV 欄位轉浮點。空字串與缺欄回傳 default，不丟例外——
    合併不同來源的批次時欄位本來就不齊，那不是錯誤。"""
    v = row.get(key, "")
    if v is None or v == "":
        return default
    return float(v)


def collect_runs(dirs: list) -> list:
    rows = []
    for d in dirs:
        rows += read_csv(Path(d) / "results.csv")
    return rows


# ---------------------------------------------------------------- 統計


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x is not None]
    return st.stdev(xs) if len(xs) > 1 else float("nan")


def pearson(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pairs) < 3:
        return float("nan")
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    ma, mb = st.fmean(a), st.fmean(b)
    na = math.sqrt(sum((x - ma) ** 2 for x in a))
    nb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if na == 0 or nb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (na * nb)


def fmt(v, n=4):
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:      # NaN
        return "—"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


# ---------------------------------------------------------------- HTML


CSS = """
:root{--bg:#ffffff;--fg:#16181d;--mut:#5b6472;--line:#e3e6eb;--acc:#1f5fd0;
      --good:#0d7a4a;--bad:#b3261e;--card:#f7f8fa}
@media (prefers-color-scheme:dark){:root{--bg:#101317;--fg:#e8eaed;--mut:#9aa3b0;
  --line:#282d35;--acc:#7aa7f5;--good:#4ec98d;--bad:#ff8a80;--card:#171b21}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.75 -apple-system,"Segoe UI","Noto Sans TC",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:40px 24px 90px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:44px 0 10px;padding-top:18px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:26px 0 8px;color:var(--fg)}
.sub{color:var(--mut);margin:0 0 28px;font-size:14px}
p{margin:10px 0}
.note{background:var(--card);border-left:3px solid var(--acc);
  padding:12px 16px;margin:16px 0;border-radius:0 6px 6px 0;font-size:14px}
.warn{border-left-color:var(--bad)}
.ok{border-left-color:var(--good)}
.tw{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;
  font-variant-numeric:tabular-nums}
th,td{padding:7px 11px;text-align:right;border-bottom:1px solid var(--line);
  white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--card);font-weight:600;position:sticky;top:0}
tbody tr:hover{background:var(--card)}
.win{color:var(--good);font-weight:600}
.lose{color:var(--bad)}
.best td{background:color-mix(in srgb,var(--good) 9%,transparent)}
code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:12.5px}
.grid{display:grid;gap:14px;margin:18px 0}
.g3{grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}
.g4{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
figure{margin:0;background:var(--card);border:1px solid var(--line);
  border-radius:8px;overflow:hidden}
figure img{width:100%;display:block;background:#888}
figcaption{padding:7px 10px;font-size:12px;color:var(--mut);line-height:1.5}
figcaption b{color:var(--fg);display:block;font-size:13px}
.miss{background:var(--card);border:1px dashed var(--line);border-radius:8px;
  padding:16px;color:var(--mut);font-size:14px;margin:14px 0}
ul{margin:10px 0;padding-left:22px}li{margin:5px 0}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:14px;
  margin:12px 0}
.kv dt{color:var(--mut)}.kv dd{margin:0}
"""


def img_tag(path: Path, side: int = 384, quality: int = 82) -> str:
    """把影像縮圖後內嵌成 data URI。檔案不在就回傳灰底佔位，不讓版面塌掉。

    一律縮圖再轉 JPEG：512² 的 PNG 每張約 300 KB，一頁擺上百張就是幾十 MB，
    寄不動也開不動。縮到 384 邊長、quality 82 之後每張約 25 KB，判讀擾動的
    紋理與編輯的差異仍然足夠——需要逐像素看的場合本來就要開 runs/ 的原檔。
    """
    if not path.exists():
        return ('<div style="aspect-ratio:1;display:grid;place-items:center;'
                'color:var(--mut);font-size:12px">缺圖</div>')
    from PIL import Image
    import io
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return ('<img src="data:image/jpeg;base64,'
            f'{base64.b64encode(buf.getvalue()).decode()}" loading="lazy">')


def page(title: str, subtitle: str, body: str) -> str:
    return (f"<!doctype html><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{CSS}</style>"
            f"<div class=wrap><h1>{title}</h1><p class=sub>{subtitle}</p>{body}</div>")


def table(head: list, rows: list, best_row: int = None) -> str:
    h = "".join(f"<th>{c}</th>" for c in head)
    body = []
    for i, r in enumerate(rows):
        cls = " class=best" if best_row is not None and i == best_row else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return (f"<div class=tw><table><thead><tr>{h}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def missing(what: str, how: str) -> str:
    return f"<div class=miss><b>缺 {what}。</b><br>補法：<code>{how}</code></div>"


def sig(v, good_above=1.0):
    """把比值染色：大於門檻是勝、小於是敗。"""
    if v != v:
        return "—"
    cls = "win" if v > good_above else "lose"
    return f'<span class="{cls}">{v:.3f}</span>'


# ================================================================ 01 外部比較


def build_external(out: Path) -> str:
    """與外部方法的比較。本輪的主目標：把 n=5 的七條件對照擴充。"""
    ext_dirs = [ROOT / "runs/hb5"] + [ROOT / f"runs/ext24/g{i}" for i in range(7)]
    ext_dirs += [ROOT / "runs/hb5_pgc"]
    ext = collect_runs([d for d in ext_dirs if d.exists()])
    arm = read_csv(ROOT / "runs/phaseA_human/results.csv")

    body = []
    body.append("<p>外部方法一律走各自論文的<b>原生預算</b>（使用者 2026-08-14 裁定"
                "不對齊）。紋理重相位與其兩個內部對照走人眼門檻。兩邊的預算軸"
                "不同，這是刻意的——十六項指標沒有一項能同時當兩個家族的預算軸"
                "（FND-035）。</p>")

    if not ext:
        return "".join(body) + missing(
            "外部條件的結果",
            "python scripts/apa_baseline.py --out runs/ext24/gN --data data/lo_aligned "
            "--conditions photoguard_c mist dia_r apa_weak --images ...")

    # 逐條件聚合
    per = defaultdict(list)
    for r in ext + arm:
        c = r["condition"]
        e = num(r, "edit_lpips")
        if e is None:
            continue
        per[c].append(r)

    rows, imgs_per_cond = [], {}
    for c in ORDER:
        rs = per.get(c, [])
        if not rs:
            continue
        imgs_per_cond[c] = sorted({r["image"] for r in rs})
        rows.append([
            LABEL.get(c, c), len(imgs_per_cond[c]),
            fmt(mean([num(r, "edit_lpips") for r in rs])),
            fmt(mean([num(r, "fid_dists") for r in rs])),
            fmt(mean([num(r, "fid_lpips") for r in rs])),
            fmt(mean([num(r, "fid_psnr") for r in rs]), 2),
            fmt(mean([num(r, "fid_ssim") for r in rs])),
            fmt(mean([num(r, "edit_clip_drop") for r in rs])),
            fmt(mean([num(r, "total_seconds") for r in rs]), 0),
        ])
    body.append("<h2>一、全條件聚合</h2>")
    body.append("<p><code>edit_lpips</code> 越大代表把編輯推得越遠，是抗編輯的主讀數。"
                "其餘欄是保真度，全部照報不挑選。</p>")
    body.append(table(
        ["條件", "n", "edit_lpips↑", "DISTS↓", "LPIPS↓", "PSNR↑", "SSIM↑",
         "CLIP 掉幅", "秒/圖"], rows))

    # 逐對手各自配對：不同對手跑過的影像不一樣，硬取全體交集會把 n 砍到最小的那個
    body.append("<h2>二、逐對手配對比較</h2>")
    body.append("<p>每一列只用<b>該對手與紋理重相位都跑過</b>的影像，所以每列的 n "
                "不同。硬取全體交集會把樣本數砍到跑得最慢的那個條件。</p>")
    ph_by = {r["image"]: num(r, "edit_lpips") for r in per.get("phase", [])}
    ph_d = {r["image"]: num(r, "fid_dists") for r in per.get("phase", [])}
    prow = []
    for c in ORDER:
        if c == "phase" or c not in per:
            continue
        other = {r["image"]: num(r, "edit_lpips") for r in per[c]}
        other_d = {r["image"]: num(r, "fid_dists") for r in per[c]}
        common = sorted(i for i in other if i in ph_by and ph_by[i] and other[i])
        if not common:
            continue
        a = [ph_by[i] for i in common]
        b = [other[i] for i in common]
        ratio = mean(a) / mean(b)
        wins = sum(1 for x, y in zip(a, b) if x > y)
        prow.append([LABEL.get(c, c), len(common), fmt(mean(a)), fmt(mean(b)),
                     sig(ratio), f"{wins}/{len(common)}",
                     fmt(mean([ph_d[i] for i in common])),
                     fmt(mean([other_d[i] for i in common]))])
    body.append(table(["對手", "n", "紋理重相位", "對手", "倍率", "逐圖勝場",
                       "相位 DISTS", "對手 DISTS"], prow))
    body.append("<div class=note><b>倍率與逐圖勝場要一起讀。</b>兩者可以指向不同的"
                "故事——聚合打平但逐圖大輸，代表本方法在少數影像上贏很多、"
                "在多數影像上小輸。<b>DISTS 兩欄也必須一起讀</b>：外部方法走各自論文的"
                "原生預算，失真水位不同，倍率本身不是公平的比較。</div>")
    body.append("<div class=note><b>比值只報「平均比平均」，不報「逐圖比值的平均」。</b>"
                "後者會被分母支配——同一個缺陷已經讓 <code>retention</code> "
                "不可解讀（FND-037，r = −0.83），詳見第二份報告。</div>")

    # 看圖用的影像：取有 photoguard_c 的那些（它是最強的對手）
    pgc_imgs = sorted({r["image"] for r in per.get("photoguard_c", [])})
    common = pgc_imgs or sorted(imgs_per_cond.get("phase", []))

    # 影像
    body.append("<h2>三、看圖</h2>")
    body.append("<p>每一列是同一張影像：原圖、未防禦的編輯，然後各條件的防禦圖與"
                "其編輯結果。<b>未防禦的編輯必須真的成功</b>，否則抗編輯那一欄的"
                "分母不成立（DEC-022）。</p>")
    shown = common[:3] if common else sorted(imgs_per_cond.get("phase", []))[:2]
    for im in shown:
        body.append(f"<h3>{im}</h3><div class='grid g4'>")
        src = find_image_dir(im, "phase", ext_dirs + [ROOT / "runs/phaseA_human"])
        body.append(f"<figure>{img_tag(src / f'{im}__orig.png')}"
                    f"<figcaption><b>原圖</b>未經任何處理</figcaption></figure>")
        for c in ORDER:
            if c not in per:
                continue
            d = find_image_dir(im, c, ext_dirs + [ROOT / "runs/phaseA_human"])
            if d is None:
                continue
            tag = f"{c}__human" if c in ("phase", "add", "phase_rand") else c
            body.append(
                f"<figure>{img_tag(d / f'{im}__{tag}__def.png')}"
                f"<figcaption><b>{LABEL.get(c,c)}</b>防禦圖</figcaption></figure>"
                f"<figure>{img_tag(d / f'{im}__{tag}__edit_def.png')}"
                f"<figcaption><b>{LABEL.get(c,c)}</b>防禦後的編輯</figcaption></figure>")
        body.append("</div>")
    return "".join(body)


def find_image_dir(image: str, cond: str, dirs: list):
    tag = f"{cond}__human" if cond in ("phase", "add", "phase_rand") else cond
    for d in dirs:
        if (d / f"{image}__{tag}__def.png").exists():
            return d
    for d in dirs:
        if (d / f"{image}__orig.png").exists():
            return d
    return None


# ================================================================ 02 預算與比值


def build_budget(out: Path) -> str:
    """既有 24 張資料的重新分析。不需要新實驗，全部從 phaseA_human 重算。"""
    rows = read_csv(ROOT / "runs/phaseA_human/results.csv")
    if not rows:
        return missing("runs/phaseA_human/results.csv",
                       "python scripts/phase_ablation.py --out runs/phaseA_human "
                       "--data data/lo_aligned --human-threshold")
    by = defaultdict(dict)
    for r in rows:
        by[r["image"]][r["condition"]] = r
    imgs = sorted(i for i in by if {"phase", "add", "phase_rand"} <= set(by[i]))
    P = [by[i]["phase"] for i in imgs]
    A = [by[i]["add"] for i in imgs]
    R = [by[i]["phase_rand"] for i in imgs]
    pe = [num(r, "edit_lpips") for r in P]
    ae = [num(r, "edit_lpips") for r in A]
    re_ = [num(r, "edit_lpips") for r in R]
    ratio = [a / b for a, b in zip(pe, ae)]

    b = []
    b.append(f"<p>來源：<code>runs/phaseA_human</code>，{len(imgs)} 張影像、"
             "三條件、人眼門檻。<b>沒有跑任何新實驗</b>——以下全部是既有資料的"
             "逐圖層級重算。</p>")

    # --- 一、預算穩定度
    b.append("<h2>一、固定的半徑不等於固定的失真</h2>")
    trow = []
    for nm, C, rad in (("紋理重相位", P, "θ = 1.30"),
                       ("加性 δ", A, "ε∞ = 1.2/255"),
                       ("隨機相位", R, "θ = 1.30")):
        for k, lab in (("fid_psnr", "PSNR"), ("fid_dists", "DISTS"),
                       ("fid_lpips", "LPIPS")):
            v = [num(r, k) for r in C]
            m, s = mean(v), sd(v)
            trow.append([f"{nm}（{rad}）", lab, fmt(m, 4), fmt(s, 4),
                         fmt(s / m, 3), f"{min(v):.4f} – {max(v):.4f}"])
    b.append(table(["條件", "指標", "平均", "標準差", "變異係數", "範圍"], trow))
    pp = [num(r, "fid_psnr") for r in P]
    ap = [num(r, "fid_psnr") for r in A]
    b.append(f"<div class='note warn'><b>加性的預算逐圖幾乎是常數，相位不是。</b>"
             f"PSNR 的變異係數：加性 {sd(ap)/mean(ap):.3f}（全距 "
             f"{max(ap)-min(ap):.2f} dB），相位 {sd(pp)/mean(pp):.3f}（全距 "
             f"<b>{max(pp)-min(pp):.2f} dB</b>）。固定 θ 在紋理多的影像上買到的"
             f"失真遠大於紋理少的影像。</div>")

    # --- 二、這個漂移預測誰贏
    b.append("<h2>二、漂移直接預測勝負</h2>")
    corrs = [
        ("比值 vs 相位自身的失真（1/PSNR）", pearson(ratio, [-x for x in pp])),
        ("比值 vs 相位的 DISTS", pearson(ratio, [num(r, "fid_dists") for r in P])),
        ("比值 vs 紋理閘有效佔比", pearson(ratio, [num(r, "active_fraction") for r in P])),
        ("比值 vs 加性的效果（分母）", pearson(ratio, ae)),
        ("比值 vs 相位的效果（分子）", pearson(ratio, pe)),
        ("相位效果 vs 加性效果", pearson(pe, ae)),
        ("相位效果 vs 幅度譜偏差", pearson(pe, [num(r, "amp_dev") for r in P])),
        ("相位效果 vs 紋理閘有效佔比", pearson(pe, [num(r, "active_fraction") for r in P])),
    ]
    b.append(table(["關係", "Pearson r"],
                   [[k, f"{v:+.3f}"] for k, v in corrs]))
    b.append("<div class='note warn'><b>比值與分母的相關是 "
             f"{pearson(ratio, ae):+.3f}。</b>專案已經因為同一個缺陷"
             "（r = −0.83）判定 <code>retention</code> 不可解讀（FND-037）；"
             "這裡的相關更強。分子與分母幾乎無關（r = "
             f"{pearson(pe, ae):+.3f}），而加性效果的變異係數是相位的 "
             f"{(sd(ae)/mean(ae))/(sd(pe)/mean(pe)):.1f} 倍，所以"
             "<b>逐圖比值主要在描述加性在哪些圖上剛好比較弱</b>。</div>")
    b.append("<div class='note ok'><b>沒有被影響的讀數：</b>逐圖勝場"
             f"（{sum(1 for x in ratio if x > 1)}/{len(ratio)}）是配對符號檢定，"
             f"誰贏誰輸是真的；聚合的「平均比平均」= "
             f"<b>{mean(pe)/mean(ae):.3f}</b> 也不受影響。"
             f"不可解讀的是逐圖比值的平均（{mean(ratio):.3f}）與其全距"
             f"（{min(ratio):.3f}–{max(ratio):.3f}）。</div>")

    # --- 三、逐圖表
    b.append("<h2>三、逐圖</h2>")
    trow = []
    for i, im in enumerate(imgs):
        trow.append([
            im, fmt(pe[i]), fmt(ae[i]), sig(ratio[i]),
            fmt(num(P[i], "fid_psnr"), 2), fmt(num(P[i], "fid_dists")),
            fmt(num(P[i], "active_fraction")), fmt(num(P[i], "amp_dev"), 4),
        ])
    b.append(table(["影像", "相位效果", "加性效果", "倍率", "相位 PSNR",
                    "相位 DISTS", "紋理閘佔比", "幅度偏差"], trow))
    lost = [imgs[i] for i in range(len(imgs)) if ratio[i] < 1]
    if lost:
        af = {imgs[i]: num(P[i], "active_fraction") for i in range(len(imgs))}
        lo = sorted(af, key=af.get)[:3]
        b.append(f"<div class=note><b>輸掉的影像：<code>{'、'.join(lost)}</code>。</b>"
                 f"紋理閘佔比最低的三張是 <code>{'、'.join(lo)}</code>——"
                 "兩份名單高度重疊。紋理少的影像撐不起足夠的擾動容量，"
                 "這正是規格 §6 風險三預先寫下的失敗情形。</div>")

    # --- 四、與隨機相位的乾淨對照
    b.append("<h2>四、和隨機相位的比較是唯一預算完全對齊的一組</h2>")
    rr = [a / c for a, c in zip(pe, re_)]
    pd_ = [num(r, "fid_dists") for r in P]
    rd = [num(r, "fid_dists") for r in R]
    lower = sum(1 for a, c in zip(pd_, rd) if a < c)
    b.append("<p>同參數化、同半徑、同損失、同步數、同種子，唯一差別是"
             "<b>有沒有最佳化</b>。</p>")
    b.append(table(["讀數", "值"], [
        ["聚合倍率（平均比平均）", f"{mean(pe)/mean(re_):.3f}"],
        ["逐圖勝場", f"{sum(1 for x in rr if x > 1)}/{len(rr)}"],
        ["相位的 DISTS", fmt(mean(pd_))],
        ["隨機相位的 DISTS", fmt(mean(rd))],
        ["逐圖相位的失真較低", f"{lower}/{len(pd_)}"],
    ]))
    b.append("<div class='note ok'><b>最佳化同時買到更大的效果與更小的失真。</b>"
             "在完全相同的預算下，這是一個嚴格的 Pareto 勝，也是本方法相對於"
             "Galerne 2011 的 RPN 的全部增益。目前文件把它定位成「內部對照」，"
             "以本輪的分析看，它是全專案唯一沒有預算爭議的比較。</div>")
    return "".join(b)


# ================================================================ 03 消融


def build_ablation(out: Path) -> str:
    b = []

    # ---------- 目標影像 ----------
    b.append("<h2>一、換掉損失的目標影像</h2>")
    b.append("<p>損失是 <code>norm(E(x_def) - E(y_target))^2</code>，把防禦圖的 latent "
             "推向 <code>y_target</code> 的 latent。灰圖是既有的選擇（DEC-023）。"
             "FND-031 記過灰圖的高位移有一部分來自「推向無內容」造成的全域降對比；"
             "換目標就是在測這件事有多大。</p>")
    tdirs = sorted((ROOT / "runs/targets").glob("*")) if (ROOT / "runs/targets").exists() else []
    tdata = {d.name: read_csv(d / "results.csv") for d in tdirs if (d / "results.csv").exists()}
    if not tdata:
        b.append(missing("目標消融", "python scripts/phase_ablation.py --out "
                         "runs/targets/NAME --human-threshold --conditions phase "
                         "--target data/targets/NAME.png"))
    else:
        common = None
        for rs in tdata.values():
            s = {r["image"] for r in rs}
            common = s if common is None else (common & s)
        common = sorted(common or [])
        b.append(f"<p>共同影像 {len(common)} 張：<code>{'、'.join(common)}</code></p>")
        trow = []
        for name, rs in sorted(tdata.items()):
            sel = [r for r in rs if r["image"] in common]
            if not sel:
                continue
            trow.append([name, len(sel),
                         mean([num(r, "edit_lpips") for r in sel]),
                         mean([num(r, "fid_dists") for r in sel]),
                         mean([num(r, "fid_psnr") for r in sel]),
                         mean([num(r, "fid_lpips") for r in sel]),
                         mean([num(r, "edit_clip_drop") for r in sel])])
        trow.sort(key=lambda r: -r[2])
        b.append(table(["目標影像", "n", "edit_lpips 越大越好", "DISTS", "PSNR",
                        "LPIPS", "CLIP 掉幅"],
                       [[r[0], r[1], fmt(r[2]), fmt(r[3]), fmt(r[4], 2),
                         fmt(r[5]), fmt(r[6])] for r in trow], best_row=0))
        b.append("<div class=grid style='grid-template-columns:repeat(auto-fit,minmax(130px,1fr))'>"
                 + "".join(
                     f"<figure>{img_tag(ROOT / 'data/targets' / (n + '.png'))}"
                     f"<figcaption><b>{n}</b></figcaption></figure>"
                     for n in sorted(tdata))
                 + "</div>")
        b.append("<div class=note><b>怎麼讀。</b>目標之間若差距很小，代表效果來自"
                 "「把 latent 推離原處」這件事本身，而不是灰圖特有的降對比；"
                 "差距很大則代表 FND-031 記的那個偏差是實質的，"
                 "報告裡必須寫明目標的選擇會影響結論。</div>")

    # ---------- 約束設定 ----------
    b.append("<h2>二、換掉約束落在哪裡（兩個閘的設定）</h2>")
    b.append("<p>三個設定都只改<b>閘</b>，也就是擾動被允許出現的位置，不改損失、"
             "不改更新規則。<code>r_min</code> 是徑向頻率閘的下限（越小越准動低頻），"
             "<code>block</code> 是重疊區塊邊長（擾動的空間尺度），"
             "<code>quantile</code> 是紋理閘的能量參考分位數（越小越多區塊被放行）。</p>")
    gdirs = sorted((ROOT / "runs/gates").glob("*")) if (ROOT / "runs/gates").exists() else []
    gdata = {d.name: read_csv(d / "results.csv") for d in gdirs if (d / "results.csv").exists()}
    base = read_csv(ROOT / "runs/phaseA_human/results.csv")
    if not gdata:
        b.append(missing("約束掃描", "python scripts/phase_ablation.py --out "
                         "runs/gates/NAME --human-threshold --conditions phase "
                         "--r-min 0.25"))
    else:
        common = None
        for rs in gdata.values():
            s = {r["image"] for r in rs if r["condition"] == "phase"}
            common = s if common is None else (common & s)
        common = sorted(common or [])
        rows = []
        base_sel = [r for r in base
                    if r["condition"] == "phase" and r["image"] in common]
        if base_sel:
            rows.append(["定案 block=32 r_min=0.12 q=0.5", len(base_sel),
                         fmt(mean([num(r, "edit_lpips") for r in base_sel])),
                         fmt(mean([num(r, "fid_dists") for r in base_sel])),
                         fmt(mean([num(r, "fid_psnr") for r in base_sel]), 2),
                         fmt(mean([num(r, "active_fraction") for r in base_sel])),
                         fmt(mean([num(r, "amp_dev") for r in base_sel]))])
        for name, rs in sorted(gdata.items()):
            sel = [r for r in rs if r["condition"] == "phase" and r["image"] in common]
            if not sel:
                continue
            rows.append([name, len(sel),
                         fmt(mean([num(r, "edit_lpips") for r in sel])),
                         fmt(mean([num(r, "fid_dists") for r in sel])),
                         fmt(mean([num(r, "fid_psnr") for r in sel]), 2),
                         fmt(mean([num(r, "active_fraction") for r in sel])),
                         fmt(mean([num(r, "amp_dev") for r in sel]))])
        b.append(f"<p>共同影像 {len(common)} 張：<code>{'、'.join(common)}</code></p>")
        b.append(table(["設定", "n", "edit_lpips 越大越好", "DISTS", "PSNR",
                        "紋理閘佔比", "幅度偏差"], rows))
        b.append("<div class=note><b>怎麼讀這張表。</b>效果大但 DISTS／PSNR 也差的"
                 "設定不算贏——那只是花了更多失真。要看的是<b>同一個失真水位下</b>"
                 "誰的效果高。<code>r_min=0.05</code> 的幅度偏差若明顯衝高，代表它"
                 "已經在造新能量而不是重排相位（規格風險一）。</div>")
        if common:
            im = common[0]
            b.append(f"<h3>看圖：{im}</h3><div class='grid g4'>")
            if base_sel:
                b.append("<figure>"
                         + img_tag(ROOT / "runs/phaseA_human" / (im + "__phase__human__def.png"))
                         + "<figcaption><b>定案</b>block 32 / r_min 0.12 / q 0.5</figcaption></figure>")
            for name in sorted(gdata):
                b.append("<figure>"
                         + img_tag(ROOT / "runs/gates" / name / (im + "__phase__human__def.png"))
                         + f"<figcaption><b>{name}</b></figcaption></figure>")
            b.append("</div>")

    # ---------- 操作點：閘與 theta 的取捨 ----------
    b.append("<h3>把省下來的失真用 θ 花掉之後</h3>")
    b.append("<p>上一張表是<b>固定 θ = 1.30</b> 量的，所以拉高 <code>r_min</code> "
             "同時把失真壓低了——效率變好一部分只是因為花得少。真正的問題是"
             "<b>在同一個失真水位上</b>誰的效果高。</p>")
    tdirs2 = sorted((ROOT / "runs/theta").glob("*")) if (ROOT / "runs/theta").exists() else []
    tt = {d.name: read_csv(d / "results.csv") for d in tdirs2 if (d / "results.csv").exists()}
    alt = read_csv(ROOT / "runs/alt_r025/results.csv")
    if not tt:
        b.append(missing("θ 掃描", "python scripts/phase_ablation.py --out "
                         "runs/theta/rR_tT --human-threshold --conditions phase "
                         "--r-min R --phase-radius T"))
    else:
        common2 = None
        for rs in tt.values():
            s2 = {r["image"] for r in rs}
            common2 = s2 if common2 is None else (common2 & s2)
        common2 = sorted(common2 or [])
        rows2 = []
        bs = [r for r in base
              if r["condition"] == "phase" and r["image"] in common2]
        if bs:
            rows2.append(["定案 r 0.12 / θ 1.30", len(bs),
                          fmt(mean([num(r, "edit_lpips") for r in bs])),
                          fmt(mean([num(r, "fid_dists") for r in bs])),
                          fmt(mean([num(r, "fid_lpips") for r in bs])),
                          fmt(mean([num(r, "fid_psnr") for r in bs]), 2),
                          fmt(mean([num(r, "amp_dev") for r in bs]))])
        for name in sorted(tt):
            sel = [r for r in tt[name] if r["image"] in common2]
            if not sel:
                continue
            rows2.append([name, len(sel),
                          fmt(mean([num(r, "edit_lpips") for r in sel])),
                          fmt(mean([num(r, "fid_dists") for r in sel])),
                          fmt(mean([num(r, "fid_lpips") for r in sel])),
                          fmt(mean([num(r, "fid_psnr") for r in sel]), 2),
                          fmt(mean([num(r, "amp_dev") for r in sel]))])
        b.append(table(["設定", "n", "edit_lpips 越大越好", "DISTS", "LPIPS",
                        "PSNR", "幅度偏差"], rows2))
        b.append("<div class='note warn'><b>兩個軸給出相反的答案。</b>"
                 "在同一個 DISTS 上，高 <code>r_min</code> 配高 θ 的效果比定案高；"
                 "但它的 LPIPS 反而更差。這與 FND-026／034 同型——"
                 "<b>操作點的選擇是人眼的事，不是指標的事。</b></div>")
    if alt:
        a_ph = [r for r in alt if r["condition"] == "phase"]
        b0 = [r for r in base if r["condition"] == "phase"
              and r["image"] in {r2["image"] for r2 in a_ph}]
        if a_ph and b0:
            b.append("<h3>換操作點跑滿 24 張</h3>")
            b.append(table(["設定", "n", "edit_lpips", "DISTS", "LPIPS", "PSNR"], [
                ["定案 r 0.12 / θ 1.30", len(b0),
                 fmt(mean([num(r, "edit_lpips") for r in b0])),
                 fmt(mean([num(r, "fid_dists") for r in b0])),
                 fmt(mean([num(r, "fid_lpips") for r in b0])),
                 fmt(mean([num(r, "fid_psnr") for r in b0]), 2)],
                ["r 0.25 / θ 2.6", len(a_ph),
                 fmt(mean([num(r, "edit_lpips") for r in a_ph])),
                 fmt(mean([num(r, "fid_dists") for r in a_ph])),
                 fmt(mean([num(r, "fid_lpips") for r in a_ph])),
                 fmt(mean([num(r, "fid_psnr") for r in a_ph]), 2)]]))
            if common2:
                im2 = sorted({r["image"] for r in a_ph})[0]
                b.append("<div class='grid g3'>"
                         + "<figure>"
                         + img_tag(ROOT / "runs/phaseA_human" / (im2 + "__phase__human__def.png"))
                         + "<figcaption><b>定案</b>r 0.12 / θ 1.30</figcaption></figure>"
                         + "<figure>"
                         + img_tag(ROOT / "runs/alt_r025" / (im2 + "__phase__human__def.png"))
                         + "<figcaption><b>換操作點</b>r 0.25 / θ 2.6</figcaption></figure>"
                         + "<figure>"
                         + img_tag(ROOT / "runs/phaseA_human" / (im2 + "__orig.png"))
                         + "<figcaption><b>原圖</b></figcaption></figure>"
                         + "</div>")

    # ---------- 第二個 prompt ----------
    b.append("<h2>三、換掉編輯 prompt</h2>")
    b.append("<p><code>prompts.yaml</code> 每一類有兩個編輯 prompt，對應兩種惡意"
             "情境：<b>0 號改掉主體</b>（a dog 改成 a cat），<b>1 號保留主體、改場景</b>"
             "（a dog in the park）。既有的全部結果都只用 0 號。</p>"
             "<p>防禦本身與 prompt 無關（損失是 encoder-targeted，完全沒看到文字），"
             "所以這一組測的是<b>同一份防禦在另一種攻擊意圖下還撐不撐得住</b>。</p>")
    p1 = read_csv(ROOT / "runs/pidx1/results.csv")
    if not p1:
        b.append(missing("第二個 prompt 的批次",
                         "python scripts/phase_ablation.py --out runs/pidx1 "
                         "--data data/lo_aligned --human-threshold --prompt-index 1"))
    else:
        p0 = read_csv(ROOT / "runs/phaseA_human/results.csv")
        imgs1 = {r["image"] for r in p1}
        rows = []
        for c in ("phase", "add", "phase_rand"):
            s1 = [r for r in p1 if r["condition"] == c]
            s0 = [r for r in p0 if r["condition"] == c and r["image"] in imgs1]
            if not s1:
                continue
            m0 = mean([num(r, "edit_lpips") for r in s0])
            m1 = mean([num(r, "edit_lpips") for r in s1])
            rows.append([LABEL[c], len(s1), fmt(m0), fmt(m1),
                         fmt(m1 / m0 if m0 == m0 and m0 else float("nan"), 3),
                         fmt(mean([num(r, "edit_clip_drop") for r in s1]))])
        b.append(table(["條件", "n", "prompt 0 改主體", "prompt 1 改場景",
                        "1 對 0 的比", "CLIP 掉幅"], rows))
        ph1 = {r["image"]: num(r, "edit_lpips") for r in p1 if r["condition"] == "phase"}
        ad1 = {r["image"]: num(r, "edit_lpips") for r in p1 if r["condition"] == "add"}
        rd1 = {r["image"]: num(r, "edit_lpips") for r in p1 if r["condition"] == "phase_rand"}
        pair = [(ph1[i], ad1[i]) for i in ph1 if i in ad1 and ad1[i]]
        pr = [(ph1[i], rd1[i]) for i in ph1 if i in rd1 and rd1[i]]
        if pair:
            b.append("<div class='note ok'>在 prompt 1 上，紋理重相位對加性的聚合倍率是 "
                     f"<b>{mean([a for a, _ in pair]) / mean([c for _, c in pair]):.3f}</b>，"
                     f"逐圖勝 {sum(1 for a, c in pair if a > c)}/{len(pair)}。")
            if pr:
                b.append("　對隨機相位是 "
                         f"<b>{mean([a for a, _ in pr]) / mean([c for _, c in pr]):.3f}</b>，"
                         f"逐圖勝 {sum(1 for a, c in pr if a > c)}/{len(pr)}。")
            b.append("</div>")
        b.append("<div class=note><b>兩個 prompt 的絕對值不可直接比大小。</b>"
                 "1 號不要求模型改掉主體，未防禦的編輯本來就改得比較少，"
                 "抗編輯那一欄的分母因此比較小。要看的是同一個 prompt 內"
                 "各條件之間的相對關係。</div>")
    return "".join(b)


# ================================================================ 04 空白地板


def build_floor(out: Path) -> str:
    b = []
    b.append("<p>把<b>未防禦的原圖</b>丟給淨化算子，再編輯，量出來的位移就是"
             "<b>算子自己造成的</b>，與有沒有防禦無關。</p>"
             "<p>沒有這一格，「淨化後某條件的絕對位移量比較大」就無法排除"
             "「該算子本來就把編輯推得比較開」這個平庸解釋。</p>")
    floor = read_csv(ROOT / "runs/hb5/retention_floor.csv")
    if not floor:
        return "".join(b) + missing(
            "空白地板",
            "python scripts/phase_retention.py --run runs/hb5 --seeds 3 --floor "
            "--images man_02 woman_02 dog_03 horse_03 cat_01 "
            "--out runs/hb5/retention_floor.csv")

    ret = []
    for p in sorted((ROOT / "runs/hb5").glob("retention_*.csv")):
        if p.name == "retention_floor.csv":
            continue
        ret += read_csv(p)
    fl = defaultdict(list)
    for r in floor:
        fl[r["purifier"]].append(num(r, "effect_mean"))
    cond_op = defaultdict(lambda: defaultdict(list))
    for r in ret:
        cond_op[r["condition"]][r["purifier"]].append(num(r, "effect_mean"))

    order = ["blur1", "noise0.05", "quantize16", "jpeg75", "jpeg30",
             "crop_resize0.1", "jpeg_then_resize75", "adverse_cleaner", "impress"]
    ops = [o for o in order if o in fl] + [o for o in fl
                                           if o not in order and o != "identity"]
    conds = [c for c in ORDER if c in cond_op]
    head = ["淨化算子", "地板 無防禦"] + [LABEL.get(c, c) for c in conds]
    rows = []
    for o in ops:
        base = mean(fl[o])
        row = [o, "<b>" + fmt(base) + "</b>"]
        for c in conds:
            v = mean(cond_op[c].get(o, []))
            if v != v:
                row.append("—")
            else:
                over = v - base
                cls = "win" if over > 0 else "lose"
                row.append(f'{v:.4f} <span class="{cls}">({over:+.4f})</span>')
        rows.append(row)
    b.append("<h2>一、淨化後的絕對位移量，與地板的差</h2>")
    b.append("<p>括號內是<b>扣掉地板之後真正屬於防禦的位移</b>。</p>")
    b.append(table(head, rows))

    b.append("<h2>二、地板佔了多少</h2>")
    frow = []
    for o in ops:
        base = mean(fl[o])
        ph = mean(cond_op.get("phase", {}).get(o, []))
        frow.append([o, fmt(base), fmt(ph),
                     fmt(base / ph if ph == ph and ph else float("nan"), 3)])
    b.append(table(["淨化算子", "地板", "紋理重相位", "地板佔比"], frow))
    b.append("<div class='note warn'><b>地板佔比接近 1 的算子，其該列的比較不具鑑別力。</b>"
             "既有紀錄已觀察到 <code>crop_resize</code> 之後七個條件收斂到 "
             "0.495 至 0.617、<code>noise 0.05</code> 之後收斂到 0.434 至 0.564，"
             "與有沒有防禦幾乎無關。這張表把那個觀察變成可判定的數字。</div>")
    return "".join(b)


# ================================================================ main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/2026-08-16")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stamp = "2026-08-16 批次"
    specs = [
        ("01_external.html", "與外部方法的比較",
         stamp + "．紋理重相位對 PhotoGuard-c／Mist／DIA-R／APA 弱 baseline",
         build_external),
        ("02_budget.html", "預算漂移與比值的分母支配",
         stamp + "．既有 24 張資料的重新分析，沒有新實驗", build_budget),
        ("03_ablation.html", "目標影像、約束設定、編輯 prompt",
         stamp + "．三組消融", build_ablation),
        ("04_floor.html", "空白地板：淨化算子自己造成的位移",
         stamp + "．effect(purify(原圖))", build_floor),
    ]
    for fn, title, sub, builder in specs:
        html = page(title, sub, builder(args.out))
        (args.out / fn).write_text(html, encoding="utf-8")
        print(f"  {args.out / fn}  ({len(html) // 1024} KB)")

    idx = ["<h2>四份報告</h2><ul>"]
    for fn, title, sub, _ in specs:
        idx.append(f'<li><a href="{fn}">{title}</a> — {sub}</li>')
    idx.append("</ul>")
    idx.append("<h2>文字報告</h2><ul>"
               "<li><code>docs/reference/SURVEY_2026-08-16.md</code> — 文獻查證</li>"
               "<li><code>reports/2026-08-16/05_maintenance.md</code> — 命名與資料維護</li>"
               "<li><code>docs/PHASE_METHOD.md</code> — 方法的完整紀錄</li></ul>")
    (args.out / "index.html").write_text(
        page("2026-08-16 批次報告", stamp, "".join(idx)), encoding="utf-8")
    print(f"  {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
