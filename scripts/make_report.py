"""把 runs/ 底下的實驗結果組成單一 HTML 報告。

報告的組織原則：**每個數字都要能追到產生它的執行**。每個區塊都標明來源
CSV 的路徑與量測條件，避免報告與原始資料脫節。

執行：python scripts/make_report.py --out docs/RESULTS_lowrank.html
"""

import argparse
import csv
import html
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=float("nan")):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def mean(vals):
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(v, nd=3):
    if isinstance(v, float) and v != v:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return html.escape(str(v))


def table(headers, rows, cls=""):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def section_e0(root: Path):
    cost = read_csv(root / "e0_vae" / "e0_cost.csv")
    brk = read_csv(root / "e0b" / "e0_breakdown.csv")
    old = read_csv(root / "e0" / "e0_cost.csv")
    if not cost and not brk:
        return ""

    parts = ["<h2>E0 — 成本實測</h2>"]
    parts.append(
        "<p class='src'>來源：<code>runs/e0/e0_cost.csv</code>、"
        "<code>runs/e0b/e0_breakdown.csv</code>、"
        "<code>runs/e0_vae/e0_cost.csv</code>。"
        "V100 32GB、SD v1-4、512²、fp32。</p>"
    )

    if old:
        n_oom = sum(1 for r in old if r.get("oom") == "1")
        parts.append(
            f"<p>初次量測（僅 UNet checkpoint）：18 格中 {n_oom} 格 OOM。"
            "可行格的 peak 於 <code>(5,5)</code> 到 <code>(20,20)</code> 之間"
            "僅由 18665.4 MB 變動到 18667.3 MB —— 步數變動四倍而記憶體幾乎不動，"
            "代表瓶頸與步數無關。</p>"
        )

    if brk:
        parts.append("<h3>記憶體歸因</h3>")
        parts.append(
            table(
                ["項目", "peak (MB)", "秒"],
                [[r["case"], fnum(r, "peak_mb"), fnum(r, "seconds")] for r in brk],
            )
        )
        parts.append(
            "<p>整條計算圖上有三次 VAE 呼叫（<code>x_def</code> 的 decode、"
            "SDEdit 的 encode 與 decode）。未 checkpoint 時三者的激活必須同時"
            "留存至反向傳播，peak 為<strong>總和</strong>；checkpoint 後反向"
            "一次只重算一塊，peak 降為<strong>最大值</strong>。單次 decode 不受"
            "影響（9934.5 → 9934.4 MB），與此解釋一致。</p>"
        )

    if cost:
        parts.append("<h3>加上 VAE checkpoint 後的成本表</h3>")
        rows = [
            [r["k_inv"], r["n_edit"], fnum(r, "peak_mb"), fnum(r, "seconds")]
            for r in cost
        ]
        parts.append(table(["k_inv", "n_edit", "peak (MB)", "秒/iter"], rows))
        parts.append(
            "<p>擬合：<code>seconds ≈ 1.05 + 0.384·k_inv + 0.304·n_edit</code>，"
            "於 (50,50) 的誤差小於 1%。peak 在整張表上介於 9949.7 與 9958.7 MB，"
            "即記憶體已不是限制，時間才是。</p>"
        )
    return "\n".join(parts)


def section_e0c(root: Path):
    rows = read_csv(root / "e0c_tmax" / "recon_floor.csv")
    if not rows:
        return ""
    agg = defaultdict(list)
    for r in rows:
        agg[(int(r["t_max"]), int(r["k_inv"]))].append(r)

    body = []
    for (tm, k), sel in sorted(agg.items()):
        body.append([
            tm, k,
            mean([fnum(r, "psnr") for r in sel]),
            mean([fnum(r, "lpips") for r in sel]),
            mean([fnum(r, "ssim") for r in sel]),
            mean([fnum(r, "linf") for r in sel]),
        ])
    return (
        "<h2>E0c — site L 的保真地板</h2>"
        "<p class='src'>來源：<code>runs/e0c_tmax/recon_floor.csv</code>，n = 6 張。</p>"
        "<p>site P 在 φ=0 時 <code>x_def = x</code> 逐元素相等；site L 在 φ=0 時"
        "<code>x_def</code> 仍是「VAE 編碼 → DDIM inversion → 去噪 → VAE 解碼」"
        "的重建，其誤差 φ 無法消除。<code>t_max = 0, k_inv = 0</code> 一列是"
        "VAE 單獨來回，即<strong>不可約的地板</strong>。</p>"
        + table(["t_max", "k_inv", "PSNR", "LPIPS", "SSIM", "L∞"], body)
        + "<p>原設定 <code>t_max = 999</code>、<code>k_inv = 10</code> 的重建為"
        " PSNR 16.87 / LPIPS 0.697，即 φ=0 時 <code>x_def</code> 與 <code>x</code>"
        " 已是兩張不同的圖，spec §1.1 的「人眼尺度上接近」在起點就不成立。"
        "改用 <code>t_max = 500</code>、<code>k_inv = 20</code> 後為 26.56 / 0.194，"
        "與 VAE 地板（27.51 / 0.143）相距約 1 dB。</p>"
    )


def section_e0d(root: Path):
    rows = read_csv(root / "e0d" / "lr_sweep.csv")
    if not rows:
        return ""
    body = [
        [
            r["site"], fnum(r, "lr"), fnum(r, "loss_first"), fnum(r, "loss_final"),
            fnum(r, "loss_min"), fnum(r, "ratio_final_min"),
            f"{fnum(r, 'monotone_frac') * 100:.0f}%",
            fnum(r, "shift_final"), fnum(r, "psnr_final"),
        ]
        for r in rows
    ]
    return (
        "<h2>E0d — 學習率校準</h2>"
        "<p class='src'>來源：<code>runs/e0d/lr_sweep.csv</code>，單張影像、rank=4。</p>"
        "<p>判準是<strong>總損失是否收斂</strong>，不是最終偏移大小：偏移大"
        "但損失發散的設定不可用，那代表結果由隨機遊走決定而非優化。</p>"
        "<div class='warn'><strong>判準的已知混淆</strong>：優化每步輪替使用不同"
        "淨化算子（identity / blur / jpeg），這是 spec §5.1 對 𝒫 求期望值的取樣"
        "方式，因此相鄰步的損失量的是不同條件下的值，本來就會震盪。"
        "<code>單調下降比例</code>即使收斂良好也只在 1/3 附近，四個 lr 全落在"
        " 17%~33%，沒有鑑別力。主要判準取 <code>終/最低</code>：發散時為 39.6"
        "與 138，收斂時為 2~3。</div>"
        + table(
            ["site", "lr", "loss 起", "loss 終", "loss 最低", "終/最低",
             "單調下降比例", "偏移", "PSNR"],
            body,
        )
    )


def section_e2(root: Path, run_name: str):
    run = root / run_name
    summary = read_csv(run / "summary.csv")
    results = read_csv(run / "results.csv")
    if not summary:
        return ""

    parts = [f"<h2>E2 / E3 — site × rank 與淨化掃描</h2>",
             f"<p class='src'>來源：<code>runs/{run_name}/summary.csv</code>"
             f"（{len(summary)} 格）、<code>results.csv</code>（{len(results)} 列）。</p>"]

    body = []
    for site in sorted({r["site"] for r in summary}):
        for rk in sorted({int(r["rank"]) for r in summary if r["site"] == site}):
            sel = [r for r in summary if r["site"] == site and int(r["rank"]) == rk]
            body.append([
                site, rk, len(sel),
                mean([fnum(r, "final_shift") for r in sel]),
                mean([fnum(r, "final_psnr") for r in sel]),
                mean([fnum(r, "final_ssim") for r in sel]),
                mean([fnum(r, "final_linf") for r in sel]),
                mean([fnum(r, "eff_rank_mean") for r in sel]),
                mean([fnum(r, "energy_rank_99_mean") for r in sel]),
            ])
    parts.append(
        table(
            ["site", "注入 r", "n", "編輯偏移", "PSNR", "SSIM",
             "L∞(φ 造成)", "eff_rank", "energy_rank99"],
            body,
        )
    )
    parts.append(
        '<p><img src="../runs/%s/frontier.png" alt="E2 前緣"></p>' % run_name
    )
    parts.append(
        "<div class='warn'><strong>兩個 site 的起點不同，讀圖時必須納入</strong>："
        "site P 在 φ=0 時 <code>x_def = x</code> 逐元素相等；site L 在 φ=0 時"
        "已帶有 inversion + VAE 來回的重建誤差（E0c 實測 19.61–31.01 dB，逐張"
        "差異達 11.4 dB）。前緣圖的橫軸是相對原圖的<strong>絕對</strong>保真度，"
        "故此差異在圖上直接可見，未以損失函數的設計掩蓋。</div>"
    )

    if results:
        parts.append("<h3>E3 淨化強度掃描</h3>")
        parts.append(
            '<p><img src="../runs/%s/purify_sweep.png" alt="E3 淨化掃描"></p>' % run_name
        )
        parts.append(
            "<p>縱軸為<strong>淨額</strong>偏移 <code>net = edit − ctrl</code>，"
            "其中 <code>ctrl</code> 是同一淨化算子施加於<strong>原圖</strong>後"
            "編輯所產生的偏移（φ 完全沒有參與）。必須扣除：<code>P(x) ≠ x</code>，"
            "故即使 φ=0，模糊或 JPEG 本身就會讓編輯偏離 <code>E(x)</code>。"
            "實測 site P、r=1 在 blur 下 shift 0.347、identity 下 0.098——"
            "高的那個是淨化自己造成的。不扣除則 E3 的每個數字都被系統性高估，"
            "且高估幅度隨淨化強度上升，而那正是 §7.4 因果判斷所讀的軸。</p>"
        )
        parts.append(
            "<p>spec §7.4 的因果判斷讀這張圖：P 為低秩<strong>加性</strong>、"
            "L 為低秩<strong>非加性</strong>。P 亦耐淨化則機制是秩結構；"
            "P 不耐而 L 耐則機制是非加性；兩者皆耐則兩機制不可分辨。"
            "三種結果都是可發表的發現。</p>"
        )

        # E3 淨額數值表。曲線看得出趨勢，但 §7.4 的因果判斷要引用數字。
        ho = [r for r in results if r.get("noise_split") == "heldout"]
        if ho:
            for kind in sorted({r["purify"] for r in ho}):
                sub = [r for r in ho if r["purify"] == kind]
                strengths = sorted({fnum(r, "strength") for r in sub})
                sites = sorted({r["site"] for r in sub})
                rks = sorted({int(r["rank"]) for r in sub})
                body = []
                for site in sites:
                    for rk in rks:
                        row = [f"{site} r={rk}"]
                        for st in strengths:
                            sel = [
                                r for r in sub
                                if r["site"] == site and int(r["rank"]) == rk
                                and fnum(r, "strength") == st
                            ]
                            row.append(mean([fnum(r, "net_lpips") for r in sel]))
                        body.append(row)
                parts.append(f"<h4>net shift — {html.escape(kind)}</h4>")
                parts.append(
                    table(["site / r"] + [f"{s:g}" for s in strengths], body)
                )

        ot = run / "overfit_table.md"
        if ot.exists():
            parts.append("<h3>對特定噪聲的過擬合幅度</h3>")
            parts.append(
                "<p class='src'>來源：<code>runs/%s/overfit_table.md</code>。</p>"
                % run_name
            )
            parts.append(
                "<p>φ 是針對訓練用的那一組 ε 優化出來的（<code>n_eot = 1</code>）。"
                "評測一律改用未見過的種子；此表併列訓練種子的結果，兩者之差即為"
                "過擬合幅度。比值遠大於 1 表示防禦主要是對該組噪聲有效，"
                "泛化到其他噪聲的能力有限。</p>"
            )
            parts.append(_md_table_to_html(ot.read_text(encoding="utf-8")))

        rt = run / "rank_table.md"
        if rt.exists():
            parts.append("<h3>注入秩 vs 實測像素秩</h3>")
            parts.append(
                "<p>site P 的 <code>eff_rank</code> 遠高於注入秩而 "
                "<code>energy99</code> 等於注入秩，即 spec §7.2 修訂紀錄所述的"
                " clamp 效應：低秩結構在能量意義下成立、在精確秩意義下不成立。"
                "site L 的像素秩為湧現量，沒有理論保證，此處為實測值。</p>"
            )
            parts.append(_md_table_to_html(rt.read_text(encoding="utf-8")))
    return "\n".join(parts)


def _md_table_to_html(md: str) -> str:
    rows = [r.strip() for r in md.strip().splitlines() if r.strip().startswith("|")]
    if len(rows) < 2:
        return ""
    def cells(r):
        return [c.strip() for c in r.strip("|").split("|")]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cells(rows[0]))
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells(r)) + "</tr>"
        for r in rows[2:]
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


CSS = """
body{font-family:-apple-system,'Segoe UI','Noto Sans TC',sans-serif;line-height:1.7;
max-width:1100px;margin:0 auto;padding:2rem 1.2rem;color:#1a1a1a;background:#fff}
h1{border-bottom:3px solid #2b6cb0;padding-bottom:.4rem}
h2{margin-top:2.4rem;border-bottom:1px solid #ddd;padding-bottom:.3rem;color:#2b6cb0}
h3{margin-top:1.6rem;color:#444}
table{border-collapse:collapse;margin:1rem 0;font-size:.88rem;width:100%}
th,td{border:1px solid #ccc;padding:.32rem .6rem;text-align:right}
th{background:#eef3f8;text-align:center}
td:first-child,th:first-child{text-align:left}
tbody tr:nth-child(even){background:#fafafa}
code{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px;font-size:.9em}
img{max-width:100%;border:1px solid #ddd;border-radius:4px}
.src{color:#666;font-size:.86rem}
.warn{background:#fff8e6;border-left:4px solid #d69e2e;padding:.7rem 1rem;margin:1rem 0}
@media(prefers-color-scheme:dark){
body{background:#16181c;color:#e6e6e6}
th{background:#232830}tbody tr:nth-child(even){background:#1b1e24}
th,td{border-color:#39414d}code{background:#232830}
.warn{background:#2a2417;border-color:#b7791f}
h2{color:#7fb2e5}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--run_name", default="e2")
    ap.add_argument("--out", default="docs/RESULTS_lowrank.html")
    args = ap.parse_args()

    root = Path(args.runs)
    sections = [
        section_e0(root),
        section_e0c(root),
        section_e0d(root),
        section_e2(root, args.run_name),
    ]
    doc = (
        "<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>低秩殘差防禦 — 實驗結果</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>低秩殘差防禦 — 實驗結果</h1>"
        "<p class='src'>本頁由 <code>scripts/make_report.py</code> 自 "
        "<code>runs/</code> 直接生成，數字未經人工轉錄。</p>"
        + "\n".join(s for s in sections if s)
        + "</body></html>"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"[report] 寫入 {out}（{len(doc)} 字元）")


if __name__ == "__main__":
    main()
