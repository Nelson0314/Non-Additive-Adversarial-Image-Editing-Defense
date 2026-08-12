#!/usr/bin/env python
"""產生 `runs/index.html` —— 全部批次報告的目錄頁。

    python scripts/report_index.py

存在理由：`runs/` 底下有二十份 HTML 報告，分散在各批次目錄裡，彼此沒有連結。
要看某一批的比對圖得先知道它叫什麼、在哪個資料夾——而批次名（`v14r_merged`、
`s3a_merged`）本身不說明它是什麼實驗。這一頁把 `docs/EXPERIMENTS.md` 的
`EXP-` 說明與磁碟上的產物接起來。

**掃 git 索引而不是檔案系統。** 本 repo 用 sparse-checkout，`runs/` 多半沒有
簽出到工作區；照檔案系統掃會得到一份空清單，而那看起來就只是「這批沒有報告」。

樣式與 `compare_page.py`／`attention_page.py` 逐項相同（同一組字級、邊框色、
深色模式規則），故三種頁面並排看不出接縫。
"""
from __future__ import annotations

import html
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 批次 → (EXP 編碼, 一句話說明)。取自 docs/EXPERIMENTS.md，改那邊要同步改這裡。
# 沒列到的批次仍會出現在「其他」區，不會靜默消失。
BATCHES = [
    ("s3t30_merged", "EXP-s3t30", "site apa + 注意力抑制，τ_train 0.30"),
    ("s3t25_merged", "EXP-s3t25", "site apa + 注意力抑制，τ_train 0.25"),
    ("s3a_merged", "EXP-s3a",
     "site apa + Lo 式 (5)，τ_train 0.50。注意力壓掉約 90%，但編輯成功率"
     "反而由未防禦的 10/15 升到 15/15"),
    ("v14r_merged", "EXP-v14r",
     "SD v1.4 重做，strength 0.4、門檻放寬。位移場與隨機對照無法區分"),
    ("v14_merged", "EXP-v14",
     "SD v1.4 第一輪，strength 0.6。20 個組合沒有一個通過 mean ≥ 3σ"),
    ("b3_merged", "EXP-b3",
     "SDXL 1024²。攻擊在 strength 0.6 上本身很弱，比 v14 更沒有資訊量"),
    ("b1_bird_03", "EXP-b1", "SDXL 首次上機。校準表在錯誤的反演設定下產生，已作廢"),
]

# 檔名 → 這份東西是什麼
KIND = {
    "compare.html": ("人眼比對", "逐格的原圖、防禦圖、編輯輸出並排"),
    "attention.html": ("注意力圖", "各層 cross-attention 的聚合圖"),
    "index.html": (None, None),
    "grid.csv": ("逐格數值", "每一格的 74 個欄位，判讀的原始資料"),
    "protocols.md": ("分層評測", "三層判準一次報完，逐 τ 分列"),
    "unavailable.md": ("算不出來的指標", "逐項寫明為什麼算不出來"),
}

CSS = """
body{font:14px/1.5 system-ui,sans-serif;margin:1.5rem;max-width:64rem}
h1{margin:0 0 .2rem}
h2{margin:2rem 0 .4rem;font-size:1.05rem}
p.lead{background:#8881;padding:.6rem .8rem;border-radius:.3rem;margin:.6rem 0 1.6rem}
table{border-collapse:collapse;margin:.4rem 0 1rem;width:100%}
th,td{border-bottom:1px solid #8884;padding:.35rem .5rem;vertical-align:top}
th{text-align:left;font-weight:600;font-size:.85em;color:#666;
   text-transform:uppercase;letter-spacing:.06em}
td.exp{white-space:nowrap;font-weight:600}
td.desc{color:#444}
td.links a{margin-right:.7rem;white-space:nowrap}
td.miss{color:#888}
code{background:#8881;padding:0 .25rem;border-radius:.2rem}
.note{color:#666;font-size:.92em}
@media(prefers-color-scheme:dark){body{background:#111;color:#ddd}
  th,.note{color:#999}td.desc{color:#bbb}}
"""


def tracked() -> set[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "runs"],
                         capture_output=True, text=True, encoding="utf-8")
    return set(out.stdout.split())


def row(batch: str, exp: str, desc: str, files: set[str]) -> str:
    links = []
    for name, (label, _) in KIND.items():
        if label is None:
            continue
        p = f"runs/{batch}/{name}"
        if p in files:
            links.append(f'<a href="{batch}/{html.escape(name)}">{label}</a>')
    # 報告以外的兩份走批次層級的檔名（class_margin / eval_protocols 的輸出）
    for suffix, label in (("_margin.csv", "類別 margin"),):
        p = f"runs/{batch.replace('_merged', '')}{suffix}"
        if p in files:
            links.append(f'<a href="{Path(p).name}">{label}</a>')
    prot = f"runs/{batch.replace('_merged', '')}_protocols/protocols.md"
    if prot in files:
        links.append(f'<a href="{batch.replace("_merged", "")}_protocols/'
                     f'protocols.md">分層評測</a>')
    cell = " ".join(links) if links else '<span class="miss">尚未產生</span>'
    return (f'<tr><td class="exp">{html.escape(exp)}</td>'
            f'<td class="desc">{html.escape(desc)}<br>'
            f'<code>runs/{html.escape(batch)}/</code></td>'
            f'<td class="links">{cell}</td></tr>')


def main() -> int:
    files = tracked()
    if not files:
        print("git ls-files runs 回傳空集合——不在 repo 內，或 runs/ 未入版控",
              file=sys.stderr)
        return 1

    rows = [row(b, e, d, files) for b, e, d in BATCHES]
    listed = {b for b, _, _ in BATCHES}
    others = sorted({p.split("/")[1] for p in files if "/" in p[5:]}
                    - listed - {b.replace("_merged", "") for b in listed})

    page = f"""<title>報告目錄 · WACV 非加性抗編輯防禦</title>
<style>{CSS}</style>
<h1>報告目錄</h1>
<p class="lead">每一批實驗的可看產物都列在這裡。批次的完整說明、
它確立了什麼、產物路徑，見 <code>docs/EXPERIMENTS.md</code> 的對應
<code>EXP-</code> 條目；結論與判準見 <code>docs/FINDINGS.md</code> 與
<code>docs/DECISIONS.md</code>。</p>

<h2>批次</h2>
<table>
<thead><tr><th>編碼</th><th>是什麼</th><th>可看的產物</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>

<h2>怎麼讀</h2>
<table>
<thead><tr><th>產物</th><th>回答什麼問題</th></tr></thead>
<tbody>
{chr(10).join(f'<tr><td class="exp">{lab}</td><td class="desc">{d}</td></tr>'
              for lab, d in KIND.values() if lab)}
<tr><td class="exp">類別 margin</td><td class="desc">
每張編輯輸出對「目標類」與「原類」的 SigLIP 差，符號即判定。
<strong>這是診斷量，不是主判準</strong>——高預算下它是反的，見 FND-012</td></tr>
</tbody>
</table>

<p class="note"><strong>主判準是連續量。</strong>
看「防禦把編輯輸出推得多遠」（<code>grid.csv</code> 的 <code>edit_lpips</code>
等第 1 層欄位，DAYN Table 1 的形式），不是二元的成功／失敗。理由見
<code>docs/DECISIONS.md</code> 的 DEC-006。</p>

<p class="note">未列在上表的 <code>runs/</code> 目錄（先驗實驗與探測）：
{', '.join(f'<code>{html.escape(o)}</code>' for o in others) or '（無）'}。
它們的地位見 <code>docs/EXPERIMENTS.md</code> 的 EXP-prior。</p>
"""
    out = ROOT / "runs" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"寫入 {out}（{len(rows)} 個批次、{len(others)} 個其他目錄）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
