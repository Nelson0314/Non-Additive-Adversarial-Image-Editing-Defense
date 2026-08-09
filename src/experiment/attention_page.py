"""attention map 對照頁 —— `reference/CODE_CONTRACTS.md` §4.2。

`compare.html` 回答「編輯被擋住了嗎」，這一頁回答**為什麼**：防禦是不是
真的讓 cross-attention 失效，以及淨化是不是把那個擾亂洗掉了。

三篇對比論文與本專案的一個條件都以 attention 為著力點：

| 方法 | 著力點 |
|---|---|
| AdvPaint | self-attn 的 Q／K／V + cross-attn 的 Q |
| PromptFlare | cross-attention decoy（把輸出拉向「只 attend BOS」） |
| **N1**（本專案） | 把注意力質量導向 shared token |

沒有這一頁，「注意力被打散了」這句話在報告裡沒有任何可查證的依據。

## 一列三張圖

    防禦側聚合圖 ‖ 對照側聚合圖 ‖ 逐層圖（僅 attn_full 的格）

前兩張並排即是判讀的全部：同一個淨化、同一個種子、同一個 prompt，
唯一的差別是防禦有沒有開。第三張是 `attn_full` 那一組才有的逐層原圖
（`CODE` §4.2 的體積控制），此處以連結列出而不內嵌——70 層 × 全部格點
內嵌會讓頁面無法使用。

## 數值一律指向 attn_stats.csv

`save_heatmap` 對每張圖各自正規化到自己的最大值（各層的質量尺度相差數個
量級，共用尺度會讓深層全黑）。**故不同圖之間的亮度不可直接比較**，
數值結論一律以 `attn_stats.csv` 為準。這一條寫在頁面上，不只寫在程式裡。
"""

from html import escape
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.experiment import grid
from src.experiment.compare_page import _as_purify, _num, _purify_label, slug


def _attn_arts(cell: Optional[Dict[str, Any]], tag: str) -> Dict[str, Any]:
    """一格的 attention 產物：聚合圖、逐層圖清單、統計表。"""
    out: Dict[str, Any] = {"agg": None, "layers": [], "stats": None}
    if not cell:
        return out
    for a in cell.get("artifacts") or []:
        if a.endswith("attn_stats.csv"):
            out["stats"] = a
        elif a.endswith(f"{tag}_agg.png"):
            out["agg"] = a
        elif "/attn/" in a.replace("\\", "/") and "_layer" in a:
            out["layers"].append(a)
    out["layers"].sort()
    return out


def _img(path: Optional[str], alt: str) -> str:
    if not path:
        return '<td class="miss" title="產物不存在">缺</td>'
    p = escape(path)
    return (f'<td><a href="{p}"><img loading="lazy" src="{p}" '
            f'alt="{escape(alt)}"></a></td>')


def _links(paths: Sequence[str], stats: Optional[str]) -> str:
    bits = []
    if stats:
        bits.append(f'<a href="{escape(stats)}">attn_stats.csv</a>')
    if paths:
        bits.append(f'{len(paths)} 張逐層圖：'
                    + " ".join(f'<a href="{escape(p)}">'
                               f'{escape(p.rsplit("_", 1)[-1][:-4])}</a>'
                               for p in paths[:70]))
    return "<td class='links'>" + (" · ".join(bits) or "—") + "</td>"


def build_attention_html(cells: Sequence[Dict[str, Any]], batch: str = "",
                         env: Optional[Dict[str, Any]] = None) -> str:
    """由 `_cells/*.json` 產生對照頁。**不碰 GPU、不讀影像內容。**"""
    by_id = {c["id"]: c for c in cells}
    evals = [c for c in cells
             if c.get("stage") == "eval" and c.get("status") == "done"]

    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for c in evals:
        cfg = c.get("config") or {}
        key = (c.get("image") or c.get("image_id") or "",
               cfg.get("tau"), _as_purify(cfg.get("purify")))
        groups.setdefault(key, []).append(c)

    body, index = [], []
    for key in sorted(groups, key=lambda k: (k[0], k[1] if k[1] is not None
                                             else -1.0, _purify_label(k[2]))):
        img, tau, purify = key
        anchor = "a-" + slug(img, "-", tau, "-", _purify_label(purify))
        title = f"{img} · τ={_num(tau, 3)} · 淨化 {_purify_label(purify)}"
        index.append(f'<a href="#{anchor}">{escape(title)}</a>')

        rows = []
        for c in sorted(groups[key],
                        key=lambda c: (str(c.get("condition")),
                                       (c.get("config") or {}).get("seed") or 0)):
            cfg = c.get("config") or {}
            cond, seed = c.get("condition") or "", cfg.get("seed")
            tag = f"tau{float(tau):g}_seed{seed}" if tau is not None \
                else f"seed{seed}"
            mine = _attn_arts(c, tag)
            ctrl = _attn_arts(
                by_id.get(grid.Cell("control", "phi0", img, purify=purify,
                                    seed=seed).cell_id()),
                f"seed{seed}")
            full = '<span class="flag">全層</span>' if c.get("attn_full") else ""
            rows.append(
                f'<tr><th scope="row">{escape(cond)}<br><small>seed {seed}'
                f'</small>{full}</th>'
                + _img(mine["agg"], f"{cond} 防禦側聚合")
                + _img(ctrl["agg"], f"{cond} 對照側聚合")
                + _links(mine["layers"], mine["stats"])
                + "</tr>"
            )
        body.append(
            f'<section id="{anchor}"><h2>{escape(title)}</h2>'
            f'<table><tr><th>條件</th><th>防禦側聚合</th><th>對照側聚合</th>'
            f'<th>逐層與數值</th></tr>{"".join(rows)}</table></section>'
        )

    e = env or {}
    meta = (f"gpu={escape(str(e.get('gpu', '?')))} · "
            f"precision={escape(str(e.get('precision', '?')))} · "
            f"commit={escape(str(e.get('commit', '?')))}")
    return _PAGE.format(batch=escape(batch), meta=meta, n_groups=len(groups),
                        index=" · ".join(index), body="".join(body))


_PAGE = """<title>attention {batch}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:1.5rem;max-width:80rem}}
table{{border-collapse:collapse;margin:.4rem 0 1rem;width:100%}}
th,td{{border-bottom:1px solid #8884;padding:.25rem;vertical-align:top}}
th[scope=row]{{text-align:left;white-space:nowrap}}
img{{width:100%;max-width:14rem;height:auto;display:block;
  background:#8881;image-rendering:pixelated}}
td.miss{{color:#c33;text-align:center;font-weight:600}}
td.links{{font-size:.8em;color:#666;word-break:break-all}}
.flag{{display:inline-block;margin-left:.3rem;padding:0 .3rem;
  border-radius:.2rem;background:#4a9;color:#fff;font-size:.72em}}
.index{{font-size:.85em;line-height:1.9;color:#666;margin-bottom:1.5rem}}
.lead{{background:#8881;padding:.6rem .8rem;border-radius:.3rem}}
section{{margin-bottom:2rem}}
@media(prefers-color-scheme:dark){{body{{background:#111;color:#ddd}}
  td.links,.index{{color:#999}}}}
.nav{{font-size:.85em;margin:0 0 1rem;padding-bottom:.5rem;
  border-bottom:1px solid #8884}}
.nav a{{margin-right:.9rem}}
.nav .here{{font-weight:600;color:#666}}
</style>
<p class="nav"><a href="../index.html">← 報告目錄</a><a href="compare.html">人眼比對</a><span class="here">注意力圖</span><a href="grid.csv">逐格數值</a></p>
<h1>cross-attention 對照 · batch {batch}</h1>
<p>{meta}</p>
<p class="lead"><b>每張熱圖各自正規化到自己的最大值</b>——各層的質量尺度
相差數個量級，共用尺度會讓深層全黑。故<b>不同圖之間的亮度不可直接比較</b>，
數值結論一律以 <code>attn_stats.csv</code> 為準。逐層原圖只在標記「全層」的
那一組完整存（主表所在的 τ、seed 0），其餘格點只有聚合圖與統計表；
理由是體積，不是那些格點不重要。</p>
<p>{n_groups} 組 ×（影像, τ, 淨化）。</p>
<div class="index">{index}</div>
{body}
"""
