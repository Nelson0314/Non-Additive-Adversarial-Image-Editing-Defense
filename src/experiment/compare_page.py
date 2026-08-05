"""人眼比對頁 —— `DESIGN_2026-08-05.md` §1.1 的**主判準**產物。

使用者 2026-08-05 定案：判準以人眼為主、數值指標為輔。`compare.html` 因此
由附屬產物升格為主要產出物，**每一格都必須有影像可看**；指標與人眼矛盾時
以人眼為準並記錄該矛盾。

## 為什麼一定要有這一頁

先驗實驗的三個門檻與全部否定結論，都是靠逐張看圖定出或確認的。三個實例：

- 編輯 LPIPS 0.2593 聽起來像「有部分效果」，逐張看是「編輯照樣成功，
  只是細節略有不同」。
- 同為 LPIPS 0.05 時，四種失真的 PSNR 由 24.44 到 36.47（差 12 dB），
  人眼看到的是四種不同的東西。
- E2–E23 全部在防禦一個不存在的攻擊，是使用者看 `compare.html` 才發現的
  （「連原始圖片被文字編輯都沒有成功」）。

## 一列六張圖，順序即因果鏈

    原圖 → 防禦圖 → 殘差 → 淨化後 → 防禦側編輯 ‖ 對照側編輯

前三張回答「防禦看得出來嗎」（主張三），後三張回答「編輯被擋住了嗎」
（主張一與二）。最後兩張並排是整頁的重點：同一個淨化、同一個種子、
同一個 prompt，唯一的差別是防禦有沒有開。

## 版面：同一組 (影像, τ, 淨化) 下九個條件並排成列

這個分組不是任意的。要判的是「非加性在**同失真、同淨化**下是否勝過加性」，
故必須讓九個條件在其他變因全部相同時上下對照。反過來以條件分組會讓
「N1 在 jpeg30 下的圖」與「photoguard 在 jpeg30 下的圖」隔開數百列。

## 體積

4,050 個 eval 格 × 6 張圖。全部 `loading="lazy"`，瀏覽器只載入捲到的部分。
seed 0 直接展開，其餘種子收在 `<details>` 內——種子之間的差異是量測噪聲，
逐張看的價值遠低於條件之間的差異，但仍必須可看（`DESIGN` §1.1 的
「每一格都必須有影像」不排除任何一格）。
"""

from html import escape
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.experiment import grid

# 一列的六張圖，(標題, 來源) 依因果鏈排列。來源在 `_row_images` 中解析。
COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("原圖", "orig"),
    ("防禦圖", "x_def"),
    ("殘差", "residual"),
    ("淨化後", "x_purified"),
    ("防禦側編輯", "edit_def"),
    ("對照側編輯", "edit_ctrl"),
)

# 表頭要顯示的數值。**排在影像之後**，位階由版面本身表達。
METRICS: Tuple[Tuple[str, str], ...] = (
    ("effect_siglip", "效果"),
    ("retention", "保留率"),
    ("fid_lpips", "LPIPS"),
    ("fid_psnr", "PSNR"),
)


def _num(v: Any, digits: int = 4) -> str:
    if v is None or v == "":
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return escape(str(v))
    if f != f:                      # NaN
        return "—"
    return f"{f:.{digits}g}"


def _artifact(cell: Optional[Dict[str, Any]], suffix: str) -> Optional[str]:
    """從一格的 `artifacts` 取出以 `suffix` 結尾的相對路徑。

    用 `artifacts` 而不是自行拼路徑：那份清單是實際寫出去的檔案，
    自行拼路徑會在產物改名時給出一堆壞掉的 `<img>`，而頁面照樣渲染得出來。
    """
    if not cell:
        return None
    for a in cell.get("artifacts") or []:
        if a.endswith(suffix):
            return a
    return None


def _row_images(cell: Dict[str, Any], by_id: Dict[str, Dict[str, Any]],
                cond: str, img: str, tau: Any, seed: Any) -> Dict[str, Optional[str]]:
    """六張圖的相對路徑。缺的回 None，由 `_img_cell` 顯示成缺圖標記。"""
    train = by_id.get(grid.Cell("train", cond, img).cell_id())
    ray = by_id.get(grid.Cell("rayscale", cond, img, tau=tau).cell_id())
    purify = cell.get("config", {}).get("purify") or (
        cell.get("purify_kind"), cell.get("purify_strength"))
    ctrl = by_id.get(
        grid.Cell("control", "phi0", img,
                  purify=_as_purify(purify), seed=seed).cell_id()
    )
    return {
        "orig": _artifact(train, "orig.png"),
        "x_def": _artifact(ray, "x_def.png") or _artifact(train, "x_def.png"),
        "residual": _artifact(ray, "residual.png")
        or _artifact(train, "residual.png"),
        "x_purified": _artifact(cell, "x_purified.png"),
        "edit_def": _artifact(cell, f"edit_seed{seed}.png"),
        "edit_ctrl": _artifact(ctrl, f"edit_seed{seed}.png"),
    }


def _as_purify(p) -> Optional[Tuple[str, float]]:
    if p is None:
        return None
    if isinstance(p, (list, tuple)) and len(p) == 2:
        kind, strength = p
        if kind is None:
            return None
        return (str(kind), float(strength))
    return None


def _img_cell(path: Optional[str], alt: str) -> str:
    if not path:
        return '<td class="miss" title="產物不存在">缺</td>'
    p = escape(path)
    return (f'<td><a href="{p}"><img loading="lazy" src="{p}" '
            f'alt="{escape(alt)}"></a></td>')


def slug(*parts: Any) -> str:
    """把任意字串壓成只含 `[A-Za-z0-9_-]` 的錨點。

    影像識別碼來自檔名，不是受控字串。直接拼進 `id=` 與 `href="#..."`
    會讓一個含引號或角括號的名稱破壞整頁的結構——而頁面仍會渲染出東西，
    只是後半段全部落在屬性值裡面。跳脫（`escape`）在此不夠：跳脫後的
    `&quot;` 放進錨點雖然安全，卻使 `href` 與 `id` 對不上而失去導覽功能。
    """
    out = []
    for p in parts:
        for ch in str(p):
            out.append(ch if (ch.isascii() and (ch.isalnum() or ch in "-_"))
                       else "_")
    return "".join(out).strip("_") or "x"


def _group_key(cell: Dict[str, Any]) -> Tuple:
    cfg = cell.get("config") or {}
    purify = _as_purify(cfg.get("purify"))
    return (cell.get("image") or cell.get("image_id") or "",
            cfg.get("tau"), purify)


def _purify_label(p: Optional[Tuple[str, float]]) -> str:
    if p is None:
        return "identity"
    kind, strength = p
    return kind if kind == "identity" else f"{kind} {strength:g}"


def _cell_row(cell: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> str:
    cfg = cell.get("config") or {}
    cond = cell.get("condition") or ""
    img = cell.get("image") or cell.get("image_id") or ""
    tau, seed = cfg.get("tau"), cfg.get("seed")
    status = cell.get("status", "")

    flags = []
    if cell.get("modified_from_paper"):
        # 報表把改寫過的 baseline 讀成原論文設定，比不標註更糟
        # （`BaselineSpec.__post_init__` 對此有建構期檢查）。
        flags.append('<span class="flag" title="非原論文設定">改寫</span>')
    if cell.get("retention_usable") is False:
        flags.append('<span class="flag" title="effect(identity) 低於 3 倍量測'
                     '標準差，該列的 retention 不進統計">不可用</span>')
    head = (f'<th scope="row">{escape(cond)}<br><small>seed {seed}'
            f'</small>{"".join(flags)}</th>')

    if status != "done":
        reason = cell.get("skipped_reason") or cell.get("error") or status
        # 不適用與失敗都不留白：`DESIGN` §1.1 要求每一格都要能看到東西，
        # 而「這格為什麼沒有圖」本身就是要看的資訊。
        return (f'<tr class="{escape(status)}">{head}'
                f'<td colspan="{len(COLUMNS)}" class="note">'
                f'{escape(status)}：{escape(str(reason).splitlines()[0])}</td>'
                f'<td class="note">—</td></tr>')

    imgs = _row_images(cell, by_id, cond, img, tau, seed)
    tds = "".join(_img_cell(imgs[key], f"{cond} {img} {label}")
                  for label, key in COLUMNS)
    nums = " · ".join(f"{label} {_num(cell.get(field))}"
                      for field, label in METRICS)
    return f'<tr>{head}{tds}<td class="num">{nums}</td></tr>'


def build_compare_html(cells: Sequence[Dict[str, Any]], batch: str = "",
                       env: Optional[Dict[str, Any]] = None) -> str:
    """由 `_cells/*.json` 的紀錄產生比對頁。**不碰 GPU、不讀影像內容。**

    只寫 `<img src>`，路徑相對於批次目錄，故頁面與產物一起搬移仍可看。
    """
    by_id = {c["id"]: c for c in cells}
    evals = [c for c in cells if c.get("stage") == "eval"]

    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for c in evals:
        groups.setdefault(_group_key(c), []).append(c)

    def sort_key(k):
        img, tau, purify = k
        return (img, tau if tau is not None else -1.0,
                _purify_label(purify))

    body, index = [], []
    for key in sorted(groups, key=sort_key):
        img, tau, purify = key
        anchor = "g-" + slug(img, "-", tau, "-", _purify_label(purify))
        title = (f"{img} · τ={_num(tau, 3)} · 淨化 {_purify_label(purify)}")
        index.append(f'<a href="#{anchor}">{escape(title)}</a>')

        rows = sorted(
            groups[key],
            key=lambda c: (str(c.get("condition")),
                           (c.get("config") or {}).get("seed") or 0),
        )
        head_seed = [c for c in rows if (c.get("config") or {}).get("seed") == 0]
        rest = [c for c in rows if (c.get("config") or {}).get("seed") != 0]

        cols = "".join(f"<th>{escape(label)}</th>" for label, _ in COLUMNS)
        table = (f'<table><tr><th>條件</th>{cols}<th>指標</th></tr>'
                 + "".join(_cell_row(c, by_id) for c in head_seed)
                 + "</table>")
        more = ""
        if rest:
            more = (f'<details><summary>其餘 '
                    f'{len({(c.get("config") or {}).get("seed") for c in rest})}'
                    f' 個種子（{len(rest)} 格）</summary>'
                    f'<table><tr><th>條件</th>{cols}<th>指標</th></tr>'
                    + "".join(_cell_row(c, by_id) for c in rest)
                    + "</table></details>")
        body.append(f'<section id="{anchor}"><h2>{escape(title)}</h2>'
                    f'{table}{more}</section>')

    e = env or {}
    meta = (f"gpu={escape(str(e.get('gpu', '?')))} · "
            f"precision={escape(str(e.get('precision', '?')))} · "
            f"commit={escape(str(e.get('commit', '?')))}")
    return _PAGE.format(
        batch=escape(batch), meta=meta, n_groups=len(groups),
        n_cells=len(evals), index=" · ".join(index), body="".join(body),
    )


_PAGE = """<title>compare {batch}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:1.5rem;max-width:100rem}}
h1{{margin:0 0 .2rem}}
table{{border-collapse:collapse;margin:.4rem 0 1rem;width:100%}}
th,td{{border-bottom:1px solid #8884;padding:.25rem;vertical-align:top}}
th[scope=row]{{text-align:left;white-space:nowrap;font-weight:600}}
img{{width:100%;max-width:13rem;height:auto;display:block;background:#8881}}
td.miss{{color:#c33;text-align:center;font-weight:600}}
td.note{{color:#888}}
td.num{{white-space:nowrap;font-variant-numeric:tabular-nums;color:#666}}
tr.skipped{{opacity:.6}}
tr.failed th[scope=row]{{color:#c33}}
.flag{{display:inline-block;margin-left:.3rem;padding:0 .3rem;border-radius:.2rem;
  background:#c33;color:#fff;font-size:.72em;font-weight:600}}
.index{{font-size:.85em;line-height:1.9;color:#666;margin-bottom:1.5rem}}
.lead{{background:#8881;padding:.6rem .8rem;border-radius:.3rem}}
section{{margin-bottom:2rem}}
@media(prefers-color-scheme:dark){{body{{background:#111;color:#ddd}}
  td.num,.index{{color:#999}}}}
</style>
<h1>人眼比對 · batch {batch}</h1>
<p>{meta}</p>
<p class="lead"><b>這一頁是主判準。</b>數值指標為輔助；兩者矛盾時以人眼為準，
並把該矛盾記進報告。每一列的六張圖依因果鏈排列：原圖 → 防禦圖 → 殘差 →
淨化後 → 防禦側編輯 ‖ 對照側編輯。最後兩張並排是重點——同一個淨化、
同一個種子、同一個 prompt，唯一的差別是防禦有沒有開。</p>
<p>{n_groups} 組 ×（影像, τ, 淨化），共 {n_cells} 格。</p>
<div class="index">{index}</div>
{body}
"""
