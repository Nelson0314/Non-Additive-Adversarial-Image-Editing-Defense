"""把一批 `carrier_search` 的輸出排成可瀏覽的 HTML。

每格一節：未防禦的編輯、`start`（現行全域仿射）的防禦圖與編輯、`searched` 的
防禦圖與編輯，圖下標該列的四項讀數。判準不由這支腳本下，圖與數字擺在一起讓人
自己看。
"""
import argparse
import csv
import html
from pathlib import Path

ARM_ORDER = ('undefended', 'start', 'searched')


def num(value, nd=3):
    if value in (None, '', 'nan'):
        return '—'
    try:
        return f'{float(value):.{nd}f}'
    except (TypeError, ValueError):
        return html.escape(str(value))


def build(root: Path) -> Path:
    rows = []
    for f in sorted(root.glob('**/carrier_search*.csv')):
        rows += list(csv.DictReader(open(f, encoding='utf-8')))
    png = {}
    for p in root.glob('**/*.png'):
        png[p.name] = p.relative_to(root).as_posix()

    cells = {}
    for r in rows:
        cells.setdefault((r['image'], r['class']), []).append(r)

    out = ['<title>carrier_search</title>', '''<style>
:root{--bg:#fbfbf9;--fg:#1a1918;--sub:#5e5d59;--line:rgba(0,0,0,.12)}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:30px 0 2px;border-top:1px solid var(--line);padding-top:14px}
.meta{color:var(--sub);font-size:12px;margin-bottom:10px}
.grid{display:flex;flex-wrap:wrap;gap:16px}
.card{width:224px}
.card img{width:224px;height:224px;object-fit:cover;border:1px solid var(--line);border-radius:4px;display:block;background:#eee}
.cap{font-size:11px;color:var(--sub);margin-top:3px}
.tag{font-weight:600;color:var(--fg)}
</style>''', '<h1>carrier_search</h1>']
    out.append(f'<div class="meta">{len(rows)} 列，{len(cells)} 格。'
               f'每格：未防禦 → start（全域仿射）→ searched（搜到的載體）。'
               f'數字是 criterion / id_norm / use_norm / dir_norm。</div>')

    for (image, cls), rs in sorted(cells.items()):
        r0 = rs[0]
        out.append(f'<h2>{html.escape(image)} · {html.escape(cls)}</h2>')
        out.append(f'<div class="meta">「{html.escape(r0["instruction"])}」 · '
                   f'搜尋種子 {r0["search_seeds"]} · 回報種子 {r0["report_seeds"]} · '
                   f'預算 {r0["search_budget"]} · 搜尋分數 {num(r0["search_score"], 4)}</div>')
        by_arm = {}
        for r in rs:
            by_arm.setdefault(r['arm'], []).append(r)
        out.append('<div class="grid">')
        for arm in ARM_ORDER:
            for r in sorted(by_arm.get(arm, []), key=lambda q: q['eval_seed']):
                s = r['eval_seed']
                if arm == 'undefended':
                    name = f'{image}__{cls}__s{s}__clean_edit.png'
                else:
                    name = f'{image}__{cls}__{arm}__s{s}__def_edit.png'
                cap = (f'<span class="tag">{html.escape(arm)} · s{s}</span><br>'
                       f'crit {num(r["criterion"])} · id {num(r["id_norm"])}<br>'
                       f'use {num(r["use_norm"])} · dir {num(r["dir_norm"])}<br>'
                       f'id_def {num(r["subject_id_edit_def"])} · '
                       f'hf {num(r["hf_rgb_total"], 3)}')
                src = png.get(name)
                img = (f'<a href="{src}"><img loading="lazy" src="{src}"></a>'
                       if src else '<div class="cap">缺圖</div>')
                out.append(f'<div class="card">{img}<div class="cap">{cap}</div></div>')
            if arm != 'undefended':
                d = png.get(f'{image}__{cls}__{arm}__def.png')
                if d:
                    r = by_arm.get(arm, [{}])[0]
                    out.append(
                        f'<div class="card"><a href="{d}">'
                        f'<img loading="lazy" src="{d}"></a><div class="cap">'
                        f'<span class="tag">{html.escape(arm)} 防禦圖</span><br>'
                        f'niqe {num(r.get("niqe_input"))} / {num(r.get("niqe_original"))}<br>'
                        f'ΔE00 {num(r.get("support_deltaE00"), 2)} · '
                        f'hf {num(r.get("hf_rgb_total"), 3)}</div></div>')
        out.append('</div>')

    target = root / 'index.html'
    target.write_text('\n'.join(out), encoding='utf-8')
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    print(build(args.root))


if __name__ == '__main__':
    main()
