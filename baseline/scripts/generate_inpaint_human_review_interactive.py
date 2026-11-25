"""SD inpainting ISR 人工複核的互動作答頁：沿用 scripts/generate_inpaint_human_review.py 已抽出的 60 格（讀其 key.csv，不重新抽樣）。

頁面為單一 HTML 檔，影像以 JPEG base64 內嵌，不含條件名與 Sonnet 判定。每次顯示一格：指令、淨化名、Input、同淨化的 Undefended 編輯、
待判編輯。作答流程：先判 Undefended 編輯是否成功（判準同 docs/isr_pilot/inpaint/rubric.md：指令內容出現、無 mismatch、無 degraded）；
判為不成功時該格不計入 ISR，其餘題目隱藏；判為成功時再對待判編輯給 content、mismatch、degraded 三個 0/1 標籤與選填備註。
頂端顯示進度（已填／60）與規則摘要；作答存於瀏覽器 localStorage；「Export CSV」下載 answers.csv，
欄位 item,undefended_success,content,mismatch,degraded,note（不計入者 content 等留空）。
鍵盤：Y／N 判 Undefended 成功／不成功，C／M／D 切換三個標籤，←／→ 換格。
輸出 <--review-dir>/review_interactive.html。
用法（遠端 baseline 專案根，CPU）：$PY scripts/generate_inpaint_human_review_interactive.py
"""
import argparse
import base64
import csv
import io
import json
from pathlib import Path

from PIL import Image

SHEETS = Path("artifacts/pexels_portraits/inpaint_isr_sheets")
PUR_LABEL = {"none": "No purification", "jpeg50": "JPEG 50", "blur2": "Gaussian blur σ2", "vae_trans": "VAE-Trans"}
RULES = ("Stable Diffusion Inpainting repaints only the background (everything outside the person). "
         "Inputs may be color-shifted; color shift alone is not a failure. "
         "An edit succeeds when the requested content is clearly present in the repainted background, with no mismatch and no "
         "degradation. content: hospital = recognizable medical setting or equipment; gym = recognizable gym equipment; "
         "snowy street = recognizable outdoor snow scene (street not required); dog = recognizable dog; faint or half-drawn = 0. "
         "mismatch: person replaced or altered, extra people, another animal or a person instead of a dog, unrelated scene. "
         "degraded: severe structural or quality breakdown; global color cast, mild blur or haze is not degraded. "
         "The band of original background color around the person outline is present in all edits and does not count.")

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inpainting ISR review</title>
<style>
:root{--bg:#fff;--fg:#111;--muted:#666;--line:#ccc;--accent:#1565c0;--done:#2e7d32}
body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;font-size:15px}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;z-index:2}
header .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
details{font-size:13px;color:var(--muted);margin-top:4px}
main{padding:8px 16px;max-width:1400px;margin:0 auto}
.imgs{display:flex;gap:8px;justify-content:center}
figure{margin:0;flex:1 1 0;max-width:420px}
figure img{width:100%;aspect-ratio:1;object-fit:contain;display:block;border:1px solid var(--line)}
figcaption{text-align:center;font-size:13px;color:var(--muted)}
.q{margin:10px 0;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
button{font-size:14px;padding:4px 10px;cursor:pointer}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.grid{display:flex;flex-wrap:wrap;gap:3px;margin-top:6px}
.grid span{width:26px;text-align:center;font-size:12px;border:1px solid var(--line);cursor:pointer}
.grid span.filled{background:var(--done);color:#fff}.grid span.cur{outline:2px solid var(--accent)}
#second.hidden{display:none}
kbd{border:1px solid var(--line);padding:0 4px;border-radius:3px;font-size:12px}
</style></head><body>
<header><div class="bar"><strong>SD Inpainting edits: human review</strong>
<span id="progress"></span><button id="export">Export CSV</button>
<span style="font-size:13px;color:var(--muted)">Keys: <kbd>Y</kbd>/<kbd>N</kbd> undefended edit succeeds, <kbd>C</kbd> <kbd>M</kbd> <kbd>D</kbd> toggle labels, <kbd>←</kbd>/<kbd>→</kbd> previous/next</span></div>
<details><summary>Rules</summary><p>__RULES__</p></details>
<div class="grid" id="grid"></div></header>
<main>
<h3 id="title"></h3>
<div class="imgs">
<figure><img id="i0" alt="input"><figcaption>Input to the editor</figcaption></figure>
<figure><img id="i1" alt="undefended edit"><figcaption>Undefended edit</figcaption></figure>
<figure><img id="i2" alt="edit to judge"><figcaption>Edit to judge</figcaption></figure>
</div>
<div class="q">1. Does the undefended edit (middle) succeed?
<button id="uy">Yes (Y)</button><button id="un">No (N) — not counted</button></div>
<div id="second" class="hidden">
<div class="q">2. Edit to judge (right):
<label><input type="checkbox" id="c"> content present (C)</label>
<label><input type="checkbox" id="m"> mismatch (M)</label>
<label><input type="checkbox" id="d"> degraded (D)</label></div>
<div class="q">Note (optional): <input type="text" id="note" size="50"></div></div>
<div class="q"><button id="prev">← Previous</button><button id="next">Next →</button></div>
</main>
<script>
const ITEMS = __ITEMS__;
const STORE = "inpaint_isr_review_answers";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) { answers = {}; }
let cur = 0;
const $ = id => document.getElementById(id);
function save() { try { localStorage.setItem(STORE, JSON.stringify(answers)); } catch (e) {} }
function get(it) { return answers[it.item] || (answers[it.item] = {undefended_success: "", content: 0, mismatch: 0, degraded: 0, note: ""}); }
function filled(it) { const a = answers[it.item]; return a && a.undefended_success !== ""; }
function render() {
  const it = ITEMS[cur], a = get(it);
  $("title").textContent = `Item ${it.item} — prompt: “${it.prompt}” — ${it.purification}`;
  ["i0", "i1", "i2"].forEach((id, k) => $(id).src = "data:image/jpeg;base64," + it.images[k]);
  $("uy").classList.toggle("on", a.undefended_success === "1");
  $("un").classList.toggle("on", a.undefended_success === "0");
  $("second").classList.toggle("hidden", a.undefended_success !== "1");
  $("c").checked = !!a.content; $("m").checked = !!a.mismatch; $("d").checked = !!a.degraded; $("note").value = a.note || "";
  const n = ITEMS.filter(filled).length;
  $("progress").textContent = `Answered ${n} / ${ITEMS.length}`;
  $("grid").innerHTML = "";
  ITEMS.forEach((x, k) => { const s = document.createElement("span"); s.textContent = x.item;
    if (filled(x)) s.classList.add("filled"); if (k === cur) s.classList.add("cur");
    s.onclick = () => { cur = k; render(); }; $("grid").appendChild(s); });
}
function setU(v) { get(ITEMS[cur]).undefended_success = v; save(); render(); }
function toggle(k) { if (get(ITEMS[cur]).undefended_success !== "1") return; const a = get(ITEMS[cur]); a[k] = a[k] ? 0 : 1; save(); render(); }
$("uy").onclick = () => setU("1"); $("un").onclick = () => setU("0");
["c", "m", "d"].forEach(id => $(id).onchange = e => { get(ITEMS[cur])[{c: "content", m: "mismatch", d: "degraded"}[id]] = e.target.checked ? 1 : 0; save(); render(); });
$("note").oninput = e => { get(ITEMS[cur]).note = e.target.value; save(); };
$("prev").onclick = () => { cur = Math.max(0, cur - 1); render(); };
$("next").onclick = () => { cur = Math.min(ITEMS.length - 1, cur + 1); render(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type === "text") return;
  const k = e.key.toLowerCase();
  if (k === "y") setU("1"); else if (k === "n") setU("0");
  else if (k === "c") toggle("content"); else if (k === "m") toggle("mismatch"); else if (k === "d") toggle("degraded");
  else if (e.key === "ArrowRight") $("next").click(); else if (e.key === "ArrowLeft") $("prev").click();
});
$("export").onclick = () => {
  const q = s => /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  const rows = ["item,undefended_success,content,mismatch,degraded,note"];
  ITEMS.forEach(it => { const a = answers[it.item] || {undefended_success: ""};
    const counted = a.undefended_success === "1";
    rows.push([it.item, a.undefended_success, counted ? a.content : "", counted ? a.mismatch : "", counted ? a.degraded : "",
               q(counted ? (a.note || "") : "")].join(",")); });
  const blob = new Blob([rows.join("\n") + "\n"], {type: "text/csv"});
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "answers.csv"; link.click();
};
render();
</script></body></html>
"""


def b64(path, size=512):
    if not Path(path).is_file():
        raise SystemExit(f"缺 {path}")
    im = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review-dir", type=Path, default=Path("artifacts/pexels_portraits/inpaint_isr_human_review"))
    a = ap.parse_args()
    cells = {(r["sheet"], r["cell"]): r for r in csv.DictReader(open(SHEETS / "cells.csv", encoding="utf-8"))}
    items = []
    for r in csv.DictReader(open(a.review_dir / "key.csv", encoding="utf-8")):
        c = cells[(f"{r['sheet']}.jpg", r["cell"])]
        items.append({"item": r["item"], "prompt": c["prompt_text"], "purification": PUR_LABEL[r["purification"]],
                      "images": [b64(r["condition_input_png"]), b64(r["undefended_same_purification_png"]), b64(r["edit_png"])]})
    if len(items) != 60:
        raise SystemExit(f"key.csv 有 {len(items)} 格，不是 60")
    page = PAGE.replace("__RULES__", RULES).replace("__ITEMS__", json.dumps(items))
    out = a.review_dir / "review_interactive.html"
    out.write_text(page, encoding="utf-8")
    print(f"[INTERACTIVE] {len(items)} 格 → {out}（{out.stat().st_size / 1e6:.1f} MB）", flush=True)


if __name__ == "__main__":
    main()
