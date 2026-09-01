"""防禦圖與淨化後影像的人眼比對頁。**不跑 GPU、不重算任何數字。**

為什麼要它
────────────────────────────────────────────────────────────────────
抗淨化那一段至今只有數字：`phase_retention.py` 把中間影像用完即棄。而本專案
的判準是「人眼為主、數值指標為輔」，指標矛盾時以人眼裁定
（`docs/GOAL.md`）——沒有圖就無法裁定。

本檔把同一條路徑上的每一階存下來並排版成可原地切換的頁面：

    原圖 → 防禦圖 → 淨化(防禦圖) 各算子
    原圖 → 未防禦編輯
    防禦圖 → 防禦後編輯

**淨化後的編輯需要 GPU**，不在本檔範圍；缺的格子明寫為「待補」而不是留白。

版面規則（沿用既有的比對頁作法）
────────────────────────────────────────────────────────────────────
- **原地切換**，不並排。並排讓注意力花在移動視線上，對低對比差異的敏感度
  差很多。
- 附 **4× 放大裁切**，位置由**原圖的梯度能量**自動選（差異在高頻處最明顯），
  同一張圖的各版本取**同一座標**，放大用最近鄰。
- 影像的環境底色在明暗主題下都固定為同一階中性灰。環境亮度會改變感知對比，
  隨主題變動會使判斷不可比。
- **指標預設收起**，先看圖再展開數字，否則就不是在測眼睛。
- 全部影像以 PNG 內嵌。**不可改用 JPEG**：本頁要看的正是高頻的擾動，
  而 JPEG 量化會把它抹掉，等於用受測對象本身當壓縮器。

用法：
    python scripts/defense_purify_gallery.py --src <擺著 PNG 的目錄> \
        --out runs/defense_purify_gallery/index.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.purify import ops as purify_ops  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512
CROP = 128              # 裁切邊長；顯示時 4× 放大到 512
VIEW = 288              # 全圖檢視的邊長（降取樣仍是無損 PNG）
# 全圖只提供整體印象，診斷靠的是 4× 裁切；裁切保持原生解析度且無損。
# 全圖不改用 JPEG——本頁要看的正是高頻擾動，用 JPEG 等於拿受測對象當壓縮器。


def gradient_peak(x01: torch.Tensor, crop: int = CROP) -> Tuple[int, int]:
    """梯度能量最高的 `crop x crop` 視窗左上角座標。

    取原圖而非防禦圖：座標必須與比較的是哪一版無關，否則各版本裁到不同位置，
    看到的差異裡混著位置的差異。
    """
    lum = (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]).unsqueeze(1)
    gx = lum[..., :, 1:] - lum[..., :, :-1]
    gy = lum[..., 1:, :] - lum[..., :-1, :]
    energy = F.pad(gx.abs(), (0, 1)) + F.pad(gy.abs(), (0, 0, 0, 1))
    pooled = F.avg_pool2d(energy, kernel_size=crop, stride=crop // 4)
    idx = int(pooled.flatten().argmax())
    w = pooled.shape[-1]
    y = (idx // w) * (crop // 4)
    x = (idx % w) * (crop // 4)
    limit = x01.shape[-1] - crop
    return min(max(y, 0), limit), min(max(x, 0), limit)


def png_data_uri(t: torch.Tensor) -> str:
    """(1,3,H,W) in [0,1] → PNG 的 data URI。無損，理由見模組 docstring。"""
    from PIL import Image
    arr = (t.clamp(0, 1)[0].permute(1, 2, 0) * 255.0).round().to(torch.uint8).numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def views(t: torch.Tensor, box: Tuple[int, int]) -> Dict[str, str]:
    """一張圖的兩個檢視：降取樣的全圖，以及**原尺寸**的裁切。

    裁切不在這裡放大——CSS 已經設了 `image-rendering: pixelated`，由瀏覽器
    做最近鄰放大到 4×。存 512² 的放大點陣等於把同一份資訊多存 16 倍，整頁
    會從 11 MB 漲到 23 MB 而畫面逐點相同。
    """
    y, x = box
    full = F.interpolate(t, size=(VIEW, VIEW), mode="area")
    patch = t[..., y:y + CROP, x:x + CROP]
    return {"full": png_data_uri(full), "zoom": png_data_uri(patch)}


def discover(src: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """把 `<image>__<cond>__<kind>.png` 依 影像 → 條件 → 種類 歸類。

    `<image>__orig.png` 沒有條件段，歸到 `__orig__`。條件名本身含底線
    （`phase_gain`），故由**右側**切三段而不是直接 split。
    """
    out: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for path in sorted(src.rglob("*.png")):
        stem = path.stem
        if stem.endswith("__orig"):
            image, cond, kind = stem[:-len("__orig")], "__orig__", "orig"
        else:
            m = re.match(r"^(.*)__([^_].*?)__(def|edit_def|edit_orig)$", stem)
            if not m:
                continue
            image, cond, kind = m.group(1), m.group(2), m.group(3)
        # 同名條件可能來自多個批次目錄，用父目錄名區分
        tag = cond if cond == "__orig__" else f"{path.parent.name}"
        out.setdefault(image, {}).setdefault(tag, {})[kind] = path
    return out


# 淨化算子。取 `phase_retention.py` 用過、且不需要擴散權重或 GPU 的四個，
# 其中 `jpeg_then_resize`（C&R 串接）與 `crop_resize` 是本方法最弱的一格。
PURIFIERS = [
    ("blur1", lambda: purify_ops.Purifier("blur", 1.0)),
    ("jpeg75", lambda: purify_ops.Purifier("jpeg", 75)),
    ("crop_resize0.1",
     lambda: purify_ops.Purifier("crop_resize", purify_ops.CROP_FRACTION_DIA)),
    ("jpeg_then_resize",
     lambda: purify_ops.Purifier("jpeg_then_resize", purify_ops.CR_JPEG_QUALITY)),
]


def purifier_set(scope: str):
    """`scope` 選到的淨化算子清單。

    `none` 給的是**條件對條件**的比較——多個方法的防禦圖與防禦後編輯並列，
    淨化階既不是要看的東西，又讓頁面體積漲三倍。名字打錯要拋錯而不是回退到
    `standard`：靜默回退會產出一份看起來正常、卻不是被要求的那一份頁面。
    """
    if scope == "none":
        return []
    if scope == "standard":
        return list(PURIFIERS)
    raise ValueError(
        f"未知的 purifiers 範圍：{scope!r}，可用的是 none／standard")


def build(src: Path, images: Optional[List[str]], conditions: Optional[List[str]],
          scope: str = "standard") -> Tuple[List[dict], List[str]]:
    found = discover(src)
    if images:
        found = {k: v for k, v in found.items() if k in images}
    if not found:
        raise SystemExit(f"{src} 底下找不到可用的 PNG")

    purifiers = []
    for name, make in purifier_set(scope):
        p = make()
        if not p.available:
            raise SystemExit(f"淨化算子 {name} 的相依不齊備，拒絕以缺格產出頁面")
        purifiers.append((name, p))

    cards: List[dict] = []
    notes: List[str] = []
    for image, by_cond in sorted(found.items()):
        orig_path = None
        for cond, kinds in by_cond.items():
            if "orig" in kinds:
                orig_path = kinds["orig"]
                break
        if orig_path is None:
            notes.append(f"{image}：找不到原圖，整張略過")
            continue
        x_orig = load_image_tensor(orig_path, torch.device("cpu"), size=RESOLUTION)
        box = gradient_peak(x_orig)

        entry = {"image": image, "box": list(box),
                 "stages": {"原圖": views(x_orig, box)}, "conditions": {}}
        for cond, kinds in sorted(by_cond.items()):
            if cond == "__orig__":
                if "edit_orig" in kinds:
                    x = load_image_tensor(kinds["edit_orig"], torch.device("cpu"),
                                          size=RESOLUTION)
                    entry["stages"]["未防禦編輯"] = views(x, box)
                continue
            if conditions and cond not in conditions:
                continue
            if "def" not in kinds:
                notes.append(f"{image}/{cond}：沒有防禦圖，該條件略過")
                continue
            if "edit_orig" in kinds and "未防禦編輯" not in entry["stages"]:
                x = load_image_tensor(kinds["edit_orig"], torch.device("cpu"),
                                      size=RESOLUTION)
                entry["stages"]["未防禦編輯"] = views(x, box)

            x_def = load_image_tensor(kinds["def"], torch.device("cpu"),
                                      size=RESOLUTION)
            stages = {"防禦圖": views(x_def, box)}
            for name, p in purifiers:
                stages[f"淨化：{name}"] = views(p.evaluate(x_def), box)
            if "edit_def" in kinds:
                x = load_image_tensor(kinds["edit_def"], torch.device("cpu"),
                                      size=RESOLUTION)
                stages["防禦後編輯"] = views(x, box)
            else:
                notes.append(f"{image}/{cond}：缺防禦後編輯")
            entry["conditions"][cond] = stages
            print(f"  {image} / {cond}：{len(stages)} 階", flush=True)
        cards.append(entry)
    return cards, notes


HTML_HEAD = """<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+TC:wght@400;500;600&display=swap">
<style>
/* 色票的中性色帶一點青藍偏色，對應頻譜量測儀器的取色；強調色是量測儀的
   青色，語意色（琥珀）只用於「缺格／警告」，兩者不混用。 */
:root{
  --bg:#eef1f3; --panel:#ffffff; --fg:#12181c; --muted:#5d6b73;
  --line:#ccd6dc; --line-soft:#e2e9ed;
  --accent:#0e6f80; --accent-fg:#ffffff;
  --warn:#8a5300; --warn-bg:rgba(180,120,20,.10);
  /* 影像的環境底色**固定**，明暗主題皆同一階中性灰。環境亮度會改變感知
     對比，隨主題變動會使不同時間看到的判斷不可比。 */
  --stage:#7f7f7f; --stage-fg:#f2f2f2;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0c1114; --panel:#131b1f; --fg:#dfe7ea; --muted:#8fa2ab;
    --line:#26343a; --line-soft:#1b262b;
    --accent:#4fc3d6; --accent-fg:#06181d;
    --warn:#e0a03c; --warn-bg:rgba(224,160,60,.10);
  }
}
:root[data-theme="dark"]{
  --bg:#0c1114; --panel:#131b1f; --fg:#dfe7ea; --muted:#8fa2ab;
  --line:#26343a; --line-soft:#1b262b;
  --accent:#4fc3d6; --accent-fg:#06181d;
  --warn:#e0a03c; --warn-bg:rgba(224,160,60,.10);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:"IBM Plex Sans TC",ui-sans-serif,system-ui,"Noto Sans TC",sans-serif;
  font-size:15px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1200px;margin:0 auto;padding:34px 22px 90px}
header{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:26px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.72rem;
  letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}
h1{font-size:1.72rem;line-height:1.25;margin:0 0 10px;font-weight:600;
  text-wrap:balance;letter-spacing:-.01em}
p.lede{color:var(--muted);margin:0;max-width:64ch}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:20px 22px;margin-bottom:24px}
.card h2{font-size:.98rem;margin:0;font-weight:600;
  font-family:"IBM Plex Mono",ui-monospace,monospace;letter-spacing:-.01em}
.meta{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;
  color:var(--muted);margin:4px 0 0;font-variant-numeric:tabular-nums}
.grouplabel{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.68rem;
  letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
  align-self:center;padding-right:2px}
.bar{display:flex;gap:7px;flex-wrap:wrap;align-items:center}
.bar+.bar{margin-top:8px}
.bars{display:flex;flex-direction:column;gap:8px;margin:16px 0 18px;
  padding:14px 16px;border:1px solid var(--line-soft);border-radius:8px}
.sep{width:1px;align-self:stretch;background:var(--line-soft);margin:0 4px}
button{font:inherit;font-size:.87rem;padding:5px 11px;border-radius:6px;
  cursor:pointer;border:1px solid var(--line);background:transparent;
  color:var(--fg);transition:background .12s,border-color .12s}
button:hover{border-color:var(--accent)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button[aria-pressed="true"]{background:var(--accent);color:var(--accent-fg);
  border-color:var(--accent);font-weight:500}
.viewer{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
.pane{background:var(--stage);border-radius:8px;padding:14px;
  display:flex;flex-direction:column;gap:9px;align-items:center}
.pane img{display:block;image-rendering:pixelated;max-width:100%;height:auto;
  border-radius:3px}
/* 裁切以 CSS 放大到 4×（128 -> 512）。放大在瀏覽器端做，頁面只存原尺寸。 */
.pane img[data-zoom]{width:512px}
@media (max-width:1120px){.pane img[data-zoom]{width:100%}}
.cap{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.74rem;
  color:var(--stage-fg);letter-spacing:.02em}
.hint{color:var(--muted);font-size:.85rem;margin:14px 0 0}
kbd{font-family:"IBM Plex Mono",ui-monospace,monospace;border:1px solid var(--line);
  border-bottom-width:2px;border-radius:4px;padding:1px 5px;font-size:.78rem}
details{margin-top:26px;border:1px solid var(--line);border-radius:10px;
  padding:14px 18px;background:var(--panel)}
summary{cursor:pointer;color:var(--muted);font-size:.9rem;font-weight:500}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;margin-top:14px;font-size:.85rem;width:100%;
  font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line-soft);padding:7px 11px;text-align:right}
th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.7rem;
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
  font-weight:500;border-bottom-color:var(--line)}
th:first-child,td:first-child{text-align:left;
  font-family:"IBM Plex Mono",ui-monospace,monospace}
.warn{border-left:3px solid var(--warn);padding:10px 14px;margin:16px 0;
  color:var(--muted);font-size:.87rem;background:var(--warn-bg);
  border-radius:0 6px 6px 0}
.warn strong{color:var(--fg);font-weight:600}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
"""


# 版面的文案隨 `--purifiers` 的範圍換。錯的說明比沒有說明更糟：讀的人會
# 照著它去解讀畫面，而兩種頁面要回答的根本不是同一個問題。
_SHARED_LEDE = (
    '切換是<strong>原地替換</strong>而不是並排——並排讓注意力花在移動視線上，'
    '對低對比差異的敏感度差很多。右側 4× 裁切的位置由'
    '<strong>原圖的梯度能量</strong>自動選，同一張圖的各版本取同一座標，'
    '最近鄰放大。影像的環境底色固定為中性灰，不隨主題變動。')

PAGE_COPY = {
    "standard": {
        "title": "擾動存活檢視台",
        "eyebrow": "InstructPix2Pix · 主線批次",
        "lede": "同一張防禦圖走過四個淨化算子之後還剩下什麼。" + _SHARED_LEDE,
        "warn": ("<strong>本頁不能回答的事</strong><br>"
                 "淨化後的<em>編輯</em>需要 GPU，不在本頁範圍。"
                 "本頁回答「淨化把擾動抹掉了多少」，"
                 "不回答「淨化之後防禦還有沒有效」——後者要看淨增益，"
                 "而那必須扣掉空白地板。"),
    },
    "none": {
        "title": "防禦條件對照台",
        "eyebrow": "InstructPix2Pix · 人眼確認服從的影像",
        "lede": "同一張原圖在各個防禦條件下，防禦圖付了多少可見失真、"
                "以及攻擊方拿到的編輯輸出還能不能用。" + _SHARED_LEDE,
        "warn": ("<strong>本頁不能回答的事</strong><br>"
                 "各條件的強度<em>沒有</em>對齊到同一個失真——那要看數字，"
                 "展開下方的指標表。本頁回答的是「同樣一張圖，"
                 "各個方法把它變成什麼樣子」。"),
    },
}


def render(cards: List[dict], notes: List[str], metrics: List[dict],
           scope: str = "standard", title: Optional[str] = None) -> str:
    if scope not in PAGE_COPY:
        raise ValueError(
            f"未知的 purifiers 範圍：{scope!r}，可用的是 {sorted(PAGE_COPY)}")
    copy = PAGE_COPY[scope]
    # 一份批次拆成多頁時各頁要有自己的名字，否則在分頁列與清單裡分不出來。
    # 換的只有名字，抬頭與說明仍由 scope 決定——那兩者講的是這一頁在回答
    # 什麼，不隨拆頁改變。
    name = title or copy["title"]
    # 用 replace 不用 format：`HTML_HEAD` 裡整段 CSS 都是大括號。
    parts = [HTML_HEAD.replace("{title}", name),
             '<div class="wrap">', "<header>"]
    parts.append(f'<p class="eyebrow">{copy["eyebrow"]}</p>')
    parts.append(f'<h1>{name}</h1>')
    parts.append(f'<p class="lede">{copy["lede"]}</p>')
    parts.append("</header>")

    parts.append(f'<div class="warn">{copy["warn"]}</div>')
    if notes:
        parts.append('<div class="warn"><strong>缺的格子</strong><br>'
                     + "<br>".join(notes) + "</div>")

    for i, card in enumerate(cards):
        parts.append('<div class="card">')
        parts.append(f'<h2>{card["image"]}</h2>')
        y, x = card["box"]
        parts.append(f'<p class="meta">裁切 (y, x) = ({y}, {x}) · 邊長 {CROP} px '
                     f'· 最近鄰 4× · 全圖降取樣至 {VIEW} px</p>')
        parts.append(f'<div class="bars" data-bars="{i}"></div>')
        parts.append(
            f'<div class="viewer">'
            f'<div class="pane"><img data-full="{i}" alt="全圖檢視">'
            f'<span class="cap" data-capfull="{i}"></span></div>'
            f'<div class="pane"><img data-zoom="{i}" alt="4 倍放大裁切">'
            f'<span class="cap" data-capzoom="{i}"></span></div>'
            f'</div>')
        parts.append('<p class="hint">按 <kbd>←</kbd> <kbd>→</kbd> 依序切換，'
                     '<kbd>1</kbd>–<kbd>9</kbd> 直接跳到該階段。</p>')
        parts.append("</div>")

    if metrics:
        parts.append("<details><summary>數字（預設收起：先看圖，否則就不是在測眼睛）"
                     "</summary>"
                     '<div class="tablewrap"><table><thead><tr>'
                     "<th>條件</th><th>LPIPS</th><th>DISTS</th><th>L/D</th>"
                     "<th>位移</th><th>位移÷LPIPS</th><th>步數</th><th>n</th>"
                     "</tr></thead><tbody>")
        for m in metrics:
            parts.append(
                "<tr><td>{cond}</td><td>{lpips:.4f}</td><td>{dists:.4f}</td>"
                "<td>{ld:.2f}</td><td>{effect:.4f}</td><td>{eff:.3f}</td>"
                "<td>{steps}</td><td>{n}</td></tr>".format(**m))
        parts.append("</tbody></table></div></details>")

    parts.append("</div>")
    parts.append("<script>const DATA = ")
    parts.append(json.dumps(cards, ensure_ascii=False))
    parts.append(""";
// 階段依「來源 / 淨化後 / 編輯結果」分群。分群encode的是這條路徑的實際結構，
// 不是排版裝飾：淨化只作用在防禦圖上，編輯是路徑的終點。
function groupOf(name){
  if (name.startsWith("淨化：")) return "淨化後";
  if (name.indexOf("編輯") >= 0) return "編輯結果";
  return "來源";
}
const ORDER = ["來源", "淨化後", "編輯結果"];

DATA.forEach((card, i) => {
  const conds = Object.keys(card.conditions);
  let cond = conds[0] || null, stage = null;

  const bars = document.querySelector(`[data-bars="${i}"]`);
  const imgFull = document.querySelector(`[data-full="${i}"]`);
  const imgZoom = document.querySelector(`[data-zoom="${i}"]`);
  const capFull = document.querySelector(`[data-capfull="${i}"]`);
  const capZoom = document.querySelector(`[data-capzoom="${i}"]`);

  function stagesFor(){
    const out = Object.assign({}, card.stages);
    if (cond && card.conditions[cond]) Object.assign(out, card.conditions[cond]);
    return out;
  }
  function ordered(){
    const names = Object.keys(stagesFor());
    return ORDER.flatMap(g => names.filter(n => groupOf(n) === g));
  }
  function paint(){
    const all = stagesFor(), names = ordered();
    if (!names.includes(stage)) stage = names[0];
    const v = all[stage];
    imgFull.src = v.full; imgZoom.src = v.zoom;
    capFull.textContent = stage + "  ·  全圖";
    capZoom.textContent = stage + "  ·  4×";
    bars.querySelectorAll("button[data-stage]").forEach(b =>
      b.setAttribute("aria-pressed", String(b.dataset.stage === stage)));
    bars.querySelectorAll("button[data-cond]").forEach(b =>
      b.setAttribute("aria-pressed", String(b.dataset.cond === cond)));
  }
  function build(){
    bars.innerHTML = "";
    const condRow = document.createElement("div");
    condRow.className = "bar";
    const cl = document.createElement("span");
    cl.className = "grouplabel"; cl.textContent = "條件";
    condRow.appendChild(cl);
    conds.forEach(c => {
      const b = document.createElement("button");
      b.textContent = c; b.dataset.cond = c;
      b.onclick = () => { cond = c; build(); paint(); };
      condRow.appendChild(b);
    });
    bars.appendChild(condRow);

    const names = ordered();
    const stageRow = document.createElement("div");
    stageRow.className = "bar";
    let last = null;
    names.forEach((name, k) => {
      const g = groupOf(name);
      if (g !== last){
        if (last !== null){
          const sp = document.createElement("span");
          sp.className = "sep"; stageRow.appendChild(sp);
        }
        const gl = document.createElement("span");
        gl.className = "grouplabel"; gl.textContent = g;
        stageRow.appendChild(gl);
        last = g;
      }
      const b = document.createElement("button");
      b.textContent = (k < 9 ? (k + 1) + " " : "") + name.replace("淨化：", "");
      b.dataset.stage = name;
      b.onclick = () => { stage = name; paint(); };
      stageRow.appendChild(b);
    });
    bars.appendChild(stageRow);
  }
  build(); paint();

  document.addEventListener("keydown", ev => {
    if (ev.target.tagName === "BUTTON" && ev.key !== "ArrowLeft"
        && ev.key !== "ArrowRight") return;
    const names = ordered();
    let k = names.indexOf(stage);
    if (ev.key === "ArrowRight") k = (k + 1) % names.length;
    else if (ev.key === "ArrowLeft") k = (k - 1 + names.length) % names.length;
    else if (/^[1-9]$/.test(ev.key)) k = Math.min(+ev.key - 1, names.length - 1);
    else return;
    ev.preventDefault(); stage = names[k]; paint();
  });
});
</script>""")
    return "\n".join(parts)


def collect_metrics(run_roots: List[Path], conditions: List[str]) -> List[dict]:
    """把頁面上出現的條件對應的數字撈出來。找不到就略過該列，不編造。"""
    import csv
    import statistics as st
    out = []
    for root in run_roots:
        for tag in conditions:
            path = root / tag / "results.csv"
            if not path.exists():
                continue
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            if not rows:
                continue
            g = lambda k: st.fmean(float(r[k]) for r in rows)
            L, D, E = g("fid_lpips"), g("fid_dists"), g("edit_lpips")
            out.append({"cond": tag, "lpips": L, "dists": D, "ld": L / D,
                        "effect": E, "eff": E / L,
                        "steps": rows[0].get("defense_steps", "?"),
                        "n": len({r["image"] for r in rows})})
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="擺著 <image>__<cond>__<kind>.png 的目錄，會遞迴尋找")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="條件標籤即批次子目錄名，如 phase_s0100")
    ap.add_argument("--metrics-from", type=Path, nargs="*", default=[],
                    help="要撈數字的批次根目錄")
    ap.add_argument("--title", default=None,
                    help="頁面名稱。不給則由 --purifiers 的範圍決定。"
                         "一份批次拆成多頁時各頁要有自己的名字")
    ap.add_argument("--purifiers", choices=("standard", "none"),
                    default="standard",
                    help="standard = 四個淨化算子（預設，逐位元等於加這個旗標"
                         "之前）；none = 只留防禦圖與防禦後編輯，用於條件對"
                         "條件的比較")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    cards, notes = build(args.src, args.images, args.conditions, args.purifiers)
    conds = sorted({c for card in cards for c in card["conditions"]})
    metrics = collect_metrics(args.metrics_from, conds) if args.metrics_from else []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render(cards, notes, metrics, args.purifiers, args.title),
        encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    print(f"{args.out}  {mb:.1f} MB  影像 {len(cards)} 張 × 條件 {len(conds)} 個")
    if mb > 16:
        print("**超過 16 MB**，Artifact 會拒收；請減少 --images 或 --conditions")


if __name__ == "__main__":
    main()
