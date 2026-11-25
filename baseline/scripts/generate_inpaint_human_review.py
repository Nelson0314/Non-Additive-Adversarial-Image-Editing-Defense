"""SD inpainting ISR 人工複核的盲看頁：從 Sonnet 判定（docs/isr_pilot/inpaint/judgments.csv）分層抽 60 格，不顯示 Sonnet 判定與條件名。

抽樣：只取同一影像、指令、淨化下 Undefended 判為成功（content=1、mismatch=0、degraded=0）的防禦條件格（條件式 ISR 的分母）；
6 個防禦條件（PhotoGuard、DIA-PT、DCT-Shield、SIFM、BlurGuard、ImmuLUT）各 10 格，其中 Sonnet 判為編輯成功與失敗各 5 格，
5 格依序輪替 4 種淨化（none、jpeg50、blur2、vae_trans；起點逐條件依序位移）；同一層內以 numpy.random.default_rng(20260812) 無放回抽樣。
某層格數不足時失敗。60 格的呈現順序再以同一個 rng 打亂，依序編號 01–60。
每格顯示：指令全文、淨化名、Input（該條件送入編輯器的影像）、Undefended 在同一淨化下的編輯、該條件的編輯。
評審規則同 docs/isr_pilot/inpaint/rubric.md（content、mismatch、degraded 三個 0/1 標籤）。
輸出 <--output-dir>（預設 artifacts/pexels_portraits/inpaint_isr_human_review/）：
    review.html    盲看頁（影像以 JPEG base64 內嵌，單檔可離線開啟）
    answers.csv    作答表（item、content、mismatch、degraded、note，標籤欄空白）
    key.csv        解盲表（item、sheet、cell、image、prompt_id、condition、purification、Sonnet 三個標籤、各影像路徑）；不交給評審
用法（遠端 baseline 專案根，CPU）：
    $PY scripts/generate_inpaint_human_review.py --judgments <judgments.csv>
"""
import argparse
import base64
import csv
import html
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 20260812
SHEETS = Path("artifacts/pexels_portraits/inpaint_isr_sheets")
COLS = {"C2": "none", "C3": "jpeg50", "C4": "blur2", "C5": "vae_trans"}
PURS = ("none", "jpeg50", "blur2", "vae_trans")
PUR_LABEL = {"none": "No purification", "jpeg50": "JPEG 50", "blur2": "Gaussian blur σ2", "vae_trans": "VAE-Trans"}
CONDS = ("PhotoGuard", "DIA-PT", "DCT-Shield", "SIFM", "BlurGuard", "ImmuLUT")
PER_OUTCOME = 5
DESCRIPTION = """Each item shows one edit by Stable Diffusion Inpainting, which repaints only the background (everything outside
the person). Left: the image fed to the editor (it may be color-shifted; color shift alone is not a failure). Middle: the edit of
the undefended image under the same purification, for reference. Right: the edit to judge. Give three independent 0/1 labels
for the right image: content (the requested content is clearly present in the repainted background), mismatch (the edit did
something other than the instruction: person replaced or altered, extra people, another animal instead of a dog, unrelated
scene), degraded (severe structural or quality breakdown; global color cast, mild blur or haze is not degraded). The band of
original background color around the person outline is present in all edits and does not count. Full rubric:
docs/isr_pilot/inpaint/rubric.md."""


def b64(path, size=384):
    if not Path(path).is_file():
        raise SystemExit(f"缺 {path}")
    im = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/pexels_portraits/inpaint_isr_human_review"))
    a = ap.parse_args()
    key = {(r["sheet"], r["row"]): r["condition"] for r in csv.DictReader(open(SHEETS / "key.csv", encoding="utf-8"))}
    cells = {(r["sheet"], r["cell"]): r for r in csv.DictReader(open(SHEETS / "cells.csv", encoding="utf-8"))}
    labels = {}
    for r in csv.DictReader(open(a.judgments, encoding="utf-8")):
        col, row = r["cell"].split("-")
        cond = key[(f"{r['sheet']}.jpg", row)]
        ok = r["content"] == "1" and r["mismatch"] == "0" and r["degraded"] == "0"
        labels[(r["sheet"], COLS[col], cond)] = (r, ok)
    strata = defaultdict(list)
    for (sheet, pur, cond), (r, ok) in sorted(labels.items()):
        if cond == "Undefended" or not labels[(sheet, pur, "Undefended")][1]:
            continue
        strata[(cond, pur, ok)].append(r)
    rng = np.random.default_rng(SEED)
    picked = []
    for ci, cond in enumerate(CONDS):
        for ok in (True, False):
            for i in range(PER_OUTCOME):
                pur = PURS[(ci + i) % len(PURS)]
                pool = [r for r in strata[(cond, pur, ok)] if all(r is not p[0] for p in picked)]
                if not pool:
                    raise SystemExit(f"層 {cond} {pur} Sonnet {'成功' if ok else '失敗'} 沒有可抽的格")
                picked.append((pool[int(rng.integers(len(pool)))], cond, pur, ok))
    order = rng.permutation(len(picked))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Inpainting ISR review</title>",
             "<style>body{font-family:sans-serif;max-width:1240px;margin:16px auto;padding:0 16px}"
             ".item{border-top:1px solid #999;padding:12px 0}.row{display:flex;gap:8px}figure{margin:0}"
             "figcaption{font-size:13px;text-align:center}img{width:384px;height:384px}</style></head><body>",
             "<h1>SD Inpainting edits: human review (60 items)</h1>", f"<p>{html.escape(DESCRIPTION)}</p>",
             "<p>Record the labels in answers.csv (one row per item).</p>"]
    keys, answers = [], []
    for n, idx in enumerate(order, start=1):
        r, cond, pur, ok = picked[idx]
        c = cells[(f"{r['sheet']}.jpg", r["cell"])]
        item = f"{n:02d}"
        parts.append(f"<div class='item'><h3>Item {item} — prompt: “{html.escape(c['prompt_text'])}” — {PUR_LABEL[pur]}</h3>"
                     "<div class='row'>"
                     f"<figure><img src='data:image/jpeg;base64,{b64(c['condition_input_png'])}'><figcaption>Input to the editor</figcaption></figure>"
                     f"<figure><img src='data:image/jpeg;base64,{b64(c['undefended_same_purification_png'])}'><figcaption>Undefended edit (reference)</figcaption></figure>"
                     f"<figure><img src='data:image/jpeg;base64,{b64(c['edit_png'])}'><figcaption>Edit to judge</figcaption></figure>"
                     "</div></div>")
        keys.append({"item": item, "sheet": r["sheet"], "cell": r["cell"], "image": c["image"], "prompt_id": c["prompt_id"],
                     "condition": cond, "purification": pur, "sonnet_content": r["content"], "sonnet_mismatch": r["mismatch"],
                     "sonnet_degraded": r["degraded"], "sonnet_success": int(ok), "edit_png": c["edit_png"],
                     "condition_input_png": c["condition_input_png"],
                     "undefended_same_purification_png": c["undefended_same_purification_png"]})
        answers.append({"item": item, "content": "", "mismatch": "", "degraded": "", "note": ""})
    parts.append("</body></html>")
    (a.output_dir / "review.html").write_text("\n".join(parts), encoding="utf-8")
    for name, data in (("key.csv", keys), ("answers.csv", answers)):
        with open(a.output_dir / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    print(f"[REVIEW] {len(keys)} 格 → {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
