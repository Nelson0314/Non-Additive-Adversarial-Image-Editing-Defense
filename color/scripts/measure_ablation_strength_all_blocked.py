"""消融降強度 ISR 中 Full、B2_lpips0p85、B3_lpips0p55 都擋下的格子：美學讀數、失敗標籤分布與對照拼圖。

B2、B3 取 ISR 與 Full 相近的設定（docs/isr_pilot/ablation_strength/summary.txt：條件式 ISR 全部合計 Full 17.6％、
B2_lpips0p85 17.9％、B3_lpips0p55 21.4％）。做法同 measure_ablation_all_blocked.py，只換條件、樣本與讀數來源。
格子＝(影像, 指令, 淨化)，30 張 × 4 句 × 淨化 none、jpeg50、blur2、vae_trans。判定同 measure_ablation_strength_isr.py：
Undefended 編輯成功（accessory＝1、mismatch＝0、degraded＝0），且三者的編輯都不成功。
讀數：各條件防禦圖（淨化前）的 LAION、MUSIQ-AVA。Full 取 color/results/ablation_isr_aesthetics/inputs/aesthetic_immulut.csv
（supp 表 4 的逐張讀數）；B2、B3 取 color/results/ablation_strength/per_image.csv 的 laion、musiq
（同一實作：baseline measure_additional_metrics --stage aesthetic）。
    逐格加權：每格計一次；影像層級：至少一格入選的影像各計一次。
    區間：影像為單位的 bootstrap 95% 百分位區間，1000 次，random.Random(20260812)；成對差 Full − B2、Full − B3 用同一組重抽。
失敗標籤：accessory＝0（指令物件未出現）、mismatch＝1、degraded＝1，一格可同時有多個。
拼圖（--cells 與 --sheets-dir 給定時）：每格一列，欄為原圖、Full／B2／B3 防禦圖（下標 LAION／MUSIQ）、同一淨化下的未防禦編輯、
Full／B2／B3 編輯；列首為影像、指令、淨化與三者的失敗標籤；每頁 10 列、tile 256 px，依淨化 → 指令 → 影像排序。
輸出（--output-dir）：cells.csv、summary.csv、summary.txt；拼圖寫入 --sheets-dir。必要輸入缺少即失敗。
用法（遠端，CPU）：python color/scripts/measure_ablation_strength_all_blocked.py --judgments <CSV> --key <CSV>
    --immulut-aesthetic <CSV> --per-image <CSV> --output-dir <目錄> [--cells <拼圖 cells.csv> --sheets-dir <目錄>]
"""
import argparse
import csv
import random
import textwrap
from collections import Counter
from pathlib import Path

PURIFS = ["none", "jpeg50", "blur2", "vae_trans"]
CONDS = ["Full", "B2_lpips0p85", "B3_lpips0p55"]
STRENGTH = {"B2_lpips0p85": "color_baseline_huesat_lpips0p85_pexels", "B3_lpips0p55": "color_baseline_affine_lpips0p55_pexels"}
METRICS = ["aes_laion", "aes_musiq_ava"]
N_IMAGES, N_PROMPTS, N_ROWS = 30, 4, 8
SEED, BOOTS, ROWS_PER_PAGE, TILE = 20260812, 1000, 10, 256


def read(path):
    if not Path(path).is_file():
        raise SystemExit(f"缺輸入：{path}")
    return list(csv.DictReader(open(path, encoding="utf-8")))


def ci(values):
    v = sorted(values)
    return v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]


def labels(r):
    out = []
    if r["accessory"] == "0":
        out.append("no accessory")
    if r["mismatch"] == "1":
        out.append("mismatch")
    if r["degraded"] == "1":
        out.append("degraded")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--immulut-aesthetic", type=Path, required=True)
    ap.add_argument("--per-image", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cells", type=Path, default=None, help="拼圖用：降強度 ISR 拼圖的 cells.csv（逐格影像路徑）")
    ap.add_argument("--sheets-dir", type=Path, default=None)
    a = ap.parse_args()
    key = {(r["sheet"].rsplit(".", 1)[0], r["row"]): r["condition"] for r in read(a.key)}
    judg = {}
    for r in read(a.judgments):
        if r["cell"] != "C1":
            k = (r["sheet"], r["purification"], key[(r["sheet"], r["row"])])
            if k in judg:
                raise SystemExit(f"judgments 重複：{k}")
            judg[k] = r
    sheets = sorted({s for s, _, _ in judg})
    if len(sheets) != N_IMAGES * N_PROMPTS or len(judg) != len(sheets) * len(PURIFS) * N_ROWS:
        raise SystemExit(f"judgments 不完整：{len(sheets)} 張拼圖、{len(judg)} 格")
    ok = lambda r: r["accessory"] == "1" and r["mismatch"] == "0" and r["degraded"] == "0"
    aes = {"Full": {r["image"]: {m: float(r[m]) for m in METRICS} for r in read(a.immulut_aesthetic)}}
    per_image = read(a.per_image)
    for c, cond in STRENGTH.items():
        rows = [r for r in per_image if r["condition"] == cond and r["status"] == "delivered"]
        aes[c] = {r["image"]: {"aes_laion": float(r["laion"]), "aes_musiq_ava": float(r["musiq"])} for r in rows}
        if len(aes[c]) != len(rows):
            raise SystemExit(f"per_image.csv 的 {cond} 有重複影像")
    chosen = []
    for s in sheets:
        img, pk = s.rsplit("__", 1)
        for pu in PURIFS:
            if ok(judg[(s, pu, "Undefended")]) and not any(ok(judg[(s, pu, c)]) for c in CONDS):
                chosen.append({"sheet": s, "image": img, "prompt_id": pk, "purification": pu,
                               **{f"labels_{c}": ";".join(labels(judg[(s, pu, c)])) for c in CONDS}})
    if not chosen:
        raise SystemExit("沒有三者都擋下的格子")
    chosen.sort(key=lambda r: (PURIFS.index(r["purification"]), r["prompt_id"], r["image"]))
    images = sorted({r["image"] for r in chosen})
    for c in CONDS:
        miss = [i for i in images if i not in aes[c]]
        if miss:
            raise SystemExit(f"{c} 缺美學讀數：{miss}")
    count = Counter(r["image"] for r in chosen)
    rng = random.Random(SEED)
    boots = [[rng.choice(images) for _ in images] for _ in range(BOOTS)]
    w_of = {"cell_weighted": lambda i: count[i], "image_level": lambda i: 1}
    mean = lambda c, m, ims, w: sum(aes[c][i][m] * w(i) for i in ims) / sum(w(i) for i in ims)
    rows = []
    for name, w in w_of.items():
        n = len(chosen) if name == "cell_weighted" else len(images)
        for m in METRICS:
            for c in CONDS:
                lo, hi = ci([mean(c, m, b, w) for b in boots])
                rows.append({"scheme": name, "metric": m, "row": c, "n": n, "mean": round(mean(c, m, images, w), 5),
                             "ci_low": round(lo, 5), "ci_high": round(hi, 5)})
            for c in CONDS[1:]:
                d = lambda b: mean("Full", m, b, w) - mean(c, m, b, w)
                lo, hi = ci([d(b) for b in boots])
                rows.append({"scheme": name, "metric": m, "row": f"Full-{c}", "n": n, "mean": round(d(images), 5),
                             "ci_low": round(lo, 5), "ci_high": round(hi, 5)})
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for fname, data in (("cells.csv", chosen), ("summary.csv", rows)):
        with open(a.output_dir / fname, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            wr.writeheader()
            wr.writerows(data)
    per_pur = Counter(r["purification"] for r in chosen)
    img_pur = {p: len({r["image"] for r in chosen if r["purification"] == p}) for p in PURIFS}
    lines = [f"Cells blocked by Full, B2_lpips0p85 and B3_lpips0p55 (undefended edit successful): {len(chosen)} cells, {len(images)} images",
             "  per purification (cells / images): " + ", ".join(f"{p} {per_pur[p]} / {img_pur[p]}" for p in PURIFS),
             "Failure labels per condition (a cell can carry several):"]
    for c in CONDS:
        lab = Counter(x for r in chosen for x in r[f"labels_{c}"].split(";") if x)
        lines.append(f"  {c}: " + ", ".join(f"{k} {lab[k]}" for k in ("no accessory", "mismatch", "degraded")))
    lines.append(f"{'scheme':14s} {'metric':14s} {'row':18s} {'n':>4s} {'mean':>8s}  95% CI")
    lines += [f"{r['scheme']:14s} {r['metric']:14s} {r['row']:18s} {r['n']:>4} {r['mean']:>8}  [{r['ci_low']}, {r['ci_high']}]" for r in rows]
    (a.output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if a.cells is None or a.sheets_dir is None:
        return
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans", fallback_to_default=False), 14)
    paths = {}
    for r in read(a.cells):
        paths[(r["image"], r["prompt_id"], r["purification"], r["condition"])] = r

    def tile(p):
        if not Path(p).is_file():
            raise SystemExit(f"缺影像：{p}")
        return Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS)

    heads = ["Original", "Full", "B2_lpips0p85", "B3_lpips0p55", "Undefended edit", "Full edit", "B2_lpips0p85 edit",
             "B3_lpips0p55 edit"]
    LEFT, HEAD, CAP, PAD = 230, 26, 20, 4
    a.sheets_dir.mkdir(parents=True, exist_ok=True)
    for page in range(0, len(chosen), ROWS_PER_PAGE):
        part = chosen[page:page + ROWS_PER_PAGE]
        sheet = Image.new("RGB", (LEFT + 8 * (TILE + PAD), HEAD + len(part) * (TILE + CAP + PAD)), "white")
        d = ImageDraw.Draw(sheet)
        for j, h in enumerate(heads):
            d.text((LEFT + j * (TILE + PAD) + TILE // 2, HEAD // 2), h, fill="black", font=font, anchor="mm")
        for i, r in enumerate(part):
            y = HEAD + i * (TILE + CAP + PAD)
            rec = {c: paths[(r["image"], r["prompt_id"], r["purification"], c)] for c in ("Undefended",) + tuple(CONDS)}
            prompt = rec["Full"]["prompt_text"]
            wrapped = "\n".join(textwrap.wrap(f"{r['prompt_id']}: {prompt}", 26))
            text = f"{r['image']}\n{wrapped}\n{r['purification']}\n" + "\n".join(
                f"{c.split('_')[0]}: {r[f'labels_{c}'] or '-'}" for c in CONDS)
            d.multiline_text((6, y + 6), text, fill="black", font=font)
            cols = [rec["Full"]["original_png"]] + [rec[c]["condition_input_png"] for c in CONDS] + [
                rec[c]["edit_png"] for c in ("Undefended",) + tuple(CONDS)]
            for j, p in enumerate(cols):
                x = LEFT + j * (TILE + PAD)
                sheet.paste(tile(p), (x, y))
                if 1 <= j <= 3:
                    v = aes[CONDS[j - 1]][r["image"]]
                    d.text((x + TILE // 2, y + TILE + CAP // 2), f"LAION {v['aes_laion']:.2f} / MUSIQ {v['aes_musiq_ava']:.2f}",
                           fill="black", font=font, anchor="mm")
        out = a.sheets_dir / f"all_blocked_page{page // ROWS_PER_PAGE + 1}.png"
        sheet.save(out)
        print(f"[SHEET] {out}", flush=True)


if __name__ == "__main__":
    main()
