"""UltraEdit（SD3）全表：分母 + 12 條件，未淨化與七道淨化算子，ip2p 場景的四條指令。

輸入（全部沿用主表已產好的影像，不重跑防禦、不重跑淨化）
────────────────────────────────────────────────────────────────────
- 分母、未淨化：`data/portraits/<類>/<圖>.png`。
- 條件、未淨化：`<--defenses-root>/<條件>/<圖>__<條件>__def.png`（預設 `artifacts/defenses`），與 FLUX 全表同一份。
- 淨化後（分母與條件皆同）：`<--purified-edits-root>/<條件>/<算子>/ip2p_<條件>_<算子>/<圖>__orig.png`
  （預設 `artifacts/purified_edits`），即主表 ip2p 那一格實際送進編輯器的影像（`run_edits` 存下的輸入）。
  兩個編輯器因此使用逐位元相同的輸入。

編輯器與協定
────────────────────────────────────────────────────────────────────
`BleachNick/SD3_UltraEdit_freeform`（管線見 `third_party/ultraedit/pipeline.py`），512×512，
50 步，`negative_prompt=""`，種子 20260812。指令句型與 guidance 由參數給定，
逐列寫進 CSV；選定的值與選法見 STATUS.md「UltraEdit」一節。

輸出版面與主表相同，`measure_edit_displacement`／`measure_purified_displacement` 可直接讀取：

| 格 | 路徑（相對 `--output-dir`） |
|---|---|
| 分母、未淨化 | `undefended_edits/ultraedit_undefended/<圖>__p<N>.png` |
| 條件、未淨化 | `defended_edits/<條件>/ultraedit_<條件>/<圖>__p<N>.png`，同層 `preflight.csv` |
| 淨化後 | `purified_edits/<條件或 undefended>/<算子>/ip2p_<條件>_<算子>/<圖>__p<N>.png` |

淨化後那一層的中段仍寫 `ip2p`，因為 `pipelines.retention.cell()` 以場景名組路徑，
而這裡的「場景」是指令來源 `edits.ip2p`，不是編輯器。

逐格 CSV：`results/ultraedit/edits/<arm>.csv`（每個 arm 一檔，兩張卡寫不同的檔），
含 id_orig／arcface_orig（參考圖是乾淨原圖）。續跑須通過協定摘要、輸入雜湊與 PNG 檢查。

用法（baseline 專案根目錄，一張卡一份）
    CUDA_VISIBLE_DEVICES=<卡> python -m immunization_baseline.cli.run_ultraedit_edits \\
        --arms undefended mist dct_shield ... --variant verbatim --guidance 2.5 --image-guidance 1.5
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from immunization_baseline import conditions, layout  # noqa: E402

import torch  # noqa: E402

from immunization_core.pipelines.editing import defended_image, identity_row, load_items  # noqa: E402
from immunization_baseline.editors import EDITORS, load_pipeline, to_pil, to_tensor  # noqa: E402
from immunization_core.purifiers.protocol import PURIFIERS, label  # noqa: E402
from immunization_core.io import load_image_tensor  # noqa: E402
from immunization_baseline.resume_state import file_digest, load_resume_rows, protocol_digest, write_rows_atomic  # noqa: E402

EDITOR = "sd3-ultraedit"
RES = 512
EDIT_SEED = 20260812
STEPS = 50
UNDEFENDED = "undefended"
OPERATORS = [label(k, s) for k, s in PURIFIERS if k != "identity"]
#: 可用的條件：configs/conditions.yaml 的全部條件（含非主表的 color_curve）。
CONDITIONS = list(conditions.CONDITIONS)

FIELDS = ["arm", "purifier", "image", "prompt_index", "variant", "prompt", "model",
          "guidance_scale", "image_guidance_scale", "steps", "seed", "seconds",
          "input_png", "png", "id_orig", "arcface_orig", "face_found_orig", "face_found_edit",
          "protocol_id", "input_sha256", "reference_png", "reference_sha256"]


def input_png(arm: str, purifier: str, item: dict, defenses: Path,
              purified_edits: Path) -> Path:
    if purifier != "none":
        path = (purified_edits / arm / purifier / f"ip2p_{arm}_{purifier}"
                / f"{item['name']}__orig.png")
        if not path.is_file():
            raise SystemExit(f"找不到淨化後的輸入：{path}")
        return path
    if arm == UNDEFENDED:
        return Path(item["path"])
    return defended_image(defenses / arm, item["name"])


def output_png(root: Path, arm: str, purifier: str, name: str, pi: int) -> Path:
    if purifier != "none":
        d = root / "purified_edits" / arm / purifier / f"ip2p_{arm}_{purifier}"
    elif arm == UNDEFENDED:
        d = root / "undefended_edits" / "ultraedit_undefended"
    else:
        d = root / "defended_edits" / arm / f"ultraedit_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}__p{pi}.png"


def write_preflight(root: Path, arm: str, targets: list, prompts: list) -> None:
    """`pipelines.displacement` 從每個條件目錄的 preflight.csv 列舉格子。"""
    path = root / "defended_edits" / arm / "preflight.csv"
    rows = [{"scenario": "ip2p", "arm": f"ultraedit_{arm}", "image": item["name"],
             "prompt_index": pi, "prompt": prompt}
            for item in targets for pi, prompt in enumerate(prompts)]
    write_rows_atomic(path, ["scenario", "arm", "image", "prompt_index", "prompt"], rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True,
                    help=f"`{UNDEFENDED}` 或 configs/conditions.yaml 的條件之一：{' '.join(CONDITIONS)}")
    ap.add_argument("--purifiers", nargs="+", default=["none"] + OPERATORS)
    ap.add_argument("--variant", default="verbatim")
    ap.add_argument("--prompt-sets", type=Path,
                    default=layout.CONFIGS / "prompts" / "ultraedit_templates.json")
    ap.add_argument("--guidance", type=float, required=True)
    ap.add_argument("--image-guidance", type=float, required=True)
    ap.add_argument("--output-dir", dest="root", type=Path, default=layout.ULTRAEDIT_EDITS)
    ap.add_argument("--defenses-root", dest="defenses", type=Path, default=layout.DEFENSES,
                    help="條件防禦圖的根目錄，`<條件>/<圖>__<條件>__def.png`")
    ap.add_argument("--purified-edits-root", dest="purified_edits", type=Path, default=layout.PURIFIED_EDITS,
                    help="主表淨化後編輯的根目錄，取其中 ip2p 格的輸入影像")
    ap.add_argument("--output-csv-dir", dest="results", type=Path, default=layout.RESULTS / "ultraedit" / "edits",
                    help="不同協定須使用獨立結果目錄與 --output-dir")
    args = ap.parse_args()

    bad = [a for a in args.arms if a != UNDEFENDED and a not in CONDITIONS]
    bad += [p for p in args.purifiers if p != "none" and p not in OPERATORS]
    if bad:
        raise SystemExit(f"未知的 arm／算子：{bad}")

    if len(set(args.arms)) != len(args.arms) or len(set(args.purifiers)) != len(args.purifiers):
        ap.error("--arms 與 --purifiers 不得重複")

    items, _ = load_items(layout.PORTRAITS)
    targets = sorted(items, key=lambda i: i["name"])
    prompts = json.loads(args.prompt_sets.read_text(encoding="utf-8"))[args.variant]
    if not targets or not prompts:
        ap.error("影像或指令集合為空")
    spec = EDITORS[EDITOR]

    states = {}
    for arm in args.arms:
        out_csv = args.results / f"{arm}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        protocol_id = protocol_digest({
            "editor": EDITOR, "model": spec["repo"], "call_kw": spec["call_kw"],
            "arm": arm, "resolution": RES, "steps": STEPS, "seed": EDIT_SEED,
            "precision": "float16", "variant": args.variant, "prompts": prompts,
            "guidance": args.guidance, "image_guidance": args.image_guidance,
            "data": str(layout.PORTRAITS.resolve()), "defenses": str(args.defenses.resolve()),
            "purified_edits": str(args.purified_edits.resolve()),
            "output": str(args.root.resolve()),
        })
        rows = load_resume_rows(out_csv, FIELDS, protocol_id,
                                ("arm", "purifier", "image", "prompt_index"))
        done = {(r["purifier"], r["image"], int(r["prompt_index"])) for r in rows}
        for purifier in args.purifiers:
            for item in targets:
                input_png(arm, purifier, item, args.defenses, args.purified_edits)  # 載入模型前驗收全部輸入。
                for pi in range(len(prompts)):
                    png = output_png(args.root, arm, purifier, item["name"], pi)
                    if (purifier, item["name"], pi) not in done and png.exists():
                        raise FileExistsError(f"未登錄的既有產物：{png}；請使用新的輸出目錄")
        states[arm] = (out_csv, protocol_id, rows, done)

    pipe = None
    for arm in args.arms:
        out_csv, protocol_id, rows, done = states[arm]
        t_arm = time.time()
        for purifier in args.purifiers:
            for item in targets:
                pending = [(pi, prompt) for pi, prompt in enumerate(prompts)
                           if (purifier, item["name"], pi) not in done]
                if not pending:
                    continue
                if pipe is None:
                    pipe = load_pipeline(EDITOR)
                    device = pipe.device
                    refs = {i["name"]: load_image_tensor(i["path"], device, size=RES) for i in targets}
                src = input_png(arm, purifier, item, args.defenses, args.purified_edits)
                x_pil = to_pil(load_image_tensor(src, device, size=RES))
                for pi, prompt in pending:
                    t0 = time.time()
                    with torch.no_grad():
                        out = pipe(prompt=prompt, image=x_pil, num_inference_steps=STEPS,
                                   guidance_scale=args.guidance,
                                   image_guidance_scale=args.image_guidance,
                                   height=RES, width=RES, **spec["call_kw"],
                                   generator=torch.Generator(device=device)
                                   .manual_seed(EDIT_SEED)).images[0]
                    dt = time.time() - t0
                    png = output_png(args.root, arm, purifier, item["name"], pi)
                    out.save(png)
                    idr = identity_row(refs[item["name"]], to_tensor(out, device))
                    rows.append({
                        "arm": arm, "purifier": purifier, "image": item["name"],
                        "prompt_index": pi, "variant": args.variant, "prompt": prompt,
                        "model": spec["repo"], "guidance_scale": args.guidance,
                        "image_guidance_scale": args.image_guidance, "steps": STEPS,
                        "seed": EDIT_SEED, "seconds": round(dt, 1),
                        "input_png": src.as_posix(), "png": png.as_posix(), **idr,
                        "protocol_id": protocol_id, "input_sha256": file_digest(src),
                        "reference_png": item["path"].as_posix(),
                        "reference_sha256": file_digest(item["path"])})
                    write_rows_atomic(out_csv, FIELDS, rows)
                    done.add((purifier, item["name"], pi))
            print(f"[DONE] {arm:18s} {purifier:14s} 累計 {len(rows)} 格  "
                  f"({time.time() - t_arm:.0f}s)", flush=True)
        if arm != UNDEFENDED and all(("none", item["name"], pi) in done
                                     for item in targets for pi in range(len(prompts))):
            write_preflight(args.root, arm, targets, prompts)
        print(f"[ARM] {arm} -> {out_csv}", flush=True)
    print("[ALLDONE]", flush=True)


if __name__ == "__main__":
    main()
