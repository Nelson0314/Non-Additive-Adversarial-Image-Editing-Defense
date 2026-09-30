"""SD 1.4／1.5 以外的 SD 系列編輯器，在主表資料集上的小樣本參數探索。

目的：找出一組「指令被執行、人沒被換掉」的參數，作為跨編輯器遷移的候選。
SDEdit（SD 1.5／2.1-base）三個參數維度都沒有交集，見 STATUS.md；這一支
換編輯器而不是換參數。

編輯器（`--editor`）
────────────────────────────────────────────────────────────────────
- `sdxl-ip2p`：`diffusers/sdxl-instructpix2pix-768`，SDXL 骨幹、以
  InstructPix2Pix 的資料重新訓練的指令式編輯器，原生 768×768。指令式，
  與 ip2p 場景的祈使句逐字相容。旋鈕是 `guidance_scale`（文字）與
  `image_guidance_scale`（原圖）。模型卡的範例值 guidance 3.0、image
  guidance 1.5、30 步是本腳本的預設值。
- `sd3-ultraedit`：`BleachNick/SD3_UltraEdit_freeform`，UltraEdit（arXiv:2407.05282）
  以 SD3-medium 為骨幹訓練的指令式編輯器，自由形式（不需遮罩）版本，512×512。
  管線不在官方 diffusers，移植自作者的 fork，見 `third_party/ultraedit/pipeline.py`。
  作者 README 的範例值是 guidance 7.5、image guidance 1.5、50 步、
  `negative_prompt=""`。
- `sdxl-img2img`、`sd3-img2img`：SDEdit（img2img），旋鈕是 `strength` 與
  `guidance_scale`，原生 1024×1024。不是指令式編輯器，祈使句只能當成描述。

指令與種子沿用 ip2p 場景（`prompts.yaml` 的 `edits.ip2p`，逐字不改，不加遮罩），
種子與 ip2p/inpaint 共用 20260812。身分讀數是 `immunization_core.pipelines.editing.identity_row`，
參考圖是 512×512 的原圖，編輯結果以原生解析度直接送進去（與 FLUX 那一支相同）。

指令句型（`--prompt-sets`）：JSON 檔，`{句型名: [四條指令]}`，四條依序對應
`edits.ip2p` 的四個配件。不給時只有一個句型 `verbatim`，即 `edits.ip2p` 原文。
句型是與參數並列的一個網格維度，逐列記在 `variant` 欄。

每一批輸出一張 CSV（`--output-csv`，每格寫完整份重寫，可續跑）與一張對照圖 `<--output-dir>/sheet.jpg`（列 = 參數組合，欄 = 影像 × 指令，
第一欄是原圖）。

用法（遠端，需要一張卡）
    CUDA_VISIBLE_DEVICES=<卡> python -m immunization_baseline.cli.sweep_editor_parameters \\
        --editor sdxl-ip2p --guidances 3 5 7.5 --image-guidances 1.2 1.5 \\
        --output-dir artifacts/sweeps/sdxl_ip2p/guidance/guidance_portrait_pair \\
        --output-csv results/sweeps/sdxl_ip2p/guidance/guidance_portrait_pair.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

from immunization_baseline import layout  # noqa: E402

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from immunization_core.io import write_rows_atomic  # noqa: E402
from immunization_core.pipelines.editing import identity_row, load_items  # noqa: E402
from immunization_core.io import load_image_tensor  # noqa: E402
from immunization_baseline.editors import EDITORS, load_pipeline, to_pil, to_tensor  # noqa: E402

REF_RESOLUTION = 512
EDIT_SEED = 20260812

FIELDS = ["editor", "model", "variant", "image", "prompt_index", "prompt", "resolution", "steps",
          "guidance_scale", "image_guidance_scale", "strength", "seed", "seconds", "png",
          "id_orig", "arcface_orig", "face_found_orig", "face_found_edit"]


def settings(args, kind: str, variants: list[str]) -> list[dict]:
    if kind == "ip2p":
        return [dict(variant=v, guidance_scale=g, image_guidance_scale=ig, strength="")
                for v, g, ig in itertools.product(variants, args.guidances,
                                                  args.image_guidances)]
    return [dict(variant=v, guidance_scale=g, image_guidance_scale="", strength=s)
            for v, s, g in itertools.product(variants, args.strengths, args.guidances)]


def tag(s: dict) -> str:
    """`verbatim` 不進標籤，句型維度加進來之前的檔名與 CSV 才對得上。"""
    parts = [] if s.get("variant", "verbatim") == "verbatim" else [s["variant"]]
    parts.append(f"g{s['guidance_scale']}")
    if s["image_guidance_scale"] != "":
        parts.append(f"ig{s['image_guidance_scale']}")
    if s["strength"] != "":
        parts.append(f"s{s['strength']}")
    return "_".join(parts)


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with csv_path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows_atomic(csv_path, FIELDS, rows)


def contact_sheet(out_dir: Path, originals: dict, cols: list, grid: list, rows: list,
                  cell: int = 192) -> Path:
    """列 = 參數組合，欄 = (影像, 指令)；第一列是原圖，每格下方標 id_orig。"""
    by_key = {(r["image"], int(r["prompt_index"]), r["_tag"]): r for r in rows}
    label_w, label_h = 150, 16
    W = label_w + cell * len(cols)
    H = (cell + label_h) * (len(grid) + 1)
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (name, pi) in enumerate(cols):
        x = label_w + j * cell
        sheet.paste(originals[name].resize((cell, cell)), (x, 0))
        draw.text((x + 2, cell + 2), f"{name} p{pi}", fill="black")
    draw.text((2, 2), "original", fill="black")
    for i, s in enumerate(grid, start=1):
        y = i * (cell + label_h)
        draw.text((2, y + 2), tag(s), fill="black")
        for j, (name, pi) in enumerate(cols):
            r = by_key.get((name, pi, tag(s)))
            if r is None:
                continue
            x = label_w + j * cell
            sheet.paste(Image.open(r["png"]).convert("RGB").resize((cell, cell)), (x, y))
            draw.text((x + 2, y + cell + 2), f"id {r['id_orig'] or 'none'}", fill="black")
    out = out_dir / "sheet.jpg"
    sheet.save(out, quality=88)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editor", choices=sorted(EDITORS), required=True)
    ap.add_argument("--data-root", dest="data", type=Path, default=layout.PORTRAITS)
    ap.add_argument("--output-dir", dest="out", type=Path, required=True, help="影像與對照圖的輸出目錄")
    ap.add_argument("--output-csv", dest="out_csv", type=Path, required=True,
                    help="這一批的 CSV；既有列視為已完成，續跑時沿用")
    ap.add_argument("--images", nargs="+", default=None, help="預設每個類別取第一張")
    ap.add_argument("--instruction-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidances", nargs="+", type=float, default=[3.0])
    ap.add_argument("--image-guidances", nargs="+", type=float, default=[1.5],
                    help="只有指令式編輯器（sdxl-ip2p）用得到")
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.5],
                    help="只有 img2img 用得到")
    ap.add_argument("--prompt-sets", type=Path, default=None,
                    help="JSON：{句型名: [四條指令]}；不給時只用 edits.ip2p 原文")
    args = ap.parse_args()

    spec = EDITORS[args.editor]
    items, edits = load_items(args.data)
    prompt_sets = {"verbatim": edits["ip2p"]}
    if args.prompt_sets is not None:
        prompt_sets = json.loads(args.prompt_sets.read_text(encoding="utf-8"))
        bad = {k: len(v) for k, v in prompt_sets.items() if len(v) != len(edits["ip2p"])}
        if bad:
            raise SystemExit(f"每個句型要剛好 {len(edits['ip2p'])} 條指令：{bad}")
    by_class = {}
    for item in items:
        by_class.setdefault(item["class"], item)
    targets = ([by_class[c] for c in sorted(by_class)] if args.images is None
               else [next(i for i in items if i["name"] == n) for n in args.images])
    grid = settings(args, spec["kind"], list(prompt_sets))
    cols = [(t["name"], pi) for t in targets for pi in args.instruction_indices]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_csv
    rows = load_rows(out_csv)
    for r in rows:
        r["variant"] = r.get("variant") or "verbatim"
        r["_tag"] = tag({"variant": r["variant"],
                         **{k: (r[k] if r[k] == "" else float(r[k]))
                            for k in ("guidance_scale", "image_guidance_scale", "strength")}})
    done = {(r["image"], int(r["prompt_index"]), r["_tag"]) for r in rows}
    print(f"editor {args.editor}，影像 {[t['name'] for t in targets]}，指令 "
          f"{args.instruction_indices}，{len(grid)} 組參數，已完成 {len(done)} 格", flush=True)

    pipe = load_pipeline(args.editor)
    device = pipe.device
    originals = {}
    for item in targets:
        x_ref = load_image_tensor(item["path"], device, size=REF_RESOLUTION)
        x_in = load_image_tensor(item["path"], device, size=spec["res"])
        originals[item["name"]] = (x_ref, to_pil(x_in))

    for s in grid:
        for name, pi in cols:
            key = (name, pi, tag(s))
            if key in done:
                continue
            x_ref, x_pil = originals[name]
            prompt = prompt_sets[s["variant"]][pi]
            kw = dict(prompt=prompt, image=x_pil, num_inference_steps=args.steps,
                      guidance_scale=s["guidance_scale"],
                      generator=torch.Generator(device=device).manual_seed(EDIT_SEED))
            kw.update(spec.get("call_kw", {}))
            if spec["kind"] == "ip2p":
                kw.update(image_guidance_scale=s["image_guidance_scale"],
                          height=spec["res"], width=spec["res"])
            else:
                kw.update(strength=s["strength"])
            t0 = time.time()
            with torch.no_grad():
                out = pipe(**kw).images[0]
            dt = time.time() - t0
            png = out_dir / f"{name}__p{pi}__{tag(s)}.png"
            out.save(png)
            idr = identity_row(x_ref, to_tensor(out, device))
            row = {"editor": args.editor, "model": spec["repo"],
                   "variant": s["variant"], "image": name,
                   "prompt_index": pi, "prompt": prompt, "resolution": spec["res"],
                   "steps": args.steps, **{k: s[k] for k in
                   ("guidance_scale", "image_guidance_scale", "strength")},
                   "seed": EDIT_SEED, "seconds": round(dt, 1), "png": png.as_posix(), **idr}
            rows.append({**row, "_tag": tag(s)})
            write_rows(out_csv, [{k: r[k] for k in FIELDS} for r in rows])
            print(f"  {name:10s} p{pi} {tag(s):24s} id_orig={idr['id_orig']}  "
                  f"arcface_orig={idr['arcface_orig']}  ({dt:.1f}s)", flush=True)

    sheet = contact_sheet(out_dir, {n: p for n, (_, p) in originals.items()}, cols, grid, rows)
    print(f"\n完成：{len(rows)} 格 -> {out_csv}\n對照圖 -> {sheet}")


if __name__ == "__main__":
    main()
