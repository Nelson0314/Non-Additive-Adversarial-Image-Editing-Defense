"""草稿（trial non_sd_editor_speed）：FLUX.2 [klein] 4B 與 VIBE 的跨編輯器編輯，輸入輸出版面同 run_ultraedit_edits。

輸入：淨化前為防禦圖（`--defenses-root/<條件>/<影像>__<條件>__def.png`，undefended 為原圖），七道與 VAE-Trans 為 ip2p 流程送進
編輯器的淨化後輸入（`--purified-edits-root/<條件>/<算子>/ip2p_<條件>_<算子>/<影像>__orig.png`）；BlurGuard 的兩批產物
分在不同根目錄，以 `--input-manifest`（CSV：arm、image、purifier、input_png）逐格指定輸入，取代上兩個根目錄。
輸出：`<輸出>/defended_edits/<條件>/<editor>_<條件>/`、`undefended_edits/<editor>_undefended/`、
`purified_edits/<條件>/<算子>/ip2p_<條件>_<算子>/<影像>__p<k>.png`；逐格表 `<--output-csv-dir>/edits_<條件>.csv`
（欄 arm、purifier、image、prompt_index、prompt、png、input_png、seconds、model、settings、seed），已完成的格續跑時略過。
身分讀數不在此計算（VIBE 的獨立 venv 沒有 ArcFace 相依），由讀數階段的 measure_semantic_identity --edits-csv 量測。
設定（測速所用）：klein 4 步、guidance 1.0；klein_base 50 步、guidance 4.0；vibe 20 步、guidance 4.5、image guidance 1.2；
vibe_distilled 20 步。512²，種子 20260812，指令為 configs/prompts/ultraedit_templates.json 的 add 句型。
用法：python run_nonsd_edits.py --editor <klein|klein_base|vibe|vibe_distilled> --data-root <資料根> --arms ... --purifiers ...
      --defenses-root <根> --purified-edits-root <根> [--input-manifest <CSV>] --output-dir <根> --output-csv-dir <目錄>
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image

RES, SEED = 512, 20260812
UNDEFENDED = "undefended"
FIELDS = ["arm", "purifier", "image", "prompt_index", "prompt", "png", "input_png", "seconds", "model", "settings", "seed"]
SETTINGS = {"klein": ("black-forest-labs/FLUX.2-klein-4B", dict(num_inference_steps=4, guidance_scale=1.0)),
            "klein_base": ("black-forest-labs/FLUX.2-klein-base-4B", dict(num_inference_steps=50, guidance_scale=4.0)),
            "vibe": ("iitolstykh/VIBE-Image-Edit", dict(image_guidance_scale=1.2, guidance_scale=4.5, num_inference_steps=20)),
            "vibe_distilled": ("iitolstykh/VIBE-Image-Edit-DistilledCFG", dict(num_inference_steps=20))}
TEMPLATES = Path(__file__).resolve().parents[2] / "configs/prompts/ultraedit_templates.json"


def load_editor(name):
    repo, kw = SETTINGS[name]
    if name.startswith("klein"):
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")

        def edit(img, prompt):
            g = torch.Generator("cuda").manual_seed(SEED)
            return pipe(prompt=prompt, image=img, height=RES, width=RES, generator=g, **kw).images[0]
    else:
        from huggingface_hub import snapshot_download
        from vibe.editor import ImageEditor
        editor = ImageEditor(checkpoint_path=snapshot_download(repo_id=repo), device="cuda", **kw)

        def edit(img, prompt):
            torch.manual_seed(SEED)
            res = editor.generate_edited_image(instruction=prompt, conditioning_image=img, num_images_per_prompt=1)
            return res[0] if isinstance(res, list) else res
    return repo, kw, edit


def input_png(args, manifest, arm, purifier, cls, name):
    if (arm, name, purifier) in manifest:
        return Path(manifest[(arm, name, purifier)])
    if purifier != "none":
        return args.purified_edits_root / arm / purifier / f"ip2p_{arm}_{purifier}" / f"{name}__orig.png"
    if arm == UNDEFENDED:
        return args.data_root / cls / f"{name}.png"
    return args.defenses_root / arm / f"{name}__{arm}__def.png"


def output_png(root, editor, arm, purifier, name, k):
    if purifier != "none":
        d = root / "purified_edits" / arm / purifier / f"ip2p_{arm}_{purifier}"
    elif arm == UNDEFENDED:
        d = root / "undefended_edits" / f"{editor}_undefended"
    else:
        d = root / "defended_edits" / arm / f"{editor}_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}__p{k}.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editor", choices=sorted(SETTINGS), required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--purifiers", nargs="+", required=True)
    ap.add_argument("--defenses-root", type=Path, required=True)
    ap.add_argument("--purified-edits-root", type=Path, required=True)
    ap.add_argument("--input-manifest", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--output-csv-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = {}
    if args.input_manifest:
        with open(args.input_manifest, encoding="utf-8", newline="") as f:
            manifest = {(r["arm"], r["image"], r["purifier"]): r["input_png"] for r in csv.DictReader(f)}
    prompts = json.loads(TEMPLATES.read_text(encoding="utf-8"))["add"]
    items = sorted((p.parent.name, p.stem) for cls in ("man", "woman") for p in (args.data_root / cls).glob("*.png"))
    plan = []
    for arm in args.arms:
        for purifier in args.purifiers:
            for cls, name in items:
                src = input_png(args, manifest, arm, purifier, cls, name)
                if not src.is_file():
                    raise SystemExit(f"缺輸入 {src}")
                plan.append((arm, purifier, name, src))
    args.output_csv_dir.mkdir(parents=True, exist_ok=True)
    edit = None
    for arm in args.arms:
        out_csv = args.output_csv_dir / f"edits_{arm}.csv"
        rows = list(csv.DictReader(open(out_csv, encoding="utf-8"))) if out_csv.is_file() else []
        done = {(r["purifier"], r["image"], r["prompt_index"]) for r in rows}
        for a, purifier, name, src in plan:
            if a != arm:
                continue
            for k, prompt in enumerate(prompts):
                if (purifier, name, str(k)) in done:
                    continue
                if edit is None:
                    repo, kw, edit = load_editor(args.editor)
                img = Image.open(src).convert("RGB").resize((RES, RES), Image.LANCZOS)
                t = time.time()
                out = edit(img, prompt)
                png = output_png(args.output_dir, args.editor, arm, purifier, name, k)
                out.save(png)
                rows.append({"arm": arm, "purifier": purifier, "image": name, "prompt_index": k, "prompt": prompt,
                             "png": png.as_posix(), "input_png": src.as_posix(), "seconds": round(time.time() - t, 2),
                             "model": repo, "settings": json.dumps(kw), "seed": SEED})
                tmp = out_csv.with_suffix(".csv.tmp")
                with open(tmp, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(rows)
                tmp.replace(out_csv)
        print(f"[ARM] {arm} {len(rows)} 格 → {out_csv}", flush=True)
    print("[ALLDONE]", flush=True)


if __name__ == "__main__":
    main()
