"""非 SD 系編輯器（FLUX.2 [klein] 4B、VIBE）對一張影像的跨編輯器編輯，輸入逐格由清單指定。

輸入清單 `--manifest`（CSV：arm、purifier、image、input_png）：每列一個（條件, 設定）的輸入影像；設定為 none（淨化前，
條件 undefended 為原圖、其餘為 ip2p 求得的防禦圖）、七道算子或 vae_trans（ip2p 流程送進編輯器的淨化後輸入）。
輸出版面同 FLUX 512（scripts/evaluate_pexels_flux_512.sh），讀數程式可直接沿用：
    淨化前  <--output-root>/edits/<條件>/<影像>__p<k>.png
    淨化後  <--output-root>/purified_edits/<條件>/<設定>/<影像>__p<k>.png
    逐格表  <--chunks-dir>/edits_<條件>__<設定>.csv（欄 arm、image、prompt_index、prompt、png、input_png、seconds、model、settings、seed）
    完成標記 <--done-dir>/<影像>__<條件>__<設定>.done；已有標記的（條件, 設定）略過，未標記者整組重做。
身分讀數不在此計算（VIBE 的獨立 venv 沒有 ArcFace 相依），由讀數階段的 measure_semantic_identity --edits-csv 量測。
設定：klein 4 步、guidance 1.0；klein_base 50 步、guidance 4.0；vibe 20 步、guidance 4.5、image guidance 1.2；
vibe_distilled 20 步。512²，種子 20260812，指令為 configs/prompts/ultraedit_templates.json 的 add 句型；輸入以 LANCZOS 讀為 512²，
輸出不是 512² 時失敗。
用法：python scripts/run_nonsd_edits.py --editor <klein|klein_base|vibe|vibe_distilled> --manifest <CSV>
      --output-root <根> --chunks-dir <目錄> --done-dir <目錄>
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image

RES, SEED = 512, 20260812
FIELDS = ["arm", "image", "prompt_index", "prompt", "png", "input_png", "seconds", "model", "settings", "seed"]
SETTINGS = {"klein": ("black-forest-labs/FLUX.2-klein-4B", dict(num_inference_steps=4, guidance_scale=1.0)),
            "klein_base": ("black-forest-labs/FLUX.2-klein-base-4B", dict(num_inference_steps=50, guidance_scale=4.0)),
            "vibe": ("iitolstykh/VIBE-Image-Edit", dict(image_guidance_scale=1.2, guidance_scale=4.5, num_inference_steps=20)),
            "vibe_distilled": ("iitolstykh/VIBE-Image-Edit-DistilledCFG", dict(num_inference_steps=20))}
TEMPLATES = Path(__file__).resolve().parents[1] / "configs/prompts/ultraedit_templates.json"


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editor", choices=sorted(SETTINGS), required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--chunks-dir", type=Path, required=True)
    ap.add_argument("--done-dir", type=Path, required=True)
    args = ap.parse_args()
    with open(args.manifest, encoding="utf-8", newline="") as f:
        plan = list(csv.DictReader(f))
    for r in plan:
        if not Path(r["input_png"]).is_file():
            raise SystemExit(f"缺輸入 {r['input_png']}")
    prompts = json.loads(TEMPLATES.read_text(encoding="utf-8"))["add"]
    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    args.done_dir.mkdir(parents=True, exist_ok=True)
    edit = None
    for r in plan:
        arm, purifier, name, src = r["arm"], r["purifier"], r["image"], Path(r["input_png"])
        marker = args.done_dir / f"{name}__{arm}__{purifier}.done"
        if marker.exists():
            continue
        if edit is None:
            repo, kw, edit = load_editor(args.editor)
        d = args.output_root / ("edits" if purifier == "none" else "purified_edits") / arm
        if purifier != "none":
            d = d / purifier
        d.mkdir(parents=True, exist_ok=True)
        img = Image.open(src).convert("RGB").resize((RES, RES), Image.LANCZOS)
        rows = []
        for k, prompt in enumerate(prompts):
            t = time.time()
            out = edit(img, prompt)
            if out.size != (RES, RES):
                raise SystemExit(f"{args.editor} 輸出 {out.size}，不是 {RES}²")
            png = d / f"{name}__p{k}.png"
            out.save(png)
            rows.append({"arm": arm, "image": name, "prompt_index": k, "prompt": prompt, "png": png.as_posix(),
                         "input_png": src.as_posix(), "seconds": round(time.time() - t, 2), "model": repo,
                         "settings": json.dumps(kw), "seed": SEED})
        out_csv = args.chunks_dir / f"edits_{arm}__{purifier}.csv"
        with open(out_csv.with_suffix(".csv.tmp"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(rows)
        out_csv.with_suffix(".csv.tmp").replace(out_csv)
        marker.touch()
        print(f"[DONE] {name} {arm} {purifier}", flush=True)
    print("[ALLDONE]", flush=True)


if __name__ == "__main__":
    main()
