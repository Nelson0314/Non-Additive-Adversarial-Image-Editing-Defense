"""Pexels 人像的 SD inpainting 跨編輯器編輯：七條件 × {淨化前、jpeg50、blur2、VAE-Trans} × 4 句，輸入為 ip2p 流程的產物，不重新求解。

條件：undefended、等失真臂 photoguard_encoder、dia_pt、dct_shield、sifm、BlurGuard、color3d_scale2p5_pexels（ImmuLUT）。
輸入解析與 scripts/evaluate_pexels_nonsd_editor.sh 相同：
    undefended     淨化前為原圖 data/pexels_portraits/<類別>/<影像>.png；淨化後取 artifacts/pexels_portraits/<根>/purified_edits/undefended/
    等失真臂       artifacts/pexels_portraits/aligned/<根>/{defenses,purified_edits}/；VAE-Trans vae_trans/<根>/inputs/aligned/
    ImmuLUT        color trial artifacts_pexels/{defenses,purified_edits}/；VAE-Trans vae_trans/<根>/inputs/color3d_scale2p5/
    blurguard      第一批在原生產物根與 vae_trans/<根>/inputs/blurguard/；第二批在 runtime/pexels_portraits/blurguard_rest/
編輯協定與 core `pipelines/editing.py` 的 inpaint 場景（`--sampler official`）相同：runwayml/stable-diffusion-inpainting、fp32、
StableDiffusionInpaintPipeline、strength 1.0、50 步、guidance 7.5、生成器種子 20260812（cuda）、512²；影像與遮罩以
`load_image_tensor(size=512)` 讀入，遮罩 data/pexels_portraits/masks/<影像>.png 以 0.5 二值化（白＝重繪＝主體的補集）；
輸出合成回遮罩外原輸入（mask·raw + (1−mask)·x），pipeline 原始輸出另存 `__raw.png`。指令為 prompts.yaml 的 edits.inpaint，
`{content}` 由類別代入。
輸出版面同 scripts/run_nonsd_edits.py：
    淨化前  <--output-root>/edits/<條件>/<影像>__p<k>.png
    淨化後  <--output-root>/purified_edits/<條件>/<設定>/<影像>__p<k>.png
    逐格表  <--chunks-dir>/<影像>/edits_<條件>__<設定>.csv
    完成標記 <--done-dir>/<影像>__<條件>__<設定>.done；已有標記者略過。
缺任何輸入時失敗。`--check-only` 只檢查輸入是否齊全並列出缺項，不載入模型。
用法（遠端 baseline 專案根，source scripts/env.sh 後）：
    $PY scripts/run_inpaint_edits.py --images <影像>... [--check-only]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml

from immunization_core.artifacts.images import save_image
from immunization_core.editors.inpainting import SDInpaintWrapper
from immunization_core.io import load_image_tensor

MODEL = "runwayml/stable-diffusion-inpainting"
RES, SEED, STEPS, GUIDANCE = 512, 20260812, 50, 7.5
CONDITIONS = ("undefended", "photoguard_encoder", "dia_pt", "dct_shield", "sifm", "blurguard", "color3d_scale2p5_pexels")
PURIFIERS = ("none", "jpeg50", "blur2", "vae_trans")
ALIGNED = ("photoguard_encoder", "dia_pt", "dct_shield", "sifm")
DATA = Path("data/pexels_portraits")
FIELDS = ["arm", "image", "prompt_index", "prompt", "png", "raw_png", "input_png", "mask_png", "seconds", "model",
          "settings", "seed"]


def build_manifest(img, sets, blurguard_first, blurguard_units):
    s = next(s for s in sets if img in s["images"])
    root = s["root"] or ""
    cls = img.split("_")[0]
    na = Path("artifacts/pexels_portraits") / root
    aa = Path("artifacts/pexels_portraits/aligned") / root
    vt = Path("artifacts/pexels_portraits/vae_trans") / root / "inputs"
    ct = Path("../color/trials/lut3d_defense_and_naturalness/artifacts_pexels")
    if img in blurguard_first:
        bg = (na / "defenses", na / "purified_edits", vt / "blurguard")
    else:
        unit = next(u for u, members in blurguard_units.items() if img in members)
        br = Path("runtime/pexels_portraits/blurguard_rest")
        bg = (br / "artifacts" / unit / "defenses", br / "artifacts" / unit / "purified_edits",
              br / "vae_trans" / unit / "inputs" / "blurguard")
    roots = {"undefended": (None, na / "purified_edits", vt / "native"),
             "blurguard": bg,
             "color3d_scale2p5_pexels": (ct / "defenses", ct / "purified_edits", vt / "color3d_scale2p5")}
    roots.update({c: (aa / "defenses", aa / "purified_edits", vt / "aligned") for c in ALIGNED})
    rows = []
    for c in CONDITIONS:
        d, p, v = roots[c]
        rows.append((c, "none", DATA / cls / f"{img}.png" if d is None else d / c / f"{img}__{c}__def.png"))
        rows += [(c, op, p / c / op / f"ip2p_{c}_{op}" / f"{img}__orig.png") for op in ("jpeg50", "blur2")]
        rows.append((c, "vae_trans", v / c / "vae_trans" / f"ip2p_{c}_vae_trans" / f"{img}__orig.png"))
    return cls, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/pexels_portraits/inpaint"))
    ap.add_argument("--chunks-dir", type=Path, default=Path("runtime/pexels_portraits/inpaint/chunks"))
    ap.add_argument("--done-dir", type=Path, default=Path("runtime/pexels_portraits/inpaint/done"))
    args = ap.parse_args()

    sets = json.loads((DATA / "image_sets.json").read_text(encoding="utf-8"))["sets"]
    first = set(Path("runtime/pexels_portraits/blurguard/images.txt").read_text(encoding="utf-8").split())
    units = {p.stem: set(p.read_text(encoding="utf-8").split())
             for p in sorted(Path("runtime/pexels_portraits/blurguard_rest/units").glob("*.txt"))}
    plans, missing = {}, []
    for img in args.images:
        cls, rows = build_manifest(img, sets, first, units)
        mask = DATA / "masks" / f"{img}.png"
        missing += [(img, c, op, x.as_posix()) for c, op, x in rows if not x.is_file()]
        if not mask.is_file():
            missing.append((img, "mask", "-", mask.as_posix()))
        plans[img] = (cls, rows, mask)
    if args.check_only:
        print(f"[CHECK] {len(args.images)} 張 × {len(CONDITIONS)} 條件 × {len(PURIFIERS)} 設定；缺 {len(missing)} 個輸入")
        for m in missing:
            print("[MISSING]", *m)
        raise SystemExit(1 if missing else 0)
    if missing:
        raise SystemExit(f"缺 {len(missing)} 個輸入，例如 {missing[:3]}")

    spec = yaml.safe_load((DATA / "prompts.yaml").read_text(encoding="utf-8"))
    templates = spec["edits"]["inpaint"]
    settings = json.dumps({"num_inference_steps": STEPS, "guidance_scale": GUIDANCE, "strength": 1.0,
                           "sampler": "official", "dtype": "float32"})
    args.done_dir.mkdir(parents=True, exist_ok=True)
    victim = None
    for img, (cls, rows, mask_png) in plans.items():
        content = spec[cls]["content"]
        chunks = args.chunks_dir / img
        chunks.mkdir(parents=True, exist_ok=True)
        for arm, purifier, src in rows:
            marker = args.done_dir / f"{img}__{arm}__{purifier}.done"
            if marker.exists():
                continue
            if victim is None:
                victim = SDInpaintWrapper(MODEL, dtype=torch.float32)
                device = victim.device
            d = args.output_root / ("edits" if purifier == "none" else "purified_edits") / arm
            if purifier != "none":
                d = d / purifier
            d.mkdir(parents=True, exist_ok=True)
            x01 = load_image_tensor(src, device, size=RES)
            mask = (load_image_tensor(mask_png, device, size=RES)[:, :1] >= 0.5).to(x01.dtype)
            out = []
            for k, template in enumerate(templates):
                prompt = template.format(content=content)
                t = time.time()
                with torch.no_grad():
                    gen = torch.Generator(device=device).manual_seed(SEED)
                    raw = victim.pipe(prompt=prompt, image=x01, mask_image=mask, strength=1.0,
                                      num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=gen,
                                      output_type="pt").images
                if tuple(raw.shape[-2:]) != (RES, RES):
                    raise SystemExit(f"inpainting 輸出 {tuple(raw.shape[-2:])}，不是 {RES}²")
                edit = mask * raw + (1.0 - mask) * x01
                png, raw_png = d / f"{img}__p{k}.png", d / f"{img}__p{k}__raw.png"
                save_image(raw, raw_png)
                save_image(edit, png)
                out.append({"arm": arm, "image": img, "prompt_index": k, "prompt": prompt, "png": png.as_posix(),
                            "raw_png": raw_png.as_posix(), "input_png": src.as_posix(), "mask_png": mask_png.as_posix(),
                            "seconds": round(time.time() - t, 2), "model": MODEL, "settings": settings, "seed": SEED})
            out_csv = chunks / f"edits_{arm}__{purifier}.csv"
            tmp = out_csv.with_suffix(".csv.tmp")
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
                w.writeheader()
                w.writerows(out)
            tmp.replace(out_csv)
            marker.touch()
            print(f"[DONE] {img} {arm} {purifier} {sum(r['seconds'] for r in out) / len(out):.1f}s/格", flush=True)
    print("[ALLDONE]", flush=True)


if __name__ == "__main__":
    main()
