"""SD inpainting 質性圖的原生預算版：加性五法以各自論文的原生預算（`docs/reference/BASELINE_PROVENANCE.md`）求得的防禦圖，
經 {淨化前、jpeg50、blur2、VAE-Trans} × 4 句編輯，並量編輯結果 LPIPS 與防禦圖 LPIPS。

條件與防禦圖：
    photoguard_encoder、dia_pt、dct_shield、sifm  ip2p 流程的原生預算防禦圖 artifacts/pexels_portraits/<根>/defenses/<條件>/
    blurguard  官方設定（`generate_blurguard_defenses` 預設：ε 16/255、150 步、sigma_weighting 10）就地求解，
               輸出 <--output-root>/<影像>/defenses/blurguard/
淨化：jpeg50、blur2 以 core `Purifier`（與 `pipelines/purification.py` 相同）；VAE-Trans 以 `VAETrans`
（color trial artifacts/purification/vae_trans/model/encoder.pt，與 `apply_vae_trans` 相同）。淨化後輸入存
<--output-root>/<影像>/inputs/<條件>/<設定>.png。
編輯協定與 scripts/run_inpaint_edits.py 相同（runwayml/stable-diffusion-inpainting、fp32、strength 1.0、50 步、guidance 7.5、
種子 20260812、512²、遮罩以 0.5 二值化、輸出合成回遮罩外輸入）。
讀數 <--output-root>/<影像>/readouts.csv：edit_lpips＝piq.LPIPS(本條件編輯, 同淨化的未防禦編輯
artifacts/pexels_portraits/inpaint/{edits,purified_edits}/undefended/)；defense_lpips＝piq.LPIPS(防禦圖, 原圖)。
用法（遠端 baseline 專案根，source scripts/env.sh 後，在已取得的卡上）：
    $PY scripts/run_native_inpaint_edits.py --images woman_14 man_81
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import piq
import torch
import yaml

from immunization_baseline.vae_trans import VAETrans
from immunization_core.artifacts.images import save_image
from immunization_core.editors.inpainting import SDInpaintWrapper
from immunization_core.io import load_image_tensor
from immunization_core.purifiers.operators import Purifier

MODEL = "runwayml/stable-diffusion-inpainting"
RES, SEED, STEPS, GUIDANCE = 512, 20260812, 50, 7.5
ARMS = ("photoguard_encoder", "dia_pt", "dct_shield", "sifm", "blurguard")
PURIFIERS = {"jpeg50": ("jpeg", 50), "blur2": ("blur", 2.0)}
DATA = Path("data/pexels_portraits")
ENC = Path("../color/trials/lut3d_defense_and_naturalness/artifacts/purification/vae_trans/model/encoder.pt")
UNDEF = Path("artifacts/pexels_portraits/inpaint")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/pexels_portraits/inpaint_native"))
    args = ap.parse_args()

    sets = json.loads((DATA / "image_sets.json").read_text(encoding="utf-8"))["sets"]
    spec = yaml.safe_load((DATA / "prompts.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    for img in args.images:
        root = next(s["root"] for s in sets if img in s["images"]) or ""
        out = args.output_root / img
        bg_dir = out / "defenses" / "blurguard"
        if not (bg_dir / f"{img}__blurguard__def.png").is_file():
            subprocess.run([sys.executable, "-m", "immunization_baseline.cli.generate_blurguard_defenses",
                            "--data-root", str(DATA), "--output-dir", str(bg_dir), "--images", img], check=True)
        defense = {c: Path("artifacts/pexels_portraits") / root / "defenses" / c / f"{img}__{c}__def.png" for c in ARMS[:4]}
        defense["blurguard"] = bg_dir / f"{img}__blurguard__def.png"
        missing = [p for p in defense.values() if not p.is_file()]
        if missing:
            raise SystemExit(f"缺防禦圖：{missing}")

    vae = VAETrans(ENC, device)
    lpips = piq.LPIPS().to(device)
    victim = SDInpaintWrapper(MODEL, dtype=torch.float32)
    for img in args.images:
        root = next(s["root"] for s in sets if img in s["images"]) or ""
        cls = img.split("_")[0]
        out = args.output_root / img
        orig = load_image_tensor(DATA / cls / f"{img}.png", device, size=RES)
        mask = (load_image_tensor(DATA / "masks" / f"{img}.png", device, size=RES)[:, :1] >= 0.5).to(orig.dtype)
        prompts = [t.format(content=spec[cls]["content"]) for t in spec["edits"]["inpaint"]]
        rows = []
        for c in ARMS:
            dpath = (Path("artifacts/pexels_portraits") / root / "defenses" / c / f"{img}__{c}__def.png" if c != "blurguard"
                     else out / "defenses" / "blurguard" / f"{img}__blurguard__def.png")
            d = load_image_tensor(dpath, device, size=RES)
            with torch.no_grad():
                dl = float(lpips(d, orig).mean())
            inputs = {"none": dpath}
            (out / "inputs" / c).mkdir(parents=True, exist_ok=True)
            for op, (kind, strength) in PURIFIERS.items():
                with torch.no_grad():
                    y = Purifier(kind, strength).evaluate(d).clamp(0, 1)
                inputs[op] = out / "inputs" / c / f"{op}.png"
                save_image(y, inputs[op])
            inputs["vae_trans"] = out / "inputs" / c / "vae_trans.png"
            save_image(vae(d), inputs["vae_trans"])
            for op, src in inputs.items():
                x01 = load_image_tensor(src, device, size=RES)
                edir = out / "edits" / c / op
                edir.mkdir(parents=True, exist_ok=True)
                for k, prompt in enumerate(prompts):
                    with torch.no_grad():
                        gen = torch.Generator(device=device).manual_seed(SEED)
                        raw = victim.pipe(prompt=prompt, image=x01, mask_image=mask, strength=1.0,
                                          num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=gen,
                                          output_type="pt").images
                    edit = mask * raw + (1.0 - mask) * x01
                    png = edir / f"{img}__p{k}.png"
                    save_image(edit, png)
                    u = (UNDEF / "edits" / "undefended" / f"{img}__p{k}.png" if op == "none"
                         else UNDEF / "purified_edits" / "undefended" / op / f"{img}__p{k}.png")
                    with torch.no_grad():
                        el = float(lpips(load_image_tensor(png, device, size=RES), load_image_tensor(u, device, size=RES)).mean())
                    rows.append({"image": img, "condition": c, "purifier": op, "prompt_index": k, "prompt": prompt,
                                 "edit_lpips": round(el, 4), "defense_lpips": round(dl, 4), "defense_png": dpath.as_posix(),
                                 "input_png": src.as_posix(), "edit_png": png.as_posix(), "undefended_png": u.as_posix()})
                print(f"[DONE] {img} {c} {op}", flush=True)
        with open(out / "readouts.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print("[ALLDONE]", flush=True)


if __name__ == "__main__":
    main()
