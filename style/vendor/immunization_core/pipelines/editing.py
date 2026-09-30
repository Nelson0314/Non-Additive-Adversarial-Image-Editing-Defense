"""對原圖或防禦圖執行 IP2P 與 inpainting 編輯，並記錄編輯結果對輸入的讀數。

兩個場景共用種子、步數與解析度；ip2p 作用於整張影像，inpainting 只重繪遮罩內
（資料集 `masks/`，白色表示重繪）。指令取自 `prompts.yaml` 的 `edits`，依場景
分組。輸入資料根與輸出目錄由呼叫端明確提供。

| 欄 | 定義 |
|---|---|
| `clip_gain` | CLIP(編輯結果, 指令) − CLIP(輸入, 指令) |
| `lpips`、`psnr` | 編輯結果對輸入 |
| `id_orig`、`arcface_orig` | FaceNet 與 ArcFace 餘弦（僅 `--identity`） |
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import yaml

from immunization_core.artifacts.images import save_image
from immunization_core.artifacts.layout import defended_image
from immunization_core.editors.inpainting import SDInpaintWrapper
from immunization_core.editors.instruct_pix2pix import (
    IP2P_SEED, IP2P_TEXT_GUIDANCE, MODEL_NAME as IP2P_MODEL, IP2PWrapper,
)
from immunization_core.io import load_image_tensor, write_csv
from immunization_core.metrics.suite import MetricSuite

INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
RESOLUTION = 512
EDIT_SEED = IP2P_SEED
# 兩條路徑統一 50 步（FaceLock Table A2、DiffusionGuard Appendix D），覆寫 IP2P
# adapter 的 100 步預設。
EDIT_STEPS = 50
INPAINT_GUIDANCE = 7.5
# 編輯使用的 IP2P image guidance；adapter 預設 1.5 不適用於本流程。
IP2P_EDIT_IMAGE_GUIDANCE = 1.8


def load_items(root: Path):
    """回傳影像清單與逐場景的指令表；缺 `edits` 時拒絕。"""
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    edits = spec.get("edits")
    if not edits:
        raise SystemExit(
            f"{root / 'prompts.yaml'} 缺 `edits`：指令須依場景分組，"
            "逐類別的 `prompts` 格式不適用")
    out = []
    for cls in sorted(k for k in spec if k != "edits"):
        for img in sorted((root / cls).glob("*.png")):
            mask = root / "masks" / f"{img.stem}.png"
            out.append({
                "name": img.stem, "class": cls, "path": img,
                "content": spec[cls]["content"],
                "mask": mask if mask.exists() else None,
            })
    return out, {k: list(v) for k, v in edits.items()}


def apply_defended(items, directory: Path):
    """以 `directory` 中的防禦圖取代輸入影像；遮罩與指令不變，缺圖即失敗。"""
    resolved = [defended_image(directory, item["name"]) for item in items]
    for item, source in zip(items, resolved):
        item["path"] = source
    return items


def identity_row(x_ref, x_edit) -> dict:
    """FaceNet 與 ArcFace 身分讀數；未偵測到臉時留空。"""
    from immunization_core.metrics import arcface
    from immunization_core.metrics.identity import embed, similarity

    e_ref, e_edit = embed(x_ref), embed(x_edit)
    fn = similarity(e_ref, e_edit)
    a_ref, a_edit = arcface.embed(x_ref), arcface.embed(x_edit)
    af = arcface.similarity(a_ref, a_edit)
    return {
        "id_orig": "" if fn is None else round(fn, 4),
        "arcface_orig": "" if af is None else round(af, 4),
        "face_found_orig": e_ref is not None,
        "face_found_edit": e_edit is not None,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", dest="data", type=Path, required=True,
                    help="資料集根目錄，含 prompts.yaml、各類別子目錄與 masks/")
    ap.add_argument("--output-dir", dest="out", type=Path, required=True)
    ap.add_argument("--scenarios", nargs="+", default=["ip2p", "inpaint"])
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--identity", action="store_true",
                    help="加量 FaceNet 與 ArcFace 身分讀數")
    ap.add_argument("--metrics-only", action="store_true",
                    help="不重新編輯，只由既有 PNG 重算讀數")
    ap.add_argument("--sampler", choices=("wrapper", "official"),
                    default="official",
                    help="inpaint 的取樣鏈：`official` 為 diffusers 的 "
                         "StableDiffusionInpaintPipeline，`wrapper` 為可微分的 "
                         "SDWrapper.inpaint（DDIM）")
    ap.add_argument("--instruction-set", default=None,
                    help="改用 prompts.yaml 的 edits 下另一個鍵作為指令來源")
    ap.add_argument("--suffix", default="",
                    help="輸出子目錄的後綴，使不同 arm 並存")
    ap.add_argument("--s-t", type=float, default=IP2P_TEXT_GUIDANCE,
                    help="IP2P 的文字 guidance，逐列記入 CSV")
    ap.add_argument("--s-i", type=float, default=IP2P_EDIT_IMAGE_GUIDANCE,
                    help="IP2P 的影像 guidance")
    ap.add_argument("--defenses-dir", dest="defended", type=Path, default=None,
                    help="改用此目錄中的 `<名稱>__def.png` 作為輸入；缺圖即失敗")
    ap.add_argument("--negative-prompt", default="",
                    help="IP2P negative prompt; recorded verbatim in CSV")
    ap.add_argument("--prompt-indices", nargs="+", type=int, default=None,
                    help="IP2P prompt indices to run, preserving their pN filenames")
    ap.add_argument("--ip2p-instruction", default=None,
                    help="Override exactly one selected IP2P instruction")
    ap.add_argument("--seed", type=int, default=None,
                    help="IP2P seed override; default is 20260812")
    ap.add_argument("--require-new-arm", action="store_true",
                    help="Refuse an existing arm directory or CSV key (for sweeps)")
    return ap


def main(argv=None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)

    ip2p_options = (args.negative_prompt or args.prompt_indices is not None
                    or args.ip2p_instruction is not None or args.seed is not None)
    if ip2p_options and args.scenarios != ["ip2p"]:
        ap.error("IP2P-only options require --scenarios ip2p")
    if args.ip2p_instruction is not None:
        if not args.ip2p_instruction.strip():
            ap.error("--ip2p-instruction must not be empty")
        if args.prompt_indices is None or len(args.prompt_indices) != 1:
            ap.error("--ip2p-instruction requires exactly one --prompt-indices value")
    if args.require_new_arm and args.metrics_only:
        ap.error("--require-new-arm cannot be used with --metrics-only")
    if args.defended is not None and not args.defended.is_dir():
        ap.error(f"--defenses-dir 不是目錄：{args.defended}")

    args.out.mkdir(parents=True, exist_ok=True)
    items, edits = load_items(args.data)
    # 先依 --images 篩選再解析防禦圖，子集目錄因此可以執行。
    if args.images:
        keep = set(args.images)
        items = [d for d in items if d["name"] in keep]
    if args.defended is not None:
        apply_defended(items, args.defended)
    if not items:
        raise SystemExit("沒有符合的影像")
    if args.instruction_set:
        if args.instruction_set not in edits:
            raise SystemExit(
                f"edits 下沒有 `{args.instruction_set}`，可用的鍵為 {sorted(edits)}")
        for s_ in args.scenarios:
            edits[s_] = edits[args.instruction_set]
    missing = [s for s in args.scenarios if s not in edits]
    if missing:
        raise SystemExit(f"prompts.yaml 的 edits 缺少這些場景的指令：{missing}")
    if args.prompt_indices is not None:
        if (len(set(args.prompt_indices)) != len(args.prompt_indices)
                or any(i < 0 or i >= len(edits["ip2p"])
                       for i in args.prompt_indices)):
            ap.error("--prompt-indices must be distinct valid IP2P indices")
        if args.ip2p_instruction is not None:
            edits["ip2p"][args.prompt_indices[0]] = args.ip2p_instruction
    print(f"{len(items)} 張影像，場景 {args.scenarios}")

    # 既有列以 arm（場景＋後綴）為鍵保留，只替換本次執行的 arm。
    csv_path = args.out / "preflight.csv"
    keep_rows = []
    touched = {s_ + args.suffix for s_ in args.scenarios}
    if args.require_new_arm:
        existing_dirs = [str(args.out / arm) for arm in touched
                         if (args.out / arm).exists()]
        if existing_dirs:
            ap.error(f"Arm already exists; choose a new --suffix: {existing_dirs}")
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        if args.require_new_arm and any(
                r.get("arm", r["scenario"]) in touched for r in existing_rows):
            ap.error("Arm already exists in CSV; choose a new --suffix")
        keep_rows = [r for r in existing_rows
                     if r.get("arm", r["scenario"]) not in touched]
        if keep_rows:
            print(f"保留 {len(keep_rows)} 列既有讀數"
                  f"（場景 {sorted({r['scenario'] for r in keep_rows})}）")

    rows = []
    for scenario in args.scenarios:
        instructions = edits[scenario]
        d = args.out / (scenario + args.suffix)
        d.mkdir(parents=True, exist_ok=True)

        model = IP2P_MODEL if scenario == "ip2p" else INPAINT_MODEL
        guidance = (args.s_t if scenario == "ip2p" else INPAINT_GUIDANCE)
        seed = (args.seed if scenario == "ip2p" and args.seed is not None
                else EDIT_SEED)
        steps = EDIT_STEPS
        if args.metrics_only:
            victim = None
            device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
        elif scenario == "ip2p":
            victim = IP2PWrapper(dtype=torch.float32)
            device = victim.device
        else:
            victim = SDInpaintWrapper(model, dtype=torch.float32)
            device = victim.device
        suite = MetricSuite(device=device)
        print(f"\n=== {scenario}（{model}）===", flush=True)

        for item in items:
            x01 = load_image_tensor(item["path"], device, size=RESOLUTION)
            save_image(x01, d / f"{item['name']}__orig.png")
            mask = None
            if scenario == "inpaint":
                if item["mask"] is None:
                    raise SystemExit(f"{item['name']} 缺少 inpainting 遮罩")
                mask = load_image_tensor(item["mask"], device,
                                         size=RESOLUTION)[:, :1]
                mask = (mask >= 0.5).to(x01.dtype)

            for pi, template in enumerate(instructions):
                if args.prompt_indices is not None and pi not in args.prompt_indices:
                    continue
                prompt = template.format(content=item["content"])
                out_png = d / f"{item['name']}__p{pi}.png"

                if args.metrics_only:
                    if not out_png.exists():
                        raise SystemExit(
                            f"--metrics-only 需要既有的 {out_png}，但它不存在")
                    edit = load_image_tensor(out_png, device, size=RESOLUTION)
                elif scenario == "ip2p":
                    with torch.no_grad():
                        edit = victim.edit(x01, prompt, seed=seed,
                                           steps=steps, s_t=args.s_t,
                                           s_i=args.s_i,
                                           negative_prompt=args.negative_prompt or None)
                elif args.sampler == "official":
                    with torch.no_grad():
                        gen = torch.Generator(device=device).manual_seed(
                            int(EDIT_SEED))
                        raw = victim.pipe(
                            prompt=prompt, image=x01, mask_image=mask,
                            strength=1.0, num_inference_steps=steps,
                            guidance_scale=guidance, generator=gen,
                            output_type="pt").images
                    # 同時保存 pipeline 原始輸出與合成回遮罩外原圖的結果。
                    save_image(raw, d / f"{item['name']}__p{pi}__raw.png")
                    edit = mask * raw + (1.0 - mask) * x01
                else:
                    with torch.no_grad():
                        emb = victim.encode_text(prompt)
                        emb_u = victim.uncond_prompt()
                        noise = victim.sample_edit_noise(
                            victim.encode_image(x01), seed=EDIT_SEED)
                        # `inpaint` 自行組成 9 通道輸入，不經 `conditioning_for`。
                        edit = victim.inpaint(
                            x01, mask, emb, noise, steps,
                            guidance_scale=guidance, emb_uncond=emb_u)
                        # 解碼後合成一次，使遮罩外逐位元等於原圖。
                        edit = mask * edit + (1.0 - mask) * x01

                tag = f"p{pi}"
                if not args.metrics_only:
                    save_image(edit, out_png)

                pair = suite.pairwise(x01, edit)
                sem_e = suite.semantic(edit, prompt)
                sem_o = suite.semantic(x01, prompt)
                row = {
                    "scenario": scenario, "image": item["name"],
                    "class": item["class"], "prompt_index": pi,
                    "prompt": prompt, "victim": model,
                    "seed": seed, "steps": steps, "guidance": guidance,
                    "arm": scenario + args.suffix,
                    "instruction_set": args.instruction_set or scenario,
                    "data": str(args.data).replace("\\", "/"),
                    "defense": args.defended.name if args.defended else "",
                    "input_png": item["path"].as_posix(),
                    "sampler": (args.sampler if scenario == "inpaint"
                                else "ip2p_pipeline"),
                    "s_t": args.s_t if scenario == "ip2p" else "",
                    "s_i": args.s_i if scenario == "ip2p" else "",
                    "instruction": prompt,
                    "negative_prompt": (args.negative_prompt
                                        if scenario == "ip2p" else ""),
                    "output_png": out_png.as_posix(),
                    "clip_edit": round(sem_e["clip"], 4),
                    "clip_orig": round(sem_o["clip"], 4),
                    "clip_gain": round(sem_e["clip"] - sem_o["clip"], 4),
                    "lpips": round(float(pair["lpips"]), 4),
                    "psnr": round(float(pair["psnr"]), 3),
                    **(identity_row(x01, edit) if args.identity else {}),
                }
                rows.append(row)
                print(f"  {item['name']} {tag}: clip_gain "
                      f"{row['clip_gain']:+.4f} lpips {row['lpips']}"
                      + (f" id {row['id_orig']}" if args.identity else ""),
                      flush=True)
                write_csv(csv_path, keep_rows + rows)
        del victim, suite
        torch.cuda.empty_cache()

    print(f"\n{len(rows)} 格 -> {csv_path}")
    if args.identity:
        same = [r for r in rows
                if r["id_orig"] != "" and float(r["id_orig"]) >= 0.55]
        print(f"FaceNet ≥ 0.55 的格數：{len(same)} / {len(rows)}")


if __name__ == "__main__":
    main()
