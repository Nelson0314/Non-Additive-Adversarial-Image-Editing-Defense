"""FLUX.1-Kontext-dev 在主表資料集上的編輯——跨編輯器遷移的第三支路線。

與 `edit_sdedit_preview.py` 同一個評測外殼(同一批影像、同一組 ip2p 指令、
同一個 `identity_row` 身分讀數、同一種 CSV 輸出),核心呼叫換成
`FluxKontextPipeline`——FLUX 是流匹配 transformer,不是 UNet+DDIM,不走
`SDWrapper` 那條路徑,`sdedit()` 的 strength/加噪起點對它沒有意義。
`FluxKontextPipeline.__call__` 本身就沒有 `strength` 參數。

硬體:transformer 12B,bf16 全精度單這一個元件就約 22.2 GiB,3090 的 24GB
放不下(還要留活化與 CUDA 開銷的空間)。這裡對 transformer 與 T5 文字編碼器
都做 4-bit(NF4)量化,兩者量化後的權重量級落在 10–14GB,3090 上跑得動。

授權:`black-forest-labs/FLUX.1-Kontext-dev` 是 gated repo,需要
`~/.hf_token` 且該帳號已在 huggingface.co 上點過同意授權——這件事只有
使用者能做,腳本這裡假設它已完成,沒完成會在載入階段直接看到
`GatedRepoError`。

解析度:`FluxKontextPipeline` 預設會把輸入與輸出放大到約 1024×1024
(`max_area`)。這裡明給 `height=width=512`,跳過那個自動放大,維持跟
`edit_sdedit_preview.py` 同一個解析度才能互相比較。

用法(遠端,需要一張卡,首次會下載約 24GB 權重)
    HF_HOME=/var/cache/huggingface CUDA_VISIBLE_DEVICES=<卡> \\
        python code/edit_flux_preview.py --out images/flux_preview
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from edit_preflight import identity_row, load_items  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
RESOLUTION = 512
EDIT_SEED = 20260812
EDIT_STEPS = 28          # FluxKontextPipeline 的預設步數,沒有跨模型的協定可對齊
EDIT_GUIDANCE = 3.5      # FluxKontextPipeline 的預設 guidance_scale


def to_pil(x01: torch.Tensor) -> Image.Image:
    arr = (x01[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def to_tensor(img: Image.Image, device) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def load_pipeline():
    from diffusers import BitsAndBytesConfig, FluxKontextPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig as BnbConfigT5, T5EncoderModel

    quant4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", quantization_config=quant4,
        torch_dtype=torch.bfloat16)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder_2",
        quantization_config=BnbConfigT5(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16)
    pipe = FluxKontextPipeline.from_pretrained(
        MODEL_NAME, transformer=transformer, text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--out", type=Path, default=paths.IMAGES / "flux_preview")
    ap.add_argument("--images", nargs="+", default=None,
                    help="預設每個類別取第一張")
    ap.add_argument("--instruction-indices", nargs="+", type=int, default=[0, 1],
                    help="`prompts.yaml` 的 edits.ip2p 底下要用哪幾條,預設前兩條")
    ap.add_argument("--steps", type=int, default=EDIT_STEPS)
    ap.add_argument("--guidance", type=float, default=EDIT_GUIDANCE)
    args = ap.parse_args()

    items, edits = load_items(args.data)
    instructions = edits["ip2p"]
    by_class = {}
    for item in items:
        by_class.setdefault(item["class"], item)
    targets = ([by_class[c] for c in sorted(by_class)] if args.images is None
              else [next(i for i in items if i["name"] == n) for n in args.images])

    print(f"影像 {[t['name'] for t in targets]}，指令索引 "
          f"{args.instruction_indices}，steps={args.steps} guidance={args.guidance}",
          flush=True)

    t_load = time.time()
    pipe = load_pipeline()
    print(f"[LOAD] {time.time() - t_load:.1f}s", flush=True)
    if torch.cuda.is_available():
        print(f"[VRAM] allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in targets:
        x01 = load_image_tensor(item["path"], device, size=RESOLUTION)
        x_pil = to_pil(x01)
        for pi in args.instruction_indices:
            prompt = instructions[pi]
            t0 = time.time()
            gen = torch.Generator(device=device).manual_seed(EDIT_SEED)
            out = pipe(image=x_pil, prompt=prompt, height=RESOLUTION, width=RESOLUTION,
                      guidance_scale=args.guidance, num_inference_steps=args.steps,
                      generator=gen).images[0]
            dt = time.time() - t0
            edit = to_tensor(out, device)
            idr = identity_row(x01, edit)
            out_png = args.out / f"{item['name']}__p{pi}.png"
            save_image(edit, out_png)
            row = {
                "image": item["name"], "prompt_index": pi, "prompt": prompt,
                "model": MODEL_NAME, "guidance_scale": args.guidance,
                "steps": args.steps, "seed": EDIT_SEED, "seconds": round(dt, 1),
                "png": out_png.as_posix(), **idr,
            }
            rows.append(row)
            print(f"  {item['name']:10s} p{pi}  id_orig={idr['id_orig']}  "
                  f"arcface_orig={idr['arcface_orig']}  ({dt:.1f}s)", flush=True)
            if torch.cuda.is_available():
                print(f"    [VRAM peak] {torch.cuda.max_memory_allocated()/1e9:.2f}GB",
                      flush=True)

    out_csv = paths.RESULTS / "flux_preview.csv"
    write_csv(out_csv, rows)
    print(f"\n完成：{len(rows)} 格 -> {out_csv}")


if __name__ == "__main__":
    main()
