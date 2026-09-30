"""FLUX.1-Kontext-dev 在主表資料集上的編輯——跨編輯器遷移的第三支路線。

與 `sweep_sdedit_parameters` 同一個評測外殼(同一批影像、同一組 ip2p 指令、
同一個 `identity_row` 身分讀數、同一種 CSV 輸出),核心呼叫換成
`FluxKontextPipeline`——FLUX 是流匹配 transformer,不是 UNet+DDIM,不走
`SDWrapper` 那條路徑,`sdedit()` 的 strength/加噪起點對它沒有意義。
`FluxKontextPipeline.__call__` 本身就沒有 `strength` 參數。

硬體:transformer 12B,bf16 全精度單這一個元件就約 22.2 GiB,3090 的 24GB
放不下(還要留活化與 CUDA 開銷的空間)。這裡對 transformer 與 T5 文字編碼器
都做 4-bit(NF4)量化。**實測**(2026-09-27,RTX 3090,4 格):載入
956.8s(多數是首次下載約 24GB 權重,權重快取後應該快得多，未重跑驗證)、
載入完 VRAM 11.43GB、單格生成後峰值 14.13GB，28 步在 1024×1024 下單格
185 秒。三者都在 24GB 卡上跑得動，沒有 OOM。

授權:`black-forest-labs/FLUX.1-Kontext-dev` 是 gated repo,需要
`~/.hf_token` 且該帳號已在 huggingface.co 上點過同意授權——這件事只有
使用者能做。2026-09-27 已完成。

解析度:`FluxKontextPipeline` **不接受**縮到 512×512。程式碼裡明給了
`height=width=512`，但實測 log 印出「Generation height and width have
been adjusted to 1024 and 1024 to fit the model requirements」——參數
被套件自己蓋掉了。也就是說 FLUX 這一臂**沒有**跟 `sweep_sdedit_parameters`
同一個解析度，兩者的位移／保真讀數不能直接並排比較解析度效應，這件事
要在報告裡標明，不是程式錯誤。

全表:`--arm undefended` 跑分母(8 影像 × 4 指令 = 32 格),`--arm <條件>` 跑該
條件的防禦圖(同樣 32 格,`--defenses-dir` 預設為 `artifacts/defenses/<條件>/`)。12 條件 + 分母 = 13 個 arm、416 格,單格 83–185 秒（視卡上其他人
負載），每個 arm 只載入一次模型。CSV 逐格 append 並 flush，中斷重跑會跳過
已經完成且協定摘要、輸入雜湊與 PNG 均一致的格，不必整個 arm 重來。

用法(遠端,需要一張卡,首次會下載約 24GB 權重)
    HF_HOME=/var/cache/huggingface CUDA_VISIBLE_DEVICES=<卡> \\
        python -m immunization_baseline.cli.run_flux_edits --arm undefended
    HF_HOME=/var/cache/huggingface CUDA_VISIBLE_DEVICES=<卡> \\
        python -m immunization_baseline.cli.run_flux_edits --arm mist
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from immunization_baseline import layout  # noqa: E402

import torch  # noqa: E402

from immunization_core.pipelines.editing import defended_image, identity_row, load_items  # noqa: E402
from immunization_core.artifacts.images import save_image  # noqa: E402
from immunization_core.io import load_image_tensor  # noqa: E402
from immunization_baseline.editors import to_pil, to_tensor  # noqa: E402
from immunization_baseline.resume_state import file_digest, load_resume_rows, protocol_digest  # noqa: E402

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
RESOLUTION = 512
EDIT_SEED = 20260812
EDIT_STEPS = 28          # FluxKontextPipeline 的預設步數,沒有跨模型的協定可對齊
EDIT_GUIDANCE = 3.5      # FluxKontextPipeline 的預設 guidance_scale


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


FIELDS = ["arm", "image", "prompt_index", "prompt", "model", "guidance_scale",
         "true_cfg_scale", "negative_prompt", "steps", "seed", "seconds", "png",
         "id_orig", "arcface_orig", "face_found_orig", "face_found_edit",
         "protocol_id", "input_png", "input_sha256"]


def load_done(csv_path: Path, protocol_id: str) -> set:
    rows = load_resume_rows(csv_path, FIELDS, protocol_id, ("arm", "image", "prompt_index"))
    return {(r["image"], r["prompt_index"]) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", dest="data", type=Path, default=layout.PORTRAITS)
    ap.add_argument("--arm", required=True,
                    help="`undefended` 跑原圖，其餘為條件名，防禦圖取自 --defenses-root/<條件>/")
    ap.add_argument("--defenses-root", dest="defenses", type=Path, default=layout.DEFENSES,
                    help="條件防禦圖的根目錄")
    ap.add_argument("--defenses-dir", dest="defended", type=Path, default=None,
                    help="覆寫單一條件的防禦圖目錄，預設為 --defenses-root/<arm>")
    ap.add_argument("--output-dir", dest="out", type=Path, default=None,
                    help="預設 artifacts/flux/edits/<arm>/")
    ap.add_argument("--output-csv", dest="out_csv", type=Path, default=None,
                    help="不同協定須使用獨立 CSV 與 --output-dir")
    ap.add_argument("--images", nargs="+", default=None,
                    help="預設全部 8 張")
    ap.add_argument("--instruction-indices", nargs="+", type=int, default=[0, 1, 2, 3],
                    help="`prompts.yaml` 的 edits.ip2p 底下要用哪幾條，預設全部 4 條")
    ap.add_argument("--steps", type=int, default=EDIT_STEPS)
    ap.add_argument("--guidance", type=float, default=EDIT_GUIDANCE)
    ap.add_argument("--true-cfg-scale", type=float, default=1.0,
                    help="真正的雙分支 CFG。FluxKontextPipeline 只在 >1 且有給"
                         "negative_prompt 時才會啟用，否則沿用蒸餾過的 guidance_scale")
    ap.add_argument("--negative-prompt", default=None,
                    help="配合 --true-cfg-scale>1 使用，往「不要偏離原圖」推，"
                         "不要提被要求改的那個部位，否則會跟正面指令互相抵銷")
    args = ap.parse_args()

    out_dir = args.out or (layout.FLUX_EDITS / args.arm)
    out_csv = args.out_csv or layout.RESULTS / "flux" / f"edits_{args.arm}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    items, edits = load_items(args.data)
    instructions = edits["ip2p"]
    if (len(set(args.instruction_indices)) != len(args.instruction_indices)
            or any(pi < 0 or pi >= len(instructions) for pi in args.instruction_indices)):
        ap.error("--instruction-indices 必須是不重複的有效索引")
    if args.images and len(set(args.images)) != len(args.images):
        ap.error("--images 不得重複")
    targets = (sorted(items, key=lambda i: i["name"]) if args.images is None
              else [next(i for i in items if i["name"] == n) for n in args.images])
    if not targets:
        ap.error("沒有符合的影像")

    if args.arm != "undefended":
        defended_dir = args.defended or args.defenses / args.arm
        for item in targets:
            item["path"] = defended_image(defended_dir, item["name"])

    protocol_id = protocol_digest({
        "editor": "flux-kontext", "model": MODEL_NAME, "arm": args.arm,
        "resolution": RESOLUTION, "seed": EDIT_SEED, "steps": args.steps,
        "guidance": args.guidance, "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt, "precision": "bfloat16-nf4",
        "prompts": instructions, "data": str(args.data.resolve()),
        "defended": str(defended_dir.resolve()) if args.arm != "undefended" else None,
        "output": str(out_dir.resolve()),
    })
    done = load_done(out_csv, protocol_id)
    todo = [(item, pi) for item in targets for pi in args.instruction_indices
           if (item["name"], str(pi)) not in done]
    print(f"arm={args.arm} 影像={[t['name'] for t in targets]} "
          f"指令索引={args.instruction_indices} steps={args.steps} "
          f"guidance={args.guidance} -- 已完成 {len(done)} 格，待做 {len(todo)} 格",
          flush=True)
    if not todo:
        print("全部完成，無需載入模型", flush=True)
        return
    for item, pi in todo:
        png = out_dir / f"{item['name']}__p{pi}.png"
        if png.exists():
            raise FileExistsError(f"未登錄的既有產物：{png}；請使用新的輸出目錄")

    t_load = time.time()
    pipe = load_pipeline()
    print(f"[LOAD] {time.time() - t_load:.1f}s", flush=True)
    if torch.cuda.is_available():
        print(f"[VRAM] allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    new_file = not out_csv.is_file()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        cache = {}
        for item, pi in todo:
            name = item["name"]
            if name not in cache:
                x01 = load_image_tensor(item["path"], device, size=RESOLUTION)
                cache[name] = (x01, to_pil(x01))
            x01, x_pil = cache[name]
            prompt = instructions[pi]
            t0 = time.time()
            gen = torch.Generator(device=device).manual_seed(EDIT_SEED)
            call_kw = dict(image=x_pil, prompt=prompt, height=RESOLUTION, width=RESOLUTION,
                          guidance_scale=args.guidance, num_inference_steps=args.steps,
                          generator=gen)
            if args.negative_prompt:
                call_kw["negative_prompt"] = args.negative_prompt
                call_kw["true_cfg_scale"] = args.true_cfg_scale
            out = pipe(**call_kw).images[0]
            dt = time.time() - t0
            edit = to_tensor(out, device)
            idr = identity_row(x01, edit)
            out_png = out_dir / f"{name}__p{pi}.png"
            save_image(edit, out_png)
            row = {
                "arm": args.arm, "image": name, "prompt_index": pi, "prompt": prompt,
                "model": MODEL_NAME, "guidance_scale": args.guidance,
                "true_cfg_scale": args.true_cfg_scale if args.negative_prompt else "",
                "negative_prompt": args.negative_prompt or "",
                "steps": args.steps, "seed": EDIT_SEED, "seconds": round(dt, 1),
                "png": out_png.as_posix(), **idr,
                "protocol_id": protocol_id, "input_png": item["path"].as_posix(),
                "input_sha256": file_digest(item["path"]),
            }
            writer.writerow(row)
            stream.flush()
            print(f"  {name:10s} p{pi}  id_orig={idr['id_orig']}  "
                  f"arcface_orig={idr['arcface_orig']}  ({dt:.1f}s)", flush=True)

    print(f"\n完成：arm={args.arm} -> {out_csv}")


if __name__ == "__main__":
    main()
