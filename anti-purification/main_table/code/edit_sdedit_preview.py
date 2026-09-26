"""SDEdit（stock SD img2img）在主表資料集上的 strength 預覽——第三個編輯器的
前置檢查，跑全表之前先看未防禦編輯有沒有把人換掉。

為什麼要有這一支
────────────────────────────────────────────────────────────────────
主表現有兩個場景（ip2p、inpaint）都不是 SD img2img——`edit_preflight.py`
docstring 記過 SD v1.4 img2img 在 strength 0.6／0.8 下把人換掉
（FaceNet 對原圖中位數 0.120／0.029，同一人門檻 0.55）。跨編輯器遷移要新增
SDEdit 當第三個編輯器，strength 是這裡才出現的新自由度，**在跑全表 736 格
之前**要先在小樣本上看過未防禦編輯的圖與身分讀數，再決定用哪個值——這是
使用者與上一位接手者都定下的規矩，本腳本只做這一步，不跑全表。

只跑未防禦編輯（沒有防禦圖介入），因為要看的是「這個編輯器本身在這個
strength 下站不站得住」，防禦的影響是下一步的事。

用法（遠端，需要一張卡）
    HF_HOME=/var/cache/huggingface CUDA_VISIBLE_DEVICES=<卡> \\
        python code/edit_sdedit_preview.py --out images/sdedit_preview
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402

from edit_preflight import identity_row, load_items  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

#: `runwayml/stable-diffusion-v1-5`：SD 1.x 家族的標準基準權重。專案內另一條
#: 舊線（`scripts/baseline_run.py`）用過 `CompVis/stable-diffusion-v1-4`，
#: 兩者 UNet 架構相同，這裡選 1.5 是因為它是目前 SD 1.x 實驗的通用預設，
#: 不是因為 1.4 不能用。
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
RESOLUTION = 512
EDIT_SEED = 20260812   # 與 ip2p/inpaint 兩個場景共用同一顆種子
EDIT_STEPS = 50        # 與 edit_preflight.py 的兩個場景對齊
EDIT_GUIDANCE = 7.5    # 見 sd.py `_eps_cfg`：w=1 時 SDEdit 退化成加噪再去噪
#: 要掃的 strength。0.55／0.8 已由 `edit_preflight.py` 記過的實測結果排除
#: 在「兩者都要」的候選之外（0.55 不服從 prompt、0.8 換人），這裡改掃中段。
STRENGTHS = (0.3, 0.4, 0.5, 0.6)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--out", type=Path, default=paths.IMAGES / "sdedit_preview")
    ap.add_argument("--images", nargs="+", default=None,
                    help="預設每個類別取第一張")
    ap.add_argument("--instruction-indices", nargs="+", type=int, default=[0, 1],
                    help="`prompts.yaml` 的 edits.ip2p 底下要用哪幾條，預設前兩條")
    ap.add_argument("--strengths", nargs="+", type=float, default=list(STRENGTHS))
    args = ap.parse_args()

    items, edits = load_items(args.data)
    instructions = edits["ip2p"]
    by_class = {}
    for item in items:
        by_class.setdefault(item["class"], item)   # 每類第一張
    targets = ([by_class[c] for c in sorted(by_class)] if args.images is None
              else [next(i for i in items if i["name"] == n) for n in args.images])

    print(f"影像 {[t['name'] for t in targets]}，指令索引 "
          f"{args.instruction_indices}，strength {args.strengths}", flush=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in targets:
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        for pi in args.instruction_indices:
            prompt = instructions[pi]
            emb, emb_u = sd.encode_text(prompt), sd.uncond_prompt()
            noise = sd.sample_edit_noise(sd.encode_image(x01), seed=EDIT_SEED)
            for strength in args.strengths:
                t0 = time.time()
                with torch.no_grad():
                    edit = sd.sdedit(x01, emb, noise, EDIT_STEPS,
                                     strength=strength,
                                     guidance_scale=EDIT_GUIDANCE,
                                     emb_uncond=emb_u)
                idr = identity_row(x01, edit)
                out_png = args.out / f"{item['name']}__p{pi}__s{strength}.png"
                save_image(edit, out_png)
                row = {
                    "image": item["name"], "prompt_index": pi, "prompt": prompt,
                    "strength": strength, "model": MODEL_NAME,
                    "guidance_scale": EDIT_GUIDANCE, "steps": EDIT_STEPS,
                    "seed": EDIT_SEED, "png": out_png.as_posix(), **idr,
                }
                rows.append(row)
                print(f"  {item['name']:10s} p{pi} strength={strength}  "
                      f"id_orig={idr['id_orig']}  arcface_orig={idr['arcface_orig']}"
                      f"  ({time.time() - t0:.1f}s)", flush=True)

    out_csv = paths.RESULTS / "sdedit_preview.csv"
    write_csv(out_csv, rows)
    print(f"\n完成：{len(rows)} 格 -> {out_csv}")


if __name__ == "__main__":
    main()
