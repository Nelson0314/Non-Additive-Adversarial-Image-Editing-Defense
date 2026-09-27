"""SDEdit（stock SD img2img）在主表資料集上的編輯——跨編輯器遷移的第三個編輯器。

協定：strength 0.5、guidance 7.5、50 步，不加任何遮罩
────────────────────────────────────────────────────────────────────
這組值取自 PhotoGuard 官方 repo 的 `demo_simple_attack_img2img.ipynb` cell 12
（`pipe_img2img(...)` 那次呼叫），逐行稽核見 `docs/reference/
AUDIT_PROMPTFLARE_PHOTOGUARD.md` §5.1「編輯（評估）階段設定」：
strength=0.5、guidance_scale=7.5、num_inference_steps=50。同一份稽核也記了
一個不一致：論文附錄 Table 8 給的是 100 步，notebook 實際跑的是 50 步——這裡
採 notebook 的值，因為那是官方**實際執行**用來產生 demo 結果的設定，不是
論文表格；100 步的版本也可能有效，只是沒有一份文獻同時給出「0.5 配 100 步」
這個特定組合。inpainting demo cell 用的是另一組（strength 0.7、100 步），
不適用於這裡的 img2img/SDEdit 場景。

採用這一組理由是**它是文獻自己用來展示「SDEdit 可以怎麼編輯一張圖」的設定，
不是本專案為了通過某個身分門檻反推出來的**。

**不加遮罩，使用者 2026-09-27 裁定。** 在 4 張影像 × 4 個 strength 的預覽中
（`results/sdedit_preview.csv`），strength 0.3–0.6 的 id_orig 全部低於
`edit_preflight.py` 記的 0.55 同一人門檻。`SDWrapper.sdedit()` 的 `keep01`
遮罩參數可以緩解這件事，但 PhotoGuard 等文獻本身展示 SDEdit 時也沒有這樣做
——對這個威脅模型，身分被大幅改動是 SDEdit 這一類攻擊本來就有的性質，不是
要修的缺陷。要不要遮罩因此不是本腳本的決定，本腳本沿用文獻沒有遮罩的做法。

跨 SD 1.x／2.x 的「統一」：`SDWrapper` 只讀 `self.pipe` 的 unet／
text_encoder／tokenizer，不寫死任何維度，同一個類別餵不同的 `--model` 即可，
不需要另一個 wrapper（FLUX 是流匹配架構，不共用這條路徑，見
`edit_flux_preview.py`）。

用法（遠端，需要一張卡）
    HF_HOME=/var/cache/huggingface CUDA_VISIBLE_DEVICES=<卡> \\
        python code/edit_sdedit_preview.py --model stabilityai/stable-diffusion-2-1 \\
            --out images/sdedit_preview_sd21
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

RESOLUTION = 512
EDIT_SEED = 20260812   # 與 ip2p/inpaint 兩個場景共用同一顆種子
EDIT_STEPS = 50        # PhotoGuard demo cell 1，同時與本專案兩個場景對齊
EDIT_GUIDANCE = 7.5    # 同上；見 sd.py `_eps_cfg`：w=1 時 SDEdit 會退化成加噪再去噪
EDIT_STRENGTH = 0.5    # 同上，見上方協定說明


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--out", type=Path, default=paths.IMAGES / "sdedit_preview")
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5",
                    help="任何 SD 1.x／2.x 的 diffusers checkpoint 名稱；"
                         "SDWrapper 不寫死維度，換這個參數就是換模型家族")
    ap.add_argument("--images", nargs="+", default=None,
                    help="預設每個類別取第一張")
    ap.add_argument("--instruction-indices", nargs="+", type=int, default=[0, 1],
                    help="`prompts.yaml` 的 edits.ip2p 底下要用哪幾條，預設前兩條")
    ap.add_argument("--strengths", nargs="+", type=float, default=[EDIT_STRENGTH],
                    help="預設用已定案的協定值 0.5；只在需要重新檢視時才覆寫")
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

    sd = SDWrapper(args.model, dtype=torch.float32)
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
                    "strength": strength, "model": args.model,
                    "guidance_scale": EDIT_GUIDANCE, "steps": EDIT_STEPS,
                    "seed": EDIT_SEED, "png": out_png.as_posix(), **idr,
                }
                rows.append(row)
                print(f"  {item['name']:10s} p{pi} strength={strength}  "
                      f"id_orig={idr['id_orig']}  arcface_orig={idr['arcface_orig']}"
                      f"  ({time.time() - t0:.1f}s)", flush=True)

    model_tag = args.model.rsplit("/", 1)[-1].replace(".", "_")
    out_csv = paths.RESULTS / f"sdedit_preview_{model_tag}.csv"
    write_csv(out_csv, rows)
    print(f"\n完成：{len(rows)} 格 -> {out_csv}")


if __name__ == "__main__":
    main()
