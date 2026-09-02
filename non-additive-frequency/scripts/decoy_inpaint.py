"""語意誘餌 v2：**文字指出主體，用 inpainting 把誘餌畫進主體以外的地方。**

與 v1（`scripts/semantic_decoy.py` ＋ `scripts/decoy_masked.py`）的差別
────────────────────────────────────────────────────────────────────
v1 對**整張圖**下編輯指令，再用手標的矩形把主體貼回去。兩個問題：

1. **邊界生硬。** 誘餌是對整張圖畫出來的，矩形攔腰切掉，剩下那半塊沒有與
   周圍接上，看得出來是貼片（`runs/ip2p_decoy_masked/` 的影像）。
2. **會溢出到主體。** 「在背景加一把粉紅色的傘」讓 IP2P 順手把花盆也變粉紅，
   等於防禦方先把攻擊做完了。矩形擋得住，但那是事後補救。

v2 換成 **inpainting**：遮罩由 CLIPSeg 用一句文字指出主體
（`src/defense/subject_mask.py`），重畫區是**主體以外**，模型只在那裡作畫、
並且是**看著周圍的內容**畫的，所以接得上。主體區域模型根本不會動到，
再加一次硬合成把它釘成逐位元相同。

遮罩的文字就是 `data/decoy_catalogue.yaml` 的 `objects` 登記的主體名稱——
**威脅模型的前提是防護對象已知**，攻擊指令不是。給多個名詞時取逐像素聯集。

三個必須照實記的事
────────────────────────────────────────────────────────────────────
1. **inpainting 的權重與攻擊模型不同**（`runwayml/stable-diffusion-inpainting`
   對 `timbrooks/instruct-pix2pix`）。防禦方用哪個生成模型是它自己的選擇，
   與白盒假設無關；但兩者不是同一個模型這件事要進報表。
2. **誘餌的提示詞是描述式的**，不是指令式的——inpainting 吃的是「這裡應該
   長什麼樣」。v1 的指令式目錄不能直接搬過來用。
3. **硬合成之後仍逐列驗證主體逐位元不變**，不是只驗一次。

用法：

    python scripts/decoy_inpaint.py --out runs/ip2p_decoy_inpaint/all \\
        --images task_attr_mod_color_11699 task_attr_mod_color_6205 \\
        --groups inpaint_object
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torchvision.utils as vutils  # noqa: E402
import yaml  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from semantic_decoy import RESERVED_KEYS, fill  # noqa: E402
from src.defense.subject_mask import (  # noqa: E402
    CLIPSEG_REPO, DILATE, FEATHER, THRESHOLD, mask_stats, subject_mask,
)
from src.metrics.standard import standard_row  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.ip2p import (  # noqa: E402
    IP2P_IMAGE_GUIDANCE, IP2P_SEED, IP2P_STEPS, IP2P_TEXT_GUIDANCE, IP2PWrapper,
)
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
INPAINT_REPO = "runwayml/stable-diffusion-inpainting"
# 防禦端的取樣種子，與攻擊端分開（理由見 `semantic_decoy` 的 docstring）。
DECOY_SEED = 20260902
# inpainting 的三個設定，**本專案指定**，故逐列進 CSV。
INPAINT_STEPS = 50
INPAINT_GUIDANCE = 7.5


def load_inpainter(device):
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_REPO, torch_dtype=torch.float32, safety_checker=None,
        requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def to_pil(x01: torch.Tensor):
    from PIL import Image

    arr = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
    return Image.fromarray(arr.astype("uint8"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--catalogue", type=Path,
                    default=Path("data/decoy_catalogue.yaml"))
    ap.add_argument("--groups", nargs="+", default=["inpaint_object"])
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decoy-seed", type=int, default=DECOY_SEED)
    ap.add_argument("--inpaint-steps", type=int, default=INPAINT_STEPS)
    ap.add_argument("--inpaint-guidance", type=float, default=INPAINT_GUIDANCE)
    ap.add_argument("--mask-threshold", type=float, default=THRESHOLD)
    ap.add_argument("--mask-dilate", type=int, default=DILATE)
    ap.add_argument("--mask-feather", type=int, default=FEATHER)
    args = ap.parse_args()

    spec = yaml.safe_load(args.catalogue.read_text(encoding="utf-8"))
    available = [k for k in spec if k not in RESERVED_KEYS]
    unknown = set(args.groups) - set(available)
    if unknown:
        raise SystemExit(
            f"目錄裡沒有這些組：{sorted(unknown)}；有的是 {sorted(available)}")
    decoys = [{"group": g, "index": i, "prompt": t,
               "condition": f"decoy_{g}_{i}"}
              for g in args.groups for i, t in enumerate(spec[g])]
    subjects = dict(spec.get("objects") or {})

    dataset = {d["name"]: d for d in load_dataset(args.data)}
    missing = [n for n in args.images if n not in dataset]
    if missing:
        raise SystemExit(f"資料集裡沒有這些影像：{missing}")

    sd = IP2PWrapper(dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    inpainter = load_inpainter(sd.device)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.images:
        item = dataset[name]
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        texts = subjects.get(name)
        if not texts:
            print(f"[skip] {name}：objects 對照表裡沒有主體名稱", flush=True)
            continue
        if isinstance(texts, str):
            texts = [texts]
        m = subject_mask(x01, texts, threshold=args.mask_threshold,
                         dilate=args.mask_dilate, feather=args.mask_feather)
        stats = mask_stats(m)
        vutils.save_image(x01.clamp(0, 1), args.out / f"{name}__orig.png")
        vutils.save_image(m.repeat(1, 3, 1, 1),
                          args.out / f"{name}__subject_mask.png")
        e_orig = sd.edit(x01.clamp(0, 1), item["prompt"], seed=IP2P_SEED,
                         steps=IP2P_STEPS, s_t=IP2P_TEXT_GUIDANCE,
                         s_i=IP2P_IMAGE_GUIDANCE)
        print(f"{name[15:]:26s}主體遮罩 {texts} {stats}", flush=True)

        for d in decoys:
            t0 = time.time()
            cond = d["condition"]
            prompt = fill(d["prompt"], name, {name: " and ".join(texts)})
            if prompt is None:
                print(f"[skip] {name}／{cond}：模板要 {{object}} 而沒有主體名稱",
                      flush=True)
                continue
            gen = torch.Generator(device=sd.device).manual_seed(args.decoy_seed)
            painted = inpainter(
                prompt=prompt, image=to_pil(x01),
                mask_image=to_pil((1.0 - m).repeat(1, 3, 1, 1)),
                num_inference_steps=args.inpaint_steps,
                guidance_scale=args.inpaint_guidance,
                generator=gen, output_type="pt",
            ).images
            painted = painted.to(sd.device).float().clamp(0, 1)
            # **硬合成**：inpainting 的管線會重新編解碼整張圖，未遮罩區也會
            # 有微小差異。合成把主體釘成逐位元相同。
            x_def = (m * x01 + (1.0 - m) * painted).clamp(0, 1)
            inside = m >= 1.0
            gap = float(((x_def - x01) * inside).abs().max()) if inside.any() else 0.0
            if gap != 0.0:
                raise RuntimeError(
                    f"{name}/{cond}：主體核心與原圖差 {gap}，應恰為 0。")

            e_def = sd.edit(x_def, item["prompt"], seed=IP2P_SEED,
                            steps=IP2P_STEPS, s_t=IP2P_TEXT_GUIDANCE,
                            s_i=IP2P_IMAGE_GUIDANCE)
            for sub, img in (("def", x_def), ("edit_orig", e_orig),
                             ("edit_def", e_def)):
                vutils.save_image(img.clamp(0, 1),
                                  args.out / f"{name}__{cond}__{sub}.png")
            fid = suite.pairwise(x01, x_def)
            eff = suite.pairwise(e_orig, e_def)
            sim = suite.image_similarity(e_orig, e_def)
            rows.append({
                "image": name, "condition": cond,
                "attacker": "instruct-pix2pix",
                "instruction": item["prompt"], "task": item.get("class", ""),
                "decoy_group": d["group"], "decoy_index": d["index"],
                "decoy_instruction": prompt, "decoy_template": d["prompt"],
                # 防禦端的設定。**全部本專案指定**，故是欄位。
                "decoy_generator": INPAINT_REPO,
                "decoy_seed": args.decoy_seed,
                "inpaint_steps": args.inpaint_steps,
                "inpaint_guidance": args.inpaint_guidance,
                # 遮罩的來源與四個旋鈕。
                "mask_model": CLIPSEG_REPO,
                "mask_text": " | ".join(texts),
                "mask_threshold": args.mask_threshold,
                "mask_dilate": args.mask_dilate,
                "mask_feather": args.mask_feather,
                **stats,
                "edit_steps": IP2P_STEPS, "s_t": IP2P_TEXT_GUIDANCE,
                "s_i": IP2P_IMAGE_GUIDANCE, "edit_seed": IP2P_SEED,
                "radius": "",
                **standard_row("fid_", fid),
                **standard_row("edit_", eff),
                "fid_deltaE00": round(fid["deltaE00"], 4),
                "edit_clip_sim": round(sim["clip"], 5),
                "edit_siglip_sim": round(sim["siglip"], 5),
                "total_seconds": round(time.time() - t0, 1),
            })
            write_csv(args.out / "results.csv", rows)
            print(f"{name[15:]:26s}{cond:26s}{prompt[:34]:36s}"
                  f"失真={fid['dists']:.4f} 位移={eff['lpips']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n表：{args.out / 'results.csv'}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
