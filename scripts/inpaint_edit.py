"""用既有的全圖防禦圖跑 inpainting 編輯。兩種場景，遮罩自動產生。

修正 2026-08-14 第一版 inpainting 批次的協定錯誤（見 FND-038 的更正）：
那一版把遮罩畫在**背景**、prompt 卻用**描述主體**的 `prompts[0]`，等於叫模型
在背景畫一隻狗、而狗本人在遮罩外原封不動——攻擊本來就不可能成立。

文獻的兩種場景（AdvPaint ICLR 2025 同時測兩者）：

    subject      遮罩＝主體，prompt＝`prompts[0]`（換掉主體：a dog → a cat）
    background   遮罩＝主體以外，prompt＝`prompts[1]`（保住主體、重畫其餘：
                 a dog in the park）

`prompts.yaml` 的檔頭本來就是這樣分工的，第一版用錯了那一欄。

**防禦加在整張圖上，遮罩在編輯時才套用。** 故不需要為 inpainting 重跑攻擊，
直接讀 img2img 批次（`runs/hb5`）存下的 `*__def.png`。

`effect` 只在遮罩內量：`sd.inpaint` 每一步把遮罩外貼回，兩條分支在該處正好
差一個防禦擾動，整張圖算 LPIPS 會把失真算成效果。

用法：
    python scripts/inpaint_edit.py --out runs/ip2/background --scenario background \
        --defended runs/hb5 runs/hb5_pgc --masks data/lo_masks_auto
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDInpaintWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

MODEL_NAME = "runwayml/stable-diffusion-inpainting"
RESOLUTION = 512
INPAINT_STEPS = 50
INPAINT_GUIDANCE = 7.5
INPAINT_SEED = 20260814

# `runs/hb5` 的七個條件與各自的檔名段。A 臂三者帶 `__human`。
TAG = {"phase": "phase__human", "add": "add__human",
       "phase_rand": "phase_rand__human", "apa_weak": "apa_weak",
       "mist": "mist", "dia_r": "dia_r", "photoguard_c": "photoguard_c"}
# prompts.yaml 的哪一欄。0＝換掉主體，1＝保住主體改其餘。
SCENARIO_PROMPT = {"subject": 0, "background": 1}


def load_items(data: Path, masks: Path, scenario: str, images=None) -> list:
    spec = yaml.safe_load((data / "prompts.yaml").read_text(encoding="utf-8"))
    idx = SCENARIO_PROMPT[scenario]
    out = []
    for c in sorted(spec):
        for img in sorted((data / c).glob("*.png")):
            if images and img.stem not in images:
                continue
            m = masks / f"{img.stem}__{scenario}.png"
            if not m.exists():
                raise FileNotFoundError(f"{img.stem} 缺 {scenario} 遮罩：{m}")
            out.append({"name": img.stem, "content": spec[c]["content"],
                        "path": img, "mask": m,
                        "prompt": spec[c]["prompts"][idx]})
    return out


def load_mask(path: Path, device) -> torch.Tensor:
    m = load_image_tensor(path, device, size=RESOLUTION)[:, :1]
    if not torch.isin(m, torch.tensor([0.0, 1.0], device=m.device)).all():
        raise ValueError(f"{path} 不是二值遮罩")
    return m


def find_def(dirs, image, cond):
    name = f"{image}__{TAG[cond]}__def.png"
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def masked_compare(y, x01, mask):
    """遮罩外換成同一張原圖，使比較只落在生成內容上。"""
    return y * mask + x01 * (1.0 - mask)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenario", choices=list(SCENARIO_PROMPT), required=True)
    ap.add_argument("--defended", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--masks", type=Path, default=Path("data/lo_masks_auto"))
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=list(TAG))
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDInpaintWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)
    items = load_items(args.data, args.masks, args.scenario, args.images)
    seeds = [INPAINT_SEED + k for k in range(args.seeds)]

    def repaint(x01, item, mask, seed):
        emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
        noise = sd.sample_edit_noise(sd.encode_image(x01), seed=seed)
        with torch.no_grad():
            return sd.inpaint(x01.clamp(0, 1), mask, emb, noise, INPAINT_STEPS,
                              guidance_scale=INPAINT_GUIDANCE, emb_uncond=emb_u)

    rows = []
    for item in items:
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        mask = load_mask(item["mask"], sd.device)
        save_image(x01, args.out / f"{item['name']}__orig.png")
        save_image(mask.expand(-1, 3, -1, -1), args.out / f"{item['name']}__mask.png")
        print(f"\n##### {item['name']} · {args.scenario} · “{item['prompt']}” "
              f"· 重畫區 {float(mask.mean()):.3f} #####", flush=True)

        base = {}
        for s in seeds:
            y = repaint(x01, item, mask, s)
            base[s] = masked_compare(y, x01, mask)
            if s == seeds[0]:
                save_image(y, args.out / f"{item['name']}__gen_orig.png")
        # 攻擊本身有沒有發生：未防禦的重畫對原圖，遮罩內。
        atk = float(suite.pairwise(
            masked_compare(x01, x01, mask), base[seeds[0]])["lpips"])

        for cond in args.conditions:
            p = find_def(args.defended, item["name"], cond)
            if p is None:
                print(f"  [略過] {cond}：找不到防禦圖", flush=True)
                continue
            t0 = time.time()
            x_def = load_image_tensor(p, sd.device, size=RESOLUTION)
            vals = []
            for s in seeds:
                y = repaint(x_def, item, mask, s)
                vals.append(float(suite.pairwise(
                    base[s], masked_compare(y, x01, mask))["lpips"]))
                if s == seeds[0]:
                    save_image(y, args.out / f"{item['name']}__{cond}__gen_def.png")
            m = suite.pairwise(x01, x_def)
            a = aes.measure(x_def)
            row = {"image": item["name"], "condition": cond,
                   "scenario": args.scenario, "prompt": item["prompt"],
                   "gen_lpips": round(sum(vals) / len(vals), 4),
                   "gen_lpips_sd": round(
                       (sum((v - sum(vals) / len(vals)) ** 2 for v in vals)
                        / max(len(vals) - 1, 1)) ** 0.5, 4) if len(vals) > 1 else "",
                   "attack_strength": round(atk, 4),
                   "mask_coverage": round(float(mask.mean()), 4),
                   "fid_lpips": round(float(m["lpips"]), 4),
                   "fid_dists": round(float(m["dists"]), 4),
                   "fid_psnr": round(float(m["psnr"]), 4),
                   "fid_ssim": round(float(m["ssim"]), 4),
                   "fid_nima": round(float(a["nima"]), 4),
                   "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(row, flush=True)
            write_csv(args.out / "results.csv", rows)

    print(f"\n表：{args.out / 'results.csv'}")


if __name__ == "__main__":
    main()
