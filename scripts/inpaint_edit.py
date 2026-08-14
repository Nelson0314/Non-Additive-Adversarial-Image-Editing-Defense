"""用既有的全圖防禦圖跑 inpainting 編輯。遮罩自動產生，場景只有一種。

**在主體上加東西**：遮罩是主體的一個局部方框（頭部或軀幹），prompt 描述
加上配件之後的整體（`a cat wearing a red party hat`）。使用者 2026-08-15
裁定移除「換掉整個主體」那一組——它逼模型從零重畫一個物件，難度高且與實際
的惡意編輯情境不符。

人物用 `torso` 而非 `head`：這批的人物在畫面中佔比大，臉部方框會涵蓋畫面
一半，重畫區太大則「攻擊有沒有成功」不再能歸因到防禦。

逐影像的 region 與 prompt 由 `data/lo_inpaint_edits.yaml` 給定，**只有這一
個地方決定它們**——2026-08-14 的協定錯誤（FND-038）正是遮罩與 prompt 各自
從不同來源取值造成的。

**防禦加在整張圖上，遮罩在編輯時才套用。** 故不需要為 inpainting 重跑攻擊，
直接讀 img2img 批次（`runs/hb5`）存下的 `*__def.png`。

`effect` 只在遮罩內量：`sd.inpaint` 每一步把遮罩外貼回，兩條分支在該處正好
差一個防禦擾動，整張圖算 LPIPS 會把失真算成效果。

用法：
    python scripts/inpaint_edit.py --out runs/ip3 --defended runs/hb5 runs/hb5_pgc         --masks data/lo_masks_auto --seeds 3
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
EDITS = Path("data/lo_inpaint_edits.yaml")


def load_items(data: Path, masks: Path, edits: Path, images=None) -> list:
    """逐影像的 region 與 prompt 由 `lo_inpaint_edits.yaml` 給定。

    不再由 `prompts.yaml` 推 prompt——那份是 img2img 的規格，兩邊各自決定
    prompt 正是 2026-08-14 那次協定錯誤的來源（FND-038）。
    """
    spec = yaml.safe_load(edits.read_text(encoding="utf-8"))
    paths = {p.stem: p for c in data.iterdir() if c.is_dir()
             for p in c.glob("*.png")}
    out = []
    for name in sorted(spec):
        if images and name not in images:
            continue
        if name not in paths:
            raise FileNotFoundError(f"{name} 不在 {data}")
        region = spec[name]["region"]
        m = masks / f"{name}__{region}.png"
        if not m.exists():
            raise FileNotFoundError(f"{name} 缺 {region} 遮罩：{m}")
        out.append({"name": name, "path": paths[name], "mask": m,
                    "region": region, "prompt": spec[name]["prompt"]})
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
    ap.add_argument("--edits", type=Path, default=EDITS)
    ap.add_argument("--orig-only", action="store_true",
                    help="只跑未防禦的重畫，供開跑前以人眼複驗攻擊是否成功")
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
    items = load_items(args.data, args.masks, args.edits, args.images)
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
        print(f"\n##### {item['name']} · {item['region']} · “{item['prompt']}” "
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

        if args.orig_only:
            continue
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
                   "region": item["region"], "prompt": item["prompt"],
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
