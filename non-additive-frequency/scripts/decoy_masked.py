"""把已生成的誘餌**合成回主體區域之外**，再讓攻擊方編輯一次。

為什麼要這一步
────────────────────────────────────────────────────────────────────
`runs/ip2p_decoy/` 的控制組量到：與攻擊指令同色的誘餌壓得最兇，但**看圖之後
那兩格都是靠違規贏的**——

- 盆栽人的粉紅傘：誘餌指令只說「在背景加一把粉紅色的傘」，IP2P 順手把**花盆**
  也變成粉紅色。攻擊指令要的正是「把盆栽變粉紅」，於是**防禦方自己先把攻擊
  做完了**，攻擊之後加不上東西。
- 瑪利歐的紅傘：誘餌把**整張圖換掉**（沙灘上一把巨傘），主體不見了。
  誘餌本身的失真 0.3615，是同組其他顏色的 3.4–4.8 倍。

四格低失真、主體完好的誘餌（藍／綠／灰）**全部沒有壓制效果**。也就是說那一批
裡沒有任何一格在「不動到主體」的前提下成功。

兩個失效模式（**溢出到主體**、**整張換掉**）都可以由構造擋掉：把誘餌只留在
主體區域之外。

    x_def = m · 原圖 + (1 − m) · 誘餌圖

`m = 1` 的地方**逐位元**是原圖。主體遮罩由發布者標出——威脅模型的前提就是
防護對象已知；本檔用 `data/subject_masks.yaml` 的矩形代替那個標注動作。

**不重跑誘餌。** 誘餌圖已存在，這一支只做合成與一次攻擊編輯。

邊緣
────────────────────────────────────────────────────────────────────
矩形外側羽化 `--feather` 像素，避免硬接縫。**羽化只往外**：框內恆為 1，
主體因此仍然逐位元保留。

用法：

    python scripts/decoy_masked.py --src runs/ip2p_decoy/dev5 \\
        --out runs/ip2p_decoy_masked/dev5 --masks data/subject_masks.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torchvision.utils as vutils  # noqa: E402
import yaml  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from src.metrics.standard import standard_row  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.ip2p import (  # noqa: E402
    IP2P_IMAGE_GUIDANCE, IP2P_SEED, IP2P_STEPS, IP2P_TEXT_GUIDANCE, IP2PWrapper,
)
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def subject_mask(box, size: int, feather: int, device, dtype) -> torch.Tensor:
    """(1,1,H,W)：矩形內為 1（保留原圖），外側羽化到 0。

    `box` 是正規化的 `[x0, y0, x1, y1]`。羽化**只往外**，故框內恆為 1，
    主體逐位元保留。
    """
    x0, y0, x1, y1 = box
    ys = torch.arange(size, device=device, dtype=dtype).view(-1, 1)
    xs = torch.arange(size, device=device, dtype=dtype).view(1, -1)
    px0, px1 = x0 * size, x1 * size
    py0, py1 = y0 * size, y1 * size
    # 到矩形的距離（框內為 0）
    dx = torch.clamp(px0 - xs, min=0) + torch.clamp(xs - px1, min=0)
    dy = torch.clamp(py0 - ys, min=0) + torch.clamp(ys - py1, min=0)
    dist = torch.sqrt(dx ** 2 + dy ** 2)
    if feather <= 0:
        m = (dist <= 0).to(dtype)
    else:
        m = torch.clamp(1.0 - dist / feather, min=0.0, max=1.0)
    return m.view(1, 1, size, size)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, nargs="+", required=True,
                    help="已生成的誘餌批次目錄（含 results.csv 與 __def.png）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--masks", type=Path,
                    default=Path("data/subject_masks.yaml"))
    ap.add_argument("--feather", type=int, default=12,
                    help="矩形外側的羽化寬度（像素）。**只往外**，框內恆為 1")
    args = ap.parse_args()

    boxes = yaml.safe_load(args.masks.read_text(encoding="utf-8"))
    dataset = {d["name"]: d for d in load_dataset(args.data)}
    sd = IP2PWrapper(dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, edit_orig_cache = [], {}
    for src in args.src:
        recs = list(csv.DictReader((src / "results.csv").open(encoding="utf-8")))
        for r in recs:
            name, cond = r["image"], r["condition"]
            if name not in boxes:
                print(f"[skip] {name}：{args.masks} 沒有這張影像的主體框",
                      flush=True)
                continue
            decoy_png = src / f"{name}__{cond}__def.png"
            if not decoy_png.exists():
                print(f"[skip] 缺 {decoy_png}", flush=True)
                continue
            t0 = time.time()
            item = dataset[name]
            x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
            decoy = load_image_tensor(decoy_png, sd.device, size=RESOLUTION)
            m = subject_mask(boxes[name]["box"], RESOLUTION, args.feather,
                             sd.device, x01.dtype)
            x_def = (m * x01 + (1.0 - m) * decoy).clamp(0, 1)
            # 由構造保證：框內逐位元是原圖。每一列都驗一次，不是只驗一次。
            inside = m >= 1.0
            if inside.any():
                gap = float(((x_def - x01) * inside).abs().max())
                if gap != 0.0:
                    raise RuntimeError(
                        f"{name}/{cond}：主體框內與原圖差 {gap}，應恰為 0。")

            if name not in edit_orig_cache:
                edit_orig_cache[name] = sd.edit(
                    x01.clamp(0, 1), item["prompt"], seed=IP2P_SEED,
                    steps=IP2P_STEPS, s_t=IP2P_TEXT_GUIDANCE,
                    s_i=IP2P_IMAGE_GUIDANCE)
                vutils.save_image(x01.clamp(0, 1),
                                  args.out / f"{name}__orig.png")
            e_orig = edit_orig_cache[name]
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
                "decoy_group": r.get("decoy_group", ""),
                "decoy_index": r.get("decoy_index", ""),
                "decoy_instruction": r.get("decoy_instruction", ""),
                "decoy_seed": r.get("decoy_seed", ""),
                "decoy_steps": r.get("decoy_steps", ""),
                "decoy_s_t": r.get("decoy_s_t", ""),
                "decoy_s_i": r.get("decoy_s_i", ""),
                # 合成的兩個設定。主體框是**發布者標的**，不是量出來的。
                "subject_box": ",".join(str(v) for v in boxes[name]["box"]),
                "subject_area": round(float(m.mean()), 5),
                "feather": args.feather,
                "source_batch": src.name,
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
            print(f"{name[15:]:26s}{cond:28s}"
                  f"合成後失真={fid['dists']:.4f} 位移={eff['lpips']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n表：{args.out / 'results.csv'}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
