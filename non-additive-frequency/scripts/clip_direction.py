"""指令服從度：CLIP 的**方向**相似度，各自以自己的輸入為參照。

    D(x) = cos( I(編輯(x)) − I(x),  T(目標敘述) − T(來源敘述) )

`I`、`T` 是 CLIP 的影像／文字嵌入（正規化後）。這個量問的是**「編輯把影像
往指令說的方向推了多遠」**，而不是「兩張輸出差多少」。

為什麼需要它
────────────────────────────────────────────────────────────────────
語意誘餌交出的是另一張真的照片，於是 `LPIPS(編輯(原圖), 編輯(誘餌圖))` 被內容
差異灌水，空白地板扣不掉、共防禦參照也用不上（誘餌不是逐點映射）。
`scripts/edit_take.py` 的 `LPIPS(x, 編輯(x))` 解決了參照問題，但它**沒有方向**
——編輯做得多可能是照著指令做，也可能是把場景重畫掉。

本支補上方向。減去 `I(x)` 使誘餌本身的內容差異在一階上抵消；
文字側減去來源敘述，使「這張圖本來就有多像那句話」不進來。

敘述怎麼來
────────────────────────────────────────────────────────────────────
**只由指令模板機械推導，不臆測。** OmniEdit 的 `attribute_modification` 指令是

    turn the color of <物件> to [be] <顏色>

於是 來源 = `a <物件>`、目標 = `a <顏色> <物件>`。**對不上這個模板的指令一律
跳過並印出來**，不猜、不套用別的模板——本專案已實測到 OmniEdit 給的是**指令**
不是描述，把指令直接當 caption 做 CLIP 對齊近乎隨機（25 張裡 15 張為正）。

讀法
────────────────────────────────────────────────────────────────────
`D` 越大代表編輯越照著指令走。**主讀數是防禦圖與原圖的 `D` 之差**：

    ΔD = D(誘餌圖) − D(原圖)

負值代表在防禦圖上，指令被執行得比在原圖上少。**這不是「擋下」的判準**，
擋下與否仍由人眼判；本支只是把「指令有沒有被照著做」這一件事量出來。

用法：

    python scripts/clip_direction.py --src runs/ip2p_decoy/dev5 \\
        --out runs/ip2p_decoy/clip_direction.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torchvision.transforms.functional import resize  # noqa: E402

from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512

# OmniEdit 的 attribute_modification 模板。`to be` 與 `to` 兩種寫法都有。
COLOR_TEMPLATE = re.compile(
    r"^turn the colou?r of (?:the |a |an )?(.+?) to (?:be )?(.+?)\.?$",
    re.IGNORECASE)


def captions(instruction: str) -> Optional[Tuple[str, str]]:
    """指令 → (來源敘述, 目標敘述)。對不上模板回傳 None。"""
    m = COLOR_TEMPLATE.match(instruction.strip())
    if not m:
        return None
    obj, colour = m.group(1).strip(), m.group(2).strip()
    return f"a {obj}", f"a {colour} {obj}"


@torch.no_grad()
def embed(suite: MetricSuite, images, texts):
    """走與 `MetricSuite.semantic_multi` 相同的前向，回傳正規化後的嵌入。"""
    suite._ensure_vlm()
    model, proc = suite._clip, suite._clip_proc
    size = proc.image_processor.size
    side = size.get("shortest_edge") or size["height"]
    mean = torch.tensor(proc.image_processor.image_mean, device=suite.device)
    std = torch.tensor(proc.image_processor.image_std, device=suite.device)
    pix = []
    for x in images:
        img = resize(x.to(suite.device).float().clamp(0, 1), [side, side],
                     antialias=True)
        pix.append((img - mean[:, None, None]) / std[:, None, None])
    pix = torch.cat(pix, dim=0)
    tok = proc.tokenizer(list(texts), return_tensors="pt", padding=True,
                         truncation=True).to(suite.device)
    res = model(pixel_values=pix, **tok)
    ie = res.image_embeds / res.image_embeds.norm(dim=-1, keepdim=True)
    te = res.text_embeds / res.text_embeds.norm(dim=-1, keepdim=True)
    return ie, te


def direction(ie_in, ie_out, te_src, te_tgt) -> float:
    di = ie_out - ie_in
    dt = te_tgt - te_src
    denom = di.norm() * dt.norm()
    if float(denom) == 0.0:
        # 編輯完全沒動、或兩句敘述的嵌入相同。方向無定義，回報 nan 不補零。
        return float("nan")
    return float((di * dt).sum() / denom)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=dev)

    rows, skipped = [], []
    for src in args.src:
        recs = list(csv.DictReader((src / "results.csv").open(encoding="utf-8")))
        base_cache = {}
        for r in recs:
            name, cond = r["image"], r["condition"]
            caps = captions(r["instruction"])
            if caps is None:
                skipped.append(f"{name}：指令對不上模板 → {r['instruction']!r}")
                continue
            src_cap, tgt_cap = caps
            paths = {
                "orig": src / f"{name}__orig.png",
                "def": src / f"{name}__{cond}__def.png",
                "edit_orig": src / f"{name}__{cond}__edit_orig.png",
                "edit_def": src / f"{name}__{cond}__edit_def.png",
            }
            missing = [k for k, p in paths.items() if not p.exists()]
            if missing:
                skipped.append(f"{name}/{cond}：缺 {missing}")
                continue
            img = {k: load_image_tensor(p, dev, size=RESOLUTION)
                   for k, p in paths.items()}
            ie, te = embed(suite,
                           [img["orig"], img["edit_orig"],
                            img["def"], img["edit_def"]],
                           [src_cap, tgt_cap])
            d_orig = direction(ie[0], ie[1], te[0], te[1])
            d_def = direction(ie[2], ie[3], te[0], te[1])
            # **絕對對齊**，用來拆掉方向讀數的一個混淆：若誘餌本身就把畫面
            # 推向目標敘述（例如指令要粉紅、而誘餌就是一把粉紅傘），
            # `I(編輯(x)) − I(x)` 剩下可走的粉紅方向本來就少，ΔD 會下降
            # **即使編輯照樣成功**。`abs_in_def` 就是那一份預先的對齊量；
            # `abs_out_def − abs_in_def` 才是編輯自己加上去的。
            abs_out_orig = float((ie[1] * te[1]).sum())
            abs_out_def = float((ie[3] * te[1]).sum())
            abs_in_orig = float((ie[0] * te[1]).sum())
            abs_in_def = float((ie[2] * te[1]).sum())
            rows.append({
                "image": name, "condition": cond,
                "instruction": r["instruction"],
                "source_caption": src_cap, "target_caption": tgt_cap,
                "decoy_instruction": r.get("decoy_instruction", ""),
                "clip_dir_orig": round(d_orig, 5),
                "clip_dir_def": round(d_def, 5),
                "delta": round(d_def - d_orig, 5),
                "abs_in_orig": round(abs_in_orig, 5),
                "abs_out_orig": round(abs_out_orig, 5),
                "abs_gain_orig": round(abs_out_orig - abs_in_orig, 5),
                "abs_in_def": round(abs_in_def, 5),
                "abs_out_def": round(abs_out_def, 5),
                "abs_gain_def": round(abs_out_def - abs_in_def, 5),
            })
            print(f"{name[15:]:26s}{cond:28s}D(原圖)={d_orig:+.4f} "
                  f"D(防禦圖)={d_def:+.4f} Δ={d_def - d_orig:+.4f}", flush=True)

    for s in skipped:
        print(f"[skip] {s}", flush=True)
    write_csv(args.out, rows)
    if rows:
        deltas = [r["delta"] for r in rows]
        print(f"\nn={len(rows)}  Δ 中位數={statistics.median(deltas):+.4f}"
              f"  最小={min(deltas):+.4f}  最大={max(deltas):+.4f}")
    print(f"寫出 {args.out}")


if __name__ == "__main__":
    main()
