"""只重繪一塊，受保護的那一塊逐位元保留：身分由構造保證。

為什麼是這條路
────────────────────────────────────────────────────────────────────
生成載體最大的失效是**防禦圖本身把人換掉**（直接交付 SDEdit 的輸出在
`strength 0.35` 上八張裡有三張的 FaceNet 餘弦掉到 0.55 以下）。低頻色彩轉移
是一種解法：只取生成模型的顏色，結構留在原圖。這一檔是另一種解法，而且更硬
——**受保護的區域根本不送進生成模型**，重繪只發生在它之外，最後再逐位元
合成回去。身分不是被「保住」，是根本沒有被碰過。

兩種區域
────────────────────────────────────────────────────────────────────
| `--region` | 重繪哪裡 | 保留哪裡 |
|---|---|---|
| `background` | 主體之外（資料集既有的遮罩，白＝重繪） | 整個主體 |
| `outside_face` | 臉以外的全部（含頭髮、衣服、背景） | 臉 |

**這兩個的預期不一樣，而且其中一個預期會失效。** inpainting 那個受害模型
重繪的區域正好就是主體之外——也就是 `background` 改動的那一塊。受害模型會
把它整個蓋掉，所以 `background` 在 inpaint 場景上預期推不動任何東西。
這與主表的 `diffvax` 是同一個幾何關係（它的擾動也只存在於重繪區之外，
inpaint 主體內位移 0.0348）。`outside_face` 改的是主體內的頭髮與衣服，
受害模型會把那一塊保留下來，所以它在 inpaint 場景上有作用。

**把這件事寫在這裡，是因為它是設計的一部分，不是事後解釋。** 兩個臂都跑、
都照報，讀數要連這個幾何關係一起讀。

合成的邊界
────────────────────────────────────────────────────────────────────
`background` 沿著主體輪廓合成——那裡原本就有一條邊，硬合成不產生新的邊，
而且這正是編輯管線自己在做的事（`edit = mask·raw + (1−mask)·x`）。

`outside_face` 的邊界落在頭的中間，硬合成會是一條看得見的接縫。所以這一個
用 C¹ 的 smoothstep 場合成（`curve_budget_defence.smooth_field`），送進 pipeline
的二值遮罩則取該場的 0.5 等高線。

**沒有失真上限。** 重繪區是生成出來的，ΔE00 對它沒有意義；這一臂與兩個直接
交付 SDEdit 的臂一樣掛在「無預算」那一類，不可與有 ΔE00 上限的臂比大小。

產出（lab 各臂共用的版面：`<影像>__<臂>__def.png`）
────────────────────────────────────────────────────────────────────
    {out}/{image}__orig.png
    {out}/{image}__{arm}__def.png
    {out}/{image}__raw.png        pipeline 的原始輸出（未合成）
    {out}/{image}__field.png      合成用的權重場（1 = 保留原圖）
    {out}/results.csv
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
import yaml  # noqa: E402

from curve_budget_defence import (  # noqa: E402
    box_support, expanded_box, smooth_field, write_rows,
)
from src.defense.color_amplitude import delta_e00  # noqa: E402
from src.metrics.identity import embed_box, face_boxes  # noqa: E402
from src.models.sd import SDInpaintWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512
INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"

#: 防禦方自己選的風格句。**攻擊指令不會進到這裡**——那是攻擊方寫的。
#: `{content}` 由該影像類別的 content 欄代入，那是防禦方選定要保護的詞。
#:
#: 句子要描述**場景**，不可以描述**媒材**
#: ────────────────────────────────────────────────────────────────
#: 第一版寫的是 `a portrait of a {content}, vintage cross-processed film
#: photograph, heavy grain, teal and orange colour grading`。結果 inpainting
#: 照字面畫了一個**實體相框**：`outside_face` 那一批的重繪區變成一圈木頭邊框
#: 把臉框在中間，`background` 那一批則因為沒有場景可畫，只糊出一圈灰白暈與
#: 暗角。原因很直接——inpainting 的 prompt 描述的是**整張結果影像**，
#: 而「vintage photograph」在模型眼裡是一張實體照片，不是一種色調。
#:
#: 所以這裡的句子一律是「主體在某個場景裡」，帶季節、光線與顏色，
#: **不出現任何媒材詞**（photograph、film、vintage、portrait、frame）。
#: 主體要寫進句子裡：本專案已量到只描述填充物的片語會讓模型照上下文
#: 補出另一張人臉。
STYLE_PROMPTS = {
    "autumn": "a {content} outdoors in autumn, warm golden and amber foliage "
              "filling the background, soft late afternoon sunlight",
    "winter": "a {content} outdoors in winter, snow-covered trees and a pale "
              "blue overcast sky behind them",
    "teal": "a {content} standing in front of a deep teal painted wall, "
            "cool even lighting",
    "sunset": "a {content} outdoors at sunset, an orange and magenta sky "
              "behind them",
}

#: 反向 prompt。前兩項擋的是上面那個相框，後兩項擋的是本專案早就量到的
#: 「模型照上下文補出第二個人」。
NEGATIVE_PROMPT = ("picture frame, border, vignette, passe-partout, canvas edge, "
                   "text, watermark, signature, collage, "
                   "multiple people, second face, duplicate person")


def load_items(root: Path, only):
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for cls in sorted(k for k in spec if k != "edits"):
        for img in sorted((root / cls).glob("*.png")):
            if only and img.stem not in only:
                continue
            mask = root / "masks" / f"{img.stem}.png"
            out.append({"name": img.stem, "class": cls, "path": img,
                        "content": spec[cls]["content"],
                        "mask": mask if mask.is_file() else None})
    if not out:
        raise SystemExit(f"{root} 找不到影像（--images 過濾後為空）")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--region", required=True,
                    choices=("background", "outside_face"))
    ap.add_argument("--style", default="autumn", choices=sorted(STYLE_PROMPTS))
    ap.add_argument("--negative-prompt", default=NEGATIVE_PROMPT,
                    help="反向 prompt。預設擋相框與第二個人，逐列寫進 CSV")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--feather", type=float, default=0.35,
                    help="outside_face 的合成過渡帶寬度，單位是臉框半徑的倍數")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    victim = SDInpaintWrapper(INPAINT_MODEL, dtype=torch.float32)
    device = victim.device
    items = load_items(args.data, set(args.images) if args.images else None)

    rows = []
    for item in items:
        started = time.time()
        x = load_image_tensor(item["path"], device, size=RESOLUTION)
        boxes = face_boxes(x, device)
        if not boxes:
            raise SystemExit(f"{item['name']} 偵測不到臉；身分讀數沒有錨點")
        box = expanded_box(x, max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])))
        face = box_support(x, box)

        if args.region == "background":
            if item["mask"] is None:
                raise SystemExit(f"{item['name']} 缺遮罩，background 模式跑不了")
            repaint = load_image_tensor(item["mask"], device,
                                        size=RESOLUTION)[:, :1]
            repaint = (repaint >= 0.5).to(x.dtype)
            keep = 1.0 - repaint                       # 硬邊，落在主體輪廓上
        else:
            field = smooth_field(x, box, args.feather)  # 1 = 臉
            keep = field
            repaint = (field < 0.5).to(x.dtype)

        prompt = STYLE_PROMPTS[args.style].format(content=item["content"])
        with torch.no_grad():
            gen = torch.Generator(device=device).manual_seed(int(args.seed))
            raw = victim.pipe(prompt=prompt, image=x, mask_image=repaint,
                              negative_prompt=args.negative_prompt or None,
                              strength=1.0, num_inference_steps=args.steps,
                              guidance_scale=args.guidance, generator=gen,
                              output_type="pt").images.to(x.dtype)
            y = (keep * x + (1.0 - keep) * raw).clamp(0, 1)
            y = (y * 255).round() / 255                 # 交付的是 PNG
            frame = torch.ones_like(x[:, :1])
            id0 = embed_box(x, box, device).reshape(1, -1)
            id_def = float(torch.nn.functional.cosine_similarity(
                id0, embed_box(y, box, device).reshape(1, -1)))
            id_raw = float(torch.nn.functional.cosine_similarity(
                id0, embed_box(raw, box, device).reshape(1, -1)))
            kept_exact = float(((keep >= 0.999) & ((y - x).abs().amax(1, True) < 1e-6))
                               .to(torch.float32).sum()
                               / (keep >= 0.999).to(torch.float32).sum().clamp_min(1))

        row = {
            "image": item["name"], "class": item["class"], "arm": args.arm,
            "carrier": "inpaint_" + args.region, "carrier_model": INPAINT_MODEL,
            "region": args.region, "style": args.style, "style_prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "steps": args.steps, "guidance": args.guidance, "seed": args.seed,
            "feather": args.feather if args.region == "outside_face" else "",
            "budget": "none（重繪區是生成的，ΔE00 對它沒有意義）",
            "repaint_frac": round(float(repaint.mean()), 5),
            "keep_frac": round(float((keep >= 0.999).to(torch.float32).mean()), 5),
            # 保留區有沒有真的逐位元保留。合成公式上應該是 1.0；
            # **不是 1.0 就代表合成寫錯了**，這一欄就是為了讓那件事有症狀。
            "kept_bitexact_frac": round(kept_exact, 5),
            "deltaE00_frame": round(float(delta_e00(x, y, frame)), 4),
            "deltaE00_face_box": round(float(delta_e00(x, y, face)), 4),
            "psnr": round(float(10 * torch.log10(1.0 / (y - x).pow(2).mean())), 4),
            "linf": round(float((y - x).abs().max()), 5),
            "id_cos": round(id_def, 5), "id_cos_raw": round(id_raw, 5),
            "seconds": round(time.time() - started, 1),
        }
        save_image(x, args.out / f"{item['name']}__orig.png")
        save_image(y, args.out / f"{item['name']}__{args.arm}__def.png")
        save_image(raw, args.out / f"{item['name']}__raw.png")
        save_image(keep.repeat(1, 3, 1, 1), args.out / f"{item['name']}__field.png")
        rows.append(row)
        write_rows(args.out / "results.csv", rows)
        print(f"[{args.arm}] {item['name']}  重繪 {row['repaint_frac']}  "
              f"保留逐位元 {row['kept_bitexact_frac']}  "
              f"ΔE 臉框 {row['deltaE00_face_box']}  身分 {row['id_cos']} "
              f"(raw {row['id_cos_raw']})  {row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 張 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
