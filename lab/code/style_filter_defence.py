"""用現成生成編輯模型**設計一張色彩濾鏡**，再把濾鏡套回原圖。

為什麼不直接交付 SDEdit 的輸出
────────────────────────────────────────────────────────────────────
直接把 SDEdit 的輸出當防禦圖，在 `strength 0.35` 上會把人**換掉**（同一批
影像上 FaceNet 餘弦掉到 0.26–0.76，八張裡有三張低於本專案「同一個人」的
0.55）。換掉人之後位移讀數很高，但那個高不是免疫——分母（未防禦的編輯）
保護的是另一個人，兩邊已經不可比。這是這條線最主要的靜默失效點。

載體
────────────────────────────────────────────────────────────────────
生成模型只用來**產生一個色彩位移場**，結構完全來自原圖：

    Δ      = SDEdit(x ; style, strength, steps, guidance) − x
    Δ_lf   = G_σ(Δ)                       低通，σ = --sigma（像素）
    α(p)   = a_bg·(1 − w(p)) + a_face·w(p)
    y      = clamp( x + α(p)·Δ_lf , 0, 1 )

三件事因此在構造上成立：

1. **身分與結構逐像素來自原圖。** 高頻完全沒有被重繪，`Δ_lf` 只帶低頻色彩。
2. **沒有硬邊。** `Δ_lf` 是平滑場、`w` 是 C¹ 的 smoothstep 場，兩者相乘仍是
   平滑場。
3. **臉與背景吃不同的預算。** `a_face` 與 `a_bg` 各自由二分搜尋定出，
   分別對上人臉框內與整圖的 ΔE00 上限。兩者用同一個平滑場插值，
   所以「不同預算」不等於「兩塊不同的區域」。

`--smoother` 的兩個值不是選項，是兩件不同的事
────────────────────────────────────────────────────────────────────
`gaussian` 在 `strength 0.6` 上會在主體輪廓外留下**一圈光暈**：那個 strength
下 SDEdit 已經把髮型與輪廓換掉，`Δ` 因此含大量結構差異，高斯低通把它抹過
輪廓。光暈不是硬接縫，但同樣是產物上看得出來的東西，逐張圖在
`runs/defence/style_filter/` 可查。

`guided` 用 guided filter，導引影像是原圖的亮度：平滑只發生在原圖的同質區
內，跨過原圖的邊界時不平均，於是色彩的轉折落在**主體自己的輪廓上**。
兩者其餘逐項相同，可以直接對照。

與全域色調曲線的差別是**空間相依**：曲線是 `x → F(x)`，同一個輸入色階在任何
位置都同一個輸出；這裡的位移場隨內容而變，容量高一個層級，而代價仍鎖在
同一道 ΔE00 上限上。

**這一臂不做最佳化。** 它回答的是「現成的生成模型當濾鏡設計器，本身推得動
多少」。要不要再對風格做搜尋，是下一步的事，不在這一臂裡。

產出（版面與 `defence_run.py` 相同）
────────────────────────────────────────────────────────────────────
    {out}/{image}__orig.png
    {out}/{image}__{arm}__def.png
    {out}/{image}__field.png       α(p)，供逐格看圖時對照
    {out}/results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402
import yaml  # noqa: E402

from curve_budget_defence import (  # noqa: E402
    box_support, expanded_box, load_images, smooth_field, write_rows,
)
from src.defense.color_amplitude import delta_e00  # noqa: E402
from src.defense.uniformity import lab_offset, tv_offset  # noqa: E402
from src.metrics.identity import embed_box, face_boxes  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512
CARRIER_MODEL = "runwayml/stable-diffusion-v1-5"

#: 風格目錄。**防禦方自己選的**，與攻擊指令屬於不同的人，故可以進求解端。
STYLE_CATALOGUE = {
    "film": ("a vintage cross-processed film photograph, heavy grain, "
             "teal and orange colour grading, faded highlights"),
    "cyan": ("a photograph with a strong cyan-magenta colour filter, "
             "high contrast, cold cinematic grade"),
    "paint": ("an oil painting with thick visible brush strokes, "
              "saturated pigments, canvas texture"),
}


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """可分離高斯低通。半徑取 3σ，`reflect` 補邊避免邊緣變暗。"""
    radius = max(1, int(round(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    pad = torch.nn.functional.pad(x, (radius, radius, 0, 0), mode="reflect")
    out = torch.nn.functional.conv2d(pad, kx, groups=c)
    pad = torch.nn.functional.pad(out, (0, 0, radius, radius), mode="reflect")
    return torch.nn.functional.conv2d(pad, ky, groups=c)


def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """可分離盒濾波，`reflect` 補邊。"""
    k = 2 * radius + 1
    c = x.shape[1]
    ones = torch.ones(c, 1, 1, k, device=x.device, dtype=x.dtype) / k
    pad = torch.nn.functional.pad(x, (radius, radius, 0, 0), mode="reflect")
    out = torch.nn.functional.conv2d(pad, ones, groups=c)
    ones = ones.view(c, 1, k, 1)
    pad = torch.nn.functional.pad(out, (0, 0, radius, radius), mode="reflect")
    return torch.nn.functional.conv2d(pad, ones, groups=c)


def guided_filter(guide: torch.Tensor, src: torch.Tensor,
                  radius: int, eps: float) -> torch.Tensor:
    """He 等人的 guided filter，導引影像是**原圖的亮度**。

    為什麼需要它：高斯低通會把 `Δ` 裡的**結構**差異抹過主體輪廓，於是
    `strength 0.6` 下 SDEdit 換掉髮型的那一圈會在交付圖上變成一道光暈。
    光暈不是硬接縫，但同樣是產物上看得出來的東西。

    guided filter 讓平滑只發生在導引影像的同質區內，跨過原圖的邊界時不平均，
    因此色彩的轉折落在**主體自己的輪廓上**，而不是輪廓外的一圈。
    """
    mean_i = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    corr_i = box_filter(guide * guide, radius)
    corr_ip = box_filter(guide * src, radius)
    var_i = (corr_i - mean_i * mean_i).clamp_min(0.0)
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box_filter(a, radius) * guide + box_filter(b, radius)


def smooth_delta(x: torch.Tensor, delta: torch.Tensor, args) -> torch.Tensor:
    if args.smoother == "gaussian":
        return gaussian_blur(delta, args.sigma)
    guide = (x * torch.tensor([0.299, 0.587, 0.114], device=x.device,
                              dtype=x.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
    return guided_filter(guide, delta, args.radius, args.eps)


def quantise(x: torch.Tensor) -> torch.Tensor:
    """交付的是 PNG，所以可行性一律在量化後的影像上檢查。"""
    return (x.detach().clamp(0, 1) * 255).round() / 255


def compose(x, delta_lf, field, a_bg, a_face):
    alpha = a_bg * (1.0 - field) + a_face * field
    return quantise(x + alpha * delta_lf), alpha


def bisect(measure, cap: float, hi: float = 3.0, tries: int = 20) -> float:
    """最大的 `a`，使 `measure(a) <= cap`。`measure` 對 `a` 單調遞增。

    `hi` 本身可行就直接回傳 `hi`：不可以靜默再縮，那會讓交付的圖比上限鬆的
    時候還是被壓掉振幅，而 CSV 上看不出來。
    """
    if measure(hi) <= cap:
        return hi
    lo = 0.0
    for _ in range(tries):
        mid = 0.5 * (lo + hi)
        if measure(mid) <= cap:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--style", default="film", choices=sorted(STYLE_CATALOGUE))
    ap.add_argument("--strength", type=float, default=0.6,
                    help="SDEdit 的 strength。這裡只取它的低頻色彩，所以可以"
                         "比直接交付那一臂高——結構不會被帶進交付的圖")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--smoother", default="gaussian",
                    choices=("gaussian", "guided"),
                    help="怎麼把 Δ 變成只帶色彩的平滑場。gaussian 會在主體輪廓"
                         "外留下一圈光暈（Δ 含結構差異時）；guided 只在原圖的"
                         "同質區內平滑，轉折落在主體自己的輪廓上")
    ap.add_argument("--sigma", type=float, default=12.0,
                    help="高斯低通的 σ（像素）。只有 --smoother gaussian 用得到")
    ap.add_argument("--radius", type=int, default=32,
                    help="guided filter 的半徑（像素）")
    ap.add_argument("--eps", type=float, default=0.01,
                    help="guided filter 的 eps。越小越貼著導引影像的邊")
    ap.add_argument("--frame-cap", type=float, default=16.0)
    ap.add_argument("--face-cap", type=float, default=8.0)
    ap.add_argument("--feather", type=float, default=0.35)
    ap.add_argument("--max-scale", type=float, default=3.0,
                    help="α 的上界。SDEdit 的低頻位移本身通常打不滿 ΔE00 16 的"
                         "整圖預算，所以允許放大；上界本身可行時就照上界交付，"
                         "不靜默再縮")
    ap.add_argument("--dtype", default="float16", choices=("float32", "float16"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sd = SDWrapper(CARRIER_MODEL, dtype=getattr(torch, args.dtype))
    prompt = STYLE_CATALOGUE[args.style]
    emb = sd.encode_text(prompt).detach()
    uncond = sd.uncond_prompt().detach()

    items = load_images(args.data, set(args.images) if args.images else None)
    rows = []
    for item in items:
        started = time.time()
        x = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        boxes = face_boxes(x, sd.device)
        if not boxes:
            raise SystemExit(f"{item['name']} 偵測不到臉；預算要分在臉與背景之間，"
                             "偵不到臉就沒有可分的界線，不要靜默退回單一預算")
        box = expanded_box(x, max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])))
        frame = torch.ones_like(x[:, :1])
        face = box_support(x, box)
        field = smooth_field(x, box, args.feather)

        with torch.no_grad():
            z = sd.encode_image(x)
            noise = sd.sample_edit_noise(z, args.seed)
            raw = sd.sdedit(x, emb, noise, num_steps=args.num_steps,
                            strength=args.strength, guidance_scale=args.guidance,
                            emb_uncond=uncond).clamp(0, 1).float()
            delta_lf = smooth_delta(x, raw - x, args)

            # 兩道上限交替二分：`a_face` 也會壓低整圖的色差，所以先臉後整圖、
            # 反覆三輪即可同時成立。單邊一次的寫法會讓後解的那一道把前一道
            # 推出可行域，而 CSV 上只會看到一個「已可行」。
            a_bg, a_face = 1.0, 1.0
            for _ in range(3):
                a_face = bisect(
                    lambda v: float(delta_e00(x, compose(x, delta_lf, field, a_bg, v)[0], face)),
                    args.face_cap, hi=args.max_scale)
                a_bg = bisect(
                    lambda v: float(delta_e00(x, compose(x, delta_lf, field, v, a_face)[0], frame)),
                    args.frame_cap, hi=args.max_scale)
            y, alpha = compose(x, delta_lf, field, a_bg, a_face)

            offset = lab_offset(x, y)
            id0 = embed_box(x, box, sd.device)
            id1 = embed_box(y, box, sd.device)
            id_cos = float(torch.nn.functional.cosine_similarity(
                id0.reshape(1, -1), id1.reshape(1, -1)))
            id_raw = float(torch.nn.functional.cosine_similarity(
                id0.reshape(1, -1),
                embed_box(raw, box, sd.device).reshape(1, -1)))

        row = {
            "image": item["name"], "class": item["class"], "arm": args.arm,
            "carrier": "sdedit_lowfreq_transfer", "carrier_model": CARRIER_MODEL,
            "style": args.style, "style_prompt": prompt,
            "strength": args.strength, "num_steps": args.num_steps,
            "guidance": args.guidance, "seed": args.seed,
            "smoother": args.smoother, "sigma": args.sigma,
            "radius": args.radius, "eps": args.eps,
            "frame_cap": args.frame_cap, "face_cap": args.face_cap,
            "feather": args.feather, "max_scale": args.max_scale,
            "a_bg": round(a_bg, 5), "a_face": round(a_face, 5),
            "deltaE00_frame": round(float(delta_e00(x, y, frame)), 4),
            "deltaE00_face_box": round(float(delta_e00(x, y, face)), 4),
            "tv_frame": round(float(tv_offset(offset, frame)), 5),
            "psnr": round(float(10 * torch.log10(1.0 / (y - x).pow(2).mean())), 4),
            "linf": round(float((y - x).abs().max()), 5),
            # `id_raw` 是 SDEdit 原始輸出的身分，`id_cos` 是交付圖的。
            # 兩欄並列，好看出「低通轉移」到底把換人的部分擋掉了多少。
            "id_cos": round(id_cos, 5), "id_cos_sdedit_raw": round(id_raw, 5),
            "seconds": round(time.time() - started, 1),
        }
        save_image(x, args.out / f"{item['name']}__orig.png")
        save_image(y, args.out / f"{item['name']}__{args.arm}__def.png")
        save_image(raw, args.out / f"{item['name']}__sdedit_raw.png")
        save_image(alpha.repeat(1, 3, 1, 1) / max(a_bg, a_face, 1e-6),
                   args.out / f"{item['name']}__field.png")
        rows.append(row)
        write_rows(args.out / "results.csv", rows)
        print(f"[{args.arm}] {item['name']}  a_bg {row['a_bg']} a_face {row['a_face']}  "
              f"ΔE frame {row['deltaE00_frame']} face {row['deltaE00_face_box']}  "
              f"id {row['id_cos']} (raw {row['id_cos_sdedit_raw']})  "
              f"{row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 張 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
