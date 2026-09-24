"""生成模型只提供「目標風格」，交付的是 `ab_warp` 那一族的全域色彩映射。

為什麼是這條路
────────────────────────────────────────────────────────────────────
`style_affine` 與 `style_opt` 取 SDEdit 輸出的色彩，但交付時經過局部仿射轉移
與臉框長出的平滑場，產物上留下不均勻的色度／亮度塊與一圈臉框邊界。
`ab_warp` 的映射是全域的（輸出只依賴該像素自己的顏色），產物自然。

本檔把兩者接起來：SDEdit 的風格輸出只當**擬合目標**，求一個 `AbWarpParam`
（CIELAB `(a,b)` 平面的 RBF 位移 ＋ 單調亮度曲線）使映射後的原圖在 Lab 上最接近
那張風格圖。結構逐像素來自原圖；位置、臉框、遮罩都不進 render。

    min_θ  mean ‖Lab(W_θ(x)) − Lab(s)‖² / 100²
    s.t.   整圖 ΔE00 ≤ 16、臉框 ΔE00 ≤ 8、同色 ΔE00 ≤ 8、彩度 p95 ≤ 1.15×

約束與 `ab_warp` 逐項相同，走同一個增廣 Lagrange（`optimise_carrier`），
不做事後縮放。兩張圖先以 4×4 平均池化到 128²，擬合的是色彩對應而不是
SDEdit 改掉的結構細節。

**這一臂沒有對編輯器做最佳化**，它回答的是「生成模型選出的風格，以全域映射
交付，推得動多少」。風格圖取自 `style_affine` 已產出的 `__sdedit_raw.png`
（`film`、strength 0.6、20 步、CFG 7.5），不重跑擴散模型。

產出（版面與 `defence_run.py` 相同）
────────────────────────────────────────────────────────────────────
    {out}/{image}__orig.png
    {out}/{image}__{arm}__def.png
    {out}/{image}__warp.png          (a,b) 平面的位移
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
import torch.nn.functional as F  # noqa: E402

from ab_warp_defence import AbWarpParam, warp_picture  # noqa: E402
from colour_support import chroma_p95, skin_colour_support  # noqa: E402
from curve_budget_defence import (  # noqa: E402
    box_support, expanded_box, load_images, write_rows,
)
from src.defense.color_amplitude import delta_e00  # noqa: E402
from src.defense.delta_e_torch import delta_e00_torch  # noqa: E402
from src.defense.immunise import Cap, optimise_carrier, quantise  # noqa: E402
from src.defense.ncf_param import rgb_to_lab  # noqa: E402
from src.metrics.identity import face_boxes  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512


class StyleFit:
    """目標：映射後的原圖與風格圖在 Lab 上的均方差（池化到 128²）。"""

    def __init__(self, style01: torch.Tensor, pool: int = 4):
        self.pool = pool
        self.target = rgb_to_lab(F.avg_pool2d(style01.clamp(0, 1), pool))

    def terms(self, y01: torch.Tensor) -> dict:
        lab = rgb_to_lab(F.avg_pool2d(y01.clamp(0, 1), self.pool))
        return {"fit": (lab - self.target).pow(2).sum(1).mean() / 100.0 ** 2}

    def score(self, y01: torch.Tensor) -> torch.Tensor:
        return self.terms(y01)["fit"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--style-root", type=Path,
                    default=Path("lab/runs/defence/style_affine"),
                    help="風格圖所在目錄，檔名 {image}__sdedit_raw.png")
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--extent", type=float, default=90.0)
    ap.add_argument("--warp-radius", type=float, default=30.0)
    ap.add_argument("--pieces", type=int, default=16)
    ap.add_argument("--l-radius", type=float, default=0.6)
    ap.add_argument("--frame-cap", type=float, default=16.0)
    ap.add_argument("--face-cap", type=float, default=8.0)
    ap.add_argument("--skin-radius", type=float, default=12.0)
    ap.add_argument("--chroma-gain", type=float, default=1.15)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lr-final-ratio", type=float, default=0.2)
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--lam-every", type=int, default=5)
    ap.add_argument("--check-every", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--noise-seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("style_warp 需要 GPU（CPU 上 900 次 ΔE00 太慢，不要靜默退回）")
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    items = load_images(args.data, set(args.images) if args.images else None)

    rows = []
    for item in items:
        started = time.time()
        style_path = args.style_root / f"{item['name']}__sdedit_raw.png"
        if not style_path.is_file():
            raise SystemExit(f"找不到風格圖 {style_path}")
        x = load_image_tensor(item["path"], device, size=RESOLUTION)
        style = load_image_tensor(style_path, device, size=RESOLUTION)
        boxes = face_boxes(x, device)
        if not boxes:
            raise SystemExit(f"{item['name']} 偵測不到臉；膚色群心定不出來")
        box = expanded_box(x, max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])))
        frame = torch.ones_like(x[:, :1])
        face = box_support(x, box)
        skin = skin_colour_support(x, face, args.skin_radius)
        c95 = float(chroma_p95(x))
        chroma_cap = c95 * args.chroma_gain

        carrier = AbWarpParam(grid=args.grid, extent=args.extent,
                              warp_radius=args.warp_radius, pieces=args.pieces,
                              l_radius=args.l_radius)
        carrier.reset(x, args.noise_seed)
        caps = [
            Cap("frame", lambda y: delta_e00_torch(x, y, frame),
                lambda y: delta_e00(x, y, frame), args.frame_cap),
            Cap("face_box", lambda y: delta_e00_torch(x, y, face),
                lambda y: delta_e00(x, y, face), args.face_cap),
            Cap("skin_colour", lambda y: delta_e00_torch(x, y, skin),
                lambda y: delta_e00(x, y, skin), args.face_cap),
            Cap("chroma_p95", lambda y: chroma_p95(y),
                lambda y: float(chroma_p95(y)), chroma_cap),
        ]
        objective = StyleFit(style)
        with torch.no_grad():
            fit_identity = float(objective.score(x))
        stats = optimise_carrier(
            carrier, x, objective, steps=args.steps, lr=args.lr, caps=caps,
            rho=args.rho, lam_every=args.lam_every, check_every=args.check_every,
            log_every=args.log_every, lr_final_ratio=args.lr_final_ratio,
            probe_every=0)
        if stats["free_amplitude_shrink"] != 1.0:
            raise RuntimeError(f"{item['name']}：可行點靠事後縮放取得，本臂不接受")

        with torch.no_grad():
            y = quantise(carrier.render(x))
            row = {
                "image": item["name"], "class": item["class"], "arm": args.arm,
                "carrier": carrier.name, "style_source": str(style_path),
                "grid": args.grid, "extent": args.extent,
                "warp_radius": args.warp_radius, "pieces": args.pieces,
                "l_radius": args.l_radius, "solver_steps": args.steps, "lr": args.lr,
                "frame_cap": args.frame_cap, "face_cap": args.face_cap,
                "skin_radius": args.skin_radius, "chroma_gain": args.chroma_gain,
                "fit_identity": round(fit_identity, 6),
                "fit_end": round(float(objective.score(y)), 6),
                "chroma_p95_orig": round(c95, 4),
                "chroma_p95_cap": round(chroma_cap, 4),
                "chroma_p95_out": round(float(chroma_p95(y)), 4),
                "deltaE00_frame": round(float(delta_e00(x, y, frame)), 4),
                "deltaE00_face_box": round(float(delta_e00(x, y, face)), 4),
                "deltaE00_skin_colour": round(float(delta_e00(x, y, skin)), 4),
                "deltaE00_style_vs_orig": round(float(delta_e00(x, style, frame)), 4),
                "psnr": round(float(10 * torch.log10(1.0 / (y - x).pow(2).mean())), 4),
                "linf": round(float((y - x).abs().max()), 5),
                "seconds": round(time.time() - started, 1),
                **{k: v for k, v in stats.items()},
            }
        save_image(x, args.out / f"{item['name']}__orig.png")
        save_image(y, args.out / f"{item['name']}__{args.arm}__def.png")
        save_image(warp_picture(carrier, device), args.out / f"{item['name']}__warp.png")
        rows.append(row)
        write_rows(args.out / "results.csv", rows)
        print(f"[{args.arm}] {item['name']}  fit {row['fit_identity']} → {row['fit_end']}  "
              f"ΔE frame {row['deltaE00_frame']} face {row['deltaE00_face_box']}  "
              f"違反 {row['free_cap_violations']}  {row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 張 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
