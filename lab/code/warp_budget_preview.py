"""`ab_warp` 族在不同色差額度下的外觀預覽：不做最佳化，只看額度放大後長什麼樣。

做法
────────────────────────────────────────────────────────────────────
每張影像抽**一個固定的隨機方向**（種子依影像固定），各額度共用同一個方向，
只改幅度——所以同一列從左到右是同一個調色往外推，差別只在額度。

    w_raw  = 單位向量 × g_i         （G×G 個 (a,b) 錨點）
    g_i    = 1 − exp(−d_i² / 2σ_s²)  d_i 為錨點到該張膚色群心的 (a,b) 距離
    亮度曲線固定為恆等：亮度一動，膚色的 ΔE00 就跟著動，整圖額度會用不到
    w      = s · w_raw               s 為幅度（Lab 單位）

`g_i` 讓膚色附近的錨點幾乎不動，模仿最佳化會做的事（臉那道上限只釘住膚色
附近的錨點）；沒有它，隨機方向會先被臉的上限卡住，整圖額度根本用不到。

每一個額度 `(整圖 F, 臉 f)` 以二分搜尋找最大的 `s`，使
整圖 ΔE00 ≤ F、臉框 ΔE00 ≤ f、同色（膚色）ΔE00 ≤ f。彩度上界**不設**，
改為照報 `chroma_p95` 對原圖的倍率，讓「高飽和」那一條看得到而不是被藏起來。
綁住的是哪一道，逐張寫進 CSV。

這是預覽，不是防禦：沒有對編輯器做任何最佳化。

用法（遠端 CPU 即可）
    CUDA_VISIBLE_DEVICES= python lab/code/warp_budget_preview.py \\
        --out lab/runs/warp_budget_preview
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402

from ab_warp_defence import AbWarpParam  # noqa: E402
from colour_support import chroma_p95, skin_colour_support  # noqa: E402
from curve_budget_defence import box_support, expanded_box, load_images  # noqa: E402
from src.defense.delta_e_torch import delta_e00_torch  # noqa: E402
from src.defense.immunise import quantise  # noqa: E402
from src.defense.ncf_param import rgb_to_lab  # noqa: E402
from src.metrics.identity import face_boxes  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

#: (整圖 ΔE00 上限, 同色 ΔE00 上限)。臉框不設上限、照報：框內含頭髮與背景，
#: 它會比膚色先卡住，實測（第一版）整圖額度因此完全用不到。
#: 第二版實測：膚色（同色）上限總是先卡住，整圖在膚色 8 下只到 7–8。額度因此沿
#: 膚色這一軸開，整圖上限放到 64（實際不會碰到），整圖 ΔE00 照報。
LEVELS = [(64, 8), (64, 12), (64, 16), (64, 24), (64, 32)]


def tag(frame, skin):
    return f"F{frame}_s{skin}"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("lab/data/portraits"))
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--extent", type=float, default=90.0)
    ap.add_argument("--pieces", type=int, default=16)
    ap.add_argument("--l-radius", type=float, default=0.6)
    ap.add_argument("--skin-sigma", type=float, default=50.0)
    ap.add_argument("--s-max", type=float, default=160.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    rows = []
    for idx, item in enumerate(load_images(args.data, None)):
        x = load_image_tensor(item["path"], device, size=512)
        boxes = face_boxes(x, device)
        if not boxes:
            raise SystemExit(f"{item['name']} 偵測不到臉")
        box = expanded_box(x, max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])))
        frame = torch.ones_like(x[:, :1])
        face = box_support(x, box)
        skin = skin_colour_support(x, face, 12.0)
        lab = rgb_to_lab(x.float())
        centre = torch.stack([(lab[:, c:c + 1] * face).sum() / face.sum() for c in (1, 2)])
        c95 = float(chroma_p95(x))

        carrier = AbWarpParam(grid=args.grid, extent=args.extent, warp_radius=1.0,
                              pieces=args.pieces, l_radius=args.l_radius)
        carrier.reset(x, 0)
        gen = torch.Generator().manual_seed(args.seed * 1000 + idx)
        # 一致的調色方向：色相旋轉 ＋ 色度平移。逐錨點獨立的隨機方向經 RBF 平均後
        # 互相抵消（第一版實測幅度推到上限、整圖 ΔE00 仍只有 13）。
        sign = 1.0 if torch.rand(1, generator=gen) < 0.5 else -1.0
        theta = sign * (25 + 35 * float(torch.rand(1, generator=gen))) * torch.pi / 180
        rot = torch.tensor([[torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
                            [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))]])
        shift = torch.randn(2, generator=gen)
        shift = 0.5 * shift / shift.norm()
        direction = carrier.anchors.cpu() @ rot.T - carrier.anchors.cpu()
        direction = direction / direction.norm(dim=-1).max() + shift
        d = (carrier.anchors - centre).norm(dim=-1, keepdim=True)
        gate = 1.0 - torch.exp(-d.pow(2) / (2 * args.skin_sigma ** 2))
        w_dir = direction * gate
        th_dir = torch.rand(args.pieces, generator=gen) * 2 - 1

        def render(s):
            carrier.w_raw = w_dir.clone()
            carrier.warp_radius = s
            carrier.th_raw = th_dir * 0.0
            return quantise(carrier.render(x))

        def measure(y):
            return {"frame": float(delta_e00_torch(x, y, frame)),
                    "face": float(delta_e00_torch(x, y, face)),
                    "skin": float(delta_e00_torch(x, y, skin))}

        save_image(x, args.out / f"{item['name']}__orig.png")
        for F, f in LEVELS:
            caps = {"frame": F, "skin": f}
            ok = lambda s: all(measure(render(s))[k] <= v for k, v in caps.items())
            lo, hi = 0.0, args.s_max
            if ok(hi):
                lo = hi
            else:
                for _ in range(18):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if ok(mid) else (lo, mid)
            y = render(lo)
            m = measure(y)
            bind = max(caps, key=lambda k: m[k] / caps[k])
            row = {"image": item["name"], "level": tag(F, f), "frame_cap": F, "skin_cap": f,
                   "amplitude": round(lo, 3), "binding": bind if lo < args.s_max else "s_max",
                   "deltaE00_frame": round(m["frame"], 3), "deltaE00_face": round(m["face"], 3),
                   "deltaE00_skin": round(m["skin"], 3), "hue_rotation_deg": round(theta * 180 / torch.pi, 1),
                   "chroma_p95_ratio": round(float(chroma_p95(y)) / c95, 3),
                   "psnr": round(float(-10 * torch.log10((y - x).pow(2).mean())), 3)}
            rows.append(row)
            save_image(y, args.out / f"{item['name']}__{tag(F, f)}.png")
            print(row, flush=True)
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
