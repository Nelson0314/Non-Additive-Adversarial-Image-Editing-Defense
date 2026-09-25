"""`ab_warp` 族的同族隨機對照（`docs/NEXT_PLAN.md` B4）。

函數類與上限與 `ab_warp_ch` 逐項相同（`AbWarpParam`、grid 7、extent 90、warp_radius 80、
pieces 16；整圖 32、臉框／同色 16、彩度 2.0×、五道分通道位移 p95）。

抽樣：候選 `j`（影像序 `i`）的種子為 `100000 + 10000·i + j`；先在
`{0.03, 0.1, 0.3, 1, 3}` 等機率選 σ，再抽 `w_raw ~ N(0, σ²)`（投影到範數 ≤ 1）與
`th_raw ~ N(0, σ²)`（夾到 [−1, 1]）。

接受：全部色彩上限在量化後成立，且輸入 LPIPS 落在參照臂該張**實際達到**的值
± `--lpips-tolerance`。依 j 順序取前 `--replicates` 個，輸出為
`{out}/{prefix}_r{k}/{影像}__{prefix}_r{k}__def.png` 與各自的 `results.csv`，
可直接當成臂名跑 `arm_chain.sh`。每張最多 `--max-candidates` 個候選；不足時照報
實際數量，不放寬上限。全部候選的種子、σ、LPIPS、各上限值與拒絕原因寫進
`{out}/{prefix}_log/candidates.csv`。不看目標函數或編輯結果。
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

from ab_warp_defence import CHANNEL_CAPS, AbWarpParam, channel_shift_p95  # noqa: E402
from colour_support import chroma_p95, skin_colour_support  # noqa: E402
from curve_budget_defence import box_support, expanded_box, load_images, write_rows  # noqa: E402
from src.defense.delta_e_torch import delta_e00_torch  # noqa: E402
from src.defense.immunise import quantise  # noqa: E402
from src.metrics.identity import face_boxes  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

IMAGE_ORDER = [f"{g}_{i:02d}" for g in ("man", "woman") for i in range(4)]
STDS = (0.03, 0.1, 0.3, 1.0, 3.0)
CH_CAPS = {"a_pos": 4, "a_neg": 15, "b_pos": 4, "b_neg": 25, "l_abs": 15}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-arm", required=True, help="對齊其逐張實際 LPIPS 的最佳化臂")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", type=Path, default=Path("lab/runs/defence"))
    ap.add_argument("--data", type=Path, default=Path("lab/data/portraits"))
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--max-candidates", type=int, default=4096)
    ap.add_argument("--lpips-tolerance", type=float, default=0.0025)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU（piq LPIPS）")
    import piq
    device = torch.device("cuda")
    lp = piq.LPIPS().to(device).eval()
    lp.requires_grad_(False)
    ref = {r["image"]: float(r["lpips_out"]) for r in
           csv.DictReader(open(args.out / args.ref_arm / "results.csv", encoding="utf-8"))}
    logdir = args.out / f"{args.prefix}_log"
    logdir.mkdir(parents=True, exist_ok=True)
    log, reps = [], {k: [] for k in range(1, args.replicates + 1)}
    for item in load_images(args.data, None):
        name, i = item["name"], IMAGE_ORDER.index(item["name"])
        x = load_image_tensor(item["path"], device, size=512)
        box = expanded_box(x, max(face_boxes(x, device), key=lambda q: (q[2]-q[0]) * (q[3]-q[1])))
        frame, face = torch.ones_like(x[:, :1]), box_support(x, box)
        skin = skin_colour_support(x, face, 12.0)
        c95 = float(chroma_p95(x)) * 2.0
        target = ref[name]
        c = AbWarpParam(grid=7, extent=90.0, warp_radius=80.0, pieces=16, l_radius=0.6)
        c.reset(x, 0)
        accepted = 0
        for j in range(args.max_candidates):
            g = torch.Generator(device="cpu").manual_seed(100000 + 10000 * i + j)
            sd = STDS[int(torch.randint(len(STDS), (1,), generator=g))]
            with torch.no_grad():
                c.w_raw.copy_((sd * torch.randn(c.w_raw.shape, generator=g)).to(device))
                c.th_raw.copy_((sd * torch.randn(c.th_raw.shape, generator=g)).to(device))
                c.project()
                y = quantise(c.render(x))
                m = {"frame": float(delta_e00_torch(x, y, frame)), "face_box": float(delta_e00_torch(x, y, face)),
                     "skin_colour": float(delta_e00_torch(x, y, skin)), "chroma_p95": float(chroma_p95(y)),
                     **{f"shift_{n}": float(channel_shift_p95(x, y, ch, sg)) for n, ch, sg in CHANNEL_CAPS},
                     "lpips": float(lp(y, x).mean())}
            lim = {"frame": 32.0, "face_box": 16.0, "skin_colour": 16.0, "chroma_p95": c95,
                   **{f"shift_{n}": CH_CAPS[n] for n, _, _ in CHANNEL_CAPS}}
            reasons = [k for k, v in lim.items() if not m[k] <= v]
            if abs(m["lpips"] - target) > args.lpips_tolerance:
                reasons.append("lpips_band")
            ok = not reasons
            log.append({"image": name, "j": j, "seed": 100000 + 10000 * i + j, "std": sd,
                        "lpips_target": round(target, 6), **{k: round(v, 5) for k, v in m.items()},
                        "feasible": int(ok), "rejection_reason": ";".join(reasons)})
            if ok:
                accepted += 1
                arm = f"{args.prefix}_r{accepted}"
                (args.out / arm).mkdir(parents=True, exist_ok=True)
                save_image(x, args.out / arm / f"{name}__orig.png")
                save_image(y, args.out / arm / f"{name}__{arm}__def.png")
                torch.save(c.state_dict(), args.out / arm / f"{name}__carrier.pt")
                reps[accepted].append({"image": name, "arm": arm, "j": j, "std": sd, **m,
                                       "lpips_target": target})
                write_rows(args.out / arm / "results.csv", reps[accepted])
                if accepted == args.replicates:
                    break
        write_rows(logdir / "candidates.csv", log)
        print(f"[{args.prefix}] {name}: accepted {accepted}/{args.replicates} "
              f"after {j + 1} candidates (target LPIPS {target:.4f})", flush=True)


if __name__ == "__main__":
    main()
