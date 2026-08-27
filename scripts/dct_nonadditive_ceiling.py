"""DCT 域非加性擾動的構造天花板：θ 要取多少才落得進失真帶。**不跑 GPU。**

模式與 `advdrop_ceiling.py`／`dct_rotation_ceiling.py` 相同：**不最佳化**，
把強度推到上界（`plane`／`shared_plane` 是每個區塊都轉滿 ±θ、平面隨機抽；
`gain` 是每個係數都乘 `exp(±g)`），量它的失真。那是該上界下最極端的可行解，
給的是天花板；最佳化只會落在它底下。

**這一支是派工前的必要步驟**：`matched_distortion_table.py` 拒絕外插，θ 猜錯
就是整批 GPU 時間白花（`warp_triad.sh` 踩過一次）。

判讀規則（跑之前就寫下）
────────────────────────────────────────────────────────────────────
N1  `plane` 在 θ = π 下的十張平均 DISTS **構不到 0.1286**（失真帶下界），
    這一族在保長的限制下到不了工作點，整條路收掉——連 GPU 都不用跑。
N2  `clip_fraction` 在帶內工作點**超過 0.10**，則「保長」在像素域已經名存實亡
    （一成的像素被值域夾掉），等失真比較必須加註，不可以宣稱保長。
N3  `plane` 與 `shared_plane` 的天花板**差不到 10%**，則逐區塊那 52 萬個參數
    沒有買到可達範圍，`plane` 的容量優勢是假的——該用 `shared_plane` 當主線
    （便宜兩個數量級），並把「容量不夠」從失敗解釋裡刪掉。

用法：
    python scripts/dct_nonadditive_ceiling.py --out runs/dct_nonadd/ceiling.csv
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.dct_nonadditive import DctNonAdditiveParam  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
THETAS = (0.1, 0.2, 0.4, 0.7, 1.1, 1.6, 2.2, math.pi)
BAND_LO, BAND_HI = 0.1286, 0.1447


@torch.no_grad()
def _align_planes(p: DctNonAdditiveParam, x01: torch.Tensor) -> None:
    """把每個區塊的 e1 對齊它自己的係數向量，v 取一個正交的隨機方向。

    這才是保長映射的**天花板**：`‖Δc‖ = 2‖Pc‖·sin(θ/2)`，而 `‖Pc‖ ≤ ‖c‖`，
    等號在 e1 ∥ c 時成立。隨機平面只抓得到 2/63 的能量，量出來的是隨機解
    不是天花板——這個區別在第一版探針裡搞錯過。

    `shared_plane` 不適用（全域共用一個平面，沒有「對齊每個區塊」這回事），
    故只處理 `mode == "plane"`。
    """
    from src.baselines.jpeg_codec import block_dct
    from src.defense.dct_nonadditive import CHANNELS, to_planes

    planes = to_planes(x01)
    rows = torch.tensor([u for u, _ in p.idx], device=x01.device)
    cols = torch.tensor([v for _, v in p.idx], device=x01.device)
    for i, name in enumerate(CHANNELS):
        if name not in p.params_:
            continue
        coef = block_dct(planes[i], p._d)[..., rows, cols]   # (N,hb,wb,k)
        d = p.params_[name]
        d["u"].copy_(coef / coef.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--modes", nargs="+",
                    default=["plane", "shared_plane", "gain"])
    ap.add_argument("--gates", nargs="+", default=["texture", "band"])
    ap.add_argument("--thetas", type=float, nargs="+", default=list(THETAS))
    ap.add_argument("--plane-init", default="aligned",
                    choices=("aligned", "random"),
                    help="`plane`／`shared_plane` 的平面怎麼取。"
                         "**aligned 才是天花板**：把 e1 對齊該區塊自己的係數"
                         "向量 c/‖c‖，此時 ‖Pc‖ = ‖c‖，位移達到保長映射的"
                         "上限 2‖c‖·sin(θ/2)。random 是**隨機平面**，在 63 維"
                         "裡平均只抓得到 2/63 ≈ 3.2% 的區塊能量，位移因此被"
                         "低估 sqrt(63/2) ≈ 5.6 倍——那不是天花板，是隨機解。")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    device = torch.device(args.device)
    suite = MetricSuite(device=device)

    rows = []
    kind = "（天花板）" if args.plane_init == "aligned" else "（隨機解，非天花板）"
    print("平面取法：" + args.plane_init + kind)
    print()
    for name in names:
        x = load_image_tensor(args.data / name / f"{name}.png", device,
                              size=RESOLUTION)
        for mode in args.modes:
            for gate in args.gates:
                for th in args.thetas:
                    p = DctNonAdditiveParam(radius=th, mode=mode, gate=gate)
                    p.reset(x, seed=args.seed)
                    key = "g" if mode == "gain" else "theta"
                    if mode == "plane" and args.plane_init == "aligned":
                        _align_planes(p, x)
                    with torch.no_grad():
                        for d in p.params_.values():
                            # 每一格都推到上界，正負號隨機——那是該上界下最極端
                            # 的可行解。正負號的樣式對 L2 失真沒有影響，但 DISTS
                            # 量的是紋理統計量，故仍固定 seed 以便重跑。
                            g = torch.Generator().manual_seed(args.seed)
                            s = (torch.randint(0, 2, d[key].shape, generator=g)
                                 .to(device=d[key].device, dtype=d[key].dtype)
                                 * 2.0 - 1.0)
                            d[key].copy_(s * th)
                    y = p.render(x)
                    m = suite.pairwise(x.float(), y.float())
                    rows.append({
                        "image": name, "mode": mode, "gate": gate,
                        "theta": th,
                        "dists": round(m["dists"], 6),
                        "lpips": round(m["lpips"], 6),
                        "psnr": round(m["psnr"], 4),
                        "ssim": round(m["ssim"], 6),
                        "rms": round(m["rms"], 6),
                        "linf": round(m["linf"], 6),
                        "clip_fraction": round(p.clip_fraction(), 6),
                        "degenerate_fraction": round(p.degenerate_fraction(), 6),
                        "plane_init": args.plane_init,
                        "n_params": sum(t.numel() for t in p.params()),
                    })
        print(f"  {name} 完成", flush=True)

    write_csv(args.out, rows)
    print(f"\n{len(names)} 張 → {args.out}")
    print(f"失真帶 DISTS {BAND_LO}–{BAND_HI}\n")
    for mode in args.modes:
        for gate in args.gates:
            sub = [r for r in rows if r["mode"] == mode and r["gate"] == gate]
            if not sub:
                continue
            line = f"{mode:>13} / {gate:<8}"
            for th in args.thetas:
                s2 = [r for r in sub if r["theta"] == th]
                line += f"  {th:.2f}:{st.fmean(r['dists'] for r in s2):.4f}"
            print(line)
    print("\n夾取比例（帶內判讀用，N2 的門檻是 0.10）")
    for mode in args.modes:
        sub = [r for r in rows if r["mode"] == mode and r["gate"] == "texture"]
        if not sub:
            continue
        line = f"{mode:>13}"
        for th in args.thetas:
            s2 = [r for r in sub if r["theta"] == th]
            line += f"  {th:.2f}:{st.fmean(r['clip_fraction'] for r in s2):.4f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
