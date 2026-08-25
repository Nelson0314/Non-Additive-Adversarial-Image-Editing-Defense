"""AdvDrop 的量化表到底走到哪裡：eps 沒有咬到的根因診斷。

`runs/ip2p_advdrop_band` 掃了 eps = 60／100／150／220 四點，四點的失真與位移
**幾乎逐位相同**（DISTS 0.0111–0.0114、位移 0.108–0.112、擋下 0/13）。
eps 在論文模式下決定的是可動區間 `q ∈ [q_init, q_init + eps]`，四個值差三倍多
卻沒有任何差異，只有一個解釋：**量化表根本沒有走到夾取邊界**，所以邊界在哪
不影響結果。

這支腳本把 `run_advdrop` 的內圈攤開，逐步記下量化表的統計量與損失，回答：

- `q` 有沒有在爬？爬到多少就停？
- 梯度的符號是一致的還是在震盪？
- 損失有沒有真的下降？

**不是要修 AdvDrop**，是要判定「AdvDrop 在這個威脅模型上沒有效果」是**它的
性質**還是**我們的移植或設定有問題**。兩者的結論完全不同：前者是可報告的
陰性結果，後者是缺陷。

用法（一張卡，一張影像，約兩分鐘）：

    python scripts/advdrop_q_trajectory.py --image task_attr_mod_color_11699 \\
        --eps 220 --out runs/ip2p_advdrop_band/q_trajectory.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from src.baselines.advdrop import (  # noqa: E402
    CHANNELS, PAPER_Q_INIT, alpha_at, init_q_tables, render_advdrop,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--image", required=True)
    ap.add_argument("--eps", type=float, default=220.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--step-size", type=float, default=4.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from src.models.ip2p import IP2PWrapper

    sd = IP2PWrapper(dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    item = {d["name"]: d for d in load_dataset(args.data)}[args.image]
    x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)

    lo, hi, init = PAPER_Q_INIT, PAPER_Q_INIT + args.eps, PAPER_Q_INIT
    q = init_q_tables(x01, init)

    rows = []
    for i in range(args.steps):
        a = torch.tensor(alpha_at(i, args.steps), device=x01.device,
                         dtype=x01.dtype)
        for t in q.values():
            t.requires_grad_(True)
        x_adv = render_advdrop(x01, q, a, "rgb", False)
        loss = sd.encode_image(x_adv).pow(2).mean()
        grads = torch.autograd.grad(loss, [q[c] for c in CHANNELS],
                                    allow_unused=True, materialize_grads=True)
        with torch.no_grad():
            g_all = torch.cat([g.flatten() for g in grads])
            q_all = torch.cat([q[c].detach().flatten() for c in CHANNELS])
            m = suite.pairwise(x01, x_adv.detach().clamp(0, 1))
            rows.append({
                "step": i, "alpha": round(float(a), 8),
                "loss": round(float(loss.detach()), 8),
                # 量化表的位置：走到哪、有沒有貼上邊界。
                "q_mean": round(float(q_all.mean()), 4),
                "q_max": round(float(q_all.max()), 4),
                "q_frac_at_hi": round(float((q_all >= hi - 1e-6).float().mean()), 5),
                "q_frac_at_lo": round(float((q_all <= lo + 1e-6).float().mean()), 5),
                # 梯度的符號分布：一面倒表示在爬，各半表示在震盪。
                "grad_frac_negative": round(float((g_all < 0).float().mean()), 5),
                "grad_abs_mean": float(g_all.abs().mean()),
                "grad_frac_zero": round(float((g_all == 0).float().mean()), 5),
                "dists": round(m["dists"], 6),
                "rms": round(m["rms"], 6),
            })
            for c, g in zip(CHANNELS, grads):
                q[c] = (q[c] - args.step_size * torch.sign(g)).clamp(
                    lo, hi).detach().requires_grad_(True)
        if i % 10 == 0 or i == args.steps - 1:
            r = rows[-1]
            print(f"step {i:3d}  alpha={r['alpha']:.5f}  loss={r['loss']:.6f}  "
                  f"q 平均 {r['q_mean']:7.3f} 最大 {r['q_max']:7.3f}  "
                  f"貼上界 {r['q_frac_at_hi']:.3f} 貼下界 {r['q_frac_at_lo']:.3f}  "
                  f"梯度<0 {r['grad_frac_negative']:.3f} 梯度=0 {r['grad_frac_zero']:.3f}  "
                  f"DISTS {r['dists']:.5f}", flush=True)

    write_csv(args.out, rows)
    first, last = rows[0], rows[-1]
    print()
    print(f"損失 {first['loss']:.6f} → {last['loss']:.6f}"
          f"（{'下降' if last['loss'] < first['loss'] else '沒有下降'}）")
    print(f"量化表 平均 {first['q_mean']:.3f} → {last['q_mean']:.3f}，"
          f"上界是 {hi:.1f}")
    print(f"最後一步貼上界的比例 {last['q_frac_at_hi']:.4f}"
          f"，貼下界 {last['q_frac_at_lo']:.4f}")
    if last["q_frac_at_hi"] < 0.01:
        print("→ **eps 沒有咬到**：量化表沒有走到上界，改 eps 不會改變任何東西。")
    print(f"表：{args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
