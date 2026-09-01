"""殘差有沒有對準影像內容：H1 的機制那一半。

`docs/reference/SURVEY_ARCHITECTURE.md` 第三節的 H1 說「DCT-Shield 的效果是
**統計驅動**的、本方法的是**實現驅動**的」。`signature.csv` 已量到兩者的空間
集中度差四到六倍（`block_gini` 0.43–0.65 對 0.10–0.11），但**集中不等於對準**
——一個把能量集中在隨機幾塊的擾動同樣會有高 Gini，而它並不是實現特定的。

這支腳本量的就是缺的那一半：**逐區塊的殘差能量與該區塊的影像紋理之間的
相關**。高相關表示殘差是照著這一張圖的內容鋪的（實現特定）；相關接近零表示
殘差鋪在哪裡與這張圖長什麼樣無關（統計驅動）。

再加一個對照：把殘差做一次 32×32 區塊置換之後重算同一個相關。置換保留每個
區塊的內容、只換位置，所以它**必定**把對準破壞掉——這一欄是「對準完全消失」
時該讀到什麼的參照線，讓另外兩欄有尺度可比。

**不跑 GPU、不需要擴散模型**，只讀 `*__def.png` 與原圖。與
`scripts/residual_signature.py` 同一組輸入、同一組 32×32 區塊基底。

用法：

    python scripts/residual_texture_alignment.py \\
        --run ours_add=runs/ip2p_axis_necessity/b_pg_r20 \\
              dct_e18=runs/ip2p_dct_band_extend/dct_e18 \\
        --out runs/residual_texture_alignment
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
BLOCK = 32          # 與 residual_signature.py 的分析基底相同


def block_energy(t: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """(1,C,H,W) → (nh*nw,)，每個方格的能量。"""
    e = t.pow(2).sum(dim=1, keepdim=True)
    return F.avg_pool2d(e, block).flatten()


def block_texture(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """逐方格的紋理量：梯度能量。

    取梯度能量而不是結構張量的 coherence，是因為 coherence 專挑**邊緣**
    （單一方向），而本方法的紋理閘正好在 coherence 高處歸零。要問的是
    「殘差有沒有跟著內容走」，梯度能量是這句話最直接、也最少預設的量。
    """
    g = x.mean(dim=1, keepdim=True)
    dx = g[..., :, 1:] - g[..., :, :-1]
    dy = g[..., 1:, :] - g[..., :-1, :]
    gx = F.pad(dx, (0, 1, 0, 0))
    gy = F.pad(dy, (0, 0, 0, 1))
    return F.avg_pool2d(gx.pow(2) + gy.pow(2), block).flatten()


def _ranks(v: torch.Tensor) -> torch.Tensor:
    """平均等第（同分取平均）。

    **同分一定要取平均，不能用 `argsort().argsort()`。** 後者給同分的是
    索引序的等第，而索引序在影像上就是列優先的空間順序——完全平坦的方格
    （梯度能量**恰為** 0，剪裁過的天空就是這樣）會被排成一條與位置相關的
    序列，而殘差能量也有空間結構，於是憑空生出相關。真實影像上這種方格
    不罕見，所以這不是理論上的顧慮。
    """
    order = v.argsort()
    raw = torch.empty(v.numel(), dtype=torch.float64)
    raw[order] = torch.arange(v.numel(), dtype=torch.float64)
    uniq, inv, counts = torch.unique(v, return_inverse=True, return_counts=True)
    sums = torch.zeros(uniq.numel(), dtype=torch.float64).index_add_(0, inv, raw)
    return (sums / counts.to(torch.float64))[inv]


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """等第相關。用等第而不是 Pearson：兩個量的分布都極度右偏
    （少數方格帶走大部分能量），Pearson 會被那幾個方格決定。"""
    ra = _ranks(a)
    rb = _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if float(denom) == 0.0:
        raise ValueError("等第全部相同，相關無定義——輸入可能是常數")
    return float((ra * rb).sum() / denom)


def crop_ring_fraction(r: torch.Tensor,
                       fraction: float = 0.10) -> float:
    """裁切會丟掉的那一圈佔殘差總能量的比例。

    `SURVEY_ARCHITECTURE` 第三節的 H4（集中度在裁切下的變異數）被標為「弱、
    沒有新增證據」，缺的就是這個量：均勻鋪開的擾動損失的是**面積比例**這個
    定值，集中的擾動損失的是**隨機的那一圈**，逐圖變異大得多。所以要看的是
    這一欄的**標準差**，不只是平均。

    幾何取 `src/purify/ops.py` 的 `crop_resize`：`fraction` 是**每邊**各裁掉
    的邊長比例，故保留中央 `(1-2f)` 的邊長。f = 0.10 時保留 410/512 的邊長、
    面積比 0.641——**丟掉的是 35.9% 的面積，不是 19%**。
    """
    h, w = r.shape[-2:]
    dh, dw = int(round(h * fraction)), int(round(w * fraction))
    total = float(r.pow(2).sum())
    if total == 0.0:
        raise ValueError("殘差能量為零，比例無定義")
    kept = float(r[..., dh:h - dh, dw:w - dw].pow(2).sum())
    return (total - kept) / total


def permute_block_energy(e: torch.Tensor, seed: int) -> torch.Tensor:
    """把方格能量重排。等價於把殘差做區塊置換之後再算能量。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return e[torch.randperm(e.numel(), generator=g).to(e.device)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True,
                    help="tag=目錄，可給多組")
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--permute-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dataset = {d["name"]: d for d in load_dataset(args.data)}
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    rows = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run 要寫成 tag=目錄，收到 {spec!r}")
        tag, d = spec.split("=", 1)
        run = Path(d)
        with (run / "results.csv").open(encoding="utf-8") as fh:
            entries = [(r["image"], r["condition"]) for r in csv.DictReader(fh)]
        if not entries:
            raise SystemExit(f"{run / 'results.csv'} 是空的")
        for name, cond in entries:
            png = run / f"{name}__{cond}__def.png"
            if not png.exists():
                raise FileNotFoundError(f"缺少防禦圖 {png}")
            x = load_image_tensor(dataset[name]["path"], device, size=RESOLUTION)
            x_def = load_image_tensor(png, device, size=RESOLUTION)
            r = x_def - x
            e = block_energy(r)
            t = block_texture(x)
            rows.append({
                "tag": tag, "image": name, "condition": cond, "block": BLOCK,
                "rho_energy_texture": round(spearman(e, t), 5),
                "rho_after_permute": round(
                    spearman(permute_block_energy(e, args.permute_seed), t), 5),
                # 與 signature.csv 同定義的集中度，放在同一列好對讀：
                # 集中而不對準、與集中且對準，是兩件事。
                "block_gini": round(_gini(e), 5),
                # 裁切丟掉的那一圈佔多少能量。H4 要看的是它的**標準差**：
                # 均勻鋪開損失的是面積比這個定值，集中的損失的是隨機的一圈。
                "crop_ring_energy_frac": round(crop_ring_fraction(r), 5),
                "residual_rms": round(float(r.pow(2).mean().sqrt()), 6),
            })
            print(f"{tag:10s} {name:34s} rho={rows[-1]['rho_energy_texture']:+.4f} "
                  f"置換後 {rows[-1]['rho_after_permute']:+.4f} "
                  f"gini={rows[-1]['block_gini']:.4f} "
                  f"裁切圈 {rows[-1]['crop_ring_energy_frac']:.4f}", flush=True)
        write_csv(args.out / "alignment.csv", rows)

    print("\n逐條件的平均（13 張）：")
    print(f"{'條件':>10s} {'ρ(能量, 紋理)':>14s} {'置換後':>10s} {'block_gini':>11s} "
          f"{'裁切圈平均':>11s} {'裁切圈 sd':>10s}")
    for tag in dict.fromkeys(r["tag"] for r in rows):
        sel = [r for r in rows if r["tag"] == tag]
        ring = [r["crop_ring_energy_frac"] for r in sel]
        print(f"{tag:>10s} {statistics.fmean(r['rho_energy_texture'] for r in sel):14.4f} "
              f"{statistics.fmean(r['rho_after_permute'] for r in sel):10.4f} "
              f"{statistics.fmean(r['block_gini'] for r in sel):11.4f} "
              f"{statistics.fmean(ring):11.4f} {statistics.stdev(ring):10.4f}")
    print()
    print("參照：`crop_resize(0.10)` 每邊各裁 10%，保留中央 410/512 的邊長，"
          "丟掉的面積是 35.9%。")
    print(f"\n表：{args.out / 'alignment.csv'}（{len(rows)} 列）")


def _gini(v: torch.Tensor) -> float:
    s = v.sort().values
    n = s.numel()
    idx = torch.arange(1, n + 1, dtype=s.dtype)
    total = s.sum()
    if float(total) == 0.0:
        raise ValueError("能量全為零，Gini 無定義")
    return float((2 * idx - n - 1).mul(s).sum() / (n * total))


if __name__ == "__main__":
    main()
