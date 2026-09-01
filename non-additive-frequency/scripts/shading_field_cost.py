"""候選二 步驟 0：乘性明暗場與等 RMS 低頻加性場的失真對照。

`docs/reference/SURVEY_ARCHITECTURE.md` 候選二第 4 點指出，該候選唯一的理論
依據是「乘性形式與加性形式在同一頻帶的感知代價不同」——乘性場保持局部對比度
比值、在暗部自動縮小絕對變化量，而人類視覺對「照明」有恆常性折扣，對「加在
亮度上的低頻雜訊」則沒有。同一節逐字寫著**這個依據目前只有理論，專案內沒有
量測**。

本檔就是那個量測。第 6 點步驟 0 的判準：

    若乘性在等 RMS 下的 DISTS 不明顯低於加性（例如低不到 1.5 倍），
    本候選唯一的理論依據不成立，當場否決，不必上 GPU。

為什麼這一步值得先做：低頻的失真效率比中高頻差兩個數量級（`RESULTS.md` 的
頻帶表，位移／DISTS 為 330 對 24772），`放行低頻（r_min 調低）` 已因此被否決。
候選二要成立，舉證責任在「乘性比加性便宜」這一句上。

**不跑最佳化、不需要防禦圖、不需要擴散模型。** 只生成隨機場並量失真。

對照的設計（成對，同一個實現）：

    低頻場   f = 雙三次上採樣(16×16 的高斯雜訊)，正規化成單位標準差
    乘性     x · exp(s·f)        再 clamp 到 [0, 1]
    加性     x + c·f             再 clamp 到 [0, 1]

`s` 與 `c` 各自用二分法解到**殘差的 RMS 等於同一個目標值**，所以兩邊唯一的
差別是「乘上去」還是「加上去」，不是強度。RMS 一律在 clamp **之後**量，否則
兩邊被 clamp 吃掉的量不同，等 RMS 就不成立。

明暗場取**單通道（消色差）**：逐通道的彩色明暗場會撞上已否決的 `顏色通道`
（`SURVEY_ARCHITECTURE.md` 候選二第 5 點），主線不取。

用法：

    python scripts/shading_field_cost.py --out runs/shading_field_cost
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
# 目標殘差 RMS。上兩點涵蓋現行工作點的殘差（純相位＋下限 0.0560、
# 相位＋增益 0.0605、DCT-Shield ε=1.4 為 0.0451），下兩點看小訊號端有沒有
# 不同的走向——比值若隨強度翻轉，「乘性比較便宜」就不是一句話講得完的事。
TARGET_RMS = (0.01, 0.02, 0.04, 0.06)


def low_freq_field(seed: int, grid: int, size: int, device) -> torch.Tensor:
    """(1,1,size,size) 的低頻場，單位標準差。

    16×16 雙三次上採樣到 512 的最高頻率約 grid/2 個週期／全圖，換算成
    Nyquist 正規化半徑是 `grid / size`（512 的 Nyquist 是 0.5 cycles/px）。
    grid = 16 時為 0.031，落在候選二宣稱的 `f_n ≲ 0.03`。實測值由
    `field_radius` 一併寫進 CSV，不靠這段註解當證據。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    coarse = torch.randn(1, 1, grid, grid, generator=g)
    f = F.interpolate(coarse.to(device), size=(size, size),
                      mode="bicubic", align_corners=False)
    return (f - f.mean()) / f.std()


def field_radius(f: torch.Tensor) -> float:
    """場的能量加權平均頻率半徑（1 = Nyquist）。純量測，不影響判定。"""
    spec = torch.fft.rfft2(f[0, 0].float())
    h, w = f.shape[-2:]
    fy = torch.fft.fftfreq(h, device=f.device).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=f.device).view(1, -1)
    r = torch.sqrt(fy ** 2 + fx ** 2) / 0.5
    e = spec.abs() ** 2
    return float((r * e).sum() / e.sum())


def residual_rms(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(((y - x) ** 2).mean()))


def solve_scale(x: torch.Tensor, f: torch.Tensor, target: float,
                multiplicative: bool) -> tuple[float, torch.Tensor]:
    """二分法解出讓 clamp 後殘差 RMS 等於 `target` 的係數。

    clamp 之後 RMS 對係數仍是單調不減的（放大係數只會讓每個像素離原值更遠
    或被夾住不動），所以二分法適用。上界由倍增找出來，不寫死——寫死的上界
    在暗圖上會不夠、在亮圖上會浪費一半的迭代。
    """
    def render(k: float) -> torch.Tensor:
        if multiplicative:
            return (x * torch.exp(k * f)).clamp(0, 1)
        return (x + k * f).clamp(0, 1)

    kind = "乘性" if multiplicative else "加性"
    hi = 0.01
    while True:
        v = residual_rms(x, render(hi))
        # **非有限值要拋，不能當成「還沒到目標」。** fp32 的 exp 在指數約 88
        # 以上溢位成 inf，而全黑像素上 `0 · inf = nan`；`nan < target` 為假，
        # 於是倍增迴圈會靜默跳出、二分法拿一個壞掉的上界繼續算，產出一列
        # 看起來正常但等 RMS 是假的數字。
        if not math.isfinite(v):
            raise RuntimeError(
                f"係數 {hi} 下{kind}場的殘差 RMS 不是有限值（exp 溢位）；"
                f"解不出係數")
        if v >= target:
            break
        hi *= 2.0
        if hi > 64.0:
            raise RuntimeError(
                f"係數超過 64 仍達不到 RMS {target}；{kind}場解不出係數")
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if residual_rms(x, render(mid)) < target:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return k, render(k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images13.txt"),
                    help="影像清單檔，一行一個名字")
    ap.add_argument("--grid", type=int, default=16,
                    help="明暗場的粗網格邊長（候選二寫 16 或 24）")
    ap.add_argument("--seeds", type=int, default=3,
                    help="每張圖抽幾個場。比值的圖間變異遠大於場間變異，"
                         "三個足以看出後者不主導")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names = [ln.strip() for ln in args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    dataset = {d["name"]: d for d in load_dataset(args.data)}
    missing = [n for n in names if n not in dataset]
    if missing:
        raise SystemExit(f"{args.data} 缺這些影像：{missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in names:
        x = load_image_tensor(dataset[name]["path"], device, size=RESOLUTION)
        for seed in range(args.seeds):
            f = low_freq_field(seed, args.grid, RESOLUTION, device)
            r_field = field_radius(f)
            for target in TARGET_RMS:
                k_mul, x_mul = solve_scale(x, f, target, multiplicative=True)
                k_add, x_add = solve_scale(x, f, target, multiplicative=False)
                m_mul = suite.pairwise(x, x_mul)
                m_add = suite.pairwise(x, x_add)
                for kind, k, met in (("multiplicative", k_mul, m_mul),
                                     ("additive", k_add, m_add)):
                    rows.append({
                        "image": name, "seed": seed, "grid": args.grid,
                        "target_rms": target, "kind": kind,
                        "scale": round(k, 6),
                        "field_radius": round(r_field, 5),
                        "rms": round(met["rms"], 6),
                        "dists": round(met["dists"], 6),
                        "lpips": round(met["lpips"], 6),
                        "psnr": round(met["psnr"], 4),
                        "ssim": round(met["ssim"], 6),
                        "vif_p": round(met["vif_p"], 6),
                        "linf": round(met["linf"], 6),
                    })
                write_csv(args.out / "results.csv", rows)
            print(f"{name} seed={seed} 完成（場半徑 {r_field:.4f}）", flush=True)

    print("\n等 RMS 下的失真（13 張 × "
          f"{args.seeds} 個場的平均）：")
    print(f"{'目標 RMS':>9s} {'乘性 DISTS':>11s} {'加性 DISTS':>11s} "
          f"{'加/乘':>7s} {'乘性 PSNR':>10s} {'加性 PSNR':>10s}")
    verdict_ratios = []
    for target in TARGET_RMS:
        sel = [r for r in rows if r["target_rms"] == target]
        mul = [r for r in sel if r["kind"] == "multiplicative"]
        add = [r for r in sel if r["kind"] == "additive"]
        d_mul = statistics.fmean(r["dists"] for r in mul)
        d_add = statistics.fmean(r["dists"] for r in add)
        ratio = d_add / d_mul
        verdict_ratios.append(ratio)
        print(f"{target:9.3f} {d_mul:11.5f} {d_add:11.5f} {ratio:7.3f} "
              f"{statistics.fmean(r['psnr'] for r in mul):10.2f} "
              f"{statistics.fmean(r['psnr'] for r in add):10.2f}")

    # 判準要在**現行工作點的殘差 RMS 上**讀，不是在最有利的那一格上讀：
    # 純相位＋下限的殘差 RMS 是 0.0560、相位＋增益 0.0605，都落在 0.06 那一列。
    work = TARGET_RMS[-1]
    ratio_work = verdict_ratios[TARGET_RMS.index(work)]
    best = max(verdict_ratios)
    print()
    print("判準（SURVEY_ARCHITECTURE 候選二 步驟 0）："
          "加/乘 的 DISTS 比值要達 1.5 倍。")
    print(f"  現行工作點所在的 RMS {work}：{ratio_work:.3f}"
          "（本方法殘差 RMS 為 0.0560-0.0605）")
    print(f"  四個強度的最大值：{best:.3f}")
    print("-> 候選二唯一的理論依據在工作點上"
          + ("成立" if ratio_work >= 1.5 else "不成立（步驟 0 否決）"))
    print(f"\n表：{args.out / 'results.csv'}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
