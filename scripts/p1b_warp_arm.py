"""P1b —— 在探針上加入第三、第四臂：純空間變形（雙線性 / 雙三次）。

動機是 P1 與 P2 的一個矛盾。P1（等 LPIPS 的模糊 vs 雜訊）判 NLPD 與 VIF
與 LPIPS 同盲區，但 P2（真實的 E15 資料）中這兩者卻把 site S 與 site P 分得
很開（2.5–2.8 倍）。兩件事不可能同時只用「對模糊收費」來解釋。

可能的解釋是：site S 的失真不只是模糊，還有幾何位移，而 P1 只測了模糊
與雜訊，從未測位移。若 NLPD/VIF 的分辨力主要來自位移而非模糊，那麼把它們
加進約束會因為 site S 是一個變形而懲罰它——這是循環論證：site S 本來就
被允許是變形，它只是不被允許變模糊。

本腳本以第三、第四臂檢驗此假說：把 site S 自己的位移場機制（粗網格 + 雙線性
上採樣，`src/residual/site_warp.py`）套上隨機平滑位移，同樣校準到 LPIPS
0.05，再看各指標怎麼計價。

兩種重取樣一併測，因為 `docs/NEXT_SESSION.md` §5 列了「site S 的 grid_sample
改 bicubic」這個待辦——E15 的鈍化有一部分來自雙線性重取樣，此處可直接量出
那一部分有多大。重取樣模式在此於腳本內指定而非改動 `site_warp.py`：是否
改動生產模塊是另一個決定，不應由一個探針順手做掉。

範圍限於 τ=0.05：只有該 τ 的 E15 逐圖影像入了版控，也是 E15 那個「1.15×
領先」宣稱所在的運作點。

輸出：`runs/p1_iso_lpips_probe/warp.csv` 與 `warp_summary.txt`。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.battery import MetricBattery
from src.residual.site_warp import WarpResidual, identity_grid

ROOT = Path(__file__).resolve().parent.parent
SRC_RUN = ROOT / "runs" / "e15_S_tau0.05"
OUT = ROOT / "runs" / "p1_iso_lpips_probe"

TARGET = 0.05
TOL = 1e-4
MAX_ITERS = 40
SCALE_HI = 20.0

# 與 E15 的 site S 設定一致：r32 即 grid_size=32。max_disp 取得夠大使其不
# 綁住——本探針的位移量由校準決定，不該再被另一道上界截斷。
GRID_SIZE = 32
MAX_DISP = 50.0
SEED = 20260731

MODES = ["bilinear", "bicubic"]

# 失真化轉換與 p1_summary 相同，此處重用以確保兩份判定可直接並排。
from p1_summary import distortion  # noqa: E402
from p1_summary import KEYS  # noqa: E402


def load(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def save(x: torch.Tensor, p: Path) -> None:
    a = (x[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(a).save(p)


def warp(x: torch.Tensor, base_flow: torch.Tensor, scale: float,
         mode: str) -> torch.Tensor:
    """套用 base_flow × scale 的位移場。

    位移場經 `WarpResidual.displacement` 產生（粗網格 + 雙線性上採樣），與
    site S 完全相同；只有最後的重取樣模式可切換。雙三次會過衝到 [0,1] 之外，
    故夾回——那是可顯示影像的定義，不是為了讓數字好看。
    """
    b, c, h, w = x.shape
    mod = WarpResidual(size=w, grid_size=GRID_SIZE, max_disp=MAX_DISP)
    mod.flow.data = base_flow * scale
    disp = mod.displacement(h, w).to(x.dtype)

    sx = 2.0 / max(w - 1, 1)
    sy = 2.0 / max(h - 1, 1)
    off = torch.stack((disp[:, 0] * sx, disp[:, 1] * sy), dim=-1)
    grid = identity_grid(h, w, x.device, x.dtype) + off
    out = F.grid_sample(x, grid.expand(b, -1, -1, -1), mode=mode,
                        padding_mode="border", align_corners=True)
    return out.clamp(0, 1)


def calibrate(make, lpips_fn, target: float, hi: float):
    """二分搜尋位移倍率，使 LPIPS 命中 target。上限不足時直接拋出。"""
    top = lpips_fn(make(hi))
    if top < target:
        raise ValueError(
            f"倍率上限 {hi} 處的 LPIPS 僅 {top:.4f}，低於目標 {target}；"
            "二分會收斂到上限並回報假的命中，需調高上限"
        )
    lo, hi_ = 0.0, hi
    for _ in range(MAX_ITERS):
        mid = 0.5 * (lo + hi_)
        img = make(mid)
        val = lpips_fn(img)
        if abs(val - target) < TOL:
            return mid, img, val
        if val < target:
            lo = mid
        else:
            hi_ = mid
    img = make(0.5 * (lo + hi_))
    return 0.5 * (lo + hi_), img, lpips_fn(img)


@torch.no_grad()
def main() -> None:
    """全程不建計算圖。

    `WarpResidual.flow` 是 `nn.Parameter`，故 `warp()` 的輸出預設帶梯度，
    存檔時會以 `Can't call numpy() on Tensor that requires grad` 失敗。
    正解是在此宣告整段為純前向評估——本探針從頭到尾不需要梯度——而不是
    在存檔處補 `.detach()` 把症狀蓋掉。
    """
    bat = MetricBattery()
    origs = sorted(SRC_RUN.glob("*/orig.png"))
    if not origs:
        raise FileNotFoundError(f"找不到原圖：{SRC_RUN}")

    # 位移場的隨機種子固定且逐圖共用：六張圖套同一個位移場，使跨圖差異
    # 只來自影像內容，不混入位移場的取樣變異。
    g = torch.Generator().manual_seed(SEED)
    base_flow = torch.randn(1, 2, GRID_SIZE, GRID_SIZE, generator=g)

    rows = []
    for op in origs:
        name = op.parent.name.split("__")[0]
        x = load(op)
        lp = lambda y: float(bat._lpips(x.clamp(0, 1), y.clamp(0, 1)))

        row = {"image": name, "target_lpips": TARGET}
        for mode in MODES:
            scale, xw, lpv = calibrate(
                lambda s: warp(x, base_flow, s, mode), lp, TARGET, SCALE_HI)
            save(xw, OUT / f"{name}__lpips0p05__warp_{mode}.png")

            mod = WarpResidual(size=512, grid_size=GRID_SIZE, max_disp=MAX_DISP)
            mod.flow.data = base_flow * scale
            st = mod.disp_stats()

            m = bat.evaluate(x, xw)
            row.update({
                f"{mode}_scale": scale, f"{mode}_lpips_actual": lpv,
                f"{mode}_disp_mean_px": st["disp_mean_px"],
                f"{mode}_disp_max_px": st["disp_max_px"],
                **{f"{mode}_{k}": v for k, v in m.items()},
            })
            print(f"{name}  {mode:>8s}  倍率={scale:.4f}  "
                  f"LPIPS={lpv:.4f}  平均位移={st['disp_mean_px']:.3f} px  "
                  f"銳利度={m['acutance_ratio']:.3f}")
        rows.append(row)

    with (OUT / "warp.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report(rows)


def report(rows: list[dict]) -> None:
    """把兩個變形臂與既有的模糊／雜訊臂並排，全部在 LPIPS = 0.05 上。

    判準：一個合格的抗模糊約束，應該對模糊收高費、對位移收低費。
    對位移收高費者，會因為 site S 是變形而懲罰它，那是循環論證。
    """
    blur_rows = [r for r in csv.DictReader((OUT / "probe.csv").open(encoding="utf-8"))
                 if abs(float(r["target_lpips"]) - TARGET) < 1e-9]
    if len(blur_rows) != len(rows):
        raise ValueError(
            f"probe.csv 在 τ={TARGET} 有 {len(blur_rows)} 張，warp 臂有 {len(rows)} 張，"
            "兩者必須是同一組影像才能並排比較"
        )

    lines = []
    hdr = (f"{'指標':>16s} {'模糊':>10s} {'雜訊':>10s} "
           f"{'變形-雙線性':>12s} {'變形-雙三次':>12s}")
    print()
    print("=== LPIPS 全部固定為 0.05 的失真量（0 為完美，越大越糟；n=6）===")
    print(hdr)
    print("-" * 66)
    for k in KEYS:
        db = np.mean([distortion(k, r, "blur") for r in blur_rows])
        dn = np.mean([distortion(k, r, "noise") for r in blur_rows])
        dw = {m: np.mean([distortion(k, r, m) for r in rows]) for m in MODES}
        line = (f"{k:>16s} {db:>10.5f} {dn:>10.5f} "
                f"{dw['bilinear']:>12.5f} {dw['bicubic']:>12.5f}")
        print(line)
        lines.append(line)

    print()
    print("--- 重取樣模式對銳利度的影響 ---")
    for mode in MODES:
        a = np.mean([r[f"{mode}_acutance_ratio"] for r in rows])
        d = np.mean([r[f"{mode}_disp_mean_px"] for r in rows])
        s = f"{mode:>8s}：銳利度保留 {100*a:.1f}%，平均位移 {d:.3f} px"
        print(s)
        lines.append(s)

    (OUT / "warp_summary.txt").write_text("\n".join([hdr] + lines), encoding="utf-8")


if __name__ == "__main__":
    main()
