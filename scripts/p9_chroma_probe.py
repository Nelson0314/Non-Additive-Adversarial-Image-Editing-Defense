"""E28 —— 色度約束的候選判別：等 LPIPS 下，誰對色調偏移收費而不對位移收費？

判別法沿用 E20，不另立新法。E20 §5.3 的作法是把數種失真校準到相同的
LPIPS，再看各候選指標分別收多少費；判準是「真的在量 X 的指標必須對 X 收費
較高」。E20 靠這個方法發現 GMSD、NLPD、VIF、HaarPSI 量的其實是位移而非鈍化，
而它們在真實資料上看似有效只是因為 site S 恰好是一個變形——那是循環論證。

本次要找的是能擋住 site C 買色調偏移的約束（E27 §4）。要求有三條，缺一不可：

1. 對色度偏移收費明顯——否則擋不住。
2. 對位移與雜訊不多收——site S 本來就被允許是變形、site P 本來就被允許是
   加性擾動，對它們收費就是把「不是我們要禁止的東西」也禁掉，重蹈 E20 §5.3
   的循環論證。
3. site P 的真實解必須遠低於 site C 的真實解——這是最終判準。前兩條是
   合成臂上的性質，這一條是真實資料上的結果，且方向已由人眼定調：使用者
   判讀 `runs/e27d_C_lr0.3/compare.html` 的回報是「P 的那兩張防禦圖人眼看起來
   跟原圖幾乎一樣，其他則有色調偏移一點點」。

臂的組成（前四臂與 E20 的四臂探針相同，第五臂是本次新增）：

| 臂 | 產生方式 | 角色 |
|---|---|---|
| 模糊 | 高斯 σ 二分搜尋至目標 LPIPS | E20 的既有臂 |
| 雜訊 | 加性高斯，同上 | site P 的失真型態 |
| 變形-雙線性 | `WarpResidual` 隨機平滑位移 | site S 的失真型態（含鈍化） |
| 變形-雙三次 | 同上、換重取樣 | site S 的失真型態（不含鈍化） |
| 色度偏移 | `ColorResidual` 隨機平滑 ΔM | site C 的失真型態 |

另把 site C 與 site P 的真實解（`runs/e27d_*` 的 `defended.png`）一併評分，
因為合成臂與真實解可能不同——E20 §4.1 就出現過兩者判定矛盾的情形。

輸出 `runs/p9_chroma_probe/{probe.csv, real.csv, summary.txt}`。
CPU 執行，不需要 GPU。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.chroma import chroma_battery  # noqa: E402
from src.metrics.local_acutance import local_acutance  # noqa: E402
from src.residual.site_color import ColorResidual  # noqa: E402
from src.residual.site_warp import WarpResidual  # noqa: E402

OUT = ROOT / "runs" / "p9_chroma_probe"
TARGET_LPIPS = 0.05          # 與 E27 校準所用的 τ 相同
SEED = 20260728
GRID = 32

# 真實解：E27 第四輪的校準結果，兩臂都貼著 τ=0.05 跑
REAL_RUNS = [("site C", "e27d_C_lr0.3"), ("site P", "e27d_P_lr0.03")]


def to_tensor(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def gaussian_blur(x, sigma):
    from src.purify.ops import gaussian_blur as gb
    return gb(x, sigma)


def make_arm(kind: str, x: torch.Tensor, amount: float) -> torch.Tensor:
    """依強度參數產生該臂的失真影像。強度與 LPIPS 單調遞增，供二分搜尋。"""
    if kind == "blur":
        return gaussian_blur(x, amount)
    if kind == "noise":
        g = torch.Generator().manual_seed(SEED)
        return (x + amount * torch.randn(x.shape, generator=g)).clamp(0, 1)
    if kind in ("warp_bilinear", "warp_bicubic"):
        mod = WarpResidual(size=x.shape[-1], grid_size=GRID, init_std=0.0,
                           max_disp=50.0, resample=kind.split("_")[1])
        g = torch.Generator().manual_seed(SEED)
        mod.flow.data = torch.randn(mod.flow.shape, generator=g) * amount
        with torch.no_grad():
            return mod.pixel_residual(x)
    if kind == "chroma":
        mod = ColorResidual(size=x.shape[-1], grid_size=GRID, init_std=0.0,
                            max_dev=10.0)
        g = torch.Generator().manual_seed(SEED)
        mod.delta.data = torch.randn(mod.delta.shape, generator=g) * amount
        with torch.no_grad():
            return mod.pixel_residual(x)
    raise ValueError(f"未知的臂：{kind!r}")


def calibrate(kind: str, x: torch.Tensor, lpips_fn, target: float,
              lo: float, hi: float, iters: int = 24):
    """二分搜尋強度參數，使該臂的 LPIPS 命中目標。

    先確認上界真的越過目標再開始搜尋——上界不足時二分會回傳上界，
    產生一個「看起來校準過但其實沒到」的臂，該情形會使整組比較失效。
    """
    with torch.no_grad():
        top = float(lpips_fn(x, make_arm(kind, x, hi).clamp(0, 1)))
    if top < target:
        raise ValueError(
            f"{kind} 的強度上界 {hi} 只能達到 LPIPS {top:.4f}，未及目標 "
            f"{target}。請提高上界，不可回傳該值"
        )
    for _ in range(iters):
        mid = (lo + hi) / 2
        with torch.no_grad():
            v = float(lpips_fn(x, make_arm(kind, x, mid).clamp(0, 1)))
        if v < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


ARMS = [
    ("blur", 0.0, 6.0),
    ("noise", 0.0, 0.3),
    ("warp_bilinear", 0.0, 3.0),
    ("warp_bicubic", 0.0, 3.0),
    ("chroma", 0.0, 3.0),
]


def score(x: torch.Tensor, y: torch.Tensor) -> dict:
    out = chroma_battery(x, y)
    out["local_acutance_dev"] = local_acutance(x, y)["local_acutance_dev"]
    return out


def main() -> None:
    import piq

    OUT.mkdir(parents=True, exist_ok=True)
    lpips = piq.LPIPS()

    def lp(a, b):
        return lpips(a.clamp(0, 1), b.clamp(0, 1))

    imgs = sorted((ROOT / "data/dayn_testset").rglob("*.png"))
    rows = []
    for path in imgs:
        x = to_tensor(path)
        print(f"\n[{path.stem}]", flush=True)
        for kind, lo, hi in ARMS:
            amt = calibrate(kind, x, lp, TARGET_LPIPS, lo, hi)
            y = make_arm(kind, x, amt).clamp(0, 1)
            with torch.no_grad():
                got = float(lp(x, y))
            m = score(x, y)
            rows.append({"image": path.stem, "arm": kind, "amount": amt,
                         "lpips": got, **m})
            print(f"  {kind:15s} 強度 {amt:7.4f}  LPIPS {got:.4f}  "
                  f"de76 {m['de76']:7.3f}  dchroma {m['dchroma']:7.3f}  "
                  f"local {m['local_dchroma_dev']:7.3f}  "
                  f"鈍化 {m['local_acutance_dev']:.4f}", flush=True)

    _write(OUT / "probe.csv", rows)

    # ---- 真實解 ----
    real = []
    for label, run in REAL_RUNS:
        d = ROOT / "runs" / run
        if not d.exists():
            print(f"[跳過] 找不到 {d}")
            continue
        for cell in sorted(p for p in d.iterdir() if p.is_dir()):
            o, r = cell / "orig.png", cell / "defended.png"
            if not (o.exists() and r.exists()):
                continue
            xo, xr = to_tensor(o), to_tensor(r)
            with torch.no_grad():
                got = float(lp(xo, xr))
            m = score(xo, xr)
            real.append({"label": label, "run": run,
                         "image": cell.name.split("__")[0],
                         "lpips": got, **m})
    if real:
        _write(OUT / "real.csv", real)

    # ---- 判定 ----
    lines = []
    keys = ["de76", "de00", "dchroma", "local_dchroma_dev",
            "local_chroma_bias", "local_acutance_dev"]
    lines.append(f"=== 合成臂（全部校準到 LPIPS = {TARGET_LPIPS}，n={len(imgs)}）===")
    lines.append(f"{'指標':>20s}" + "".join(f"{k:>16s}" for k, _, _ in ARMS))
    for k in keys:
        vals = {a: np.mean([r[k] for r in rows if r["arm"] == a]) for a, _, _ in ARMS}
        lines.append(f"{k:>20s}" + "".join(f"{vals[a]:>16.4f}" for a, _, _ in ARMS))

    lines.append("")
    lines.append("=== 判準一：對色度收費是否明顯高於位移與雜訊 ===")
    for k in keys:
        v = {a: np.mean([r[k] for r in rows if r["arm"] == a]) for a, _, _ in ARMS}
        others = max(v["noise"], v["warp_bilinear"], v["warp_bicubic"])
        ratio = v["chroma"] / max(others, 1e-9)
        ok = ratio > 2.0
        lines.append(f"  {k:>20s}  色度/max(雜訊,兩個變形) = {ratio:6.2f}  "
                     f"{'通過' if ok else '不通過'}")

    if real:
        lines.append("")
        lines.append("=== 判準二：真實解上 site C 是否明顯高於 site P ===")
        lines.append("（人眼已定調：site P 看不出來、site C 有色調偏移）")
        lines.append(f"{'指標':>20s}{'site C':>12s}{'site P':>12s}{'倍率':>8s}  判定")
        for k in keys:
            c = np.mean([r[k] for r in real if r["label"] == "site C"])
            p = np.mean([r[k] for r in real if r["label"] == "site P"])
            ratio = c / max(p, 1e-9)
            lines.append(f"{k:>20s}{c:>12.4f}{p:>12.4f}{ratio:>8.2f}  "
                         f"{'揭穿' if ratio > 2.0 else '分不出'}")

    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\n寫入 {OUT}")


def _write(path: Path, rows) -> None:
    if not rows:
        raise ValueError(f"沒有資料可寫入 {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
