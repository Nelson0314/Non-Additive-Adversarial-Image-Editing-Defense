"""把所有臂的防禦圖失真放到同一張表上，四個指標一起量。

為什麼需要這一支
────────────────────────────────────────────────────────────────────
各臂的 `results.csv` 是各自的求解腳本寫的，欄位隨載體而異：
`curve_*`、`ab_warp` 有 ΔE00 沒有 LPIPS，`inpaint_*` 兩者都沒有意義，
只有兩個 `style_*` 直接交付臂寫了 LPIPS。於是**沒有一個軸能把十二個臂排在
一起**——而主表那條線的等失真錨點是 LPIPS 0.3344（`colour_curve_ours` 原生
設定的八張平均），要掛上去就得先有這一欄。

這也正是本專案剛寫下的那條限制的自我檢查：**任一失真指標固定住時，其餘指標
的離散度同量級或更大，所以等失真比較一定要四個指標一起報**。本目錄自己的
報表在這一支之前並沒有做到。

量什麼
────────────────────────────────────────────────────────────────────
逐張 `<名稱>__<臂>__def.png` 對同目錄的 `<名稱>__orig.png`：

| 欄 | 怎麼算 |
|---|---|
| `lpips` | `piq.LPIPS()`（VGG），與主表與對齊那條線同一個實作 |
| `deltaE00` | `skimage` 的 `deltaE_ciede2000`，全圖平均（量測路徑，不是可微那份） |
| `psnr` | `10·log10(1/MSE)` |
| `linf` | `max |y − x|` |
| `rms` | `sqrt(MSE)` |

**都在量化後的 PNG 上算**，因為交付的就是 PNG。

用法
    python lab/code/defence_fidelity.py --out lab/results/fidelity.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402

from curve_budget_defence import write_rows  # noqa: E402
from src.defense.color_amplitude import delta_e00  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512

#: 主表等失真對齊的錨：`colour_curve_ours` 原生設定的八張平均 LPIPS。
#: **那是另一批的數字**，這裡只當參照欄，不當判準。
ANCHOR_LPIPS = 0.3344

#: 哪些臂本來就沒有失真預算。它們的 ΔE00／PSNR 不與有上限的臂比大小。
NO_BUDGET = {"style_random", "style_low", "inpaint_bg", "inpaint_outside_face"}


def pairs(arm_dir: Path, arm: str):
    out = []
    for def_png in sorted(arm_dir.glob(f"*__{arm}__def.png")):
        name = def_png.name.split("__")[0]
        orig = arm_dir / f"{name}__orig.png"
        if not orig.is_file():
            raise SystemExit(f"{arm_dir} 缺 {orig.name}；防禦圖沒有對應的原圖，"
                             "失真算不了，不要靜默跳過")
        out.append((name, orig, def_png))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("lab/runs/defence"))
    ap.add_argument("--out", type=Path, default=Path("lab/results/fidelity.csv"))
    ap.add_argument("--arms", nargs="+", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("拿不到 GPU。本專案的預設是靜默退回 CPU，而 CPU 上"
                         "這一支跑得完只是慢兩個數量級——停住比跑完難發現。")
    import piq
    lpips = piq.LPIPS().to(device)

    arms = args.arms or sorted(d.name for d in args.root.iterdir() if d.is_dir())
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        d = args.root / arm
        found = pairs(d, arm)
        if not found:
            print(f"[SKIP] {arm}：沒有 __{arm}__def.png", flush=True)
            continue
        for name, orig, defended in found:
            x = load_image_tensor(orig, device, size=RESOLUTION)
            y = load_image_tensor(defended, device, size=RESOLUTION)
            with torch.no_grad():
                lp = float(lpips(y, x))
                mse = float((y - x).pow(2).mean())
            rows.append({
                "arm": arm, "image": name,
                "budget": "none" if arm in NO_BUDGET else "ΔE00 16 / 臉 8",
                "lpips": round(lp, 4),
                "lpips_vs_anchor": round(lp / ANCHOR_LPIPS - 1.0, 4),
                "deltaE00": round(float(delta_e00(x, y, torch.ones_like(x[:, :1]))), 4),
                "psnr": round(float("inf") if mse == 0 else 10 * torch.log10(
                    torch.tensor(1.0 / mse)).item(), 4),
                "linf": round(float((y - x).abs().max()), 5),
                "rms": round(mse ** 0.5, 6),
                "anchor_lpips": ANCHOR_LPIPS,
                "anchor_source": "主表 colour_curve_ours 原生設定的八張平均（另一批）",
            })
            write_rows(args.out, rows)
        sub = [r for r in rows if r["arm"] == arm]
        med = lambda k: sorted(r[k] for r in sub)[len(sub) // 2]
        print(f"{arm:30s} n={len(sub)}  LPIPS {med('lpips'):.4f} "
              f"({med('lpips_vs_anchor'):+.1%} vs 錨)  ΔE00 {med('deltaE00'):6.2f}  "
              f"PSNR {med('psnr'):6.2f}  L∞ {med('linf'):.3f}", flush=True)
    print(f"完成：{len(rows)} 列 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
