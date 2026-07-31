"""P3 —— 用已校準好的影像檢驗局部銳利度偏差。

P1/P1b 產生的四臂影像（模糊／雜訊／變形-雙線性／變形-雙三次）全部已校準到
LPIPS = 0.05 並存檔，故新指標可以直接在這些檔案上評分，**不需要重跑任何
二分搜尋**。P2 的 E15 真實資料同理。

要檢驗的三件事：

1. **對位移不敏感**：雙三次變形（實測銳利度 99.9%）應被判為接近無失真，
   與雜訊同級；雙線性變形（85.0%）應被判為介於雜訊與模糊之間。
2. **不可抵銷**：`tests/test_local_acutance.py` 的單元測試負責，此處不重複。
3. **在真實資料上仍能分辨**：E15 的 site S 應明顯高於 site P。
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.acutance import acutance
from src.metrics.local_acutance import local_acutance

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "runs" / "p1_iso_lpips_probe"
RUNS = ROOT / "runs"
OUT = RUNS / "p3_local_acutance"

ARMS = [("blur", "模糊"), ("noise", "雜訊"),
        ("warp_bilinear", "變形-雙線性"), ("warp_bicubic", "變形-雙三次")]


def load(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # 影像清單取自 E15 的 run 目錄而非 glob 探針目錄：後者還有比對頁用的
    # zoom__*__orig.png，其檔名同樣以 __orig.png 結尾，會被誤當成一張原圖。
    names = sorted(d.name.split("__")[0]
                   for d in (RUNS / "e15_S_tau0.05").iterdir() if d.is_dir())
    if not names:
        raise FileNotFoundError("runs/e15_S_tau0.05 沒有逐圖子目錄")

    rows = []
    for name in names:
        x = load(PROBE / f"{name}__orig.png")
        for arm, _ in ARMS:
            p = PROBE / f"{name}__lpips0p05__{arm}.png"
            if not p.exists():
                raise FileNotFoundError(
                    f"缺少 {p}。四臂必須齊備才能比較；warp 兩臂由 "
                    "scripts/p1b_warp_arm.py 產生")
            y = load(p)
            rows.append({"image": name, "arm": arm,
                         "acutance_ratio": acutance(x, y)["acutance_ratio"],
                         **local_acutance(x, y)})

    e15 = []
    for site in ("S", "P"):
        run = RUNS / f"e15_{site}_tau0.05"
        for d in sorted(run.iterdir()):
            o, v = d / "orig.png", d / "defended.png"
            if not (d.is_dir() and o.exists() and v.exists()):
                continue
            x, y = load(o), load(v)
            e15.append({"site": site, "image": d.name.split("__")[0],
                        "acutance_ratio": acutance(x, y)["acutance_ratio"],
                        **local_acutance(x, y)})
    if len(e15) != 12:
        raise ValueError(f"E15 τ=0.05 應有 12 張（S/P 各 6），實得 {len(e15)}")

    for fn, data in (("probe_arms.csv", rows), ("e15.csv", e15)):
        with (OUT / fn).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    keys = ["acutance_ratio", "local_acutance_dev",
            "local_acutance_signed", "local_acutance_worst"]

    print("=== 四臂（LPIPS 全部 = 0.05，n=6）===")
    print(f"{'臂':>14s} " + " ".join(f"{k:>22s}" for k in keys))
    print("-" * 106)
    for arm, lab in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        vals = " ".join(f"{np.mean([r[k] for r in sub]):>22.5f}" for k in keys)
        print(f"{lab:>14s} {vals}")

    print()
    print("=== E15 τ=0.05 真實資料（n=6 各）===")
    print(f"{'site':>14s} " + " ".join(f"{k:>22s}" for k in keys))
    print("-" * 106)
    for site in ("S", "P"):
        sub = [r for r in e15 if r["site"] == site]
        vals = " ".join(
            f"{np.mean([r[k] for r in sub]):>15.5f}±{np.std([r[k] for r in sub]):<6.4f}"
            for k in keys)
        print(f"{'site ' + site:>14s} {vals}")


if __name__ == "__main__":
    main()
