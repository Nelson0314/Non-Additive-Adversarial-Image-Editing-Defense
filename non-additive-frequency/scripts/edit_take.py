"""編輯有沒有「吃進去」：`LPIPS(輸入, 編輯(輸入))`，不需要任何參照影像。

為什麼需要它
────────────────────────────────────────────────────────────────────
語意誘餌交出的是**另一張真的照片**，於是現行的位移讀數
`LPIPS(編輯(原圖), 編輯(誘餌圖))` 被內容差異灌水：兩張輸出本來就不同場景，
分不出「編輯失敗了」與「內容本來就不一樣」。空白地板也扣不掉那一份，
因為地板本身依賴誘餌是什麼。共防禦參照也用不上——誘餌不是逐點映射。

這一支換一個問法：**攻擊方的指令對它自己吃進去的那張圖，改動了多少？**

    吃進去的量(x) = LPIPS(x, 編輯(x))

`x` 取原圖時是「這個指令在未防禦的圖上改了多少」，取誘餌圖時是「在防禦後的圖
上改了多少」。兩者**各自只用自己的輸入當參照**，內容差異因此不進來。
比值小代表編輯在誘餌圖上做得少。

**這不是「擋下」的判準。** 編輯做得少可能是擋下，也可能是誘餌把畫面變得
與指令無關；反過來，模型重畫成無關場景會讓這個量變**大**而不是變小。
判定仍然要看圖，本支只是把「編輯有沒有動」這一件事量出來，讓看圖有個順序。

用法：

    python scripts/edit_take.py --src runs/ip2p_decoy/dev5 --out runs/ip2p_decoy/edit_take.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, nargs="+", required=True,
                    help="含 results.csv 與 __def/__edit_def/__edit_orig 圖的目錄")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=dev)

    rows = []
    for src in args.src:
        recs = list(csv.DictReader((src / "results.csv").open(encoding="utf-8")))
        for r in recs:
            name, cond = r["image"], r["condition"]
            paths = {
                "orig": src / f"{name}__orig.png",
                "def": src / f"{name}__{cond}__def.png",
                "edit_orig": src / f"{name}__{cond}__edit_orig.png",
                "edit_def": src / f"{name}__{cond}__edit_def.png",
            }
            missing = [k for k, p in paths.items() if not p.exists()]
            if missing:
                print(f"[skip] {src.name}/{name}/{cond}：缺 {missing}", flush=True)
                continue
            img = {k: load_image_tensor(p, dev, size=RESOLUTION)
                   for k, p in paths.items()}
            take_orig = float(suite.pairwise(img["orig"], img["edit_orig"])["lpips"])
            take_def = float(suite.pairwise(img["def"], img["edit_def"])["lpips"])
            rows.append({
                "image": name, "condition": cond,
                "decoy_instruction": r.get("decoy_instruction", ""),
                "fid_dists": r.get("fid_dists", ""),
                "edit_lpips": r.get("edit_lpips", ""),
                "take_orig": round(take_orig, 5),
                "take_def": round(take_def, 5),
                "take_ratio": round(take_def / take_orig, 4) if take_orig else "",
            })
            print(f"{name[:30]:32s}{cond:26s}take_orig={take_orig:.4f} "
                  f"take_def={take_def:.4f} ratio={rows[-1]['take_ratio']}",
                  flush=True)

    write_csv(args.out, rows)
    if rows:
        ratios = [r["take_ratio"] for r in rows if r["take_ratio"] != ""]
        print(f"\nn={len(ratios)}  take_ratio 中位數={statistics.median(ratios):.3f}"
              f"  最小={min(ratios):.3f}  最大={max(ratios):.3f}")
    print(f"寫出 {args.out}")


if __name__ == "__main__":
    main()
