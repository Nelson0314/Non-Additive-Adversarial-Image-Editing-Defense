"""從已存的 PNG 補回 `results.csv` 缺掉的量測列。**不重跑最佳化。**

為什麼需要它
────────────────────────────────────────────────────────────────────
`--skip-existing` 早期的版本跳過影像時沒有把舊列帶回來，而每張寫檔是整份
重寫，於是先前的量測被截掉——防禦圖還在、數字沒了，報表上只看得到列數變少，
不會拋錯（實際踩過：5 張防禦圖只剩 2 列）。根因已修，這一支是把已經掉了的
補回來。

能補回什麼、不能補回什麼
────────────────────────────────────────────────────────────────────
可以：所有由影像算出來的量——`fid_*`（原圖對防禦圖）、`edit_*`（原圖的編輯
對防禦圖的編輯）、兩個語意讀數。這些只需要四張 PNG，全部都在。

**不可以**：`stop_reason`／`stopped_at`／`best_eval`／`total_seconds` 這些
只有訓練當下才知道的量，補不回來就留空，**不填猜的值**。同理 `radius` 等
旗標欄由 CLI 補齊，缺了就必須明給。

用法：
    python scripts/rebuild_rows.py --run runs/ip2p_ig_converge/ig_d21 \\
        --condition phase_gain --radius 2.1 --loss image_guidance \\
        --ig-zt diffuse_src --steps 8000 --step-size 0.01
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESOLUTION = 512


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--condition", default="phase_gain")
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--loss", required=True)
    ap.add_argument("--ig-zt", default="")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--step-size", type=float, required=True)
    args = ap.parse_args()

    import torch  # noqa: F401  （載入順序與其他驅動一致）
    from apa_baseline import load_dataset
    from src.metrics.standard import (
        SIGLIP_BLOCKED_THRESHOLD, blocked_by_siglip, standard_row,
    )
    from src.metrics.suite import MetricSuite
    from src.utils.io import load_image_tensor, write_csv

    out = args.run / "results.csv"
    have = []
    if out.exists():
        with out.open(encoding="utf-8") as f:
            have = list(csv.DictReader(f))
    known = {r["image"] for r in have}

    ds = {d["name"]: d for d in load_dataset(args.data, prompt_index=0)}
    suite = MetricSuite()
    c = args.condition
    added = 0
    for png in sorted(args.run.glob(f"*__{c}__def.png")):
        name = png.name[: -len(f"__{c}__def.png")]
        if name in known:
            continue
        need = {s: args.run / f"{name}__{c}__{s}.png"
                for s in ("def", "edit_orig", "edit_def")}
        need["orig"] = args.run / f"{name}__orig.png"
        missing = [k for k, p in need.items() if not p.exists()]
        if missing:
            print(f"  {name}：缺 {missing}，跳過（不編造）")
            continue
        img = {k: load_image_tensor(p, suite.device, size=RESOLUTION)
               for k, p in need.items()}
        fid = suite.pairwise(img["orig"], img["def"])
        prot = suite.pairwise(img["edit_orig"], img["edit_def"])
        sim = suite.image_similarity(img["edit_orig"], img["edit_def"])
        have.append({
            "image": name, "condition": c, "attacker": "instruct-pix2pix",
            "instruction": ds.get(name, {}).get("prompt", ""),
            "task": ds.get(name, {}).get("class", ""),
            "radius": round(args.radius, 6),
            "loss": args.loss, "ig_zt": args.ig_zt,
            "defense_steps": args.steps, "step_size": args.step_size,
            # 只有訓練當下才知道的量補不回來，留空而不是填猜的值。
            "stop_reason": "", "stopped_at": "", "best_eval": "",
            "total_seconds": "",
            "rebuilt_from_png": 1,
            **standard_row("fid_", fid),
            **standard_row("edit_", prot),
            "fid_linf": round(fid["linf"], 5),
            "fid_rms": round(fid["rms"], 5),
            "edit_lpips": round(float(prot["lpips"]), 5),
            "edit_clip_sim": round(float(sim["clip"]), 5),
            "edit_siglip_sim": round(float(sim["siglip"]), 5),
            "blocked": blocked_by_siglip(sim["siglip"]),
            "siglip_blocked_threshold": SIGLIP_BLOCKED_THRESHOLD,
        })
        added += 1
        print(f"  補回 {name}  dists={fid['dists']:.4f} "
              f"effect={float(prot['lpips']):.4f}")

    have.sort(key=lambda r: r["image"])
    write_csv(out, have)
    print(f"{out}：補回 {added} 筆，現在共 {len(have)} 筆")


if __name__ == "__main__":
    main()
