"""取樣本身的離散 `D_seed`：同一輸入、不同評估種子的兩次 ip2p 編輯之間的 LPIPS。

未防禦的五個種子（主種子 20260812 ＋ 20260813–16）兩兩配對，每格 10 對，
分全圖／主體／背景（`split_displacement`，與位移同一個量法）。輸出
`lab/results/passthrough/seed_spread.csv`。這是位移與 `D_T` 的雜訊底線：
兩個條件的差若小於它，無法與「換一顆種子」區分。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402

from passthrough_readout import IMAGES, R, edit_png, load, subject_mask  # noqa: E402
from src.metrics.regional import RegionalLPIPS, split_displacement  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

SEEDS = [None, 20260813, 20260814, 20260815, 20260816]     # None = 主種子的既有檔


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)
    rows = []
    for name in IMAGES:
        mask = subject_mask(name, device)
        for k in range(4):
            imgs = {s: load(edit_png("undefended", name, k, s), device) for s in SEEDS}
            for s1, s2 in itertools.combinations(SEEDS, 2):
                with torch.no_grad():
                    d = split_displacement(regional, imgs[s1], imgs[s2], mask)
                rows.append({"image": name, "prompt_index": k,
                             "seed_a": s1 or 20260812, "seed_b": s2,
                             **{f"Dseed_{r}": round(float(v), 5) for r, v in d.items()}})
        write_csv(R / "lab/results/passthrough/seed_spread.csv", rows)
        print(f"[seed_spread] {name} done", flush=True)


if __name__ == "__main__":
    main()
