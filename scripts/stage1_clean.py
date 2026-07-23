"""stage1：乾淨情況比較 — SPEC.md §6.2、STRUCTURE.md §4.3。

方法矩陣 = {PhotoGuard-enc, PhotoGuard-diff, AdvDiff-based, APA-based, Hybrid}
         × {SD V1.4, SD V2.0} × {sdedit, inpaint}，每結果 20 seed 平均。

校準檢查點：PhotoGuard 數值須與 SPEC §2.7 Encoder/Diffusion 欄接近。
輸出 experiments/stage1/results.csv。
"""
