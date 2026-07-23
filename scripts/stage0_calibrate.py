"""stage0：相似性約束校準 — SPEC.md §6.1、STRUCTURE.md §4.2。

1. 對 PhotoGuard（κ=0.06）計算 LPIPS(protected, original) 平均，得基準 L_ref
2. 對非加性方法掃描 similarity_budget，找出 LPIPS ≈ L_ref 之設定
3. 寫入 configs/nonadditive.yaml，輸出校準曲線至 experiments/stage0/
"""
