"""APA-based 非加性保護 — SPEC.md §4.4。

- Stage 1：LoRA 式 ∆θ 更新（APA 式 6）
- Stage 2：trajectory-level（式 7）與 step-level（式 8、11，用 sgn(m)）注入，
  搭配中間步淨化（式 9、10）與 diffusion augmentation（式 12）
- 參數：T_a=10, N=10, ε_a=0.4, µ=0.04；variant "gc"（T=10）/"sg"（T=50）
- LoRA rank 待確認（SPEC §8 第 4 項）
"""
