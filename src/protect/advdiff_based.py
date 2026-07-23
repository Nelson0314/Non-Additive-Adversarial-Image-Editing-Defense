"""AdvDiff-based 非加性保護 — SPEC.md §4.3。

- DDIM inversion 取得 z_T（本專案設計；原文由隨機噪聲出發）
- 兩注入點（DDIM 形式）：每步改 epsilon（Alg.2 line 6）、
  起始噪聲注入（Alg.2 line 11）
- guidance 僅在反向過程末段 (0, 0.2] 施加（附錄 E）
- reward 採 SPEC §4.2 方案一（注意力抑制損失）
- 對 z_T 加入投影約束以貼近原圖（本專案設計）
"""
