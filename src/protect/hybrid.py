"""Hybrid 非加性保護 — SPEC.md §4.5（本專案設計）。

以 APA 為骨架（inversion + 兩階段解耦），
Stage 2 的注入改用 AdvDiff 的兩注入點形式。
實驗須驗證是否優於單獨任一者；若未優於，如實報告。
"""
