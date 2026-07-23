"""Seed 協定 — SPEC.md §2.3、STRUCTURE.md §2.3。

先在原圖上搜尋能產生合理編輯結果的 seed，
再以同一 seed 編輯受保護影像，確保編輯條件完全相同。
此為實驗公平性的必要條件，不可省略。DAYN 以 20 個 seed 平均。

「合理」的判準需與指導者確認；預設以編輯結果與 prompt 的
CLIP score 高於門檻為準。
"""
