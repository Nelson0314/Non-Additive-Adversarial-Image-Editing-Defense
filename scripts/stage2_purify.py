"""stage2：淨化後比較 — SPEC.md §6.3、STRUCTURE.md §4.4。

淨化清單 = {jpeg×4, blur×3, crop_resize×3, advclean_bf, advclean_bfgf, gridpure×3}
流程：淨化 → 編輯（同 stage1 seed）→ 測編輯劣化。
核心輸出：drop = (clean_effect − purified_effect) / clean_effect，
並繪「淨化強度 vs 防禦效果」曲線（勿只比單點）。
執行前務必估算時數並回報。
"""
