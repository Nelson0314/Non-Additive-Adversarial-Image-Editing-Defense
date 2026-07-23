"""GrIDPure — SPEC.md §5.3（Zhao et al., CVPR 2024）。

四步驟：grid 切分（512×512 → stride 128，九個 256×256 + 四角落組成第十個）
→ 每 grid 以小步 SDEdit（0.1T，pixel-space 無條件 diffusion model）淨化
→ 合併（重疊區取平均）→ 與前一輪混合 x_{i+1} = (1−γ)·x̃_i + γ·x_i。

成本警告：單張 512×512 於 V100 約 2 分鐘，執行前須估算總時數。
預設參數依 L1.3 對官方 repo 之確認結果校正。
"""
