# 設計規格：紋理重相位（紋理重相位）

2026-08-13 定案。判準沿用 `docs/FINDINGS.md`／`docs/DECISIONS.md`，本檔不新增判準。

---

## 1. 問題與機制假設

三筆既有事實指向同一個機制：

- FND-013：同 τ 下非加性換位移的效率約差三倍（τ=0.20 時為最佳 baseline 的 0.31×）。
- FND-004：位移場（flow）與同失真隨機對照無法區分。
- FND-026：加性擾動住在高頻（Mist 的 L∞ 僅 0.0137）。

合起來的解釋是：加性方法把能量放在高頻，VAE encoder 對高頻敏感而人眼不敏感；
既有的非加性形式（warp、latent 球）產生的是自然、合法的低頻結構改動，VAE 忠實
編碼它，於是每單位可見失真換到的 latent 位移少。「自然」同時買到抗淨化
（FND-018／020 的 7/7）與低位移——這是同一件事的兩面。

**要被否證的那一句**：DISTS 的設計目標明文包含 tolerant to texture resampling
（Ding et al. 2020），人眼判紋理亦靠統計量，而 VAE encoder 的卷積特徵對相位敏感。
若此落差存在，則在同一可見失真下，相位擾動能換到比加性更大的 latent 位移。

## 2. 算子構造

文獻依據為 Random Phase Noise（Galerne, Gousseau, Morel, IEEE TIP 2011）：
把影像的傅立葉相位隨機化，微紋理的知覺外觀不變。本設計把「隨機」換成「最佳化」。

    x' = OLA( irfft2( |rfft2(P_b)| · exp(i·(∠rfft2(P_b) + g_b · m_ω · θ_b)) ) )

| 符號 | 內容 |
|---|---|
| `P_b` | Hann 窗、32×32 區塊、hop 16 取出的重疊區塊；`OLA` 為重疊相加 |
| `θ_b` | 每區塊每頻率一個相位偏移，形狀 (B, 32, 17)，**RGB 三通道共用** |
| `g_b` | 紋理度閘，固定不可學，逐區塊純量 |
| `m_ω` | 徑向頻率閘，固定不可學，歸一化半徑 < `r_min` 的頻格為 0 |

**恆等保證**：重建式取 Griffin & Lim (1984) 的最小平方解——分析窗與合成窗
各乘一次，再除以 `OLA(w²)`。**依賴的是 NOLA 而非 COLA**：COLA 是充分條件，
必要的只是 `Σw² > 0` 處處成立。故 `θ = 0` 時 `x' = x`。與
`site_warp.py` 同一條「由構造保證恆等」的原則，且必須由測試實測，不得假設。

**通道共用相位**：依 2026-08-13 的顏色結論——等亮度色度擾動的位移比 RGB 獨立
低 31%，而 RGB 獨立那組的效果全部來自它順帶改變的亮度。共用相位使擾動落在
結構而非色度上。參數量約 1024 × 544 ≈ 5.6e5，與加性 δ 的 7.9e5 同數量級。

**紋理度閘**：取結構張量的 coherence，

    g_b = (1 − ((λ₁ − λ₂)/(λ₁ + λ₂ + ξ))²) · sat(梯度能量)

邊緣（coherence 高）與平坦區（梯度能量低）皆得 g ≈ 0。前者避免鬼影，後者避免
平滑區的可見振鈴。

**徑向頻率閘**：低頻格帶著區塊的位置與結構，動它會在重疊相加後產生接縫，故
歸一化半徑低於 `r_min` 的頻格不參與擾動。

**約束**：`‖θ‖∞ ≤ θ_max ≤ π`，逐元素夾。

**已知的不精確之處**：50% 重疊的相鄰區塊各自改相位後相加會部分抵消，故「局部
幅度譜保留」是近似而非恆等。此偏差必須實測並列入報告，不得假設為零。

## 3. 兩臂與對照組

### 像素臂（像素空間，科學主體）

損失、優化器、步數、種子完全相同，**唯一變因是參數化**：

| 條件 | 參數化 |
|---|---|
| `add` | δ 加性，L∞ 投影 |
| `phase` | θ 相位，本設計 |
| `phase_rand` | 隨機 θ 同幅度，即 RPN 本身 |

損失固定為 encoder-targeted `‖E(x′) − E(y_target)‖²`（PhotoGuard 的 encoder
attack 形式）。選它的理由有二：只跑 VAE 編碼器故成本是分鐘量級；以及它與弱
baseline 的 targeted 形式同源。

`phase_rand` 自第一天即存在。FND-004 與 FND-018 兩次都是被「贏不過同失真隨機」
擋下來的，事後補對照組等於重蹈覆轍。

### latent 臂（latent，相容性檢查）

把 APA 階段二的 `δ ∈ L∞ 球` 換成 latent 上的相位 θ（8×8 區塊、hop 4），其餘
五個位置維持原生，對照現行弱 baseline。

此臂**不能**與像素臂共用損失：直接擾動 latent 會使 encoder 損失退化為平凡解。
故沿用 APA 的 `−‖D(z̄₀) − y_target‖²`。

### 外部水位

`photoguard_c`／`mist`／`dia_r` 照跑，作為絕對水位。

## 4. 預算軸與量測

- **取樣點**：三個條件在同一迭代數下跑完，沿迭代軌跡取 DISTS 最接近目標的
  那一格，記錄取到第幾步。比逐條件二分搜尋便宜，且失真沿 PGD 迭代近似單調。
- **兩個預算點**：DISTS@256 ≈ 0.075（現行非加性水位）與 ≈ 0.04（加性水位）。
- **循環論證的處置**：本算子是為了在 DISTS 下便宜而設計的，拿 DISTS 當預算軸
  有循環論證的嫌疑。處置三件：(a) 同一組結果另外在 LPIPS 對齊軸上重報一次，
  答案若翻轉就明寫；(b) 全部指標照報，不挑選；(c) `compare.html` 出同失真配對
  交由人眼判定。判準以人眼為主是專案既有規則。
- **抗淨化**：既有七個算子，另加 `ROBUSTNESS_TESTS.md` §1 的 C&R 串接
  （JPEG q75 → 0.5× Lanczos）。
- **編輯評測**：SDEdit strength 0.55（DEC-022）。未防禦編輯必須先看圖確認
  真的成功，否則抗編輯那一欄的分母不成立。

## 5. 程式落點

| 檔 | 狀態 | 內容 |
|---|---|---|
| `src/residual/texture_rephase.py` | 新增 | 算子本體，`site = "F"`（E／L／S／W 已佔用） |
| `src/baselines/encoder_target.py` | 新增 | 像素臂共用的 encoder-targeted 損失與 spec |
| `scripts/phase_ablation.py` | 新增 | 像素臂驅動 |
| `src/defense/apa_native_stage2.py` | 修改 | latent 臂：新增 latent 相位參數化分支 |
| `src/purify/ops.py` | 修改 | 新增 `jpeg75_then_resize` |
| `tests/test_texture_rephase.py` | 新增 | 見下 |

測試要釘住的性質：

1. `θ = 0` 時輸出逐位等於原圖（NOLA 恆等；測試以不滿足 COLA 的 hop 驗證）。
2. 輸出為實數（由 `rfft2`／`irfft2` 結構保證，仍須實測 dtype 與虛部）。
3. 單區塊、無重疊時，幅度譜逐位保留。
4. 梯度能傳到 `θ`。
5. 紋理閘在合成邊緣圖上接近 0、在合成雜訊紋理上接近 1。
6. `‖θ‖∞ ≤ θ_max` 投影後成立。

## 6. 風險

先寫下三個會讓它失敗的情形，避免事後合理化：

1. **相位擾動可能只是被紋理遮蔽的高頻噪聲。** 若是，它會繼承加性的抗淨化弱點，
   `retention` 不會贏。判別方式：量防禦圖的局部幅度譜偏差；構造上應接近 0，
   實測偏大即說明它在造新能量而非重排相位。
2. **`phase_rand` 追平 `phase`。** 代表最佳化沒有貢獻，與 FND-004／018 同型。
   此時仍有一個可報的結論（RPN 本身作為零成本防禦的水位），但主張須降級。
3. **紋理閘讓有效自由度塌陷。** 真實照片的紋理區佔比可能不足以支撐足夠位移。
   在跑最佳化之前先量 `g_b` 的面積佔比。

## 7. 引用

- Griffin, Lim. Signal Estimation from Modified Short-Time Fourier Transform. IEEE TASSP 32(2):236–243, 1984.（重建式、一致性投影）
- Allen, Rabiner. A Unified Approach to Short-Time Fourier Analysis and Synthesis. Proc. IEEE 65(11):1558–1564, 1977.（STFT 框架、加窗）
- Weickert. Coherence-Enhancing Diffusion Filtering. IJCV 31:111–127, 1999.（結構張量的 coherence）
- Galerne, Gousseau, Morel. Random Phase Textures: Theory and Synthesis. IEEE TIP 20(1):257–267, 2011.
- Ding, Ma, Wang, Simoncelli. Image Quality Assessment: Unifying Structure and Texture Similarity. arXiv:2004.07728.
- Salman et al. Raising the Cost of Malicious AI-Powered Image Editing. ICML 2023. arXiv:2302.06588.
- Xiao et al. Spatially Transformed Adversarial Examples. ICLR 2018. arXiv:1801.02612.
- NAPPure. arXiv:2510.14025（其可反演的三個家族為 blur／occlusion／flow，不含相位）。
