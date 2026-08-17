# 文獻清單

本專案至 2026-08-16 引用過的全部文獻，含網址。**這是索引不是判準來源**——
逐篇的查證細節在同目錄的其他檔案（`SOURCE_AUDIT.md`、`SURVEY.md`、
`ROBUSTNESS_TESTS.md`、`dia_apa.md`、`mist_diffvax.md`、
`promptflare_photoguard.md`、`advpaint_dia_promptflare.md`、`purify.md`）。

「狀態」欄的意思：

- **已實作** —— 程式在 `src/baselines/` 或 `src/purify/`，跑得出數字
- **已查證** —— 讀過原始碼或論文並留下紀錄，未實作
- **僅引用** —— 只在論述中引用，未查證細節

---

## 1. 防護擾動：本專案的 baseline

| 論文 | 場景 | 狀態 | 連結 |
|---|---|---|---|
| **PhotoGuard**（ICML 2023） | img2img ＋ inpainting | 已實作 | [arXiv:2302.06588](https://arxiv.org/abs/2302.06588) ／ [repo](https://github.com/MadryLab/photoguard) |
| **Mist**（ICML 2023 Oral） | 風格模仿 | 已實作 | [arXiv:2305.12683](https://arxiv.org/abs/2305.12683) ／ [arXiv:2302.04578](https://arxiv.org/abs/2302.04578) ／ [repo](https://github.com/mist-project/mist) |
| **DIA**（ICCV 2025） | inversion-based editing | 已實作（PT／R 兩變體） | [arXiv:2510.00778](https://arxiv.org/abs/2510.00778) ／ [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Hong_DIA_The_Adversarial_Exposure_of_Deterministic_Inversion_in_Diffusion_Models_ICCV_2025_paper.pdf) ／ [repo](https://github.com/sohn1029/DIA) |
| **AdvPaint**（ICLR 2025） | inpainting | 已實作（改寫為全圖 mask） | [arXiv:2503.10081](https://arxiv.org/abs/2503.10081) ／ [OpenReview](https://openreview.net/forum?id=m73tETvFkX) ／ [repo](https://github.com/JoonsungJeon/AdvPaint) |
| **PromptFlare**（ACM MM 2025） | inpainting | 已實作（改寫為全圖 mask） | [arXiv:2508.16217](https://arxiv.org/abs/2508.16217) ／ [ACM](https://dl.acm.org/doi/10.1145/3746027.3755763) ／ [repo](https://github.com/NAHOHYUN-SKKU/PromptFlare) |
| **DiffVax**（ICLR 2026） | inpainting，前饋 UNet++ | 已查證，**移出 baseline 清單** | [arXiv:2411.17957](https://arxiv.org/abs/2411.17957) ／ [repo](https://github.com/ozdentarikcan/DiffVax) ／ [專案頁](https://diffvax.github.io/) |
| **Glaze**（USENIX Sec 2023） | 風格模仿 | 僅引用 | [arXiv:2302.04222](https://arxiv.org/abs/2302.04222) |
| **DCT-Shield**（ICCV 2025 Highlight） | img2img 編輯，**DCT 係數上的加性擾動** | 已查證，**待實作**，見 `SURVEY_2026-08-16.md` §1 | [arXiv:2504.17894](https://arxiv.org/abs/2504.17894) ／ [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Bala_DCT-Shield_A_Robust_Frequency_Domain_Defense_against_Malicious_Image_Editing_ICCV_2025_paper.pdf) ／ [專案頁](https://dct-shield.github.io/project-page/) |

移出 DiffVax 的理由見 `SOURCE_AUDIT.md` §5：其免疫器吃 masked image、
只支援 inpainting、無 L∞ 預算，在無 mask 的 SDEdit 下忠實重現結構上不可能。

## 2. 本方法的來源：APA 與非加性擾動

| 論文 | 用途 | 連結 |
|---|---|---|
| **APA**（本專案弱 baseline 的原型） | 兩階段：LoRA 對齊 ＋ dual-path latent 攻擊 | [arXiv:2506.01511](https://arxiv.org/abs/2506.01511) ／ [repo](https://github.com/deep-kaixun/APA) |
| **stAdv**（ICLR 2018） | 空間變形對抗樣本，已刪除的位移場模組的構造來源 | [arXiv:1801.02612](https://arxiv.org/abs/1801.02612) |
| **Lo et al.**（CVPR 2024） | 指導者的基準論文，注意力抑制損失（式 5） | 見 `archive/` 的先驗紀錄 |
| **Asymmetric VQGAN** | 解碼器側的重建改善 | [arXiv:2306.04632](https://arxiv.org/abs/2306.04632) |

## 2b. 紋理重相位的構造來源與相位擾動的前例

2026-08-16 補。前五筆是本方法**實際依賴**但索引檔漏收的；後五筆是同日查證
新增。逐篇的查證細節在 `SURVEY_2026-08-16.md`。

| 論文 | 在本專案的角色 | 連結 |
|---|---|---|
| **Galerne, Gousseau, Morel**. Random Phase Textures: Theory and Synthesis. IEEE TIP 20(1):257-267, 2011 | **紋理重相位的構造來源**：隨機化傅立葉相位可保留微紋理外觀。`phase_rand` 即 RPN 本身 | — |
| **Ding, Ma, Wang, Simoncelli**. Unifying Structure and Texture Similarity. TPAMI 2021 | DISTS。預算軸，以及「對紋理重取樣寬容」這半個機制假設的依據 | [arXiv:2004.07728](https://arxiv.org/abs/2004.07728) |
| **Madry et al.** Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018 | **PGD 本身**。`param_pgd.py` 的 sign 更新式此前沒有引用出處 | [arXiv:1706.06083](https://arxiv.org/abs/1706.06083) |
| **Oppenheim & Lim**. The Importance of Phase in Signals. Proc. IEEE 69(5):529-541, 1981 | 相位比幅度更決定可辨識內容。**本方法必須正面處理的矛盾**——兩個閘就是答案 | [PDF](https://dsp-group.mit.edu/wp-content/uploads/2024/11/ImportancePhaseSignals_1981.pdf) |
| **Griffin, Lim**. Signal Estimation from Modified Short-Time Fourier Transform. IEEE TASSP 32(2):236–243, 1984 | **重建式 `OLA(w²·x)/OLA(w²)` 的出處**——它是「由被修改過的 STFT 還原訊號」的最小平方最佳解。本方法逐區塊轉相位後的係數一般不一致，該式即把它投影回一致集合；`amplitude_deviation` 就是這個投影誤差 | — |
| **Allen, Rabiner**. A Unified Approach to Short-Time Fourier Analysis and Synthesis. Proc. IEEE 65(11):1558–1564, 1977 | STFT 的分析／合成框架：切塊、加窗、重疊相加 | — |
| **Weickert**. Coherence-Enhancing Diffusion Filtering. IJCV 31:111–127, 1999 | 紋理閘用的 coherence `(λ₁−λ₂)/(λ₁+λ₂)`，這個量與名稱的來源 | [doi:10.1023/A:1008009714131](https://doi.org/10.1023/A:1008009714131) |
| 結構張量本身（Förstner & Gülch 1987、Bigün & Granlund 1987） | 紋理閘的梯度外積統計 | — |
| **Perturbing the Phase**（2026） | **相位攻擊的前例**：只動相位、幅度逐點保留，且相位攻擊比幅度攻擊有效。其幅度相依的相位上限 `2·arcsin(eps/(2·mag))` 可解決本專案「固定 theta 不等於固定失真」的問題 | [arXiv:2602.06577](https://arxiv.org/html/2602.06577) |
| **Black box phase-based adversarial attacks on image classifiers**（JEI 34(1):013041, 2025） | 相位攻擊的前例，黑盒、分類任務 | [doi:10.1117/1.JEI.34.1.013041](https://doi.org/10.1117/1.JEI.34.1.013041) |
| **PPD: Permutation Phase Defense** | 把相位置換當**防禦**使用 | [arXiv:1812.10049](https://arxiv.org/pdf/1812.10049) |
| **Interpreting Structured Perturbations**（2025） | Glaze／Nightshade 在頻域上是「沿影像主頻率軸重分配能量」，不是隨機噪聲。支撐本方法的框架 | [arXiv:2512.08329](https://arxiv.org/abs/2512.08329) |
| **NeuralRemaster: Phase-Preserving Diffusion** | 相位承載結構，擴散脈絡下的確認 | [arXiv:2512.05106](https://arxiv.org/html/2512.05106v2) |

**本方法的訊號處理骨架全部是教科書內容**：切塊、加窗、FFT、重疊相加是 STFT
（Allen & Rabiner 1977），`OLA(w²)` 正規化是 Griffin & Lim (1984) 的最小平方解，
Hann 窗與 COLA／NOLA、實數 FFT 的共軛對稱都是標準性質。**這不是缺點——方法
建立在成熟的基礎上是優點——但必須引用。** 真正屬於本專案的是兩個閘、
把 RPN 的隨機換成最佳化，以及 `fx=0`／`fx=N/2` 兩行的處理（Galerne 不切塊，
遇不到那個問題）。

**新穎性主張因此收窄**：不能宣稱首次把相位擾動用於對抗攻擊。可宣稱的是
首次把**加窗重疊區塊的頻譜相位旋轉**用於**擴散編輯防護**，並以兩個由原圖
決定的閘限制作用範圍。理由見 `SURVEY_2026-08-16.md` §2.4。

## 3. 淨化與破解：對防護擾動的攻擊

| 論文 | 內容 | 狀態 | 連結 |
|---|---|---|---|
| **DiffPure**（ICML 2022） | 擴散淨化，本專案的 `diffpure` 算子 | 已實作 | [arXiv:2205.07460](https://arxiv.org/abs/2205.07460) ／ [repo](https://github.com/NVlabs/DiffPure) ／ [專案頁](https://diffpure.github.io/) |
| **IMPRESS**（NeurIPS 2023） | 以重建一致性檢驗擾動，本專案的 `impress` 算子 | 已實作 | [arXiv:2310.19248](https://arxiv.org/abs/2310.19248) ／ [repo](https://github.com/AAAAAAsuka/Impress) |
| **PDM-Pure / Pixel is a Barrier** | 像素空間擴散當通用淨化器，繞過幾乎所有既有防護 | 已查證，協定見 `ROBUSTNESS_TESTS.md` §2 | [arXiv:2404.13320](https://arxiv.org/abs/2404.13320) |
| **NAPPure** | **針對非加性擾動的淨化**（模糊／遮擋／flow field） | 已查證，協定見 `ROBUSTNESS_TESTS.md` §3 | [arXiv:2510.14025](https://arxiv.org/html/2510.14025) |
| **Real-world Transformations** | JPEG ＋ resize **串接**後防護全面失效 | 已查證，協定見 `ROBUSTNESS_TESTS.md` §1 | [arXiv:2604.23688](https://arxiv.org/html/2604.23688) |
| **Adversarial Perturbations Cannot Reliably Protect Artists** | 影像上採樣等低成本手段即可繞過 | 僅引用 | [arXiv:2406.12027](https://arxiv.org/pdf/2406.12027) |
| **Purify Once, Edit Freely** | model mismatch 下破解防護 | 僅引用 | [arXiv:2603.13028](https://arxiv.org/html/2603.13028) |
| **BridgePure**（NeurIPS 2025） | 洩漏少量 (clean, protected) 配對即可學出反函數。**對前饋 generator 是直接威脅** | 僅引用 | [arXiv:2412.21061](https://arxiv.org/abs/2412.21061) |
| **AdverseCleaner** | 導向濾波去噪，本專案的 `adverse_cleaner` 算子 | 已實作 | [repo](https://github.com/lllyasviel/AdverseCleaner) |
| **Countering Adversarial Images using Input Transformations** | crop-resize 等輸入變換的來源 | 僅引用 | [arXiv:1711.00117](https://arxiv.org/abs/1711.00117) |
| **IPT-V2**（NTIRE 2023 相關） | CNN 去噪算子的替代對象，**權重未公開** | 缺權重，未實作 | [arXiv:2404.00633](https://arxiv.org/abs/2404.00633) |

## 4. 抗淨化的防禦側

| 論文 | 內容 | 連結 |
|---|---|---|
| **AntiPure**（ICCV 2025） | 形式化 anti-purification；頻率引導 ＋ 錯誤 timestep 引導。**加性** | [arXiv:2509.13922](https://arxiv.org/abs/2509.13922) |
| **BlurGuard**（NeurIPS 2025） | 對噪聲做 per-region 高斯模糊以重塑頻譜；主張由 imperceptibility 轉向 irreversibility。**加性** | [arXiv:2511.00143](https://arxiv.org/html/2511.00143) |
| **DiffusionGuard** | 對抗擴散編輯的 robust defense | [arXiv:2410.05694](https://arxiv.org/pdf/2410.05694) |
| **GuardDoor** | 保護性後門，需控制模型端 | [arXiv:2503.03944](https://arxiv.org/pdf/2503.03944) |

## 5. 語意注入與 amortization

| 論文 | 內容 | 連結 |
|---|---|---|
| **Universal Image Immunization via Semantic Injection** | 單一 image-agnostic UAP；target injection ＋ source suppression 雙損失（Eq.5／6）。**加性** | [arXiv:2602.14679](https://arxiv.org/html/2602.14679) |
| **DiffVax** | 前饋 immunizer，optimization-free | 見 §1 |

2026-08-13 曾把 Eq.5／6 移植到非加性載體上，四個方向全部否決，
歸檔在 `FINDINGS.md` 末段。

## 6. 顏色空間的對抗擾動

2026-08-13 的顏色通道實驗（已放棄）查到的前例，**全部是分類攻擊，
沒有擴散編輯防禦**：

| 論文 | 內容 | 連結 |
|---|---|---|
| **Spatial Chroma-Shift** | 只在 YUV 色度通道做空間變形 | [arXiv:2108.02502](https://arxiv.org/abs/2108.02502) |
| **PerC-AL**（CVPR 2020） | 以 CIEDE2000 感知色差取代 L∞ 當約束 | [arXiv:1911.02466](https://arxiv.org/abs/1911.02466) |
| **Adversarial Perturbations Prevail in the Y-Channel** | **相反結論**：對抗能量集中在亮度而非色度 | [arXiv:2003.00883](https://arxiv.org/pdf/2003.00883) |
| **Chroma Backdoor** | UV 通道的高頻小波注入 | [doi:10.3390/sym17071014](https://doi.org/10.3390/sym17071014) |

## 7. 對防護有效性的否定證據

| 論文 | 結論 | 連結 |
|---|---|---|
| **Is Perturbation-Based Image Protection Disruptive to Image Editing?**（ICIP 2025） | 多數情況下受保護影像的編輯**仍符合 prompt**；加噪聲甚至可能提高與 prompt 的關聯 | [arXiv:2506.04394](https://arxiv.org/abs/2506.04394) |
| **Off-The-Shelf Image-to-Image Models Are All You Need** | — | [arXiv:2602.22197](https://arxiv.org/html/2602.22197) |

第一篇與本專案的 FND-024／029／030 獨立測到同一現象，是可引用的外部確證。

## 8. 底層模型與方法

| 項目 | 用途 | 連結 |
|---|---|---|
| **SDEdit** | 本專案的攻擊方流程 | [repo](https://github.com/ermongroup/SDEdit) |
| **guided-diffusion** | DiffPure 的檢查點來源 | [repo](https://github.com/openai/guided-diffusion) |
| **Score SDE** | — | [repo](https://github.com/yang-song/score_sde) |
| **SD3 / Rectified Flow Transformers** | 說明 SD 3.x 起改為 MMDiT、無 cross-attention | [arXiv:2403.03206](https://arxiv.org/abs/2403.03206) |
| **Stable Diffusion v1.4** | 本專案的模型 | [HuggingFace](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) |

## 9. 指標

| 項目 | 用途 | 連結 |
|---|---|---|
| **ShiftTolerant-LPIPS** | 位移容忍的感知指標，空間變形的量測參考 | [arXiv:2207.13686](https://arxiv.org/abs/2207.13686) |
| **Deep Detail Network**（CVPR 2017） | 去雨，`local_acutance` 的區塊尺度參考 | [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Fu_Removing_Rain_From_CVPR_2017_paper.html) |
| **Attentive GAN**（CVPR 2018） | 去雨滴 | [CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Qian_Attentive_Generative_Adversarial_CVPR_2018_paper.html) |

## 10. 其他曾查證但未納入的

`SDA`（[arXiv:2505.19425](https://arxiv.org/abs/2505.19425)）、
`Attention Attack`（[arXiv:2509.10359](https://arxiv.org/abs/2509.10359)）、
`Structured Perturbations`（[arXiv:2512.08329](https://arxiv.org/abs/2512.08329)）、
`SIFM`（[arXiv:2512.14320](https://arxiv.org/html/2512.14320)）、
`DANP`（[arXiv:2512.14333](https://arxiv.org/abs/2512.14333)）、
`TDAE`（[arXiv:2512.14341](https://arxiv.org/html/2512.14341)）、
`HYPE-EDIT-1`（[arXiv:2602.00105](https://arxiv.org/pdf/2602.00105)）、
`Spatially Transformed AE 的後續`（[arXiv:1804.07493](https://arxiv.org/abs/1804.07493)）。

納入與否的理由在 `SURVEY.md` 與 `SOURCE_AUDIT.md`。

**NatADiff**（ICLR 2026，[arXiv:2505.20934](https://arxiv.org/abs/2505.20934)）——
Collins、Vice、French、Mian（Univ. of Western Australia）。攻擊**分類器**而非
text-guided editing，故不是本專案的 baseline，但有兩點可引用。

方法：用 SD1.5 生成 natural adversarial sample。在 classifier-free guidance 上多
加一個指向真類與對抗類**交集**的方向 `v_{y∩ỹ} = ε_θ(x_t,t,y∩ỹ) − ε_θ(x_t,t)`，
交集的條件是文字 prompt「`<ỹ 類名> and <y 類名>`」，不經過 victim classifier；
再配上分類器增廣（對 `x̂_0` 套可微變換後平均 logits）與 time-travel sampling。
`μ=0` 時退化成 AdvDiff。

兩點可引用：

1. **加性與非加性的 transferability 差距**。ResNet-50 surrogate 下 white-box ASR
   都近 100%，但平均 ASR：PGD 17.6%、AutoAttack 18.4%、AdvDiff 45.7%、
   NatADiff 68.2%。論文歸因於 ε-ball 擾動依賴的對抗口袋在不同架構間不對齊。
2. **對淨化的抵抗**（附錄 M）。變換式淨化（裁切／旋轉／灰階多視角平均）對它
   無實質削弱；DiffPure 只把平均 ASR 降 **7.9 個百分點**。adversarial training
   的 ResNet／Inception 也沒有提供有意義的防護。

成本 103.1 s/張（RTX 4090），對照 PGD 的 0.3 s。
