# 文獻清單

本專案至 2026-08-13 引用過的全部文獻，含網址。**這是索引不是判準來源**——
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

移出 DiffVax 的理由見 `SOURCE_AUDIT.md` §5：其免疫器吃 masked image、
只支援 inpainting、無 L∞ 預算，在無 mask 的 SDEdit 下忠實重現結構上不可能。

## 2. 本方法的來源：APA 與非加性擾動

| 論文 | 用途 | 連結 |
|---|---|---|
| **APA**（本專案弱 baseline 的原型） | 兩階段：LoRA 對齊 ＋ dual-path latent 攻擊 | [arXiv:2506.01511](https://arxiv.org/abs/2506.01511) ／ [repo](https://github.com/deep-kaixun/APA) |
| **stAdv**（ICLR 2018） | 空間變形對抗樣本，`site_warp.py` 的構造來源 | [arXiv:1801.02612](https://arxiv.org/abs/1801.02612) |
| **Lo et al.**（CVPR 2024） | 指導者的基準論文，注意力抑制損失（式 5） | 見 `archive/` 的先驗紀錄 |
| **Asymmetric VQGAN** | 解碼器側的重建改善 | [arXiv:2306.04632](https://arxiv.org/abs/2306.04632) |

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
