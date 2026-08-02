# 文獻調查：擾動式影像保護對抗擴散編輯

<!-- STATUS-BLOCK -->
| | |
|---|---|
| **狀態** | 現行。2026-08-03 建立，涵蓋至 2026 年 6 月的 arXiv |
| **範圍** | 只收與本專案研究問句直接相關者：白盒、外掛模組、抵抗文字引導編輯、匹配人眼可辨失真 |
| **相關** | 設計依據 `docs/specs/2026-08-02-e31-positive-control.md` §2；主張索引 `docs/LEDGER.md` |

> 本檔按**問句**組織而非按論文組織。每一節先給答案，再列證據。
> 論文只在能回答某個問句時出現，不做逐篇摘要。

---

## 0. 一頁摘要

| 問句 | 目前的答案 | 對本專案的意涵 |
|---|---|---|
| 擾動式保護真的擋得住編輯嗎？ | **多半擋不住**，且受保護影像常**更**服從 prompt | E29 的否定結果是已知現象的白盒複現，不是實作瑕疵 |
| 領域用什麼判準？ | 尚未收斂。2026 年的 TPAMI 論文仍用「與未防禦編輯的距離」 | 本專案的判準方法學領先，是可發表的材料 |
| 目標函數有幾族？ | 至少四族，本專案只跑過其中一族 | `targeted`、`suppress` 待跑；**早期步噪聲範數**這一族從未考慮過 |
| 標準失真預算？ | ε∞ = 16/255；LPIPS 0.267–0.362 | 本專案的 LPIPS 低一個量級，但 L∞ 高六倍 |
| 標準攻擊設定？ | 多數是**遮罩式 inpainting**；img2img 用 strength 0.2–0.3 | 本專案用全域 SDEdit strength 0.5，是文獻中最難的設定 |
| 保護撐得住淨化嗎？ | 撐不住。現成的 img2img 模型即可清除 | 抗淨化是附帶目標，但這條線的天花板很低 |
| 非加性有人做嗎？ | 有，且 2025–2026 開始被明確提出 | 本專案的 site S／site C 與該線同向，但那條線也還沒有正面結果 |

---

## 1. 擾動式保護真的擋得住編輯嗎

**答案：多半擋不住，而且方向可能是反的。**

- [Is Perturbation-Based Image Protection Disruptive to Image Editing?](https://arxiv.org/abs/2506.04394)
  （ICIP 2025）測 PhotoGuard／Mist／Glaze，SD v1.5、五個隨機種子，判準用
  CLIP-S 與 PAC-S++ 的 image-text alignment。結論是保護不阻止編輯，且
  **61.82–67.79% 的受保護影像對 prompt 的對齊度上升**。
  作者的說法是「adversarial perturbations enhance alignment」。

- 本專案 E29 §4.2 的 site P 為 0.1277 對未防禦的 0.1202（+0.0075），方向與
  量級都落在該區間內。**這使 E29 由「我們的方法失敗」升格為「已知現象的
  白盒複現」。**

- 即使是報正面結果的論文，幅度也有限。[DiffusionGuard](https://arxiv.org/pdf/2410.05694)
  （ICLR 2025）在 inpainting 上的人類偏好勝率是 41.9%（見過的遮罩）與
  39.0%（未見過的遮罩），對照 PhotoGuard 的 25.0%／22.2%。
  **勝率 42% 不是「擋下來了」，是「比另一個方法好一點」。**

- [SIFM](https://arxiv.org/html/2512.14320) 報的 ISR 較高（79–97%），
  但其 ISR 的定義是聯集（語意不符**或**感知劣化），且評測的是直接的
  text-guided editing 而非高 strength 的全域 SDEdit。

**注意事項。** 上列數字來自不同的實作、資料集與判準，不可直接互比。
可靠的只有方向：這條線沒有一個方法能在多數情況下讓編輯失敗。

---

## 2. 判準：領域還沒收斂

**答案：三種判準並存，彼此不相容，而最常見的那一種本專案已證實無效。**

| 判準族 | 代表 | 量的是什麼 | 本專案的判定 |
|---|---|---|---|
| 與未防禦編輯的距離 | TDAE（TPAMI 2026）用 PSNR／SSIM／FSIM／VIFP／LPIPS | 輸出移動了多少 | **無效**。E25 §1.2 在 726 格中語意失敗 0 格；E29 §4.3 的 `edit_shift` 達 0.42–0.47 而編輯照常發生 |
| 影像—文字對齊 | ICIP 2025 用 CLIP-S／PAC-S++ | 輸出服不服從 prompt | 可用，但 **CLIP 未通過本專案的 edit_effect 對照**（E25 §1.1），SigLIP 通過 |
| 聯集式 | SIFM 的 ISR（MLLM 判定）；Attention Attack 的 Caption Similarity + semantic IoU | 語意不符 **或** 感知劣化 | 本專案 E31 採用，但用 NR 品質指標取代 MLLM |

三點值得記下：

1. **2026 年的 TPAMI 論文仍在用第一族。** [TDAE](https://arxiv.org/html/2512.14341v2)
   明確以「較低的 PSNR／SSIM 與較高的 LPIPS 代表免疫更成功」為判準。
   本專案兩次獨立量到該量與編輯是否失敗不對應。
2. **判準的選擇會改變結論的方向。** 同一批資料在第一族下看起來防禦有效
   （`edit_shift` 0.42），在第二族下看起來防禦無效甚至有害（Δsiglip +0.0075）。
3. **聯集式判準的兩半難度差很多。** 在高 strength 的全域編輯下，輸出主要由
   prompt 重新生成，「語意不符」在原理上幾乎不可達成；文獻上真正被達成的是
   「感知劣化」那一半。E25 之後本專案只取了前半，這是 E31 要補的。

---

## 3. 目標函數：至少四族，本專案只跑過一族

| 族 | 形式 | 代表 | 本專案的狀態 |
|---|---|---|---|
| **A. 無目標輸出距離** | max d(y_def, y_orig) | PhotoGuard diffusion attack 的無目標變體、TDAE | **只有這一族跑過**。59 個有記錄的 `env.json` 100% 是它（E29 §5.4）。它最大化的正是已被判定不對應防禦成功的量 |
| **B. 有目標** | min d(y_def, y_target)，y_target 常取灰圖 | PhotoGuard encoder／diffusion attack | 已實作、有測試，**從未在真實 SD 上跑過**。E31 待跑 |
| **C. 表示層／注意力** | 壓低內容 token 的注意力質量，或推離參考注意力圖 | [Attention Attack](https://arxiv.org/abs/2509.10359)（ACM MM 2025）、[DANP](https://arxiv.org/abs/2512.14333)、SIFM（中間層特徵） | `suppress` 已實作，**從未在真實 SD 上跑過**。E31 待跑 |
| **D. 早期步的噪聲預測範數** | max ‖ε_θ(z_t, t=T, c)‖₂，只在**第一個**去噪步施力 | [DiffusionGuard](https://arxiv.org/pdf/2410.05694) | **本專案從未考慮過**。見下方 |

**D 族值得單獨記一筆。** DiffusionGuard 的主張是不必對整條去噪鏈最佳化，只要
把**初始去噪步**的預測噪聲範數推大即可，理由是初期步決定了整體佈局。實測
效果是 inpainting 上的 SOTA，且最佳化成本約 11 秒對 PhotoGuard 的 90 秒。

對本專案的意涵：它與 `crossattn` 一樣不需要跑完 `n_edit` 步的鏈，成本結構相同
（E0 的成本模型中 `0.304·n_edit` 那一項消失），但著力點完全不同。若 E31 的
`suppress` 失敗，D 族是下一個成本可負擔的候選。

**另一個設計細節。** Attention Attack 用**原圖的自動生成 caption 當作編輯
prompt 的代理**，因為防禦方不知道攻擊方會用什麼 prompt。本專案是白盒設定、
直接用真實的編輯 prompt，故不受此限——但這也意味著本專案的結果是該線的
**上界**：連知道 prompt 都擋不住，不知道 prompt 只會更差。

---

## 4. 失真預算：兩個軸，本專案在兩軸上的位置相反

**標準是 ε∞ = 16/255 的 L∞ 球**（≈ 0.063），源自 PhotoGuard 並被後續沿用。
DiffusionGuard 另報 6/255 下仍有效。SIFM 用 ε = 0.03、100 步。

以 LPIPS 計，文獻的運作點是 0.267–0.362（DCT-Shield 自報 0.267）。

本專案 τ_lpips = 0.10 的實測（E31 規格 §3）：

| 軸 | 本專案 | 文獻 | 比 |
|---|---|---|---|
| LPIPS | 0.0856 | 0.267–0.362 | 低 3–4 倍 |
| RMS | 0.0319 | 約 0.06（PGD 近飽和） | 低 2 倍 |
| L∞ | 0.373 | 0.063 | **高 6 倍** |
| 超過 16/255 的像素比例 | 5.3% | 接近 100% | — |

**結論：「本專案的預算比文獻低 5–8 倍」只在 LPIPS 上成立。** 本專案的擾動是
稀疏尖峰型（95% 的像素遠低於文獻球、5% 遠高於），這是 `beta_linf = 0` 的
直接後果。跨論文比較預算時必須同時報兩個軸。

---

## 5. 威脅模型與攻擊設定：本專案選了最難的那一個

| 設定 | 文獻主流 | 本專案 |
|---|---|---|
| 編輯型態 | **遮罩式 inpainting**（PhotoGuard、DiffusionGuard、Anti-Inpainting、Structure Disruption） | 全域 SDEdit |
| img2img 的 strength | 0.2–0.3（PhotoGuard 的 SDEdit 評測） | **0.5** |
| guidance scale | 未必明說，多為預設 7.5 | 7.5（E26 修正後） |
| 攻擊方模型 | SD v1.5／v2／SDXL／SD3／InstructPix2Pix | stock SD v1.4 |

**inpainting 對防禦方有結構性優勢**：遮罩外的區域保留原像素，擾動整片留在
條件裡；遮罩內的內容雖然重生成，但要與遮罩外縫合，故受條件影響很大。

**全域 SDEdit 在 strength = 0.5 下相反**：整張圖被加到一半的噪聲再重新去噪，
擾動的高頻成分大部分被噪聲淹沒，而 prompt 有很大的自由度重新生成內容。
在這個設定下要求「輸出語意不符 prompt」，接近要求防禦壓過整條文字條件。

E31 把 strength 0.3 納入網格正是為了量出這一項的貢獻，而不是用推論帶過。

---

## 6. 破解側：這條線的天花板

**答案：現成的 img2img 模型就能清掉保護，且模型越強清得越乾淨。**

- [Off-The-Shelf Image-to-Image Models Are All You Need To Defeat Image
  Protection Schemes](https://arxiv.org/html/2602.22197)（2026-02）測 8 個案例、
  6 類保護（UnGANable、PRC Watermark、VINE、SIREN、Mist、Tree-Ring）。
  能力排序 **GPT-4o > FLUX > SD3 > SDXL > SD1.5**，越大的模型越會清。
  作者的結論是這類保護提供的是「false sense of security」。
- [Purify Once, Edit Freely](https://arxiv.org/html/2603.13028)：異質模型本身
  即為淨化器，攻擊方先清再編輯。
- [Do Protective Perturbations Really Protect Portrait Privacy under Real-world
  Image Transformations?](https://arxiv.org/pdf/2604.23688)：縮放與色彩壓縮
  這類日常變換就會削弱保護。
- IMPRESS、CAT、BlurGuard 各自從不同角度證實同一件事。

**對本專案的意涵。** 抗淨化在本專案是附帶目標（「有更好、沒有也沒關係」），
這是對的取捨——即使做出抗淨化的優勢，這條線的天花板仍由上列結果決定。
但它也提高了否定結果的價值：本專案量的是**淨化之前**保護就已經無效，
比「淨化之後才無效」更前面一步。

---

## 7. 非加性與結構化擾動

**答案：這條線 2025–2026 開始被明確提出，但同樣沒有正面結果。**

- [Interpreting Structured Perturbations in Image Protection Methods for
  Diffusion Models](https://arxiv.org/abs/2512.08329)：分析 Glaze 與 Nightshade
  的擾動，結論是它們是「structured, low-entropy perturbations tightly coupled
  to underlying image content」，在頻域上**沿影像自身的主頻軸重新分配能量**
  而非加入擴散式雜訊。這解釋了為何它們視覺上不顯眼卻穩定可偵測。
  作者明講後續方向應「移出結構化、與內容耦合的擾動」，並點名非加性。
- **STP-Diff**（Information Fusion 2025）：spatial transformation perturbations，
  以幾何扭曲像素座標取代加性雜訊，並只作用在非顯著區域以抵抗擴散式淨化。
  這與本專案的 site S（空間變形）是同一個構想。
- **NAPPure**（[arXiv:2510.14025](https://arxiv.org/html/2510.14025)）：從**淨化側**
  處理非加性擾動，把受擾影像的生成過程建模成一個變換再聯合最佳化。
  它的存在本身說明非加性擾動已被視為需要專門處理的一類。

**對本專案的意涵。** 非加性不是本專案獨有的想法，但本專案在**量測方法**上
走得比較遠：等 LPIPS 多臂探針、`local_acutance_dev`、`local_chroma_bias`
這三項是為了回答「非加性到底用什麼買到它的效果」而做的，而上列論文都沒有
處理這個問題（STP-Diff 只報 LPIPS 與 SSIM）。**「匹配失真」三次被證明是假的**
這個結果本身就是對該線的貢獻。

---

## 8. 尚未有人回答的

按對本專案的相關性排序：

1. **在高 strength 的全域 img2img 上，有沒有任何方法能讓編輯失敗？**
   查不到正面案例。文獻的成功案例集中在 inpainting 與低 strength。
2. **在匹配失真的條件下，非加性能否勝過加性？** 沒有論文用可通過判別的
   保真約束做過這個比較——STP-Diff 只報 LPIPS 與 SSIM，而 E20 已證明
   SSIM 會補貼模糊。
3. **判準的三族之間如何換算？** 沒有人報過同一批資料在三族判準下的結果。
   本專案的 `runs/p12_isr_rejudge/` 有這個材料。
4. **早期步噪聲範數（D 族）在全域 SDEdit 上有效嗎？** DiffusionGuard 只測
   inpainting。

---

## 9. 引用清單

依本檔出現順序。

| 代號 | 標題 | 出處 |
|---|---|---|
| ICIP 2025 | Is Perturbation-Based Image Protection Disruptive to Image Editing? | [arXiv:2506.04394](https://arxiv.org/abs/2506.04394) |
| DiffusionGuard | A Robust Defense Against Malicious Diffusion-based Image Editing | [arXiv:2410.05694](https://arxiv.org/pdf/2410.05694)（ICLR 2025） |
| SIFM | Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity | [arXiv:2512.14320](https://arxiv.org/html/2512.14320) |
| TDAE | Towards Transferable Defense Against Malicious Image Edits | [arXiv:2512.14341](https://arxiv.org/html/2512.14341v2)（TPAMI） |
| Attention Attack | Immunizing Images from Text to Image Editing via Adversarial Cross-Attention | [arXiv:2509.10359](https://arxiv.org/abs/2509.10359)（ACM MM 2025） |
| DANP | Dual Attention Guided Defense Against Malicious Edits | [arXiv:2512.14333](https://arxiv.org/abs/2512.14333) |
| PhotoGuard | Raising the Cost of Malicious AI-Powered Image Editing | [arXiv:2302.06588](https://arxiv.org/pdf/2302.06588) |
| Off-The-Shelf | Off-The-Shelf Image-to-Image Models Are All You Need To Defeat Image Protection Schemes | [arXiv:2602.22197](https://arxiv.org/html/2602.22197) |
| Purify Once | Purify Once, Edit Freely: Breaking Image Protections under Model Mismatch | [arXiv:2603.13028](https://arxiv.org/html/2603.13028) |
| Real-world | Do Protective Perturbations Really Protect Portrait Privacy under Real-world Image Transformations? | [arXiv:2604.23688](https://arxiv.org/pdf/2604.23688) |
| Structured | Interpreting Structured Perturbations in Image Protection Methods for Diffusion Models | [arXiv:2512.08329](https://arxiv.org/abs/2512.08329) |
| NAPPure | Adversarial Purification for Robust Image Classification under Non-Additive Perturbations | [arXiv:2510.14025](https://arxiv.org/html/2510.14025) |
| BlurGuard | A Simple Approach for Robustifying Image Protection Against AI-Powered Editing | [arXiv:2511.00143](https://arxiv.org/html/2511.00143v1) |
| STP-Diff | Synergistic fusion of spatial transformation perturbations and diffusion models for robust face privacy protection | Information Fusion 2025 |
