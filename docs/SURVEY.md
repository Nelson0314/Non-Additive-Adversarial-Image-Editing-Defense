# 文獻調查：擾動式影像保護對抗擴散編輯

<!-- STATUS-BLOCK -->
| | |
|---|---|
| **狀態** | 現行。2026-08-03 建立並於同日補入基準論文，涵蓋至 2026 年 6 月的 arXiv |
| **範圍** | 只收與本專案研究問句直接相關者：白盒、外掛模組、抵抗文字引導編輯、匹配人眼可辨失真 |
| **相關** | 設計依據 `docs/specs/2026-08-03-lo-aligned-protocol.md`；主張索引 `docs/LEDGER.md` |

> 本檔按**問句**組織而非按論文組織。每一節先給答案，再列證據。
> 論文只在能回答某個問句時出現，不做逐篇摘要。

---

## 0. 基準論文

其餘各節都是圍繞這一篇的脈絡，**先讀這一段**。

> Ling Lo, Cheng Yu Yeo, Hong-Han Shuai, Wen-Huang Cheng,
> **Distraction is All You Need: Memory-Efficient Image Immunization against
> Diffusion-Based Image Editing**, CVPR 2024, pp. 24462–24471.
> [CVF open access](https://openaccess.thecvf.com/content/CVPR2024/html/Lo_Distraction_is_All_You_Need_Memory-Efficient_Image_Immunization_against_Diffusion-Based_CVPR_2024_paper.html)

| 面向 | 該論文的做法 |
|---|---|
| 方法 | 加性 `x_adv = x + δ`；損失是遮罩內 cross-attention 反應的 L1（式 5） |
| 更新 | PGD sign + L∞ 投影；逐 timestep 反傳再平均（Algorithm 1） |
| 約束 | **只有 L∞ ≤ 0.06**，N = 100，T = 10，所有方法一致 |
| 判準 | Table 1：PSNR↓ SSIM↓ VIFp↓ FSIM↓ LPIPS↑，全部是第 1 類「與未防禦編輯的距離」 |
| baseline | PhotoGuard 的 encoder attack 與 diffusion attack |
| 資料 | 擴散生成 150 張、3 物件 × 2 prompt、**20 種子平均** |
| 自陳限制 | 模糊或 JPEG 壓縮可能消掉免疫效果 |

**這一篇的地位不同於本檔其他論文**：其約束、判準與 baseline 已定為本專案的
必要對齊項（使用者 2026-08-03）。逐項的對齊狀態見新規格 §3。

它同時決定了本檔以下各節的讀法：第 2 節說「第 1 類判準無效」，那是**本專案
在自己的資料上的判定**，不是可以直接套到這一篇上的結論——順序見新規格 §7。

---

## 0.1 一頁摘要

| 問句 | 目前的答案 | 對本專案的意涵 |
|---|---|---|
| 擾動式保護真的擋得住編輯嗎？ | **多半擋不住**，且受保護影像常**更**服從 prompt | E29 的否定結果是已知現象的白盒複現，不是實作瑕疵 |
| 領域用什麼判準？ | 尚未收斂。基準論文與 2026 年的 TPAMI 論文都用「與未防禦編輯的距離」 | 本專案有一份該判準與另兩類之相關性的量測（ρ = 0.140／−0.207／0.014）。**2026-08-04 起可以提出**：兩個 PhotoGuard 變體 已在基準自己的協定上重現（LEDGER 3.15），且 3.24 的逐圖判讀是在該協定與預算上取得的直接證據，不是拿自己的失敗去質疑尺 |
| 目標函數有幾類？ | 至少四類，本專案只跑過其中一類 | `targeted`、`suppress` 待跑；**「只攻第一個去噪步」**這一類（兩個成員）從未考慮過 |
| 標準失真預算？ | ε∞ = 16/255；LPIPS 0.267–0.362 | 本專案的 LPIPS 低一個量級，但 L∞ 高六倍 |
| 標準攻擊設定？ | 多數是**遮罩式 inpainting**；img2img 用 strength 0.2–0.3 | 本專案用全域 SDEdit strength 0.5，是文獻中最難的設定 |
| 保護撐得住淨化嗎？ | 撐不住。現成的 img2img 模型即可清除 | 抗淨化是附帶目標，但這條線的天花板很低 |
| 非加性有人做嗎？ | 有，且 2025–2026 開始被明確提出 | 本專案的 site S／site C 與該線同向，但那條線也還沒有正面結果 |

---

## 1. 擾動式保護真的擋得住編輯嗎

**答案：多半擋不住，而且方向可能是反的。**

- [Is Perturbation-Based Image Protection Disruptive to Image Editing?](https://arxiv.org/abs/2506.04394)
  （ICIP 2025，Tang, Ayambem, Chuah, Bharati，Lehigh University）。
  SD **v1.5**，五個固定種子（9222、999、123、66、42），guidance scale／
  strength／步數一律沿用 PhotoGuard 的預設。判準是 image-text alignment
  （ITA）：CLIP-S 與 PAC-S++。**逐圖判定的是「Actual Change ≥ 0」**，即
  「由受保護影像編輯出來的圖，其對齊度沒有比由乾淨影像編輯出來的低」。

  **2026-08-04 逐項核對原文後修正（本檔原本把兩張表混為一談）：**

  | 表 | 實驗 | 保護方法 | 資料集 | CLIP-S | PAC-S++ |
  |---|---|---|---|---|---|
  | Table 1 | image-to-image | **只有 PhotoGuard** | Flickr8k（近／遠 caption） | 54.23% / 61.54% | 61.82% / 67.79% |
  | Table 2 | 風格化 | PhotoGuard | Flickr1024 | 77.54% | 68.43% |
  | Table 2 | 風格化 | Mist | WikiArt | 54.15% | 56.90% |
  | Table 2 | 風格化 | Glaze | WikiArt | 54.12% | 56.90% |

  本專案先前引用的「61.82–67.79%」**只是 Table 1 的 PAC-S++ 那一列**，
  且該表只測 PhotoGuard、不含 Mist 與 Glaze。同一張表的 CLIP-S 是
  54.23–61.54%，明顯較低。**引用時應報全距 54.12–77.54%，或明確指出是
  哪一個指標與哪一張表。** 作者的說法是 adversarial perturbation 可能
  「paradoxically increase their association with given text prompts」。

  另一個可引用的旁證：作者量到生成影像的 BRISQUE 為 17.88，原圖為 22.27，
  即**受保護影像編出來的圖在無參考品質上並沒有比較差**。這與本專案 L3 量到
  `pg_encoder` 的 Δniqe = +5.81 方向相反，差別在他們用的是 PhotoGuard 的
  預設預算與 strength，而本專案的 `pg_encoder` 跑滿 κ = 0.06、N = 100。

- 本專案 E29 §4.2 的 site P 為 0.1277 對未防禦的 0.1202（+0.0075），方向與
  該現象一致。**這使 E29 由「我們的方法失敗」升格為「已知現象的白盒複現」。**

- 本專案 L3 在基準論文自己的協定上獨立複現同一現象，且量級可比：
  `runs/l3_criterion_axes/` 的 72 格中，`pg_encoder` 有 16/24 格、
  `pg_diffusion` 有 21/24 格的 Δsiglip 為正（即受保護影像編出來的圖**更**
  服從 prompt），平均 +0.0203。**這比 ICIP 的證據更直接**：同一份資料、
  同一組評測種子、逐元素相同的 ε，而且是在 κ = 0.06 這個文獻標準預算上。

- 即使是報正面結果的論文，幅度也有限。[DiffusionGuard](https://arxiv.org/pdf/2410.05694)
  （ICLR 2025）在 inpainting 上的人類評測：DiffusionGuard 41.9%（見過的遮罩）
  與 39.0%（未見過的遮罩），對照 PhotoGuard 的 25.0%／22.2%，其餘約
  33%／39% 為平手。**勝率 42% 不是「擋下來了」，是「比另一個方法好一點」。**

  **2026-08-04 逐項核對後補上三個界定**：(a) 這是 2100 對的**兩兩 A/B**，
  不是絕對成功率；(b) 「勝」的定義是「該方法的輸出**沒有**被選中，即被判為
  較差」——所以勝率確實是保護成功的量；(c) 評分者被要求「同時依影像品質與
  編輯 prompt 服從度」判斷，**即它也是一個聯集式判準**，與 ISR 有同一個
  問題（本專案 LEDGER 1.17：聯集在實務上由劣化那一半主導）。

- [SIFM](https://arxiv.org/html/2512.14320) 報的 ISR 較高，但其 ISR 的定義是
  聯集（語意不符**或**感知劣化），且評測的是直接的 text-guided editing 而非
  高 strength 的全域 SDEdit。**2026-08-04 逐項核對後修正**：79% 與 97% 是
  兩個不同設定的數字——79% 是 SD3 上的 λ 消融最佳值，97% 是 HQ-Edit 模型上
  原始 prompt 的成績（換成未見過的 prompt 掉到 71%，另一處報 65%）。
  同一張表上**baseline 的 ISR 也都很高**（PhotoGuard encoder 84%、
  diffusion 88%、ACE 92%、SDS 91%、MIST 88%），即在該判準下這條線的方法
  彼此差距不大而絕對值都高——與人類偏好勝率 41.9% 的落差因此更需要解釋。
  本專案 L3 提供了一個機制解釋：ISR 由劣化那一半主導（LEDGER 1.17）。

**注意事項。** 上列數字來自不同的實作、資料集與判準，不可直接互比。
可靠的只有方向：這條線沒有一個方法能在多數情況下讓編輯失敗。

---

## 2. 判準：領域還沒收斂

**答案：三種判準並存，彼此不相容，而最常見的那一種本專案已證實無效。**

| 判準類別 | 代表 | 量的是什麼 | 本專案的判定 |
|---|---|---|---|
| 與未防禦編輯的距離 | **基準論文 Table 1**、TDAE（TPAMI 2026），都用 PSNR／SSIM／FSIM／VIFp／LPIPS | 輸出移動了多少 | **本專案已定為必要判準**（使用者 2026-08-03）。本專案另有兩份證據：(a) 它與語意軸幾乎不相關（ρ = 0.140）、與劣化軸負相關（ρ = −0.207），`runs/p16_criterion_correlation/`；(b) **在基準自己的協定上逐圖判讀**，該判準把 `pg_encoder` 排第一，而它的高分是用全域由模糊換來的、多數列已無從判斷編輯有沒有成功（LEDGER 3.24，`runs/figs/compare.html`）。(b) 不受新規格 §7 的順序限制——它不是拿自己的失敗去質疑尺 |
| 影像—文字對齊 | ICIP 2025 用 CLIP-S／PAC-S++ | 輸出服不服從 prompt | 可用，但 **CLIP 未通過本專案的 edit_effect 對照**（E25 §1.1），SigLIP 通過 |
| 聯集式 | SIFM 的 ISR（MLLM 判定）；Attention Attack 的 Caption Similarity + semantic IoU | 語意不符 **或** 感知劣化 | 本專案 E31 採用，但用 NR 品質指標取代 MLLM |

**MLLM 判準的可靠度已被量過**（2026-08-04 逐項核對 SIFM 原文 Table I）：

| 一致率 | 語意不符 | 品質劣化 | 最終 ISR |
|---|---|---|---|
| 人類 ↔ 人類 | 84% | **79%** | **76%** |
| MLLM ↔ MLLM | 80% | 70% | 74% |
| 人類 ↔ MLLM | 73% | 72% | 74% |

判讀者是 Gemini 2.5 Pro 與 Flash，人類是三位獨立評分者、150 個分層抽樣樣本。

兩件事：

1. **MLLM 與人類的差距（74%）小於人類之間本來就有的分歧（76%）。** 這支持
   ISR 作為代理判準。
2. **但 ISR 本身的人類一致率只有 76%，劣化那一半更只有 79%。** 也就是說
   這個判準的雜訊下限不低，尤其是劣化那一半——而本專案 L3 量到 ISR 在實務上
   正是由劣化那一半主導（LEDGER 1.17）。引用 ISR 的數字時要記得它的解析度。

**SIFM 自己的資料就顯示傳統指標不追蹤 ISR。** 其 λ 消融表（Table VII，SD3）
上 ISR 由 70% 變到 79%，而同一批列的 PSNR 只由 15.76 動到 15.85、SSIM 由
0.4747 到 0.4832、LPIPS 由 0.4915 到 0.5046——**傳統指標幾乎不動而 ISR 動了
9 個百分點**。這是由提出 ISR 的那篇論文自己提供的、支持本專案 1.16 的證據。

另一份編輯基準（HYPE-EDIT-1）報 VLM judge 與人類多數決的一致率約 80%，
但傾向過嚴，在人類仍可接受的細微改動上會判失敗。

對本專案的意涵：E31 用無參考品質指標（NIQE 等）取代 MLLM 是成本考量下的
折衷，而**折衷的代價現在有數字可引用**。若 E31 的感知劣化那一半成為結論的
關鍵，值得補一次 MLLM 判定作為交叉驗證。

三點值得記下：

1. **2026 年的 TPAMI 論文仍在用第一類。** [TDAE](https://arxiv.org/html/2512.14341v2)
   明確以「較低的 PSNR／SSIM 與較高的 LPIPS 代表免疫更成功」為判準。
   本專案兩次獨立量到該量與編輯是否失敗不對應。
2. **判準的選擇會改變結論的方向。** 同一批資料在第一類下看起來防禦有效
   （`edit_shift` 0.42），在第二類下看起來防禦無效甚至有害（Δsiglip +0.0075）。
3. **聯集式判準的兩半難度差很多。** 在高 strength 的全域編輯下，輸出主要由
   prompt 重新生成，「語意不符」在原理上幾乎不可達成；文獻上真正被達成的是
   「感知劣化」那一半。E25 之後本專案只取了前半，這是 E31 要補的。

---

## 3. 目標函數：至少四類，本專案只跑過一類

| 類 | 形式 | 代表 | 本專案的狀態 |
|---|---|---|---|
| **A. 無目標輸出距離** | max d(y_def, y_orig) | PhotoGuard diffusion attack 的無目標變體、TDAE | **只有這一類跑過**。59 個有記錄的 `env.json` 100% 是它（E29 §5.4）。它最大化的正是已被判定不對應防禦成功的量 |
| **B. 有目標** | min d(y_def, y_target)，y_target 常取灰圖 | PhotoGuard encoder／diffusion attack | 已實作、有測試，**從未在真實 SD 上跑過**。E31 待跑 |
| **C. 表示層／注意力** | 壓低內容 token 的注意力質量，或推離參考注意力圖 | [Attention Attack](https://arxiv.org/abs/2509.10359)（ACM MM 2025）、[DANP](https://arxiv.org/abs/2512.14333)、SIFM（中間層特徵） | `suppress` 已實作，**從未在真實 SD 上跑過**。E31 待跑 |
| **D. 早期步的噪聲預測範數** | max ‖ε_θ(z_t, t=T, c)‖₂，只在**第一個**去噪步施力 | [DiffusionGuard](https://arxiv.org/pdf/2410.05694) | **本專案從未考慮過**。見下方 |

**D 類值得單獨記一筆，而且它其實有兩個成員。**

| 方法 | 在第一個去噪步上攻擊什麼 |
|---|---|
| [DiffusionGuard](https://arxiv.org/pdf/2410.05694)（ICLR 2025） | 預測噪聲的 L2 範數 `max ‖ε_θ(z_T, T, c)‖₂` |
| [Structure Disruption / SDA](https://arxiv.org/abs/2505.19425)（2025-05） | **self-attention 的 query**，理由是輪廓在初期步成形 |

兩者獨立提出、著力點不同（噪聲預測 vs 自注意力），但共用同一個假設：
**初期去噪步決定整體佈局，破壞它就夠了，不必對整條鏈最佳化。**
DiffusionGuard 的成本是約 11 秒對 PhotoGuard 的 90 秒。

對本專案有兩層意涵：

1. **成本結構與 `crossattn` 相同。** 兩者都不需要跑完 `n_edit` 步的鏈，
   E0 成本模型中的 `0.304·n_edit` 那一項消失。若 E31 的 `suppress` 失敗，
   D 類是下一個成本可負擔的候選，而且是**兩個**候選不是一個。
2. **本專案的 `crossattn` 可能把力氣攤太平。** `optimize.py` 的取樣是
   `t_list = linspace(0, t_edit, attn_timesteps+1)[1:]`，即均分於整個
   `[0, t_edit]` 區間，`attn_timesteps` 預設 4。文獻的兩個成功案例都集中在
   **最高的那個 t**（第一個去噪步）。若 `suppress` 在 E31 失敗，先檢查的
   應該是「有沒有把權重放在該放的 t 上」，而不是換目標函數。這是一個
   已寫在程式裡、可用一行改動檢驗的假設。

**注意力那一類內部也分兩種。** cross-attention（文字—影像綁定，Attention
Attack、DANP、本專案的 `suppress`）與 self-attention（影像內部的結構，SDA）
是不同的東西。前者攻擊「哪個位置聽哪個 token」，後者攻擊「輪廓怎麼長出來」。
本專案只實作了前者。

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

這一點在文獻裡是被明講的。[SDA](https://arxiv.org/abs/2505.19425) 的動機
就是「global perturbation-based methods fail in mask-guided editing tasks
due to spatial constraints」——注意方向相反：**全域擾動在遮罩式編輯上失效**，
因為擾動大部分落在被重新生成的區域之外。於是該線的方法轉為攻擊遮罩內的
結構生成（self-attention query）。

本專案的處境是另一邊：全域 SDEdit 下沒有遮罩，擾動全部參與，但也全部被
strength 決定的噪聲稀釋。兩種設定的失效機制不同，**不可互相引用結論**。

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
  攻擊方法是拿現成 img2img 模型配一句「Denoise this image」。

  > **2026-08-04 核對後修正兩處。** (a) 原文的排序是「**FLUX and SD3
  > outperforms SDXL and SD1.5**」，GPT-4o 只用在 SD3 失敗的剩餘影像上、
  > 把 Matching Rate 由 77.8% 推到 78.6%（+0.8 個百分點）。本檔先前寫的
  > 「GPT-4o > FLUX > SD3 > SDXL > SD1.5」是過度解讀。
  > (b)「false sense of security」是原文引述**先前研究**的說法，不是他們
  > 自己的結論句。另注意其判準是人臉 Matching Rate，與抗編輯不同。
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
  Diffusion Models](https://arxiv.org/abs/2512.08329)（arXiv 預印本）。
  以可解釋 AI 的方法分析 Glaze、Nightshade、Mist 的擾動，結論是它們是
  「structured, low-entropy perturbations that remain tightly coupled to
  underlying image content across representational, spatial, and spectral
  domains」。這解釋了為何它們穩定**可被偵測與淨化**。

  > **2026-08-04 逐項核對後大幅修正。本檔先前對這一篇的引用有兩處錯誤。**
  >
  > 1. 先前寫「作者明講後續方向應『移出結構化、與內容耦合的擾動』，
  >    並點名非加性」。**原文沒有這句話。** Limitations and Future Work 的
  >    原文是「designing provably **unpredictable, spectrally diffuse, and
  >    entropy-adaptive** protection schemes represents a promising path
  >    toward stronger **evasion under explainable defenses**」——目標是
  >    規避偵測與淨化，不是抵抗編輯，而且 spectrally diffuse 與非加性
  >    參數化是兩件不同的事。
  > 2. 全文只有一處 `non-additive`，指的是**把兩種保護疊加時偵測行為的
  >    非加性**（「sequential application amplifies rather than suppresses
  >    the detectable perturbation structure」），與擾動的參數化無關。
  > 3. 「在頻域上沿影像自身的主頻軸重新分配能量」在原文中查無對應句。
  >
  > 此外這一篇評測的是 Glaze／Nightshade／Mist，即**風格模仿**那一類，
  > 依本檔 §7.1 其結論不可直接搬到單張影像的文字引導編輯上。
  >
  > **這一篇不再作為非加性方向的文獻支持。** 它仍可引用，但只能用於
  > 「現行保護擾動是結構化、低熵、與內容耦合，因而可被偵測」這一點。
- **STP-Diff**（Information Fusion 2025，
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253525011315)）：
  spatial transformation perturbations，以幾何扭曲像素座標取代加性雜訊，
  只作用在非顯著的周邊區域——理由是把 STP 施加在關鍵臉部特徵上會造成可見的
  變形與模糊。**參數化與本專案的 site S 是同一個構想。**

  > **2026-08-04 核對後補上界定。** 這一篇的**任務不同**，不可當成
  > 「已經有人用非加性做抗編輯」的證據：
  >
  > | | STP-Diff | 本專案 |
  > |---|---|---|
  > | 防的是什麼 | **人臉辨識系統**認出你 | **文字引導編輯**改掉你的照片 |
  > | 威脅模型 | **黑盒**、有目標 | **白盒**、stock SD |
  > | 判準 | Privacy Similarity Ratio 81.09%、FID 8.79 | Table 1 五指標／語意軸／劣化軸 |
  >
  > 它證明的是「空間變形作為一個非加性擾動類存在且可用」，不是
  > 「空間變形在抗編輯上有效」。

**把上面兩項界定合起來，非加性用於抗文字編輯這條線目前查不到直接的前作。**
這對本專案的新穎性是有利的，但也意味著**沒有外部結果可以借用**——
site S 至今沒有正面結果這件事，在文獻上沒有對照可比。
- **NAPPure**（[arXiv:2510.14025](https://arxiv.org/html/2510.14025)）：從**淨化側**
  處理非加性擾動，把受擾影像的生成過程建模成一個變換再聯合最佳化。
  它的存在本身說明非加性擾動已被視為需要專門處理的一類。
  **界定（2026-08-04）**：其標題明寫任務是 **image classification** 的
  強健性，不是抗編輯。與 STP-Diff 同樣屬於「參數化相關、任務不同」。

**對本專案的意涵。** 非加性不是本專案獨有的想法，但本專案在**量測方法**上
走得比較遠：等 LPIPS 多條件探針、`local_acutance_dev`、`local_chroma_bias`
這三項是為了回答「非加性到底用什麼換到它的效果」而做的，而上列論文都沒有
處理這個問題（STP-Diff 只報 LPIPS 與 SSIM）。**「匹配失真」三次被證明是假的**
這個結果本身就是對該線的貢獻。

---

## 7.1 一個必須劃清的範圍：三個不同的任務

這條線的論文常被混在一起引用，但它們防的是三件不同的事，**結論不互通**：

| 任務 | 攻擊方做什麼 | 代表 |
|---|---|---|
| **文字引導編輯**（本專案） | 拿**單一張**受保護影像，用 prompt 改它 | PhotoGuard、DiffusionGuard、SDA、本專案 |
| 風格模仿 | 拿**一批**受保護影像**微調**模型，再生成新作品 | Glaze、Nightshade、Mist |
| 個人化／DreamBooth | 拿數張人臉微調，再生成該人的新影像 | Anti-DreamBooth、CAT、`arXiv:2509.13922` |

差別在攻擊方是否**訓練**。後兩者的擾動要在梯度平均與資料增強之後仍然存活，
成功條件與失效機制都與單張編輯不同。

這一點在數字上很明顯：Glaze 自報防護率 >92%，看起來遠優於編輯那一類的
41.9% 人類偏好勝率——但那是兩個不可比的量。而且
[Adversarial Perturbations Cannot Reliably Protect Artists From Generative
AI](https://arxiv.org/pdf/2406.12027) 指出 Glaze v2.0 在強健模仿下沒有改善，
「noisy upscaling」幾乎達成完美模仿。

**本專案引用該類只用於方法學**（例如 `arXiv:2512.08329` 對 Glaze 擾動結構的
分析），不引用其防護率。

---

## 8. 尚未有人回答的

按對本專案的相關性排序：

1. **在高 strength 的全域 img2img 上，有沒有任何方法能讓編輯失敗？**
   查不到正面案例。文獻的成功案例集中在 inpainting 與低 strength。
2. **在匹配失真的條件下，非加性能否勝過加性？** 沒有論文用可通過判別的
   保真約束做過這個比較——STP-Diff 只報 LPIPS 與 SSIM，而 E20 已證明
   SSIM 會補貼模糊。
3. **判準的三類之間如何換算？** 沒有人報過同一批資料在三類判準下的結果。
   本專案的 `runs/p12_isr_rejudge/` 有這個材料。
4. **早期步的攻擊（D 類）在全域 SDEdit 上有效嗎？** DiffusionGuard 與 SDA
   都只測 inpainting，而 SDA 的動機恰恰是「全域擾動在遮罩式編輯上失效」——
   反過來會不會成立（早期步攻擊在全域編輯上失效）沒有人測過。
5. **無參考品質指標能不能代替 MLLM 判準？** SIFM 量過 MLLM 與人類的一致率
   （74%，對照人類彼此的 76%），但沒有人量過 NIQE 這類指標與人類的一致率。
   E31 的 `runs/p11_degrade_ladder/` 是為此準備的材料。

---

## 9. E31 之後的候選方向

為後續做的預備。每一條都附成本與**可能的否定結果**，避免事後把任何結果都
詮釋成正面。這一節不是計畫，是選單。

### 若 E31 找到正對照（某一格擋得下編輯）

那麼原本的研究問句終於問得出來：**在那個運作點上，非加性能否以較低的
人眼可辨失真達成同樣的阻擋。** 作法是把 site C 放回去，用 E31 定出的
逐預算門檻跑同一組格子。成本約等於 E31 的網格（1.5–2 小時）。

可能的否定結果：正對照只在 τ=0.28 這種明顯可見的預算上成立，而該預算下
「匹配人眼可辨失真」的比較本身失去意義——兩個條件都已經看得見，比的是哪一種
難看法比較有效。這個風險現在就該想清楚要怎麼寫。

### 若 E31 什麼都沒找到

四個選項，依成本排序。

| # | 方向 | 作法 | 成本 | 可能的否定結果 |
|---|---|---|---|---|
| A | **把 `crossattn` 的力氣集中在最高的 t** | `optimize.py:761` 的 `t_list` 改為只取 `t_edit`。文獻的兩個成功案例都集中在第一個去噪步（§3） | 一行改動 + 4 格 | 集中之後仍無效，那就排除「攤太平」這個解釋，D 類的假設在全域編輯上不成立 |
| B | **實作第四類：只攻第一個去噪步** | DiffusionGuard 的 `max ‖ε_θ(z_T, T, c)‖₂`，或 SDA 的 self-attention query 擾動。兩者都不跑完整條鏈，成本結構與 `crossattn` 相同 | 一個新的 objective + 校準 + 小網格，約 3–4 小時 | 兩個成員都只在 inpainting 上驗證過。若在全域 SDEdit 上失敗，那本身是對該類適用範圍的界定，值得寫 |
| C | **改成 inpainting 威脅模型** | 文獻的成功案例集中在這裡。需新寫攻擊路徑（遮罩式）並改變威脅模型定義 | 新的攻擊實作 + 完整重跑，數天 | SDA 的動機是「全域擾動在遮罩式編輯上失效」，即本專案既有的參數化在該設定下可能更差。這不是換個容易的題目，是換一個題目 |
| D | **寫成方法學／否定結果的論文** | 見下 | 0 GPU | 審稿人問「有沒有在文獻的預算與設定上量過」——E31 跑完就有答案 |

### D 的發表先例

「保護無效」與「判準錯了」這一類論文在這個領域是進得了場的：

| 論文 | 主張 | 場次 |
|---|---|---|
| [Rethinking the Invisible Protection against Unauthorized Image Usage in Stable Diffusion](https://www.usenix.org/system/files/usenixsecurity24-an.pdf) | 隱形擾動保護有結構性弱點 | **USENIX Security 2024** |
| [DiffHammer: Rethinking the Robustness of Diffusion-Based Purification](https://proceedings.neurips.cc/paper_files/paper/2024/file/a2fac827e992d55dcfdd4263e98528f4-Paper-Conference.pdf) | 單次評測低估風險，應改為 N 次評測 | **NeurIPS 2024** |
| [Is Perturbation-Based Image Protection Disruptive to Image Editing?](https://arxiv.org/abs/2506.04394) | 保護不阻止編輯，反而提高 prompt 對齊度 | **ICIP 2025** |
| [Rethinking and Defending Protective Perturbation in Personalized Diffusion Models](https://arxiv.org/html/2406.18944v4) | 三階段淨化即可破解 | ICLR 2025 |

**DiffHammer 的論點值得單獨記。** 它主張擴散過程的隨機性使**單次評測**系統性
低估風險，應改為 N 次評測。這與本專案 E25 定下的 n ≥ 2 規則是同一個方向，
但更進一步——本專案的 n 是**影像數**，每格只用一個評測噪聲種子（另加一個
訓練種子的對照）。DiffHammer 說的是**同一張影像要換多個種子**。

對 E31 的意涵：目前每格 2 張影像 × 1 個 held-out 種子。若某一格出現正例，
在下結論之前應該先換幾個種子確認它不是抽樣運氣。成本是線性的，且評測那一半
可以在本機跑（`runs/logs/e31_local_probe.log`）。

### D 的資產盤點

若要走這條，手上已有的東西是：

1. **判準的證偽。** 兩次獨立量到 `edit_shift`／`net_lpips` 與編輯是否失敗
   不對應（E25 的 726 格、E29 的 0.42–0.47），而 2026 年的 TPAMI 論文仍在
   用那一類（§2）。
2. **保真約束的方法學。** 等 LPIPS 多條件探針、`local_acutance_dev`、
   `local_chroma_bias`，以及「匹配失真三次被證明是假的」的完整紀錄。
   沒有其他論文處理過「非加性到底用什麼換到它的效果」這個問題（§7）。
3. **白雜訊參照條件。** 分辨「這個解在作弊」與「門檻是在小五倍的預算上定的」
   的手法（`docs/RESULTS_E25-E31.md` §2）。
4. **逐預算門檻，以及它的失效。** 門檻定出規則在 chroma 軸獨立重現了人眼判定
   （0.802 對 0.8），而 acut 軸的分離度隨預算塌掉——後者說明「匹配失真」
   這個概念本身有一個預算上限。這是本專案最原創的一項。
5. **有效約束診斷。** 四個誤判的有效約束的完整案例。
6. **外部佐證。** ICIP 2025（保護反而提高 prompt 對齊度）與 Off-The-Shelf
   2026（現成模型即可清除保護）。本專案的否定結果不是孤例。

缺的是第 7 項：**在文獻的預算與攻擊設定上量過**。那正是 E31 在做的事。

---

## 9.1 逐篇核對紀錄（2026-08-04）

取回原文 PDF、以文字比對本檔引用的每一個數字與說法。**下一次不必重驗這幾篇。**

| 論文 | 結果 | 改了什麼 |
|---|---|---|
| **Lo et al.** CVPR 2024 | **全部相符** | 無。Table 1 六欄逐格核對、κ = 0.06／N = 100／T = 10 相符；確認全文**沒有出現 `strength` 一詞**、guidance 只出現在參考文獻標題裡，即 3.13「論文未公布」成立 |
| **ICIP 2025** 2506.04394 | **引用有誤** | 把 Table 1（只測 PhotoGuard、Flickr8k）與 Table 2（風格化，Mist／Glaze）混為一談。61.82–67.79% 只是 Table 1 的 PAC-S++ 一列，同表 CLIP-S 是 54.23–61.54%。全距應報 54.12–77.54%。另補：判定是「Actual Change **≥ 0**」，且他們量到生成影像的 BRISQUE **比原圖好** |
| **SIFM** 2512.14320 | **引用不精確** | 「79–97%」是兩個不同設定（SD3 的 λ 消融 79%、HQ-Edit 的 97%，換未見 prompt 掉到 71%／65%）。補上 Table I 的完整一致率矩陣，以及一個對本專案有利的發現：**SIFM 自己的 λ 消融表上 ISR 動了 9 個百分點而 PSNR/SSIM/LPIPS 幾乎不動** |
| **DiffusionGuard** 2410.05694 | **方向正確，界定不足** | 補上三點：這是 2100 對的兩兩 A/B、「勝」的定義是「輸出被判為較差」、評分者被要求**同時**依影像品質與 prompt 服從度判斷（即它也是聯集式判準） |
| **SDA** 2505.19425 | **相符** | 無。「global perturbation-based methods fail in mask-guided editing tasks due to spatial constraints」為摘要原文 |
| **Structured Perturbations** 2512.08329 | **誤引，已移出非加性支持文獻** | 見 §7 的修正框。全文唯一的 `non-additive` 講的是堆疊保護時的偵測行為；future work 講的是規避偵測而非抗編輯；「沿主頻軸重新分配能量」查無對應句 |
| **STP-Diff** Information Fusion 2025 | **任務不同，已加界定** | 它防的是人臉辨識、威脅模型是黑盒、判準是 PSR/FID。參數化相同，結論不可搬用 |
| **Attention Attack** 2509.10359 | **相符，且提供兩個獨立佐證與一個本專案缺的作法** | 見下 |
| **PhotoGuard** 2302.06588 | **推翻了本專案 2026-08-04 稍早的一個說法** | 見下 |
| **DANP** 2512.14333 | **相符，且值得單獨記** | 摘要原文：DANP「functions over multiple timesteps to manipulate **both cross-attention maps and the noise prediction process**, using a dynamic threshold to generate masks」。**它是唯一同時屬於 C 類（注意力）與 D 類（噪聲預測）的方法**，而本專案的 §3 把兩類分開列。若 C 類單獨不夠，DANP 是「兩類合用」的現成參考 |
| **BlurGuard** 2511.00143 | **相符** | 摘要原文：對抗雜訊「should not only be **imperceptible**⋯but also **irreversible**」。本檔 §6 的定位正確 |
| **Glaze** 2302.04222 | **相符** | 「>92%」的出處是「highly successful at disrupting mimicry under normal conditions (>92%)」，另有「93% of artists rate the protection is successful」。屬**風格模仿**類，依 §7.1 其防護率不可搬到編輯那一類 |
| **Purify Once** 2603.13028 | **相符** | 原文：「protections optimized against a surrogate model may break down when attackers purify or edit images using a **mismatched pipeline** built on a different model」 |
| **Real-world** 2604.23688 | **相符** | 原文：影像在跨裝置顯示與傳播中會經歷「**scale transformations and color compression**」，直接改變像素值 |
| **TDAE** 2512.14341 | **全部相符** | 無。原文：「For these metrics [PSNR, SSIM, VIFP, FSIM], **lower values indicate a more effective immunization**⋯LPIPS, where **higher values**⋯**more effective immunization**」，即純第 1 類判準。venue 亦確認：arXiv 註記為「accepted by IEEE TPAMI」。**這是 2025 年 12 月的論文仍在用該判準的直接證據** |

### PhotoGuard（arXiv:2302.06588）的三項

**(a) 它有隨機雜訊對照，本檔稍早寫「五篇都沒有」是錯的。** Table 6：

| 方法 | FID ↓ | SSIM ↑ | PSNR ↑ | VIFp ↑ | FSIM ↑ |
|---|---|---|---|---|---|
| Immunization baseline (**Random noise**) | 82.57 | 0.75 | 19.21 | 0.43 | 0.83 |
| Immunization (Encoder attack) | 130.6 | 0.58 | 14.91 | 0.30 | 0.73 |
| Immunization (Diffusion attack) | 167.6 | 0.50 | 13.58 | 0.24 | 0.69 |

原文結論：隨機雜訊「is **not effective** at disrupting the SDM, and yields
edits **almost identical** to those of non immunized images」。

**(b) 但它匹配的是振幅，不是感知失真。** 原文界定：「adds uniform random
noise (**of the same intensity** as the perturbations used in our proposed
immunization method)」。而本專案 `p17` 已量到同一張影像上 PGD 解的 LPIPS
是 0.2935、同振幅隨機 sign 只有 0.1497、同 RMS 高斯只有 0.1084
（LEDGER 2.19）——**匹配振幅時隨機的可辨失真只有最佳化解的 1/2 到 1/3**。

> **兩個結果因此一致而非矛盾**：在**同振幅**上隨機很弱（PhotoGuard），
> 在**同 LPIPS**上隨機取得 60–74%（本專案，1.20）。
> **這是本專案「匹配軸應該是人眼可辨失真而不是 L∞」這個主張的直接證據，
> 而且證據來自 baseline 自己的實驗。**

**(c) 超參數**：ℓ∞、ε = **16/255**、step size 2/255、**N = 200 步**。
本檔 §4「文獻的標準失真預算是 ε = 16/255」成立。注意步數是 200 而非 100
（Lo et al. 為了公平比較把所有方法統一到 N = 100）。

**(d) 未能從論文核實的一項**：本專案規格 §4.1.1 以「PhotoGuard 的 img2img
評測用 strength 0.2／0.3」為由選定 strength = 0.3。**全文搜尋 `strength`
沒有任何出現**，且該論文的主要評測是**遮罩式 inpainting** 而非 img2img。
該值可能出自其程式碼而非論文。**此為待查項**，不影響 L1 與 Lo et al. 的
對照（後者本來就未公布 strength），但引用時不應寫成「論文說」。

### Attention Attack（ACM MM 2025）的三項

**(a) 獨立佐證了本專案 L3 的視覺判讀。** 原文：「differently from other
attacks, which mostly produce **noisy or blurry results**, our attention-based
attack makes final images **spatially and semantically inconsistent**」——
這正是 LEDGER 3.24 逐圖判讀出來的型態（`pg_encoder` 模糊褪色、
`pg_diffusion` 疊雜訊、`semantic` 保留構圖但語意被擋），
而本專案是**量化**的（1.15、1.17）。

**(b) 獨立佐證了判準不足。** 原文：「we argue that existing metrics are
insufficient. In fact, they are either focused on low-level image properties,
such as SSIM, PSNR and LPIPS, or are too high-level like CLIP score」，
並補「LPIPS is not capable of detecting small semantic changes」。
他們的替代品是 **Caption Similarity + semantic IoU**，與 SIFM 的 ISR 不同。

> **三個 2025 年的獨立群組各自判定標準判準不足，而三者提出的替代品互不相同**
> （SIFM 的 ISR、Attention Attack 的 CaptionSim + IoU、本專案的三類分解）。
> 這本身就是「領域尚未收斂」的直接證據，也使「**為什麼它們會不一致**」
> 成為一個有價值的問題——而那正是本專案 L3 量出來的東西。

**(c) 一個本專案沒做而應該做的過濾。** 原文：「We manually filtered the
dataset to ensure reliable results... **selecting images where edits are
successful** using different editing methods」。即**先排除掉未防禦編輯本來
就失敗的影像**。本專案沒有做這道過濾，而 `runs/figs/2026-08-04_l1_three_attacks.png`
逐圖看得到至少三張的未防禦編輯是失敗的（`woman_00` → a man 仍是女性、
`man_02` → a woman 仍是男性、`bird_00` → a butterfly 仍是鳥）。
**在編輯本來就不會成功的影像上量免疫效果沒有意義。**

**核對之後最重要的一件事**：把 2512.08329 移出、把 STP-Diff 加上界定之後，
**「非加性用於抗文字編輯」查不到任何直接的前作**。新穎性因此比本檔先前
描述的更高，但也代表**沒有外部結果顯示這個方向可行**——與 L3 量到的
「文獻最強基準在 5.4 倍失真上語意失敗也只有 7/72」（LEDGER 1.15）
及「隨機擾動就拿到 60% 的效果」（1.18）合起來看，
`docs/CONVERGENCE.md` 的兩層策略建議因此更站得住。

---

## 10. 引用清單

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
| SDA | Structure Disruption: Subverting Malicious Diffusion-Based Inpainting via Self-Attention Query Perturbation | [arXiv:2505.19425](https://arxiv.org/abs/2505.19425) |
| HYPE-EDIT-1 | Benchmark for Measuring Reliability in Frontier Image Editing Models | [arXiv:2602.00105](https://arxiv.org/pdf/2602.00105) |
| Glaze | Protecting Artists from Style Mimicry by Text-to-Image Models | [arXiv:2302.04222](https://arxiv.org/abs/2302.04222)（USENIX Security 2023） |
| Hönig et al. | Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI | [arXiv:2406.12027](https://arxiv.org/pdf/2406.12027) |
| BlurGuard | A Simple Approach for Robustifying Image Protection Against AI-Powered Editing | [arXiv:2511.00143](https://arxiv.org/html/2511.00143v1) |
| STP-Diff | Synergistic fusion of spatial transformation perturbations and diffusion models for robust face privacy protection | Information Fusion 2025 |
