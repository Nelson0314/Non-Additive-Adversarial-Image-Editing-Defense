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
| 判準 | Table 1：PSNR↓ SSIM↓ VIFp↓ FSIM↓ LPIPS↑，全部是第 1 族「與未防禦編輯的距離」 |
| baseline | PhotoGuard 的 encoder attack 與 diffusion attack |
| 資料 | 擴散生成 150 張、3 物件 × 2 prompt、**20 種子平均** |
| 自陳限制 | 模糊或 JPEG 壓縮可能消掉免疫效果 |

**這一篇的地位不同於本檔其他論文**：其約束、判準與 baseline 已定為本專案的
必要對齊項（使用者 2026-08-03）。逐項的對齊狀態見新規格 §3。

它同時決定了本檔以下各節的讀法：第 2 節說「第 1 族判準無效」，那是**本專案
在自己的資料上的判定**，不是可以直接套到這一篇上的結論——順序見新規格 §7。

---

## 0.1 一頁摘要

| 問句 | 目前的答案 | 對本專案的意涵 |
|---|---|---|
| 擾動式保護真的擋得住編輯嗎？ | **多半擋不住**，且受保護影像常**更**服從 prompt | E29 的否定結果是已知現象的白盒複現，不是實作瑕疵 |
| 領域用什麼判準？ | 尚未收斂。基準論文與 2026 年的 TPAMI 論文都用「與未防禦編輯的距離」 | 本專案有一份該判準與另兩族之相關性的量測（ρ = 0.140／−0.207／0.014），但**在重現落後基準時不提出**（新規格 §7） |
| 目標函數有幾族？ | 至少四族，本專案只跑過其中一族 | `targeted`、`suppress` 待跑；**「只攻第一個去噪步」**這一族（兩個成員）從未考慮過 |
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
| 與未防禦編輯的距離 | **基準論文 Table 1**、TDAE（TPAMI 2026），都用 PSNR／SSIM／FSIM／VIFp／LPIPS | 輸出移動了多少 | **本專案已定為必要判準**（使用者 2026-08-03）。本專案另有一份量測顯示它與語意軸幾乎不相關（ρ = 0.140）、與劣化軸負相關（ρ = −0.207），資料在 `runs/p16_criterion_correlation/`；依新規格 §7，該論證在第一層重現通過之前不提出 |
| 影像—文字對齊 | ICIP 2025 用 CLIP-S／PAC-S++ | 輸出服不服從 prompt | 可用，但 **CLIP 未通過本專案的 edit_effect 對照**（E25 §1.1），SigLIP 通過 |
| 聯集式 | SIFM 的 ISR（MLLM 判定）；Attention Attack 的 Caption Similarity + semantic IoU | 語意不符 **或** 感知劣化 | 本專案 E31 採用，但用 NR 品質指標取代 MLLM |

**MLLM 判準的可靠度已被量過。** SIFM 報告 ISR 的
**人類—MLLM 一致率為 74%，而人類彼此的一致率是 76%**——即 MLLM 判準與人類
判準的差距，小於人類之間本來就有的分歧。另一份編輯基準（HYPE-EDIT-1）報
VLM judge 與人類多數決的一致率約 80%，但傾向過嚴，在人類仍可接受的細微改動
上會判失敗。

對本專案的意涵：E31 用無參考品質指標（NIQE 等）取代 MLLM 是成本考量下的
折衷，而**折衷的代價現在有數字可引用**。若 E31 的感知劣化那一半成為結論的
關鍵，值得補一次 MLLM 判定作為交叉驗證。

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

**D 族值得單獨記一筆，而且它其實有兩個成員。**

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
   D 族是下一個成本可負擔的候選，而且是**兩個**候選不是一個。
2. **本專案的 `crossattn` 可能把力氣攤太平。** `optimize.py` 的取樣是
   `t_list = linspace(0, t_edit, attn_timesteps+1)[1:]`，即均分於整個
   `[0, t_edit]` 區間，`attn_timesteps` 預設 4。文獻的兩個成功案例都集中在
   **最高的那個 t**（第一個去噪步）。若 `suppress` 在 E31 失敗，先檢查的
   應該是「有沒有把權重放在該放的 t 上」，而不是換目標函數。這是一個
   已寫在程式裡、可用一行改動檢驗的假設。

**注意力那一族內部也分兩種。** cross-attention（文字—影像綁定，Attention
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

## 7.1 一個必須劃清的範圍：三個不同的任務

這條線的論文常被混在一起引用，但它們防的是三件不同的事，**結論不互通**：

| 任務 | 攻擊方做什麼 | 代表 |
|---|---|---|
| **文字引導編輯**（本專案） | 拿**單一張**受保護影像，用 prompt 改它 | PhotoGuard、DiffusionGuard、SDA、本專案 |
| 風格模仿 | 拿**一批**受保護影像**微調**模型，再生成新作品 | Glaze、Nightshade、Mist |
| 個人化／DreamBooth | 拿數張人臉微調，再生成該人的新影像 | Anti-DreamBooth、CAT、`arXiv:2509.13922` |

差別在攻擊方是否**訓練**。後兩者的擾動要在梯度平均與資料增強之後仍然存活，
成功條件與失效機制都與單張編輯不同。

這一點在數字上很明顯：Glaze 自報防護率 >92%，看起來遠優於編輯那一族的
41.9% 人類偏好勝率——但那是兩個不可比的量。而且
[Adversarial Perturbations Cannot Reliably Protect Artists From Generative
AI](https://arxiv.org/pdf/2406.12027) 指出 Glaze v2.0 在強健模仿下沒有改善，
「noisy upscaling」幾乎達成完美模仿。

**本專案引用該族只用於方法學**（例如 `arXiv:2512.08329` 對 Glaze 擾動結構的
分析），不引用其防護率。

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
4. **早期步的攻擊（D 族）在全域 SDEdit 上有效嗎？** DiffusionGuard 與 SDA
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
「匹配人眼可辨失真」的比較本身失去意義——兩臂都已經看得見，比的是哪一種
難看法比較有效。這個風險現在就該想清楚要怎麼寫。

### 若 E31 什麼都沒找到

四個選項，依成本排序。

| # | 方向 | 作法 | 成本 | 可能的否定結果 |
|---|---|---|---|---|
| A | **把 `crossattn` 的力氣集中在最高的 t** | `optimize.py:761` 的 `t_list` 改為只取 `t_edit`。文獻的兩個成功案例都集中在第一個去噪步（§3） | 一行改動 + 4 格 | 集中之後仍無效，那就排除「攤太平」這個解釋，D 族的假設在全域編輯上不成立 |
| B | **實作第四族：只攻第一個去噪步** | DiffusionGuard 的 `max ‖ε_θ(z_T, T, c)‖₂`，或 SDA 的 self-attention query 擾動。兩者都不跑完整條鏈，成本結構與 `crossattn` 相同 | 一個新的 objective + 校準 + 小網格，約 3–4 小時 | 兩個成員都只在 inpainting 上驗證過。若在全域 SDEdit 上失敗，那本身是對該族適用範圍的界定，值得寫 |
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
   用那一族（§2）。
2. **保真約束的方法學。** 等 LPIPS 多臂探針、`local_acutance_dev`、
   `local_chroma_bias`，以及「匹配失真三次被證明是假的」的完整紀錄。
   沒有其他論文處理過「非加性到底用什麼買到它的效果」這個問題（§7）。
3. **白雜訊參照臂。** 分辨「這個解在作弊」與「門檻是在小五倍的預算上定的」
   的手法（`docs/RESULTS_E31_local.md` §2）。
4. **逐預算門檻，以及它的失效。** 定錨規則在 chroma 軸獨立重現了人眼判定
   （0.802 對 0.8），而 acut 軸的分離度隨預算塌掉——後者說明「匹配失真」
   這個概念本身有一個預算上限。這是本專案最原創的一項。
5. **綁定者診斷。** 四個假的綁定者的完整案例。
6. **外部佐證。** ICIP 2025（保護反而提高 prompt 對齊度）與 Off-The-Shelf
   2026（現成模型即可清除保護）。本專案的否定結果不是孤例。

缺的是第 7 項：**在文獻的預算與攻擊設定上量過**。那正是 E31 在做的事。

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
