# PAPER_VERIFICATION.md — 論文原文核驗紀錄

日期：2026-07-23（preflight 第二部分之二）。
目的：檢驗 SPEC.md 是否正確反映論文原文（SPEC 由指導者自論文重建，實作依 SPEC 與官方 code）。
方法：逐篇取回原文（arXiv HTML／ar5iv／PDF 文字抽取／CVF PDF／官方 repo 與文件），
對照 SPEC §7 公式—出處對照表與各章節記載。引文憑證見文末「取回來源」。

影響程度：**高**＝實作行為錯誤或結論不成立；**中**＝影響參數或效果強弱；**低**＝措辭或標註。

**總結：無「高」等級問題。發現 2 項「中」（其一已修正實作）、多項「低」。**

---

## DAYN（Lo et al., CVPR 2024, "Distraction is All You Need"）

**一致**：
- Table 1 全部 30 個數值（5 指標 × V1.4/V2.0 × Encoder/Diffusion/Ours）與 SPEC §2.7 **逐字一致**
- §4.1：κ=0.06、N=100（所有攻擊一致）、diffusion/semantic attack 之 T=10、SD V1.4（HuggingFace）
- 式 (2) attention map（softmax(QK^T/√d)）、式 (3) bicubic 上採樣至 initial feature map 尺寸後逐像素相加、
  式 (4) mask 由**原圖**聚合注意力以 τ 二值化、式 (5) L=‖Att(M⊙x_adv, c_a)‖₁ —— 均與 SPEC §3.4 一致
- 測試集：150 張、3 類物件、每類 2 prompt（兩種惡意情境）、20 random seeds 平均 —— SPEC §2.4 一致
- SDEdit strength：全文未出現 —— SPEC §2.8/§8 標記「須向作者索取」正確

**不一致**：無數值性不一致。

**SPEC 遺漏（對本專案有影響）**：
1. **Table 1 之實驗情境為 image editing（img2img，§4.3）**；inpainting（§4.2）僅質性比較（Figure 3）。
   → 校準比對**只能用 sdedit 列**，勿用 inpaint 列。影響：中。已修正 stage1 summary 之校準說明。
2. **測試集為「以 diffusion model 生成」之影像**（§4.3：»we first generate 150 images … using the
   diffusion model«），非自然照片。→ 資料索取未果時可依 supplementary 自行生成之備援方案成立。
   影響：中（資料策略）。已記入 TWCC_CHECKLIST。
3. **Algorithm 1 之更新為 `δ ← δ + s·sign(grad)`、`δ ← clip(δ, −κ, κ)`** —— 即 ℓ∞ 投影，
   且「for all attacks」同設定。→ 部分解決 SPEC §8 第 4 項（範數類型）：DAYN 自身演算法為 ℓ∞，
   支持 config 預設 `norm: "linf"`（T2 仍照計畫驗證）。影響：中（提高 linf 先驗）。
4. Algorithm 1 第 13 行 `x_adv ← x_adv − δ`（減號，與 PhotoGuard Alg.1 同形式）——
   與 SPEC §3.2 投影註之處理一致。影響：低。

**詮釋存疑**：SPEC §4.2 方案一之 reward 未套用式 (4) 之 mask M（全圖注意力抑制）；
此為本專案自行設計（SPEC 已標明「非 DAYN 重現」），非誤讀。

---

## PhotoGuard（Salman et al., ICML 2023, arXiv 2302.06588）

**一致**：
- 式 (4)：`argmin_{‖δ‖∞≤ε} ‖ℰ(x+δ)−z_targ‖²`，z_targ 為任意目標、灰階僅為例示 ——
  SPEC §3.2「論文通式＋官方 zeros 特例」之記載正確
- 式 (5)：`argmin ‖f(x+δ)−x_targ‖²`，x_targ 灰階或噪聲 —— 一致
- Table 9：ε=16/255、step 2/255、N=200、ℓ∞ —— SPEC §3.1 右欄一致
- Table 8：512/512/7.5/100/eta=1 —— SPEC §2.2 一致
- Appendix A.1 seed 協定：»first search for a good random seed that leads to a realistic
  modification« —— 判準僅「realistic」，**無量化標準** → SPEC 留白、實作以 CLIP 門檻
  操作化（config 可關）之處理正確
- §3 記憶體註：A100 40GB 反傳全程爆記憶體、僅反傳少數步 —— 逐字證實
- 使用 SD **v1.5** —— SPEC §2.1 註記正確（本專案依 DAYN 用 v1.4）

**不一致**：
- piq 引用註腳：SPEC 記「§4.2 註 8」，取回版本為**註 9**。影響：低（標註）。

---

## AdvDiff（Dai et al., ECCV 2024, arXiv 2307.12499）

**一致**：
- 式 (9)：`x*_{t-1} = x_{t-1} + σ_t²·s·∇ log p_f(y_a|x_{t-1})` —— 論文**有** σ_t² 係數；
  SPEC v4 記載「官方 code 將係數全部註解停用」與論文形式並陳，正確。
  註：DDIM eta=0 時 σ_t=0，係數若保留 guidance 恆為零——官方 code 停用係數與此一致，
  本專案採 code 形式（後置加法無係數）有其必然性。
- 式 (11)：`x_T = (μ+σε) + σ̄_T²·a·∇_{x_0} log p_f(y_a|x_0)` —— 梯度對 x_0（最終影像）計算
  ＝skip-gradient 之論文根據；SPEC §4.3.2 正確
- 附錄 E：noise sampling step »(0, 0.5] for MNIST, **(0, 0.2] for the ImageNet**« —— 逐字證實
  SPEC guidance 區間 (0, 0.2]
- 附錄 H：untargeted 形式 `−∇_{x_{t-1}} log p_f(y|x_{t-1})`（y 為真實標籤）—— 與 SPEC §4.2 一致；
  論文明言 untargeted »attack transferability significantly improved with a decrease in
  generation quality«（遷移性升、生成品質降）—— 與 SPEC 記載一致
- DiffPure 防禦下：AutoAttack ASR 22.2%（ResNet50）、AdvDiff-Untargeted **75.2%** ——
  SPEC §7「非加性抗淨化之證據」數值正確（指標為 ASR）
- ImageNet 超參數：N(K)=5、s=0.7、a=0.5、LDM+DDIM 200 步 —— 一致

**不一致**：
- 表格編號隨版本浮動（SPEC 記「Table 2」，v4 版該數據非編號 2）。影響：低（引用時註明版本）。

**SPEC 遺漏**：
- 附錄 F 失敗案例：**s=10 產生噪聲紋理、a=10 產生噪聲影像** —— 對本專案 s/a 重調係
  上界護欄有用。影響：低-中（調參指引）。已記於此。

---

## APA（arXiv 2506.01511）

**一致**：
- 兩階段解耦動機：論文原文即用 **reward hacking** 一詞（»Joint optimization with reward
  weighting often results in reward hacking (e.g., one shortcut to improving attack success is
  to reduce visual consistency)«）—— SPEC §4.4 記載正確
- 兩階段必要性有消融支持：Table 4（dual-path＋augmentation 使 black-box ASR 48.28→75.32）、
  Figure 5(b)（one-stage 隨視覺一致性權重升高 ASR 下降＝reward hacking、two-stage 較接近
  Pareto 前緣）
- 式 (4) DDIM inversion、式 (6) R_s、式 (7) 軌跡動量（∇_{z_T}、ℓ1 動量跨輪、µ·sgn、
  Π_{z⁰_T+ε_a} 投影）、式 (8)（ε−√(1−ᾱ)∇R_a）、式 (9)(10)(11)（x̂₀、混合 z_in、ℓ1 動量）、
  式 (12)（ϱ((D(z̄₀^t)+D(z̄₀))/2)）—— 逐條與 SPEC §4.4.1 一致
- 參數：T_a=10、N=10、ε_a=0.4、µ=0.04、SD v1.5 —— 一致
- **SG 之近似在論文中有形式根據**：»Skip gradient approximates g_tr as ρ·∇_{z̄_0}R_a(...)« ——
  即「最終 latent 梯度 × 縮放常數」；官方 code 之 14.58 為 ρ 取值（論文未給值）。
  SPEC §4.4.3「官方未說明來源」可補注：**形式出自論文、數值出自 code**。影響：低。
- GC 於論文以 »DDIM inversion steps: 10« 呈現 —— 語意不足，v4 依 code 判讀為部分
  inversion（前 10 格點、t≈0.2T）。論文與 code 不衝突，code 為準之處理正確。

**不一致**：
- 式 (8) 論文寫原始梯度 ∇；官方 code（v4 已核）用 sgn(動量)（式 11 之 m_st 代入）。
  屬論文式與演算法組合之詮釋空間；依修正原則以 code 為準，維持現行實作。影響：低。

**SPEC 遺漏**：
- LoRA rank／訓練步數論文正文未載（僅 code 有）—— SPEC §4.4.2 已註明出處為
  visual_alignment.py，正確。

---

## GrIDPure（Zhao et al., CVPR 2024, arXiv 2312.00084）

**一致**：
- 四步驟與混合式 `x_{i+1} = (1−γ)·x̃_i + γ·x_i`、γ=0.1 —— 一致
- Grid：256×256、stride 128、512×512 共 **10 個** grid（含四角落合併之第十個）—— 一致
- 成本：»approximately 2 minutes to purify a 512×512 image on a single V100« —— 逐字證實
- 評測任務為 **fine-tuning**（LoRA/DreamBooth 等），非 editing —— SPEC §6.3「僅借用淨化
  方法、流程不同」之注記正確

**不一致**：
1. **淨化設定**：論文 §B.5 預設 **pure_steps=10 × iterations=10**；README 範例 10×20；
   官方腳本預設 100×1。SPEC 記「README 建議 10×20」對 README 正確，但漏論文預設 10×10。
   影響：低-中（掃描軸已涵蓋；config 註解已補三來源）。
2. **「LDM 淨化器無效（對抗保護跨 LDM 遷移佳）」之負面結果不在本論文**——
   實際出處為 **Pixel is a Barrier §6.3／Appendix D（LDM-Pure）**：»LDMs can not be used to
   purify the protected images … the adversarial protection transfers well between different
   LDMs«。SPEC §5.3 出處誤置。影響：低-中（僅引用位置錯誤；實作本就採 pixel-space
   checkpoint，行為不受影響）。**建議 SPEC 下版將此句移至 §5.2 並改引 PiaB。**

---

## Pixel is a Barrier（Xue & Chen, arXiv 2404.13320）

**一致**：
- Appendix D：JPEG quality **65**、crop **20%** 後 resize 回原尺寸 —— SPEC §5.2 一致
- 擴散淨化 t*=**0.1T**（respace 100 步、跑 10 步）—— 一致
- LDM-Pure 無效與跨 LDM 遷移之論述（見上，GrIDPure 第 2 點）——此為該主張之正確出處

**SPEC 遺漏／注**：
- **Gaussian blur 之 sigma 值未見於 PiaB**；SPEC §5.2 併引 DiffusionGuard 附錄 F，
  sigma ∈ {0.5, 1.0, 1.5} 應出自後者。影響：低（建議 SPEC 分列兩出處）。

---

## AdverseCleaner（lllyasviel, 2023；原 repo 已下架，經 fork 核對）

**不一致（本次核驗之最大發現，已修正）**：
- 官方 clean.py（fork 逐字取回）：
  ```python
  for _ in range(64): y = cv2.bilateralFilter(y, 5, 8, 8)
  for _ in range(4):  y = guidedFilter(img, y, 4, 16)
  ```
  即 **64 次 bilateral ＋ 4 次 guided filter**（float32、0–255 域）。
  SPEC §5.1 偽碼記為 `n_bf_iter=3`＋單次 GF —— **迭代次數有誤**，淨化強度遠弱於官方。
  影響：**中**（stage2 之 AdverseCleaner 條件會系統性低估淨化能力）。
  **處置：已修正** `src/purify/adverse_cleaner.py` 與 `configs/purify.yaml`
  （bf_iterations 3→64、新增 gf_iterations=4）。d=5、σ_color=8、σ_space=8、r=4、eps=16 原即正確。
  **建議 SPEC 下版更新 §5.1 偽碼。**

---

## SDEdit（Meng et al., ICLR 2022, arXiv 2108.01073）

**一致**：
- diffusers 官方文件（v0.39）：»The StableDiffusionImg2ImgPipeline uses the
  diffusion-denoising mechanism proposed in SDEdit« —— SPEC §2.8 對應關係證實
- strength 語意（加噪程度，1=完全忽略原圖）與 SPEC 記載一致

---

## 核驗結論與待辦

| # | 發現 | 影響 | 處置 |
|---|---|---|---|
| 1 | AdverseCleaner 64×BF+4×GF（SPEC 記 3×BF+1×GF） | 中 | **已修正實作與 config** |
| 2 | DAYN Table 1 僅 img2img 情境（inpaint 為質性） | 中 | **已修正** stage1 校準說明 |
| 3 | DAYN 測試集為 SD 生成影像（備援方案成立） | 中 | 已記 TWCC_CHECKLIST |
| 4 | DAYN Alg.1 為 sign+clip（ℓ∞）→ 支持 norm=linf 預設 | 中 | T2 掃描不變，先驗提高 |
| 5 | 「LDM 淨化無效」出處＝PiaB 非 GrIDPure | 低-中 | 待 SPEC 下版修引用 |
| 6 | GrIDPure 論文預設 10×10（另有 README 10×20、腳本 100×1） | 低-中 | config 註解已補 |
| 7 | AdvDiff s=10/a=10 失敗案例（調參上界護欄） | 低-中 | 記於本文件 |
| 8 | APA SG 之 ρ·∇ 近似形式出自論文、14.58 出自 code | 低 | 待 SPEC 補注 |
| 9 | APA 式(8) ∇ vs code 之 sgn(動量) | 低 | code 為準，維持 |
| 10 | piq 註腳 8→9、AdvDiff 表號版本差、blur sigma 出處為 DiffusionGuard | 低 | 待 SPEC 措辭修正 |

**每個實作決定之依據**可循：SPEC §7 對照表 →（本文件逐篇「一致」清單）→ NOTES.md
之 L1.3／v4／preflight 條目（官方 code 核對）三層追溯。

## 取回來源

- DAYN：CVF open access PDF（本地抽取全文）
- PhotoGuard：ar5iv 2302.06588
- AdvDiff：arxiv.org/html/2307.12499（v1/v4）＋ PDF 全文抽取（附錄 E/F/H、表格）
- APA：arxiv.org/html/2506.01511
- GrIDPure：ar5iv 2312.00084 ＋ 官方 repo README
- Pixel is a Barrier：arxiv.org/html/2404.13320
- AdverseCleaner：raw.githubusercontent.com/shidoto/AdverseCleaner/main/clean.py（fork 逐字）
- SDEdit 對應：huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img
