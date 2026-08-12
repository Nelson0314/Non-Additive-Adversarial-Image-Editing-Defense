# 實驗批次

每一筆自足。欄位固定：設定、規模、狀態、產物、該批確立了什麼。
`FND-`／`DEF-` 是指認，不是閱讀順序。

---

## EXP-b1 · SDXL 首次上機

- **設定**：SDXL 1.0 base、1024²、bf16
- **規模**：段 0 校準 + 段 1 兩張圖
- **狀態**：**作廢**。校準表在錯誤的反演設定下產生（`t_max` 未指定、
  未開 BDIA），N3 的兩個學習率校準的是另一個問題
- **產物**：`runs/b1`、`runs/b1_bird_03`、`runs/b1_cat_02`、`runs/b1_dog_03`
- **確立**：DEF-001、DEF-002、DEF-003（段 1 的三個崩潰）；FND-001（生成路徑重建下限）

## EXP-b2 · 反演修正的驗證

- **設定**：同 EXP-b1 加 `--exact-inversion --t-max 200 --purify-mode all`
- **規模**：只跑 N3，三張圖
- **狀態**：完成，目的達成
- **產物**：`runs/b2_<影像>`（已於整理時刪除，內容為 EXP-b3 的子集）
- **確立**：FND-002（BDIA + t_max 200 使 `G(x;φ=0)` 落在 VAE 下限上方 0.008–0.010）

## EXP-b3 · SDXL 正式批

- **設定**：SDXL 1.0 base、1024²、bf16、strength 0.6
- **規模**：段 0–4 全部完成，三張圖
- **狀態**：完成。**結論比 EXP-v14 更沒有資訊量**——攻擊在 strength 0.6 上
  本身很弱（可辨區間 0.038，對 v14 的 0.091）
- **產物**：`runs/b3`、`runs/b3_merged`
- **確立**：FND-010（攻擊本身太弱時防禦效果無從量起）

## EXP-v14 · SD v1.4 第一輪

- **設定**：SD v1.4、512²、fp32、strength 0.6、img2img
- **規模**：3294 格（train 21 / rayscale 84 / control 285 / eval 2940），零失敗
- **狀態**：完成
- **產物**：`runs/v14`、`runs/v14_merged`
- **確立**：FND-003（20 個「影像 × 條件」組合沒有一個通過 `mean ≥ 3σ`）；
  DEF-005（訓練實際被 `tau_acut` 而非 `tau_lpips` 綁住）

## EXP-v14r · SD v1.4 重做

- **設定**：同 EXP-v14，改 strength **0.4**、`tau_acut` 0.16、`tau_chroma` 3.2
- **規模**：3294 格，零失敗
- **狀態**：完成。**重做的前提本身不成立**——放寬門檻解開了訓練約束，
  但那不是失效的原因
- **產物**：`runs/v14r`、`runs/v14r_merged`、`runs/v14r_margin.csv`、
  `runs/v14r_protocols/`
- **確立**：FND-004（位移場與隨機對照無法區分）；FND-005（約束確實解開，
  6/6 → 0/6）；FND-006（N3 只用掉三分之一預算就早停）；
  FND-007（三個判準給出三個相反的結論）；FND-008（非加性的 ΔNIQE 為負）

## EXP-ip3 · inpainting

- **設定**：SD inpainting 權重、512²、fp32、`--mask-mode attention_box`
- **狀態**：由另一個 session 負責，本 session 未介入
- **產物**：`runs/ip*`（本 repo 未收，在對方的工作目錄）
- **相關**：DEF-011（遮罩與 `c_a` 的位置關係做反了，影響本批）

## EXP-s3a · site apa + 注意力抑制，τ_train 0.50

- **設定**：SD v1.4、512²、fp32、strength 0.4、img2img、`--attn-mask-tau 0.5`、
  `--attn-timesteps 2`
- **條件**：MTH-N4、MTH-Ra、MTH-photoguard_c、MTH-mist、MTH-dia_r
- **影像**：bird_03、cat_02、dog_03，各 5 seed
- **規模**：2028 格（train 15 / rayscale 75 / control 285 / eval 1725），零失敗
- **機時**：段 0 = 1 h 08 m（單卡）；段 1–3 = 4 h 55 m（三卡平行）
- **狀態**：完成
- **產物**：`runs/s3a`、`runs/s3a_merged`、`runs/s3a_margin.csv`、
  `runs/s3a_protocols/`、`runs/s3a_identity.csv`
- **確立**：FND-011（注意力目標可最佳化，壓掉約 90%）；
  FND-012（二元判準在高預算下是反的）；FND-013（改用連續量後的 τ 掃描）；
  DEF-008、DEF-009、DEF-010

## EXP-s3t25 · 同上，τ_train 0.25

- **設定**：與 EXP-s3a 逐字相同，只改 `--tau-train 0.25`
- **影像**：horse_00、horse_03、woman_03（DEC-008）
- **規模**：2028 格
- **狀態**：**執行中**（2026-08-09 17:09 起）
- **產物**：`~/wacv_runs/s3t25*`（伺服器，尚未回收）

## EXP-s3t30 · 同上，τ_train 0.30

- **設定**：與 EXP-s3t25 逐字相同，只改 `--tau-train 0.30`
- **狀態**：**排隊中**，EXP-s3t25 成功收工後自動接
- **產物**：`~/wacv_runs/s3t30*`（伺服器，尚未回收）

## EXP-preflight · 選圖用的未防禦編輯

- **設定**：φ=0 那一側，無任何防禦，strength 0.4
- **規模**：horse ×4、man ×4、woman ×4，各 5 seed
- **狀態**：完成
- **產物**：`runs/s3_preflight/`、`runs/s3_preflight2/`
- **確立**：FND-014（man 那組的編輯失敗是構圖問題）；DEC-008（選圖）

## EXP-prior · 先驗實驗

- **範圍**：`runs/e15_*`、`runs/e21_*`、`runs/e23_*`、`runs/lrp*`、
  `runs/gate_*`、`runs/memtest`、`runs/diag*`、`runs/p6_purify_retention`
- **狀態**：2026-08-04 定案，截至該日的所有實驗一律視為先驗實驗，
  程式與評測流程整批重造
- **地位**：其結論已被後續批次取代或已寫入 `FINDINGS.md`；
  原始紀錄在 `archive/PRIOR_FINDINGS.md`

## EXP-s3t20 · site apa，τ_train 0.20，評測改用相對 DISTS Δ=0.04

- **狀態**：段 1–4 完成，判定層完成
- **設定**：SD v1.4／512²／fp32／strength 0.4；條件 `apa Ra photoguard_c mist dia_r`；
  影像 horse_00／horse_03／woman_03；段 1 綁 τ_LPIPS=0.20，段 2 改用
  `--budget-delta 0.04 --budget-metric dists`（DEC-015）
- **規模**：合併後 1800 格，`grid.csv` 1425 列。eval 每片 450 done／25 skipped
  （skipped 全為 `cnn_denoise_substitute`，其權重與套件未公開）
- **產物**：`runs/s3t20_merged/`（grid.csv、margin.csv、compare.html）、
  `runs/s3t20_protocols/protocols.md`
- **結論**：FND-017（無人防得住，photoguard_c 的位移全來自劣化）、
  FND-018（抗淨化 7/7 成立但優勢來自參數化）、FND-019（集中度）

## EXP-ip20 · inpainting，DEC-014 重訓，相對 DISTS Δ=0.04

- **狀態**：**段 1–4 完成，判定層完成**。段 3 曾在 eval 300–350/475 停過一次
  （使用者裁決轉向 DEC-016），同日續跑補完（`resumed` 326–347 格）
- **設定**：runwayml/stable-diffusion-inpainting／512²／fp32／無 strength；
  `--masks data/lo_masks --data data/lo_inpaint --prompt-index 1
  --attn-mask-tau 0.5 --attn-timesteps 2`；影像 horse_00／man_00／bird_03
- **段 1 端點**（DEC-014，式 (5) 對整個 M）：apa 的 `fid_lpips`
  0.3720／0.2784／0.3214，三張都跑滿 250 步；`attn_mask_kept`
  0.904／0.754／0.824
- **段 2**（Δ=0.04）：apa 的 k = 0.109／0.156／0.125
- **規模**：合併後 `grid.csv` 1425 列。eval 每片 done+resumed=450／skipped 25
  （skipped 全為 `cnn_denoise_substitute`，權重與套件未公開）
- **產物**：`runs/ip20_merged/`（grid.csv、margin.csv、compare.html、attention.html）、
  `runs/ip20_protocols/protocols.md`。分片目錄依 DEC-009 刪除
- **人眼**：三張 apa 防禦圖比 DEC-012 受限版乾淨很多；man_00 襯衫仍有青紫色紋
- **結論**：FND-020（抗淨化 7/7 重現，Ra 仍高於 apa）、
  FND-021（類別 margin 判準在本威脅模型下無效，`control` 分母為零）

### 續跑時踩到的坑

kill tmux 之後 `.writer.lock` 成為殘留檔，`ProgressWriter` 依設計拒絕啟動並
要求手動刪除（它刻意不自動判定行程是否存活，誤判會讓兩個寫入者同時動同一批
資料）。處置是先用 `ps -p` 逐一確認鎖檔記錄的 pid 已結束，再刪鎖續跑。
**不要略過那一步**——鎖的存在理由就是它。

## EXP-taupreview · Δ 掃描（不重訓，只換段 2 的目標）

- **狀態**：完成
- **內容**：用既有 φ 以 `scripts/tau_preview.py --metric dists --tol 0.001`
  求 Δ = 0.02／0.025／0.03 的縮放係數與防禦圖
- **產物**：`runs/tp_sweep2/`（img2img）、`runs/tpi_sweep/`（inpainting）、
  `runs/tp_dists/`、`runs/tpd_ip/`（Δ=0.05 的先期版本）
- **用途**：使用者挑 Δ 的比對頁。裁決結果是 **Δ=0.04 可接受**
  （「除了女人照片那張之外，其他的照片失真都在可接受範圍」），
  woman_03 的銳利度比在 Δ=0.04 為 0.664、Δ=0.02 仍只有 0.730，
  **降 Δ 救不了它**，屬臉部平滑化而非幅度問題

## EXP-s3t20_r · 換掉階段一（latent 對齊 + 解碼器逐圖微調）

- **狀態**：完成
- **變因**：唯一改動是階段一。原文的 UNet LoRA 對齊（FND-022 證明為空操作）
  換成解 `z*` 使 `decode(z*) ≈ x` 加上解碼器 GroupNorm affine 與 conv bias
  的逐圖微調。條件為 `apa`（注意力抑制，hinge 約束）與 `Ra`
- **規模**：3 影像 × 5 種子 × 22 個淨化設定
- **產物**：`runs/s3t20_r_merged/`
- **結果**：FND-023。φ=0 下限 LPIPS 0.128 → 0.080；345 個配對格全部保真變好，
  防禦勝率 55%。階段一成本 910 秒／格 → 0

## EXP-s3t20_rd · 損失變因：注意力抑制改取比例

- **狀態**：完成
- **變因**：`L_def` 由 `‖Att ⊙ M‖₁` 改為 `‖Att ⊙ M‖₁ / ‖Att‖₁`。取比例後
  無法靠整體攤平降低，只有把質量移出遮罩才會下降
- **產物**：`runs/s3t20_rd_merged/`、`runs/s3t20_rd_protocols/`
- **結果**：質量確實被移出（horse_00 −56%），編輯照樣成功；位移量 0.0880，
  低於原式的 0.1010。見 FND-024 第 4 條

## EXP-s3t20_pj · 約束變因：hinge 改為投影

- **狀態**：完成（第一版三格因停止準則失效跑滿 250 步，已重跑）
- **變因**：保真約束由罰項改為每步投影回 Δ 球面（DEC-019）
- **產物**：`runs/s3t20_pj_merged/`、`runs/s3t20_pj_protocols/`
- **結果**：`scale_k` 0.938–1.000（訓練即評測，射線縮放為空操作）；
  `|1−銳利度比|` 由 0.2235 降至 0.0623，woman_03 由 0.646 拉到 0.960——
  該張先前被判定為唯一不可接受且「降 Δ 救不了」。防禦落差 −4%

## EXP-s3t20_tj · 著力點變因：投影 + targeted_output

- **狀態**：完成
- **變因**：投影約束不變，`L_def` 改為 `‖SDEdit(x_def; c_∅) − y_target‖²`
  ——直接最小化評測所量的那個量
- **產物**：`runs/s3t20_tj_merged/`
- **結果**：位移量 0.0776，**四個 arm 中最差**，配對較優比例僅 25%。
  訓練期 `edit_shift` 0.142–0.167（10 步代理鏈）在 50 步的真實攻擊鏈上掉到
  0.078——代理鏈與真實鏈的落差大於損失形式的差別

## EXP-ca_probe · c_a 抑制的 timestep 掃描

- **狀態**：完成
- **內容**：以 c_a 為條件、用式 (4) 的同一組遮罩，掃 12 個 timestep 量遮罩內
  注意力 L1 相對未防禦的變化，分「訓練施力點附近」與「其餘 t」
- **產物**：`runs/ca_probe/`（非加性四條件）、`runs/ca_probe_base/`（加性三方法）
- **結果**：FND-024 第 2、3 條
- **一個作廢的版本**：第一版問「抑制有沒有活過攻擊鏈」，但攻擊鏈以攻擊方的
  prompt 為條件，而 `token_span("horse")` 回傳 (1,2)——`a zebra` 的第 1 格是
  `a`。量到的是冠詞的注意力，且數值看起來完全正常（−0.7% 到 +2.2%）。
  c_a 不在攻擊 prompt 裡，該問法無解

## EXP-lo_semantic · DAYN 的忠實重現（像素 PGD + L∞）

- **狀態**：完成
- **內容**：`scripts/run_lo_baseline.py --attacks semantic`，κ=0.06、100 步，
  與本專案同一組影像
- **產物**：`runs/lo_semantic_s3/`（同三張圖）、`runs/lo_semantic/`（資料集前三張）
- **結果**：注意力抑制 70–90%、`edit_lpips` 0.51–0.56，但 `pert_lpips`
  0.52–0.59——**失真是本專案工作點（fid_lpips 0.13）的四倍以上**
- **用途**：切開「損失無效」與「參數化無效」。結論是損失有效，DAYN 的效果
  建立在一個本專案明確拒絕的工作點上

## EXP-suppress_sweep · 把既有 φ 放大到高抑制區

- **狀態**：完成
- **內容**：`scripts/suppression_sweep.py`，對 EXP-s3t20_pj 的 φ 掃
  k ∈ {1,…,64}，逐點量抑制並渲染防禦圖與編輯圖。**不重新訓練**
- **產物**：`runs/suppress_render/`、`runs/suppression_ceiling.html`
- **結果**：FND-025

## EXP-apa1 · APA 原生階段一重現（LoRA vs 我方 z*+decoder）

- **狀態**：完成
- **設定**：SD v1.4／512²／fp32；影像取自 **APA 官方 repo** 的 `images_un/`
  三張（panda／butterfly／coot，見 `data/apa_native/provenance.json`），
  類別標籤用官方 `data.json` 的 `class`
- **臂**：`floor`（φ=0）／`lora_native`（官方 Eq.6 目標，200 步固定）／
  `recon`（DEC-016 的 z*+decoder）
- **產物**：`runs/apa_native_probe/`、對照組 `runs/apa_native_ddim_control/`
- **結論**：FND-027

## EXP-apa2 · APA 原生階段二完整重現，四軸消融，對照三個加性 baseline

- **狀態**：完成
- **設定**：同上模型與影像，只取 butterfly + coot；階段二為 APA-GC 的
  dual-path guidance + latent L∞ 球（`src/defense/apa_native_stage2.py`）；
  編輯評測 SDEdit strength **0.55**（DEC-022）、CFG 7.5、共用噪聲
- **四個軸**：階段一（原生 LoRA／z*+decoder）× 架構（DDIM／BDIA）×
  reward（注意力抑制／targeted／**分類器 CE**）× 保真控制（latent 球／
  DISTS 進 loss × sign 或 Adam／影像空間投影 apa_pj）
- **規模**：96 格。`apa_pj` 另走 `run_stage.py`（重用 `s3t20_pj` 的校準，
  context 逐項相符），以平台停止收斂（180／140 步）
- **產物**：`runs/apa_native_full_v3`（3 影像，strength 0.4，含 latent 球
  失真對照用的 panda）、`apa_native_full_v4`、`apa_pj_eval`、`apa_exp2`、
  `apa_native_anchor`、`apa_edit_sweep`
- **結論**：FND-028、FND-029、FND-030
- **重現過程修掉的三處實作偏離**（見 DEC-021 與模組 docstring）：
  階段二 CFG 7.5→1.0、反演改為 APA-GC 的淺噪聲帶（50 格排程只執行前 11 格、
  T_a=10）、reward 量級正規化。`fid_lpips` 三批演進 0.51–0.82 → 0.42–0.48 →
  0.23–0.34（APA 原文 Table 3 為 0.23–0.25）
