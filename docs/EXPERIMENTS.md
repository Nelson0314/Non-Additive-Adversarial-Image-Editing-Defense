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
