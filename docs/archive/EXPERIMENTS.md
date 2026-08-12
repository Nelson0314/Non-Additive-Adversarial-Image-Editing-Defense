# 實驗批次

每一筆自足。欄位固定：設定、規模、狀態、產物、該批確立了什麼。
`FND-`／`DEF-` 是指認，不是閱讀順序。

---

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
