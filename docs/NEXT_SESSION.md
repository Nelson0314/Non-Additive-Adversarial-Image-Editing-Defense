# 下一階段：攻擊端修好之前，沒有任何既有數字可以引用

寫於 2026-08-01（E25–E26 之後），取代同日稍早的版本（該版寫的是
「主軸的方向需要重新思考」，其 §1–§3 建立在 `net_lpips` 上，而 E26 已使
該量失效）。完整交代見 `docs/RESULTS_E25-E26.md`。

---

## 1. 一句話交代現況

**E2–E23 全部是在防禦一個不存在的攻擊。** `src/models/sd.py` 從來沒有
classifier-free guidance，等同 w = 1，而 SD v1.4 在 w = 1 下幾乎不服從
prompt。那些實驗量到的 `net_lpips` 是兩次隨機去噪之間的漂移。

發現的順序是：E20 修好保真那一軸 → E25 重判防禦那一軸得到「726 格語意
失敗 0 格」→ 使用者看比對頁後指出「連原始圖片被文字編輯都沒有成功」→
E26 追出根因並以真實 SD v1.4 量出 w 的影響。

---

## 2. 哪些結論還活著

**活著（純失真量測，完全不經過 SDEdit）：**

| 結論 | 出處 |
|---|---|
| 用單一純量定義「匹配失真」，在失真種類不同的兩族之間不成立 | E20 §0、§5.3 |
| 四臂等 LPIPS 探針：位移主導 vs 鈍化追蹤的判別法，以及人眼確認 | E20 §5.3、§7 |
| `local_acutance_dev` 同時具備「對位移不敏感」與「不可抵銷」 | E20 §6 |
| SSIM 在保真項中補貼模糊 | E20 §3.3(1) |
| site S 的鈍化來源是雙線性重取樣（85.0% vs 99.9%） | E20 §5.2 |
| A 族（site L/E/W）的 VAE 重建地板高於加性運作點 | E17–E19 |
| site C 的色度變換在 max_dev∈[0.10,0.15] 進得了 τ∈[0.02,0.10] | E26 §5.4 |
| CLIP 在本資料上分不出「編輯有沒有發生」，SigLIP 可以 | E25 §1.1 |

**死了（全部依賴 `net_lpips`，而該量在 w=1 下無意義）：**

- E15 的「site S 領先 1.15×」（E20 已撤回，E26 再補一刀）
- E21 的「site S 失去 67.3%」——**該數字本身仍是兩個 net 的比值，方向可能
  仍對，但絕對意義沒了**
- E23 的「Sbic / P = 0.85×」
- E25 §2 的淨化保留率（量的是漂移的保留率）
- E22 的位移飽和度診斷（診斷的對象仍在，結論的用途沒了）

---

## 3. 已經修好、等著上 GPU 的東西

全部已在 CPU 跑通並有測試（208 passed / 1 skipped，基準為 186/1）：

| 項目 | 狀態 |
|---|---|
| `sdedit(guidance_scale=, emb_uncond=)` | 已實作，w=1.0 逐位元回到舊行為 |
| `--guidance_scale` 貫通訓練與評測 | 已接通並以 tiny-SD 端到端驗過 |
| `attn_mode="suppress"`（修 φ=0 零梯度） | 已實作並成為預設 |
| `defense_mode="targeted"` | 已跑通（此前 4882 列全是 untargeted） |
| site C（色度矩陣場） | 已實作、13 項測試、容量檢查通過 |

---

## 4. 下一步該做什麼

**第一優先：以 w = 7.5 重跑主網格。** 在此之前任何倍率、任何 site 之間的
比較都不應引用。重跑時同時改四件事，否則會再跑出一批不能用的資料：

1. **w = 7.5**（E26 §3），且 **strength 與 w 一起掃描**——car_01 在 s=0.7
   的對齊度反而低於 s=0.5，最佳 strength 逐圖不同。
2. **判準改為語意軸**（E25 §1）。`net_lpips` 不可再當主要判準。SigLIP 通過
   對照可用；ISR 式的 MLLM 判準（arXiv:2512.14320）更完整但需外部模型。
3. **停止準則改為「約束啟動並穩定」**，不要固定步數（E21–E23 §5.4，這條
   在 E26 之後仍然成立）。
4. **不含 site S**（使用者 2026-08-01 決定）。它的 1.15× 與 0.85× 都已失效，
   沒有既有理由保留；作為對照的價值留到最後再評估。非加性一側改由 site C
   承擔，加性基準 site P 不變。

**第二優先（可與上面同批跑）：**

- site C 的四臂探針：現行約束集對它的特徵失真（色偏、假色）收不收費？
  不做這一步就是重蹈 site S 用 LPIPS 買模糊的覆轍。
- site C 的 `max_dev` 逐圖校準：色度能量跨影像差 2 倍。
- targeted 與 suppress 兩個目標的實測。
- 失真預算提高到文獻區間：現行 `defimg_lpips` 0.036–0.059，而 DCT-Shield
  (ICCV 2025) 報告自身 0.267、PhotoGuard 等 0.284–0.362。

---

## 5. 不要重走的死路

- **對抗性強健的感知度量**（E-LPIPS / R-LPIPS / LipSim）——解的是相反的失效。
- **NLPD、VIF、GMSD、HaarPSI 當保真約束**——四臂探針證明它們量的是位移。
- **low rank**——使用者 2026-07-30 明確排除。
- **site L / E / W（A 族）**——VAE 重建地板高於加性運作點。
- **在固定步數的網格上比較兩個 site**——φ 量綱與 lr 都不同。
- **`net_lpips` 當防禦成功的判準**——E25/E26，這是新增的一條。

---

## 6. 值得知道的文獻位置

- **arXiv:2512.14320**（Semantic Mismatch and Perceptual Degradation）——
  直接否定「與未防禦編輯的視覺距離」這個判準，提出 ISR。本專案 E25 是它的
  獨立佐證。
- **arXiv:2604.23688**（Do Protective Perturbations Really Protect...）——
  JPEG(q=75) + Lanczos 縮放擊垮八個方法。並提出「不可見性悖論」。
- **arXiv:2412.18791**——已記錄「用單一純量界定可見度預算會使跨方法比較
  失效」。E20 的四臂判別法與 `local_acutance_dev` 是這個問題的一個解。
- **NAPPure（ICCV 2025, arXiv:2510.14025）**——非加性擾動需要專門的淨化器。
- **STP-Diff（Information Fusion 2025-12）**——非加性空間變形保護的先行工作。
  他們**不敢單獨用 STP**（會扭曲關鍵區域），只在非顯著區用，是 E20/E21
  結論的獨立外部佐證。
- **DCT-Shield（ICCV 2025, arXiv:2504.17894）**——DCT 域的擾動，明講
  「pixel budgets are not a suitable metric for noise perceptibility」。
- **ReColorAdv / cAdv / RetouchUAA / Natural Color Fool**——色彩類非加性
  參數化，site C 的文獻依據。

---

## 7. 程式與資料狀態

- 分支 `claude/e20-fidelity-constraint`，**未併入 main**。
- 測試基準 **208 passed / 1 skipped**。
- 新增：`src/residual/site_color.py`、`src/models/sd.py` 的 `_eps_cfg`、
  `src/models/attention.py` 的 `attention_content_suppression`。
- runs 新增 `p5_semantic_axis`（含 `compare.html`）、`p6_purify_retention`、
  `p7_attack_sanity`（含 `compare.html`）、`p8_site_c_capacity`。
- **兩張人眼比對頁**：`runs/p5_semantic_axis/compare.html`（編輯有沒有被
  擋下來）與 `runs/p7_attack_sanity/compare.html`（w=1 vs w=7.5）。

## 8. 環境

- TWCC 容器 203.145.216.165:57683 仍存在但閒置，**刪不掉**（帳號到期、
  API 路由全被擋），需由使用者從 TWCC 網頁刪除。
- SD v1.4 權重**本機也有快取**（4.27 GB），CPU 上跑 512² 每步約數秒，
  E26 的驗證即在本機完成，不需 GPU。
- `/work/nelson0314` 是 NFS，跨容器保留。執行前 source
  `/work/nelson0314/WACV/env.sh`（repo 內副本 `scripts/twcc_env.sh`）。
- 容器預裝的 NGC torch 不支援 V100 的 sm_70，必須用 conda env 的 torch cu118。
- 遠端 `git pull` 常因 `runs/` 未追蹤檔 abort，先 `mv` 到
  `/work/nelson0314/pull_backup/` 再 pull。**`env.sh` 會被一併移走，記得還原。**
