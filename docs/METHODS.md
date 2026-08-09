# 條件與方法

每一筆自足：參數化在哪、損失是什麼、對照是誰、現況。

`grid.CONDITIONS` 是**已定義條件的登記表**，不是某一批要跑的清單；
哪些進入某一批由 `--conditions` 決定，記在 `scripts/shard.sh` 的 profile。

---

## MTH-N1 · 位移場 + 注意力導向

- **參數化**：site warp（`WarpResidual`），控制點網格上的位移場
- **損失**：`targeted_attn` —— `1 − (shared token 分到的注意力質量)`，
  全部 query 位置取平均
- **目標從哪來**：空 prompt 的 CLIP 編碼中恆為 shared 的位置（BOS 或末位 PAD），
  **prompt-free**
- **隨機對照**：MTH-R
- **現況**：DEC-005 移出格點，原始碼保留

## MTH-N2 · 位移場 + 目標影像

- **參數化**：site warp
- **損失**：`targeted_output`（`target_metric="lpips"`）—— `LPIPS(y_def, y_target)`
- **隨機對照**：MTH-R
- **現況**：DEC-005 移出格點，原始碼保留

## MTH-N3 · site apa + 目標影像

- **參數化**：site apa（LoRA 階段一 + latent 階段二的複合模塊）
- **損失**：`targeted_output`（`target_metric="mse"`）
- **隨機對照**：MTH-Ra
- **現況**：未在第三階段的格點內。其早停型態見 FND-006

## MTH-N4 · site apa + 注意力抑制（Lo et al. 式 5）

- **參數化**：site apa，與 MTH-N3 相同
- **損失**：`suppress_attn_ca` —— `‖Att(x_adv, c_a) ⊙ M‖₁`，
  **只在式 (4) 的遮罩 M 內**
- **目標從哪來**：**防禦方指名要保護的詞 c_a**（`ImageEntry.content`，
  來自 `data/lo_aligned/prompts.yaml` 的 `content` 欄）
- **威脅模型的改變**：前三個條件都是 prompt-free，本條件**不是**——
  防禦方必須說出要保護什麼。prompt 是攻擊方寫的、c_a 是防禦方選的，
  在威脅模型裡屬於不同的人。論文必須寫明這項不對稱：
  三個加性 baseline 都不需要這個輸入
- **與 MTH-N1 的差別**：N1 把注意力質量**導向** decoy 且作用於全域、不需要 c_a；
  N4 把指定詞的反應**壓低**且只在遮罩內、必須有 c_a
- **監看量**：`attn_suppressed`（起點的遮罩內 L1 − 當前值）。
  L_def 本身隨進展**下降**，不可直接監看
- **記憶體**：注意力前向恆不能 checkpoint，是本批最緊的一條路徑。見 DEC-011、DEF-008
- **隨機對照**：MTH-Ra
- **現況**：EXP-s3a 已跑完，EXP-s3t25／s3t30 執行中

## MTH-R · 位移場的同失真隨機對照

- **作法**：與被比較的條件走**同一個參數化**（位移場），參數取高斯隨機，
  不最佳化。種子由影像 id 的 CRC32 決定
- **已知偏離**：四個 τ 共用同一個隨機方向（射線縮放的結構所致）
- **現況**：隨 DEC-005 移出格點

## MTH-Ra · site apa 的同失真隨機對照

- **作法**：同一個 apa 參數化，**同樣跑階段一的保真對齊**，
  只把射線縮放要乘的方向參數抽成高斯
- **為什麼必須跑對齊**：不對齊的話它的 `x_base` 是未對齊的 VAE 重建，
  同一個 τ 之下得先付掉更大的重建誤差，留給隨機方向的預算比 N4 少
  ——那會讓對照系統性偏弱，而這條對照存在的唯一理由就是判斷
  「有沒有勝過隨便擾動一下」
- **測試**：`test_隨機對照不是選配且逐參數化各一個` 釘住
  「每個非加性條件都要有同參數化的隨機對照」

## MTH-inpaint · inpainting 威脅模型下的配置（尚未起跑）

不是一個新條件，而是上列條件換到 inpainting 權重時**共同改變的設定**。
條件本身不變：N4、Ra 加三個 baseline，位移場三個不放回來（DEC-005）。

- **攻擊 prompt**：取 `prompts[1]`（保留 c_a、改動別處），由 `--prompt-index 1`
  選定。img2img 各批一律 `prompts[0]`。全部呼叫點只准經
  `ImageEntry.attack_prompt`，`test_攻擊prompt只有一個入口` 以原始碼掃描釘住
- **遮罩**：**人工繪製**，逐影像一張 PNG，由 `--masks <目錄>` 指定
  （`scripts/draw_masks.py`）。畫的時候 **c_a 要留在遮罩外**——Lo et al.
  Figure 3 的要求，也是 DEF-011 的處置。文獻同此作法：PIE-Bench 附標註
  遮罩、PhotoGuard 與 AdvPaint 用人工遮罩。遮罩內容進 `config_hash`，
  改一張遮罩即改變全部格的雜湊
- **不可帶 `--strength`**：inpainting 沒有這個參數，`SDWrapper.edit` 會拒絕
- **`--attn-mask-tau` 必須明給**：`suppress_attn_ca` 不接受預設值
- **`--attn-timesteps 2`**：DEC-011，記憶體實測 15126 MiB 對 23924 MiB
- **選圖**：涵蓋率窗口 [0.15, 0.45] 由繪製工具即時顯示，落在窗外只警告
  不阻擋（全域常數正是本專案重複踩過的缺陷型態）。改人工之後約束變成
  「主體之外還有沒有地方畫得下這麼大一塊」——人像 8 張主體佔滿畫面，
  結論多半不變但要重新判讀。**選圖尚未定案**
- **τ_train**：不沿用 0.25／0.30。DEC-001 與 DEC-007 的工作點都是在 img2img
  上由人眼定的，inpainting 的失真型態不同，須重新掃描後由使用者選定
- **主判準**：連續量，不用二元 margin（DEC-006）

## MTH-photoguard_c · PhotoGuard 的 diffusion attack

- **類型**：加性（L∞ 球上的 PGD）
- **原生任務**：inpainting 與 img2img
- **成本**：段 1 單格約 107 分鐘，佔整段的六成以上
- **特殊處置**：`grad_mask` —— 擾動只落在不會被重繪的區域（原作逐字如此）

## MTH-mist · Mist 的 textural loss

- **類型**：加性
- **外部素材**：`data/targets/MIST.png`，缺少時該條件的每一格明確失敗
- **成本**：段 1 單格約 1.3 分鐘

## MTH-dia_r · DIA 的 R 變體

- **類型**：加性
- **成本**：段 1 單格約 2.4 分鐘
- **附註**：`dia_pt` 變體已排除——其 `l1_ball` 起點在某些輸入下遠超 eps 球
  （實測 eps=0.05 下 ‖d‖∞ 達 1.499，30 倍）

## 已排除的方法

| 方法 | 理由 |
|---|---|
| `dia_pt` | 起點超出自己的 eps 球 30 倍 |
| `diffvax` | 其免疫器實際餵入的是編輯區已歸零的 masked image，且硬編碼 9 通道 inpainting |
| `advpaint`、`promptflare`、`impress` | 機時裁決（2026-08-06），程式保留但不執行 |
