# NOTES.md — 執行紀錄

記錄硬體資訊、套件版本、L1.3 官方 repo 關鍵發現、每個決定的理由。

## 環境

- 本地開發機：Windows 11 Home（win32），無 GPU，以 CPU 開發。
- 目標執行環境：TWCC 開發型容器，Tesla V100-SXM2-32GB（CC 7.0）。
- conda env：`wacv`（Python 3.11）。
  - 理由：TWCC 容器預裝 torch 2.4.1（支援 Python ≤3.12），本機 miniconda base 為 3.13，
    故另建 3.11 env 以保持與 TWCC 可對齊。
- 裝置資訊（由 `src/utils/device.py get_device_info()` 產生）：（L1.2 填入）

## 套件版本

2026-07-23 安裝（conda env `wacv`，pip，最新穩定版不釘版本；TWCC 端 torch 版本本就不同，
程式碼不綁定版本）：

| 套件 | 版本 |
|---|---|
| torch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| diffusers | 0.39.0 |
| transformers | 5.14.1 |
| accelerate | 1.14.0 |
| piq | 0.8.0 |
| opencv-contrib-python | 5.0.0 |
| pytest | 9.1.1 |

驗證（2026-07-23 實測）：`piq.vif_p`、`piq.fsim` 可呼叫且回傳合理值；
`cv2.ximgproc.guidedFilter` 存在且可執行。

## L1.3 官方 repo 關鍵發現

### photoguard（校準錨點）— 2026-07-23 確認

**結論：官方實作為標準 PGD「以原圖為中心之 ℓ∞ ball 投影」，非 Algorithm 1 字面的累積扣減。**
解決 SPEC §3.2 註與 §8 待確認第 3 項；SPEC 採 from_original 之判斷正確。

encoder attack（`notebooks/demo_simple_attack_img2img.ipynb`，函式 `pgd`）關鍵片段：

```python
X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()  # 隨機初始化於 ε-ball 內
for i in pbar:
    actual_step_size = step_size - (step_size - step_size / 100) / iters * i  # 線性衰減
    loss = (model(X_adv).latent_dist.mean).norm()          # 目標為「零 latent」
    X_adv = X_adv - grad.detach().sign() * actual_step_size
    X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)   # ★ 投影回原圖 X±ε
    X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
```

與 SPEC §3.2 虛擬碼之差異（實作 photoguard.py 時須決定）：

1. **loss 目標**：官方 demo 最小化 `‖ℰ(x')‖`（隱含目標 = 零 latent），非 SPEC 的
   `‖ℰ(x') − ℰ(x_gray)‖²`。註：灰階 0.5 影像在 [-1,1] 空間為零「像素」，但其 latent 不為零，兩者不同。
2. **值域尺度**：notebook 在 [-1,1] 像素空間操作，demo 之 eps=0.06 換算 [0,1] 尺度僅 0.03。
   DAYN κ=0.06 所指尺度未載明——與 DAYN 對齊時的潛在誤差來源，列入待確認。
3. 隨機初始化 + step size 線性衰減（衰減至 1/100）；demo 用 iters=1000（論文 Table 9 為 200）。

diffusion attack（`notebooks/demo_complex_attack_inpainting.ipynb`）：

- `attack_forward` 為完整可微分 sampling loop，demo 用 **num_inference_steps=4**
  （印證「僅反傳少數步」；SPEC 取 T=10 對齊 DAYN）
- loss = `‖image − target‖₂`，demo 之 target = **零張量**（亦可用目標影像）
- 梯度以 **grad_reps=10 次平均**（EOT，對隨機起始 latent 取期望）——SPEC 虛擬碼未含此點
- demo 預設用 **L2 版**（eps=16、step=1、iters=200、`torch.renorm` 投影回原圖 L2 ball）；
  ℓ∞ 版（`super_linf`）以註解提供（eps=0.1、step=0.006），投影同樣以原圖為中心
- 此 notebook 步長為常數（衰減式被註解掉）

### GrIDPure — 2026-07-23 確認

**grid 切分與 SPEC §5.3 一致**（`gridpure.py get_crop_box` / `get_corner_box`）：
512×512 → 左上角座標 {0,128,256}² 共九個 256×256（stride 128），四個 128×128 角落
拼成第十個 256×256；重疊區以計數平均合併；混合式 `(1−γ)·純化 + γ·前一輪`，γ=0.1。

參數（尺度為 1000 步 DDPM 之原始 timestep）：

- **README 建議（= SPEC 所述常用值）**：`pure_steps=10, pure_iter_num=20, gamma=0.1`
- **腳本預設值**：`pure_steps=100, pure_iter_num=1`（即 DiffPure 標準 0.1T 單次設定）
- SPEC §5.3 的「小步 SDEdit 噪聲 0.1T」對應腳本預設（t=100），「t=10、20 次迭代」對應
  README 建議；兩者為不同設定，configs/purify.yaml 的 `sdedit_noise: 0.1` 與
  `timesteps: [5,10,15]` 語意須統一為「原始 timestep 數」（實作 gridpure.py 時處理）
- 淨化器為 **pixel-space 無條件 guided diffusion**（`256x256_diffusion_uncond.pt`，
  ImageNet 256×256，linear schedule 1000 步）；TWCC 上須另下載該 checkpoint（約 2 GB）
- 流程：前向加噪至 t（closed-form）→ 由 t 逐步 `p_sample` 去噪至 0，`sample_step=1`
- 官方 `__main__` 有小 bug（`args.pure_model` 應為 `args.pure_model_dir`），不影響本專案
  （我們自行實作，不呼叫其腳本）

### APA（deep-kaixun/APA）— 2026-07-23 確認

檔案：`attack_alignment.py`（Stage 2 入口）、`visual_alignment.py`（Stage 1 LoRA 訓練）、
`pipe_ours.py`（核心 pipeline）、`utils.py`（增強變換）。

**與 SPEC §4.4 一致**：µ(alpha)=0.04、ε_a(eps)=0.4、N(niters)=10、T_a=10
（`index_cond=40` 於 50 步排程 = 最後 10 步）、SD v1.5、prompt = class 名稱、
式 (9)(10) 中間步淨化逐字一致（`fac=√(1−ᾱ)`，z_in = fac·z_ori + (1−fac)·ẑ0，
z_ori 為原圖 latent）、式 (8) `noise_pred − √β·sign(m)`（m 為 ℓ1 正規化梯度動量、
每輪軌跡重置）、式 (7) 軌跡動量跨輪累積 + µ·sign + ℓ∞ clamp、
guidance_scale=1（inversion 與攻擊均無 CFG）。

**解決 SPEC §8 第 4 項——LoRA 設定（visual_alignment.py）**：
- **rank = 8**、lora_alpha = 8、target_modules = `["to_k","to_q","to_v","to_out.0"]`、
  init="gaussian"、以 **peft**（`unet.add_adapter(LoraConfig(...))`）實作
- AdamW lr=1e-4、weight_decay=1e-2、grad clip 1.0、constant scheduler、**200 步**
  （SPEC config 誤植 100）、noise_offset=0.1、單圖、caption=class、latent 用 `.sample()`
- Stage 1 = 標準 diffusion loss 最小化（與本專案實作方向一致）

**與 SPEC 不一致（以官方為準待修）**：
1. **APA-SG 之「skip gradient」語意**：軌跡梯度**不反傳穿過採樣鏈**——
   在最終 latent（z̄_0）上算 grad、乘常數 **14.58**（來源不明，經驗值）、
   直接套用到 z_T。SPEC 式 (7) 寫 ∇_{z_T} 未說明此近似。
2. **APA-GC 之 T=10 語意**：inversion 只做「部分」——50 步格點只走前 10 步
   （反推至 t≈0.2T），採樣只跑最後 11 步、`torch.utils.checkpoint` 反傳至 z_T；
   非「全噪聲範圍 10 步粗格點」（本專案目前實作為後者，須修正）。
   GC 的 reward 另有 **−10·MSE(z_ori, z_final)** latent 貼近正規化（SG 無）。
3. **式 (12) 之 z̄_0 = 最終生成 latent**（增強影像 = (D(x̂0_t) + 最終生成影像)/2），
   非原圖 latent（本專案目前與原圖 z0 平均，須修正）；la_list 之 x̂0 為 detached，
   梯度僅經最終影像那一半流動。
4. **ϱ（式 12 前處理）**= `ori_trans`：random resize（90–100%）+ random 零 padding
   回原尺寸 + interpolate；brightness 在 code 中被註解停用。reward 為對整批
   增強影像之 CE（等價於 1/T·Σ）。
5. 排程用模型預設 scheduler（PNDM/PLMS）50 步之 `scheduler.step`，非純手動 DDIM。

### AdvDiff（EricDai0/advdiff）— 2026-07-23 確認

檔案：`advdiff.py`（driver）、`ldm/models/diffusion/ddim_adv.py`（核心，位置與先前查閱一致）。
情境：class-conditional LDM（cin256-v2）、DDIM 200 步、eta=0、CFG scale 3.0、
由隨機噪聲生成（非保護任務）。

**與 SPEC §4.3 一致**：guidance 區間 `0 < index ≤ 0.2·total_steps`（DDIM 步 index 計，
最後 20% 步）、a=0.5、K(=N)=5、200 步、x_T 無投影（SPEC 之投影確為本專案自加）。

**與 SPEC 不一致（以官方為準待修）**：
1. **每步注入形式**：LDM+DDIM 下官方仍用「**後置加法**」——`p_sample_ddim` 完成後
   `img = img + s·gradient`（直接加在 x_{t-1} 上），**非** Alg.2 之改 epsilon；
   且所有係數（σ²_t / √(1−ᾱ)·×10）皆被註解掉，實際**無係數**。
   SPEC 採 Alg.2 改-epsilon 形式與官方不符。
2. **起始噪聲注入為 skip-gradient**：gradient 於**最終 latent** 上計算
   （= 論文式 11 之 ∇_{x_0}），直接 `pri_img += a·gradient` 套用到 x_T，
   **不反傳穿過採樣鏈**。SPEC 偽碼 `grad(R_final, z_T)`（全鏈反傳）與官方不符；
   官方作法無需保留採樣計算圖，記憶體大幅較低。
3. s：driver 預設 1.0、函式簽名預設 0.75（論文 §4 寫 0.7）——不一致，屬官方自身版本差。

**code 有而 SPEC 未提**：targeted 用動態「第二名標籤」（`get_target_label`：
分類正確時取 top-2，否則取 top-1）；攻擊成功即 early exit；只回傳成功樣本；
gradient clamp（±0.3）被註解停用。以上屬分類器情境，本專案 untargeted reward
不直接適用，但記錄供追溯。

### SDEdit — 2026-07-23

原始 repo（ermongroup/SDEdit，DDPM-based）僅作參考；本專案實際採 diffusers 之
img2img pipeline（SPEC §2.8 已確認其對應關係），不使用此 repo 程式碼。

## 決策紀錄

- 2026-07-23（v2.1 補丁）：PhotoGuard 為校準錨點，凡官方實作與 SPEC 偽碼衝突處一律以官方為準。
  據此：encoder attack 預設 `target_latent: zeros`（官方）；新增 `epsilon_scale`（pm1/01）、
  `norm`（linf/l2）、`target_latent`（zeros/gray）三個 config 參數，均列 SPEC §8 待確認
  （第 6–8 項），於 T2 校準階段以 DAYN Table 1 實驗定奪，不事先假設。
  官方三細節（隨機初始化、step 線性衰減、diffusion attack EOT grad_reps=10）一併採用。
  官方實作與 SPEC 偽碼之完整差異清單見上方「L1.3 官方 repo 關鍵發現」。
- 2026-07-23：GrIDPure 兩組設定（iterative 多次淺淨化 / single_deep 單次深淨化）皆列入
  configs/purify.yaml，階段二兩者都測，兼作「迭代設計是否優於單次深淨化」之檢驗。
- 2026-07-23（指令 B）：
  - `img2img_differentiable` 以手動 DDIM（eta=0）實作而非 diffusers scheduler：
    確保計算圖完整可反傳、步數可精確控制（T=10 對齊 DAYN）；此實作亦為
    §4.1 DDIM inversion 之基礎。prompt 為空字串時略過 CFG（uncond 與 cond
    相同，數學上等價、省一半記憶體；官方 attack_forward 對空 prompt 仍做 CFG，屬冗餘）。
  - `edit_image()` 較 STRUCTURE §2.2 介面多一個 `sd` 參數（SDWrapper）：
    原介面未含模型握把，為必要補充。
  - inpaint 用與編輯相同之基礎模型（4-channel UNet，diffusers 逐步 latent 混合），
    不用 runwayml inpainting 專用權重（SPEC §2.1 統一 v1.4；tiny 模型驗證可跑通）。
  - cross-attention 擷取以自訂 AttnProcessor 實作（context manager，僅掛 attn2），
    跨層聚合（DAYN 式 3）留給保護方法（指令 D）。
  - seed 協定：`seed_clip_threshold: null` 時不篩選、回傳連續 seed（判準待與
    指導者確認）；同 seed 編輯之位元級可重現性已由測試驗證。
  - placeholder 資料集為決定性合成影像（seeded 低頻雜訊上採樣），3 類 × 2 prompt，
    config `is_placeholder: true` 標記，取得 DAYN 資料集後切換 `_folder_dataset`。
  - FID 為資料集層級指標，不入 `compute_all`（per-pair 無意義），另供 `compute_fid()`。
- 2026-07-23（指令 D，非加性三方法）：
  - reward（SPEC §4.2 方案一）抽出為共用模組 `src/protect/rewards.py`
    （STRUCTURE §1 未列此檔，為避免三方法重複而新增）。
    `R = −‖A_concept‖₁`；聚合依 DAYN 式 (3)（head 平均、bicubic 上採樣、逐層相加）。
    方向正確性經測試驗證（沿 ∇R 上升一步 R 確實增加）。
  - AdvDiff：z_T 投影採「相對 L2 ball」（半徑 = eps_latent × ‖z_T‖），
    eps_latent=0.2 為暫定值，stage0 以 LPIPS 校準。
  - APA 實作簡化：式 (11)/(12) 之 f(·) 原為作用於解碼影像之分類器；本專案 reward
    以 UNet 注意力定義，直接於 latent（z^t_in、(z^t_0+z̄_0)/2）評估、不經 VAE
    解碼—編碼往返，語意等價且大幅降低內迴圈記憶體。「最後 T_a 步」實作為
    去噪末段（低噪聲端），與 AdvDiff 附錄 E 高 t 無效之發現一致。
  - LoRA 手刻（LoRALinear：to_q/to_k/to_v，up 零初始化），不引入 peft；
    注入當下不改變輸出、protect 結束後精確還原（皆經測試，torch.equal）。
    lora_rank 暫用 4（SPEC §8 第 4 項待確認，屬假設）。
  - Stage 1 之 LoRA 訓練即最小化單圖 diffusion loss（式 6 之 maximize R_s 等價形式）。
  - Hybrid 以子類別覆寫 APA 兩個注入 hook（step-level 原始梯度×s、
    trajectory-level +a·g），投影維持 APA 骨架之 ℓ∞ ball。
  - 記憶體風險（SPEC §4.2 已知風險）：trajectory 梯度須保留 T 步 UNet 計算圖，
    tiny 模型無虞；真實 SD 於 V100 若不足，後續以 gradient checkpointing 處理。
    （此條為 v3 時之評估，v4 已因 skip-gradient 發現而下修，見下）
- 2026-07-23（v4 補丁，依 APA/AdvDiff 官方 repo 全面修正非加性實作）：
  - **AdvDiff**（advdiff_based.py）：每步注入改為「後置加法」`z ← z + s·g`
    （標準 DDIM 一步後加在 x_{t-1} 上，無任何係數——官方將 σ²_t、√(1−ᾱ)×10
    全部註解停用）；z_T 更新改 **skip-gradient**（梯度於最終 latent 計算後直接
    套用，不反傳穿鏈）。s 採論文值 0.7（官方 driver 1.0／簽名 0.75 不一致）。
    z_T 相對 L2 ball 投影保留（本專案設計；STRUCTURE v4 範本漏列 eps_latent，
    SPEC §4.3.6 偽碼仍用，config 保留並註記）。
  - **APA**（apa_based.py）：LoRA 改用 **peft**（rank=8、alpha=8、
    to_k/q/v/to_out.0、gaussian、AdamW+wd 1e-2+clip 1.0、200 步、noise_offset
    0.1、latent .sample()，均官方值）；protect 後 `delete_adapters` 還原，
    torch.equal 驗證通過。SG＝完整 inversion（50 步格點）＋軌跡梯度 skip
    （最終 latent 上計算 ×14.58）；GC＝**部分 inversion**（格點前
    inversion_steps 步，t≈0.2T）＋checkpoint 反傳＋`−10·MSE(z_ori,z_final)`
    正規化。式 (12) 增強改為與**最終生成 latent** 平均（v3 誤用原圖）；
    ϱ = random resize(90–100%)+random 零 padding（brightness 不啟用）；
    ϱ 與混合於 latent 上進行（reward 為注意力定義之既有決定）。
    GC 之 sampling_steps=11 與 inversion_steps=10：官方迭代計數含邊界，
    本實作以 10 個格點區間上下對稱走訪，語意相同。
  - **Hybrid**：注入形式同步更新為 AdvDiff v4（步後加法 s·g、軌跡 a·g 無動量
    無 14.58），骨架（inversion、LoRA、ℓ∞ 投影）維持 APA。
  - **scheduler 疑點（SPEC §8 第 7 項）調查結果**：官方 APA 之 inversion 為
    手動 DDIM 數學式（格點取自 `scheduler.set_timesteps(50)`）、採樣為
    `scheduler.step` = **v1.5 預設 PNDMScheduler**（skip_prk_steps → PLMS），
    全 repo 無 scheduler 覆寫、無補償措施；repo 無成對 demo 輸出可觀察。
    本地 tiny 模型量化實驗**無效**：tiny 為隨機權重、eps 預測無意義，
    DDIM 匹配與 PNDM 兩者重建皆完全失敗（latent MSE 3–10 vs 訊號功率 0.003），
    無從比較。結論：重建品質差異須於 TWCC 真實 SD 上量測（建議 T2 附帶小實驗）。
    本專案現行實作為 inversion 與採樣**同用手動 DDIM（匹配）**，
    「錨定原圖」性質自洽，暫不改動，官方混用組合列為 TWCC 消融選項。
  - **記憶體量測**（tiny/CPU，process peak working set，baseline=僅載入模型
    361 MB，各值為單方法單張 protect 之峰值）：pg_enc 391、pg_diff 711、
    advdiff 429、apa_sg 640、apa_gc 662、hybrid 638 MB。
    pg_diff（EOT×10＋T=10 保留計算圖）最重；advdiff 因 skip-gradient 最輕，
    支持 v4 之風險下修。CUDA 上另有 peak_memory_mb()（torch.cuda 統計）。
  - 新增套件：peft（APA 官方依賴）、psutil（記憶體量測，開發輔助）。

- 2026-07-23（指令 E，淨化模組＋三支 stage 腳本）：
  - **淨化模組**：`purify()` 統一入口較 STRUCTURE §2.4 介面多 `config`（濾波參數）
    與 `purifier`（GrIDPure 淨化器握把）兩個關鍵字參數，為必要補充（同 edit_image
    之 sd）。AdverseCleaner 於 0–255 像素域運作（官方 guided filter eps=16 為該
    尺度）；GrIDPure grid 機制忠實移植官方 gridpure.py（切分／四角落拼合第十
    grid／重疊平均／γ 混合），淨化器以 callable 注入使 grid 機制可於 CPU 獨立
    驗證（恆等淨化器 → 輸出=輸入，經測試）；正式淨化器
    `load_guided_diffusion_purifier()` 沿用官方 guided_diffusion 套件與
    imagenet.yml 設定（use_fp16 依裝置：cuda 開、cpu 關）。
  - **stage0**：L_ref 以 PhotoGuard **encoder**（κ=0.06）為基準（假設：DAYN 之
    κ=0.06 加性基準；diffusion attack 同 κ，成本約 10 倍，不影響 LPIPS 基準）。
    掃描參數＝各方法相似性控制鈕（advdiff: eps_latent；apa/hybrid: eps_a），
    掃描值域為初始猜測、TWCC 實測後調整。回寫策略：similarity_budget 以逐行
    regex 改值（保留 yaml 註解）；各方法選定值寫入
    `configs/nonadditive_calibrated.yaml` overlay（stage1 自動合併），
    避免 yaml.dump 重寫整份 nonadditive.yaml 破壞註解。
  - **stage1**：保護一律以 protect_model 生成、一次完成後存 png，再逐 eval model
    編輯評測（保護與評測解耦，跨模型遷移即 SD V2.0 欄位）。protected 與
    edited_orig 均存檔供 stage2 重用（png 8-bit 無失真，量化屬保護輸出之
    現實情境）。每列含 prompt_idx（每類 2 prompt，STRUCTURE 欄位之必要補充）。
    FID 逐 (method, model, edit) 群組以 piq InceptionV3 特徵計算，
    `--no-fid` 可略（本地 smoke 樣本數過少、FID 無意義）。
  - **stage2**：generation/edit 設定一律讀 stage1 之 config_snapshot.yaml 確保
    條件一致；淨化與模型/編輯無關、逐 (方法,影像,op) 執行一次後存檔。
    drop 逐指標計算，方向解讀依 METRIC_HIGHER_IS_BETTER；曲線以 lpips 為
    主效果軸。`--dry-run` 先印成本估算（STRUCTURE §4.4 要求）；
    `--gridpure-fake`（Gaussian blur 假淨化器）僅供本地流程驗證。
  - `--smoke` 模式（三腳本共通）：tiny 模型縮減參數之流程驗證，
    解析度依模型原生尺寸推得，數值不具比較意義。
  - 本地驗證：測試 34/34（新增 test_purify.py 13 項）；stage0→1→2 煙霧串跑
    成功（stage1 40 列、stage2 400 列＝40 組合×10 淨化設定，drop 欄位與
    曲線圖正常產出）。config 回寫 regex 於副本驗證（註解保留、yaml 可解析）。
  - 新增套件：matplotlib（曲線繪製）。
- 2026-07-23（preflight 部署前檢查）：
  - **四項假設依裁定調整**：stage0 雙基準（encoder 採用＋diffusion 5 張併列、
    >10% 差距警告）；掃描範圍移入 config `stage0_scan`（apa_sg/apa_gc 獨立值域、
    hybrid 掃 eps_a——其約束為 APA 骨架 ℓ∞ ball）＋選中值落端點自動警告；
    遮罩規格入 config（edit.inpaint_mask）＋summary/manifest 標記；prompt_idx 維持。
  - **穩健性**：stage1/2 改逐筆寫入（IncrementalCsv，每列 flush）＋ `--resume`
    斷點續跑（保護影像/edited_orig/結果列皆可跳過，煙霧實測 8–9 秒空跑無重複）；
    drop 除零與負基準防護（drop_valid 欄）；config 回寫維持單行 regex＋overlay
    （評估後不引入 ruamel.yaml，理由記於 PREFLIGHT [2.5]）。
  - **稽核發現並修正之 bug**：`apply_smoke` 之 EOT 縮減用錯鍵名
    （diffusion_eot→grad_reps，STRUCTURE 範本與實作鍵名不一致所致）。
  - **論文原文核驗**（PAPER_VERIFICATION.md）：高等級 0、中等級 2 均處置——
    (1) AdverseCleaner 官方為 **64×BF＋4×GF**（SPEC 記 3×BF+1×GF），已修正
    adverse_cleaner.py 與 purify.yaml；(2) DAYN Table 1 僅為 img2img 情境，
    校準比對限 sdedit 列（stage1 summary 已改）。另：DAYN 測試集為 SD 生成影像
    （自生成備援可行）、DAYN Alg.1 為 sign+clip（ℓ∞ 先驗提高）、「LDM 淨化無效」
    出處實為 Pixel is a Barrier §6.3 非 GrIDPure、GrIDPure 論文預設 10×10。
  - **新增測試 5 項**（39/39）：guidance 區間位元級等價、eps_latent 投影生效、
    APA ℓ∞ clamp、指標方向極端案例（piq 五項＋FID）。
  - **成本估算**（preflight_report.py，V100 單位假設待實測校正）：完整 SPEC 方案
    約 14,000h（fp32）——瓶頸為 stage2 編輯 204 萬次之乘數效應，單卡不可行；
    建議路徑 C（~43h 首日）→ B（~213h，批次化後 ~70–90h）。
  - 產出：PREFLIGHT.md（preflight_report.py 自動生成，全項 PASS＋1 WARN）、
    PAPER_VERIFICATION.md、TWCC_CHECKLIST 新增 9 項首次執行驗證。
  - 新增開發依賴：pypdf（論文 PDF 文字抽取，非執行期依賴，不入 environment.yml）。
- 2026-07-24（v5 決策落地：SPEC/STRUCTURE 更新至 v5 後之實作調整）：
  - **批次化編輯（最重要之技術裁定）**：編輯純推論無梯度，已實作批次版
    （`edit_image_batch`，src/edit）。批次中每樣本各自 `torch.Generator`＋seed，
    diffusers 對 generator list 逐樣本取樣——**噪聲協定與 batch=1 位元級一致**
    （test_noise_protocol），故 seed 協定（SPEC §2.3）未被破壞。但 v5 啟用判準
    「batch=1 vs batch=4 逐位元相同」**實測未通過**：批次矩陣運算之歸約順序隨
    batch 尺寸而異，float32 差 ~5e-6（8-bit 存檔後 ~0.02% 像素 ±1 LSB），
    位元級等價原理上不可達。依判準 `edit_batch_size` **維持 1**（批次化不啟用）。
    tests/test_edit_batching.py 之位元級測試設為「啟用閘門」（=1 時 skip、>1 時
    執行且依實測失敗，阻止未經裁定之啟用）；容差測試（≤1e-4）驗證非協定破壞。
    **待指導者裁定**：若放寬判準為「噪聲協定位元級一致＋輸出容差 ≤1e-4」則可
    啟用，編輯時數 ÷4-6。TWCC 上須以真實模型重跑該測試。
  - **GrIDPure 預設（v5）**：purify.yaml settings 改為 paper 10×10（主設定）＋
    readme 10×20＋single_deep 100×1（三來源，後兩者為強度掃描端點）；
    pure_steps_scan 改 [5,15]（與 paper 之 10 合成 5/10/15 曲線）。
  - **stage1 summary（v5）**：依有無 DAYN 錨點分兩表——sdedit（img2img，主要
    條件，校準比對限此表 pg_* 列）與 inpaint（無錨點，僅以自實作 PhotoGuard
    為對照，不可比 DAYN Table 1）。
  - **T2 校準（v5，入 TWCC_CHECKLIST）**：norm 固定 linf（DAYN Alg.1 為
    sign+clip=ℓ∞），strength 候選 {0.8,0.7,0.5,0.3}（0.8 為 diffusers 預設，
    最可能值）；分階段搜尋（固定 0.8 掃 epsilon_scale 2 次 → 掃 strength 3 次 →
    對角確認 1 次，共 7 次，皆 encoder attack），格點由 4 組降為 2 組。
  - **閘門式規模方案（preflight [5] 重算）**：保真度花在有錨點條件（V1.4×img2img
    跑滿 20 seed），縮減集中在 stage2（其結論為 drop 相對差異，不需絕對精度）。
    完整 A ~14,200h fp32/~7,600h fp16（記帳基準）；**建議 G 閘門式 ~229h fp16
    單卡（4 卡 ~57h、8 卡 ~29h）**；首日 C ~43h。批次化封鎖後編輯加速僅剩 fp16
    （÷2）與多卡平行；成本模型改為 block 結構（stage1 各條件可不同 seed 數）。
  - 全測試 44 pass + 1 skip（新增 test_edit_batching.py 6 項）。
- 2026-07-23：專案根目錄採 `C:\WACV`（SPEC.md 所在處），不另建子目錄。
- 2026-07-23：configs/*.yaml 直接填入 STRUCTURE.md §3 範本內容（文件已完整給定）。

### 裝置資訊 2026-07-23 12:06（get_device_info()）

```json
{
  "torch_version": "2.13.0+cpu",
  "python_version": "3.11.15",
  "platform": "Windows-10-10.0.26200-SP0",
  "backend": "cpu",
  "vendor": "AMD64",
  "model": "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel",
  "total_memory_gb": 15.71
}
```
