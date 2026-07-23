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
