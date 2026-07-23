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
