# 專案結構、介面與 Pipeline

配合 `SPEC.md`（演算法規格）與 `PROMPTS.md`（Claude Code 指令）使用。本文件定義「怎麼組織」，SPEC.md 定義「怎麼實作」。

---

## §1 目錄結構

```
nonadd-immunization/
├── SPEC.md                      # 演算法規格（唯一實作依據）
├── PROMPTS.md                   # Claude Code 分階段指令
├── STRUCTURE.md                 # 本文件
├── NOTES.md                     # 執行紀錄：硬體、套件版本、決策理由
├── README.md
├── environment.yml              # 硬體確認後填入
│
├── configs/
│   ├── base.yaml                # 模型、資料、編輯、指標
│   ├── additive.yaml            # PhotoGuard 參數
│   ├── nonadditive.yaml         # 非加性參數（相似性約束由 stage0 填入）
│   └── purify.yaml              # 淨化方法參數
│
├── data/
│   ├── dayn_testset/            # 向作者索取，勿進版控
│   └── README.md                # 記錄資料來源與取得日期
│
├── src/
│   ├── models/
│   │   └── sd_wrapper.py        # SD V1.4/V2.0 封裝，暴露 cross-attention
│   ├── protect/
│   │   ├── base.py              # 統一介面（見 §2.1）
│   │   ├── photoguard.py        # 加性 · 校準錨點
│   │   ├── advdiff_based.py     # 非加性 · AdvDiff 注入
│   │   ├── apa_based.py         # 非加性 · APA inversion + 兩階段
│   │   └── hybrid.py            # 非加性 · APA 骨架 + AdvDiff 注入
│   ├── edit/
│   │   ├── sdedit.py            # img2img
│   │   ├── inpaint.py
│   │   └── seed_protocol.py     # SPEC §2.3 的 seed 搜尋與固定
│   ├── purify/
│   │   ├── lightweight.py       # JPEG / blur / crop-resize
│   │   ├── adverse_cleaner.py   # BF only + BF+GF 兩變體
│   │   └── gridpure.py
│   ├── metrics/
│   │   └── quality.py           # piq 封裝，含方向定義
│   ├── data/
│   │   └── dataset.py           # 載入 DAYN 測試集
│   └── utils/
│       ├── device.py            # 裝置抽象（禁用 .cuda()）
│       ├── seed.py
│       └── io.py                # 結果存取、csv 輸出
│
├── scripts/
│   ├── stage0_calibrate.py
│   ├── stage1_clean.py
│   └── stage2_purify.py
│
├── experiments/                 # 輸出（勿進版控）
│   ├── stage0/
│   ├── stage1/
│   └── stage2/
│
├── external/                    # clone 的官方 repo（勿進版控）
│   ├── photoguard/
│   ├── GrIDPure/
│   └── SDEdit/
│
└── tests/
    └── test_smoke.py            # 極小規模全流程煙霧測試
```

---

## §2 關鍵介面

### 2.1 保護方法統一介面（`src/protect/base.py`）

所有方法實作此介面，使 stage 腳本可統一呼叫、確保公平比較。加性與非加性僅在 `protect()` 內部不同，對外一致。

```python
from abc import ABC, abstractmethod
import torch


class ProtectionMethod(ABC):
    """所有保護方法的抽象介面。"""

    def __init__(self, sd_wrapper, config: dict):
        self.sd = sd_wrapper
        self.cfg = config

    @abstractmethod
    def protect(self, image: torch.Tensor, concept: str) -> torch.Tensor:
        """產生受保護影像。

        Args:
            image:   (1,3,H,W)，值域 [0,1]
            concept: 欲保護之語意概念，如 "dog"。
                     PhotoGuard 不使用此參數。

        Returns:
            (1,3,H,W)，值域 [0,1]
        """

    @property
    @abstractmethod
    def is_additive(self) -> bool: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    def peak_memory_mb(self) -> float:
        """回傳上次 protect() 的峰值記憶體，供資源比較用。"""
```

### 2.2 編輯介面（`src/edit/`）

```python
def edit_image(
    image: torch.Tensor,        # (1,3,H,W)，[0,1]
    prompt: str,
    seed: int,
    method: str,                # "sdedit" | "inpaint"
    mask: torch.Tensor = None,  # inpaint 時必要
    config: dict = None,
) -> torch.Tensor:
    """對影像執行編輯。

    生成超參數依 SPEC §2.2：
        height=512, width=512, guidance_scale=7.5,
        num_inference_steps=100, eta=1
    SDEdit 的 strength 由 config 提供（SPEC §2.8 標記待確認）。
    """
```

### 2.3 Seed 協定（`src/edit/seed_protocol.py`）

實作 SPEC §2.3。此為實驗公平性的必要條件。

```python
def search_valid_seeds(
    image: torch.Tensor,
    prompt: str,
    n_seeds: int = 20,
    config: dict = None,
) -> list[int]:
    """在原圖上搜尋能產生合理編輯結果的 seed。

    回傳的 seed 清單將同時用於原圖與受保護影像的編輯，
    確保兩者編輯條件完全相同。

    「合理」的判準需與指導者確認；預設以編輯結果與
    prompt 的 CLIP score 高於門檻為準。
    """
```

### 2.4 淨化介面（`src/purify/`）

```python
def purify(
    image: torch.Tensor,
    method: str,  # "jpeg"|"blur"|"crop_resize"|"advclean_bf"|"advclean_bfgf"|"gridpure"
    strength,     # 各方法的強度參數，見 configs/purify.yaml
) -> torch.Tensor:
    """移除保護訊號。流程為「淨化 → 編輯 → 測劣化」。"""
```

### 2.5 指標介面（`src/metrics/quality.py`）

```python
# 方向定義：True 表示「數值越高、防禦越成功」
# 採 DAYN 慣例（SPEC §2.6）。PhotoGuard 用相反慣例，勿混淆。
METRIC_HIGHER_IS_BETTER = {
    "psnr":  False,
    "ssim":  False,
    "vifp":  False,
    "fsim":  False,
    "lpips": True,
    "fid":   True,
    "clip":  False,
}


def compute_all(
    edited_protected: torch.Tensor,
    edited_original: torch.Tensor,
    prompt: str = None,   # CLIP score 需要
) -> dict[str, float]:
    """計算七項指標。

    衡量的是兩個「編輯結果」之間的差異，差異越大代表保護越成功。
    PSNR/SSIM/VIFp/FSIM/LPIPS 一律用 piq（SPEC §2.6），不可自行實作。
    """
```

### 2.6 裝置抽象（`src/utils/device.py`）

```python
def get_device() -> torch.device:
    """偵測可用裝置。GPU 廠牌不確定，可能非 NVIDIA。

    全專案禁用 .cuda()，一律經此函式。
    """


def get_device_info() -> dict:
    """回傳廠牌、型號、記憶體、後端（cuda/rocm/cpu），寫入 NOTES.md。"""
```

---

## §3 設定檔範本

### configs/base.yaml

```yaml
model:
  protect_model: "CompVis/stable-diffusion-v1-4"   # 生成保護時攻擊之模型
  eval_models:                                      # 評測模型
    - "CompVis/stable-diffusion-v1-4"
    - "stabilityai/stable-diffusion-2-base"

generation:            # SPEC §2.2，取自 PhotoGuard Table 8
  height: 512
  width: 512
  guidance_scale: 7.5
  num_inference_steps: 100
  eta: 1

edit:
  methods: ["sdedit", "inpaint"]
  sdedit_strength: null      # SPEC §2.8 待確認，向作者索取
  n_seeds: 20                # SPEC §2.4

data:
  root: "data/dayn_testset"
  source: "向 DAYN 作者索取"
  is_placeholder: true       # 取得原始資料集後改為 false
  n_images: 150              # SPEC §2.4
  n_object_classes: 3
  prompts_per_class: 2

metrics:
  backend: "piq"             # SPEC §2.6，不可換
  compute: ["psnr", "ssim", "vifp", "fsim", "lpips", "fid", "clip"]

runtime:
  # SPEC §1.4 之 batch_size=1 係為「保護生成」而設（有梯度、計算圖深）
  protect_batch_size: 1

  # v5：編輯為純推論、無梯度，可批次化。此為最大加速槓桿（stage2 佔約 93% 成本）
  # 前提：批次中每個樣本須使用各自的 generator 並各自 seed，
  #       否則 diffusers 在 batch>1 時的噪聲取樣與 batch=1 不同，
  #       將破壞 PhotoGuard seed 協定（SPEC §2.3）
  # 驗證要求：4 張影像分別以 batch=1 與 batch=4 執行，輸出須逐位元相同
  edit_batch_size: 4

  seed: 42
```

### configs/additive.yaml

```yaml
# SPEC §3.1：採 DAYN 重現用設定，非 PhotoGuard 原設定
# v3：依官方程式碼核對結果，新增三個待校準參數
photoguard:
  # --- 待 T2 校準階段以實驗決定的三個參數 ---
  epsilon_scale: "pm1"      # "pm1" (-1~1) | "01" (0~1)
                            # SPEC §3.2.1：官方在 [-1,1] 運作，eps=0.06 換算至
                            # [0,1] 僅 0.03。DAYN 未指明尺度，兩者差兩倍。
  norm: "linf"              # "linf" | "l2"
                            # SPEC §3.2.2：官方 demo 預設 L2(eps=16)，
                            # ℓ∞ 僅於註解提供；DAYN 未指明。
  target_latent: "zeros"    # "zeros" (官方預設) | "gray" (論文文字描述)
                            # SPEC §3.2：官方最小化 ‖ℰ(x')‖，即 z_targ=0。
                            # 灰階 latent 非零向量，兩者為不同目標。

  # --- 已確定的參數 ---
  epsilon: 0.06             # DAYN 用 0.06；PhotoGuard 原文為 16/255
  step_size: 0.006          # epsilon / 10
  step_decay: "linear"      # v3：官方有 step size 線性衰減
  random_init: true         # v3：官方有隨機初始化
  n_iter: 100               # DAYN 用 100；PhotoGuard 原文為 200
  diffusion_T: 10           # diffusion attack 反傳步數
  grad_reps: 10             # v3：官方 diffusion attack 梯度取 10 次平均(EOT)
                            # v5 命名修正：v4 範本誤寫為 diffusion_eot，與實作不符
                            # 此為 diffusion attack 成本約 10 倍於 encoder 之主因
  projection: "from_original"   # 見 SPEC §3.2 註
```

**T2 校準階段的參數組合**：`epsilon_scale`(2) × `norm`(2) = 4 種組合，各跑一次 PhotoGuard 並與 DAYN Table 1 的 Encoder/Diffusion 兩欄比對，以最接近者為準。若 `target_latent` 亦需試，則為 8 種組合。建議先固定 `target_latent="zeros"`（官方預設）跑 4 種，必要時再擴充。

### configs/nonadditive.yaml

```yaml
common:
  similarity_metric: "lpips"
  similarity_budget: null     # 由 stage0_calibrate.py 填入

advdiff:                       # SPEC §4.3（v4：依官方 ddim_adv.py 修正）
  T: 10
  N: 5                         # 官方 K=5
  a: 0.5                       # 起始噪聲注入強度
  guidance_range: [0.0, 0.2]   # 附錄 E：僅在反向過程末段施加
  eta: 0.0
  reward: "attention"          # SPEC §4.2 方案一

  # v4 修正：官方在 LDM+DDIM 下仍用後置加法，且無係數
  injection_form: "post_hoc"   # 非 "modify_epsilon"
  injection_coeff: none        # 官方將 σ²_t、√(1−ᾱ) 等係數全部註解停用

  # v4 修正：起始噪聲注入採 skip-gradient，不反傳穿鏈
  # 梯度於最終 latent 計算後直接套用至 z_T
  ztT_update: "skip_gradient"

  # s 值官方自身不一致：論文 0.7 / driver 1.0 / 函式簽名 0.75
  # 採論文值為起點；本專案 reward 定義不同，預期需重調
  s: 0.7

apa:                           # SPEC §4.4（v4：依官方 repo 修正）
  variant: "gc"                # "sg" | "gc"
  T_a: 10                      # 50 步排程中 index_cond=40，即最後 10 步
  N: 10
  eps_a: 0.4
  mu: 0.04
  guidance_scale: 1.0          # inversion 與攻擊均無 CFG
  reward: "attention"

  # --- Stage 1（v4：官方值，LoRA rank 待確認項已解決）---
  lora_impl: "peft"            # 官方使用 peft
  lora_rank: 8                 # 官方值（v1 誤植為 4，該值實為 AntiPure 設定）
  lora_alpha: 8
  lora_target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
  lora_init: "gaussian"
  lora_lr: 1.0e-4
  lora_weight_decay: 1.0e-2
  lora_grad_clip: 1.0
  lora_steps: 200              # v3 誤設 100
  noise_offset: 0.1

  # --- SG 與 GC 的實質差異（v4 重大修正）---
  sg:
    # 軌跡梯度不反傳穿鏈，於最終 latent 計算後乘常數直接套用
    trajectory_grad: "skip"
    scale_constant: 14.58      # 官方經驗常數，未說明來源；本專案預期需重調
    inversion: "full"          # 完整 inversion（50 步）

  gc:
    # 「T=10」非全程 10 步，而是部分 inversion 至 t≈0.2T
    trajectory_grad: "checkpoint"   # torch.utils.checkpoint
    inversion: "partial"
    inversion_steps: 10        # 於 50 步格點上只走前 10 步
    sampling_steps: 11         # 只跑最後 11 步
    mse_reg_weight: 10.0       # GC 版 reward 含 −10·MSE(z_ori, z_final)，SG 版無

  # --- 式 (12) diffusion augmentation（v4 修正）---
  aug:
    # 官方：增強影像 = (D(x̂_0^t) + 最終生成影像)/2，x̂_0^t 為 detached
    # v3 誤實作為與原圖 latent 平均
    base: "final_generated"    # 非 "original"
    resize_range: [0.9, 1.0]   # random resize 後零 padding 回原尺寸
    brightness: false          # 官方 code 中註解停用

hybrid:                        # SPEC §4.5，本專案設計
  base: "apa"
  injection: "advdiff"
```

**v4 註**：AdvDiff 與 APA-SG 均採 skip-gradient，不需保留採樣計算圖，V100 32GB 記憶體壓力較 v3 評估顯著降低（見 SPEC §4.2）。

### configs/purify.yaml

```yaml
# SPEC §5。DiffPure 已移出獨立條件，其機制存在於 GrIDPure 內部
jpeg:
  quality: [90, 80, 65, 50]    # 65 為社群標準設定

blur:
  sigma: [0.5, 1.0, 1.5]

crop_resize:
  ratio: [0.1, 0.2, 0.3]       # 0.2 為標準設定

adverse_cleaner:               # 即 BF+GF，兩變體
  variants: ["bf_only", "bf_gf"]
  # v5 修正：官方 clean.py 為 64×BF + 4×GF，v4 記為 3+1 低估約二十倍
  bf_iterations: 64
  gf_iterations: 4
  bf_d: 5
  bf_sigma_color: 8
  bf_sigma_space: 8
  gf_radius: 4
  gf_eps: 16

gridpure:                      # SPEC §5.3
  grid_size: 256
  stride: 128
  n_grids: 10                  # 九個 + 四角落組成的第十個
  gamma: 0.1

  # v3：參數尺度為 1000 步 DDPM 的原始 timestep
  # 官方提供兩組性質不同的設定，階段二應兩組都測
  presets:
    paper_default:             # v5：論文預設，以此為主要設定
      pure_steps: 10
      iterations: 10
    shallow_iterative:         # README 建議值：迭代較多
      pure_steps: 10
      iterations: 20
    deep_single:               # 腳本預設值：單次深淨化（退化為 DiffPure 行為）
      pure_steps: 100
      iterations: 1

  # 強度掃描（用於繪製「淨化強度 vs 防禦效果」曲線）
  pure_steps_sweep: [5, 10, 20]
  iterations_sweep: [10, 20, 30]

  # checkpoint：pixel-space 無條件 guided diffusion，約 2GB，TWCC 須另行下載
  # 成本警告：單張 512x512 在 V100 上約 2 分鐘
```

---

## §4 Pipeline 流程

### 4.1 執行順序與依賴

```
指令 A  環境偵查 + 骨架 + clone 官方 repo
   ↓
指令 B  模型封裝 + 資料載入 + 編輯 + 指標
   ↓         └─ 煙霧測試：3 張圖跑通全流程
指令 C  PhotoGuard（校準錨點）
   ↓         └─ ★ 校準比對：與 DAYN Table 1 的 Encoder/Diffusion 欄對照
指令 D  非加性三方法
   ↓         └─ 小規模驗證：梯度可算、方向正確、記憶體可控
指令 E  淨化模組 + 三支實驗腳本
   ↓
stage0 → stage1 → stage2
```

★ 為硬性檢查點：PhotoGuard 若無法對齊 DAYN 報告的數值，代表資料集、編輯設定或實作有誤，後續所有比較均失去意義，須先解決。

### 4.2 stage0：相似性約束校準

```
輸入：測試集、PhotoGuard（κ=0.06）、非加性三方法

1. 對每張測試影像跑 PhotoGuard，計算 LPIPS(protected, original)
   → 取平均得基準值 L_ref
2. 對每個非加性方法，掃描 similarity_budget（由嚴至寬）
   → 記錄各設定下的 LPIPS 平均
3. 選出使 LPIPS ≈ L_ref 的設定
4. 寫入 configs/nonadditive.yaml 的 similarity_budget
5. 輸出校準曲線圖至 experiments/stage0/

輸出：configs/nonadditive.yaml（已填值）、calibration_curve.png、calibration.csv
```

此步為 stage1 結論可信度的前提。非加性無 pixel norm 約束，若不校準，「效果較好」可能僅因允許更大改動。

### 4.3 stage1：乾淨情況比較

```
方法矩陣 = {PhotoGuard-enc, PhotoGuard-diff,
            AdvDiff-based, APA-based, Hybrid}
         × {SD V1.4, SD V2.0}
         × {sdedit, inpaint}

對每個組合、每張測試影像：
  1. seeds = search_valid_seeds(x, prompt)        # SPEC §2.3
  2. protected = method.protect(x, concept)
  3. 對每個 seed：
       edited_orig = edit(x, prompt, seed)
       edited_prot = edit(protected, prompt, seed)
       m = compute_all(edited_prot, edited_orig, prompt)
  4. 20 seed 取平均

輸出：experiments/stage1/results.csv
欄位：method, model, edit_method, image_id, prompt_idx,
      psnr, ssim, vifp, fsim, lpips, fid, clip,
      peak_memory_mb, elapsed_sec

註：prompt_idx 為 v5 新增（每類物件 2 個 prompt 須逐一評測）。

比較對象：SPEC §2.7 的 DAYN Table 1（引用值，不重現）
校準檢查：PhotoGuard 數值 vs 該表 Encoder / Diffusion 欄
通過條件：非加性至少一變體在多數指標上不劣於該表 DAYN 欄
```

### 4.4 stage2：淨化後比較

```
淨化清單 = {jpeg×4, blur×3, crop_resize×3,
            advclean_bf, advclean_bfgf, gridpure×3}

對 stage1 產生的每張受保護影像：
  1. purified = purify(protected, method, strength)
  2. edited_pur = edit(purified, prompt, seed)     # 同 stage1 的 seed
  3. m_purified = compute_all(edited_pur, edited_orig, prompt)
  4. drop = (m_clean − m_purified) / m_clean

輸出：experiments/stage2/results.csv
      experiments/stage2/purify_strength_curve.png

核心比較：加性 vs 非加性的 drop 值
待驗證假設（非預設結論）：非加性的 drop 顯著較小
```

**執行前務必估算時數並回報**。GrIDPure 單張 512×512 在 V100 上約 2 分鐘；150 張 × 5 保護方法 × 3 個 timestep 設定約 75 小時，須評估平行化或縮減規模。

### 4.5 曲線圖要求

stage2 必須繪製「淨化強度 vs 防禦效果」曲線，而非僅比較單點。原因：非加性可能在乾淨情況下僅與加性持平，於淨化後才拉開差距（交叉現象）。單點比較會漏掉這個關鍵訊號。

---

## §5 輸出規範

所有實驗結果須附完整設定與隨機種子，確保可重現。每次執行在 `experiments/<stage>/<timestamp>/` 下產生：

```
config_snapshot.yaml      # 該次執行的完整設定
results.csv               # 原始數據
summary.md                # 人可讀的摘要
env.json                  # 硬體、套件版本
```

`summary.md` 須明確區分兩類數值：**引用值**（DAYN Table 1）與**實測值**（本專案執行）。比較時揭露此層次，勿混為一談。

---

## §6 版控排除

`.gitignore` 至少包含：

```
data/dayn_testset/
experiments/
external/
*.ckpt
*.safetensors
__pycache__/
```

資料集為向作者索取，未經同意勿公開。
