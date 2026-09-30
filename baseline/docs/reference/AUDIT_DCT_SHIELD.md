# 逐行查證：DCT-Shield（`dct_shield`、`dct_shield_y`）

Bala, Gupta, Jain et al.，*DCT-Shield: A Robust Frequency Domain Defense
against Malicious Image Editing*，ICCV 2025 Highlight，
[arXiv:2504.17894](https://arxiv.org/abs/2504.17894)。

實作：`src/immunization_baseline/attacks/dct_shield.py`。主表兩列
`dct_shield`（位移排名第一）與 `dct_shield_y`（第三）都由這一支產生。

## 0. 這份查證與其他 `AUDIT_*.md` 的差別

其餘各篇的查證是**論文正文 vs 官方原始碼**逐行對照。DCT-Shield 沒有這一半：
官方 repo [`SamsungLabs/dct-shield`](https://github.com/SamsungLabs/dct-shield)
在 2026-08-18 查證時是空的（GitHub API 回 `This repository is empty.`、零分支），
project page 標 code coming soon。

**因此 `results/defense_dct_shield*.csv` 的 `modified_from_paper=False` 的依據是
論文正文與補充材料 Algorithm 1，不是程式碼比對**——那個比對在可查證的範圍內
不存在。同一列的 `spec_source` 欄寫的是「arXiv:2504.17894 補充材料 Algorithm 1」，
與其他條件的 `spec_source`（多半含 repo commit hash）性質不同。

同樣地，`BASELINE_PROVENANCE.md` 記載該篇 Table 1 的 baseline 分支（encoder 還是
complex）與 LPIPS backbone 未載，補充材料 C.1.5 只說用 `pyiqa`。**它報的數字在
本專案只能當 as reported 引用**，不構成本專案實作正確與否的對照。

## 1. 逐行對應補充材料 Algorithm 1

`src/immunization_baseline/attacks/dct_shield.py` 的 docstring 記了這個對應，原文：

| Algorithm 1 | 實作 |
|---|---|
| 行 1 `δ ← 0` | `delta` 初始化為零 |
| 行 2 `α ← JPEG_E(x; Q_alg)` | `jpeg_encode`（式 7） |
| 行 5 `for i = 0 … N−1` | `steps`，定案值 1000 |
| 行 6 `η ← (1 − i/N)·γ` | 線性衰減到 0，本檔自行實作 |
| 行 8 `α'_c ← α_c + δ_c` | 逐通道加，`channels` 限定範圍 |
| 行 9 `x' ← JPEG_D(α'; Q_alg)` | `jpeg_decode`（式 9） |
| 行 10 `L ← ‖E(x')‖₂` | `make_latent_norm_loss`，latent 全元素 L2 範數 |
| 行 11 `δ ← δ − sign(∇_δ L)·η` | 下降 |
| 行 12 `δ ← clamp(δ, −ε, +ε)` | `eps` |
| 行 14 `δ_c ← δ_c ⊙ M_c` | `_component_masks` |

定案超參數取自正文 §5.4：`Q_alg = 0.95`、`ε = 1`、`γ = 0.1`、`N = 1000`、512²。

## 2. 論文未載而實作必須自己決定的三處

這三處都不在 `modification_note` 裡，因為兩個 spec 都標 `modified_from_paper=False`。
它們是實作為了能跑而做的決定，出處只到「論文沒寫」為止：

1. **步長排程的實作位置。** `(1 − i/N)γ` 線性衰減到 0 不在 `pgd.py` 的
   `STEP_SCHEDULES`（只有 `constant` 與 `linear_decay_1pct`）之中，由
   `dct_shield.py` 自行實作。理由寫在該檔：DCT-Shield 不是像素加性，套進
   五篇共用的 PGD 骨幹會使 `BaselineSpec` 的欄位語意失效。

2. **`component_wise_masks`（行 4）的細節未載。** 實作取「8×8 區塊內有任一像素
   被遮罩就整塊保留」，理由是 DCT 係數的支撐是整個區塊。原始碼註明
   **「這是本檔的推斷，不是論文的規定」**。

3. **Y-only 變體的定位。** `dct_shield_y` 取 `channels=("Y",)`、`Q_alg=0.85`，
   出處是 §4.3 與 §6.3。論文那組設定用於**圖 6 的 JPEG 強健度圖**，不是
   §5.4 的定案編輯設定；主表把它當成與 `dct_shield` 並列的一個條件。

## 3. 實作端已量到、與該篇報的數字不一致的地方

出處為 `BASELINE_ALIGNMENT.md` 與 `dct_shield.py` 的診斷註解，協定不同的兩組
數字並排時各自的協定如下。

| 量 | 本專案 | 該篇 Table 1 | 協定差異 |
|---|---|---|---|
| LPIPS（防禦圖 vs 原圖） | 0.5532（VGG） | 0.267 | 該篇 150 張 OmniEdit、backbone 未載；本專案 8 張人像、`piq.LPIPS()` VGG |
| PSNR | 29.75 | 27.61 | 同上 |
| rms | 0.0326 | 0.0416\* | \* 非表中欄位，由 PSNR 反推（規則 4） |

`dct_shield.py` 記錄的診斷：實作出的 δ 在每個係數上都吃滿 ±1（Y 通道 |δ| 中位
0.903、50.3% 超過 0.9），DC 是最飽和的位置（Y 1.02、Cb 1.13、Cr 0.96）。
**能量更小卻失真更大**（PSNR 高 1.83 dB、LPIPS 1.76 倍）。為檢定「擾動放錯位置」
這個假設，該檔另有一個 `skip_dc` 旗標；它不在論文裡，`__post_init__` 強制要求
啟用它就必須標 `modified_from_paper`，主表兩列都未啟用。

## 4. 該篇自己聲明的適用邊界

- **`ε ≥ 1` 是抗 JPEG 的必要條件**（§4.2）：擾動必須至少造成一個量化級的改變，
  否則攻擊方以相同品質重壓時會被四捨五入回原值。`__post_init__` 把
  `eps < 1` 且未標 `modified_from_paper` 的設定直接拒絕。
- **抗 JPEG 的保證是單向的**（補充材料 D.4）：`Q_alg = q` 產生的影像只在攻擊方
  壓縮品質 `q' ≥ q` 時有效。預設 `Q_alg = 0.95` 只涵蓋品質 95 以上的壓縮。
- **δ=0 時輸出不是原圖**，而是 `Q_alg` 品質的 JPEG 壓縮圖（失真地板）。
- **免疫影像不可存成 JPEG**：δ 是連續值、加在整數係數上，再壓一次會被四捨五入掉。
  本專案一律存 PNG。

## 5. 測試覆蓋

`archive/anti-purification/tests/test_baselines.py` 的 `AUDIT` 與 `AUDIT_STEP` 兩個字典斷言
`set(AUDIT) == set(REGISTRY)`，而該 `REGISTRY` 是 `src/immunization_baseline/attacks/__init__.py` 的
六個共用 PGD 骨幹的 spec（`photoguard_c`、`mist`、`dia_pt`、`dia_r`、`advpaint`、
`promptflare`）。**DCT-Shield 走自己的 `DCTShieldSpec` 與自己的 `REGISTRY`，
不在那個集合內**，故上述斷言不涵蓋它的任何常數。

`dct_shield.py` 自帶的 `__post_init__` 檢查涵蓋三件事：未知通道、`skip_dc` 與
`eps < 1` 未標註。這三項以外的常數（`Q_alg`、`γ`、`N`、`channels`）由
`tests/test_conditions.py` 對 `results/defense_*.csv` 釘住。
