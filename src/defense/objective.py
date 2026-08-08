"""損失函數 — `docs/DESIGN_2026-08-05.md` §2.1、§3.1、§4。

    L(φ) = λ_def · L_def + λ_fid · L_fid

## 防禦項 L_def：三個條件都是 targeted

`docs/LOGIC_CHECK_2026-08-05.md` A1 的處置。三個非加性條件的著力點不同，
但**沒有一個是無目標的最大化**：

| 條件 | `defense_mode` | L_def | 目標從哪裡來 |
|---|---|---|---|
| N1 | `targeted_attn` | `1 − (shared token 分到的注意力質量)` | 空 prompt 的 CLIP 編碼（CFG 的無條件嵌入）中恆為 shared 的位置 |
| N2 | `targeted_output`（`target_metric="lpips"`） | `LPIPS(y_def, y_target)` | 固定的目標影像 |
| N3 階段二 | `targeted_output`（`target_metric="mse"`） | `‖SDEdit_sub(x_adv; c_∅) − x_target‖²` | 同上，度量改為 MSE（`DESIGN` §4） |
| N4 | `suppress_attn_ca` | `‖Att(x_adv, c_a) ⊙ M‖₁`（Lo 式 5） | **防禦方指名要保護的詞 c_a**，遮罩 M 由原圖的注意力取（式 4） |

前三者都是 prompt-free：N1 的 shared token 在**任何**攻擊 prompt 中都出現且
語意無資訊，N2／N3 的去噪期一律餵空 prompt。防禦方因此不需要知道攻擊方的
prompt，也不需要為原圖產生 caption。

### `suppress_attn_ca` 與 N1 的 `targeted_attn` 是兩件不同的事

兩者都關於 cross-attention，但著力點、作用範圍與威脅模型都不同：

| | N1 `targeted_attn` | N4 `suppress_attn_ca` |
|---|---|---|
| 施力方向 | 把注意力質量**導向** decoy token（BOS／末位 PAD） | 把指定詞的注意力反應**壓低** |
| 作用位置 | 全部 query 位置取平均 | **只在式 (4) 的遮罩 M 內** |
| 需不需要 c_a | **不需要**，decoy 的索引由 tokenizer 結構決定 | **需要**，沒有 c_a 就沒有攻擊目標 |
| 量的形狀 | 每層壓成純量再平均（全域綁定強度） | 保留空間維度的聚合圖，取遮罩內的 L1 |
| 威脅模型 | prompt-free | **防禦方必須指名要保護什麼** |

最後一列是實質的改變，不是實作細節。c_a 由 `data/lo_aligned/prompts.yaml`
的 `content` 欄給定（`ImageEntry.content`），而編輯 prompt 由同檔的 `prompts`
欄給定——**兩者在威脅模型裡屬於不同的人**：prompt 是攻擊方寫的，c_a 是
防禦方選的。該檔第 8–11 行逐字寫明這件事，`run_lo_baseline.load_dataset`
的 docstring 亦然。選錯 c_a 會讓損失壓到別的區域而**沒有任何症狀**，故
`LossConfig` 對本模式的缺項一律拋出，不接受預設值。

本模式的三個函式實作在 `src/models/attention.py`，對應 Lo et al.
(CVPR 2024) 的式 (3)(4)(5)：`aggregate_token_attention`、
`attention_region_mask`、`masked_attention_l1`。它們在 2026-08-03 就已寫好
並由 `tests/test_lo_protocol.py` 釘住，但在本次改動之前**沒有任何訓練條件
在用**——只有 `src/defense/linf_attack.py` 的文獻基準路徑（PGD on L∞ ball）
呼叫它們。本模式把同一組式子接到本專案的參數化最佳化上。

### 刻意不做：`attention_entropy`

`src/models/attention.py` 另有 `attention_entropy`（注意力分佈在 token 維度
上的熵，**不需要 c_a**）。它與本模式問的是不同的問題——「破壞綁定」對
「把綁定改指向別處」哪一種比較能跨 prompt 泛化——**本輪刻意不納入格點**，
以免把批次成本翻倍。列為待辦：要做時它已經在那裡，接法與本模式相同，
差別只在不需要 c_a 與遮罩，故威脅模型退回 prompt-free。

### 為什麼 N1 取「導向 shared token」而不是「壓低內容 token」

`attention_content_suppression`（本專案 E25 的變體）需要攻擊 prompt 的內容
token 區間才有定義，那與 prompt-free 的前提衝突：`token_span` 要拿得到攻擊方
的 prompt 才算得出 span。改為對 shared token 施力之後，作用的位置只依賴
CLIP tokenizer 的結構（`[BOS]` 恆在第 0 格），與 prompt 的內容無關。
形式取自 PromptFlare 的 cross-attention decoy，其 `L_CA` 正是對 `M^BOS` 取的。

### 修訂紀錄：untargeted 已移除

    before（本檔 2026-08-04 版 line 87、204-208）
        defense_mode: str = "untargeted"
        ...
        elif mode == "untargeted":
            terms = [
                torch.clamp(self.cfg.margin - self.distance(yd, yo), min=0.0)
                for yd, yo in zip(y_def_list, y_orig_list)
            ]

    after
        該模式整個移除。`LossConfig.__post_init__` 對 "untargeted" 明確拋出，
        訊息指向缺陷 A1。

原因：`max(0, m − LPIPS(y_def, y_orig))` 在 `y_def = y_orig` 處取到**最小值**，
故 φ = 0 且淨化算子為 identity 時起點梯度精確為零（實測 step 0 的
`grad_norm = 0.000e+00`）。它至今能跑起來，靠的是訓練期淨化算子輪替到第 1 步
換成 blur 才打破對稱——也就是「抗淨化」這個附帶目標同時是主損失的啟動器。
該缺陷涵蓋先驗實驗 59 個有紀錄批次的 **100%**。留著一個「可以選」的錯誤模式，
等於留著一條會靜默產生同一批無效資料的路徑。行為的記錄與淘汰理由由
`tests/test_zero_gradient.py` 保存。

`margin` 一併移除：hinge 存在的理由是「無界的最大化會發散」，targeted 是
最小化一個下界為 0 的距離，不會發散，不需要 hinge。

## 保真項 L_fid：四道 hinge 取交集

    L_fid = β  ·max(0, L∞      − τ_linf)    β   = beta_linf
          + γ_l·max(0, LPIPS   − τ_lpips)   γ_l = gamma_lpips
          + γ_a·max(0, acut    − τ_acut)    γ_a = gamma_acut
          + γ_c·max(0, chroma  − τ_chroma)  γ_c = gamma_chroma
          + γ  ·max(0, τ_psnr  − PSNR)      γ   = gamma_psnr，預設 0.0

取交集而非加權和的理由：加權和永遠可以用便宜的軸換貴的軸，交集式的可行域
不允許這種交換。每一項都是 `clamp(·, min=0)`，故**沒有任何一項可以為負**，
一道 hinge 的懲罰不可能被另一道的改善抵銷——這是「交集」在程式上的形式。

四道 hinge 的對象都是 `x_base = G(x; φ=0)`（該位置未施加防禦時就已產生的
圖），不是原圖 `x`。量的因此是「防禦本身加了多少」，兩個 site 彼此可比。
對原圖的絕對值仍以 `fid_lpips`、`fid_psnr_total`、`fid_linf_total` 記錄。

`acut` 是 `local_acutance_dev`（鈍化），`chroma` 是 `local_chroma_bias`
（空間連貫的色偏）。兩者都是由等 LPIPS 多條件探針判別出來的。

### 修訂紀錄：加權和的兩項殘留已移除

    before（本檔 2026-08-04 版 line 94、108、303-311）
        alpha_ssim: float = 0.0
        alpha_lpips: float = 1.0
        ...
        total = (
            c.alpha_lpips * lpips
            + c.alpha_ssim * (1.0 - ssim)
            + c.beta_linf * pen_linf
            + ...
        )

    after
        兩個係數與其在 `total` 中的項全部刪除；LPIPS 與 SSIM 仍逐步記錄於
        `parts`（`fid_lpips`、`fid_ssim`），只是不再進入梯度。

兩個理由，各自獨立成立：

1. **SSIM 是已排除的度量**（`LOGIC_CHECK` A8）。SSIM／NLPD／VIF／GMSD／
   HaarPSI／ΔE 系列只能出現在回報欄位，不得進入 hinge。E20 實測 SSIM 在等
   LPIPS 下把雜訊判得比模糊貴約 2 倍，留在梯度裡等於補貼模糊。
2. **`alpha_lpips` 是加權和的另一半**。E27 實測：它是一個持續把失真往零拉的
   力，最佳化停在「邊際防禦收益 = 邊際 LPIPS 成本」的平衡點，而該點在 τ
   之下——site C 末端 LPIPS 0.031–0.045、site P 0.040，τ=0.05 的 hinge 在
   60 步中只啟動 0–8 步。τ 因此不是綁定的那道約束，兩個條件停在各自不同的
   失真上，「匹配失真的比較」不成立。本輪以射線縮放量整條曲線
   （`DESIGN` §3.2），更不能容許這個把預算往回拉的力。

## 門檻的適用範圍

`tau_acut` 與 `tau_chroma` 是在 τ_lpips = 0.05 的量級上定出來的**絕對值**。
本輪主表在 τ_lpips = 0.20，這兩個門檻必須重新以人眼判讀定出，不可沿用。
沿用未經該預算校準的門檻正是本專案重複十次的缺陷型態。

**2026-08-08 修訂：改為正比於 τ_lpips**（使用者裁決，處置 A）。

before：`LossConfig.tau_acut = 0.04`、`tau_chroma = 0.8` 為絕對值，
與 `tau_lpips` 無任何連動（本檔第 213、220 行）。
after：新增 `ACUT_PER_TAU` / `CHROMA_PER_TAU` 兩個比例常數與
`scaled_thresholds()`，由呼叫端依該批的 τ 導出。**兩個 dataclass 預設值不動**
——它們是 τ=0.05 的既有結果所依賴的值，改掉會讓舊批次無法重現。

理由是第一階段的實測：六個訓練格的 `fid_acut` 全部精確頂在 0.04015–0.04038，
即三個非加性條件**從未在主表的 τ=0.20 上最佳化過**，停在 LPIPS 0.052–0.174
就被鈍化 hinge 擋下，之後由段 2 的射線縮放放大上去。由
`runs/tau_threshold_probe_v14.csv` 線性內插，各條件撞到 acut=0.04 時的 LPIPS
為 0.066（bird_03·R）至 0.193（dog_03·N2），全部低於主表的 0.20；而加性
baseline（PhotoGuard-c）到 τ=0.35 都沒撞到，綁定它的是 `tau_lpips`。
也就是說這道 hinge 對非加性單方面加價，**匹配失真的比較在訓練期就不成立**。

比例的錨點是使用者在 τ_lpips = 0.05 上的原判讀：0.04 = 0.8 × 0.05、
0.8 = 16 × 0.05。該規則同時重現另外兩次獨立的人眼判讀：τ=0.20 時
（`RESULTS_2026-08-06` §10.1 判為可接受）門檻 0.16 不觸發，τ=0.35 時
（同節判為「明顯壞掉」）N3 實測的 0.343／0.355 仍高於門檻 0.28 而觸發。

## 修訂史

2026-08-04 以前的五次修訂在 `docs/OBJECTIVE_HISTORY.md`。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.metrics.chroma import local_chroma_bias
from src.metrics.local_acutance import local_acutance_dev

# 已被淘汰的 `defense_mode`，與其淘汰理由。留在程式裡是為了讓誤用有一個
# 具名的攔截點：靜默接受一個舊字串會讓整批實驗跑完才發現量的是別的東西。
RETIRED_MODES = {
    "untargeted": (
        "`max(0, m − LPIPS(y_def, y_orig))` 在 y_def = y_orig 處是最小值，"
        "φ=0 且淨化為 identity 時起點梯度精確為零（實測 grad_norm = 0.000e+00），"
        "該缺陷涵蓋先驗實驗 59 個有紀錄批次的 100%。"
        "見 LOGIC_CHECK A1 與 tests/test_zero_gradient.py"
    ),
}

DEFENSE_MODES = ("targeted_output", "targeted_attn", "suppress_attn_ca")
TARGET_METRICS = ("lpips", "mse")

# `suppress_attn_ca` 的起點：遮罩內參照注意力的 L1 下限。低於此值即拋出。
#
# 存在理由與 `optimize.MIN_SHARED_MASS` 完全同型（`LOGIC_CHECK` A1）：
# 損失是 `‖Att ⊙ M‖₁`，其最小值為 0。若原圖在遮罩內本來就幾乎沒有該詞的
# 注意力反應，損失一開始就坐在最小值上，梯度趨近零，**最佳化不會產生任何
# 更新，而 log 上看起來只是「損失很低」**——那比「損失不動」更難察覺，
# 因為一個接近零的防禦損失看起來像是已經成功。
#
# 取 1e-3 而非 0：式 (3) 的 Att 是 L 層相加後在遮罩內求和，其量級由層數與
# 遮罩面積決定，真實模型上遠大於此（SD v1.4／16 層／512² 的量級為 10⁰–10¹）。
# 這道閘攔的是「結構上沒有東西可壓」，不是一個需要調的旋鈕。
MIN_MASKED_ATTENTION = 1e-3

# 第二、三道 hinge 的門檻相對 τ_lpips 的比例。錨點是使用者在 τ_lpips = 0.05
# 上的人眼判讀（0.04 與 0.8），見模組 docstring「門檻的適用範圍」。
ACUT_PER_TAU = 0.8
CHROMA_PER_TAU = 16.0


def scaled_thresholds(tau_lpips: float) -> Dict[str, float]:
    """依 τ_lpips 等比例給出 `tau_acut` 與 `tau_chroma`。

    回傳的是**具體數值**，由呼叫端明給進 `LossConfig` 並進入 `config_hash`：
    這裡不改任何預設值，也不在損失內部即時換算。門檻若只以「比例」的形式
    存在，`config_hash` 就記不下當次實際用的絕對值，而事後查表要的是後者。
    """
    if not tau_lpips > 0:
        raise ValueError(
            f"tau_lpips 須為正數，收到 {tau_lpips!r}；門檻是它的固定倍數，"
            "非正值會讓兩道 hinge 恆為啟動或恆不啟動而看不出症狀"
        )
    return {"tau_acut": ACUT_PER_TAU * float(tau_lpips),
            "tau_chroma": CHROMA_PER_TAU * float(tau_lpips)}


@dataclass
class LossConfig:
    """全部係數集中於此，避免散落在呼叫端。"""

    lam_def: float = 1.0
    lam_fid: float = 1.0

    # ---- 防禦項 ----
    # "targeted_output" — d(y_def, y_target)，把編輯結果推向固定目標影像
    #                     （N2；N3 階段二改 target_metric="mse"）
    # "targeted_attn"   — 1 − shared token 的注意力質量（N1）
    defense_mode: str = "targeted_output"
    # targeted_output 的度量。N2 取 LPIPS（與 B1／B2 的形式對齊且為感知距離），
    # N3 階段二取 MSE（`DESIGN` §4 的 R_a 定義即 ‖·‖²，不改寫成別的度量）。
    target_metric: str = "lpips"
    # targeted_attn 要把注意力質量導向的 token 位置。
    #
    # CLIP tokenizer 的輸出是 [BOS] tok₁ … tok_n [EOS] [PAD]×…。要當 decoy 的
    # 位置必須同時滿足兩件事：**索引固定**（防禦方不知道攻擊 prompt 的長度 n）
    # 與**不承載語意**。合格的只有兩端：第 0 格的 [BOS]，以及第 76 格的
    # 末位 [PAD]（任何短於 76 token 的 prompt 都會補到那裡）。中間的 EOS 與
    # 前段 PAD 位置隨 n 移動，不可用。
    #
    # **本輪在 SDXL 上取 76 而非 0（`--shared-tokens 76`）。**
    # 2026-08-06 實測 70 層 attn2 的平均 token 質量：
    #
    #   token          空 prompt      "a cat"
    #   0（BOS）        7.2e-06        5.6e-06     ← 等於零
    #   76（末位 PAD）  1.59e-02       4.84e-03
    #
    # BOS 拿不到質量的原因也量到了：它的嵌入 L2 範數是 1193，其餘 token 只有
    # 24–37。那是 CLIP 文字編碼器的 massive-activation 現象——BOS 是個高範數
    # 的暫存槽，不是被 attend 的對象。質量為零時 `1 − mass` 坐在**最大值**上，
    # 梯度為零，最佳化不會產生任何更新，而 log 上看起來只是「損失還沒降」。
    # 這與已移除的 `untargeted` 完全同型（A1）。`optimize.MIN_SHARED_MASS`
    # 因此在第 0 步就檢查並拋出。
    #
    # PromptFlare 的 `L_CA` 對 M^BOS 取是在 SD v1.x 上成立的，換到 SDXL 不成立。
    # 預設值保留 0 以忠於該原形式；本輪的選擇由 CLI 明給並進入 `config_hash`。
    shared_tokens: Tuple[int, ...] = (0,)

    # ---- suppress_attn_ca 專用 ----
    #
    # 防禦方指名要保護的那個詞（Lo et al. 的 c_a）。**沒有預設值**：
    # 由 prompt 猜 c_a 會讓損失壓到別的區域而毫無症狀，而 prompt 是攻擊方
    # 寫的、c_a 是防禦方選的，兩者在威脅模型裡屬於不同的人。
    # 值由 `ImageEntry.content` 逐影像給定（`data/lo_aligned/prompts.yaml`
    # 的 `content` 欄），故此欄在 `LossConfig` 上是逐格填入而非整批共用。
    content: str = ""
    # 式 (4) 的遮罩門檻，作用在**峰值正規化後**的 [0,1] 尺度上
    # （見 `attention_region_mask`：論文未給值，且式 (3) 的值域上界隨 UNet
    # 層數而變，絕對門檻換模型就失去意義）。
    #
    # `None` 表示未指定。`suppress_attn_ca` 下拋出而不是回退到 0.5——
    # 它決定「壓哪一塊」，與 `shared_tokens` 決定「導向哪一格」同一種量，
    # 而後者已於 2026-08-06 因為不在 `config_hash` 內而被點名（A7）。
    attn_mask_tau: Optional[float] = None

    # ---- 保真項：四道 hinge 取交集 ----
    #
    # **L∞ 的 hinge 預設關閉（beta_linf = 0）。**
    #
    # 2026-08-05 修訂。before：`beta_linf = 100.0`。after：`0.0`。
    #
    # 理由是實測：位移場在 LPIPS 0.0593（剛越過 τ=0.05）時 L∞ 已達 0.9386，
    # 其 hinge 懲罰 0.8786 是 LPIPS 懲罰 0.0093 的 **95 倍**。開著它，
    # 綁定約束就不是 LPIPS 而是 L∞——而整個實驗設計的共同貨幣是 τ_LPIPS
    # （`DESIGN` §3.2），射線縮放也是沿 LPIPS 求解。兩者不一致時，
    # 最佳化實際停在 L∞ 的邊界上，報表卻以為它停在 τ_LPIPS 上。
    #
    # 更根本的是 L∞ 對非加性參數化不具鑑別力：把一條邊緣移動一個像素，
    # 其 L∞ 可接近 1.0，人眼卻幾乎看不出來。先驗實測 site S 在 LPIPS 0.1077
    # 時 L∞ 為 0.5439，而加性在相近 LPIPS 下只有 0.030——差 28.2 倍
    # （`PRIOR_FINDINGS` §3.3）。用它當約束等於對非加性單方面加價。
    #
    # 仍保留這個旋鈕而非刪除：五篇 baseline 有四篇原生就是 ℓ∞ 約束的，
    # 重現它們時需要它。**但那條路徑走的是 `src/baselines/pgd.py` 的
    # `project()`，不是這裡的 hinge**——投影是硬約束，hinge 是軟懲罰，
    # 兩者不可互相替代。故本預設值為 0 不影響 baseline 的忠實度。
    #
    # `fid_linf` 仍然逐格記錄，只是不進梯度。報表必須報它：
    # 「同一個 LPIPS 下 L∞ 差 28.2 倍」正是本專案要呈現的事實之一。
    beta_linf: float = 0.0
    tau_linf: float = 0.06     # ≈ 15/255，對抗擾動文獻的常見上限量級

    # 綁定的保真度約束：LPIPS(x_def, x_base)，即「防禦本身造成多少感知
    # 可辨的改變」。τ 由呼叫端依該格的預算給定；預設 0.05 是先驗實驗的
    # 運作點，本輪主表在 0.20、訓練在 0.35，必須覆寫。
    gamma_lpips: float = 100.0
    tau_lpips: float = 0.05

    # 第二道：鈍化。τ_acut = 0.04 是在 τ_lpips = 0.05 上定出的絕對值，
    # 只在 τ_lpips = 0.05 的批次上成立；其餘批次由 `scaled_thresholds` 導出
    # 並由呼叫端明給（2026-08-08，見模組 docstring）。
    gamma_acut: float = 100.0
    tau_acut: float = 0.04

    # 第三道：色度偏壓。τ = 0.8 由人眼判讀（2026-08-01，`runs/p10_chroma_ladder/`）：
    # 「0.3 還有 0.6 都看不出來，1.0 以上才開始有一些細微色調變化」。取 0.8 而非
    # 1.0——1.0 是已確認看得見的那一級，把門檻設在該處等於允許約束放行一個
    # 可見的失真，約束就失去意義。同 `tau_acut`，其餘批次的值由
    # `scaled_thresholds` 導出。
    gamma_chroma: float = 100.0
    tau_chroma: float = 0.8

    # 第四道之外的 PSNR 下限：保留計算與記錄，預設不參與梯度。
    # `align()` 會以 `align_gamma_psnr` 覆蓋它——重建對齊是逐像素準確度確實
    # 重要的場合，防禦階段不是。
    gamma_psnr: float = 0.0
    psnr_floor: float = 34.0   # dB，相對 x_base

    def __post_init__(self):
        if self.defense_mode in RETIRED_MODES:
            raise ValueError(
                f"defense_mode={self.defense_mode!r} 已移除：\n  "
                f"{RETIRED_MODES[self.defense_mode]}\n"
                f"可用的模式為 {list(DEFENSE_MODES)}，全部為 targeted。"
            )
        if self.defense_mode not in DEFENSE_MODES:
            raise ValueError(
                f"未知的 defense_mode: {self.defense_mode!r}，"
                f"只接受 {list(DEFENSE_MODES)}"
            )
        if self.target_metric not in TARGET_METRICS:
            raise ValueError(
                f"未知的 target_metric: {self.target_metric!r}，"
                f"只接受 {list(TARGET_METRICS)}"
            )
        if self.defense_mode == "targeted_attn" and not self.shared_tokens:
            raise ValueError(
                "targeted_attn 需要非空的 shared_tokens。空集合下注意力質量恆為 0，"
                "L_def 退化成常數 1，最佳化不會產生任何更新"
            )
        if self.defense_mode == "suppress_attn_ca":
            if not self.content.strip():
                raise ValueError(
                    "suppress_attn_ca 需要 content（Lo et al. 的 c_a），"
                    "即防禦方指名要保護的那個詞。不接受預設值也不由 prompt "
                    "推導：prompt 是攻擊方寫的、c_a 是防禦方選的，兩者在威脅"
                    "模型裡屬於不同的人，猜錯會讓損失壓到別的區域而毫無症狀。"
                    "來源是 data/lo_aligned/prompts.yaml 的 content 欄"
                    "（ImageEntry.content）"
                )
            if self.attn_mask_tau is None:
                raise ValueError(
                    "suppress_attn_ca 需要 attn_mask_tau（式 4 的遮罩門檻）。"
                    "不回退到 0.5：它決定損失作用在影像的哪一塊，與 "
                    "shared_tokens 同一種量，而後者正因為未進 config_hash "
                    "被列為缺陷 A7。由 CLI 明給並進入 config_hash"
                )
            if not 0.0 < self.attn_mask_tau < 1.0:
                raise ValueError(
                    f"attn_mask_tau 須落在開區間 (0, 1)，收到 "
                    f"{self.attn_mask_tau}；它作用在峰值正規化後的尺度上"
                )


class DefenseObjective:
    """L(φ) 的計算。持有 LPIPS 的可微實作。

    `y_orig` 對 φ 為常數，由呼叫端算好並傳入；本類別不負責快取，以免把
    「哪些量對 φ 為常數」這個關鍵前提藏在實作細節裡。

    ## 保真度一律在 fp32 上量

    **本類別的每一個度量都在 fp32 上計算，不論骨幹跑什麼精度。**
    進來的張量在各入口 `.float()`，出去的純量因此也是 fp32。

    兩個理由，各自獨立成立：

    1. **一致性。** `lpips_rel` 是綁定的保真約束，而段 2 的射線縮放解的是
       **同一個 τ**——後者走 `MetricSuite`，輸入是 fp32 的影像張量
       （`ray_scale.lpips_against`）。兩者若在不同精度上算，訓練期滿足的 τ
       與對齊期解出的 τ 就是兩個不同的量，而「匹配失真」是全案最關鍵的
       前提（`RUNBOOK` §0.5），該前提已被證偽四次。

    2. **混合精度下的輸入本來就不同調。** bf16 時 `x_gen`／`y_def` 是半精度，
       而 `x01`、`y_target`（由 PNG 載入）是 fp32。`piq` 的 LPIPS 與 SSIM
       都會依**第一個引數**建核或轉模型，再拿它算第二個引數，故混著傳一定
       中止。2026-08-06 於段 0 連續遇到兩次（`_perceptual` 與 `piq.ssim`）。

    降精度換記憶體在這裡不划算：保真度是本研究的判準本身，不是中間量。
    新增任何度量時，輸入一律先 `.float()`，由
    `test_保真度的入口一律轉fp32` 釘住。
    """

    def __init__(self, cfg: LossConfig, device: torch.device):
        import piq

        self.cfg = cfg
        self.device = device
        # piq.LPIPS 可微，訓練與評測共用同一實作，避免兩者定義不一致
        self._lpips = piq.LPIPS().to(device)

    def _perceptual(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """LPIPS，**一律在 fp32 上算**。本類別所有 LPIPS 都必須經由此處。

        兩個獨立的理由：

        1. **正確性。** `piq` 的 `ContentLoss.forward` 第一行是
           `self.model.to(x)`——它把 VGG 轉成**第一個引數**的 dtype 與裝置，
           再用同一個模型對第二個引數取特徵。故兩個引數 dtype 不同時直接
           以 `RuntimeError: Input type (float) and bias type (c10::BFloat16)
           should be the same` 中止。本專案在 bf16 下的 `y_def` 是半精度而
           `y_target`（MIST.png）是 fp32，正是這個情形（2026-08-06 於段 0
           的 N2 探測實測）。更糟的是它會**持久改寫** `self.model` 的 dtype，
           故失敗與否還取決於此前哪一次呼叫先到。

        2. **一致性，這一項更重要。** `lpips_rel` 是綁定的保真約束，而段 2
           的射線縮放解的是同一個 τ——後者走 `MetricSuite` 的 LPIPS，輸入是
           fp32 的影像張量（`ray_scale.lpips_against`）。兩者若在不同精度上
           算，訓練期滿足的 τ 與對齊期解出的 τ 就是兩個不同的量，而「匹配
           失真」是全案最關鍵的前提（`RUNBOOK` §0.5，該前提已被證偽四次）。

        `.float()` 對 autograd 是可微的，φ 的梯度照常回傳。

        2026-08-06 新增。before：三個呼叫點各自直接呼叫 `self._lpips`，
        dtype 取決於呼叫端傳進來的張量。GPU 上從未跑過，故未暴露。
        """
        return self._lpips(a.float(), b.float())

    # ---- 距離 d(·,·) ----

    def distance(self, y_a: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        """兩張影像的感知距離。取 LPIPS，因其可微且為感知距離的標準指標。

        本函式固定為 LPIPS，與 `target_metric` 無關：它同時是 `edit_shift`
        的定義，而該量必須跨條件可比，不能隨損失設定而換一種尺度。
        """
        return self._perceptual(y_a.clamp(0, 1), y_b.clamp(0, 1))

    def target_distance(
        self, y: torch.Tensor, y_target: torch.Tensor
    ) -> torch.Tensor:
        """L_def 實際最小化的那個距離，由 `target_metric` 決定。

        `mse` 不對 `y` 取 clamp：`G` 的輸出已經 clamp 過，此處再 clamp 只會
        在飽和處把梯度截斷，而 `DESIGN` §4 的 R_a 是對輸出直接取 ‖·‖²。
        """
        if self.cfg.target_metric == "lpips":
            return self.distance(y, y_target)
        # 與 `_perceptual` 同一個理由：`y` 在 bf16 下是半精度而 `y_target`
        # 是由 PNG 載入的 fp32。見類別 docstring。
        return torch.nn.functional.mse_loss(y.float(), y_target.float())

    # ---- 防禦項（一）：targeted_output ----

    def defense_term(
        self,
        y_def_list: List[torch.Tensor],
        y_orig_list: List[torch.Tensor],
        y_target: torch.Tensor,
    ) -> torch.Tensor:
        """對 (淨化算子, 噪聲) 的取樣求平均，即對 𝒫 與 ε 的期望值估計。

        兩條分支的噪聲必須逐元素相同，否則量到的偏移主要來自噪聲差異。
        此不變量由呼叫端以共用 `noise` 保證（見 `SDWrapper.sdedit` 的介面）。

        `y_orig_list` 只用於長度檢查與 `edit_shift` 的記錄，**不進入梯度**
        ——targeted 的定義是「往哪裡去」，不是「離哪裡遠」。它仍是必要引數：
        兩側逐一配對的長度檢查是「同噪聲配對」這個不變量唯一的程式化把關。
        """
        if len(y_def_list) != len(y_orig_list):
            raise ValueError(
                f"兩側取樣數不符：{len(y_def_list)} vs {len(y_orig_list)}，"
                "每個防禦分支必須對上使用相同噪聲的原圖分支"
            )
        if y_target is None:
            raise ValueError(
                "targeted_output 需要 y_target。缺少時不可退回任何無目標形式："
                "無目標在起點的梯度精確為零（LOGIC_CHECK A1）"
            )
        terms = [self.target_distance(yd, y_target) for yd in y_def_list]
        return torch.stack(terms).mean()

    # ---- 防禦項（二）：targeted_attn ----

    def shared_token_mass(self, maps: List[torch.Tensor]) -> torch.Tensor:
        """shared token 分到的注意力質量，逐層算完再平均。值域 [0, 1]。

        每個 map 的形狀為 (B, Q, T)：Q 是影像 latent 的空間位置數、T 是 77 個
        token 格。softmax 已在 token 維度上取過，故沿 `shared_tokens` 求和
        即為「這些位置聽 shared token 的程度」。

        逐層平均而非加總：層數隨模型而異（SD1.5 的 attn2 有 16 層、SDXL 有
        70 層），取平均後這個量在不同模型之間才可比。

        **一律在 fp32 上歸約**，見類別 docstring「本類別的每一個度量都在
        fp32 上計算」。2026-08-06 修正，before／after：

            before（403–404 行）  terms.append(a.index_select(-1, sel)
                                              .sum(dim=-1).mean())
            after                 …index_select(-1, sel).float().sum(…).mean()

        理由是實測而非原則。bf16 只有 7 位尾數，間距隨指數跳：mass ≈ 0.0156
        落在 2⁻⁶，間距 1.2e-04；而 `attention_term` 要算的 `1 − mass` ≈ 0.984
        落在 2⁻¹，間距 **3.9e-03**——相減本身把解析度砍掉 32 倍。後果是
        [0.0156, 0.0176] 區間內的任何質量都被映到同一個 `l_def = 0.984375`，
        回報的 `shared_mass` 則恆為 `1 − 0.984375 = 0.015625`。

        2026-08-06 於 RTX 3090 的段 0 實測到的症狀：`lr_probe.csv` 中 N1 的
        五個候選學習率（1e-4 至 1e-1，跨 1000 倍）末端損失與監看量**逐位元
        相同**，`_pick_best` 因而在完全無訊號的情況下挑了格點第一個值，
        `_pick_stop_tol` 算出 `stop_tol.shared_mass = 0`；段 1 的 250 步
        log 上 `loss=0.9844` 一格未動，而 `|g|` 是正常的 2e-04 量級。
        亦即梯度一直是對的，被量化掉的是**判準與監看量**。

        這與 `untargeted`（A1）同樣是「跑完了但什麼都沒動」，但成因相反：
        A1 是梯度真的為零，此處梯度非零而**量表讀不出來**。`MIN_SHARED_MASS`
        只在第 0 步檢查起點質量，擋得住前者，擋不住後者。
        """
        if not maps:
            raise ValueError(
                "注意力圖清單為空；CrossAttentionRecorder 未記錄到任何層，"
                "此時 L_def 沒有定義"
            )
        idx = self.cfg.shared_tokens
        terms = []
        for a in maps:
            n_tok = a.shape[-1]
            bad = [i for i in idx if not 0 <= i < n_tok]
            if bad:
                raise IndexError(
                    f"shared_tokens 含超出範圍的位置 {bad}；該層的 token 數為 "
                    f"{n_tok}。位置錯了會靜默地對別的 token 施力"
                )
            if len(set(idx)) >= n_tok:
                raise ValueError(
                    f"shared_tokens 涵蓋了全部 {n_tok} 個 token 格。"
                    "全部 token 的注意力質量和恆為 1，L_def 會退化成常數，"
                    "最佳化不會產生任何更新"
                )
            sel = torch.as_tensor(idx, device=a.device, dtype=torch.long)
            # `.float()` 擺在 `index_select` **之後**：挑格是純搬移、不做算術，
            # 在 bf16 上是精確的，故先挑再轉——fp32 張量只有 (B, Q, |idx|) 而
            # 非 (B, Q, 77)，省 77 倍。N1 不能開 UNet checkpoint，是全批記憶體
            # 最緊的一條路徑（實測峰值 22964 MiB / 23.56 GB），這個順序不是
            # 風格問題。
            terms.append(a.index_select(-1, sel).float().sum(dim=-1).mean())
        return torch.stack(terms).mean()

    def attention_term(
        self, maps_list: Sequence[List[torch.Tensor]]
    ) -> torch.Tensor:
        """`1 − shared token 的質量`，對取樣求平均。越小代表 decoy 越成功。

        起點梯度非零的理由：φ = 0 時 shared token 的質量是一個由影像決定的
        中間值（實測 BOS 分到可觀但遠不及全部的質量），不是極值。這與
        `attention_divergence` 那類「與未防禦參照的散度」不同——後者在 φ=0
        時兩個分佈逐元素相同，散度取到最小值，梯度精確為零，換一種散度並
        不能解決（LEDGER 5.3／7.4）。

        質量有界於 [0, 1]，故本項不需要 hinge：無界的最大化才會發散。
        """
        if not maps_list:
            raise ValueError("targeted_attn 需要至少一組注意力圖")
        return torch.stack(
            [1.0 - self.shared_token_mass(m) for m in maps_list]
        ).mean()

    # ---- 防禦項（三）：suppress_attn_ca（Lo et al. 式 5）----

    def masked_attention_term(
        self,
        maps_list: Sequence[List[torch.Tensor]],
        span: tuple,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """`‖Att(x_adv, c_a) ⊙ M‖₁`，對取樣求平均。越小代表該詞越找不到落點。

        逐取樣先走式 (3) 的跨層聚合再取遮罩內的 L1，**不是**先平均注意力圖
        再聚合：式 (3) 的上採樣與相加是逐次前向內部的運算，把不同 timestep
        的注意力圖先平均會抹掉「哪一個 t 上的綁定被破壞」，而綁定強度隨 t
        變化正是 `attn_timesteps` 要取樣多個 t 的理由。

        `reduce="sum"` 即論文式 (3) 的 Σ。值域上界隨 UNet 層數而變，故本項的
        絕對值**跨模型不可比**（SD v1.5 的 16 層對 SDXL 的 70 層差 4.4 倍）；
        遮罩不受影響，因為式 (4) 以峰值正規化。跨模型引用時要改看
        `attn_suppressed_rel`（見 `__call__`），那是相對起點的比值。

        與 `attention_term`（N1）的差別見模組 docstring 的對照表：那一項對
        全部 query 位置取平均且不需要 c_a，本項只計遮罩內且必須有 c_a。
        """
        from src.models.attention import (
            aggregate_token_attention,
            masked_attention_l1,
        )

        if not maps_list:
            raise ValueError("suppress_attn_ca 需要至少一組注意力圖")
        side = int(mask.shape[-1])
        terms = []
        for maps in maps_list:
            att = aggregate_token_attention(maps, span, side=side,
                                            reduce="sum")
            terms.append(masked_attention_l1(att, mask))
        return torch.stack(terms).mean()

    # ---- VAE 編碼器目標（PhotoGuard 的 encoder attack 形式）----

    def encoder_term(
        self, z_def: torch.Tensor, z_target: torch.Tensor
    ) -> torch.Tensor:
        """‖E_vae(x_def) − z_target‖²，不經過 UNet。

        本輪的 9 個訓練條件都不用它（`DESIGN` §3.1），保留是因為它是 targeted
        且成本結構完全不同：每步只剩一次 VAE 編碼，可以跑 1000 步而非 25 步，
        且目標不依賴任何特定的編輯 prompt 或噪聲取樣。作為機制分析的備用路徑。
        """
        return torch.nn.functional.mse_loss(z_def, z_target)

    # ---- 保真項 ----

    def fidelity_term(
        self,
        x_def: torch.Tensor,
        x: torch.Tensor,
        x_base: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """回傳 (L_fid, 各分項的純量值)。分項一併回傳供逐步診斷。

        `x_base` 是 φ=0 時的輸出，即該 site 在未施加任何防禦時就已經產生的
        圖。像素側為 `x` 本身；生成路徑為 inversion + VAE 來回的重建。
        四道 hinge 全部以 `x_def − x_base` 為對象，對原圖的絕對值另行記錄。
        """
        import piq

        c = self.cfg
        # 三者一律轉 fp32，見類別 docstring「保真度一律在 fp32 上量」。
        # 生成路徑在 bf16 下的 `x_gen` 是半精度而 `x01` 是 fp32，混著進
        # `piq.ssim` 會以 `RuntimeError: expected scalar type Float but found
        # BFloat16` 中止（2026-08-06 於段 0 的 N3 對齊實測）。
        xd = x_def.clamp(0, 1).float()
        xr = x.clamp(0, 1).float()
        xb = xr if x_base is None else x_base.clamp(0, 1).float()

        # 對原圖的絕對值：只記錄，不進梯度。
        lpips = self._perceptual(xd, xr)
        # SSIM 是已排除的度量（LOGIC_CHECK A8），只能出現在回報欄位。
        ssim = piq.ssim(xd, xr, data_range=1.0)

        # L∞ 取「φ 造成的改變」而非「與原圖的總差距」。E0c 實測 VAE 單獨
        # 來回的 L∞ 平均已達 0.707：單一個飽和像素就能讓 L∞ 飽和，該值
        # 幾乎完全由重建誤差決定，與防禦強度無關。若以總差距為對象，
        # τ=0.06 對生成路徑不可達，hinge 恆為啟動，L_fid 被一個 φ 無法改善的
        # 常數主導，防禦項完全失效。
        linf = (xd - xb).abs().max()

        # PSNR 同樣以 x_base 為對象，理由與 L∞ 相同且更強：E0c 實測各影像的
        # 重建誤差下限由 19.61 dB（car_00）到 31.01 dB（person_00），相差
        # 11.4 dB，任何全域固定的 psnr_floor 都不可能同時適用。
        mse = torch.nn.functional.mse_loss(xd, xb)
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        mse_total = torch.nn.functional.mse_loss(xd, xr)
        psnr_total = 10.0 * torch.log10(1.0 / mse_total.clamp_min(1e-12))

        # 綁定的感知約束，對象與 L∞/PSNR 一致為 x_base。像素側的 x_base 即 x，
        # 此時本項與上方的 lpips 相同；多算一次是為了讓兩個 site 的 hinge
        # 定義一致，不因 site 而分支。
        lpips_rel = lpips if x_base is None else self._perceptual(xd, xb)

        # 鈍化：量的是「防禦本身讓影像鈍化了多少」。此量對位移不敏感，
        # 故空間變形只要不模糊就不被罰。
        acut = local_acutance_dev(xb, xd)

        # 色度偏壓：量的是空間上連貫的色偏，而非色度誤差的量值——P9 實測
        # ΔE 那一類分不出加性雜訊與可見色偏（等 LPIPS 下 2.44 對 2.79），
        # 而人眼分得出來。此量對隨機擾動不敏感（區塊內相消）。
        chroma = local_chroma_bias(xb, xd)

        pen_linf = torch.clamp(linf - c.tau_linf, min=0.0)
        pen_lpips = torch.clamp(lpips_rel - c.tau_lpips, min=0.0)
        pen_acut = torch.clamp(acut - c.tau_acut, min=0.0)
        pen_chroma = torch.clamp(chroma - c.tau_chroma, min=0.0)
        pen_psnr = torch.clamp(c.psnr_floor - psnr, min=0.0)

        # 交集：全部項都是 clamp(·, min=0) 乘上非負係數，故沒有任何一項可以
        # 為負，一道 hinge 的懲罰不可能被另一道的改善抵銷。這是「交集」在
        # 程式上的形式，不是註解上的宣稱。
        total = (
            c.beta_linf * pen_linf
            + c.gamma_lpips * pen_lpips
            + c.gamma_acut * pen_acut
            + c.gamma_chroma * pen_chroma
            + c.gamma_psnr * pen_psnr
        )
        parts = {
            # acut 與 chroma 直接參與梯度，故轉純量前顯式 detach；其餘各項由
            # piq 回傳時已不帶圖。這是記錄用的轉換，不影響 total 的計算圖。
            "fid_acut": float(acut.detach()),
            "fid_pen_acut": float(pen_acut.detach()),
            "fid_chroma": float(chroma.detach()),
            "fid_pen_chroma": float(pen_chroma.detach()),
            "fid_linf_total": float((xd - xr).abs().max()),
            "fid_psnr_total": float(psnr_total),
            "fid_lpips": float(lpips),
            "fid_lpips_rel": float(lpips_rel),
            "fid_ssim": float(ssim),
            "fid_linf": float(linf),
            "fid_psnr": float(psnr),
            "fid_pen_linf": float(pen_linf),
            "fid_pen_lpips": float(pen_lpips),
            "fid_pen_psnr": float(pen_psnr),
        }
        return total, parts

    # ---- 總損失 ----

    def __call__(
        self,
        x_def: torch.Tensor,
        x: torch.Tensor,
        *,
        x_base: Optional[torch.Tensor] = None,
        y_def_list: Optional[List[torch.Tensor]] = None,
        y_orig_list: Optional[List[torch.Tensor]] = None,
        y_target: Optional[torch.Tensor] = None,
        attn_maps: Optional[Sequence[List[torch.Tensor]]] = None,
        attn_span: Optional[tuple] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_ref_l1: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """依 `defense_mode` 取用對應的引數。缺少者拋出，不改走另一種模式。

        全部防禦項引數改為 keyword-only：兩種模式需要的東西完全不同
        （一邊是編輯輸出、一邊是注意力圖），位置引數會讓誤傳看起來像是能跑。
        """
        c = self.cfg
        if c.defense_mode == "targeted_output":
            if y_def_list is None or y_orig_list is None:
                raise ValueError(
                    "targeted_output 需要 y_def_list 與 y_orig_list"
                )
            l_def = self.defense_term(y_def_list, y_orig_list, y_target)
            with torch.no_grad():
                # 平均編輯偏移是最直接的進展指標，與 L_def 分開記錄：L_def 量的
                # 是「離目標多近」，這一項量的是「離原編輯多遠」，也是評測所報
                # 的量。兩者不可互相取代。
                shift = torch.stack(
                    [self.distance(yd, yo)
                     for yd, yo in zip(y_def_list, y_orig_list)]
                ).mean()
            extra = {"edit_shift": float(shift)}
        elif c.defense_mode == "suppress_attn_ca":
            if attn_maps is None or attn_span is None or attn_mask is None:
                raise ValueError(
                    "suppress_attn_ca 需要 attn_maps、attn_span 與 attn_mask "
                    "三者。span 由 c_a 的 token 區間決定、mask 由原圖的注意力"
                    "取（式 4），缺任一都不可落回全域——那會變成另一個目標"
                )
            if attn_ref_l1 is None:
                raise ValueError(
                    "suppress_attn_ca 需要 attn_ref_l1（φ=0 時遮罩內的 L1）。"
                    "本模式的 L_def 隨進展**下降**，不能直接當平台停止的監看量；"
                    "監看的是由它導出的 `attn_suppressed`，而那需要起點值"
                )
            l_def = self.masked_attention_term(attn_maps, attn_span, attn_mask)
            # 監看量必須隨進展**上升**（`optimize.DEFENSE_MONITOR` 的規則：
            # 餵一個下降的量進 `plateau_stop` 會在 min_steps 一到就判定收斂，
            # 且沒有症狀）。故記「已壓下多少」而不是損失本身。
            #
            # 同時記相對量：絕對值的尺度隨 UNet 層數與遮罩面積而變，跨模型
            # 與跨影像都不可比，而報表要並列不同影像。
            suppressed = attn_ref_l1 - float(l_def.detach())
            extra = {
                "edit_shift": float("nan"),   # 本路徑沒有 y_orig 可比
                "attn_masked_l1": float(l_def.detach()),
                "attn_ref_l1": float(attn_ref_l1),
                "attn_suppressed": suppressed,
                "attn_suppressed_rel": suppressed / max(abs(attn_ref_l1), 1e-12),
            }
        else:
            if attn_maps is None:
                raise ValueError("targeted_attn 需要 attn_maps")
            l_def = self.attention_term(attn_maps)
            # 本路徑沒有 y_orig 可比，`edit_shift` 在此沒有定義。記為 NaN 而
            # 不是 0：0 是一個合法的偏移值，會讓「沒有定義」與「量到零」在
            # 事後分不出來。`plateau_stop` 對 NaN 監看量直接拋出。
            extra = {
                "edit_shift": float("nan"),
                "shared_mass": float(1.0 - l_def.detach()),
            }

        l_fid, parts = self.fidelity_term(x_def, x, x_base=x_base)
        total = c.lam_def * l_def + c.lam_fid * l_fid

        log = {
            "loss": float(total),
            "L_def": float(l_def),
            "L_fid": float(l_fid),
            "defense_mode": c.defense_mode,
            **extra,
            **parts,
        }
        return total, log
