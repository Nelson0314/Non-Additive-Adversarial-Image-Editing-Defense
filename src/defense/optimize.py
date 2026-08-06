"""逐圖優化 φ — `docs/ARCH_2026-08-05.md` §5、`docs/CODE_2026-08-05.md` §1.1。

每張影像獨立優化一組 φ。防禦者擁有原圖與完整權重（威脅模型），故逐圖優化
符合威脅模型，不需泛化到未見影像。

## 這一版改了什麼

三件事，各自對應 `docs/LOGIC_CHECK_2026-08-05.md` 的一條既知缺陷：

| 缺陷 | 改動 |
|---|---|
| A1 起點零梯度 | 目標函數一律 targeted，見 `objective.py`。本檔配合移除 `attn_mode` 的三個無目標／散度模式 |
| A2 學習率跨位置沿用 | **學習率只能由 `Calibration.get()` 取得**。`OptimConfig` 不再有 `lr` 欄位，沒有預設值、沒有內插、沒有退回 |
| A5 `edit_shift` 不是成功判定 | 它降為 `train.csv` 的監看量（`plateau_stop` 用），不進任何判定 |

另加兩項結構改動：

- **分階段優化**。`OptimConfig.stages` 是一串 `StageSpec`，每個階段指定要更新
  `module.param_groups()` 的哪一組、用哪個校準鍵取學習率、步數上限多少。
  APA 移植（N3）階段一只更新 LoRA、階段二只更新 latent，靠的就是這個。
- **步數上限逐條件由設定給**，不寫死。`StageSpec.max_steps` 是唯一的入口。

## 成本模型

E0 實測（V100 32GB, SD v1-4, 512², fp32, UNet+VAE checkpoint）：

    seconds ≈ 1.05 + 0.384·k_inv + 0.304·n_edit·n_eot
    peak    ≈ 9.95 GB，於 k_inv、n_edit ∈ [5,50] 幾乎不變

記憶體與步數無關是因為兩條 UNet 鏈與三次 VAE 呼叫都已 checkpoint；時間則與
步數線性相關。故 n_eot 直接乘在時間上，是本迴圈最貴的參數。
SDXL 1024² 的係數須由段 0 的微基準重量，上式不得直接沿用。
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.defense.generator import DefenseGenerator
from src.defense.objective import DefenseObjective, LossConfig
from src.models.sd import SDWrapper
from src.purify.ops import Purifier
from src.residual.base import ResidualModule
from src.utils.calibration import Calibration, CalibrationMismatch


@dataclass(frozen=True)
class StageSpec:
    """一個優化階段：更新哪一組參數、學習率的校準鍵、步數上限。

    `group` 是 `module.param_groups()` 的鍵。單階段的條件（N1／N2）用預設的
    `"default"`；APA 移植（N3）用 `"stage1"`（LoRA）與 `"stage2"`（latent）。

    `lr_key` 是 `calibration.json` 的鍵，例 `"lr.N3_stage2"`。**這裡放的是鍵
    不是數值**：把數值放進設定就等於開了一條「未校準也能跑」的路徑，而先驗
    實驗十次事後診斷都是同一句話——一個為某個對象校準的值被沿用到另一個
    對象上，而且沒有症狀（實測位移場與加性的校準值相差 12.5 倍）。

    `max_steps` 是上限不是實際步數：開啟 `stop_on_plateau` 後實際步數由
    `plateau_stop` 決定。逐條件不同（DIA 20 步、Mist 100、PhotoGuard 200、
    N1/N2/N3 上限 250），故必須由設定給。
    """

    group: str
    lr_key: str
    max_steps: int

    def __post_init__(self):
        if self.max_steps <= 0:
            raise ValueError(
                f"階段 {self.group!r} 的 max_steps={self.max_steps} 不是正整數；"
                "步數上限必須由設定明確給定"
            )


# 每個 defense_mode 對應的平台停止監看量。
#
# 監看量必須是「越大代表越有進展」的量：`plateau_stop` 判的是
# `per_step < tol`，餵一個隨進展下降的量進去會在 min_steps 一到就立刻判定
# 收斂，而且沒有任何症狀。
#
# targeted_output 的 L_def 是「離目標多近」，隨進展**下降**，不可直接監看；
# 改監看 `edit_shift`（離原編輯多遠），隨進展上升且已有校準值。
# targeted_attn 監看 `shared_mass`（導向 shared token 的注意力質量），
# 隨進展上升。
DEFENSE_MONITOR = {
    "targeted_output": "edit_shift",
    "targeted_attn": "shared_mass",
}


@dataclass
class OptimConfig:
    # ---- 階段（必填，無預設）----
    # 沒有預設值是刻意的：預設一組階段就等於預設一個學習率校準鍵與一個步數
    # 上限，而這兩者都必須逐條件給定。
    stages: Tuple[StageSpec, ...]

    k_inv: int = 10
    t_max: Optional[int] = None   # inversion 的 timestep 上限，見 E0c
    # True 改用 BDIA 精確反演（arXiv 2307.10829）取代 DDIM。tiny-SD 實測
    # latent 空間來回誤差由 1.41 降到 1.37e-04（k=20, t_max=500）。
    # 影像空間的下限不會歸零：VAE 編解碼來回誤差（真實 SD 實測
    # 27.51 dB / LPIPS 0.143）與反演無關，BDIA 不觸及該項。
    exact_inversion: bool = False
    n_edit: int = 10
    n_eot: int = 1              # 每步的噪聲取樣數
    # 淨化算子的取樣方式：
    #   "rotate" — 每步只用一個算子，逐步輪替
    #   "all"    — 每步對全部算子求梯度後平均，即對 𝒫 的期望值以完整列舉估計
    #
    # 本輪的預設由 "rotate" 改為 "all"。理由不只是估計品質：輪替下 Adam 的
    # 動量在數個目標之間來回拉扯，等於數個互相衝突的梯度訊號輪流覆寫，而非求
    # 其共同下降方向。先前 untargeted 的損失需要靠輪替第 1 步換成 blur 才能
    # 脫離零梯度起點（LOGIC_CHECK A1），改為 targeted 之後這個副作用不再被
    # 需要，也就沒有理由再保留一個較差的期望值估計。成本為每步時間乘以算子數。
    purify_mode: str = "all"

    # ---- 階段一：保真對齊（可選）----
    # 先訓練 φ 使 G(x; φ) 逼近 x，再以該 φ 熱啟動防禦訓練。
    #
    # 借鑑 APA（arXiv 2506.01511）「先立保真、再訓攻擊」的順序。APA 的階段一
    # 是噪聲空間的 ‖ε − ε̂‖²；此處是影像空間的 L_fid(G(x;φ), x)，因為我們的 G
    # 是固定長度的可微 DDIM 鏈，能端對端反傳到輸出影像本身，不需要噪聲空間的
    # 代理量。**這是一個有記錄的偏離**，見 `DESIGN` §4 的移植表。
    #
    # align_steps = 0 表示不執行。
    align_steps: int = 0
    align_lr_key: str = ""      # 非零 align_steps 時必填，無預設學習率
    align_group: str = "default"
    # 階段一專用的 PSNR 係數，覆蓋 LossConfig.gamma_psnr（防禦階段為 0.0）。
    # 重建對齊是逐像素準確度確實重要的場合；防禦階段不是。
    align_gamma_psnr: float = 1.0

    strength: float = 0.5
    # site S 專用：位移場的硬上界，單位為像素。空間變形的保真度預算是位移量
    # 而非 L∞（把一條邊緣移動一像素，L∞ 可接近 1.0 卻幾乎看不出來），
    # 故此值必須與 tau_lpips 併列記錄。
    warp_max_disp: float = 1.5
    # site S 的 grid_sample 插值模式。E20 §5.2 量出 bicubic 可把銳利度保留率
    # 由 85.0% 拉到 99.9%。
    warp_resample: str = "bicubic"
    # 攻擊方的 classifier-free guidance 權重。E26 實測 w=1 時 SD 幾乎不服從
    # prompt，該設定下的「編輯」與「什麼都沒做」在指標上分不出來
    # （Δclip +0.0116，而對照的噪聲範圍是 ±0.0169）。`DESIGN` §2 定為 7.5。
    #
    # 訓練期一律餵空 prompt（prompt-free，`DESIGN` §2.1），但**條件分支與
    # 無條件分支不是同一個張量**：後者由 `sd.uncond_prompt()` 依模型自己的
    # 規則產生。stock SDXL base 的 `force_zeros_for_empty_prompt = true`，
    # 其無條件分支是零張量而非空 prompt 的編碼。
    #
    # 2026-08-05 修訂。before：此處註明「兩者逐元素相同，CFG 在數值上是恆等的」。
    # 那個說法在 SD v1.x 上為真、在 SDXL 上為假，而且即使為真也不該那樣實作：
    # 兩支相同時 `ε_u + w·(ε_c − ε_u)` 的括號恆為零，guidance_scale 設多少
    # 都等同 w=1——那正是 `LOGIC_CHECK` A6 列為硬規則要避免的失效模式。
    #
    # 評測期的 guidance 由評測程式另行指定，與此欄位無關。
    guidance_scale: float = 7.5

    # ---- 停止準則 ----
    stop_on_plateau: bool = True
    stop_patience: int = 20     # 觀察窗長度，切成前後兩半比較改善量
    # 每步的絕對改善量門檻，低於此值視為進展已停。
    # `None` 表示依監看量取 MONITOR_TOL 的值；給定數值則覆寫。
    stop_tol: Optional[float] = None
    stop_min_steps: int = 25    # 下限，讓約束有機會啟動
    # 不在約束仍被違反時停止。本輪的比較是匹配失真的比較，一律開啟。
    stop_require_feasible: bool = True

    # ---- cross-attention 目標（N1）專用 ----
    # 每步取樣幾個 timestep。cross-attention 的綁定強度隨 t 變化，只在單一 t
    # 上施力等於只防住編輯鏈的其中一步。取樣點均分於 [0, t_edit]。
    attn_timesteps: int = 4

    prompt_def: str = ""        # 防禦生成的 prompt
    seed: int = 20260728
    unet_ckpt: bool = True      # E0：512² 下關閉必 OOM
    vae_ckpt: bool = True       # E0b：三次 VAE 呼叫的激活由總和降為最大值
    log_every: int = 10
    grad_clip: float = 1.0

    def __post_init__(self):
        if not self.stages:
            raise ValueError(
                "OptimConfig.stages 為空；至少要有一個階段。"
                "每個階段必須指定 group、lr_key 與 max_steps"
            )
        if self.align_steps > 0 and not self.align_lr_key:
            raise ValueError(
                "align_steps > 0 但沒有給 align_lr_key。"
                "階段一的學習率同樣只能由校準表取得，沒有預設值"
            )


# 每個監看量各自的停止門檻。**不可共用一個值。**
#
# `plateau_stop` 比較的是「每步的絕對改善量」，而不同監看量的動態範圍差很多，
# 同一個絕對門檻在兩者上是兩件不同的事。2026-08-04 實測（同一張影像、
# 同一個 site PF、60 步）：
#
#   edit_shift  0.0000 → 0.3665   總改善 0.3665   平均每步 7.8e-03
#   attn_div    0.9847 → 0.9942   總改善 0.0095   平均每步 1.6e-04
#
# **相差 39 倍。** 把 `edit_shift` 的 1e-4 套到 `attn_div` 上，門檻只比平均
# 改善率低 1.6 倍而不是一個量級，實測會在第 30 步（共 60 步）就判定收斂，
# 砍掉 53% 的改善，而且沒有任何症狀——它會回報「已收斂」。
#
# 取值規則：比實測的平均每步改善低一個量級。
#
# 下表是**先驗實驗的紀錄，不是本輪可用的值**，故名為 LEGACY。
# 兩個值都量在 SD v1.4／512²／site PF 上；本輪是 SDXL／1024²／位移場，
# LPIPS 的動態範圍與注意力層數都不同，沿用它們就是 A2 的形狀
# （「一個為某個對象校準的值被沿用到另一個對象上，而且沒有症狀」）。
#
# 2026-08-05 修正。before：`MONITOR_TOL` 同時是紀錄與**回退值**，
# `resolve_stop_tol(None, "edit_shift")` 會靜默回傳 1e-4；`shared_mass`
# 因不在表內而拋出。即同一道規則對兩個監看量鬆緊不一。
# after：全部監看量一律向校準表索取，本表只作為量級參考留存。
# 原因見 `docs/INDEX.md` §3「停止門檻沿用舊模型的校準值且無 context 檢查」。
# N1 起點的 shared token 質量下限。低於此值即拋出，不讓最佳化空跑。
#
# 這道防護的存在理由是 `LOGIC_CHECK` A1：已移除的 `untargeted` 模式在起點的
# 梯度精確為零，而該缺陷涵蓋先驗實驗 59 個有紀錄批次的 100%——因為「損失
# 不動」在 log 上看起來就只是「還沒開始降」，沒有人會在第 10 步就停下來查。
#
# 取 1e-3 的依據是實測（2026-08-06，SDXL、70 層 attn2、1024²）：
#   token 0（BOS）    質量 7.2e-06（空 prompt）／5.6e-06（"a cat"）  ← 等於零
#   token 76（末位 PAD）質量 1.59e-02 ／ 4.84e-03                    ← 可用
# 兩者相差三個數量級，1e-3 落在中間且遠離兩端，不是一個需要調的旋鈕。
#
# BOS 拿不到質量的原因也量到了：它的嵌入 L2 範數為 1193，其餘 token 只有
# 24–37。那是 CLIP 文字編碼器的 massive-activation 現象，BOS 是高範數的
# 暫存槽而非被 attend 的對象。PromptFlare 的 `L_CA` 對 BOS 施力是在
# SD v1.x 上成立的，換到 SDXL 不成立。
MIN_SHARED_MASS = 1e-3

LEGACY_MONITOR_TOL = {
    "edit_shift": 1e-4,     # E23 實測平均 5.4e-4（SD v1.4／512²／site PF）
    "attn_div": 1e-5,       # 2026-08-04 實測平均 1.6e-4（同上）
}


def resolve_stop_tol(cfg_tol: Optional[float], monitor_key: str) -> float:
    """回傳這次要用的 `stop_tol`。**只有呼叫端明給這一個入口。**

    沒有任何回退路徑，與 `resolve_lr` 同構：門檻與學習率是同一種量——
    為某個模型／解析度／參數化量出來的值，換了對象就不再成立，而沿用它
    不會有症狀（門檻過嚴會回報「已收斂」，過鬆等於沒開停止準則）。

    段 0 的 `calibrate_lr` 會為每個條件的監看量量出 `stop_tol.<監看量>`
    並寫進 `calibration.json`；`executors.optim_config` 由該表取得後傳入。
    """
    if cfg_tol is not None:
        return cfg_tol
    raise KeyError(
        f"監看量 {monitor_key!r} 的 stop_tol 未給。它只能由段 0 的校準表取得"
        f"（鍵名 stop_tol.{monitor_key}），沒有預設值也沒有回退。"
        f"先驗實驗的量級參考為 {LEGACY_MONITOR_TOL}，但那是 SD v1.4／512² "
        f"上的值，不可沿用到本輪的 SDXL／1024²"
    )


def resolve_lr(
    calib: Optional[Calibration], lr_key: str, context: Dict[str, Any]
) -> float:
    """逐階段的學習率。**只有這一個入口，且沒有任何回退路徑。**

    `Calibration.get()` 在鍵未校準、或 context 與表中記錄的 `calibrated_for`
    不完全相等時拋出 `CalibrationMismatch`。本函式不接住它、不代入預設值、
    不由鄰近的鍵內插、不因為「差一個欄位」而放行。

    `calib` 為 None 同樣拋出而不是走一條「沒有校準表就用預設」的路：那條路
    存在的話，忘記傳校準表的症狀就會是「跑完了，數字看起來正常」。

    參數量綱逐注入位置不同（位移像素 vs 像素值 vs latent 值），Adam 每步位移
    約等於 lr，同一個數值在三個位置代表三種步長。實測位移場與加性像素殘差的
    校準值相差 **12.5 倍**。
    """
    if calib is None:
        raise CalibrationMismatch(
            f"要取學習率 {lr_key!r} 但沒有提供校準表。"
            "不會回退到預設值——未校準的值沿用正是本專案重複十次的缺陷。"
            "請先執行段 0 產生 calibration.json 並傳入。"
        )
    value = calib.get(lr_key, context)
    lr = float(value)
    if not lr > 0.0:
        raise CalibrationMismatch(
            f"校準表中 {lr_key!r} 的值為 {value!r}，不是正數。"
            "學習率非正代表該鍵記錄的不是學習率，或段 0 的校準失敗"
        )
    return lr


def stage_parameters(
    module: ResidualModule, group: str
) -> List[nn.Parameter]:
    """取出該階段要更新的參數。組名不存在直接拋出，不退回全部參數。

    退回 `module.parameters()` 會讓「階段設定寫錯」變成「兩個階段都更新了
    全部參數」，而輸出看起來完全正常——APA 的階段二本來就該凍結 LoRA，
    沒凍結的話量到的是兩組參數的聯合效果，不是論文的兩階段結構。
    """
    groups = module.param_groups()
    if group not in groups:
        raise KeyError(
            f"模塊沒有參數組 {group!r}。可用的組：{sorted(groups)}。"
            "不會退回 module.parameters()——組名寫錯時全部參數一起更新，"
            "與設定的兩階段結構是兩件不同的事，而且沒有症狀"
        )
    params = list(groups[group])
    if not params:
        raise ValueError(f"參數組 {group!r} 是空的，該階段沒有東西可以優化")
    return params


@contextmanager
def only_trainable(module: nn.Module, params: Sequence[nn.Parameter]):
    """在區塊內只讓 `params` 可訓練，其餘參數 `requires_grad=False` 且梯度清空。

    只把 `params` 交給優化器是不夠的：其他組的參數仍會在 backward 時累積
    `.grad`，下一階段建立優化器時那些殘留的梯度會在第一次 `step()` 前就存在，
    使「上一階段有沒有污染這一階段」無法從外部判斷。此處直接讓它們收不到梯度
    ——區塊結束後 `.grad` 仍為 `None`，這是可斷言的性質。
    """
    keep = {id(p) for p in params}
    for p in params:
        if not p.requires_grad:
            raise ValueError(
                "本階段的參數 requires_grad=False，優化器會靜默地什麼都不更新"
            )
    saved = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in saved:
        if id(p) not in keep:
            p.requires_grad_(False)
            p.grad = None
    try:
        yield
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)


def active_constraint_keys(loss_cfg: LossConfig) -> tuple:
    """回傳「係數非零、因而真的會綁住優化」的 hinge 對應的記錄鍵。

    不能寫死成 LPIPS 與鈍化那兩道。`fidelity_term` 一律計算並記錄全部四道
    hinge 的懲罰值，但係數為零的那幾道不進梯度、不構成約束。反過來，係數非零
    的任何一道都可能是實際綁住這一格的那道——tiny-SD 的端到端測試即出現
    `pen_lpips = pen_acut = 0` 而 `pen_linf = 0.098`（×100 = 9.8，佔 L_fid 的
    全部）的情形，與 E13 量到的「L∞ hinge 把 site S 完全節流」同一回事。

    若只認那兩道，被 L∞ 綁住的格子會永遠判不到「約束已啟動」而跑滿上限。
    """
    pairs = (
        (loss_cfg.gamma_lpips, "fid_pen_lpips"),
        (loss_cfg.gamma_acut, "fid_pen_acut"),
        (loss_cfg.gamma_chroma, "fid_pen_chroma"),
        (loss_cfg.beta_linf, "fid_pen_linf"),
        (loss_cfg.gamma_psnr, "fid_pen_psnr"),
    )
    keys = tuple(k for coef, k in pairs if coef != 0.0)
    if not keys:
        raise ValueError(
            "所有保真 hinge 的係數都是零，沒有任何約束會綁住優化。"
            "此時「約束啟動並穩定」的停止準則沒有定義，"
            "請至少開啟一道 hinge，或關閉 stop_on_plateau"
        )
    return keys


def plateau_stop(
    history: List[Dict],
    patience: int,
    tol: float,
    min_steps: int,
    require_constraint: bool = True,
    constraint_keys: tuple = ("fid_pen_lpips", "fid_pen_acut"),
    monitor_key: str = "edit_shift",
    require_feasible: bool = False,
    feas_tol: float = 1e-6,
) -> tuple:
    """該不該停？回傳 (要不要停, 原因)。

    固定步數是錯的協議。E21–E23 §5.4 量出的問題：兩個 site 的 φ 量綱不同
    （site S 是位移像素、site P 是像素值），學習率也不同，故「同樣跑 N 步」
    對兩者從來不是同一件事。實測後果是每一格被不同的東西綁住——τ=0.10 沒有
    任何一格碰到失真預算、末端 6/6 仍在上升，那一格量到的是「25 步走到哪裡」
    而不是「該方法在此預算下的能力」；而 τ=0.05 由 25 步改到 100 步後，
    site S 對 site P 的比值由 1.14× 反轉為 0.85×。

    匹配失真的比較要求每一格都被同一道約束綁住且已在該約束下收斂。
    故停止準則有兩個條件，缺一不可：

    1. 約束確實啟動過（觀察窗內某一步的懲罰大於零）。沒有這一條，一格可能
       因為還沒碰到預算就「收斂」，那是步數不足不是收斂。
    2. 進展已停（觀察窗後半的監看量相對前半的改善低於 `tol`）。

    監看量必須是「越大代表越有進展」的量，且**不可是 hinge 過的 L_def**：
    後者飽和後恆為 0，看不出偏移還在不在增加。targeted 之後 L_def 是「離目標
    多近」，隨進展下降，直接監看它會在 min_steps 一到就判定收斂——故
    `DEFENSE_MONITOR` 逐模式指定監看量，不由呼叫端隨手給。

    `tol` 是每步的絕對改善量，不是相對改善率。初版寫的是相對量
    `(b − a) / max(|a|, 1e-8)`，在監看量接近零時分母趨近零、判定被噪聲主導
    ——tiny-SD 的端到端煙霧測試中 shift 在 1e-4 量級來回抖動，相對改善動輒
    ±0.16，40 步跑滿都不會停。改用絕對量的理由不只是數值穩定：`edit_shift`
    是 LPIPS 距離，其尺度跨 site 相同（那正是選它當監看量的原因），故絕對
    門檻在兩個 site 上意義一致，而相對門檻會讓 shift=0.02 與 shift=0.2 的
    「收斂」定義不同。

    預設 `tol = 1e-4` 由 E23 的實測定出：site P 由 25 步的 net 0.0784 走到
    100 步的 0.1186，平均每步 5.4e-4，且末端 6/6 仍在上升——該格必須被判為
    「還在上升」。1e-4 比它低一個量級，留有餘裕。

    抽成純函數是為了能在沒有 SD 的情況下驗證取樣邏輯——這段決定了每一格
    跑多久，是本協議的核心，不該只能靠跑完整實驗才看得出對錯。

    2026-08-02（E31）新增 `monitor_key`。

        修訂前（本檔 line 214-215）
            a = sum(h["edit_shift"] for h in window[:half]) / half
            b = sum(h["edit_shift"] for h in window[half:]) / (patience - half)

        修訂後
            monitor_key: str = "edit_shift"      ← 新參數，預設維持既有行為
            a = sum(h[monitor_key] for h in window[:half]) / half
            b = sum(h[monitor_key] for h in window[half:]) / (patience - half)

    理由：cross-attention 路徑的 history 把 `edit_shift` 記為 `float("nan")`
    ——它沒有 y_orig 可比，那個欄位在該路徑上沒有定義。NaN 的一切比較恆為
    False，直接沿用會讓停止準則**永遠不觸發**，症狀是「加了停止準則但沒有
    作用」而非報錯。

    缺鍵與 NaN 都改為拋出而非略過：這兩種情形下「不停」與「該停但沒停」
    在外部完全分不出來。
    """
    if patience < 2:
        raise ValueError(
            f"patience={patience} 無法切成前後兩半來比較改善量；至少要 2"
        )
    if len(history) < max(min_steps, patience):
        return (False, "")

    window = history[-patience:]
    if require_constraint:
        engaged = any(
            any(h.get(k, 0.0) > 0.0 for k in constraint_keys) for h in window
        )
        if not engaged:
            return (False, "")
    if require_feasible:
        # 「啟動過」不等於「已滿足」。實測（2026-08-03，site PF）：第 31 步
        # 時 LPIPS 罰項仍有 34（即失真約 0.44，預算 0.10 的 4.4 倍），而
        # `edit_shift` 正在**下降**——最佳化正在把失真拉回預算內，那正是
        # 該發生的事，卻被判成「沒進展」而中止，該格因此停在 3.3 倍預算上。
        latest = history[-1]
        violated = [k for k in constraint_keys if latest.get(k, 0.0) > feas_tol]
        if violated:
            return (False, "")

    for h in window:
        if monitor_key not in h:
            raise KeyError(
                f"history 沒有監看鍵 {monitor_key!r}。停止準則的監看量必須"
                f"存在——缺席時靜默不停的症狀是「加了停止準則但沒有作用」，"
                f"在外部與「該停但沒停」分不出來"
            )
        v = h[monitor_key]
        if v != v:      # NaN
            raise ValueError(
                f"監看鍵 {monitor_key!r} 的值是 NaN。NaN 的一切比較恆為 "
                f"False，用它當監看量會永遠不停。cross-attention 路徑的 "
                f"edit_shift 沒有定義（無 y_orig 可比），該路徑須改用 "
                f"monitor_key='shared_mass'"
            )

    half = patience // 2
    a = sum(h[monitor_key] for h in window[:half]) / half
    b = sum(h[monitor_key] for h in window[half:]) / (patience - half)
    # 兩半的中心相距 patience/2 步，故每步改善量為總差除以該距離
    per_step = (b - a) / (patience / 2.0)
    if per_step < tol:
        return (True, f"約束已啟動且 {monitor_key} 在最近 {patience} 步的每步"
                      f"改善 {per_step:+.2e} 低於 {tol:.1e}")
    return (False, "")


def eot_pairs(mode: str, step: int, n_eot: int, n_purifiers: int) -> List[tuple]:
    """該步要評估的 (淨化算子索引, 噪聲索引) 清單。

    抽成純函數是為了能在沒有 SD 模型的情況下驗證取樣邏輯——這段決定了
    梯度訊號的構成，是對 𝒫 求期望值的實作，不該只能靠跑完整實驗才看得出對錯。
    """
    if mode == "all":
        return [(pi, i) for pi in range(n_purifiers) for i in range(n_eot)]
    if mode == "rotate":
        return [((step * n_eot + i) % n_purifiers, i) for i in range(n_eot)]
    raise ValueError(f"未知的 purify_mode: {mode!r}，只接受 'rotate' 或 'all'")


@dataclass
class OptimResult:
    history: List[Dict] = field(default_factory=list)
    x_def: Optional[torch.Tensor] = None
    # 防禦訓練所用的保真基準。未執行對齊時為 G(x; φ=0)；執行後為
    # G(x; φ_align)，即對齊結束時該位置實際能做到的重建。
    x_base: Optional[torch.Tensor] = None
    x_base0: Optional[torch.Tensor] = None  # 恆為 G(x; φ=0)，作為對照保留
    x0_trace: List[torch.Tensor] = field(default_factory=list)
    align_history: List[Dict] = field(default_factory=list)
    align_seconds: float = 0.0
    seconds: float = 0.0
    steps_done: int = 0
    # 逐階段的收尾狀況：{group, lr, steps, stop_reason}。多階段時
    # 單一個 `stop_reason` 不足以說明哪一階段用盡了上限。
    stage_reports: List[Dict] = field(default_factory=list)
    # 最後一個階段的停止原因。空字串表示跑滿上限（即上限用盡而非收斂），
    # 該格不可用於跨條件比較——那正是 E21–E23 §5.4 的問題。
    stop_reason: str = ""


def run_stages(
    module: ResidualModule,
    cfg: OptimConfig,
    calib: Optional[Calibration],
    calib_context: Dict[str, Any],
    step_fn: Callable[[StageSpec, int, int], Tuple[torch.Tensor, Dict]],
    constraint_keys: tuple,
    monitor_key: str,
    result: Optional[OptimResult] = None,
    tag: str = "",
) -> OptimResult:
    """分階段的優化驅動迴圈。**不碰 SD**，故可在 CPU 上單獨驗證。

    `step_fn(stage, stage_step, global_step) -> (loss, log)` 負責一次前向並
    回傳純量損失與該步要記錄的欄位；backward、梯度裁剪、優化器步進、記錄與
    停止判定都在此處。把 SD 的部分收在 `step_fn` 裡，使「哪些參數在這一階段
    被更新、學習率從哪裡來、跑幾步、什麼時候停」這四件事可以在沒有模型權重的
    情況下被測試——它們正是先驗實驗出過缺陷的地方。

    每個階段：

    1. 由 `param_groups()` 取出該組參數，其餘參數在本階段內 `requires_grad=False`。
    2. 學習率由 `resolve_lr` 向校準表索取，未校準即拋出。
    3. 各自建立新的 Adam。**不沿用上一階段的優化器狀態**：兩組參數的量綱不同，
       動量與二階矩沿用等於把上一組的步長帶進這一組。
    4. `plateau_stop` 只看**本階段**的歷史切片。跨階段共用觀察窗會讓階段二的
       前 20 步被階段一的尾巴決定，而那兩段是不同的目標在不同的參數上。
    """
    result = result or OptimResult()
    for stage in cfg.stages:
        params = stage_parameters(module, stage.group)
        lr = resolve_lr(calib, stage.lr_key, calib_context)
        start = len(result.history)
        reason = ""

        with only_trainable(module, params):
            opt = torch.optim.Adam(params, lr=lr)
            print(f"  [{tag}stage {stage.group}] lr={lr:g}（{stage.lr_key}）"
                  f"上限 {stage.max_steps} 步", flush=True)

            for s in range(stage.max_steps):
                opt.zero_grad(set_to_none=True)
                loss, log = step_fn(stage, s, len(result.history))
                loss.backward()

                grads = [p.grad for p in params if p.grad is not None]
                if not grads:
                    raise RuntimeError(
                        f"階段 {stage.group!r} 的參數在 backward 之後沒有任何"
                        "梯度。φ 沒有進入計算圖，優化器會靜默地什麼都不更新"
                    )
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()

                log = dict(log)
                log["step"] = len(result.history)
                log["stage"] = stage.group
                log["stage_step"] = s
                log["lr"] = lr
                log["grad_norm"] = float(
                    torch.stack([g.norm() for g in grads]).norm()
                )
                result.history.append(log)

                if s % cfg.log_every == 0 or s == stage.max_steps - 1:
                    print(
                        f"  [{tag}{stage.group}] step {s:>4d}  "
                        f"loss={log['loss']:.4f}  L_def={log['L_def']:.4f}  "
                        f"L_fid={log['L_fid']:.4f}  "
                        f"|g|={log['grad_norm']:.3e}",
                        flush=True,
                    )

                del loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if cfg.stop_on_plateau:
                    stop, reason = plateau_stop(
                        result.history[start:], cfg.stop_patience,
                        resolve_stop_tol(cfg.stop_tol, monitor_key),
                        cfg.stop_min_steps,
                        constraint_keys=constraint_keys,
                        monitor_key=monitor_key,
                        require_feasible=cfg.stop_require_feasible,
                    )
                    if stop:
                        print(f"  [stop] 階段 {stage.group} 第 {s} 步停止："
                              f"{reason}", flush=True)
                        break

        if cfg.stop_on_plateau and not reason:
            # 跑滿上限而非收斂。這一格不可用於跨條件比較，理由與 E21–E23 §5.4
            # 相同：量到的是「走到哪裡」不是「能力」。
            print(f"  [stop] 階段 {stage.group} 用盡上限 {stage.max_steps} 步"
                  f"仍未達停止準則，該格不可用於跨條件比較", flush=True)
        result.stage_reports.append({
            "group": stage.group,
            "lr_key": stage.lr_key,
            "lr": lr,
            "max_steps": stage.max_steps,
            "steps": len(result.history) - start,
            "stop_reason": reason,
        })
        result.stop_reason = reason

    result.steps_done = len(result.history)
    return result


class Diverged(RuntimeError):
    """φ 發散到非有限值。**是結果，不是程式錯誤。**

    段 0 的 lr 格點刻意跨數個量級，故一定會有候選發散；`_pick_best` 依
    `finite` 欄剔除它們。發散本身要被辨識出來並如實記錄，而不是讓 NaN
    一路流進指標套件——`fidelity_term` 的 `clamp(0, 1)` 不消除 NaN，
    `piq.ssim` 會以 `AssertionError: Expected values to be greater or equal
    to 0, got nan` 中止，而那個訊息看不出真正原因，且會讓整個段 0 陪葬
    （2026-08-06 實測於 SD v1.4／512²，`lr.N3_stage1 = 0.1`）。
    """


def raise_if_diverged(x: torch.Tensor, step: int) -> None:
    if not torch.isfinite(x).all():
        raise Diverged(
            f"第 {step} 步：生成結果出現非有限值（NaN 或 inf）。"
            "這個學習率下 φ 已發散，該候選不可用"
        )


@torch.no_grad()
def recon_floor_thresholds(x_floor: torch.Tensor, x01: torch.Tensor,
                           perceptual) -> Dict[str, float]:
    """階段一四道 hinge 的門檻，逐影像取自該影像自己的重建下限。

    判準是「**不可以比 VAE 自己造成的更差**」：門檻取 `decode(encode(x))`
    對原圖的實測值，任何劣於它的都受罰。

    **門檻取純 VAE 來回而不是 `G(x; φ=0)`**，兩者差別是後者多了 inversion
    與去噪。這個選擇是必要的：`LowRankResidual` 的初始化是 `U ~ N(0,σ)`、
    `V = 0`，故初始 Δ 恆為零（`lowrank.py` 檔頭），即
    `G(x; φ_init)` 與 `G(x; φ=0)` **逐位元相同**。門檻若取後者，第 0 步的
    四道 hinge 全部恰好落在門檻上、損失為零、梯度為零，階段一成為空操作。
    取純 VAE 來回則留下 inversion 與去噪造成的那一段作為可改善的餘裕，而
    那正是 LoRA 唯一能作用的地方（它改 UNet，不改 VAE）。

    2026-08-06 改動（使用者裁決）。before：`psnr_floor` 固定 34.0 dB、
    `tau_acut` 0.04、`tau_chroma` 0.8、`tau_lpips` 取 τ_train，四者皆為
    與影像無關的常數。實測 SDXL/1024² 下純 VAE 來回的 PSNR 為 36.71
    （bird_03）／29.68（cat_02）／33.68（dog_03） dB，即 34 dB 這道門檻對
    後兩張**永遠不可達**，200 步全在被一個常數推，反而把 LPIPS 推壞
    （cat_02 +0.054，佔其總預算四成）。`objective.py` 的
    「E0c 實測各影像的重建誤差下限由 19.61 到 31.01 dB，相差 11.4 dB，
    任何全域固定的 psnr_floor 都不可能同時適用」講的正是這件事，但該處的
    處置（改對 x_base 量）只用在防禦階段；階段一量的是**絕對**保真度，
    不能改對象，只能改門檻。

    與那個失效模式的關鍵差別：這裡的門檻**存在一個 φ 能達成**（即 φ=0），
    而 34 dB 沒有任何 φ 能達成。階段一要做的就是用 LoRA 把 latent 注入
    （`latent_init_std`）造成的劣化補回到這條線上。

    `tau_linf` 不在此列：`fidelity_term` 的 L∞ 一律以 `x_base` 為對象，
    階段一的 `x_base=None` 使其等於原圖，而 `beta_linf` 預設為 0。

    `perceptual` 取 `DefenseObjective._perceptual`，不自建第二個 `piq.LPIPS`
    ——門檻與被它約束的量必須由同一個實作算出，否則門檻本身就有偏差。
    """
    from src.defense.objective import local_acutance_dev, local_chroma_bias

    xb = x_floor.clamp(0, 1).float()
    xr = x01.clamp(0, 1).float()
    mse = torch.nn.functional.mse_loss(xb, xr)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    return {
        "psnr_floor": float(psnr),
        "tau_acut": float(local_acutance_dev(xr, xb)),
        "tau_chroma": float(local_chroma_bias(xr, xb)),
        "tau_lpips": float(perceptual(xb, xr)),
    }


def align(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    gen: DefenseGenerator,
    calib: Optional[Calibration],
    calib_context: Dict[str, Any],
    x_base0: torch.Tensor,
) -> Tuple[torch.Tensor, List[Dict]]:
    """保真對齊：訓練 φ 使 G(x; φ) 逼近 x。回傳 (x_align, history)。

    損失以 `fidelity_term(G(x;φ), x, x_base=None)` 為基礎，即對原圖的絕對
    保真度。`x_base=None` 使四道 hinge 的對象為原圖本身；hinge 在容差內不施力
    是此處要的行為——對齊到 τ 以內即停止，不必把重建誤差壓到零。

    但係數與防禦階段不同：`gamma_psnr` 由 `cfg.align_gamma_psnr` 覆蓋。
    防禦階段把 PSNR 移出梯度是對的（逐像素平方誤差與人眼可辨性關聯薄弱），
    但重建對齊正是逐像素準確度確實重要的場合，同一組係數不該同時適用兩者。

    這是 E9 直接量到的問題：200 步對齊後 car_00 的 LPIPS 由 0.2032 降到
    0.0950，PSNR 卻只由 19.62 動到 20.54；car_01 的 PSNR 甚至掉了 1.84 dB
    （22.28 → 20.44）。PSNR 當時完全不在損失裡，那些數字反映的是自由漂移。

    這個階段可能失敗，而失敗本身是結果：低秩注入未必有足夠容量吸收 VAE 與
    DDIM 的重建誤差。E9 實測 car_01 在第 50 步即停在 LPIPS 0.166，其後 150 步
    無改善——那是容量天花板，不是步數不足。回傳的 history 記錄逐步的 LPIPS 與
    PSNR，呼叫端據此判斷，不得假設對齊必然成功。

    `align_group` 使本階段可以只更新其中一組參數。APA 移植的階段一即
    `align_group="stage1"`（只訓練 LoRA），與階段二的 latent 完全分開。
    """
    # 只覆蓋 gamma_psnr：保真度的「定義」不變，變的是逐像素項在這個階段
    # 要不要參與梯度。四道 hinge 的門檻另由該影像自己的重建下限決定。
    obj = DefenseObjective(replace(loss_cfg, gamma_psnr=cfg.align_gamma_psnr),
                           x01.device)
    with torch.no_grad():
        x_floor = sd.decode_latent(sd.encode_image(x01)).detach()
    floors = recon_floor_thresholds(x_floor, x01, obj._perceptual)
    obj.cfg = replace(obj.cfg, **floors)

    params = stage_parameters(module, cfg.align_group)
    lr = resolve_lr(calib, cfg.align_lr_key, calib_context)
    history: List[Dict] = []
    best_loss, best_step, best_state = float("inf"), -1, None

    with only_trainable(module, params):
        opt = torch.optim.Adam(params, lr=lr)
        print(f"  [align] group={cfg.align_group} lr={lr:g}"
              f"（{cfg.align_lr_key}）{cfg.align_steps} 步", flush=True)
        print(f"  [align] 門檻取自本圖的重建下限："
              f"psnr≥{floors['psnr_floor']:.2f}dB "
              f"lpips≤{floors['tau_lpips']:.4f} "
              f"acut≤{floors['tau_acut']:.4f} "
              f"chroma≤{floors['tau_chroma']:.4f}", flush=True)

        for step in range(cfg.align_steps):
            opt.zero_grad(set_to_none=True)
            ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
            x_gen = gen.generate(x01, ctx, use_ckpt=cfg.unet_ckpt,
                                 vae_ckpt=cfg.vae_ckpt)

            # 發散到 NaN 時當場停下並如實記錄，不交給指標套件。
            #
            # 2026-08-06 加入。段 0 的 lr 格點刻意跨數個量級，**本來就預期
            # 會有候選發散**——`_pick_best` 依 `finite` 欄剔除它們。但
            # `fidelity_term` 的 `clamp(0, 1)` 不會消除 NaN（NaN 夾出來仍是
            # NaN），於是 `piq.ssim` 的輸入檢查以
            # `AssertionError: Expected values to be greater or equal to 0,
            # got nan` 中止，整個段 0 陪葬。實測於 SD v1.4／512²，
            # `lr.N3_stage1 = 0.1` 的第 20 步之後。
            #
            # 記 inf 而非跳過：這個候選確實發散了，該事實要進 `lr_probe.csv`
            # 供報告引用；靜默略過會讓它看起來只是步數比較少。
            if not torch.isfinite(x_gen).all():
                print(f"  [align] step {step:>4d}  發散：x_gen 出現非有限值，"
                      "本候選就此停止", flush=True)
                history.append({"step": step, "align_loss": float("inf"),
                                "fid_lpips": float("nan"),
                                "fid_psnr_total": float("nan"),
                                "diverged": True})
                break

            loss, parts = obj.fidelity_term(x_gen, x01, x_base=None)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)

            # 保留軌跡最佳的 φ，最後還原它，而不是拿最後一步的。
            #
            # E12 實測 8 個 (載體 × 影像) 組合中有 7 個的最後一步比自己的最佳
            # 值差，且劣化幅度隨參數量遞增：site L（163,840 參數）+0.0316、
            # site W r=4（397,824）+0.0404、site W r=16（1,591,296）+0.0526。
            # 拿最後一步等於系統性低報每個載體的能力，而且低報的程度與參數量
            # 相關——那會讓「容量」與「優化穩定性」兩個變因無法分離。
            #
            # 選擇判準用總損失而非單一 LPIPS：損失才是這個階段在最小化的量，
            # 挑 LPIPS 最低的一步可能挑到 PSNR 被犧牲掉的那一步。
            #
            # **存檔必須在 `opt.step()` 之前。** 2026-08-06 修正。
            # before：`opt.step()` 在本區塊之上，於是 `state_dict()` 取到的是
            # **更新之後**的參數，而 `loss` 是更新之前那組算出來的——保留的 φ
            # 比它被選中的理由晚一步。實測（tiny-SD、lr=5.0）還原後重新前向
            # 得 0.0042，而記錄的最佳損失是 0.0000，兩者本應相等。
            # 該缺陷使 E12 訂下這條規則的目的落空：留下的不是最佳步的 φ。
            if float(loss) < best_loss:
                best_loss, best_step = float(loss), step
                best_state = {k: v.detach().clone()
                              for k, v in module.state_dict().items()}
            opt.step()

            parts["step"] = step
            parts["align_loss"] = float(loss)
            # 門檻逐影像不同，故必須隨軌跡落盤——否則 align.csv 的懲罰值
            # 事後無從還原是對哪一條線量的。
            parts.update({f"align_{k}": v for k, v in floors.items()})
            history.append(parts)
            if step % cfg.log_every == 0 or step == cfg.align_steps - 1:
                print(f"  [align] step {step:>4d}  loss={float(loss):.4f}  "
                      f"lpips={parts['fid_lpips']:.4f}  "
                      f"psnr={parts['fid_psnr_total']:.2f}", flush=True)

            del x_gen, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best_state is not None and best_step != cfg.align_steps - 1:
        module.load_state_dict(best_state)
        print(f"  [align] 還原第 {best_step} 步的 φ（loss={best_loss:.4f}）；"
              f"最後一步為 {history[-1]['align_loss']:.4f}", flush=True)
    for h in history:
        h["align_best_step"] = best_step

    with torch.no_grad():
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_align = gen.generate(x01, ctx).detach()
    # 第 0 步就發散時沒有可還原的 φ，`x_align` 仍是 NaN。交出去會讓 NaN 從
    # `x_base` 一路流進防禦階段，最後在某個指標上以看不出來源的訊息中止。
    # 此處明講：這是一個發散的對齊，不是一個可用的基準。
    raise_if_diverged(x_align, len(history))
    return x_align, history


def optimize(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    purifiers: List[Purifier],
    calib: Optional[Calibration] = None,
    calib_context: Optional[Dict[str, Any]] = None,
    y_target: Optional[torch.Tensor] = None,
) -> OptimResult:
    """三個非加性條件共用的進入點，依 `loss_cfg.defense_mode` 分派。

        targeted_output   N2、N3 階段二：走完整的 SDEdit 鏈，推向 y_target
        targeted_attn     N1：單步 UNet 前向，把注意力質量導向 shared token

    分派依「損失的著力點」而非條件名稱，與 `generator.py` 依能力分派同一原則：
    新增一個用同一著力點的條件不需要動這裡。

    **訓練期一律餵空 prompt**（`DESIGN` §2.1 的 prompt-free 設計）。防禦方不
    知道攻擊方的 prompt，用自動 caption 代替會引入一個無法量測的落差；空
    prompt 的 CLIP 編碼是 CFG 的無條件嵌入，在攻擊方的任何 prompt 下都存在。

    `y_orig` 對 φ 為常數，故對每個噪聲取樣預先算好並快取；這省下每步一條
    n_edit 長度的無梯度 UNet 鏈。
    """
    device = x01.device
    calib_context = calib_context or {}
    gen = DefenseGenerator(sd, module, k_inv=cfg.k_inv, t_max=cfg.t_max,
                           exact_inversion=cfg.exact_inversion)
    obj = DefenseObjective(loss_cfg, device)
    monitor_key = DEFENSE_MONITOR[loss_cfg.defense_mode]
    constraint_keys = (
        active_constraint_keys(loss_cfg) if cfg.stop_on_plateau else ()
    )
    if cfg.stop_on_plateau:
        print(f"  [stop] 監看 {monitor_key}；約束："
              f"{', '.join(constraint_keys)}", flush=True)

    # prompt-free 的兩個分支。**兩者不是同一個張量。**
    #
    # 2026-08-05 修訂。before：`emb_cond = sd.encode_text("")`，且呼叫
    # `sdedit` 時把同一個張量同時當成 `emb` 與 `emb_uncond`。
    # after：條件分支仍為空 prompt 的編碼，無條件分支改由 `sd.uncond_prompt()`
    # 依模型自己的規則產生。
    #
    # 兩個理由，各自獨立成立：
    #
    # 1. **CFG 會退化成 w=1。** 若條件與無條件分支逐元素相同，
    #    `ε = ε_u + w·(ε_c − ε_u)` 的括號恆為零，guidance_scale 無論設多少
    #    都等同 w=1。而 w=1 下 SD 幾乎不服從 prompt，SDEdit 退化成
    #    「加噪再去噪」——那是本專案已實測過並列為硬規則的失效模式
    #    （`LOGIC_CHECK` A6）。防禦訓練若跑在 w=1 上，等於在防一個不會發生的攻擊。
    #
    # 2. **stock SDXL 的無條件分支不是空 prompt 的編碼。** 其
    #    `force_zeros_for_empty_prompt = true`（已查證 `model_index.json`），
    #    無條件分支是零張量。用 `encode_text("")` 代替，模擬的攻擊方就不是
    #    stock 模型，而該差異沒有任何症狀。
    #
    # `uncond_prompt()` 由模型的 config 決定回傳零張量還是空 prompt 的編碼，
    # 故同一段程式在 SD v1.x 與 SDXL 上都正確。
    emb_cond = sd.encode_text("").detach()
    emb_uncond = sd.uncond_prompt().detach()

    # x_base0 = G(x; φ=0)，即該 site 未施加防禦時就已產生的圖。像素側為 x
    # 本身；生成路徑為 inversion + VAE 來回的重建。停用模塊即可取得。
    with torch.no_grad():
        was_enabled = module.enabled
        module.disable()
        try:
            ctx0 = gen.prepare(x01, prompt_def=cfg.prompt_def)
            x_base0 = gen.generate(x01, ctx0).detach()
        finally:
            if was_enabled:
                module.enable()

    result = OptimResult(x_base0=x_base0)
    x_base = x_base0

    # ---- 可選的保真對齊 ----
    if cfg.align_steps > 0:
        if torch.equal(x_base0, x01):
            # G(x; φ=0) 已逐元素等於原圖，沒有重建誤差可吸收。以數值判定而非
            # 依 site 名稱分支：判準是「這個位置有沒有重建誤差」。
            print("  [align] G(x; φ=0) 已逐元素等於 x，略過保真對齊", flush=True)
        else:
            ta = time.perf_counter()
            x_base, result.align_history = align(
                sd, module, x01, cfg, loss_cfg, gen, calib, calib_context,
                x_base0)
            result.align_seconds = time.perf_counter() - ta
            # 保真基準改為對齊實際達成的重建。對齊若成功 x_base ≈ x，相對
            # hinge 與絕對 hinge 自然合流；若失敗，x_base 仍是誠實的基準，
            # 不會把對齊沒解決的重建誤差算到防禦頭上。

    if loss_cfg.defense_mode == "targeted_output":
        step_fn = _build_output_step(
            sd, gen, obj, cfg, x01, emb_cond, emb_uncond, purifiers, y_target,
            x_base, result)
    else:
        step_fn = _build_attn_step(
            sd, gen, obj, cfg, x01, emb_cond, emb_uncond, purifiers, x_base, result)

    t0 = time.perf_counter()
    run_stages(module, cfg, calib, calib_context, step_fn,
               constraint_keys=constraint_keys, monitor_key=monitor_key,
               result=result)
    result.seconds = time.perf_counter() - t0

    # x̂₀ 軌跡是診斷量，以最終的 φ 在 no_grad 下重算一次。開啟停止準則時
    # 無法預先知道哪一步是最後一步，原地收集會在每一步都付一次代價；重算的
    # 數值與原地收集相同。
    with torch.no_grad():
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        gen.generate(x01, ctx, collect_x0=True)
        result.x0_trace = [t.detach().clone() for t in ctx.x0_trace]

    result.x_base = x_base
    return result


def _build_output_step(sd, gen, obj, cfg, x01, emb_cond, emb_uncond, purifiers,
                       y_target, x_base, result):
    """targeted_output（N2、N3 階段二）的單步前向。

    兩條分支的噪聲逐元素相同：噪聲跨 step 固定而非每步重抽，因為目標是對
    「這組噪聲下的編輯」造成偏移，每步換噪聲會讓梯度訊號被取樣噪音淹沒。
    """
    device = x01.device
    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    noises = [
        sd.sample_edit_noise(torch.empty(lat, device=device), seed=cfg.seed + i)
        for i in range(cfg.n_eot)
    ]
    with torch.no_grad():
        y_origs = [
            sd.sdedit(x01, emb_cond, n, cfg.n_edit, strength=cfg.strength,
                      guidance_scale=cfg.guidance_scale, emb_uncond=emb_uncond)
            for n in noises
        ]

    def step_fn(stage, stage_step, global_step):
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_def = gen.generate(x01, ctx, use_ckpt=cfg.unet_ckpt,
                             vae_ckpt=cfg.vae_ckpt)
        raise_if_diverged(x_def, global_step)

        # (淨化算子索引, 噪聲索引) 的取樣清單。defense_term 對清單取平均，
        # 故 "all" 模式下一次 backward 得到的就是全算子的平均梯度。
        pairs = eot_pairs(cfg.purify_mode, global_step, cfg.n_eot,
                          len(purifiers))
        y_defs, y_refs = [], []
        for pi, i in pairs:
            y_defs.append(
                sd.sdedit(
                    purifiers[pi].forward(x_def), emb_cond, noises[i],
                    cfg.n_edit, strength=cfg.strength,
                    use_ckpt=cfg.unet_ckpt, vae_ckpt=cfg.vae_ckpt,
                    guidance_scale=cfg.guidance_scale, emb_uncond=emb_uncond,
                )
            )
            y_refs.append(y_origs[i])

        total, log = obj(x_def, x01, x_base=x_base, y_def_list=y_defs,
                         y_orig_list=y_refs, y_target=y_target)
        # 每步覆寫，因為任何一步都可能是最後一步。一張 512² 的 clone 約 3 MB，
        # 相對 E0 量到的 9.95 GB 峰值可忽略。
        result.x_def = x_def.detach().clone()
        return total, log

    return step_fn


def _build_attn_step(sd, gen, obj, cfg, x01, emb_cond, emb_uncond, purifiers,
                     x_base, result):
    """targeted_attn（N1）的單步前向。

    著力點與輸出端不同：直接作用在使文字編輯得以定位的機制上——UNet 的
    cross-attention 把每個 token 綁到影像的特定區域，質量被導向語意無資訊的
    shared token 則編輯無從落點。

    只做單步 UNet 前向，不走完整的 SDEdit 鏈。綁定是逐 timestep 的性質，在
    取樣到的 t 上把它改掉即可。成本因此由 `0.304·n_edit`（10 步）降為
    `attn_timesteps` 次單步前向。

    **這條前向不能開 UNet checkpoint。** 實測（tiny-SD）backward 以
    RuntimeError 中止：

        A different number of tensors was saved during the original forward
        and recomputation. Number of tensors saved during forward: 477
        Number of tensors saved during recomputation: 459.

    原因是 hook 在原前向時是掛著的，它額外算的 to_q / to_k / QKᵀ 進了
    checkpoint 區塊的圖；而 backward 觸發重算時 context manager 早已離開、
    hook 已卸除，重算的圖少了那一段，兩次存檔的張量數對不上。此處以
    `sd._eps(...)` 不傳 use_ckpt（預設 False）迴避。

    不需要參考分佈：targeted 的目標是「把質量導向哪裡」，不是「離原分佈多遠」。
    這正是移除 `attn_mode="divergence"` 的理由——任何「與未防禦參照的散度」
    形式在 φ=0 時都取到最小值，梯度精確為零（實測 grad_norm = 0.000e+00）。
    """
    from src.models.attention import CrossAttentionRecorder

    device = x01.device
    rec = CrossAttentionRecorder(sd.unet)

    # 取樣 timestep：均分於 [0, t_edit]，即 SDEdit 在該 strength 下實際走過
    # 的區間。超出該區間的 t 對攻擊者的編輯不起作用，在那裡施力是浪費預算。
    t_edit = min(int(sd.num_train_timesteps * cfg.strength),
                 sd.num_train_timesteps - 1)
    t_list = torch.linspace(0, t_edit, cfg.attn_timesteps + 1)[1:].round().long()
    abar = sd.alphas_cumprod(device)

    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    noises = [
        sd.sample_edit_noise(torch.empty(lat, device=device), seed=cfg.seed + i)
        for i in range(len(t_list))
    ]

    def step_fn(stage, stage_step, global_step):
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_def = gen.generate(x01, ctx, use_ckpt=cfg.unet_ckpt,
                             vae_ckpt=cfg.vae_ckpt)
        raise_if_diverged(x_def, global_step)

        pairs = eot_pairs(cfg.purify_mode, global_step, len(t_list),
                          len(purifiers))
        maps_list = []
        for pi, ti in pairs:
            z_def = sd.encode_image(purifiers[pi].forward(x_def),
                                    use_ckpt=cfg.vae_ckpt)
            t, n = t_list[ti], noises[ti]
            zt = abar[t].sqrt() * z_def + (1 - abar[t]).sqrt() * n
            with rec:
                sd._eps(zt, t, emb_cond)   # 見 docstring：此處不開 checkpoint
            maps_list.append(list(rec.maps))
            rec.clear()

        total, log = obj(x_def, x01, x_base=x_base, attn_maps=maps_list)
        if global_step == 0 and log["shared_mass"] < MIN_SHARED_MASS:
            raise RuntimeError(
                f"起點的 shared token 質量為 {log['shared_mass']:.3e}，"
                f"低於下限 {MIN_SHARED_MASS}。"
                f"`shared_tokens={obj.cfg.shared_tokens}` 在這個模型上幾乎"
                "拿不到注意力，`1 − mass` 因此坐在它的最大值上，梯度為零，"
                "最佳化不會產生任何更新。改用一個確實被 attend 的 token 格"
                "（`--shared-tokens`）。SDXL 的實測見 `RUNBOOK` §3。"
            )
        result.x_def = x_def.detach().clone()
        return total, log

    return step_fn


def optimize_encoder(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    purifiers: List[Purifier],
    calib: Optional[Calibration] = None,
    calib_context: Optional[Dict[str, Any]] = None,
    z_target: Optional[torch.Tensor] = None,
) -> OptimResult:
    """VAE 編碼器目標（PhotoGuard 的 encoder attack 形式）。

        min_φ  ‖E_vae(P(x_def)) − z_target‖²  +  λ_fid · L_fid(x_def, x)

    **不在本輪的 9 個訓練條件內**（`DESIGN` §3.1），保留為機制分析的備用路徑。
    與 `optimize()` 是不同的方法而不是它的一個選項：這裡完全沒有 SDEdit、
    沒有 y_orig、沒有編輯 prompt，共用同一個進入點只會讓兩者的差異被埋在
    條件分支裡。

    成本差距來自 E0 的成本模型 `秒 ≈ 1.05 + 0.384·k_inv + 0.304·n_edit`：
    此處 `n_edit` 與 `k_inv` 兩項都不存在，每步只剩一次 VAE 編碼。

    `z_target` 預設為零張量。零是 latent 空間的一個退化點，把 x_def 推向它
    等同要求 VAE 把防禦圖看成「沒有內容」，這是 PhotoGuard encoder attack 的
    常見選擇；呼叫端可傳入其他目標。

    停止準則在本路徑上關閉：`edit_shift` 與 `shared_mass` 在此都沒有定義，
    而為 `L_def` 另訂一個監看量需要它自己的校準值（見 MONITOR_TOL）。
    """
    device = x01.device
    calib_context = calib_context or {}
    gen = DefenseGenerator(sd, module, k_inv=cfg.k_inv, t_max=cfg.t_max,
                           exact_inversion=cfg.exact_inversion)
    obj = DefenseObjective(loss_cfg, device)

    with torch.no_grad():
        was_enabled = module.enabled
        module.disable()
        try:
            x_base0 = gen.generate(
                x01, gen.prepare(x01, prompt_def=cfg.prompt_def)).detach()
        finally:
            if was_enabled:
                module.enable()
        if z_target is None:
            z_target = torch.zeros_like(sd.encode_image(x01))

    result = OptimResult(x_base0=x_base0, x_base=x_base0)

    def step_fn(stage, stage_step, global_step):
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_def = gen.generate(x01, ctx, use_ckpt=cfg.unet_ckpt,
                             vae_ckpt=cfg.vae_ckpt)
        raise_if_diverged(x_def, global_step)
        # 淨化仍以 EOT 取樣：編碼器目標不會自動帶來耐淨化性，那是兩件事
        pairs = eot_pairs(cfg.purify_mode, global_step, 1, len(purifiers))
        z_defs = [sd.encode_image(purifiers[pi].forward(x_def),
                                  use_ckpt=cfg.vae_ckpt)
                  for pi, _ in pairs]
        l_def = torch.stack(
            [obj.encoder_term(z, z_target) for z in z_defs]).mean()
        l_fid, parts = obj.fidelity_term(x_def, x01, x_base=x_base0)
        total = loss_cfg.lam_def * l_def + loss_cfg.lam_fid * l_fid
        result.x_def = x_def.detach().clone()
        log = {"loss": float(total), "L_def": float(l_def),
               "L_fid": float(l_fid), "edit_shift": float("nan"),
               "defense_mode": "encoder", **parts}
        return total, log

    enc_cfg = replace(cfg, stop_on_plateau=False)
    t0 = time.perf_counter()
    run_stages(module, enc_cfg, calib, calib_context, step_fn,
               constraint_keys=(), monitor_key="edit_shift",
               result=result, tag="enc ")
    result.seconds = time.perf_counter() - t0
    return result
