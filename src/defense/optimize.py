"""逐圖優化 φ — spec §5、§7。

每張影像獨立優化一組 φ。防禦者擁有原圖與完整權重（spec §2 威脅模型），
故逐圖優化符合威脅模型，不需泛化到未見影像。

成本模型（E0 實測，V100 32GB, SD v1-4, 512², fp32, UNet+VAE checkpoint）：

    seconds ≈ 1.05 + 0.384·k_inv + 0.304·n_edit·n_eot
    peak    ≈ 9.95 GB，於 k_inv、n_edit ∈ [5,50] 幾乎不變

記憶體與步數無關是因為兩條 UNet 鏈與三次 VAE 呼叫都已 checkpoint；時間則
與步數線性相關。故 n_eot 直接乘在時間上，是本迴圈最貴的參數。
"""

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import torch

from src.defense.generator import DefenseGenerator
from src.defense.objective import DefenseObjective, LossConfig
from src.models.sd import SDWrapper
from src.purify.ops import Purifier
from src.residual.base import ResidualModule


@dataclass
class OptimConfig:
    steps: int = 100
    lr: float = 0.05
    k_inv: int = 10
    t_max: Optional[int] = None   # inversion 的 timestep 上限，見 E0c
    # True 改用 BDIA 精確反演（arXiv 2307.10829）取代 DDIM。tiny-SD 實測
    # latent 空間來回誤差由 1.41 降到 1.37e-04（k=20, t_max=500）。
    # 影像空間的下限不會歸零：VAE 編解碼來回誤差（真實 SD 實測
    # 27.51 dB / LPIPS 0.143）與反演無關，BDIA 不觸及該項。預期效果是把
    # 重建誤差下限由 LPIPS 0.194 降到 0.143，仍高於像素側加性位置實際運作的
    # 0.063，故此旗標是量測反演佔下限多少的工具，不是消除該下限的手段。
    exact_inversion: bool = False
    n_edit: int = 10
    n_eot: int = 1              # 每步的噪聲取樣數
    # 淨化算子的取樣方式：
    #   "rotate" — 每步只用一個算子，逐步輪替（原始行為）
    #   "all"    — 每步對全部算子求梯度後平均，即 spec §5.1 對 𝒫 的期望值
    #              以完整列舉估計，而非以輪替近似
    # 改用 "all" 的理由：主網格中模糊 σ=1.0 就在訓練集裡，測試時防禦在該
    # 條件下卻只保留 4.4%。輪替下 Adam 的動量在三個目標之間來回拉扯，等於
    # 三個互相衝突的梯度訊號輪流覆寫，而非求其共同下降方向；25 步分給三個
    # 算子，每個只有約 8 次更新。成本為每步時間乘以算子數。
    purify_mode: str = "rotate"

    # ---- 階段一：保真對齊 ----
    # 先訓練 φ 使 G(x; φ) 逼近 x，再以該 φ 熱啟動防禦訓練。
    #
    # 借鑑 APA（arXiv 2506.01511）「先立保真、再訓攻擊」的順序，其餘不同：
    # APA 在 UNet 另掛一組 LoRA、以噪聲空間的 ‖ε − ε̂‖² 訓練、階段二凍結它；
    # 此處不加任何新模組，就用同一組 φ，損失直接是影像空間的 L_fid(G(x;φ), x)。
    # 我們的 G 是固定長度的可微 DDIM 鏈，能端對端反傳到輸出影像本身，不需要
    # 噪聲空間的代理量。兩階段之間只重置 Adam 狀態，不換參數集。
    #
    # 動機是實測結果：latent 注入的 φ=0 對照與訓練 25 步的結果在每個淨化條件
    # 下都相同到小數點後三位，而該位置的重建誤差比 φ 的效果大 20–36 倍。
    # 若階段一能把重建誤差吸收掉，階段二的注入才有可能不被淹沒。
    #
    # align_steps = 0 表示不執行階段一（既有行為，保留可比性）。
    align_steps: int = 0
    align_lr: float = 0.008
    # 階段一專用的 PSNR 係數，覆蓋 LossConfig.gamma_psnr（防禦階段為 0.0）。
    # 理由見 align() 的 docstring：重建對齊是逐像素準確度確實重要的場合。
    align_gamma_psnr: float = 1.0
    strength: float = 0.5
    # site S 專用：位移場的硬上界，單位為像素。空間變形的保真度預算是
    # 位移量而非 L∞（把一條邊緣移動一像素，L∞ 可接近 1.0 卻幾乎看不出來），
    # 故此值必須與 tau_lpips 併列記錄，才能說清楚該格的失真預算是什麼。
    warp_max_disp: float = 1.5
    # site S 的 grid_sample 插值模式。預設維持 bilinear 使 E13–E19 可重現；
    # E20 §5.2 量出 bicubic 可把銳利度保留率由 85.0% 拉到 99.9%。
    warp_resample: str = "bilinear"
    # site C 專用：色度矩陣場偏離單位陣的硬上界。與 warp_max_disp 同一角色
    # ——本位置的保真度預算是矩陣偏離量而非 L∞，故必須與 tau_lpips 併列記錄。
    color_max_dev: float = 0.15
    # 攻擊方的 classifier-free guidance 權重。預設 1.0 只為了讓 E2–E23 的
    # 數字可重現，它不是正確的威脅模型：E26 實測 w=1 時 SD v1.4 幾乎不服從
    # prompt，該設定下的「編輯」與「什麼都沒做」在指標上分不出來（Δclip
    # +0.0116，而對照的噪聲範圍是 ±0.0169）。新實驗一律指定 7.5。
    # 見 docs/RESULTS_E25-E31.md §3。
    guidance_scale: float = 1.0

    # ---- 停止準則 ----
    # 預設 False 使 `steps` 維持既有語意（跑滿），既有 53 個 run 可重現。
    # 開啟後 `steps` 變成上限，實際步數由 `plateau_stop` 決定，見該函式的
    # docstring 與 docs/RESULTS_E13-E23.md §5.4。正式重跑必須開啟。
    stop_on_plateau: bool = False
    stop_patience: int = 20     # 觀察窗長度，切成前後兩半比較改善量
    # 每步的絕對改善量門檻，低於此值視為進展已停。
    #
    # **這個值與監看的量綁在一起，換監看量就必須換它**（見 MONITOR_TOL）。
    # 預設 1e-4 是為 `edit_shift` 校準的：E23 實測 site P 在 25→100 步之間
    # 平均每步改善 5.4e-4 且末端仍在上升，該格必須被判為未收斂，故取低一個
    # 量級的 1e-4。用絕對量而非相對率的理由見 `plateau_stop` 的 docstring。
    #
    # `None` 表示依 `monitor_key` 自動取 MONITOR_TOL 的值；給定數值則覆寫。
    stop_tol: Optional[float] = None
    stop_min_steps: int = 25    # 下限，讓約束有機會啟動；與 E15–E23 的步數一致
    # 不在約束仍被違反時停止。預設 False 以保留 E21–E31 的可重現性；
    # 匹配失真的比較一律開啟，理由見 plateau_stop 的 `require_feasible`。
    stop_require_feasible: bool = False
    # ---- cross-attention 目標專用 ----
    # "divergence" — 把防禦圖的注意力分佈推離原圖的（改變綁定的指向）
    # "entropy"    — 直接把分佈推向均勻（瓦解綁定本身，不需要參考分佈）
    # "suppress"   — 降低內容 token 分到的注意力質量（需要 attn_content_only）
    #
    # 預設由 "divergence" 改為 "suppress"（2026-08-01，E25）。divergence
    # 在 φ=0 的梯度精確為零（KL 在最小值處），最佳化永遠離不開起點，拿它去
    # 跑會不會產生任何更新（E20 §9 實測 grad_norm = 0.000e+00）。留著它是為了
    # 讓該缺陷有具名的位置與釘住它的測試，但它不該是預設。suppress 的最佳點
    # 不在 φ=0，故起步梯度非零，見 src/models/attention.py 的同名函式。
    attn_mode: str = "suppress"
    # 每步取樣幾個 timestep。cross-attention 的綁定強度隨 t 變化，只在單一
    # t 上施力等於只防住編輯鏈的其中一步。取樣點均分於 [0, t_edit]，
    # t_edit 為 SDEdit 在該 strength 下的起始 timestep。
    attn_timesteps: int = 4
    # 只作用在 prompt 的內容 token 上（排除 BOS/EOS/padding），見 token_span
    attn_content_only: bool = True
    prompt_def: str = ""        # 防禦生成的 prompt，spec §1.1 要求也測空 prompt
    prompt_edit: str = "a photo"
    seed: int = 20260728
    unet_ckpt: bool = True      # E0：512² 下關閉必 OOM，故預設開啟
    vae_ckpt: bool = True       # E0b：三次 VAE 呼叫的激活由總和降為最大值
    log_every: int = 10
    grad_clip: float = 1.0


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
# `attn_div` 的範圍為何這麼小：它是「1 − 內容 token 的注意力質量」，而內容
# token 的質量本來就只有約 1.5%（BOS/EOS/padding 吸走大部分），故可達的改善
# 上限就是那 1.5%。這也是本專案的變體與論文式 (5) 的一個實質差異——後者取
# 聚合反應的 L1、保留空間維度，動態範圍大得多（LEDGER 5.7）。
#
# 取值規則與 `edit_shift` 相同：比實測的平均每步改善低一個量級。
MONITOR_TOL = {
    "edit_shift": 1e-4,     # E23 實測平均 5.4e-4
    "attn_div": 1e-5,       # 2026-08-04 實測平均 1.6e-4
}


def resolve_stop_tol(cfg_tol: Optional[float], monitor_key: str) -> float:
    """回傳這次要用的 `stop_tol`。呼叫端給了就用它，否則依監看量取。

    未知的監看量直接拋出，不退回 `edit_shift` 的值：那正是這道缺陷的形狀
    ——一個為某個量校準的門檻被沿用到另一個量上，而且沒有症狀。
    """
    if cfg_tol is not None:
        return cfg_tol
    if monitor_key not in MONITOR_TOL:
        raise KeyError(
            f"監看量 {monitor_key!r} 沒有校準過的 stop_tol。"
            f"已校準的是 {sorted(MONITOR_TOL)}。"
            "不可沿用別的量的門檻——不同監看量的動態範圍差數十倍，"
            "同一個絕對門檻在兩者上是兩件不同的事（見 MONITOR_TOL）"
        )
    return MONITOR_TOL[monitor_key]


def active_constraint_keys(loss_cfg: LossConfig) -> tuple:
    """回傳「係數非零、因而真的會綁住優化」的 hinge 對應的記錄鍵。

    不能寫死成 LPIPS 與鈍化那兩道。`fidelity_term` 一律計算並記錄全部
    四道 hinge 的懲罰值，但係數為零的那幾道不進梯度、不構成約束。反過來，
    係數非零的任何一道都可能是實際綁住這一格的那道——tiny-SD 的端到端測試
    即出現 `pen_lpips = pen_acut = 0` 而 `pen_linf = 0.098`（×100 = 9.8，
    佔 L_fid 的全部）的情形，與 E13 量到的「L∞ hinge 把 site S 完全節流」
    同一回事，site C 也有（色度變換的 L∞ 大而人眼感知小）。

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

    1. 約束確實啟動過（觀察窗內某一步的 LPIPS 或鈍化懲罰大於零）。沒有
       這一條，一格可能因為還沒碰到預算就「收斂」，那是步數不足不是收斂。
    2. 進展已停（觀察窗後半的 `edit_shift` 相對前半的改善低於 `tol`）。

    監看 `edit_shift` 而非 `L_def`：後者是 hinge 過的，飽和後恆為 0，看不出
    偏移還在不在增加（見 `objective.py` 的 `__call__`）。

    `tol` 是每步的絕對改善量，不是相對改善率。初版寫的是相對量
    `(b − a) / max(|a|, 1e-8)`，在 `edit_shift` 接近零時分母趨近零、判定被
    噪聲主導——tiny-SD 的端到端煙霧測試中 shift 在 1e-4 量級來回抖動，
    相對改善動輒 ±0.16，40 步跑滿都不會停。改用絕對量的理由不只是數值穩定：
    `edit_shift` 是 LPIPS 距離，其尺度跨 site 相同（那正是選它當監看量的
    原因），故絕對門檻在兩個 site 上意義一致，而相對門檻會讓 shift=0.02 與
    shift=0.2 的「收斂」定義不同。

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

    理由：`optimize_crossattn` 原本是固定步數，而固定步數的網格是 E21–E23
    §5.4 記錄的方法問題，E31 要把平台停止接上去。但該路徑的 history 把
    `edit_shift` 記為 `float("nan")`（本檔 `optimize_crossattn` 的 log 區塊）
    ——它沒有 y_orig 可比，那個欄位在該路徑上沒有定義。NaN 的一切比較恆為
    False，直接沿用會讓停止準則**永遠不觸發**，症狀是「加了停止準則但沒有
    作用」而非報錯。crossattn 改監看 `attn_div`，其方向與 `edit_shift` 一致
    （兩者都是越大代表防禦越有進展），故不等式與 `tol` 的符號都不變。

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
        #
        # 本函式的 docstring 要求「每一格都被同一道約束綁住**且已在該約束
        # 下收斂**」。在超標時停止不符合這個意圖，故加這道條件。預設關閉
        # 以保留 E21–E31 既有 run 的可重現性。
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
                f"False，用它當監看量會永遠不停。crossattn 路徑的 "
                f"edit_shift 沒有定義（無 y_orig 可比），該路徑須改用 "
                f"monitor_key='attn_div'"
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
    梯度訊號的構成，是 spec §5.1 期望值估計的實作，不該只能靠跑完整實驗
    才看得出對錯。
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
    # 防禦訓練所用的保真基準。未執行階段一時為 G(x; φ=0)；執行後為
    # G(x; φ_align)，即階段一結束時該位置實際能做到的重建。
    x_base: Optional[torch.Tensor] = None
    x_base0: Optional[torch.Tensor] = None  # 恆為 G(x; φ=0)，作為對照保留
    x0_trace: List[torch.Tensor] = field(default_factory=list)
    align_history: List[Dict] = field(default_factory=list)
    align_seconds: float = 0.0
    seconds: float = 0.0
    steps_done: int = 0
    # 停止的原因。空字串表示跑滿 `steps`（即上限用盡而非收斂），該格
    # 不可用於跨 site 比較——那正是 E21–E23 §5.4 的問題。
    stop_reason: str = ""


def align(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    gen: DefenseGenerator,
) -> tuple[torch.Tensor, List[Dict]]:
    """階段一：訓練 φ 使 G(x; φ) 逼近 x。回傳 (x_align, history)。

    損失以 `fidelity_term(G(x;φ), x, x_base=None)` 為基礎，即對原圖的絕對
    保真度。`x_base=None` 使兩道 hinge 的對象為原圖本身；hinge 在容差內
    不施力是此處要的行為——對齊到 τ 以內即停止，不必把重建誤差壓到零。

    但係數與階段二不同：`gamma_psnr` 由 `cfg.align_gamma_psnr` 覆蓋。
    防禦階段把 PSNR 移出梯度是對的（逐像素平方誤差與人眼可辨性關聯薄弱，
    見 objective 的修訂之二），但重建對齊正是逐像素準確度確實重要的場合，
    同一組係數不該同時適用兩者。

    這是 E9 直接量到的問題：200 步對齊後 car_00 的 LPIPS 由 0.2032 降到
    0.0950，PSNR 卻只由 19.62 動到 20.54；car_01 的 PSNR 甚至掉了 1.84 dB
    （22.28 → 20.44）。PSNR 當時完全不在損失裡，那些數字反映的是自由漂移，
    不是容量限制，因此無法用來判斷任何事。

    這個階段可能失敗，而失敗本身是結果：低秩 ε 注入未必有足夠容量吸收
    VAE 與 DDIM 的重建誤差。E9 實測 car_01 在第 50 步即停在 LPIPS 0.166，
    其後 150 步無改善——那是容量天花板，不是步數不足。回傳的 history 記錄
    逐步的 LPIPS 與 PSNR，呼叫端據此判斷，不得假設對齊必然成功。
    """
    # 只覆蓋 gamma_psnr，其餘係數與階段二一致：保真度的「定義」不變，
    # 變的是逐像素項在這個階段要不要參與梯度。
    align_cfg = replace(loss_cfg, gamma_psnr=cfg.align_gamma_psnr)
    obj = DefenseObjective(align_cfg, x01.device)
    opt = torch.optim.Adam(module.parameters(), lr=cfg.align_lr)
    history: List[Dict] = []
    best_loss, best_step, best_state = float("inf"), -1, None

    for step in range(cfg.align_steps):
        opt.zero_grad(set_to_none=True)
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_gen = gen.generate(x01, ctx, use_ckpt=cfg.unet_ckpt,
                             vae_ckpt=cfg.vae_ckpt)
        loss, parts = obj.fidelity_term(x_gen, x01, x_base=None)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
        opt.step()

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
        if float(loss) < best_loss:
            best_loss, best_step = float(loss), step
            best_state = {k: v.detach().clone()
                          for k, v in module.state_dict().items()}

        parts["step"] = step
        parts["align_loss"] = float(loss)
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
    return x_align, history


def optimize(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    purifiers: List[Purifier],
    y_target: Optional[torch.Tensor] = None,
) -> OptimResult:
    """優化 φ 使編輯偏移最大化，同時維持 x_def 與 x 的保真。

    `y_target` 僅在 `loss_cfg.defense_mode == "targeted"` 時使用，為編輯
    結果要被推向的固定目標影像。無目標模式下傳入會被忽略。

    `y_orig` 對 φ 為常數（spec §5.1），故對每個噪聲取樣預先算好並快取；
    這省下每步一條 n_edit 長度的無梯度 UNet 鏈。
    """
    device = x01.device
    gen = DefenseGenerator(sd, module, k_inv=cfg.k_inv, t_max=cfg.t_max,
                          exact_inversion=cfg.exact_inversion)
    obj = DefenseObjective(loss_cfg, device)
    opt = torch.optim.Adam(module.parameters(), lr=cfg.lr)

    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    emb_edit = sd.encode_text(cfg.prompt_edit).detach()
    # 無條件嵌入即空字串的 CLIP 編碼，與 diffusers 一致。w=1 時不會被用到，
    # 但仍無條件算出來：讓「有沒有開 CFG」只由 guidance_scale 決定，
    # 不再多一個「emb_uncond 在不在」的隱藏分支。
    emb_uncond = sd.encode_text("").detach()

    # 每個 EOT 取樣配一組固定噪聲。噪聲跨 step 固定而非每步重抽：目標是
    # 對「這組噪聲下的編輯」造成偏移，每步換噪聲會讓梯度訊號被取樣噪音淹沒。
    noises = [
        sd.sample_edit_noise(torch.empty(lat, device=device), seed=cfg.seed + i)
        for i in range(cfg.n_eot)
    ]

    # y_orig 不依賴 φ，整個優化過程只算一次
    with torch.no_grad():
        y_origs = [
            sd.sdedit(x01, emb_edit, n, cfg.n_edit, strength=cfg.strength,
                      guidance_scale=cfg.guidance_scale, emb_uncond=emb_uncond)
            for n in noises
        ]

        # x_base = G(x; φ=0)，即該 site 未施加防禦時就已產生的圖。site P 為
        # x 本身；site L 為 inversion + VAE 來回的重建。用於把 L∞ hinge 的
        # 對象限定在「φ 造成的改變」（見 objective.fidelity_term）。停用模塊
        # 即可取得，之後還原原本的啟用狀態。
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

    # ---- 階段一：保真對齊 ----
    if cfg.align_steps > 0:
        if torch.equal(x_base0, x01):
            # G(x; φ=0) 已逐元素等於原圖，沒有重建誤差可吸收。以數值判定而非
            # 依 site 名稱分支：判準是「這個位置有沒有重建誤差」，不是「這是
            # 哪個 site」，將來新增位置不必改這裡。
            print("  [align] G(x; φ=0) 已逐元素等於 x，略過階段一", flush=True)
        else:
            ta = time.perf_counter()
            x_base, result.align_history = align(
                sd, module, x01, cfg, loss_cfg, gen)
            result.align_seconds = time.perf_counter() - ta
            # 保真基準改為階段一實際達成的重建。階段一若成功，x_base ≈ x，
            # 相對 hinge 與絕對 hinge 自然合流；若失敗，x_base 仍是誠實的
            # 基準，不會把階段一沒解決的重建誤差算到防禦頭上。
            opt = torch.optim.Adam(module.parameters(), lr=cfg.lr)

    t0 = time.perf_counter()
    # 哪些 hinge 真的會綁住這次優化，由係數決定而非寫死。見 active_constraint_keys。
    constraint_keys = (
        active_constraint_keys(loss_cfg) if cfg.stop_on_plateau else ()
    )
    if cfg.stop_on_plateau:
        print(f"  [stop] 監看的約束：{', '.join(constraint_keys)}", flush=True)

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)

        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        # 開啟停止準則時無法預先知道哪一步是最後一步，故訓練期一律不收集
        # x0 軌跡，改在迴圈結束後以最終的 φ 在 no_grad 下重算一次。該軌跡
        # 是診斷量，重算與原地收集的數值相同。關閉時維持原路徑不變。
        x_def = gen.generate(
            x01, ctx, use_ckpt=cfg.unet_ckpt, vae_ckpt=cfg.vae_ckpt,
            collect_x0=(not cfg.stop_on_plateau and step == cfg.steps - 1),
        )

        # (淨化算子索引, 噪聲索引) 的取樣清單。defense_term 對清單取平均，
        # 故 "all" 模式下一次 backward 得到的就是全算子的平均梯度，而不是
        # 輪替下「這一步只朝某一個算子下降」的訊號。
        pairs = eot_pairs(cfg.purify_mode, step, cfg.n_eot, len(purifiers))

        y_defs, y_refs = [], []
        for pi, i in pairs:
            p = purifiers[pi]
            y_defs.append(
                sd.sdedit(
                    p.forward(x_def), emb_edit, noises[i], cfg.n_edit,
                    strength=cfg.strength,
                    use_ckpt=cfg.unet_ckpt, vae_ckpt=cfg.vae_ckpt,
                    guidance_scale=cfg.guidance_scale, emb_uncond=emb_uncond,
                )
            )
            y_refs.append(y_origs[i])

        total, log = obj(x_def, x01, y_defs, y_refs,
                         x_base=x_base, y_target=y_target)
        total.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
        opt.step()

        log["step"] = step
        log["grad_norm"] = float(
            torch.stack(
                [p.grad.norm() for p in module.parameters() if p.grad is not None]
            ).norm()
        )
        result.history.append(log)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            print(
                f"  step {step:>4d}  loss={log['loss']:.4f}  "
                f"L_def={log['L_def']:.4f}  L_fid={log['L_fid']:.4f}  "
                f"shift={log['edit_shift']:.4f}  psnr={log['fid_psnr']:.2f}  "
                f"linf={log['fid_linf']:.4f}  |g|={log['grad_norm']:.3e}",
                flush=True,
            )

        if cfg.stop_on_plateau:
            # 每步覆寫，因為任何一步都可能是最後一步。一張 512² 的 clone
            # 約 3 MB，相對 E0 量到的 9.95 GB 峰值可忽略。
            result.x_def = x_def.detach().clone()
        elif step == cfg.steps - 1:
            result.x_def = x_def.detach().clone()
            result.x0_trace = [t.detach().clone() for t in ctx.x0_trace]

        del x_def, y_defs, total
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cfg.stop_on_plateau:
            stop, reason = plateau_stop(
                result.history, cfg.stop_patience,
                resolve_stop_tol(cfg.stop_tol, "edit_shift"),
                cfg.stop_min_steps, constraint_keys=constraint_keys,
                require_feasible=cfg.stop_require_feasible,
            )
            if stop:
                result.stop_reason = reason
                print(f"  [stop] 第 {step} 步停止：{reason}", flush=True)
                break

    result.steps_done = len(result.history)
    if cfg.stop_on_plateau:
        if not result.stop_reason:
            # 跑滿上限而非收斂。這一格不可用於跨 site 比較，理由與
            # E21–E23 §5.4 相同：量到的是「走到哪裡」不是「能力」。
            result.stop_reason = ""
            print(f"  [stop] 用盡上限 {cfg.steps} 步仍未達停止準則，"
                  f"該格不可用於跨 site 比較", flush=True)
        with torch.no_grad():
            ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
            gen.generate(x01, ctx, collect_x0=True)
            result.x0_trace = [t.detach().clone() for t in ctx.x0_trace]

    result.x_base = x_base
    result.seconds = time.perf_counter() - t0
    return result


def optimize_encoder(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    purifiers: List[Purifier],
    z_target: Optional[torch.Tensor] = None,
) -> OptimResult:
    """VAE 編碼器目標（PhotoGuard 的 encoder attack 形式）。

        min_φ  ‖E_vae(P(x_def)) − z_target‖²  +  λ_fid · L_fid(x_def, x)

    與 optimize() 是不同的方法，不是它的一個選項，故獨立成一個函式：
    這裡完全沒有 SDEdit、沒有 y_orig、沒有編輯 prompt，共用同一個迴圈只會
    讓兩者的差異被埋在條件分支裡。

    成本差距來自 E0 的成本模型 `秒 ≈ 1.05 + 0.384·k_inv + 0.304·n_edit`：
    此處 `n_edit` 與 `k_inv` 兩項都不存在，每步只剩一次 VAE 編碼。1000 步
    的成本仍低於 optimize() 的 25 步。

    同時消除本專案最大的兩個過擬合來源——目標不依賴任何特定的編輯 prompt
    或噪聲取樣（實測噪聲過擬合 3.3 倍，prompt 過擬合從未量過）。代價是它
    不再針對「這個編輯」最佳化，是泛化性換特異性的取捨，須實測而非假設。

    `z_target` 預設為零張量。零是 latent 空間的一個退化點，把 x_def 推向
    它等同要求 VAE 把防禦圖看成「沒有內容」，這是 PhotoGuard encoder attack
    的常見選擇；呼叫端可傳入其他目標。
    """
    device = x01.device
    gen = DefenseGenerator(sd, module, k_inv=cfg.k_inv, t_max=cfg.t_max,
                          exact_inversion=cfg.exact_inversion)
    obj = DefenseObjective(loss_cfg, device)
    opt = torch.optim.Adam(module.parameters(), lr=cfg.lr)

    with torch.no_grad():
        was_enabled = module.enabled
        module.disable()
        try:
            x_base0 = gen.generate(x01, gen.prepare(x01, prompt_def=cfg.prompt_def))
            x_base0 = x_base0.detach()
        finally:
            if was_enabled:
                module.enable()
        if z_target is None:
            z_target = torch.zeros_like(sd.encode_image(x01))

    result = OptimResult(x_base0=x_base0, x_base=x_base0)
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_def = gen.generate(x01, ctx, use_ckpt=cfg.unet_ckpt,
                             vae_ckpt=cfg.vae_ckpt)

        # 淨化仍以 EOT 取樣：編碼器目標不會自動帶來耐淨化性，那是兩件事
        pairs = eot_pairs(cfg.purify_mode, step, 1, len(purifiers))
        z_defs = [sd.encode_image(purifiers[pi].forward(x_def),
                                  use_ckpt=cfg.vae_ckpt)
                  for pi, _ in pairs]
        l_def = torch.stack([obj.encoder_term(z, z_target) for z in z_defs]).mean()
        l_fid, parts = obj.fidelity_term(x_def, x01, x_base=x_base0)
        total = loss_cfg.lam_def * l_def + loss_cfg.lam_fid * l_fid
        total.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
        opt.step()

        log = {"step": step, "loss": float(total), "L_def": float(l_def),
               "L_fid": float(l_fid), "edit_shift": float("nan"),
               "defense_mode": "encoder", **parts}
        log["grad_norm"] = float(torch.stack(
            [p.grad.norm() for p in module.parameters() if p.grad is not None]
        ).norm())
        result.history.append(log)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            print(f"  [enc] step {step:>4d}  loss={float(total):.4f}  "
                  f"L_def={float(l_def):.4f}  psnr={parts['fid_psnr_total']:.2f}  "
                  f"|g|={log['grad_norm']:.3e}", flush=True)

        if step == cfg.steps - 1:
            result.x_def = x_def.detach().clone()

        del x_def, z_defs, total
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result.seconds = time.perf_counter() - t0
    result.steps_done = cfg.steps
    return result


def optimize_crossattn(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    purifiers: List[Purifier],
) -> OptimResult:
    """cross-attention 目標：破壞 prompt token 與影像位置之間的綁定。

        max_φ  E_t[ D( A(P(x_def), c, t), A(x, c, t) ) ]  −  λ_fid · L_fid

    著力點與 optimize() 不同。optimize() 在輸出端量測「編輯結果被推開
    多少」；此處直接作用在使文字編輯得以定位的機制上——UNet 的 cross-attention
    把每個 token 綁到影像的特定區域，綁定被破壞則編輯無從落點。
    （Xu et al., arXiv 2509.10359, ACM MM 2025 採取相同的著力點。）

    只做單步 UNet 前向，不走完整的 SDEdit 鏈。綁定是逐 timestep 的性質，
    在取樣到的 t 上把它破壞掉即可，不需要把整條 n_edit 步的鏈跑完。成本因此
    由 `0.304·n_edit`（10 步）降為 `attn_timesteps` 次單步前向。

    這條前向不能開 UNet checkpoint。實測（tiny-SD）：開了之後 backward
    以 RuntimeError 中止，訊息為

        A different number of tensors was saved during the original forward
        and recomputation. Number of tensors saved during forward: 477
        Number of tensors saved during recomputation: 459.

    原因是 hook 在原前向時是掛著的，它額外算的 to_q / to_k / QKᵀ 進了
    checkpoint 區塊的圖；而 backward 觸發重算時 context manager 早已離開、
    hook 已卸除，重算的圖少了那一段，兩次存檔的張量數對不上。此處以
    `sd._eps(...)` 不傳 use_ckpt（預設 False）迴避，單步前向不開 checkpoint
    的記憶體是可負擔的——真正需要 checkpoint 的是 n_edit 步串接的那條鏈。

    參考分佈 A(x, c, t) 不依賴 φ，故對每個取樣到的 t 只算一次並快取。
    兩條分支共用同一組加噪 ε（spec §5.1），否則量到的差異主要來自噪聲不同。

    2026-08-02（E31）接上平台停止。

        修訂前
            for step in range(cfg.steps):        ← 恆跑滿
                ...
                if step == cfg.steps - 1:
                    result.x_def = x_def.detach().clone()
            result.steps_done = cfg.steps

        修訂後
            constraint_keys = active_constraint_keys(loss_cfg) if ... else ()
            for step in range(cfg.steps):
                ...
                if cfg.stop_on_plateau or step == cfg.steps - 1:
                    result.x_def = x_def.detach().clone()   ← 每步覆寫
                if cfg.stop_on_plateau:
                    stop, reason = plateau_stop(..., monitor_key="attn_div")
                    if stop: result.stop_reason = reason; break
            result.steps_done = len(result.history)

    理由：固定步數的網格是 E21–E23 §5.4 記錄的方法問題——兩個 site 的 φ
    量綱與學習率都不同，「同樣跑 N 步」對兩者從來不是同一件事，實測後果是
    每一格被不同的東西綁住。E31 要把三個 defense_mode 放進同一個網格比較，
    這條路徑不能是唯一一條跑固定步數的。

    監看量取 `attn_div` 而非 `edit_shift`：本路徑沒有 y_orig 可比，該欄位
    記為 NaN，而 NaN 的比較恆為 False，沿用會讓停止準則永遠不觸發。
    """
    from src.models.attention import (
        CrossAttentionRecorder, attention_content_suppression,
        attention_divergence, attention_entropy, token_span,
    )

    device = x01.device
    gen = DefenseGenerator(sd, module, k_inv=cfg.k_inv, t_max=cfg.t_max,
                          exact_inversion=cfg.exact_inversion)
    obj = DefenseObjective(loss_cfg, device)
    opt = torch.optim.Adam(module.parameters(), lr=cfg.lr)
    rec = CrossAttentionRecorder(sd.unet)
    # 與 optimize() 相同：只認係數非零、真的會綁住優化的那幾道 hinge。
    constraint_keys = (
        active_constraint_keys(loss_cfg) if cfg.stop_on_plateau else ()
    )

    emb_edit = sd.encode_text(cfg.prompt_edit).detach()
    span = token_span(sd.tokenizer, cfg.prompt_edit) if cfg.attn_content_only else None
    if span is not None and span[1] <= span[0]:
        # 空 prompt 沒有內容 token；對空區間取平均會得到 nan，須明確落回全域
        print("  [attn] prompt 無內容 token，改用全部 77 格", flush=True)
        span = None
    if cfg.attn_mode == "suppress" and span is None:
        # 落回全域對 suppress 不是「比較粗糙」而是「恆等於常數 0」：全部 77 格
        # 的注意力質量和恆為 1。此處提前拒絕，而不是讓它跑完才發現什麼都沒動。
        raise ValueError(
            "attn_mode='suppress' 需要非空的內容 token 區間，"
            f"但 attn_content_only={cfg.attn_content_only}、"
            f"prompt_edit={cfg.prompt_edit!r} 得到的 span 為空。"
            "全部 token 的注意力質量和恆為 1，該目標會退化成常數"
        )

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

    with torch.no_grad():
        was_enabled = module.enabled
        module.disable()
        try:
            x_base0 = gen.generate(
                x01, gen.prepare(x01, prompt_def=cfg.prompt_def)).detach()
        finally:
            if was_enabled:
                module.enable()

        # 參考分佈：原圖在同一組 (t, ε) 下的注意力
        z_orig = sd.encode_image(x01)
        ref_maps = []
        for t, n in zip(t_list, noises):
            zt = abar[t].sqrt() * z_orig + (1 - abar[t]).sqrt() * n
            with rec:
                sd._eps(zt, t, emb_edit)
            ref_maps.append([m.detach() for m in rec.maps])

    result = OptimResult(x_base0=x_base0, x_base=x_base0)
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_def = gen.generate(x01, ctx, use_ckpt=cfg.unet_ckpt,
                             vae_ckpt=cfg.vae_ckpt)

        pairs = eot_pairs(cfg.purify_mode, step, len(t_list), len(purifiers))
        divs = []
        for pi, ti in pairs:
            z_def = sd.encode_image(purifiers[pi].forward(x_def),
                                    use_ckpt=cfg.vae_ckpt)
            t, n = t_list[ti], noises[ti]
            zt = abar[t].sqrt() * z_def + (1 - abar[t]).sqrt() * n
            with rec:
                sd._eps(zt, t, emb_edit)      # 見 docstring：此處不開 checkpoint
            if cfg.attn_mode == "entropy":
                divs.append(attention_entropy(rec.maps, span))
            elif cfg.attn_mode == "divergence":
                divs.append(attention_divergence(rec.maps, ref_maps[ti], span))
            elif cfg.attn_mode == "suppress":
                divs.append(attention_content_suppression(rec.maps, span))
            else:
                raise ValueError(
                    f"未知的 attn_mode: {cfg.attn_mode!r}，"
                    "只接受 'divergence'、'entropy' 或 'suppress'"
                )
            rec.clear()

        # 目標是把散度/熵推大，故取負號後最小化，與 defense_term 的符號一致
        l_def = -torch.stack(divs).mean()
        l_fid, parts = obj.fidelity_term(x_def, x01, x_base=x_base0)
        total = loss_cfg.lam_def * l_def + loss_cfg.lam_fid * l_fid
        total.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
        opt.step()

        log = {"step": step, "loss": float(total), "L_def": float(l_def),
               "L_fid": float(l_fid), "edit_shift": float("nan"),
               "attn_div": float(-l_def), "defense_mode": "crossattn",
               "attn_mode": cfg.attn_mode, **parts}
        log["grad_norm"] = float(torch.stack(
            [p.grad.norm() for p in module.parameters() if p.grad is not None]
        ).norm())
        result.history.append(log)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            # div 以科學記號輸出：實測其量級跨越數個數量級，定點格式在
            # 小的那一端全部印成 0.0000，等於沒有記錄
            print(f"  [attn] step {step:>4d}  loss={float(total):.4f}  "
                  f"div={-float(l_def):.4e}  lpips={parts['fid_lpips']:.4f}  "
                  f"psnr={parts['fid_psnr_total']:.2f}  "
                  f"|g|={log['grad_norm']:.3e}", flush=True)

        if cfg.stop_on_plateau or step == cfg.steps - 1:
            # 開啟停止準則後每步覆寫，因為任何一步都可能是最後一步。
            # 一張 512² 的 clone 約 3 MB，相對 E0 量到的峰值可忽略。
            result.x_def = x_def.detach().clone()

        del x_def, divs, total
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cfg.stop_on_plateau:
            # 監看 attn_div 而非 edit_shift：後者在本路徑恆為 NaN（見上方
            # log 區塊），NaN 的比較恆為 False，用它會永遠不停。
            stop, reason = plateau_stop(
                result.history, cfg.stop_patience,
                resolve_stop_tol(cfg.stop_tol, "attn_div"),
                cfg.stop_min_steps, constraint_keys=constraint_keys,
                monitor_key="attn_div",
            )
            if stop:
                result.stop_reason = reason
                print(f"  [stop] 第 {step} 步停止：{reason}", flush=True)
                break

    result.seconds = time.perf_counter() - t0
    result.steps_done = len(result.history)
    if cfg.stop_on_plateau and not result.stop_reason:
        # 跑滿上限而非收斂。這一格不可用於跨 site 比較，理由與 E21–E23
        # §5.4 相同：量到的是「走到哪裡」不是「能力」。
        print(f"  [stop] 用盡上限 {cfg.steps} 步仍未達停止準則，"
              f"該格不可用於跨 site 比較", flush=True)
    return result
