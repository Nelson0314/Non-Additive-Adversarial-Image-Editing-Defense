"""五段的計算層 — `docs/ARCHITECTURE.md` §5、`docs/reference/CODE_CONTRACTS.md` §4。

`src/experiment/runner.py` 只管「哪些格要跑、跑過了沒有」；本模組是它呼叫的
`executor`，負責實際的訓練、縮放、評測與彙整。兩者分離的理由見 `runner.py`
的模組 docstring：骨架的錯誤事後看不出來，故它必須有測試；而計算層需要
真實權重與 GPU，測試只能驗到契約。

## 契約

    executor(cell, ctx) -> (artifacts, extra_meta)

`artifacts` 是**相對批次目錄**的路徑清單，續跑判定會逐一檢查其存在性
（`ProgressWriter.is_done`）。故凡是列進去的路徑，執行結束時必須真的存在；
列了卻沒產生的路徑會讓該格永遠重跑，列少了則會讓「產物被清掉」判不出來。

`ctx` 只有一個鍵 `"res"`，值為 `Resources`。把昂貴物件（SD 權重、指標模型、
校準表、影像張量）收在一個物件裡而不是每格重建，是因為 SDXL 的載入時間
以分鐘計，4449 格各載一次不可行。

## 五段的分工

| 段 | 入口 | 產物 |
|---|---|---|
| calib | `run_calibration()`（非格點式） | `calib/calibration.json` 等 |
| train | `make_executor("train")` | `<條件>/<影像>/phi.pt`、`train.csv` |
| rayscale | `make_executor("rayscale")` | `<條件>/<影像>/phi_tau{τ}.pt` |
| control | `make_executor("control")` | `control/<影像>/purify/…/edit_seed{k}.png` |
| eval | `make_executor("eval")` | `<條件>/<影像>/purify/…/metrics_seed{k}.json` |
| report | `run_report()`（非格點式） | `grid.csv` |

`calib` 與 `report` 不走 `run_stage`：`grid.plan()` 沒有它們的格點，硬塞進
格點框架只會得到一個「零格、永遠成功」的段。

## 學習率只有一個入口

三個非加性條件的學習率全部由 `Calibration.get()` 取得，本模組不提供任何
預設值、不內插、不因為「差一個欄位」而放行——那正是 `resolve_lr` 存在的
理由。未跑過段 0 就跑段 1 會拋 `CalibrationMismatch`，這是刻意的。

## φ=0 對照為什麼要單獨一段

`grid.control_cells()` 對每個 `(影像, 淨化, 種子)` 只列一格，跨 9 個條件共用。
若每個條件各算一次即 9 倍的重複計算，而評測的成本主要就在這裡。代價是
`eval` 必須先跑完 `control`：呼叫端（`scripts/run_stage.py`）保證這個順序。

**對照側的參照影像由 PNG 讀回**（8-bit）。這是刻意的取捨：以 float 張量留存
每一格對照，SDXL 1024² 下是 25 影像 × 20 淨化 × 5 種子 × 12 MB ≈ 30 GB，
而 `runs/` 必須全部入版控。8-bit 量化使 PSNR 的上限約 48 dB，遠高於本專案
編輯比較實際落在的 10–25 dB 區間，故對結論無影響；但這件事必須寫在這裡，
不能讓讀者以為兩側是同精度的。
"""

import copy
import csv
import json
import math
import statistics
import time
import zlib
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from src.baselines import REGISTRY as BASELINE_REGISTRY
from src.baselines.pgd import run_pgd
from src.defense.objective import LossConfig, scaled_thresholds
from src.defense.generator import DefenseGenerator
from src.defense.optimize import Diverged, OptimConfig, StageSpec, optimize
from src.experiment import grid
from src.experiment.attention_page import build_attention_html
from src.experiment.attn_capture import AttnCapture, capture_span, sampled_steps
from src.experiment.compare_page import build_compare_html
from src.experiment.runner import cell_config
from src.metrics.ray_scale import metric_against, solve_k
from src.purify.ops import Purifier, default_train_set
from src.residual.site_apa import APA_STAGE1_STEPS, build_apa
from src.residual.site_warp import WarpResidual
from src.utils.artifacts import (
    save_history_plot, save_image, save_json, save_residual, save_x0_trace,
)
from src.utils.calibration import REQUIRED_CONTEXT, Calibration
from src.utils.cellid import config_hash
from src.utils.progress import load_cells, write_atomic


def read_env(batch_dir: Path) -> Dict[str, Any]:
    """讀 `env.json`。缺檔時回空字典——頁面會顯示 `?`，與 `dashboard.py` 一致。

    不拋出：`env.json` 由 `run_stage.py` 在跑任何一段之前寫入，缺它表示
    這是一個舊批次或手工建的目錄，那時「卡別未知」是正確的呈現，
    而不是讓整個報表段失敗。卡別與精度的權威來源是每格 `meta.json` 內的
    `config_hash`（兩者都進雜湊），不是這個頁首。
    """
    p = Path(batch_dir) / "env.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> Path:
    """HTML 產物一律經 `write_atomic`：儀表板與比對頁都可能被另一個
    session 同時開著，直接寫會讓對方讀到半份檔案。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, text)
    return path

# 評測噪聲與訓練噪聲必須錯開。φ 是針對訓練用的那一組 ε 優化出來的，
# 沿用同一組 ε 量到的是訓練集表現，偏移量會被系統性高估且幅度未知。
# 與 `scripts/run_defense.py`、`scripts/run_lo_baseline.py` 同一個常數。
EVAL_SEED_OFFSET = 10_000

# 段 0 的 strength 掃描格點。目的是確認「未防禦編輯確實成功」，
# 不是找最強的編輯——過強的 strength 會讓原圖語意整個消失，那時
# 「防禦有沒有效」不再可判。
STRENGTH_GRID: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

# 段 0 的學習率候選。跨三個量綱（位移像素、latent 值、LoRA 矩陣元素），
# 故格點必須跨數個數量級——實測不同注入位置的校準值可差 12.5 倍。
LR_GRID: Tuple[float, ...] = (1e-4, 1e-3, 5e-3, 2e-2, 1e-1)

# x̂₀ 軌跡的解碼取樣間隔（`CODE` §4.1）。每步都解碼的成本是
# 步數 × 一次 VAE decode；固定含第 0 步與最後一步。
X0_TRACE_EVERY = 5


# ---------------------------------------------------------------------------
# 條件表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionSpec:
    """一個訓練條件在計算層需要的全部設定。

    **與 `grid.CONDITIONS` 一一對應且由測試釘住。** 條件表在兩處各寫一份
    （格點列舉一份、計算層一份）是必然的——前者不該知道 LoRA 的秩，後者
    不該知道 τ 的格點。但兩份的鍵集合必須相同，否則會出現「格點列了某個
    條件、計算層卻不認識它」，而症狀是整段跑到該條件才失敗。
    """

    name: str
    kind: str                 # "nonadditive" | "baseline" | "random"
    site: str = ""            # "warp" | "apa"，baseline 為空
    defense_mode: str = ""
    target_metric: str = "lpips"
    lr_key: str = ""
    align_lr_key: str = ""
    # `plateau_stop` 的監看量。由 `optimize.DEFENSE_MONITOR` 依 defense_mode
    # 決定，此處只記錄，用於判斷要不要向校準表索取 stop_tol。
    monitor: str = ""
    # **這個旗標只管生成／編輯路徑**（`gen.generate` 與 `sd.edit`），
    # 不管注意力前向。
    #
    # 2026-08-09 修正這段說明。before 寫的是「cross-attention 擷取與 UNet
    # checkpoint 不相容，N1 因此關閉」——那句話把**症狀**與**處置的位置**
    # 對錯了。不相容是真的（backward 以「兩次存檔的張量數 477 vs 459」中止，
    # 因為 hook 在原前向掛著、重算時已卸除），但擋住它的不是這個旗標：
    # `_build_attn_step` 呼叫 `sd._eps(...)` 時**不傳 `use_ckpt`**（預設
    # False），那是寫死的、與本旗標無關。
    #
    # 對 N1 而言本旗標是**空轉**的：site warp 的 `gen.generate` 直接回傳
    # `pixel_residual`，整條路徑不碰 UNet。故它設成什麼都沒有差別，
    # 而那正是這個誤解能存活的原因。
    #
    # 對走生成路徑的條件就不是空轉。2026-08-09 於 RTX 3090 實測：apa 沿用
    # `unet_ckpt=False` 之後，段 0 的 `_probe_align_lr` 在**保真對齊**那一步
    # OOM（23.42 GiB / 23.56 GiB，死在 backward 重算 VAE decode）——那一步
    # 根本不碰注意力目標，純粹是 k_inv=10 步的去噪鏈少了 checkpoint，
    # 10 份完整的 512² fp32 UNet 計算圖同時存活。
    unet_ckpt: bool = True


CONDITION_SPECS: Dict[str, ConditionSpec] = {
    "N1": ConditionSpec(
        "N1", "nonadditive", site="warp", defense_mode="targeted_attn",
        lr_key="lr.N1", monitor="shared_mass", unet_ckpt=False,
    ),
    "N2": ConditionSpec(
        "N2", "nonadditive", site="warp", defense_mode="targeted_output",
        target_metric="lpips", lr_key="lr.N2", monitor="edit_shift",
    ),
    # N3 的階段一（LoRA 視覺一致性）走 `optimize` 的 `align()`，階段二
    # （latent 攻擊）走 `stages`。兩者的學習率分開校準：LoRA 的參數是矩陣
    # 元素、latent 注入的是噪聲量級，Adam 每步位移約等於 lr，同一個數值
    # 對兩者代表兩種步長。
    "N3": ConditionSpec(
        "N3", "nonadditive", site="apa", defense_mode="targeted_output",
        target_metric="mse", lr_key="lr.N3_stage2",
        align_lr_key="lr.N3_stage1", monitor="edit_shift",
    ),
    # 2026-08-09（第三階段）。與 N3 同一個參數化與同一組學習率校準鍵——
    # 階段一（LoRA 保真對齊）逐字相同，差別只在階段二的損失。共用
    # `lr.N3_stage*` 是刻意的：那兩個鍵校準的是「這個參數化的步長」，
    # 而參數化沒有改變；另立 `lr.N4_stage*` 會讓段 0 多探測兩組而量到同一件事。
    #
    # `unet_ckpt=False` 與 N1 同一個理由：cross-attention 的 forward pre-hook
    # 與 UNet checkpoint 不相容（backward 以「兩次存檔的張量數 477 vs 459」
    # 中止）。**這一格因此是本批記憶體最緊的一條路徑**，見 `attention.py`。
    # 2026-08-09（第三階段）。與 N3 同一個參數化，差別只在階段二的損失。
    #
    # **但學習率的鍵必須自己一組。** `_pick_best` 的判準是「固定步數後末端
    # 總損失最小者」，而總損失是模式相依的：N3 的 L_def 是 MSE、apa 是
    # `‖Att ⊙ M‖₁`，兩者的尺度與地景都不同，argmin 沒有理由相同。共用
    # `lr.N3_stage2` 還會讓同批同時跑 N3 與 apa 時，`calibrate_lr` 的
    # `out[spec.lr_key]` 被後者**靜默覆寫**。
    #
    # **`unet_ckpt` 維持預設的 True，不照抄 N1 的 False。** 兩者的注意力前向
    # 同樣不能開 checkpoint，但擋住那件事的是 `_build_attn_step` 裡寫死的
    # `sd._eps(...)`（不傳 `use_ckpt`），不是這個旗標——本旗標只管生成路徑，
    # 詳見 `ConditionSpec.unet_ckpt` 的說明與 2026-08-09 的 OOM 實測。
    #
    # **這一格仍是本批記憶體最緊的一條路徑**：注意力前向恆不 checkpoint，
    # 而 `--purify-mode all` × `attn_timesteps 4` 是 12 份完整的 UNet 計算圖，
    # 再加上生成路徑（已 checkpoint）。段 0 若仍 OOM，先降 `--attn-timesteps`。
    "apa": ConditionSpec(
        "apa", "nonadditive", site="apa", defense_mode="suppress_attn_ca",
        lr_key="lr.N4_stage2", align_lr_key="lr.N4_stage1",
        monitor="attn_suppressed",
    ),
    "R": ConditionSpec("R", "random", site="warp"),
    # apa 上的同失真隨機對照。**它也跑階段一的保真對齊**（`align_lr_key`
    # 非空），理由見 `_train_random`：不對齊的話它的 `x_base` 是未對齊的
    # VAE 重建，同一個 τ 之下留給隨機方向的預算比 apa 少，比較就偏向我方。
    #
    # 階段一與 apa 共用 `lr.N4_stage1` 是安全的，與上面階段二的情形相反：
    # `_probe_align_lr` 的判準是**對齊損失**，那是保真度，與防禦模式無關，
    # 兩者會探出同一個值。共用因此省下一組探測而不改變任何數值。
    "Ra": ConditionSpec(
        "Ra", "random", site="apa", align_lr_key="lr.N4_stage1",
    ),
}
for _b in grid.BASELINES:
    CONDITION_SPECS[_b] = ConditionSpec(_b, "baseline")


def condition_spec(name: str) -> ConditionSpec:
    if name not in CONDITION_SPECS:
        raise KeyError(
            f"條件 {name!r} 在計算層沒有定義；已定義的是 "
            f"{sorted(CONDITION_SPECS)}。grid.CONDITIONS 與 CONDITION_SPECS "
            "必須逐鍵對應，否則整段會跑到該條件才失敗"
        )
    return CONDITION_SPECS[name]


# ---------------------------------------------------------------------------
# 資料集
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageEntry:
    """一張待防禦影像。訓練與評測用的攻擊 prompt 一律由 `attack_prompt` 取。

    `group` 是 `prompts.yaml` 的類別（man／dog／…），在報表中作為
    PIE-Bench `subtask` 欄位的對應物——本輪不用 PIE-Bench（遠端機器連不上
    HuggingFace），故分層的單位是這個類別。
    """

    image_id: str
    x01: torch.Tensor
    prompts: Tuple[str, ...]
    content: str
    group: str
    # 本批用 `prompts` 的哪一個。0 = 改掉 c_a（img2img 各批一律如此）、
    # 1 = 保留 c_a 改動別處（Lo 的 inpainting 情境）。逐 entry 帶著而不是
    # 讓各呼叫點各自去查設定：`prompts[0]` 原本寫死在十二處，在其中若干處
    # 改而另幾處沒改，症狀是同一格的訓練與評測用了不同的攻擊 prompt，
    # 而報表上完全看不出來。
    prompt_index: int = 0
    # inpainting 威脅模型的遮罩，(1,1,H,W)、1 表示攻擊方要重畫的區域。
    # img2img 下恆為 None——`SDWrapper.edit` 在兩種形態下各自拒絕對方的參數，
    # 故「這一批有沒有遮罩」不會被靜默沿用。由 `run_stage.build_resources`
    # 在載入模型之後填入（需要模型自己的 cross-attention，見 `data/masks.py`）。
    mask: Optional[torch.Tensor] = None

    @property
    def attack_prompt(self) -> str:
        """本批實際使用的攻擊 prompt。**全部呼叫點只准用這個入口。**

        `tests/test_executors.py::test_攻擊prompt只有一個入口` 以原始碼掃描
        釘住 `executors.py` 裡不再有任何 `prompts[0]`。
        """
        if not 0 <= self.prompt_index < len(self.prompts):
            raise IndexError(
                f"{self.image_id}：prompt_index={self.prompt_index} 超出 "
                f"{len(self.prompts)} 個 prompt 的範圍"
            )
        return self.prompts[self.prompt_index]


def load_lo_aligned(root, size: int, device, ids: Optional[Sequence[str]] = None,
                    n: Optional[int] = None, seed: int = 0,
                    prompt_index: int = 0) -> List[ImageEntry]:
    """讀 `data/lo_aligned/`。收錄的內容恰好等於 `prompts.yaml` 宣告的內容。

    不用 `rglob("*.png")`：根目錄的 `overview.png`（資料集總覽圖）會被當成
    一張待防禦影像，而該錯誤在輸出上沒有症狀。未宣告卻放了 PNG 的子目錄
    一律拒絕——那與「忘了宣告」分不出來。

    `n` 是樣本數的唯一入口（`ARCH` §1 第 2 條）。抽樣依類別輪流取，使前 k
    張必落在 k 個不同類別上；`ids` 明給時覆蓋 `n`。

    `prompt_index` 選本批用哪一個攻擊 prompt：0 = 改掉 c_a、1 = 保留 c_a
    改動別處（Lo 的 inpainting 情境，DEC-010）。它寫進每個 entry，使各
    呼叫點不必各自去查設定。
    """
    import yaml
    from PIL import Image
    import torchvision.transforms as T

    root = Path(root)
    pf = root / "prompts.yaml"
    if not pf.exists():
        raise FileNotFoundError(
            f"{pf} 不存在。prompt 是攻擊方的輸入，缺少時無法定義要防禦什麼"
        )
    spec = yaml.safe_load(pf.read_text(encoding="utf-8")) or {}

    by_group: Dict[str, List[ImageEntry]] = {}
    for cls in sorted(spec):
        entry = spec[cls]
        d = root / cls
        if not d.is_dir():
            raise FileNotFoundError(f"{pf} 宣告了類別 {cls!r}，但 {d} 不存在")
        if "prompts" not in entry or "content" not in entry:
            raise KeyError(
                f"{pf} 的類別 {cls!r} 缺 prompts 或 content；"
                "c_a 是防禦方選的、prompt 是攻擊方寫的，兩者都不接受預設值"
            )
        prompts = tuple(entry["prompts"])
        if not 0 <= prompt_index < len(prompts):
            raise ValueError(
                f"{pf} 的類別 {cls!r} 只有 {len(prompts)} 個 prompt，"
                f"取不到 prompt_index={prompt_index}"
            )
        for p in sorted(d.glob("*.png")):
            img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
            x = T.ToTensor()(img).unsqueeze(0).to(device)
            by_group.setdefault(cls, []).append(ImageEntry(
                image_id=p.stem, x01=x, prompts=prompts,
                content=entry["content"], group=cls,
                prompt_index=prompt_index,
            ))

    stray = [p.name for p in root.iterdir()
             if p.is_dir() and p.name not in spec and any(p.glob("*.png"))]
    if stray:
        raise KeyError(
            f"這些目錄有 PNG 但沒有在 {pf} 裡宣告：{stray}。"
            "靜默忽略與「忘了宣告」分不出來"
        )

    if ids is not None:
        flat = {e.image_id: e for g in by_group.values() for e in g}
        missing = [i for i in ids if i not in flat]
        if missing:
            raise KeyError(
                f"這些影像 id 不在 {root} 裡：{missing}。"
                f"可用的是 {sorted(flat)}"
            )
        return [flat[i] for i in ids]

    ordered = [e for g in by_group.values() for e in g]
    if n is None:
        return ordered
    return _stratified(by_group, n, seed)


def _stratified(by_group: Dict[str, List[ImageEntry]], n: int,
                seed: int) -> List[ImageEntry]:
    """依類別輪流取 n 張。前 k 張必落在 k 個不同類別上。"""
    if n <= 0:
        raise ValueError(f"n 必須為正，收到 {n}")
    import random

    rng = random.Random(seed)
    pool = {k: list(v) for k, v in by_group.items()}
    for v in pool.values():
        rng.shuffle(v)
    total = sum(len(v) for v in pool.values())
    if total < n:
        raise ValueError(f"可用影像 {total} 張少於要求的 n={n}")

    picked: List[ImageEntry] = []
    round_idx = 0
    while len(picked) < n:
        for k in sorted(pool):
            if len(picked) >= n:
                break
            if round_idx < len(pool[k]):
                picked.append(pool[k][round_idx])
        round_idx += 1
    return picked


# ---------------------------------------------------------------------------
# 整批設定
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """整批共用的計算層設定。與 `base_config`（進雜湊的那份）分開：

    `base_config` 是「哪些變因會使既有結果失效」的宣告，本結構是「怎麼算」。
    兩者有重疊（resolution、strength…），但不是同一件事——例如 `log_every`
    改了不該讓 4449 格重跑。凡是會改變數值結果的欄位都必須同時出現在
    `base_config["loss_params"]` 裡，見 `scripts/run_stage.py::base_config`。
    """

    resolution: int = 1024
    guidance: float = 7.5
    strength: float = 0.6
    steps: int = 50               # 攻擊方（評測期）的去噪步數
    seed: int = 20260805

    # ---- 防禦訓練 ----
    train_n_edit: int = 10        # 訓練期代理編輯鏈的步數，與評測期分開
    n_eot: int = 1
    k_inv: int = 10
    t_max: Optional[int] = None
    exact_inversion: bool = False
    purify_mode: str = "all"
    max_steps: int = 250
    align_steps: int = APA_STAGE1_STEPS
    stop_patience: int = 20
    stop_min_steps: int = 25
    attn_timesteps: int = 4
    grad_clip: float = 1.0
    log_every: int = 10
    unet_ckpt: bool = True
    vae_ckpt: bool = True

    # 評測期是否擷取 cross-attention（`CODE` §4.2）。**預設開，且必須維持
    # 預設開**：attention map 是主判準的一部分，AdvPaint、PromptFlare 與 N1
    # 三者都以 attention 為著力點，沒有它就無法說明「防禦是否真的讓那些層
    # 失效」。存在這個旗標的唯一理由是擷取需要真實的 UNet
    # （`CrossAttentionRecorder` 掃 `attn2` 層），而測試用的 SD 替身沒有；
    # 關掉它是測試替身的明示宣告，不是效能選項。
    capture_attn: bool = True

    # ---- 保真約束 ----
    tau_train: float = grid.TRAIN_TAU
    tau_acut: float = LossConfig.tau_acut
    tau_chroma: float = LossConfig.tau_chroma
    beta_linf: float = LossConfig.beta_linf
    tau_linf: float = LossConfig.tau_linf
    # N1 要把注意力質量導向哪些 token 格。**定義 N1 攻擊的是什麼**，
    # 故必須進 `config_hash`（見 `loss_params`）。
    shared_tokens: Tuple[int, ...] = LossConfig.shared_tokens
    # apa（`suppress_attn_ca`）式 (4) 的遮罩門檻，作用在峰值正規化後的尺度上。
    # **`None` 表示本批不用該模式**，此時它不出現在 `loss_params()`，故 v14／
    # v14r 既有格點的 `config_hash` 逐位不變（與 `warp_mask_gate` 同一慣例）。
    attn_mask_tau: Optional[float] = None
    # 取遮罩時要平均幾個 timestep。0 表示沿用 `attn_timesteps`。
    attn_mask_timesteps: int = 0

    # ---- 參數化 ----
    warp_grid_size: int = 32
    warp_max_disp: float = 1.5
    warp_resample: str = "bicubic"
    # 位移場是否在遮罩內歸零（2026-08-08，處置 B）。**只在 inpainting 下
    # 可為 True**；`run_stage` 在沒有 `--mask-mode` 時拒絕這個旗標。
    warp_mask_gate: bool = False
    apa_lora_rank: int = 8
    apa_latent_max_rank: int = 32
    apa_latent_const_rank: int = 8
    random_init_std: float = 0.5

    # ---- A 段：重建下限（DEC-016）----
    # `recon=False` 時以下全部不進 `config_hash`，故既有批次的每一格雜湊
    # 逐位不變（與 `warp_mask_gate`、`attn_mask_tau` 同一慣例，由
    # `tests/test_recon_wiring.py` 釘住）。開啟時它取代 site apa 的階段一。
    recon: bool = False
    recon_objective: str = "lpips"
    recon_a1_steps: int = 200
    recon_a1_lr: float = 0.02
    recon_a2_steps: int = 200
    recon_a2_lr: float = 2e-3
    # A2 的停止目標＝本圖舊下限 × 這個比例。硬停止的理由見 `recon` 模組
    # docstring：解碼器背熟原圖會讓 latent 上的擾動傳不到輸出。
    recon_floor_ratio: float = 0.5
    recon_gamma_acut: float = 1.0
    recon_acut_band: float = 0.05
    # 感知項與逐像素項的相對權重，逐目標校準，不可跨目標沿用。
    recon_w_pixel: float = 0.5
    recon_resp_scale: float = 0.05

    # ---- 本批的格點 ----
    # 這一批實際要跑的條件，以及失真預算軸。**兩者都不進 `config_hash`**：
    # 它們決定「列舉哪些格」，而每一格自己已經帶著 condition 與 tau
    # （`Cell.cell_id`），把批次層級的清單也放進雜湊會讓同一格在兩批之間
    # 算出不同的雜湊而無法續跑。
    #
    # 段 0 需要 `conditions`：`measure_warp_reach` 依它判斷本批有沒有位移場
    # 條件，`recon_floor` 的判定依它判斷有沒有生成路徑條件。段 0 原本完全
    # 不看格點，於是第三階段把隨機對照移到 apa 之後會在指標內部炸開。
    conditions: Tuple[str, ...] = grid.CONDITIONS
    tau_plan: grid.TauPlan = grid.DEFAULT_TAU_PLAN

    # ---- 段 0 ----
    lr_grid: Tuple[float, ...] = LR_GRID
    probe_steps: int = 12
    edit_effect_threshold: float = 0.0

    # ---- 外部檔案 ----
    target_image: str = "data/targets/gray.png"
    mist_target: str = ""         # Mist 的 MIST.png，無則該條件會明確拋出
    diffpure_ckpt: str = ""

    def _loss_params_base(self) -> Dict[str, Any]:
        """定義**損失本身**的欄位：目標、約束門檻、以及損失所走的前向鏈。

        > 2026-08-05 拆分。before：本方法回傳全部旋鈕，`config_hash` 只有
        > `loss_params` 一個必填鍵承載它們。那使「參數化容量」與「最佳化步數」
        > 沒有各自的宣告位置——`cellid.REQUIRED_KEYS` 檢查的是鍵的存在，
        > 呼叫端漏放容量參數不會有任何症狀（A7）。
        > after：拆成 `loss_params` / `module_params` / `optim_params` 三份，
        > 三者皆為 `REQUIRED_KEYS`，缺任一即拋 `ConfigIncomplete`。
        """
        return {
            "train_n_edit": self.train_n_edit,
            "n_eot": self.n_eot,
            "k_inv": self.k_inv,
            "t_max": self.t_max,
            "exact_inversion": self.exact_inversion,
            "purify_mode": self.purify_mode,
            "attn_timesteps": self.attn_timesteps,
            "tau_train": self.tau_train,
            "tau_acut": self.tau_acut,
            "tau_chroma": self.tau_chroma,
            "beta_linf": self.beta_linf,
            "tau_linf": self.tau_linf,
            "target_image": self.target_image,
            # 2026-08-06 補入。before：本欄不在雜湊內，而它決定 N1 的
            # 著力點落在哪一格 token——改了它等於換一個 L_def，舊格卻會
            # 被續跑判定視為完成而靜默沿用。這正是 A7 點名的缺陷型態。
            "shared_tokens": list(self.shared_tokens),
        }

    def loss_params(self) -> Dict[str, Any]:
        """`_loss_params_base` 加上只在啟用時出現的鍵。"""
        out = self._loss_params_base()
        # apa 的兩個鍵**只在啟用時出現**，理由與 `module_params` 的
        # `warp_mask_gate` 逐字相同：`config_hash` 吃整個 dict，無條件加入會
        # 改變 img2img 既有批次的每一格雜湊，續跑時把已完成的格全部判為未完成。
        # 鍵存在與否本身就承載「本批有沒有用式 (5) 這個目標」。
        if self.attn_mask_tau is not None:
            out["attn_mask_tau"] = self.attn_mask_tau
            # 只在遮罩取樣點與施力取樣點不同時才進雜湊。相同時它不是一個
            # 獨立的變因（`attn_timesteps` 已經在表內），多一個鍵只會讓
            # 兩種等價的寫法算出不同的雜湊。
            if self.attn_mask_timesteps:
                out["attn_mask_timesteps"] = self.attn_mask_timesteps
        return out

    def module_params(self) -> Dict[str, Any]:
        """**參數化的容量**。A7 原文點名的「控制點 32 與 128」就在這裡。

        把控制點 32 與 128 的結果合併統計，那個平均正好抹掉要量的效應，
        而輸出看起來完全正常。它們必須讓 `config_hash` 分得開。
        """
        out = {
            "warp_grid_size": self.warp_grid_size,
            "warp_max_disp": self.warp_max_disp,
            "warp_resample": self.warp_resample,
            "apa_lora_rank": self.apa_lora_rank,
            "apa_latent_max_rank": self.apa_latent_max_rank,
            "apa_latent_const_rank": self.apa_latent_const_rank,
            "random_init_std": self.random_init_std,
        }
        # A 段的鍵**只在啟用時出現**，理由與下方的 `warp_mask_gate` 逐字
        # 相同：`config_hash` 吃整個 dict，無條件加入會改變既有批次每一格
        # 的雜湊，續跑時把已完成的格全部判為未完成。鍵存在與否本身就承載
        # 「本批的生成路徑有沒有換過起點與解碼器」。
        if self.recon:
            out["recon"] = {
                "objective": self.recon_objective,
                "a1_steps": self.recon_a1_steps, "a1_lr": self.recon_a1_lr,
                "a2_steps": self.recon_a2_steps, "a2_lr": self.recon_a2_lr,
                "floor_ratio": self.recon_floor_ratio,
                "gamma_acut": self.recon_gamma_acut,
                "acut_band": self.recon_acut_band,
                "w_pixel": self.recon_w_pixel,
            }
        # 遮罩閘的鍵**只在開啟時出現**，理由與 `base_config` 的 `mask` 鍵
        # 相同（`run_stage.base_config`）：`config_hash` 吃整個 dict，無條件
        # 加入會改變 img2img 既有批次的每一格雜湊，續跑時把已完成的格全部
        # 判為未完成。此鍵存在與否本身就承載「位移場有沒有加閘」。
        if self.warp_mask_gate:
            out["warp_mask_gate"] = True
        return out

    def optim_params(self) -> Dict[str, Any]:
        """**最佳化過程**的旋鈕。改了它們，同一個 φ 的解不同。

        學習率不在此：它由校準表決定、逐格記在 `meta.json` 的 `lr` 欄，
        而 `lr` 自己就是 `REQUIRED_KEYS` 的一員。
        """
        return {
            "max_steps": self.max_steps,
            "align_steps": self.align_steps,
            "stop_patience": self.stop_patience,
            "stop_min_steps": self.stop_min_steps,
        }


@dataclass
class Resources:
    """跨格共用的昂貴物件。每批建立一次。"""

    sd: Any
    suite: Any
    batch_dir: Path
    base_config: Dict[str, Any]
    cfg: RunConfig
    images: Dict[str, ImageEntry] = field(default_factory=dict)
    calib: Optional[Calibration] = None
    y_target: Optional[torch.Tensor] = None
    # 評測期重建 x_def 的單格快取。`grid.eval_cells` 的迴圈順序是
    # 條件 → 影像 → τ → 淨化 → 種子，故同一個 (條件, 影像, τ) 的
    # 20 × 5 格連續出現，一格快取即可省下 99% 的重建成本。
    _xdef_cache: Dict[str, Any] = field(default_factory=dict)

    # ---- 查詢 ----

    def image(self, image_id: str) -> ImageEntry:
        if image_id not in self.images:
            raise KeyError(
                f"影像 {image_id!r} 不在本批的清單裡；本批為 "
                f"{sorted(self.images)}"
            )
        return self.images[image_id]

    @property
    def device(self):
        return self.sd.device

    @property
    def calib_context(self) -> Dict[str, Any]:
        """校準表的查詢 context。**必須恰好是 `REQUIRED_CONTEXT`**：

        `Calibration.get` 比對的是完全相等而非子集，多一個鍵會讓查詢全部
        失敗、少一個鍵會讓「校準時多記了一個變因」變成「比對時少檢查一個」。
        """
        missing = [k for k in REQUIRED_CONTEXT if k not in self.base_config]
        if missing:
            raise KeyError(
                f"base_config 缺少校準 context 的必填欄位 {missing}；"
                f"必填為 {list(REQUIRED_CONTEXT)}"
            )
        return {k: self.base_config[k] for k in REQUIRED_CONTEXT}

    def require_calib(self) -> Calibration:
        if self.calib is None:
            from src.utils.calibration import CalibrationMismatch

            raise CalibrationMismatch(
                f"{self.batch_dir / 'calib' / 'calibration.json'} 尚未產生。"
                "請先跑段 0（scripts/run_stage.py calib）。"
                "不會回退到預設學習率——未校準的值沿用正是本專案重複十次的缺陷"
            )
        return self.calib

    # ---- 路徑 ----

    def cell_dir(self, condition: str, image_id: str) -> Path:
        return self.batch_dir / condition / image_id

    def rel(self, path: Path) -> str:
        """相對批次目錄的 posix 路徑。續跑判定用它檢查存在性。"""
        return Path(path).resolve().relative_to(
            self.batch_dir.resolve()).as_posix()


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """欄位取全部列的聯集，依首次出現排序。

    不取第一列的鍵：`optimize` 的 history 在階段切換時會多出欄位，只認第一列
    會靜默丟掉後面的欄位，而 CSV 看起來仍然完整。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def load_image_tensor(path: Path, device, size: Optional[int] = None
                      ) -> torch.Tensor:
    """讀回 PNG 成 (1,3,H,W)、[0,1]。對照側的參照走這條，見模組 docstring。

    `size` 明給時縮放到 `size × size`。**外部素材必須經過這一步**：它們的
    尺寸由檔案決定，與本批的 `--resolution` 無關，而 latent 的邊長是影像邊長
    ÷ 8，兩者不符時錯誤發生在損失函式裡（`mse_loss` 的 broadcast），
    看不出來源是一張沒有縮放的素材。留存的產物（`x_def.png` 等）本來就
    是本批解析度，故預設不縮放。
    """
    from PIL import Image
    import torch.nn.functional as F
    import torchvision.transforms as T

    img = Image.open(path).convert("RGB")
    x = T.ToTensor()(img).unsqueeze(0).to(device)
    if size is not None and x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic",
                          antialias=True).clamp(0, 1)
    return x


def purify_dir_name(kind: str, strength: float) -> str:
    return f"{kind}_{strength:g}"


def make_purifier(kind: str, strength: float, seed: int, res: Resources
                  ) -> Purifier:
    """依 `cell.purify` 建立算子。相依不齊時 `Purifier.available` 為 False，
    呼叫 `evaluate` 會明確拋出——不在此處靜默略過。"""
    opts: Dict[str, Any] = {}
    if kind == "impress":
        opts["sd"] = res.sd
    if kind == "diffpure" and res.cfg.diffpure_ckpt:
        opts["ckpt"] = res.cfg.diffpure_ckpt
    return Purifier(kind, strength, seed=seed, **opts)


def eval_noise_seed(res: Resources, seed_idx: int) -> int:
    """評測噪聲的種子。與訓練錯開 `EVAL_SEED_OFFSET`（見該常數的說明）。

    對照側與防禦側必須用同一條式子，否則兩者的編輯噪聲不同，
    量到的偏移主要來自噪聲差異而非防禦。
    """
    return res.cfg.seed + EVAL_SEED_OFFSET + int(seed_idx)


def _sample_trace(trace: List[torch.Tensor], every: int = X0_TRACE_EVERY
                  ) -> List[torch.Tensor]:
    """`CODE` §4.1 的取樣規則：每 `every` 步一張，固定含第 0 與最後一步。"""
    if not trace:
        return []
    idx = sorted({0, len(trace) - 1} | set(range(0, len(trace), every)))
    return [trace[i] for i in idx]


def write_meta(res: Resources, cell: grid.Cell, path: Path,
               extra: Dict[str, Any]) -> str:
    """每格一份 `meta.json`（`CODE` §4）。回傳該格的 `config_hash`。

    與 `_cells/*.json`（進度的真相來源）重複是刻意的：後者以格點識別碼攤平在
    一個目錄裡，而讀報告的人是從產物目錄看過去的，那裡必須自帶「這批圖是用
    什麼設定產生的」，否則兩者對不起來時無從判斷哪一份是舊的。
    """
    cfg = cell_config(cell, res.base_config)
    chash = config_hash(cfg)
    save_json({"cell_id": cell.cell_id(), "config_hash": chash,
               "config": cfg, **extra}, path)
    return chash


# ---------------------------------------------------------------------------
# φ 的落盤與重建
# ---------------------------------------------------------------------------

PHI_FORMAT = 1


def _module_build_kwargs(condition: str, res: Resources,
                         entry: ImageEntry) -> Dict[str, Any]:
    """建構參數存進 `phi.pt`，使段 2／段 3 不依賴當次的 CLI 參數。

    重建時若改讀當下的設定，換一組 `--warp-grid-size` 再跑段 3 就會用錯誤的
    形狀去載入舊的 state_dict——而 `load_state_dict` 只在形狀不符時才報錯，
    形狀恰好相同時完全沒有症狀。
    """
    spec = condition_spec(condition)
    if spec.site == "warp":
        kw = {
            "site": "warp", "size": res.cfg.resolution,
            "grid_size": res.cfg.warp_grid_size,
            "max_disp": res.cfg.warp_max_disp,
            "resample": res.cfg.warp_resample,
        }
        if res.cfg.warp_mask_gate:
            if entry.mask is None:
                raise ValueError(
                    f"warp_mask_gate 為 True，但影像 {entry.image_id!r} 沒有"
                    "遮罩。閘由遮罩決定，沒有遮罩時靜默不加閘會讓同一批裡"
                    "有些格加閘、有些沒有，而兩者的 config_hash 相同"
                )
            # 存粗網格上的閘（32² = 4 KB）而非整張遮罩（512² = 1 MB）：
            # 實際乘進位移場的就是它，而 `phi.pt` 逐格落盤且全部入版控。
            kw["gate"] = WarpResidual.coarse_gate(
                entry.mask.detach().cpu(), res.cfg.warp_grid_size)
        return kw
    if spec.site == "apa":
        lat = res.sd.latent_shape(res.cfg.resolution, res.cfg.resolution)
        return {
            "site": "apa", "steps": res.cfg.k_inv,
            "latent_size": int(lat[-1]), "latent_channels": int(lat[1]),
            "lora_rank": res.cfg.apa_lora_rank,
            "latent_max_rank": res.cfg.apa_latent_max_rank,
            "latent_const_rank": res.cfg.apa_latent_const_rank,
            "k_inv": res.cfg.k_inv, "t_max": res.cfg.t_max,
            "exact_inversion": res.cfg.exact_inversion,
        }
    raise ValueError(f"條件 {condition!r} 沒有模塊參數化（它是 baseline？）")


def build_module(condition: str, res: Resources, entry: ImageEntry,
                 seed: int, init_std: float = 0.0):
    """依條件建立殘差模塊。**baseline 不走這裡**（它們是像素 δ）。"""
    kw = _module_build_kwargs(condition, res, entry)
    if kw["site"] == "warp":
        return WarpResidual(
            size=kw["size"], grid_size=kw["grid_size"],
            max_disp=kw["max_disp"], resample=kw["resample"],
            init_std=init_std, seed=seed, gate=kw.get("gate"),
        ).to(res.device)
    return build_apa(
        res.sd.unet, steps=kw["steps"], latent_size=kw["latent_size"],
        latent_channels=kw["latent_channels"], lora_rank=kw["lora_rank"],
        latent_max_rank=kw["latent_max_rank"],
        latent_const_rank=kw["latent_const_rank"], seed=seed,
    ).to(res.device)


def rebuild_module(payload: Dict[str, Any], res: Resources):
    """由 `phi.pt` 的內容重建模塊並載入參數。"""
    kw = payload["build"]
    if kw["site"] == "warp":
        mod = WarpResidual(
            size=kw["size"], grid_size=kw["grid_size"],
            max_disp=kw["max_disp"], resample=kw["resample"],
            init_std=0.0, gate=kw.get("gate"),
        ).to(res.device)
    elif kw["site"] == "apa":
        mod = build_apa(
            res.sd.unet, steps=kw["steps"], latent_size=kw["latent_size"],
            latent_channels=kw["latent_channels"], lora_rank=kw["lora_rank"],
            latent_max_rank=kw["latent_max_rank"],
            latent_const_rank=kw["latent_const_rank"],
        ).to(res.device)
    else:
        raise ValueError(f"未知的參數化 {kw['site']!r}")
    mod.load_state_dict({k: v.to(res.device)
                         for k, v in payload["state_dict"].items()})
    return mod


def direction_param(module) -> torch.nn.Parameter:
    """射線縮放要乘的那個參數。

    **不是「全部參數乘 k」。** N3 的階段一（LoRA）是保真對齊的結果，把它一起
    縮放等於在改變重建本身；要縮放的只有階段二的 latent 偏移，即攻擊的那一半。
    位移場沒有這個切分，整個 `flow` 就是方向。
    """
    if isinstance(module, WarpResidual):
        return module.flow
    if hasattr(module, "members"):
        return module.members[1].tensor.V
    raise TypeError(
        f"{type(module).__name__} 沒有定義射線縮放的方向參數；"
        "新增參數化時必須在此明寫要縮放哪一個，不得預設為全部參數"
    )


def save_phi(path: Path, condition: str, image_id: str, res: Resources,
             entry: ImageEntry, module=None,
             delta01: Optional[torch.Tensor] = None, recon=None,
             extra: Optional[Dict[str, Any]] = None) -> Path:
    """統一的 φ 落盤格式。加性（baseline／R 的像素形式）走 `delta01`。

    A 段的產物（`recon.ReconAdapter`）跟著 φ 一起存，**不另開一個檔**：
    段 2 的射線縮放與段 3 的評測拿到 payload 就拿到 `z*` 與那組解碼器參數，
    不可能與訓練期用到不同的重建起點。多出來的體積是 16 KB 的 `z*` 加上
    143 KB 的解碼器參數。
    """
    if (module is None) == (delta01 is None):
        raise ValueError("save_phi 必須且只能給定 module 與 delta01 其中之一")
    payload: Dict[str, Any] = {
        "format": PHI_FORMAT, "condition": condition, "image_id": image_id,
        "scale_k": 1.0, **(extra or {}),
    }
    if module is not None:
        payload["parameterization"] = _module_build_kwargs(
            condition, res, entry)["site"]
        payload["build"] = _module_build_kwargs(condition, res, entry)
        payload["state_dict"] = {k: v.detach().cpu()
                                 for k, v in module.state_dict().items()}
    else:
        payload["parameterization"] = "additive"
        payload["build"] = {"site": "additive"}
        payload["delta01"] = delta01.detach().cpu()
    if recon is not None:
        payload["recon"] = recon.to_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_phi(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != PHI_FORMAT:
        raise ValueError(
            f"{path} 的 φ 格式為 {payload.get('format')!r}，本程式只讀 "
            f"{PHI_FORMAT}。舊格式不自動轉換——轉錯不會有症狀"
        )
    return payload


@torch.no_grad()
def materialize(payload: Dict[str, Any], res: Resources, entry: ImageEntry,
                k: float = 1.0) -> torch.Tensor:
    """由 φ 產生 x_def，方向參數乘上 k。**只有前向、無梯度。**

    三種參數化的縮放意義不同，故不能統一寫成影像空間的線性內插：

    - 加性：`clip(x + k·δ)`，精確。
    - 位移場：把**位移量**乘上 k 再重新取樣。影像空間的線性內插是交叉淡入，
      只在小位移下是它的一階近似（見 `ray_scale.solve_k` 的 docstring）。
    - APA：把 latent 偏移乘上 k 後重跑一次生成。
    """
    if payload["parameterization"] == "additive":
        delta = payload["delta01"].to(res.device)
        return (entry.x01 + k * delta).clamp(0, 1)

    module = rebuild_module(payload, res)
    try:
        p = direction_param(module)
        p.data = p.data * float(k)
        if payload["parameterization"] == "warp":
            out = module.pixel_residual(entry.x01)
            if out is None:
                raise RuntimeError("位移場模塊沒有回傳像素側輸出")
            return out.detach()
        kw = payload["build"]
        # A 段的產物若在 payload 內就一定要用上：`x_def` 是拿它算出來的，
        # 少了它重建的是另一條路徑上的圖，而那個差異看起來只像是「重建差
        # 一點」。沒有這個鍵時 `recon=None`，行為與先前逐位元相同。
        recon = None
        if "recon" in payload:
            from src.defense.recon import ReconAdapter
            recon = ReconAdapter.from_payload(payload["recon"]).to(res.device)
        gen = DefenseGenerator(res.sd, module, k_inv=kw["k_inv"],
                               t_max=kw["t_max"],
                               exact_inversion=kw["exact_inversion"],
                               recon=recon)
        return gen.generate(entry.x01, gen.prepare(entry.x01)).detach()
    finally:
        module.remove()


def scaled_payload(payload: Dict[str, Any], res: Resources, k: float
                   ) -> Dict[str, Any]:
    """回傳方向參數已乘上 k 的新 φ。原 payload 不變。"""
    out = copy.deepcopy(payload)
    out["scale_k"] = float(payload.get("scale_k", 1.0)) * float(k)
    if out["parameterization"] == "additive":
        out["delta01"] = out["delta01"] * float(k)
        return out
    module = rebuild_module(payload, res)
    try:
        p = direction_param(module)
        with torch.no_grad():
            p.data = p.data * float(k)
        out["state_dict"] = {kk: v.detach().cpu()
                             for kk, v in module.state_dict().items()}
    finally:
        module.remove()
    return out


# ---------------------------------------------------------------------------
# 段 1：訓練
# ---------------------------------------------------------------------------


def loss_config(res: Resources, spec: ConditionSpec,
                entry: Optional[ImageEntry] = None) -> LossConfig:
    """訓練期的損失設定。τ 取 `grid.TRAIN_TAU`（最大預算），其餘 τ 由段 2 取得。

    `tau_acut` 與 `tau_chroma` 沿用 `LossConfig` 的預設值，而那兩個值是在
    τ_lpips = 0.05 的量級上由人眼判讀定出的**絕對值**（見 `objective.py` 的
    「門檻的適用範圍」）。本輪訓練在 0.35，兩者理應重新判讀。程式不代為推測，
    改由 `RunConfig.tau_acut` / `tau_chroma` 明給並進入 `config_hash`：
    改了值即視為另一組實驗，不會靜默沿用。**這一項需要主 session 裁決。**

    `entry` 只在 `suppress_attn_ca` 下被用到，取其 `content`（Lo et al. 的
    c_a，來自 `data/lo_aligned/prompts.yaml` 的 `content` 欄）。它**逐影像
    不同**，故不能像 `shared_tokens` 那樣放進整批共用的 `RunConfig`。
    漏傳 `entry` 時 `LossConfig.__post_init__` 會以「需要 content」中止，
    不會靜默落回空字串——空 c_a 的 span 是空區間，式 (5) 沒有定義。

    `defense_mode` 為空字串的條件（`R`／`Ra` 這兩個隨機對照）不建立損失，
    此處回傳的物件只供保真門檻使用，故仍走 `LossConfig` 的預設模式。
    """
    mode = spec.defense_mode or LossConfig.defense_mode
    content = ""
    attn_mask_tau = None
    if mode == "suppress_attn_ca":
        if entry is None:
            raise ValueError(
                f"條件 {spec.name!r} 走 suppress_attn_ca，需要 entry 才能取得 "
                "c_a（ImageEntry.content）。c_a 逐影像不同，不可由整批共用的 "
                "RunConfig 提供，也不可由 prompt 推導"
            )
        content = entry.content
        attn_mask_tau = res.cfg.attn_mask_tau
    return LossConfig(
        defense_mode=mode,
        target_metric=spec.target_metric,
        tau_lpips=res.cfg.tau_train,
        tau_acut=res.cfg.tau_acut,
        tau_chroma=res.cfg.tau_chroma,
        beta_linf=res.cfg.beta_linf,
        tau_linf=res.cfg.tau_linf,
        shared_tokens=tuple(res.cfg.shared_tokens),
        content=content,
        attn_mask_tau=attn_mask_tau,
    )


def optim_config(res: Resources, spec: ConditionSpec,
                 entry: Optional[ImageEntry] = None,
                 recon=None) -> OptimConfig:
    """訓練期的優化設定。

    `entry` 只用來取該影像的 inpainting 遮罩。訓練期的代理編輯鏈必須與評測期
    走同一種攻擊——分歧的症狀是防禦對著另一個威脅模型最佳化，而報表上兩者
    都叫「編輯」。img2img 下遮罩為 None，`SDWrapper.edit` 會在收到遮罩時拋出，
    故漏傳不會靜默通過。

    `stop_tol` **一律**向校準表索取，不分監看量。

    > 2026-08-05 修正。before：只有 `shared_mass` 向校準表索取，`edit_shift`
    > 走 `optimize.MONITOR_TOL` 的 1e-4——那是 SD v1.4／512²／site PF 的實測
    > 值，靜默沿用到 SDXL／1024²／位移場。門檻過嚴會在第一個觀察窗就回報
    > 「已收斂」、過鬆等於沒開停止準則，兩者都沒有症狀。
    > after：兩個監看量同權，未校準即拋 `CalibrationMismatch`。
    """
    stop_tol = None
    if spec.monitor:
        stop_tol = float(res.require_calib().get(f"stop_tol.{spec.monitor}",
                                                 res.calib_context))
    align_steps = res.cfg.align_steps if spec.align_lr_key else 0
    # A 段取代階段一：兩者的目的相同（把 G(x; φ=0) 拉近原圖），但階段一的
    # LoRA 在 UNet 上、重建誤差在 VAE 上，實測 200 步只移動 LPIPS 0.0015
    # 卻花 910 秒（FND-016）。留著它跑等於每格白燒那段時間。
    if res.cfg.recon and spec.site == "apa":
        align_steps = 0
    return OptimConfig(
        recon=recon,
        stages=(StageSpec(group=("stage2" if spec.site == "apa" else "default"),
                          lr_key=spec.lr_key, max_steps=res.cfg.max_steps),),
        k_inv=res.cfg.k_inv,
        t_max=res.cfg.t_max,
        exact_inversion=res.cfg.exact_inversion,
        n_edit=res.cfg.train_n_edit,
        n_eot=res.cfg.n_eot,
        purify_mode=res.cfg.purify_mode,
        align_steps=align_steps,
        align_lr_key=spec.align_lr_key,
        align_group="stage1" if spec.site == "apa" else "default",
        strength=(None if (entry is not None and entry.mask is not None)
                  else res.cfg.strength),
        edit_mask=(entry.mask if entry is not None else None),
        warp_max_disp=res.cfg.warp_max_disp,
        warp_resample=res.cfg.warp_resample,
        guidance_scale=res.cfg.guidance,
        stop_tol=stop_tol,
        stop_patience=res.cfg.stop_patience,
        stop_min_steps=res.cfg.stop_min_steps,
        attn_timesteps=res.cfg.attn_timesteps,
        attn_mask_timesteps=res.cfg.attn_mask_timesteps,
        prompt_def="",
        seed=res.cfg.seed,
        unet_ckpt=res.cfg.unet_ckpt and spec.unet_ckpt,
        vae_ckpt=res.cfg.vae_ckpt,
        log_every=res.cfg.log_every,
        grad_clip=res.cfg.grad_clip,
    )


def baseline_kwargs(name: str, res: Resources, entry: ImageEntry
                    ) -> Dict[str, Any]:
    """五篇各自的 `prepare` 需要的東西。

    **查不到的項目不填看起來合理的值。** 各篇的 `prepare` 對缺項會拋
    `NotImplementedError` 並寫明缺什麼，本函式只負責把本專案威脅模型明確
    決定的那幾項傳進去：

    - `strength`：三篇原作是 inpainting、原始碼沒有這個數，由本專案的威脅
      模型（img2img／SDEdit）指定，報表須標為我方設定。
    - `prompt`：AdvPaint 的攻擊 prompt 原始碼沒有預設。本專案是 prompt-free
      （`DESIGN` §2.1），故給空字串，同樣是我方設定。
    - `target01`：Mist 需要 MIST.png，該檔無法由描述重建。未給時 `prepare`
      會拋出——**不可用 PhotoGuard 的零張量代用**（`SOURCE_AUDIT` §3.4）。
    """
    kw: Dict[str, Any] = {}
    if name in ("photoguard_c", "advpaint", "promptflare"):
        # 這三篇原作**就是 inpainting**。遮罩給定時它們回到原生形態，
        # `strength` 隨之消失——那個參數在原始碼裡本來就不存在，是本專案
        # 為了移植到 img2img 才引入的（`photoguard.py:124` 等三處各自拒絕
        # 預設值並寫明理由）。兩者一起切換，不留一個不起作用的值。
        kw["mask"] = entry.mask
        kw["strength"] = None if entry.mask is not None else res.cfg.strength
    if name == "advpaint":
        kw["prompt"] = ""
        kw["guidance_scale"] = res.cfg.guidance
    if name.startswith("dia_"):
        # DIA 把整條反演（DIA-R 再加整條重建）留在同一張圖上，1024² 下必須
        # checkpoint 才放得下。數值中性，故沿用本批的 `unet_ckpt`，不新增旗標。
        kw["use_ckpt"] = res.cfg.unet_ckpt
        # 2026-08-07 補：只包 UNet 仍不夠，dia_r 改在 `sd.vae.decode` OOM。
        # 圖上另有一次 VAE 編碼與一次 VAE 解碼，1024² 下兩者的激活同時留存。
        kw["vae_ckpt"] = res.cfg.vae_ckpt
    if name == "mist":
        # mist 的 fused mode 同樣把兩次 VAE 編碼與一次完整 UNet 前向放在
        # 同一張圖上。2026-08-07 實測只包 UNet 不夠，主導成本是那兩次
        # 1024² 的 VAE 編碼，故兩個開關都要給。取樣的重參數化寫法見
        # `mist._encode_sampled`。
        kw["use_ckpt"] = res.cfg.unet_ckpt
        kw["vae_ckpt"] = res.cfg.vae_ckpt
    if name == "mist" and res.cfg.mist_target:
        # MIST.png 是 1440×1440 的固定素材（`mist.py` 的 NotImplementedError
        # 記其規格），與本批的 1024² 不符。
        #
        # 2026-08-06 修正。before：`load_image_tensor(path, res.device)`，
        # 不縮放。症狀是 `mist.loss_fn` 的 `mse_sum(zx, ctx.z_target)` 以
        # `RuntimeError: The size of tensor a (128) must match the size of
        # tensor b (180)` 中止——1440 ÷ 8 = 180，1024 ÷ 8 = 128。
        # `--target-image` 走的 `build_resources` 一直都有這段縮放，
        # `--mist-target` 是另一個入口而漏了，故兩張圖的 mist 格全數失敗。
        kw["target01"] = load_image_tensor(
            Path(res.cfg.mist_target), res.device, size=res.cfg.resolution)
    return kw


def solve_recon(res: Resources, entry: ImageEntry, out_dir: Path
                ) -> Tuple[object, Path, Dict[str, Any]]:
    """跑 A 段並落盤，回傳 `(adapter, recon.csv 路徑, summary)`。

    只有走生成路徑的條件會呼叫本函式（site apa）。像素側的 `build(0)` 就是
    原圖，沒有重建下限可壓。

    `adapter` 之後會跟著 `phi.pt` 一起存（`save_phi`），故段 2 的射線縮放與
    段 3 的評測不需要重跑 A 段，也不可能與訓練期用到不同的 `z*`。
    """
    from src.defense import recon as reconmod

    key = res.cfg.recon_objective
    perceptual = {"lpips": res.suite.lpips_module,
                  "dists": res.suite.dists_module}[key]

    def measure(y):
        return res.suite.pairwise(entry.x01, y)

    t0 = time.perf_counter()
    adapter, history, summary = reconmod.solve(
        res.sd, entry.x01, perceptual, measure, res.suite.pairwise, key=key,
        a1_steps=res.cfg.recon_a1_steps, a1_lr=res.cfg.recon_a1_lr,
        a2_steps=res.cfg.recon_a2_steps, a2_lr=res.cfg.recon_a2_lr,
        floor_ratio=res.cfg.recon_floor_ratio, w_pixel=res.cfg.recon_w_pixel,
        gamma_acut=res.cfg.recon_gamma_acut, acut_band=res.cfg.recon_acut_band,
        resp_seed=res.cfg.seed, resp_scale=res.cfg.recon_resp_scale,
        log_every=res.cfg.log_every)
    summary["recon_seconds"] = round(time.perf_counter() - t0, 2)
    path = write_csv(out_dir / "recon.csv", [dict(h) for h in history])
    return adapter.to(res.device), path, summary


def _train_nonadditive(cell: grid.Cell, res: Resources, out_dir: Path
                       ) -> Tuple[List[str], Dict[str, Any]]:
    spec = condition_spec(cell.condition)
    entry = res.image(cell.image_id)
    module = build_module(cell.condition, res, entry, seed=res.cfg.seed)
    recon = recon_csv = None
    recon_summary: Dict[str, Any] = {}
    if res.cfg.recon and spec.site == "apa":
        recon, recon_csv, recon_summary = solve_recon(res, entry, out_dir)
    try:
        result = optimize(
            res.sd, module, entry.x01, optim_config(res, spec, entry, recon),
            loss_config(res, spec, entry), default_train_set(),
            calib=res.require_calib(), calib_context=res.calib_context,
            y_target=(res.y_target if spec.defense_mode == "targeted_output"
                      else None),
        )
        x_def = result.x_def
        if x_def is None:
            raise RuntimeError(
                f"{cell.cell_id()}：optimize 沒有回傳 x_def，該格沒有防禦圖"
            )
        arts = [save_phi(out_dir / "phi.pt", cell.condition, cell.image_id,
                         res, entry, module=module, recon=recon)]
        if recon_csv is not None:
            arts.append(recon_csv)
        rows = [dict(h) for h in result.history]
        arts.append(write_csv(out_dir / "train.csv", rows))
        if result.align_history:
            arts.append(write_csv(out_dir / "align.csv",
                                  [dict(h) for h in result.align_history]))
            save_image(result.x_base, out_dir / "aligned_phi.png")
        save_history_plot(rows, out_dir / "history.png",
                          title=cell.cell_id())
        arts.append(out_dir / "history.png")

        trace = _sample_trace(result.x0_trace)
        if trace:
            save_x0_trace(trace, res.sd, out_dir / "x0_trace")
            arts.append(out_dir / "x0_trace")

        last = rows[-1] if rows else {}
        extra = {
            "steps_used": result.steps_done,
            "stop_reason": result.stop_reason,
            "stage_reports": result.stage_reports,
            "align_seconds": round(result.align_seconds, 2),
            "seconds_optimize": round(result.seconds, 2),
            **recon_summary,
            "lr": (result.stage_reports[-1]["lr"]
                   if result.stage_reports else None),
            "final_L_def": last.get("L_def"),
            "final_L_fid": last.get("L_fid"),
            "final_lpips": last.get("fid_lpips"),
            "final_edit_shift": last.get("edit_shift"),
        }
        # `suppress_attn_ca` 專用的三個量。**必須逐格落盤**：
        # `attn_suppressed` 是「壓下多少」，而它的量級取決於損失實際算在
        # 多大的區域上——`attn_mask_coverage` 是那塊區域佔格點的比例、
        # `attn_ref_l1` 是起點值、`attn_mask_kept`（僅 inpainting）是 M 有
        # 多少比例落在攻擊方遮罩外（DEC-012）。缺了它們，兩張影像之間的
        # `attn_suppressed` 不可比，而報表上看不出這件事。
        for k in ("attn_mask_coverage", "attn_ref_l1", "attn_mask_kept"):
            v = getattr(result, k, None)
            if v is not None:
                extra[k] = v
        if hasattr(module, "disp_stats"):
            extra.update(module.disp_stats())
        imgs, gain = _save_train_images(res, entry, x_def, out_dir,
                                        x_base=result.x_base)
        extra["residual_gain"] = gain
        return _finish_train(res, cell, out_dir, arts + imgs, extra, x_def)
    finally:
        # site W 的 hook 註冊在 SD 的模組上，模塊被回收不會移除它們。
        # 殘留的 hook 會污染後續每一格，症狀是「別的條件的結果莫名被改動」。
        module.remove()


def _train_random(cell: grid.Cell, res: Resources, out_dir: Path
                  ) -> Tuple[List[str], Dict[str, Any]]:
    """R：同參數化、參數取高斯隨機，不最佳化。

    `DESIGN` §6.3 (b) 要求「最佳化取得了多少，超過同樣形狀的隨機擾動」，
    故 R 走**與被比較的非加性條件相同的參數化**（位移場），而不是加性雜訊
    ——後者會混入參數化本身的效果。

    種子由影像 id 的 CRC32 決定，使同一張影像在任何一次執行都得到同一個
    隨機方向，且不同影像彼此獨立。

    **已知偏離**：`ray_scale.gaussian_control` 建議「種子隨失真等級而變」，
    否則各 τ 之間相關。本格點結構是「訓練一次、射線縮放到四個 τ」，四個
    τ 因此共用同一個隨機方向。要改成逐 τ 獨立抽樣，得讓 R 在段 2 各自抽，
    那會使 R 與其他條件走不同的流程。**此項需主 session 裁決。**

    ## 兩個 site 的隨機化位置不同（2026-08-09）

    位移場（`R`）的隨機來自 `WarpResidual(init_std=…)`，即建構時就把 `flow`
    抽成高斯，之後直接取 `pixel_residual`。

    site apa（`Ra`）不能照做，三個理由：

    1. **`build_apa` 不吃 `init_std`。** 它的兩個因子刻意「一半高斯、一半零」
       （LoRA 慣例），故 φ 在建構後恆為零殘差——`pixel_residual` 回傳 `None`，
       `x_def` 會是 `None` 而不是一張圖。要隨機的是**射線縮放要乘的那個方向
       參數**，即 `direction_param`（階段二的 latent V），不是整包參數。
    2. **它走生成路徑**，`x_def = G(x; φ)` 要經 inversion、去噪與 VAE 解碼，
       沒有像素側的捷徑。
    3. **它必須跑階段一的保真對齊。** 不對齊的話 `x_base` 是未對齊的 VAE
       重建，而段 2 的 τ 是對**原圖**求解的：同一個 τ=0.50 之下，未對齊的
       Ra 得先付掉更大的重建誤差，留給隨機方向的預算比 apa 少。那會讓對照
       系統性地偏弱，也就讓 apa 的任何勝出不可解讀——而這條對照存在的唯一
       理由就是判斷「有沒有勝過隨便擾動一下」。
    """
    from src.defense.generator import DefenseGenerator
    from src.defense.optimize import align

    spec = condition_spec(cell.condition)
    entry = res.image(cell.image_id)
    seed = res.cfg.seed + zlib.crc32(cell.image_id.encode("utf-8"))
    module = build_module(cell.condition, res, entry, seed=seed,
                          init_std=res.cfg.random_init_std)
    extra: Dict[str, Any] = {
        "steps_used": 0, "stop_reason": "random control：無最佳化",
        "random_seed": seed, "lr": None,
    }
    rows = [{"step": 0, "note": "random control：無最佳化",
             "random_seed": seed, "init_std": res.cfg.random_init_std,
             "site": spec.site}]
    recon = recon_csv = None
    try:
        if spec.site == "warp":
            with torch.no_grad():
                x_def = module.pixel_residual(entry.x01)
        else:
            # A 段對 `Ra` 與 `apa` 逐字相同，理由與階段一那條相同（docstring
            # 第 3 點）：兩者的 `x_base` 必須走在同一條重建基準上，否則同一個
            # τ 之下留給隨機方向的預算比 apa 少，比較系統性地偏向我方。
            if res.cfg.recon and spec.site == "apa":
                recon, recon_csv, recon_summary = solve_recon(res, entry,
                                                              out_dir)
                extra.update(recon_summary)
            optim_cfg = optim_config(res, spec, entry, recon)
            loss_cfg = loss_config(res, spec, entry)
            gen = DefenseGenerator(
                res.sd, module, k_inv=optim_cfg.k_inv, t_max=optim_cfg.t_max,
                exact_inversion=optim_cfg.exact_inversion, recon=recon)
            # 階段一：與 apa 逐字相同的保真對齊（見 docstring 第 3 點）。
            if optim_cfg.align_steps > 0:
                with torch.no_grad():
                    module.disable()
                    x_base0 = gen.generate(
                        entry.x01, gen.prepare(entry.x01,
                                               prompt_def=optim_cfg.prompt_def)
                    ).detach()
                    module.enable()
                _, align_hist = align(
                    res.sd, module, entry.x01, optim_cfg, loss_cfg, gen,
                    res.require_calib(), res.calib_context, x_base0)
                rows.extend(align_hist)
                extra["align_steps_used"] = len(align_hist)
            # 階段二：只把**方向參數**抽成高斯。其餘參數（階段一的 LoRA）
            # 是對齊的結果，動它等於在改變重建本身，那正是 `direction_param`
            # 的 docstring 指明不可做的事。
            gd = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                p = direction_param(module)
                p.data = (torch.randn(p.shape, generator=gd,
                                      dtype=torch.float32)
                          .to(p.device, p.dtype) * res.cfg.random_init_std)
                x_def = gen.generate(
                    entry.x01,
                    gen.prepare(entry.x01, prompt_def=optim_cfg.prompt_def),
                ).detach()

        if x_def is None:
            raise RuntimeError(
                f"{cell.cell_id()}：隨機對照沒有產生 x_def。"
                f"site={spec.site!r} 的模塊既沒有像素側輸出也沒有走生成路徑"
            )
        arts = [save_phi(out_dir / "phi.pt", cell.condition, cell.image_id,
                         res, entry, module=module, recon=recon,
                         extra={"random_seed": seed})]
        if recon_csv is not None:
            arts.append(recon_csv)
        arts.append(write_csv(out_dir / "train.csv", rows))
        if hasattr(module, "disp_stats"):
            extra.update(module.disp_stats())
        imgs, gain = _save_train_images(res, entry, x_def, out_dir)
        extra["residual_gain"] = gain
        return _finish_train(res, cell, out_dir, arts + imgs, extra, x_def)
    finally:
        module.remove()


def _train_baseline(cell: grid.Cell, res: Resources, out_dir: Path
                    ) -> Tuple[List[str], Dict[str, Any]]:
    entry = res.image(cell.image_id)
    spec = BASELINE_REGISTRY[cell.condition]
    # PhotoGuard-c 的 `attack_forward` 逐字為 `grad = grad * (1 - cur_mask)`：
    # 擾動只落在**不會被重繪**的區域。那是它在 inpainting 下的忠實項，也是
    # 唯一有意義的放置處——遮罩內的像素攻擊方會整片覆寫掉。其餘四篇的原始碼
    # 沒有這一步，故只有本篇傳 `grad_mask`；靠 `spec.grad_outside_mask` 分派而不是
    # 比對名稱，與 `generator.py` 依能力分派同一原則。
    result = run_pgd(
        res.sd, entry.x01, spec, seed=res.cfg.seed,
        log_every=res.cfg.log_every,
        grad_mask=(entry.mask if spec.grad_outside_mask else None),
        **baseline_kwargs(cell.condition, res, entry),
    )
    arts = [save_phi(out_dir / "phi.pt", cell.condition, cell.image_id,
                     res, entry, delta01=result.delta01)]
    arts.append(write_csv(out_dir / "train.csv",
                          [dict(h) for h in result.history]))
    extra = {
        "steps_used": spec.steps,
        "stop_reason": f"{spec.name}：步數照原論文（{spec.steps}）",
        "seconds_optimize": round(result.seconds, 2),
        "lr": None,
        "modified_from_paper": spec.modified_from_paper,
        "modification_note": spec.modification_note,
        "discrepancy_note": spec.discrepancy_note,
        "eps_pixel01": spec.eps_pixel01,
        "final_delta_linf01": (result.history[-1]["delta_linf01"]
                               if result.history else None),
    }
    imgs, gain = _save_train_images(res, entry, result.x_adv01, out_dir)
    extra["residual_gain"] = gain
    return _finish_train(res, cell, out_dir, arts + imgs, extra,
                         result.x_adv01)


def _save_train_images(res: Resources, entry: ImageEntry,
                       x_def: torch.Tensor, out_dir: Path,
                       x_base: Optional[torch.Tensor] = None
                       ) -> Tuple[List[Path], float]:
    """spec §8.3 要求留存的影像。殘差一律標註放大倍率，回傳實際倍率。

    倍率必須進 `meta.json`：殘差圖的量級通常在 1e-2 以下，不放大則整張近乎
    全灰；放大而不記錄倍率，兩張不同倍率的圖會被讀成同一尺度。
    """
    save_image(entry.x01, out_dir / "orig.png")
    save_image(x_def, out_dir / "x_def.png")
    gain = save_residual(x_def - entry.x01, out_dir / "residual.png")
    paths = [out_dir / "orig.png", out_dir / "x_def.png",
             out_dir / "residual.png"]
    if x_base is not None and not torch.equal(x_base, entry.x01):
        # 生成路徑的 φ=0 重建。留存它，讀者才分得清哪些失真來自防禦、
        # 哪些是這個注入位置本來就有的重建誤差。
        save_image(x_base, out_dir / "baseline_phi0.png")
        paths.append(out_dir / "baseline_phi0.png")
    return paths, gain


def _finish_train(res: Resources, cell: grid.Cell, out_dir: Path,
                  arts: List[Path], extra: Dict[str, Any],
                  x_def: torch.Tensor) -> Tuple[List[str], Dict[str, Any]]:
    """共用的收尾：量保真度、寫 `meta.json`、把路徑轉成相對批次目錄。

    保真度以**記憶體中的 x_def**量測而非讀回 PNG：訓練側的 φ 是浮點的，
    存檔的 8-bit 只是給人看的。評測側才必須讀 PNG（見模組 docstring）。
    """
    entry = res.image(cell.image_id)
    fid = res.suite.pairwise(entry.x01, x_def)
    extra = dict(extra)
    extra.update({f"fid_{k}": v for k, v in fid.items()})
    extra["fid_niqe"] = res.suite.niqe(x_def)
    extra["group"] = entry.group
    extra["prompt"] = entry.attack_prompt
    write_meta(res, cell, out_dir / "meta.json", extra)
    # 訓練換了 φ，段 3 的 x_def 快取必須失效，否則同一批次內先跑 eval
    # 再重跑 train 會拿到上一版的防禦圖。
    res._xdef_cache.clear()
    arts = list(arts) + [out_dir / "meta.json"]
    return [res.rel(p) for p in arts], extra


def train_executor(cell: grid.Cell, ctx: Dict[str, Any]
                   ) -> Tuple[List[str], Dict[str, Any]]:
    res: Resources = ctx["res"]
    # 段 1 全程不用語意指標，而它接下來要建的訓練圖是全案最吃顯存的一段。
    # 理由與 `calibrate_lr` 同，見 `MetricSuite.release_vlm`。
    res.suite.release_vlm()
    out_dir = res.cell_dir(cell.condition, cell.image_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = condition_spec(cell.condition).kind
    if kind == "nonadditive":
        return _train_nonadditive(cell, res, out_dir)
    if kind == "random":
        return _train_random(cell, res, out_dir)
    return _train_baseline(cell, res, out_dir)


# ---------------------------------------------------------------------------
# 段 2：射線縮放
# ---------------------------------------------------------------------------


def rayscale_executor(cell: grid.Cell, ctx: Dict[str, Any]
                      ) -> Tuple[List[str], Dict[str, Any]]:
    """把訓練好的 φ 沿參數射線縮放到 `cell.tau`。只有前向。

    `solve_k` 到不了目標時拋 `ValueError`，本函式**不接住**：「達不到
    τ=0.35」與「達到了」在下游完全不同，靜默取最接近的值會讓那一格的失真
    宣告與實際不符，而整張匹配失真的比較表都建立在該宣告上。
    """
    res: Resources = ctx["res"]
    entry = res.image(cell.image_id)
    out_dir = res.cell_dir(cell.condition, cell.image_id)
    phi_path = out_dir / "phi.pt"
    if not phi_path.exists():
        raise FileNotFoundError(
            f"{phi_path} 不存在：段 2 需要段 1 的產物。"
            "請先跑 `run_stage.py train --batch <批次>`"
        )
    payload = load_phi(phi_path)
    tau = float(cell.tau)
    tag = f"tau{tau:g}"

    plan = res.cfg.tau_plan
    build = lambda kk: materialize(payload, res, entry, kk)  # noqa: E731
    metric_fn = metric_against(res.suite, entry.x01, plan.metric)

    # 相對預算軸（`grid.budget_tau_plan`）：原點是這一格自己的 `build(0)`，
    # 不是原圖。加性位置的 `build(0)` 就是原圖，故 floor≈0，同一條規則不必
    # 分兩種寫法；生成路徑的 floor 是它的 VAE 來回下限，減掉之後兩類位置
    # 比的是**同一份可見的增量**。
    floor = metric_fn(build(0.0)) if plan.relative else 0.0
    target = tau + floor

    t0 = time.perf_counter()
    x_tau, got, k = solve_k(metric_fn, build, target)
    seconds = time.perf_counter() - t0

    out = scaled_payload(payload, res, k)
    torch.save(out, out_dir / f"phi_{tag}.pt")
    save_image(x_tau, out_dir / f"x_def_{tag}.png")
    save_residual(x_tau - entry.x01, out_dir / f"residual_{tag}.png")

    fid = res.suite.pairwise(entry.x01, x_tau)
    row = {"condition": cell.condition, "image_id": cell.image_id,
           "tau_target": tau, "tau_achieved": got - floor, "scale_k": k,
           "budget_metric": plan.metric, "budget_relative": plan.relative,
           "budget_floor": round(floor, 6), "budget_target_abs": target,
           "budget_achieved_abs": got,
           "niqe": res.suite.niqe(x_tau),
           **{f"fid_{kk}": v for kk, v in fid.items()}}
    if payload["parameterization"] == "warp":
        module = rebuild_module(out, res)
        try:
            row.update(module.disp_stats())
        finally:
            module.remove()
    write_csv(out_dir / f"fidelity_{tag}.csv", [row])

    extra = dict(row)
    extra["seconds_rayscale"] = round(seconds, 2)
    write_meta(res, cell, out_dir / f"meta_{tag}.json", extra)
    res._xdef_cache.clear()

    arts = [out_dir / f"phi_{tag}.pt", out_dir / f"x_def_{tag}.png",
            out_dir / f"residual_{tag}.png",
            out_dir / f"fidelity_{tag}.csv", out_dir / f"meta_{tag}.json"]
    return [res.rel(p) for p in arts], extra


# ---------------------------------------------------------------------------
# 段 3：φ=0 對照與評測
# ---------------------------------------------------------------------------


def _sdedit(res: Resources, x: torch.Tensor, prompt: str, seed_idx: int,
            attn_dir: Optional[Path] = None, attn_tag: str = "",
            attn_full: bool = False,
            mask: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, List[Path]]:
    """攻擊方的編輯。兩側必須逐元素共用同一組噪聲，故噪聲由種子決定而非現抽。

    `attn_dir` 給定時一併擷取 cross-attention（`CODE` §4.2）。擷取只在取樣步
    上開啟，非取樣步的 hook 是空操作——recorder 會實體化 (Q, 77) 的注意力
    矩陣，那正是 SDPA 融合核所避免的，全程開著會大幅拉高前向成本。

    `mask` 給定時走 inpainting（`SDWrapper.edit` 依權重分派）。**兩種形態的
    參數不互相沿用**：img2img 傳 `strength` 不傳 `mask`，inpainting 反之，
    兩者都由 `edit` 檢查。故此處把 strength 也交給它判斷，不在這裡先分支
    ——分支寫在這裡就等於多了一個要與 `edit` 保持同步的地方。

    回傳 (編輯結果, attention 產物路徑)。
    """
    emb = res.sd.encode_text(prompt).detach()
    emb_u = res.sd.uncond_prompt()
    lat = res.sd.latent_shape(x.shape[-2], x.shape[-1])
    noise = res.sd.sample_edit_noise(
        torch.empty(lat, device=x.device), seed=eval_noise_seed(res, seed_idx))
    kw = dict(guidance_scale=res.cfg.guidance, emb_uncond=emb_u,
              mask=mask, strength=(None if mask is not None
                                   else res.cfg.strength))
    if attn_dir is None:
        with torch.no_grad():
            return res.sd.edit(x, emb, noise, res.cfg.steps, **kw), []

    cap = AttnCapture(res.sd, res.cfg.steps, capture_span(res.sd, prompt))
    with cap, torch.no_grad():
        y = res.sd.edit(x, emb, noise, res.cfg.steps,
                        step_hook=cap.step_hook, **kw)
    return y, cap.write(attn_dir, attn_tag, full=attn_full)


def control_dir(res: Resources, image_id: str, purify: Tuple[str, float]
                ) -> Path:
    return (res.batch_dir / "control" / image_id / "purify"
            / purify_dir_name(*purify))


def control_executor(cell: grid.Cell, ctx: Dict[str, Any]
                     ) -> Tuple[List[str], Dict[str, Any]]:
    """φ=0 的同淨化對照。**跨條件共用**，每 (影像, 淨化, 種子) 只算一次。

    它量的是「同一個淨化算子施加在原圖上、再走同一條編輯鏈」的結果。
    沒有它，`d(E(P(x_def)), E(x))` 會把淨化本身造成的偏移算成防禦效果——
    實測 site P r=1 在 identity 下 shift 0.095、在 blur 下 0.347，高的那個
    是淨化自己造成的。
    """
    res: Resources = ctx["res"]
    entry = res.image(cell.image_id)
    kind, strength = cell.purify
    out_dir = control_dir(res, cell.image_id, cell.purify)
    out_dir.mkdir(parents=True, exist_ok=True)

    pur = make_purifier(kind, strength, eval_noise_seed(res, cell.seed), res)
    x_p = pur.evaluate(entry.x01)
    # 對照側同樣要存 attention（`CODE` §4.2「兩側都要」），否則無從相減。
    # 對照與 τ 無關，故 `attn_full` 只看種子。
    tag = f"seed{cell.seed}"
    y_ctrl, attn_arts = _sdedit(
        res, x_p, entry.attack_prompt, cell.seed,
        attn_dir=(out_dir / "attn") if res.cfg.capture_attn else None,
        attn_tag=tag, attn_full=(cell.seed == 0), mask=entry.mask)

    save_image(x_p, out_dir / "x_purified.png")
    edit_png = out_dir / f"edit_seed{cell.seed}.png"
    save_image(y_ctrl, edit_png)

    sem = res.suite.semantic(y_ctrl, entry.attack_prompt)
    extra = {
        "image_id": cell.image_id, "group": entry.group,
        "purify_kind": kind, "purify_strength": strength, "seed": cell.seed,
        "prompt": entry.attack_prompt,
        "ctrl_clip": sem["clip"], "ctrl_siglip": sem["siglip"],
        "purify_available": pur.available,
        "purify_differentiable": pur.differentiable,
        "attn_full": cell.seed == 0,
        "attn_steps": len(sampled_steps(res.cfg.steps)),
    }
    meta_path = out_dir / f"meta_seed{cell.seed}.json"
    write_meta(res, cell, meta_path, extra)
    arts = [out_dir / "x_purified.png", edit_png, meta_path] + attn_arts
    return [res.rel(p) for p in arts], extra


def _x_def_for(res: Resources, condition: str, image_id: str,
               tau: float) -> torch.Tensor:
    """評測用的 x_def。單格快取，見 `Resources._xdef_cache` 的說明。"""
    key = f"{condition}/{image_id}/tau{tau:g}"
    if res._xdef_cache.get("key") == key:
        return res._xdef_cache["value"]
    path = res.cell_dir(condition, image_id) / f"phi_tau{tau:g}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在：段 3 需要段 2 的產物。"
            "請先跑 `run_stage.py rayscale --batch <批次>`"
        )
    entry = res.image(image_id)
    x_def = materialize(load_phi(path), res, entry)
    res._xdef_cache.clear()
    res._xdef_cache.update({"key": key, "value": x_def})
    return x_def


def eval_executor(cell: grid.Cell, ctx: Dict[str, Any]
                  ) -> Tuple[List[str], Dict[str, Any]]:
    """淨化 → SDEdit → 指標。對照側由 `control` 段的產物讀回。"""
    res: Resources = ctx["res"]
    entry = res.image(cell.image_id)
    kind, strength = cell.purify
    tau = float(cell.tau)
    out_dir = (res.cell_dir(cell.condition, cell.image_id) / "purify"
               / purify_dir_name(kind, strength))
    out_dir.mkdir(parents=True, exist_ok=True)

    ctrl_png = control_dir(res, cell.image_id, cell.purify) / \
        f"edit_seed{cell.seed}.png"
    if not ctrl_png.exists():
        raise FileNotFoundError(
            f"{ctrl_png} 不存在：φ=0 的同淨化對照必須先跑完。"
            "對照跨條件共用，故它與 eval 一起跑而不是各條件各算一次"
        )
    y_ctrl = load_image_tensor(ctrl_png, res.device)

    x_def = _x_def_for(res, cell.condition, cell.image_id, tau)
    pur = make_purifier(kind, strength, eval_noise_seed(res, cell.seed), res)
    x_p = pur.evaluate(x_def)
    # `CODE` §4.2 的體積控制：逐層原圖只在主表所在的 τ、seed 0 完整存，
    # 其餘格點只留聚合圖與 attn_stats.csv。理由是體積不是重要性——
    # 數值在每一格都齊全，結論不受影響。
    attn_full = (tau == grid.MAIN_TAU and cell.seed == 0)
    tag = f"tau{tau:g}_seed{cell.seed}"
    y_def, attn_arts = _sdedit(
        res, x_p, entry.attack_prompt, cell.seed,
        attn_dir=(out_dir / "attn") if res.cfg.capture_attn else None,
        attn_tag=tag, attn_full=attn_full, mask=entry.mask)

    # 產物檔名必須帶 τ。2026-08-08 修。
    #
    # before：`x_purified.png` 與 `edit_seed{k}.png`，**都不帶 τ**。同一個
    # （條件, 影像, 淨化算子, seed）在四個 τ 上各寫一次同一個檔名，於是只有
    # **最後跑完的那個 τ** 留在磁碟上。實測 v14／v14r 兩批留下的都是 τ=0.35，
    # 也就是說**主表所在的 τ=0.20 的編輯輸出全部不存在**。
    #
    # 這不是體積控制（§4.2 那條規則是明確地只留聚合圖，且 attn 的檔名一直
    # 都帶 τ），是漏帶鍵。危險之處在於它不留痕跡：`compare.html` 的每一個
    # τ 的列都照樣渲染得出圖，只是那張圖屬於另一個 τ——**缺圖看得出來，
    # 錯圖看不出來**。而使用者 2026-08-08 定案人眼判讀為主判準，這些正是
    # 被判讀的檔案。
    #
    # after：改用與 attn 相同的 `tag`（`tau{τ:g}_seed{k}`）。
    #
    # `x_purified` 只在 seed 0 落盤：它是 `x_def_tau{τ}.png`（已逐 τ 落盤）
    # 經淨化算子的結果，可由既有產物重建，故取一個代表性樣本即可；確定性的
    # 算子（blur／jpeg／quantize／crop_resize／identity）逐 seed 本來就相同。
    # 這是**明講的**取捨，與上面那個漏帶鍵不同。
    edit_png = out_dir / f"edit_{tag}.png"
    save_image(y_def, edit_png)
    purified_png = out_dir / f"x_purified_{tag}.png"
    if cell.seed == 0:
        save_image(x_p, purified_png)

    m = res.suite.full(y_ctrl, y_def, prompt=entry.attack_prompt)
    row: Dict[str, Any] = {
        "condition": cell.condition, "image_id": cell.image_id,
        "group": entry.group, "tau": tau,
        "purify_kind": kind, "purify_strength": strength, "seed": cell.seed,
        "prompt": entry.attack_prompt,
        **{f"edit_{k}": v for k, v in m.items()},
        **{f"defimg_{k}": v for k, v in
           res.suite.pairwise(entry.x01, x_p).items()},
    }
    # DIA 報 MSE，而 `MetricSuite` 只有 PSNR（兩者一一對應但尺度不同）。
    # 逐欄對照該篇的表需要 MSE 本身，故在此補算而不是要讀者自己換算。
    row["edit_mse"] = float((y_ctrl - y_def).pow(2).mean())
    # `DESIGN` §5.3：effect = 對照側的語意對齊 − 防禦側的語意對齊。
    # 對照側已經是「同一淨化下的未防禦結果」，故淨化本身的效果被扣除。
    row["effect_clip"] = m["clip_a"] - m["clip_b"]
    row["effect_siglip"] = m["siglip_a"] - m["siglip_b"]
    # CLIP 分不出編輯是否發生（實測 +0.0101 ± 0.0169，標準差大於均值），
    # SigLIP 通過同一對照。判定用 SigLIP、對齊文獻用 CLIP，兩者都報。
    row["effect_abs"] = row["effect_siglip"]
    if not pur.differentiable:
        row["proxy_gap"] = pur.proxy_gap(x_def)
    row["purify_available"] = pur.available
    row["attn_full"] = attn_full
    row["attn_steps"] = len(sampled_steps(res.cfg.steps))

    meta_path = out_dir / f"metrics_{tag}.json"
    write_meta(res, cell, meta_path, row)
    # `x_purified` 只在 seed 0 存在，故只有存在時才進 `artifacts`——這份清單
    # 是 `compare_page._artifact` 的唯一依據，列一個沒寫出去的檔會讓頁面
    # 指向不存在的路徑。
    arts = [edit_png, meta_path]
    if cell.seed == 0:
        arts.insert(0, purified_png)
    arts += attn_arts
    return [res.rel(p) for p in arts], row


# ---------------------------------------------------------------------------
# 分派
# ---------------------------------------------------------------------------

_EXECUTORS: Dict[str, Callable] = {
    "train": train_executor,
    "rayscale": rayscale_executor,
    "control": control_executor,
    "eval": eval_executor,
}


def make_executor(stage: str) -> Callable:
    if stage not in _EXECUTORS:
        raise KeyError(
            f"段 {stage!r} 沒有格點式的計算層；格點式的是 "
            f"{sorted(_EXECUTORS)}。calib 與 report 走 run_calibration() 與 "
            "run_report()——它們沒有格點，硬塞進格點框架只會得到一個"
            "「零格、永遠成功」的段"
        )
    return _EXECUTORS[stage]


# ---------------------------------------------------------------------------
# 段 0：前置校準
# ---------------------------------------------------------------------------


def _edit_effect(res: Resources, entry: ImageEntry, strength: float,
                 seed_idx: int = 0) -> Dict[str, float]:
    """`SigLIP(編輯輸出, target) − SigLIP(原圖, target)`。

    量的是「未防禦的編輯確實把影像推向了攻擊 prompt」。先驗實驗 24 張裡有
    6 張不成立，其中一張為 −0.0007（編輯後反而更遠離 target）；在那類影像上
    量免疫效果沒有意義。

    **CLIP 不可用於此判定**（實測 +0.0101 ± 0.0169，標準差大於均值），
    但仍一併回報以便與文獻的欄位對齊。
    """
    emb = res.sd.encode_text(entry.attack_prompt).detach()
    emb_u = res.sd.uncond_prompt()
    lat = res.sd.latent_shape(entry.x01.shape[-2], entry.x01.shape[-1])
    noise = res.sd.sample_edit_noise(
        torch.empty(lat, device=res.device), seed=eval_noise_seed(res, seed_idx))
    with torch.no_grad():
        # inpainting 沒有 strength，故 `calibrate_strength` 的掃描在該威脅
        # 模型下退化為單一點（`STRENGTH_GRID` 的每一格算出同一個東西）。
        # 那是正確的行為而不是缺陷：段 0 該量的是「編輯有沒有效」，而
        # inpainting 的攻擊強度由遮罩面積決定，不由一個純量決定。
        y = res.sd.edit(entry.x01, emb, noise, res.cfg.steps,
                        mask=entry.mask,
                        strength=(None if entry.mask is not None else strength),
                        guidance_scale=res.cfg.guidance, emb_uncond=emb_u)
    a = res.suite.semantic(entry.x01, entry.attack_prompt)
    b = res.suite.semantic(y, entry.attack_prompt)
    return {"clip_orig": a["clip"], "clip_edit": b["clip"],
            "siglip_orig": a["siglip"], "siglip_edit": b["siglip"],
            "effect_clip": b["clip"] - a["clip"],
            "effect_siglip": b["siglip"] - a["siglip"]}


def calibrate_strength(res: Resources, calib_dir: Path) -> Dict[str, Any]:
    """SDEdit strength 掃描。回傳建議值與逐格結果。

    建議值取「平均 SigLIP 效果最大」的那一點。這是一個明寫的判準而不是唯一
    正確的判準——過強的 strength 會讓原圖語意整個消失，那時「防禦有沒有效」
    不再可判。故建議值只寫進 `calibration.json` 供對照，**實跑用的 strength
    由 CLI 指定並進 `config_hash`**：換 strength 是換一組實驗，不是換一個旋鈕。
    """
    rows = []
    for s in STRENGTH_GRID:
        for e in res.images.values():
            rows.append({"strength": s, "image_id": e.image_id,
                         "group": e.group, **_edit_effect(res, e, s)})
    write_csv(calib_dir / "strength_sweep.csv", rows)
    by_s: Dict[float, List[float]] = {}
    for r in rows:
        by_s.setdefault(r["strength"], []).append(r["effect_siglip"])
    means = {s: statistics.fmean(v) for s, v in by_s.items()}
    best = max(means, key=lambda s: means[s])
    return {"recommended": best, "mean_effect_siglip": means}


def filter_editable(res: Resources, calib_dir: Path) -> Dict[str, Any]:
    """編輯有效性過濾。在**實跑用的 strength** 上逐張量測。"""
    rows = []
    for e in res.images.values():
        r = _edit_effect(res, e, res.cfg.strength)
        r.update({"image_id": e.image_id, "group": e.group,
                  "strength": res.cfg.strength,
                  "passed": r["effect_siglip"] > res.cfg.edit_effect_threshold})
        rows.append(r)
    write_csv(calib_dir / "editable_filter.csv", rows)
    passed = [r["image_id"] for r in rows if r["passed"]]
    return {"threshold": res.cfg.edit_effect_threshold,
            "passed": passed, "n_passed": len(passed), "n_total": len(rows)}


def _probe_lr(res: Resources, condition: str, entry: ImageEntry,
              lr_key: str, lr: float, steps: int) -> Dict[str, Any]:
    """以候選學習率跑 `steps` 步，回傳末端的損失與監看量。

    校準表是查詢介面，此處把候選值放進一張**臨時**的表再交給 `optimize`
    ——不繞過 `resolve_lr`。繞過它就等於在校準流程裡開了一條「不經校準表也能
    取得學習率」的路徑，而那正是本模組要消滅的東西。
    """
    spec = condition_spec(condition)
    tmp = Calibration()
    ctx = res.calib_context
    tmp.put(lr_key, lr, ctx, note="段 0 的候選值，非最終校準結果")
    if spec.align_lr_key and spec.align_lr_key != lr_key:
        # 探測階段二時階段一必須有一個值才跑得起來。取候選格點的中位數，
        # 並在 note 裡註明它只是探測用——最終的 stage1 值由自己的探測決定。
        tmp.put(spec.align_lr_key, statistics.median(res.cfg.lr_grid), ctx,
                note="段 0 探測階段二時的暫用值，非最終校準結果")
    if spec.monitor:
        tmp.put(f"stop_tol.{spec.monitor}", 0.0, ctx,
                note="段 0 探測期不啟用平台停止")

    probe = dc_replace(res.cfg, max_steps=steps, align_steps=0)
    probe_res = dc_replace(res, cfg=probe, calib=tmp)
    cfg = optim_config(probe_res, spec, entry)
    cfg.stop_on_plateau = False        # 探測要跑滿固定步數，否則各候選不可比
    module = build_module(condition, res, entry, seed=res.cfg.seed)
    try:
        result = optimize(
            res.sd, module, entry.x01, cfg, loss_config(res, spec, entry),
            default_train_set(), calib=tmp, calib_context=ctx,
            y_target=(res.y_target if spec.defense_mode == "targeted_output"
                      else None),
        )
    except Diverged as e:
        # 發散是這個候選的**結果**，不是段 0 的錯誤。格點刻意跨數個量級，
        # 上端本來就該發散；`_pick_best` 依 `finite` 欄剔除它。不接住的話
        # 一個候選發散就會讓整個段 0 中止（2026-08-06 實測）。
        print(f"  [probe] {lr_key}={lr:g} 發散：{e}", flush=True)
        return {"condition": condition, "lr_key": lr_key, "lr": lr,
                "steps": 0, "image_id": entry.image_id,
                "final_loss": float("inf"), "final_L_def": float("nan"),
                "final_L_fid": float("nan"), "finite": False,
                "diverged": True}
    finally:
        module.remove()
    last = result.history[-1]
    # `diverged` 明寫 False 而非省略：`write_csv` 取全部列的鍵**聯集**，
    # 省略時該欄只在恰好有候選發散的批次裡出現，於是讀 `lr_probe.csv` 的人
    # 無法分辨「沒有候選發散」與「這一欄根本沒被寫出來」。
    out = {"condition": condition, "lr_key": lr_key, "lr": lr,
           "steps": len(result.history), "image_id": entry.image_id,
           "final_loss": last["loss"], "final_L_def": last["L_def"],
           "final_L_fid": last["L_fid"], "finite": _finite(last["loss"]),
           "diverged": False}
    if spec.monitor in last:
        out["monitor"] = spec.monitor
        out["monitor_first"] = result.history[0][spec.monitor]
        out["monitor_last"] = last[spec.monitor]
        out["monitor_per_step"] = (
            (last[spec.monitor] - result.history[0][spec.monitor])
            / max(len(result.history) - 1, 1))
    return out


def _probe_align_lr(res: Resources, condition: str, entry: ImageEntry,
                    lr: float, steps: int) -> Dict[str, Any]:
    """階段一（保真對齊）的候選探測。判準是對齊損失本身。

    與 `_probe_lr` 同樣把候選值放進一張**臨時**的表再交出去。`optim_config`
    會向校準表索取 `stop_tol`，而段 0 正在產生那張表、`res.calib` 此時必為
    None，故**必須把 `tmp` 換進去**——用原本的 `res` 會拿到
    `CalibrationMismatch: calibration.json 尚未產生`。

    2026-08-06 修正。before：`cfg = optim_config(res, spec)`，且 `tmp` 只放了
    `align_lr_key`。本函式只有帶 `align_lr_key` 的條件（N3／apa）會走到，
    而段 0 從未在 GPU 上跑過，故該缺陷未被暴露。
    """
    from src.defense.optimize import align

    spec = condition_spec(condition)
    tmp = Calibration()
    ctx = res.calib_context
    tmp.put(spec.align_lr_key, lr, ctx, note="段 0 的候選值")
    if spec.monitor:
        tmp.put(f"stop_tol.{spec.monitor}", 0.0, ctx,
                note="段 0 探測期不啟用平台停止")
    cfg = optim_config(dc_replace(res, calib=tmp), spec, entry)
    cfg.align_steps = steps
    module = build_module(condition, res, entry, seed=res.cfg.seed)
    try:
        gen = DefenseGenerator(res.sd, module, k_inv=res.cfg.k_inv,
                               t_max=res.cfg.t_max,
                               exact_inversion=res.cfg.exact_inversion)
        # x_base0 = G(x; φ=0)，階段一的四道 hinge 門檻由它決定
        # （`optimize.recon_floor_thresholds`）。探測與正式訓練必須用同一條
        # 基準線，否則探出來的學習率對應的是另一個目標函數。
        was_enabled = module.enabled
        module.disable()
        try:
            with torch.no_grad():
                x_base0 = gen.generate(entry.x01, gen.prepare(entry.x01)).detach()
        finally:
            if was_enabled:
                module.enable()
        _, hist = align(res.sd, module, entry.x01, cfg,
                        loss_config(res, spec, entry), gen, tmp, ctx, x_base0)
    except Diverged as e:
        # 第 0 步就發散時 `align` 沒有可還原的 φ 而拋出；理由同 `_probe_lr`。
        print(f"  [probe] {spec.align_lr_key}={lr:g} 發散：{e}", flush=True)
        return {"condition": condition, "lr_key": spec.align_lr_key, "lr": lr,
                "steps": 0, "image_id": entry.image_id,
                "final_loss": float("inf"), "final_lpips": float("nan"),
                "final_psnr": float("nan"), "finite": False, "diverged": True}
    finally:
        module.remove()
    last = hist[-1]
    # `align` 的發散有兩條路，此處必須涵蓋兩條：
    #
    #   1. 第 0 步就發散 → `raise_if_diverged` 拋 `Diverged` → 上面的 except。
    #   2. 迴圈中途出現非有限值 → `optimize.align` 以 `break` 記一列
    #      `diverged=True` 的 history 後正常返回（還原最佳步的 φ，故
    #      `x_align` 仍可用，不該拋例外）。
    #
    # before：本返回值不看 history 的 `diverged`，於是第 2 條路的候選只留下
    # `final_loss=inf` 與 `finite=False`，而 `optimize.py:782` 明寫「該事實要進
    # lr_probe.csv 供報告引用」。2026-08-07 於 v14 段 0 實測暴露：
    # `lr.N3_stage1=0.1` 在第 46 步發散，CSV 卻整張表都沒有 `diverged` 欄。
    return {"condition": condition, "lr_key": spec.align_lr_key, "lr": lr,
            "steps": len(hist), "image_id": entry.image_id,
            "final_loss": last["align_loss"],
            "final_lpips": last["fid_lpips"],
            "final_psnr": last["fid_psnr_total"],
            "finite": _finite(last["align_loss"]),
            "diverged": bool(last.get("diverged", False))}


def _finite(v: float) -> bool:
    return v == v and abs(v) != float("inf")


def calibrate_lr(res: Resources, calib_dir: Path) -> Dict[str, Any]:
    """逐條件的學習率。判準明寫於此，不藏在註解裡。

    對每個候選跑 `probe_steps` 步固定步數，取**末端總損失最小**者。理由：
    總損失正是這個階段在最小化的量，太小的 lr 使 L_def 幾乎不動、太大的 lr
    把 L_fid 的 hinge 撐爆，兩端都會被這個判準排除。非有限值（發散）一律
    先剔除。

    **限制（必須寫進報告）**：判準是「固定 12 步之後誰的損失低」，不是
    「跑到收斂誰最好」。步數短的探測偏好大 lr；本專案接受這個偏差，因為
    正式訓練開著平台停止，過大的 lr 會在 `stop_require_feasible` 下無法收斂
    而暴露出來。真正的替代方案是逐候選跑滿 250 步，成本是 20 倍。
    """
    # 語意指標的兩份權重（CLIP + SigLIP，合計 1,352 MB）到此為止都用不到了，
    # 而下面每個候選都要建 N1 的訓練圖——那條路徑不能開 UNet checkpoint，
    # 1024² 下 24 GB 的卡差約 600 MB 就 OOM。見 `MetricSuite.release_vlm`。
    res.suite.release_vlm()

    rows: List[Dict[str, Any]] = []
    entry = next(iter(res.images.values()))
    out: Dict[str, Any] = {}
    # 停止門檻的鍵是**監看量**而非條件（N2 與 N3 共用 `edit_shift`），故逐監看量
    # 彙總全部候選再算一次，不讓後一個條件覆寫前一個的值。
    by_monitor: Dict[str, List[Dict[str, Any]]] = {}
    # 哪個條件先認領了某個 lr 鍵。共用是合法的（Ra 與 apa 共用階段一），
    # 但**只有在兩者會探出同一個值時才合法**，故記下認領者的識別特徵。
    claimed: Dict[str, Tuple[str, tuple]] = {}

    def _claim(key: str, cond: str, ident: tuple) -> bool:
        """回傳是否要真的跑這一組探測。已被等價的條件認領過就跳過。"""
        if key not in claimed:
            claimed[key] = (cond, ident)
            return True
        owner, owner_ident = claimed[key]
        if owner_ident != ident:
            raise ValueError(
                f"條件 {cond!r} 與 {owner!r} 共用學習率鍵 {key!r}，但兩者的"
                f"探測對象不同（{ident} vs {owner_ident}）。判準是「固定步數後"
                f"末端總損失最小者」，而總損失是模式相依的，argmin 沒有理由"
                f"相同；照跑會讓後者靜默覆寫前者。請給其中一個條件自己的鍵"
            )
        print(f"  [probe] {key} 已由 {owner!r} 探測，{cond!r} 共用（探測對象"
              f"相同：{ident}）", flush=True)
        return False

    # **只探測這一批要跑的條件。** before：一律走 `grid.NONADDITIVE`，於是
    # 第三階段只跑 apa／Ra 的批次仍會為 N1／N2／N3 各跑 5 個候選，而那三個
    # 條件的 φ 這一批根本不會被用到。段 0 在 v14r 實測 4.9 小時，其中學習率
    # 探測是主成本。
    for cond in res.cfg.conditions:
        spec = condition_spec(cond)
        if spec.align_lr_key:
            # 對齊探測的判準是**對齊損失**（保真度），只由參數化決定。
            ident = ("align", spec.site)
            if _claim(spec.align_lr_key, cond, ident):
                probes = [
                    _probe_align_lr(res, cond, entry, lr, res.cfg.probe_steps)
                    for lr in res.cfg.lr_grid
                ]
                rows += probes
                out[spec.align_lr_key] = _pick_best(probes, spec.align_lr_key)
        if not spec.lr_key:
            # 隨機對照與 baseline 沒有要最佳化的階段二，沒有學習率可探。
            continue
        ident = ("defense", spec.site, spec.defense_mode, spec.target_metric)
        if not _claim(spec.lr_key, cond, ident):
            continue
        probes = [_probe_lr(res, cond, entry, spec.lr_key, lr,
                            res.cfg.probe_steps) for lr in res.cfg.lr_grid]
        rows += probes
        out[spec.lr_key] = _pick_best(probes, spec.lr_key)
        if spec.monitor:
            by_monitor.setdefault(spec.monitor, []).extend(probes)

    for monitor, probes in by_monitor.items():
        out[f"stop_tol.{monitor}"] = _pick_stop_tol(probes, monitor)

    write_csv(calib_dir / "lr_probe.csv", rows)
    return out


def _pick_best(probes: Sequence[Dict[str, Any]], lr_key: str) -> float:
    ok = [p for p in probes if p["finite"]]
    if not ok:
        raise RuntimeError(
            f"{lr_key}：全部候選學習率都發散（{[p['lr'] for p in probes]}）。"
            "校準失敗，不會挑一個看起來合理的值——那正是本專案重複十次的缺陷"
        )
    return float(min(ok, key=lambda p: p["final_loss"])["lr"])


def _pick_stop_tol(probes: Sequence[Dict[str, Any]], monitor: str) -> float:
    """該監看量的平台停止門檻。

    `optimize.LEGACY_MONITOR_TOL` 的取值規則是「比實測的平均每步改善低一個
    量級」，此處照辦：取各候選中每步改善的最大值，除以 10。刻意用最大值而非
    平均：門檻取得過高會讓最好的那個候選在正式訓練中被提早判為收斂。
    """
    vals = [abs(p["monitor_per_step"]) for p in probes
            if p["finite"] and "monitor_per_step" in p]
    if not vals:
        raise RuntimeError(
            f"沒有任何候選回報 {monitor} 的每步改善量，無法定出 stop_tol。"
            "不可沿用另一個監看量的門檻——動態範圍差數十倍（見 "
            "optimize.LEGACY_MONITOR_TOL 的說明）"
        )
    return max(vals) / 10.0


def measure_recon_floors(res: Resources,
                         calib_dir: Optional[Path] = None
                         ) -> Dict[str, float]:
    """每張影像走生成路徑的失真下限：`LPIPS(decode(encode(x)), x)`。

    走生成路徑的條件（N3）無論 φ 取什麼，其輸出都要經過一次 VAE 來回，
    故這個值是該影像上 τ 的硬下界。**它逐影像差很多**，不能用一個常數代替
    ——SD v1.4／512² 實測 bird_03 0.1330、dog_03 0.1403、cat_02 **0.2398**，
    最後這個高於 τ=0.20（`grid.GENERATIVE_LPIPS_FLOOR` 的名目值是 0.1434）。

    只做一次 VAE 來回，`micro_bench` 實測 0.40 秒／張，故真正執行的每一段
    都可以現場量而不必依賴段 0 的產物；`calib_dir` 給定時另寫一份 CSV 存證。

    無梯度、無隨機性：`encode_image` 取後驗均值，`decode_latent` 是純函式，
    故同一張圖在同一個模型與精度下量幾次都相同。
    """
    floors: Dict[str, float] = {}
    rows = []
    for e in res.images.values():
        with torch.no_grad():
            x_rec = res.sd.decode_latent(res.sd.encode_image(e.x01))
        m = res.suite.pairwise(e.x01, x_rec)
        floors[e.image_id] = float(m["lpips"])
        rows.append({"image_id": e.image_id, "group": e.group,
                     "lpips": float(m["lpips"]), "psnr": float(m["psnr"]),
                     "ssim": float(m["ssim"])})
    if calib_dir is not None:
        write_csv(Path(calib_dir) / "recon_floor.csv", rows)
    return floors


def measure_warp_reach(res: Resources, calib_dir: Path,
                       conditions: Sequence[str] = ()) -> Dict[str, Any]:
    """位移場在 `warp_max_disp` 下能達到的最大 LPIPS。

    存在理由：`max_disp` 是**硬上界**（`site_warp.displacement` 直接 clamp），
    而段 2 要把 φ 縮放到 τ = 0.35。若該上界下可達的 LPIPS 低於 0.35，
    `solve_k` 會拋出，而那不是程式錯誤而是「這個預算組合不可能達成」。
    在耗掉段 1 的機時之前先量出來。

    ## 本批沒有位移場條件時明確跳過（2026-08-09）

    before：本函式寫死 `build_module("R", ...)`，且 `run_calibration` 無條件
    呼叫它——**完全不看格點有哪些條件**。第三階段的隨機對照移到 site apa
    之後，`build_apa` 產生的 `CompositeResidual` 兩個成員都不是像素側，
    `pixel_residual()` 依 `composite.py` 回傳 `None`，接著
    `res.suite.pairwise(e.x01, None)` 會在指標內部以看不出來源的訊息中止。

    after：由呼叫端傳入本批實際要跑的條件，沒有任何 site warp 的條件時回傳
    `{"skipped": True, ...}` 並且**不寫入校準表**——寫一個沒有量過的
    `warp.min_lpips_at_bound` 進去，比不寫更危險。

    apa 那一側的對應量不是這個上界而是**下界**：`decode(encode(x))` 的重建
    誤差，由 `measure_recon_floors` 量出並逐影像存進 `calib/recon_floor.csv`。
    兩者不可互相代替——位移場受限於「最多能扭多遠」，apa 受限於「最少也有
    這麼糊」，故本函式跳過時該檢查由 `recon_floor` 承擔，見
    `run_calibration` 對兩者的處置。
    """
    warp_conditions = [c for c in (conditions or (grid.RANDOM_CONTROL,))
                       if condition_spec(c).site == "warp"]
    if not warp_conditions:
        reason = (
            f"本批的條件 {list(conditions)} 沒有任何一個走 site warp，"
            "位移場的可達上界對本批不適用。apa 那一側的對應量是重建誤差"
            "**下界**，見 calib/recon_floor.csv"
        )
        print(f"  [warp_reach] 跳過：{reason}", flush=True)
        return {"skipped": True, "reason": reason}

    probe = warp_conditions[0]
    rows = []
    for e in res.images.values():
        seed = res.cfg.seed + zlib.crc32(e.image_id.encode("utf-8"))
        mod = build_module(probe, res, e, seed=seed,
                           init_std=res.cfg.random_init_std)
        try:
            with torch.no_grad():
                p = direction_param(mod)
                # 乘一個大到必定被 max_disp 夾住的倍率，量的是上界本身
                p.data = p.data * (100.0 / max(float(p.data.abs().max()), 1e-8))
                x = mod.pixel_residual(e.x01)
            rows.append({
                "image_id": e.image_id, "group": e.group,
                "condition": probe,
                "max_disp": res.cfg.warp_max_disp,
                "grid_size": res.cfg.warp_grid_size,
                "lpips_at_bound": float(
                    res.suite.pairwise(e.x01, x)["lpips"]),
            })
        finally:
            mod.remove()
    write_csv(calib_dir / "warp_reach.csv", rows)
    reach = min(r["lpips_at_bound"] for r in rows)
    return {"min_lpips_at_bound": reach,
            "covers_train_tau": reach >= res.cfg.tau_train}


def micro_bench(res: Resources, calib_dir: Path) -> Dict[str, Any]:
    """`ARCH` §7 的實測倍率所需的基本計時。報告不得引用估計值。

    只量三個最基本的單位成本；`ARCH` §7 的七項措施各自的倍率仍需另行量測，
    此處提供的是它們共同的分母。
    """
    entry = next(iter(res.images.values()))
    rows = []

    def timed(name: str, fn):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        rows.append({"op": name, "seconds": time.perf_counter() - t})

    with torch.no_grad():
        timed("vae_roundtrip",
              lambda: res.sd.decode_latent(res.sd.encode_image(entry.x01)))
        timed(f"sdedit_{res.cfg.steps}steps",
              lambda: _sdedit(res, entry.x01, entry.attack_prompt, 0,
                              mask=entry.mask))
    write_csv(calib_dir / "micro_bench.csv", rows)
    return {r["op"]: r["seconds"] for r in rows}


def calibrate_precision_equiv(res: Resources, calib_dir: Path) -> Dict[str, Any]:
    """`precision_equiv.csv` —— 半精度對 fp32 的等價性（`ARCH` §7.1）。

    比的是**同一組權重在不同計算精度下的結果**，不是兩份不同的權重檔
    （`SDXLWrapper._load_pipeline` 的 `variant` 只選存檔精度，兩者都是官方
    發布的同一個模型）。故此處以同一個 `model_name` 另建一個 fp32 的
    wrapper，跑同一組輸入逐項比對。

    量三件事，涵蓋三條會分別出錯的路徑：

    | op | 路徑 | 為什麼要單獨量 |
    |---|---|---|
    | `vae_roundtrip` | VAE 編解碼 | fp16 下 SDXL 的 VAE 會溢位成全黑圖，`resolve_precision` 因此強制它留在 fp32；這一列是那條規則的實測依據 |
    | `eps` | 單次 UNet 前向 | ε 在 fp16 下只有 10 bit 尾數，而 PGD 的梯度品質直接取決於它 |
    | `sdedit` | 完整編輯鏈 | 逐步累積的誤差，前兩項各自看起來都很小時仍可能在此發散 |

    **本函式不自行判定「等價」**。門檻沒有出處，寫一個看起來合理的數字
    再據以自動退回 fp32，等於用一個猜測決定整批資料的精度。此處只把數值
    落盤並寫進校準表，由人讀了之後決定是否退回 fp32 並重估耗時
    （`RUNBOOK` §1.1）。

    計算精度已是 fp32 時不載入第二份權重：差值恆為零，該列直接標記。
    """
    from src.models.sd import SDWrapper, SDXLWrapper

    calib_dir.mkdir(parents=True, exist_ok=True)
    entry = next(iter(res.images.values()))
    run_dtype = res.sd.compute_dtype
    rows: List[Dict[str, Any]] = []

    if run_dtype == torch.float32:
        rows.append({"op": "all", "dtype": "float32", "reference": "float32",
                     "max_abs": 0.0, "rel_l2": 0.0, "psnr": float("inf"),
                     "note": "本批以 fp32 執行，與參考同一路徑，無須比對"})
        write_csv(calib_dir / "precision_equiv.csv", rows)
        return {"dtype": "float32", "compared": False, "rows": rows}

    def probes(sd) -> Dict[str, torch.Tensor]:
        lat = sd.latent_shape(entry.x01.shape[-2], entry.x01.shape[-1])
        noise = sd.sample_edit_noise(
            torch.empty(lat, device=sd.device), seed=eval_noise_seed(res, 0))
        emb = sd.encode_text(entry.attack_prompt).detach()
        emb_u = sd.uncond_prompt()
        t = torch.tensor(int(sd.num_train_timesteps * res.cfg.strength) - 1,
                         device=sd.device)
        with torch.no_grad():
            z = sd.encode_image(entry.x01)
            out = {
                "vae_roundtrip": sd.decode_latent(z).float().cpu(),
                "eps": sd._eps(z, t, emb).float().cpu(),
                "sdedit": sd.sdedit(
                    entry.x01, emb, noise, res.cfg.steps,
                    strength=res.cfg.strength,
                    guidance_scale=res.cfg.guidance,
                    emb_uncond=emb_u).float().cpu(),
            }
        return out

    got = probes(res.sd)
    cls = type(res.sd)
    if cls not in (SDWrapper, SDXLWrapper):
        cls = SDXLWrapper if isinstance(res.sd, SDXLWrapper) else SDWrapper
    print(f"[calib] 另載入 {cls.__name__}({res.sd.model_name}) fp32 作為參考",
          flush=True)
    # 本批的權重先讓開：兩份 SDXL 同時常駐在 24 GB 的卡上放不下（RTX 3090
    # 實測 OOM）。`got` 已算完且落在 CPU，故區塊內不需要 `res.sd`。
    # 搬動只換裝置不換 dtype，離開時搬回，數值路徑不變（見 `SDWrapper.offloaded`）。
    with res.sd.offloaded():
        ref_sd = cls(res.sd.model_name, dtype=torch.float32)
        try:
            ref = probes(ref_sd)
        finally:
            del ref_sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    name = str(run_dtype).replace("torch.", "")
    for op, a in got.items():
        b = ref[op]
        if a.shape != b.shape:
            raise RuntimeError(
                f"{op} 在兩個精度下的形狀不同（{tuple(a.shape)} vs "
                f"{tuple(b.shape)}）。這不是精度差異，是路徑不同"
            )
        d = (a - b)
        mse = float(d.pow(2).mean())
        rows.append({
            "op": op, "dtype": name, "reference": "float32",
            "max_abs": float(d.abs().max()),
            "rel_l2": float(d.norm() / b.norm().clamp_min(1e-12)),
            "psnr": (float("inf") if mse == 0
                     else float(10.0 * math.log10(1.0 / mse))),
            "note": "",
        })
    write_csv(calib_dir / "precision_equiv.csv", rows)
    return {"dtype": name, "compared": True, "rows": rows}


def run_calibration(res: Resources) -> Dict[str, Any]:
    """段 0。產出 `calib/calibration.json` 與五個逐項 CSV。

    **未涵蓋的一項**，需要主 session 決定後才能做：

    - `attn_norm`（SDXL 的 cross-attention 層清單與正規化常數）：本輪的 N1
      走 `shared_token_mass`，它逐層取平均後才跨層平均，不需要外部的正規化
      常數；`DESIGN` §6 第 3 項是為舊的散度形式寫的。若改回需要正規化的
      形式，此處必須補。
    """
    calib_dir = res.batch_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    ctx = res.calib_context
    table = Calibration()
    summary: Dict[str, Any] = {}

    summary["micro_bench"] = micro_bench(res, calib_dir)
    summary["precision_equiv"] = calibrate_precision_equiv(res, calib_dir)
    for r in summary["precision_equiv"]["rows"]:
        # 進校準表而不只是 CSV：報告要引用「本批的半精度與 fp32 差多少」，
        # 而校準表的 context 已含 gpu 與 precision，數值與其條件綁在一起。
        table.put(f"precision_equiv.{r['op']}.rel_l2", r["rel_l2"], ctx,
                  note=f"{r['dtype']} 對 fp32 的相對 L2；判定由人讀數字決定")

    summary["strength"] = calibrate_strength(res, calib_dir)
    table.put("strength.recommended", summary["strength"]["recommended"], ctx,
              note="SigLIP 平均編輯效果最大的一點；實跑值由 CLI 指定")

    summary["editable"] = filter_editable(res, calib_dir)
    if not summary["editable"]["passed"]:
        raise RuntimeError(
            f"在 strength={res.cfg.strength} 下沒有任何影像通過編輯有效性過濾"
            f"（門檻 {res.cfg.edit_effect_threshold}）。"
            "在無效編輯上量免疫效果沒有意義，此處停下而不是繼續跑段 1"
        )

    summary["warp_reach"] = measure_warp_reach(res, calib_dir,
                                               res.cfg.conditions)
    if not summary["warp_reach"].get("skipped"):
        table.put("warp.min_lpips_at_bound",
                  summary["warp_reach"]["min_lpips_at_bound"], ctx,
                  note=f"max_disp={res.cfg.warp_max_disp} 下可達的最小 LPIPS")

    # 逐影像的生成路徑下限。段 2／段 3 據此把結構上不可達的格標成 skipped
    # 而非讓 `solve_k` 拋出（2026-08-07 的 cat_02 事故）。此處存 CSV 供報告
    # 引用；真正執行時各段會自己再量一次（一次 VAE 來回 0.4 秒），故不必
    # 為了取得它而重跑段 0。
    summary["recon_floor"] = measure_recon_floors(res, calib_dir)

    # 走生成路徑的條件在場時，重建下限就是這一批的可達性判準（位移場那一側
    # 是 `warp_reach` 的上界，兩者不可互相代替，見 `measure_warp_reach`）。
    # 逐影像的下限高於訓練點時，該影像在訓練點上結構上不可達——v14r 的
    # cat_02（0.2398）在 τ=0.20 就是這個情形。這裡把判定寫進 summary，
    # 段 0 一跑完就看得到，不必等段 2 的 `solve_k` 拋出才知道。
    gen_conditions = [c for c in res.cfg.conditions
                      if c in grid.GENERATIVE_CONDITIONS]
    if gen_conditions:
        floors = summary["recon_floor"]
        blocked = {k: v for k, v in floors.items() if v >= res.cfg.tau_train}
        summary["recon_floor_check"] = {
            "conditions": gen_conditions,
            "tau_train": res.cfg.tau_train,
            "max_floor": max(floors.values()) if floors else None,
            "covers_train_tau": not blocked,
            "blocked_images": blocked,
        }
        if blocked:
            print(f"  [recon_floor] **注意** 這些影像的重建下限已達或超過 "
                  f"τ_train={res.cfg.tau_train}：{blocked}。"
                  f"{gen_conditions} 在該影像的訓練點上結構上不可達，"
                  f"段 1／段 2 會把它標成 skipped", flush=True)

    for key, value in calibrate_lr(res, calib_dir).items():
        table.put(key, value, ctx,
                  note=(f"段 0 探測 {res.cfg.probe_steps} 步、取末端總損失最小者；"
                        f"候選 {list(res.cfg.lr_grid)}"))
    summary["calibration_keys"] = sorted(table.entries)

    path = calib_dir / "calibration.json"
    table.save(path)
    save_json(summary, calib_dir / "calib_summary.json")
    return {"path": path, "summary": summary}


# ---------------------------------------------------------------------------
# 段 4：報表
# ---------------------------------------------------------------------------

# `grid.csv` 的欄位順序（`CODE` §4）。不在此表內的欄位仍會輸出，接在後面
# ——少一欄比多一欄危險，故不做白名單過濾。
GRID_COLUMNS = (
    "cell_id", "config_hash", "condition", "image_id", "subtask", "tau",
    "purify_kind", "purify_strength", "seed",
    "fid_psnr", "fid_lpips", "fid_ssim", "fid_linf", "fid_fsim", "fid_vif_p",
    "fid_dists", "fid_niqe", "fid_acutance_ratio", "fid_rms",
    "fid_frac_gt_16_255",
    "edit_clip_b", "edit_siglip_b", "edit_psnr", "edit_lpips", "edit_ssim",
    "edit_mse", "edit_fsim", "edit_vif_p",
    "effect_abs", "effect_clip", "effect_siglip",
    "effect_control", "retention", "retention_usable",
    "steps_used", "stop_reason", "seconds", "disp_mean_px", "disp_max_px",
    "proxy_gap", "modified_from_paper", "skipped_reason",
)


def run_report(res: Resources) -> Dict[str, Any]:
    """段 4：彙整 `grid.csv`。

    資料來源是 `_cells/*.json`（進度的真相來源）而不是掃檔案樹：每格的
    `extra_meta` 已經是那一格的完整量測結果，重新從產物解析一次等於讓
    同一份數字有兩條互相可能不一致的來源。

    `compare.html`（人眼比對頁，**主判準**）與 `attention.html` 同樣在此產生，
    來源也是 `_cells/*.json`：兩者只寫 `<img src>`，不讀影像內容、不碰 GPU。
    """
    cells = load_cells(res.batch_dir)
    by_id = {c["id"]: c for c in cells}
    rows: List[Dict[str, Any]] = []

    for c in cells:
        if c.get("stage") != "eval":
            continue
        if c.get("status") == "skipped":
            # 不適用的格由 `runner` 以最小的 meta 記錄（只有 config_hash 與
            # skipped_reason），故條件與影像必須由識別碼還原——少了這兩欄，
            # 「哪些格沒跑」在報表上就只剩一串路徑。
            parts = c["id"].split("/")
            rows.append({"cell_id": c["id"], "condition": parts[1],
                         "image_id": parts[2],
                         "skipped_reason": c.get("skipped_reason", "")})
            continue
        if c.get("status") != "done":
            continue
        row = {k: v for k, v in c.items()
               if k not in ("id", "config", "artifacts", "error")}
        row["cell_id"] = c["id"]
        row["subtask"] = c.get("group", "")
        cfg = c.get("config") or {}
        cond, img, tau = c.get("condition"), c.get("image"), cfg.get("tau")
        row["tau"] = tau
        _join(row, by_id, grid.Cell("train", cond, img),
              ("steps_used", "stop_reason", "disp_mean_px", "disp_max_px",
               "modified_from_paper", "lr"))
        _join(row, by_id, grid.Cell("rayscale", cond, img, tau=tau),
              ("tau_achieved", "scale_k", "fid_psnr", "fid_lpips", "fid_ssim",
               "fid_linf", "fid_fsim", "fid_vif_p", "fid_dists",
               "fid_acutance_ratio", "fid_rms", "fid_frac_gt_16_255", "niqe"))
        if "niqe" in row:
            row["fid_niqe"] = row.pop("niqe")
        rows.append(row)

    _fill_retention(rows)
    ordered = [_order_row(r) for r in rows]
    path = write_csv(res.batch_dir / "grid.csv", ordered)

    # `_fill_retention` 只改 `rows`，而比對頁讀的是 `cells`。把兩個算出來的
    # 欄位回填，頁面才看得到「這一格的 retention 不可用」——那正是先驗實驗
    # 的 −43／−98 應該在資料層就被標出的地方。
    for r in rows:
        c = by_id.get(r.get("cell_id"))
        if c is not None:
            for k in ("retention", "retention_usable", "effect_control"):
                if k in r:
                    c[k] = r[k]

    env = read_env(res.batch_dir)
    write_text(res.batch_dir / "compare.html",
               build_compare_html(cells, batch=res.batch_dir.name, env=env))
    write_text(res.batch_dir / "attention.html",
               build_attention_html(cells, batch=res.batch_dir.name, env=env))

    save_json({"n_rows": len(ordered),
               "conditions": sorted({r.get("condition") for r in rows
                                     if r.get("condition")}),
               "usable_rows": sum(1 for r in rows
                                  if r.get("retention_usable") is True)},
              res.batch_dir / "report_summary.json")
    return {"path": path, "n_rows": len(ordered),
            "compare": res.rel(res.batch_dir / "compare.html"),
            "attention": res.rel(res.batch_dir / "attention.html")}


def _join(row: Dict[str, Any], by_id: Dict[str, Dict], cell: grid.Cell,
          keys: Sequence[str]) -> None:
    src = by_id.get(cell.cell_id())
    if src is None:
        return
    for k in keys:
        if k in src and k not in row:
            row[k] = src[k]
    if "seconds" in src:
        row.setdefault("seconds", src["seconds"])


def _fill_retention(rows: List[Dict[str, Any]]) -> None:
    """`retention = effect(淨化) / effect(identity)`，並標出不可用的列。

    `CODE` §4：`effect(identity) < 3 × 量測標準差` 時 `retention_usable`
    為 false，該列的 retention 不進任何統計。這一欄的存在是為了讓先驗實驗
    那種 −43、−98 的數字在資料層就被標出，而不是等到報表階段才發現。
    """
    base: Dict[Tuple, List[float]] = {}
    for r in rows:
        if r.get("purify_kind") == "identity" and "effect_abs" in r:
            base.setdefault(
                (r["condition"], r["image_id"], r["tau"]), []
            ).append(float(r["effect_abs"]))

    stats = {}
    for key, vals in base.items():
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) >= 2 else float("nan")
        stats[key] = (mean, sd)

    for r in rows:
        if "effect_abs" not in r:
            continue
        key = (r.get("condition"), r.get("image_id"), r.get("tau"))
        if key not in stats:
            continue
        mean, sd = stats[key]
        r["effect_control"] = mean
        r["retention"] = (float(r["effect_abs"]) / mean
                          if mean != 0 else float("nan"))
        # n < 2 時樣本標準差沒有定義；`grid.MIN_SEEDS` 已擋住該情形，
        # 此處若仍為 NaN 表示資料不完整，標為不可用而非當成通過。
        r["retention_usable"] = bool(sd == sd and mean >= 3.0 * sd)


def _order_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: row.get(k) for k in GRID_COLUMNS if k in row}
    for k, v in row.items():
        if k not in out:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 事前檢查
# ---------------------------------------------------------------------------


def preflight(res: Resources, conditions: Sequence[str] = grid.CONDITIONS
              ) -> List[str]:
    """在耗掉機時之前列出「這批一定會失敗的條件與原因」。

    不自動跳過它們：缺 MIST.png 是一個要補的檔案、PromptFlare 的 512² 限制是
    一個要決定的白名單，兩者都需要人來處理。靜默跳過會讓比較表少一列而
    讀者無從得知。
    """
    warns: List[str] = []
    for cond in conditions:
        if cond not in BASELINE_REGISTRY:
            continue
        if cond == "mist" and not res.cfg.mist_target:
            warns.append(
                "mist：需要原作的 MIST.png（1440×1440，黑底白字密集平鋪）。"
                "該檔無法由描述重建，且不可用 PhotoGuard 的零張量代用"
                "（SOURCE_AUDIT §3.4）。請以 --mist-target 指定路徑，"
                "否則該條件的每一格都會失敗"
            )
        if cond == "promptflare" and res.cfg.resolution != 512:
            warns.append(
                f"promptflare：其 loss_depth 是 token 數白名單，對應 512² "
                f"影像；本批解析度為 {res.cfg.resolution}²，`prepare` 會拒絕。"
                "白名單必須依「排除最外層」的原則重新決定（SOURCE_AUDIT §2.5），"
                "在未決定前不得沿用原值——沿用會讓損失恆為 0 或涵蓋錯的層"
            )
    # 2026-08-08 改。before：判準是「等於 `LossConfig` 的預設值」。after：判準
    # 改為「偏離 τ_train 的比例規則」。理由是處置 A 之後門檻由
    # `scaled_thresholds` 導出，舊判準只在 τ_train 恰為 0.05 時才會觸發，而
    # 那正是唯一不需要警告的情形——判準與症狀反了。
    want = scaled_thresholds(res.cfg.tau_train)
    for key in ("tau_acut", "tau_chroma"):
        got = getattr(res.cfg, key)
        if abs(got - want[key]) > 1e-9:
            warns.append(
                f"{key} 為 {got}，而 τ_train={res.cfg.tau_train} 依比例規則"
                f"應為 {want[key]:.4g}。門檻是在 τ_lpips=0.05 上由人眼判讀"
                "定出的，用在別的預算上必須等比例放大，否則非加性條件會在"
                "到達 τ 之前就被這道 hinge 擋下（objective.py「門檻的適用範圍」）"
            )
    return warns


def unavailable_purifiers(res: Resources) -> Dict[Tuple[str, float], str]:
    """相依不齊的淨化算子與原因。

    回傳非空時，呼叫端會把對應的格標成 `skipped` 而非讓它們逐一失敗——
    `runner.CONSECUTIVE_FAILURE_LIMIT` 是 10，而同一個算子連續有 5 個種子，
    兩個不可用的算子相鄰就會誤觸「系統性失敗」中止，把整段停掉。

    標成 `skipped` 不等於隱瞞：`skip_reason` 會寫進該格的紀錄與 `grid.csv`
    的 `skipped_reason` 欄，儀表板也把 skipped 與 failed 分開計。
    """
    out: Dict[Tuple[str, float], str] = {}
    seen = set(grid.MAIN_PURIFIERS) | set(grid.SWEEP_PURIFIERS)
    for kind, strength in sorted(seen):
        p = make_purifier(kind, strength, 0, res)
        if not p.available:
            out[(kind, strength)] = (
                f"淨化算子 {kind} 的相依不齊（權重或套件缺席），"
                "本批無法執行；見 src/purify/ops.py 對該算子的說明"
            )
    return out


def annotate_unavailable(cells: Sequence[grid.Cell], res: Resources
                         ) -> List[grid.Cell]:
    """把用到不可用算子的格標成 skipped。已經有 skip_reason 的不覆蓋。"""
    bad = unavailable_purifiers(res)
    if not bad:
        return list(cells)
    out = []
    for c in cells:
        if c.purify is not None and not c.skipped and c.purify in bad:
            out.append(dc_replace(c, skip_reason=bad[c.purify]))
        else:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# 已知缺口（`CODE` §4 尚未涵蓋的部分）
# ---------------------------------------------------------------------------
#
# 2026-08-05：原列的第 1–3 項均已完成，逐項如下。
#
# 1. **attention map 的留存**（`CODE` §4.2）：`src/experiment/attn_capture.py`。
#    取樣規則不需另行裁決——`CODE` §4.2 本來就已寫定（每 5 步、固定含首末步、
#    兩側都存、逐層原圖只在主表 τ 與 seed 0 完整存）。記憶體的顧慮以
#    `CrossAttentionRecorder.enabled` 解決：非取樣步的 hook 是空操作，
#    故 70 層 × (Q, 77) 的矩陣只在 11 個取樣步上實體化。
#
# 2. **`compare.html`**：`src/experiment/compare_page.py`。版面由判準決定而非
#    自由選擇——要判的是「非加性在同失真、同淨化下是否勝過加性」，故以
#    (影像, τ, 淨化) 分組、九個條件並排成列，六張圖依因果鏈排序。
#
# 3. **`precision_equiv.csv`**：`calibrate_precision_equiv`。
#
# 4. **FID／Precision**。`DESIGN` §5.1 已把兩者降為「參考值、不作判定用」
#    （N=3 下分布層級指標無意義），故本輪不計算。擴大 N 後才需要補。
#    **這是唯一仍未做的一項，且是刻意不做。**
