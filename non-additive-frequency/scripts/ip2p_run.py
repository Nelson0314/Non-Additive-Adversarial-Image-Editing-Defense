"""主線驅動：在 **InstructPix2Pix** 上跑紋理重相位與 DCT-Shield（DEC-031）。

**這一支取代 SDEdit 那條線成為主線**（使用者 2026-08-19 裁定）；
`phase_ablation.py`／`dct_shield_run.py` 那條 SDEdit 線保留但凍結，
不要再往上加新條件——兩條線的攻擊機制不同，數字不可並列。

換了什麼、沒換什麼
────────────────────────────────────────────────────────────────────
**換的只有攻擊方。** 防禦這一側逐字沿用既有的程式：

    參數化      src/residual/texture_rephase.py（相位）、AdditiveParam（加性）
    損失        make_encoder_target_loss ＝ ‖E(x_def) − E(y_target)‖²
    更新規則    src/defense/param_pgd.py 的 sign PGD
    DCT-Shield  src/baselines/dct_shield.py 的 Algorithm 1

`E` 現在是 **IP2P 自己的 VAE**——白盒假設是「攻擊方的模型已知」，換了攻擊方
就該打它的編碼器。`IP2PWrapper.encode_image` 與 `SDWrapper.encode_image`
同名同語意，故上面四件全部不必改。

**不要沿用 SDEdit 線的結論。** FND-055 那條機制（相位擾動用得上「被噪聲稀釋
後仍留在 latent 裡的原圖訊號」）在 IP2P 上不成立：它把未加噪的 `E(x)` 由第
一層卷積直接餵進去。本方法在 IP2P 下是強是弱屬於待測。

三個必須先驗收的前提
────────────────────────────────────────────────────────────────────
1. **未防禦的編輯必須真的成功**（DEC-022），否則位移量的分母不成立。
   `--check-only` 只跑未防禦的編輯並報 CLIP／SigLIP 對齊，先看服從率。
2. **θ 的人眼門檻要重定**。θ=1.30 是在 `set0817`（人物／動物特寫）上定的，
   OmniEdit 是通用場景，紋理閘的作用面積會不同。先跑
   `phase_distortion_sweep.py`，不要沿用 1.30。
3. **推論參數是本專案指定的**（論文 §5.3 沒給步數與兩個導引尺度），
   逐列寫進 CSV。

用法：
    python scripts/ip2p_run.py --out runs/i0820/check --check-only
    python scripts/ip2p_run.py --out runs/i0820/g0 --conditions phase dct_shield \\
        --phase-radius 1.30 --eps 1.0 --images task_obj_add_441549 ...
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.utils as vutils  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from dataclasses import replace  # noqa: E402

from phase_ablation import WARP_GRID, build  # noqa: E402
from src.baselines.advdrop import (
    PAPER_Q_INIT, AdvDropSpec, run_advdrop,
)
from src.baselines.dct_watermark import (  # noqa: E402
    PAPER_ADV_DIAGONALS, PAPER_MU, PAPER_TAU, DJSMASpec, run_djsma,
)
from src.baselines.dct_shield import (  # noqa: E402
    PAPER_DEFAULT_QUALITY, PAPER_EPS, PAPER_GAMMA, PAPER_STEPS,
    DCTShieldSpec, run_dct_shield,
)
from src.baselines.encoder_target import make_encoder_target_loss  # noqa: E402
from src.baselines.jpeg_codec import (  # noqa: E402
    jpeg_roundtrip, jpeg_roundtrip_ste, normalize_quality,
)
from src.defense.consistency_loss import (  # noqa: E402
    make_consistency_term,
)
from src.defense.linf_deliver import clamp_residual_ste  # noqa: E402
from src.defense.purify_aware import (  # noqa: E402
    BROAD_CLASSES,
    BROAD_FRACTIONS,
    BROAD_SIGMAS,
    STAGE2_OPS, STAGE2_ORDERS, make_eot_geometry_transform,
    make_eot_jpeg_transform, make_eot_ops_transform, make_fixed_jpeg_transform,
    make_jpeg_transform, make_sequenced_ops_transform,
    make_eot_broad_transform,
)
from src.defense.param_pgd import (  # noqa: E402
    fit_to_budget, run_param_pgd, run_stage2_pgd, step_scale_of,
)
from src.defense.stadv_flow import (  # noqa: E402
    NEIGHBOURHOODS, flow_tv_loss,
)
from src.defense.fixedpoint_loss import make_normalised_term  # noqa: E402
from src.defense.image_guidance_loss import (  # noqa: E402
    ZT_MODES, make_image_guidance_loss,
)
from src.residual.perceptual_weight import (  # noqa: E402
    FREQ_WEIGHTS,
    SURVIVAL_WEIGHTS,
)
from src.residual.texture_rephase import (  # noqa: E402
    FLOOR_ENVELOPES,
    FLOOR_ENVELOPE_SCOPES,
    FLOOR_GATES,
)
from src.metrics.standard import (  # noqa: E402
    SIGLIP_BLOCKED_THRESHOLD, blocked_by_siglip, standard_row,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.ip2p import (  # noqa: E402
    IP2P_IMAGE_GUIDANCE, IP2P_SEED, IP2P_STEPS, IP2P_TEXT_GUIDANCE, IP2PWrapper,
)
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
# 色散變形（`src/defense/dispersion.py`）：K 個頻帶各自一個隨機位移。
# `disp_k1` 是古典位移場、`disp_kfull` 是逐頻格獨立的隨機相位。三者都是
# **隨機、不最佳化**的對照，`params()` 為空，`run_param_pgd` 不更新任何東西。
DISP_CONDS = ("disp_k1", "disp_k2", "disp_k3", "disp_k4", "disp_k8",
              "disp_kfull",
              # `_opt` = 可學 ＋ 接上紋理閘與帶級知覺定價。
              "disp_k1_opt", "disp_k2_opt", "disp_k4_opt", "disp_k8_opt")
PHASE_CONDS = ("phase", "phase_rand", "add", "phase_gain", "gain_only",
               "floor_only", "shading", "shading_rand",
               # 色彩重映射（`src/defense/color_param.py`）。強度旗鈕都是
               # `--radius`，但**兩族的單位不同**：曲線族是斜率剖面的動態
               # 範圍、網格族是仿射係數的 L∞ 偏移，不可互相換算。
               "color_curve", "color_curve_rand",
               "color_grid", "color_grid_rand",) + DISP_CONDS + (
               # WaNet 式三元對照（`runs/ip2p_warp/`）。強度旗鈕是 `--radius`，
               # 單位是最大位移像素數。三格的分工見 `phase_ablation.build`。
               "warp", "warp_rand", "warp_roundtrip")
DCT_CONDS = ("dct_shield", "dct_shield_y")
# DCT 域的保長配對旋轉（`runs/dct_phase_design/README.md`）。強度旗鈕是
# `--radius`，單位是旋轉角的上界 theta_max。**它自己就交付壓縮圖**
# （參數作用在量化後的整數係數上、解碼即輸出），故不可再疊 `--deliver-jpeg`。
DCT_ROTATE_CONDS = ("dct_rotate", "dct_rotate_rand")
# DCT 域的**非加性**擾動族（`src/defense/dct_nonadditive.py`）。與上面那一族的
# 差別：旋轉的平面是**學出來的**，不是釘在兩個固定座標上。它作用在未量化的
# 浮點係數上、輸出浮點影像，所以 `--deliver-jpeg` 是可以疊的獨立旗鈕。
DCT_NONADD_CONDS = ("dct_nonadd", "dct_nonadd_rand")
# **整併版**：學出來的旋轉平面 ＋ 量化後的整數係數 ＋ 交付即參數
# （`src/defense/dct_unified.py`）。與 `DCT_ROTATE_CONDS` 同樣**自己就
# 交付壓縮圖**，故一樣不可再疊 `--deliver-jpeg`；交付品質用 `--dct-qd`。
DCT_UNIFIED_CONDS = ("dct_unified", "dct_unified_rand")
# `advdrop_max` 不是最佳化的解，是 AdvDrop 在該 eps 下**可行集的邊界**：量化表
# 整片推到 `q = 1 + eps`。存在的理由見 `runs/ip2p_advdrop_band/README.md`——
# 最佳化收斂不到帶內（500 步只到 DISTS 0.0378），但天花板在 eps = 100 時是
# 0.1414，正落在帶內。要回答「8×8 格點的非加性擾動抗不抗裁切」就用這一點，
# 那個問題與最佳化收不收斂無關。**它不是 AdvDrop 這個方法本身**，
# `modified_from_paper` 恆為真，報表上不可寫成 AdvDrop 的結果。
ADVDROP_CONDS = ("advdrop", "advdrop_max")
WM_CONDS = ("dct_wm",)
# 補丁載體（`src/defense/patch_param.py`）。此前這個元組在四處各展開一次，
# 而 `DEFECTS.md` 記過同型的坑（政策謂詞散在多處，只改其中一部分不會報錯）。
PATCH_CONDS = ("patch", "patch_rand")
# `--deliver-jpeg` 放行的條件。**它是本方法自己的交付步驟**，故凡是本方法的
# 參數化都該放行；`DCT_ROTATE_CONDS`／`DCT_UNIFIED_CONDS` 例外，因為那兩族
# 自己就交付壓縮圖（上面另有一條守門），而別人的方法（DCT-Shield／AdvDrop／
# 浮水印）不放行，理由見那條守門的註解。
#
# 補丁族此前不在清單裡，原因是這個守門寫在補丁族加進來之前，不是裁定。
# 相位族已量到量化交付把 jpeg75 的存活由 0.362 拉到 0.493、jpeg30 由 0.146
# 拉到 0.238、blur 由 0.118 拉到 0.188（`runs/ip2p_deliver_jpeg/README.md`），
# 而補丁族正是現在唯一在身分讀數上有效、卻死在 JPEG 與低通的載體。
#
# **一個必須寫在報表上的代價**：JPEG 往返會壓到整張圖，含受保護的臉，
# 於是「受保護主體逐位元不動」在這個組合下變成「受保護主體只被壓到 QD」。
DELIVER_JPEG_CONDS = PHASE_CONDS + DCT_NONADD_CONDS + PATCH_CONDS
# 不做最佳化的對照條件：參數是抽出來的、`params()` 是空的。分階段訓練
# 沒有階段一的解可以接，故一律拒絕而不是靜默跳過階段二。
NO_OPT_CONDS = ("phase_rand", "shading_rand", "warp_rand",
                "color_curve_rand", "color_grid_rand",
                "dct_rotate_rand", "dct_nonadd_rand", "dct_unified_rand")


def defense_steps(args, cond: str) -> int:
    """該條件實際用了幾步最佳化。

    每個條件走的是不同的旗標（本方法 `--steps` 預設 100、DCT-Shield
    `--dct-steps` 預設 1000、AdvDrop 50、浮水印 300），而頭對頭表把它們並排
    在同一列上。**預算差十倍這件事此前沒有出現在任何欄位裡**，於是「誰比較
    強」與「誰跑比較久」在報表上分不開。這個函式只做取值，不做判斷。
    """
    if cond in DCT_CONDS:
        return int(args.dct_steps)
    if cond in ADVDROP_CONDS:
        return int(args.advdrop_steps)
    if cond in WM_CONDS:
        # DJSMA 是貪婪迭代，迭代上限就是 tau（同時是 l0 界）。
        return int(args.wm_tau)
    return int(args.steps)


def _purify_transform(args):
    """`--purify-aware` 選到的可微分淨化算子。`none` 回傳 None（預設，
    行為與加入此旗標之前逐位元相同）。"""
    if args.purify_aware == "none":
        return None
    if args.purify_aware == "curriculum":
        return make_jpeg_transform(args.steps)
    if args.purify_aware == "fixed75":
        return make_fixed_jpeg_transform(75)
    if args.purify_aware == "eot_jpeg":
        return make_eot_jpeg_transform(tuple(args.eot_qualities), seed=args.seed)
    if args.purify_aware == "eot_geometry":
        return make_eot_geometry_transform(seed=args.seed)
    if args.purify_aware == "eot_broad":
        # 四類算子、每類再抽一個參數。`eot_ops` 的模糊與裁切是**固定值**，
        # 固定的變換會被 co-adapt，那正是要撐開的東西。
        return make_eot_broad_transform(
            tuple(args.eot_qualities), tuple(args.eot_sigmas),
            tuple(args.eot_fractions), seed=args.seed,
            classes=tuple(args.eot_classes))
    return make_eot_ops_transform((75,), seed=args.seed)


def deliver_quality(args):
    """`--deliver-jpeg` 換算成 1–100 的整數品質；0 代表關閉，回傳 `None`。

    `normalize_quality` 同時吃論文式的小數（0.85）與整數（85），與
    `--q-alg` 用同一條換算路徑，兩個旗標的 0.85 因此指同一張量化表。
    """
    if not args.deliver_jpeg:
        return None
    return normalize_quality(args.deliver_jpeg)


def _forward_transform(args):
    """最佳化迴圈的前向要套的變換。

    **與已否決的 `--purify-aware` 差在交付什麼，不是差在迴圈裡看到什麼。**
    `RESULTS.md` 否決過「針對淨化最佳化沒有改善抗淨化」——那三個變體
    （fixed75／curriculum／多算子 EOT）把可微分 JPEG 放進 PGD 前向，但
    **交出去的是未壓縮的圖**，於是最佳化找到的「壓縮活得下來的位置」在交付
    的那一刻就被丟掉了：攻擊方拿到的是連續值，一被重新量化仍然散掉。

    `--deliver-jpeg QD` 是另一件事：迴圈的前向套
    `jpeg_roundtrip_ste(·, QD)`，而**交付與存檔的也是 `jpeg_roundtrip(·, QD)`
    的輸出**（`defend()` 裡那一步）。輸出因此被約束在 QD 的量化格點上，
    攻擊方以同品質或更高品質重壓時近似恆等——這正是 DCT-Shield 抗 JPEG 的
    全部來源（它把 δ 直接加在量化後的整數係數上，見
    `src/baselines/jpeg_codec.py` 的 docstring）。

    兩者同時給時，順序是「先自壓、再讓攻擊方淨化」，與實際發生的順序一致。
    `--deliver-jpeg 0` 且 `--purify-aware none` 時回傳 `None`，行為與加入這個
    旗標之前逐位元相同。
    """
    purify = _purify_transform(args)
    q = deliver_quality(args)
    if q is None:
        return purify
    if purify is None:
        def transform(x01, step):
            return jpeg_roundtrip_ste(x01, q)
        return transform

    def transform(x01, step):
        return purify(jpeg_roundtrip_ste(x01, q), step)
    return transform


def _deliver_projection(args, x01):
    """`--linf-deliver` 給定時回傳交付前要套的像素域 L∞ 投影，否則 `None`。

    投影是**我方交付前**做的，套在 `param.render` 的輸出上；`--purify-aware`
    是攻擊方之後做的，套在它後面。兩者的順序即 `transform(deliver(render))`，
    與實際發生的順序一致。

    走 STE 版：硬夾取在球外梯度為零，被夾住的座標拿不到訊號、再也回不來。
    前向仍然是硬夾取，所以**交出去的圖真的滿足界限**。
    """
    if not args.linf_deliver:
        return None

    def deliver(x_def):
        return clamp_residual_ste(x_def, x01, float(args.linf_deliver))

    return deliver


def _with_consistency(param, x01, args, loss_fn):
    """`--consistency-weight` 給定時，在主損失上加一項一致性懲罰。

    **評估函數也要包**。`run_param_pgd` 用 `loss_fn._fixed_eval` 判收斂，
    不包的話判的是另一個量——訓練最小化 A、收斂判定看 B，早停會在錯的地方
    觸發，而且不會有症狀。

    只接在有 `module` 的參數化上（相位族）。接到 DCT-Shield 或加性上等於改掉
    別人的方法，**寧可拒絕**。
    """
    if not args.consistency_weight:
        return loss_fn
    # **檢查屬性存不存在，不檢查它的值。** `PhaseParam.module` 在 `reset()`
    # 之前是 `None`，而 `reset()` 在 `run_param_pgd` 內部才發生；讀它的值等於
    # 在還沒建好的時候判它不合格。實際踩過：三個工作點全部被自己的守門擋下。
    # 只有相位族有這個屬性（`AdditiveParam`／DCT／位移場都沒有），故它足以
    # 擋掉「接到別人的方法上」。
    if not hasattr(param, "module"):
        raise SystemExit(
            f"--consistency-weight 只接在本方法的相位參數化上，"
            f"收到條件 {args.conditions}")

    term = make_consistency_term(lambda: param.module, x01,
                                 args.consistency_weight,
                                 steps=args.steps,
                                 decay_frac=args.consistency_decay)

    def combined(x_def):
        return loss_fn(x_def) + term(x_def)

    # 排程的推進掛在**訓練迴圈**上，不掛在損失的呼叫上——評估也會呼叫損失，
    # 讓它推進排程會使權重隨評估次數漂掉。
    combined._advance = term.advance
    base_eval = getattr(loss_fn, "_fixed_eval", None)
    if base_eval is not None:
        def combined_eval(x_def):
            return base_eval(x_def) + term(x_def)
        combined._fixed_eval = combined_eval
    return combined


def _with_flow_tv(param, x01, args, loss_fn):
    """`--flow-tau` 給定時，在主損失上加 stAdv 的流場總變差項。

    總目標變成原文 §3.2 的 `argmin_f L_adv + τ·L_flow`
    （`src/defense/stadv_flow.py`）。**`L_adv` 不是原文的那一個**：原文是
    分類器 logits 上的 Carlini–Wagner 式，這裡是 `--loss` 選到的擴散模型損失，
    故 τ 不可沿用原文的 0.05，它是 CSV 的欄位 `flow_tau`。

    **正則項作用在上採樣後的稠密場上**（`param.flow_field`），不是粗網格
    係數：原文的 `f` 就是逐像素的，而 `--warp-grid 512` 時上採樣退化成恆等，
    兩者相同。

    **評估函數也要包**（與 `_with_consistency` 同一條理由）：`run_param_pgd`
    用 `loss_fn._fixed_eval` 判收斂，不包的話訓練最小化 `L_adv + τ·L_flow`
    而收斂判定看 `L_adv`，早停會在錯的地方觸發，且沒有症狀。

    只接在有 `flow_field` 的參數化上（位移場族）。接到相位或 DCT 上等於把
    一個位移場的正則項加到不是位移場的東西上，**寧可拒絕**。
    """
    if not args.flow_tau:
        return loss_fn
    if not hasattr(param, "flow_field"):
        raise SystemExit(
            f"--flow-tau 只接在位移場參數化上（有 flow_field 的那一族），"
            f"收到條件 {args.conditions}")

    def term():
        return args.flow_tau * flow_tv_loss(
            param.flow_field(x01), eps=args.flow_eps,
            neighbourhood=args.flow_neighbourhood)

    def combined(x_def):
        return loss_fn(x_def) + term()

    advance = getattr(loss_fn, "_advance", None)
    if advance is not None:
        combined._advance = advance
    base_eval = getattr(loss_fn, "_fixed_eval", None)
    if base_eval is not None:
        def combined_eval(x_def):
            return base_eval(x_def) + term()
        combined._fixed_eval = combined_eval
    return combined


def _with_color_tv(param, x01, args, loss_fn):
    """`--color-tv` 給定時，在主損失上加色彩場的空間總變差項。

    為什麼需要它
    ────────────────────────────────────────────────────────────
    `runs/ip2p_color_hunt/README.md` 記到：把雙邊網格由 8×8 放大到 16×16 讓
    等失真倍率由 1.09 走到 1.30，但**把防禦圖變成明顯不自然的東西**——
    最佳化器拿到空間自由度之後就去畫高頻色彩噪點，臉部變形、彩虹斑塊。
    那違反「產物自然」這個前提，也把這一族原本的賣點（平滑的全域色彩改動）
    換掉了。

    這一項要問的是：**能不能保住容量帶來的效率，同時把場壓回平滑。**
    正則項作用在**網格係數本身**的空間差分上，因為那正是「色彩變換隨位置
    變化的快慢」；作用在輸出影像上會連原圖本身的邊緣一起罰，那是另一回事。

    亮度維（`luma_bins`）**不罰**：沿亮度變化是色調曲線的正常自由度
    （暗部與亮部本來就該有不同的處理），罰它等於把雙邊網格降級回全域曲線。

    **評估函數也要包**（與 `_with_flow_tv` 同一條理由）：不包的話訓練最小化
    `L + λ·TV` 而收斂判定看 `L`，早停會在錯的地方觸發且沒有症狀。

    只接在有 `a`（雙邊網格係數）的參數化上，其餘拒絕而不是靜默忽略。
    """
    if not args.color_tv and not args.color_tv_luma:
        return loss_fn
    if not hasattr(param, "identity_grid"):
        raise SystemExit(
            f"--color-tv 只接在雙邊網格參數化上（color_grid），"
            f"收到條件 {args.conditions}")

    def term():
        a = param.a
        if a is None:
            return torch.zeros((), device=x01.device, dtype=x01.dtype)
        # a 的形狀是 (1, 12, luma, gy, gx)。
        out = torch.zeros((), device=x01.device, dtype=a.dtype)
        if args.color_tv:
            dy = (a[..., 1:, :] - a[..., :-1, :]).abs().mean()
            dx = (a[..., :, 1:] - a[..., :, :-1]).abs().mean()
            out = out + args.color_tv * (dy + dx)
        if args.color_tv_luma:
            # 沿**亮度**維的差分。實測空間平滑救不了「不自然」：產物上的
            # 彩虹邊紋來自相鄰色調被映射到不同顏色，那是亮度維的自由度，
            # 不是空間維的。罰它等於要求「色調曲線本身是平滑的」，而那正是
            # 一個自然的調色該有的樣子。
            out = out + args.color_tv_luma * (
                a[..., 1:, :, :] - a[..., :-1, :, :]).abs().mean()
        return out

    def combined(x_def):
        return loss_fn(x_def) + term()

    advance = getattr(loss_fn, "_advance", None)
    if advance is not None:
        combined._advance = advance
    base_eval = getattr(loss_fn, "_fixed_eval", None)
    if base_eval is not None:
        def combined_eval(x_def):
            return base_eval(x_def) + term()
        combined._fixed_eval = combined_eval
    return combined


def _with_patch_tv(param, x01, args, loss_fn):
    """`--patch-tv` 給定時，在主損失上加補丁內容的總變差項。

    三個「美化」旋鈕裡的第三個（另兩個是 `--patch-tile` 與 `--patch-alpha`，
    它們改的是構造）。這一項要的是**柔和的色塊而不是逐像素噪點**：學出來的
    補丁預設長得像雜訊，那讀起來是「這塊壞掉了」而不是「這裡有一個標記」。

    作用在**可學張量本身**：平舖時那就是那一塊磚，故懲罰的是磚的內部結構，
    與磚之間的接縫無關（接縫由重複造成，不是自由度）。

    **評估函數也要包**（與 `_with_flow_tv` 同一條理由）：不包的話訓練最小化
    `L + λ·TV` 而收斂判定看 `L`，早停會在錯的地方觸發且沒有症狀。
    """
    if not args.patch_tv:
        return loss_fn
    if not hasattr(param, "_field"):
        raise SystemExit(
            f"--patch-tv 只接在補丁參數化上，收到條件 {args.conditions}")

    def term():
        c = param.c
        if c is None:
            return torch.zeros((), device=x01.device, dtype=x01.dtype)
        dy = (c[..., 1:, :] - c[..., :-1, :]).abs().mean()
        dx = (c[..., :, 1:] - c[..., :, :-1]).abs().mean()
        return args.patch_tv * (dy + dx)

    def combined(x_def):
        return loss_fn(x_def) + term()

    advance = getattr(loss_fn, "_advance", None)
    if advance is not None:
        combined._advance = advance
    base_eval = getattr(loss_fn, "_fixed_eval", None)
    if base_eval is not None:
        def combined_eval(x_def):
            return base_eval(x_def) + term()
        combined._fixed_eval = combined_eval
    return combined


def _subject_of(x01, texts, args):
    """受保護主體。`face` 走 ATR 的 Face+Hair，`clipseg` 走登記的名詞。"""
    if args.subject_source == "face":
        from src.defense.carrier_mask import face_subject_mask

        return face_subject_mask(x01, dilate=args.subject_mask_dilate,
                                 feather=args.subject_mask_feather)
    from src.defense.subject_mask import subject_mask

    return subject_mask(x01, texts, threshold=args.subject_mask_threshold,
                        dilate=args.subject_mask_dilate,
                        feather=args.subject_mask_feather)


def _carrier_of(x01, mask, args):
    """載體，含四個貼合與融入的旗標。四者全為 0 時與原本逐位元相同。

    順序是**吸附 → 挖掉臉框 → 內縮 → 往內羽化 → 分散**，每一步都只會讓權重
    變小或不變，故「主體那一側恆為 0」的保證在任何組合下都成立。

    挖掉臉框不是可選的：ATR 沒有「這一塊是皮膚不是衣服」的把關，特寫人像上
    它會把臉頰與額頭整片標成 Upper-clothes（實測一張 0.6561），那時載體就是
    人臉。MTCNN 的框是獨立於 ATR 的訊號。
    """
    from src.defense.carrier_mask import (MAX_AREA, MIN_AREA, carrier_mask,
                                          erode_mask, exclude_boxes,
                                          feather_inward, guided_refine,
                                          lattice_support, scatter_support)

    # `None` 代表沒給，沿用模組的定案值。**不要在 argparse 裡填那兩個常數**
    # ——填了之後模組改預設而 CLI 沒改，兩處會靜默分岔。
    c = carrier_mask(
        x01, args.patch_carrier,
        min_area=MIN_AREA if args.carrier_min_area is None
        else args.carrier_min_area,
        max_area=MAX_AREA if args.carrier_max_area is None
        else args.carrier_max_area)
    if args.carrier_refine:
        c = (guided_refine(c, x01, args.carrier_refine, 1e-3) > 0.5).to(c.dtype)
    from src.metrics.identity import face_boxes

    c = exclude_boxes(c, face_boxes(x01), margin=8)
    if args.carrier_erode:
        c = erode_mask(c, args.carrier_erode)
    if args.carrier_feather:
        c = feather_inward(c, args.carrier_feather)
    if args.carrier_ring:
        # 環帶不與衣物載體相交——它問的是「緊貼臉的那一圈」，而那一圈多半
        # 不是衣服。故它**取代**載體而不是與它相乘。
        from src.defense.carrier_mask import ring_support

        c = ring_support(mask.to(c.device), args.carrier_ring_inner,
                         args.carrier_ring)
    if args.carrier_lattice:
        # 點陣與分散大斑互斥（上面的守門已擋），故這裡是 elif 的語意。
        # 面積不由呼叫端指定，是 pitch 與 radius 的函數；實際拿到多少由
        # `patch_area` 那一欄說話。
        c = lattice_support(c * (mask.to(c.device) <= 0.0).to(c.dtype),
                            args.carrier_lattice, args.carrier_dot_radius)
    if args.carrier_scatter:
        legal = c * (mask.to(c.device) <= 0.0).to(c.dtype)
        # 種子走 `--seed`（防禦端的逐圖種子），不是 `--edit-seed`。
        # 名字寫錯時是 AttributeError 而不是靜默失效，但仍會整批死掉。
        c = scatter_support(legal, args.carrier_scatter, args.radius, args.seed)
    if args.carrier_target_area:
        # **在全部修整之後才對齊**：吸附、挖臉框、內縮、羽化每一步都會改變
        # 面積，先對齊再修整的話交出去的面積不是報表上那個數字。
        from src.defense.carrier_mask import match_area

        c = match_area(c, mask, args.carrier_target_area,
                       mode=args.carrier_match)
    return c


def attack_instruction(spec, image: str, category: str) -> str:
    """由攻擊指令目錄取出某張影像、某一類的指令。

    **目錄裡沒有該影像時拋錯。** 靜默沿用資料集自帶的句子會讓那一列的
    `attack_category` 欄寫著新的類別、而實際跑的是舊指令。
    """
    rec = (spec.get("images") or {}).get(image)
    if rec is None:
        raise SystemExit(
            f"{image} 沒有登記在攻擊指令目錄裡。**不沿用資料集自帶的句子**"
            "——那會讓 attack_category 欄與實際跑的指令對不上。")
    got = (rec.get("prompts") or {}).get(category)
    if got is None:
        raise SystemExit(f"{image} 的攻擊指令目錄裡沒有 {category} 這一類。")
    return got


def validate_attack_args(args) -> None:
    """攻擊指令的守門。**在載入權重之前**呼叫。"""
    if args.attack_category and args.attack_prompts is None:
        raise SystemExit(
            f"--attack-category {args.attack_category} 必須同時給 "
            "--attack-prompts：沒有目錄檔時指令會靜默沿用資料集自帶的句子，"
            "而報表上的 attack_category 欄仍然寫著新的類別。")
    if args.attack_prompts is not None and not args.attack_category:
        raise SystemExit("--attack-prompts 必須同時給 --attack-category。")
    if args.attack_prompts is not None and not args.attack_prompts.exists():
        raise SystemExit(f"找不到攻擊指令目錄 {args.attack_prompts}")


def carrier_apply_where(carrier, mask):
    """`載體 ∧ 主體補集`，供色彩族的 `apply_where` 用。

    **只有這一份實作。** 同一條算式散在多處時，漏改其中一處的結果是行為不
    一致而輸出看起來完全正常（`docs/DEFECTS.md` 的「政策謂詞散在多處」）。

    交集的兩邊都是硬條件：載體大於 0.5、主體遮罩逐像素等於零（羽化帶不算）。
    回傳的權重在主體上恆為 0，而 `w = 0` 的地方色彩族輸出**逐位元等於原圖**。
    """
    return ((carrier > 0.5) & (mask.to(carrier.device) <= 0.0)).to(mask.dtype)


def _with_patch_tint(param, x01, args, loss_fn):
    """`--patch-tint` 給定時，加一項「粗尺度上要像原本那件衣服」的懲罰。

        λ · mean_over_support( ( blur_σ(c) − blur_σ(x) )² )

    與 `--patch-tv` 針對的東西不同，兩者不可互相取代：tv 罰掉支撐內**所有**
    局部變化，於是花紋本身也被壓平（實測平舖配 tv 0.3 只剩基準的 32%，而
    兩者單獨用時是 86% 與 90%）。tint 只罰**低頻**——局部色調被拉向原本的
    衣服，高頻的花紋結構完全不受懲罰。

    模糊用**盒式平均**（`avg_pool2d` 加同尺寸的反射填充），不是高斯：σ 在
    這裡的意義是「多大的鄰域算同一個色調」，盒式讓那個尺度直接可讀，而且
    與 `--patch-tile` 的磚長可以並排比較。

    **評估函數也要包**（與 `_with_patch_tv` 同一條理由）：不包的話訓練最小化
    `L + λ·tint` 而收斂判定看 `L`，早停會在錯的地方觸發且沒有症狀。
    """
    if not args.patch_tint:
        return loss_fn
    if not hasattr(param, "_field"):
        raise SystemExit(
            f"--patch-tint 只接在補丁參數化上，收到條件 {args.conditions}")

    # **與 `--patch-lowfreq` 共用同一個實作**：同一個 σ 在兩個旗標下必須指
    # 同一個尺度，否則報表上兩者並排看起來像是可比的。
    from src.defense.patch_param import box_blur

    k = max(1, int(args.patch_tint_sigma))

    def blur(t):
        return box_blur(t, k)

    target = blur(x01).detach()

    def term():
        c = param.c
        if c is None:
            return torch.zeros((), device=x01.device, dtype=x01.dtype)
        # 平舖時 `c` 是一塊磚，要先攤成整張畫面才能與原圖對位。
        field = param._field(x01)
        sup = param.support.to(field.dtype)
        d = (blur(field) - target) * sup
        denom = sup.sum().clamp_min(1.0) * field.shape[1]
        return args.patch_tint * (d.pow(2).sum() / denom)

    def combined(x_def):
        return loss_fn(x_def) + term()

    advance = getattr(loss_fn, "_advance", None)
    if advance is not None:
        combined._advance = advance
    base_eval = getattr(loss_fn, "_fixed_eval", None)
    if base_eval is not None:
        def combined_eval(x_def):
            return base_eval(x_def) + term()
        combined._fixed_eval = combined_eval
    return combined


def _stage1_alpha(args, param) -> float:
    """階段一實際用的步長。**與 `run_param_pgd` 內的公式同一條**，寫兩次會在
    改動時只改到一邊，而症狀只是「階段二的步長比例不是你以為的那個」。"""
    if args.step_size is not None:
        return float(args.step_size)
    return step_scale_of(param) / max(1.0, args.steps * args.saturate_at)


def _run_stage2(x01, param, loss_fn, args):
    """分階段訓練的第二段：在階段一的解附近找一個比較耐淨化的鄰居。

    回傳 `(x_def, extras)`，`extras` 的每一欄都會寫進該列 CSV——信賴域退了
    幾次、最後守住多少、步長掉到哪裡，這些不記下來的話「沒有改善」與「安全繩
    一路咬著不放」在報表上長得一模一樣。
    """
    if args.update != "sign":
        raise SystemExit(
            f"--stage2-steps 只支援 sign 更新，收到 --update {args.update}。"
            "理由見 param_pgd.run_stage2_pgd 的 docstring（退參數而不退 Adam "
            "動量會讓步長的語意斷掉）。")
    transform = make_sequenced_ops_transform(
        args.stage2_ops, args.stage2_steps, order=args.stage2_order,
        seed=args.seed, ramp=bool(args.stage2_ramp))
    res = run_stage2_pgd(
        x01, param, loss_fn,
        steps=args.stage2_steps,
        alpha=_stage1_alpha(args, param) * args.stage2_step_scale,
        transform=transform,
        trust_frac=args.stage2_trust,
        check_every=args.stage2_check_every,
        log_every=1)
    extras = {
        "stage2_reverts": res.reverts,
        "stage2_checks": res.checks,
        "stage2_steps_run": res.steps_run,
        "stage2_alpha_init": round(res.alpha_init, 8),
        "stage2_alpha_final": round(res.alpha_final, 8),
        "stage2_gain_stage1": round(res.gain_stage1, 6),
        "stage2_gain_final": round(res.gain_final, 6),
        # 守住的比例。信賴域保證它 ≥ --stage2-trust；**低於門檻只可能出現在
        # 「一次都沒通過檢查」那種情形**，屆時這一欄會等於 1.0（退回階段一）。
        "stage2_gain_ratio": round(res.gain_final / res.gain_stage1, 6),
        "stage2_stopped_early": int(res.stopped_early),
    }
    return res.x_def, extras


def defend(ip2p, suite, cond, x01, args, loss_fn):
    """回傳 `(x_def, radius, unreachable, modified_from_paper, extras)`。

    兩條路徑：相位／加性走本專案共用的 sign PGD；DCT-Shield 走它自己論文的
    Algorithm 1。**兩者都不因為換了攻擊方而改動**——改的只有 `loss_fn` 裡的
    那個 `E`。

    `extras` 是要併進該列 CSV 的額外欄位，只有 `--deliver-jpeg` 開著時非空。
    """
    if args.stage2_steps and args.radius is None:
        # 預算模式對半徑二分搜，每一輪都是一次完整的階段一。階段二要接在
        # 「哪一輪」後面沒有唯一答案，而靜默只接在最後一輪會讓 CSV 的
        # `radius` 欄講的是階段一的半徑、圖卻是階段二的——**寧可拒絕**。
        raise SystemExit(
            "--stage2-steps 不可與預算模式（不給 --radius）併用："
            "二分搜的每一輪都是一次完整的階段一，階段二該接在哪一輪沒有定義。"
            "請明給 --radius。")
    if args.resume_weights is not None and args.radius is None:
        # 預算模式的每一輪二分搜都是一次完整的階段一，而且**每一輪的半徑都
        # 不同**：同一組權重載進不同半徑的輪次，投影會把它們夾成不同的東西，
        # 「續跑」指的是哪一輪沒有唯一答案。而此前這一支根本不呼叫
        # `_load_weights`，`resumed` 欄也不會被寫出來——給了旗標卻整批從零
        # 開始練，報表上完全看不出來。**寧可拒絕**。
        raise SystemExit(
            "--resume-weights 不可與預算模式（不給 --radius）併用："
            "二分搜的每一輪都是一次完整的階段一、半徑各不相同，"
            "續跑該接在哪一輪沒有定義。請明給 --radius。")
    if args.stage2_steps and cond in NO_OPT_CONDS:
        raise SystemExit(
            f"--stage2-steps 不可用於 {cond}：這個條件不做最佳化，"
            "沒有階段一的解可以接。")
    if args.deliver_jpeg and cond in DCT_ROTATE_CONDS + DCT_UNIFIED_CONDS:
        # `dct_rotate` 的參數就作用在量化後的整數係數上，`render` 出來的已經
        # 是 QD 品質的壓縮圖（交付即參數）。再套一次 `--deliver-jpeg` 是壓兩次，
        # 而且第二次的品質未必等於第一次，量出來的東西沒有意義。
        raise SystemExit(
            f"--deliver-jpeg 不可用於 {cond}：它自己就交付壓縮圖，"
            "交付品質請用 --dct-qd 指定")
    if args.deliver_jpeg and cond not in DELIVER_JPEG_CONDS:
        # 交付自壓是接在本方法的參數化後面的一步。套到 DCT-Shield 上等於把
        # 別人的方法改掉一半（它自己就把 δ 加在量化係數上），套到 AdvDrop／
        # 浮水印上則是換掉它們的輸出。**寧可拒絕，不要靜默照跑**。
        raise SystemExit(
            f"--deliver-jpeg 只接在本方法的參數化上，收到條件 {cond}。"
            f"允許的條件：{' '.join(DELIVER_JPEG_CONDS)}")
    if cond in WM_CONDS:
        # DJSMA（The Imaging Science Journal 2026）。無公開程式碼，由掃描 PDF
        # 逐頁判讀後依 Algorithm 1 與式 (7)–(9) 實作。
        #
        # **顯著圖必須換掉**：論文的式 (8)(9) 需要類別 logits，而擴散編輯防護
        # 沒有分類器。`saliency="grad"` 把 S± 換成本專案共用損失對該係數的
        # 偏導，其餘（一次一個係數、±1、E345、tau／mu 的意義）不變。那不是
        # 論文的方法，故 `modified_from_paper=True`。
        spec = DJSMASpec(
            name=cond, tau=args.wm_tau, mu=args.wm_mu,
            diagonals=tuple(args.wm_diagonals), q_embed=args.wm_q_embed,
            saliency="grad", modified_from_paper=True,
            modification_note=(
                "顯著圖由論文式 (8)(9) 的類別 logits 換成擴散編碼器目標損失"
                "對係數的偏導；本威脅模型沒有分類器"))
        res = run_djsma(x01, spec, loss_fn=loss_fn, log_every=0)
        # 強度旋鈕是 tau（l0 界），不是某個 eps——DJSMA 是貪婪 JSMA 不是 PGD。
        return res.x_def, float(spec.tau), False, True, {}

    if cond == "advdrop_max":
        # 不跑最佳化：直接把量化表整片推到上界，軟四捨五入的硬度取與最佳化
        # 最後一步相同的 `PAPER_ALPHA_LO`。回傳的「radius」欄是 eps。
        from src.baselines.advdrop import (  # noqa: E402
            PAPER_ALPHA_LO, init_q_tables, render_advdrop,
        )
        q = init_q_tables(x01, PAPER_Q_INIT + args.advdrop_eps)
        alpha = torch.tensor(PAPER_ALPHA_LO, device=x01.device, dtype=x01.dtype)
        with torch.no_grad():
            x_def = render_advdrop(x01, q, alpha, "rgb", False)
        return x_def, float(args.advdrop_eps), False, True, {}

    if cond in ADVDROP_CONDS:
        # AdvDrop（ICCV 2021）是唯一另一個明確的「非加性頻域」方法，也是本
        # 專案新穎性主張必須辨明的前例。**它原本不是編輯防護**，故有兩處
        # 明確改寫，兩者都必須出現在報表上：
        #
        #   損失   原文是分類的交叉熵 log p_y；本威脅模型沒有分類器，改用
        #          與 DCT-Shield 同型的 ‖E(x')‖ 類目標
        #   步長   原文式 (7) 是 sign 更新、隱含步長 1，但本專案在它自己的
        #          威脅模型上實測：50 步 × 步長 1 只走到 q 平均 7.9，未定向
        #          成功率 32.5%（論文 98.55–100%）；步長 4 才重現到 96.0%。
        #          見 runs/advdrop_repro/repro.csv
        spec = AdvDropSpec(
            name=cond, q_init=PAPER_Q_INIT, eps=args.advdrop_eps,
            steps=args.advdrop_steps, step_size=args.advdrop_step_size,
            modified_from_paper=True,
            modification_note=(
                "損失由分類 CE 換成擴散編碼器目標；步長 "
                f"{args.advdrop_step_size:g} 而非論文隱含的 1"),
            source="arXiv:2108.09034 §3.1／§4.3")
        res = run_advdrop(ip2p, x01, spec, loss_fn=loss_fn, log_every=0)
        return res.x_def, spec.eps, False, True, {}

    if cond in DCT_CONDS:
        if args.mode == "paper":
            notes = []
            if args.eps < PAPER_EPS:
                notes.append("eps 低於論文的 1，抗 JPEG 條件失效")
            if args.skip_dc:
                notes.append("排除 DC 係數——論文沒有這一步，是重現落差的檢定")
            spec = DCTShieldSpec(
                name=cond, q_alg=args.q_alg, eps=args.eps, gamma=PAPER_GAMMA,
                steps=args.dct_steps,
                channels=("Y",) if cond.endswith("_y") else ("Y", "Cb", "Cr"),
                skip_dc=args.skip_dc,
                modified_from_paper=bool(notes),
                modification_note="；".join(notes),
                source="arXiv:2504.17894 補充材料 Algorithm 1")
            # `--dct-loss paper`（預設）**不傳 loss_fn**：`run_dct_shield`
            # 用的是該篇自己的損失 `‖E(x')‖₂`（§4.2 末段）。傳別的損失進去
            # 等於把那篇的方法換掉一半，那是**消融不是 baseline**——兩者都
            # 只經過 VAE 編碼器，換掉不會拋錯也看不出來，故在此明寫，並在
            # `dct_loss` 欄逐列記下跑的是哪一個。
            #
            # `--dct-loss project` 是刻意的消融：`runs/ig_probe/` 量到
            # `image_guidance` 在同失真下比 `latent_norm` 多壓四成殘差，而
            # DCT-Shield 的**參數化**是目前位移／失真比最好的一個。兩者從未
            # 組合過。這一支一律標 `modified_from_paper`。
            if args.dct_loss == "project":
                spec = replace(
                    spec, modified_from_paper=True,
                    modification_note="；".join(
                        notes + [f"損失換成本專案的 {args.loss}"
                                 "（論文用 ‖E(x')‖₂），這是消融不是 baseline"]))
            res = run_dct_shield(
                ip2p, x01, spec, log_every=250,
                loss_fn=None if args.dct_loss == "paper" else loss_fn)
            return res.x_def, spec.eps, False, spec.modified_from_paper, {}
        raise SystemExit(
            "DCT-Shield 的預算對齊模式尚未接到 IP2P 線上。曲線協定（DEC-029）"
            "是掃 eps 畫取捨曲線，錨點由 tradeoff_curve.py 內插求得，"
            "不需要在這裡二分搜尋")

    param, lo, hi = build(cond, args.seed, block=args.block, r_min=args.r_min,
                          disp_field_grid=args.disp_field_grid,
                          hop=args.hop,
                          r_max=args.r_max,
                          quantile=args.quantile, gl_iters=args.gl_iters,
                          pixel_gate_sigma=args.pixel_gate_sigma,
                          gain_ratio=args.gain_ratio,
                          gate_edge_power=args.gate_edge_power,
                          freq_weight=args.freq_weight,
                          freq_weight_power=args.freq_weight_power,
                          survival_weight=args.survival_weight,
                          gain_weight=args.gain_weight,
                          channels=args.phase_channels,
                          spectral_floor=args.spectral_floor,
                          floor_gate=args.floor_gate,
                          floor_r_min=args.floor_r_min,
                          floor_r_max=args.floor_r_max,
                          floor_survival=args.floor_survival,
                          floor_envelope=args.floor_envelope,
                          floor_envelope_k=args.floor_envelope_k,
                          floor_envelope_scope=args.floor_envelope_scope,
                          theta_budget=args.theta_budget,
                          coarsen=args.coarsen,
                          warp_grid=args.warp_grid,
                          dct_qd=args.dct_qd, dct_pairing=args.dct_pairing,
                          dct_gate=args.dct_gate,
                          warp_init_std=args.warp_init_std,
                          dct_mode=args.dct_mode,
                          dct_plane_weight=args.dct_plane_weight,
                          color_pieces=args.color_pieces,
                          color_bound_mode=args.color_bound_mode,
                          color_grid=args.color_grid,
                          color_luma_bins=args.color_luma_bins,
                          color_rand_draw=args.color_rand_draw,
                          patch_placement=args.patch_placement,
                          patch_count=args.patch_count,
                          patch_crop_keep=args.patch_crop_keep,
                          patch_init=args.patch_init,
                          patch_tile=args.patch_tile,
                          patch_res=args.patch_res,
                          patch_palette=args.patch_palette,
                          patch_palette_temp=args.patch_palette_temp,
                          patch_seeds=args.patch_seeds,
                          patch_seed_temp=args.patch_seed_temp,
                          patch_polar=args.patch_polar,
                          patch_polar_bins=args.patch_polar_bins,
                          patch_alpha=args.patch_alpha,
                          patch_lowfreq=args.patch_lowfreq,
                          patch_chroma=args.patch_chroma)
    # 主體之外才動：把 `apply_where` 設成 `1 − 主體遮罩`，遮罩由 CLIPSeg 用
    # 一句文字指出（`src/defense/subject_mask.py`）。文字是防禦方本來就知道的
    # 主體名稱，不是攻擊指令。**只有支援 `apply_where` 的參數化吃得到這個
    # 旗標**，其餘一律拋錯而不是靜默忽略——靜默忽略的症狀是「主體被改了但
    # 報表上寫著有遮罩」。
    run_extras: dict = {"subject_mask_text": ""}
    if cond in PATCH_CONDS:
        # 補丁**必須**知道主體在哪裡才放得下去，故遮罩不是選用的。
        import yaml as _yaml

        from src.defense.subject_mask import mask_stats, subject_mask

        # `face` 來源的主體由 ATR 的 Face+Hair 直接給，**不需要名詞目錄**。
        # 仍去查目錄的話，換一個目錄檔就會死在「objects 裡沒有這張影像」——
        # 而那個錯誤訊息與真正的原因無關。
        if args.subject_source == "face":
            texts = []
        else:
            if args.subject_mask is None:
                raise SystemExit(
                    f"條件 {cond} 需要 --subject-mask：補丁要放在主體之外，"
                    "沒有主體遮罩就不知道「之外」是哪裡。**不預設放中央**。")
            spec = _yaml.safe_load(args.subject_mask.read_text(encoding="utf-8"))
            texts = (spec.get("objects") or {}).get(args._cur_image)
            if not texts:
                raise SystemExit(
                    f"{args.subject_mask} 的 objects 裡沒有 {args._cur_image} 的"
                    "主體名稱。**不猜**。")
            texts = [texts] if isinstance(texts, str) else list(texts)
        m = _subject_of(x01, texts, args)
        param.mask = m
        run_extras["subject_mask_text"] = ("" if args.subject_source == "face"
                                           else " | ".join(texts))
        run_extras.update(mask_stats(m))
        # 語意載體：把支撐再交集一層衣物區域。**在遮罩之後掛**，因為交集
        # 的另一半就是遮罩的補集，而「主體逐位元不動」由 `reset` 裡的構造
        # 保證（見 `src/defense/carrier_mask.py` 與 `PatchParam.reset`）。
        if args.patch_carrier != "none":
            from src.defense.carrier_mask import carrier_stats

            c = _carrier_of(x01, m, args)
            param.set_carrier(c, args.patch_carrier)
            run_extras.update(carrier_stats(c))
    # 補丁族在上面已經把遮罩用掉了（它用來決定補丁放哪裡，不是 apply_where），
    # 故這一段只跑色彩族。兩個用途共用同一個旗標與同一組形狀參數，但意義不同。
    wants_mask = (args.subject_mask is not None
                  or args.subject_source == "face")
    if wants_mask and cond not in PATCH_CONDS:
        import yaml as _yaml

        from src.defense.subject_mask import mask_stats

        if not hasattr(param, "apply_where"):
            raise SystemExit(
                f"--subject-mask 用在 {cond} 上，但該參數化沒有 apply_where。"
                f"支援的是色彩族（color_curve／color_grid）。")
        # `face` 主體由 ATR 的 Face+Hair 直接給，不需要名詞目錄——與補丁族
        # 共用 `_subject_of`，兩族的「受保護主體」因此是同一個物件而不是兩份
        # 各自演化的實作。
        if args.subject_source == "face":
            texts = []
        else:
            spec = _yaml.safe_load(args.subject_mask.read_text(encoding="utf-8"))
            texts = (spec.get("objects") or {}).get(args._cur_image)
            if not texts:
                raise SystemExit(
                    f"{args.subject_mask} 的 objects 裡沒有 {args._cur_image} 的"
                    f"主體名稱。**不猜**——沒有主體名稱就沒有「主體之外」"
                    "這個概念。")
            texts = [texts] if isinstance(texts, str) else list(texts)
        m = _subject_of(x01, texts, args)
        # 載體給定時，色彩重映射**只作用在衣服上**：
        # `apply_where = 載體 ∧ 主體補集`。不給載體時維持原本的「主體之外」。
        if args.patch_carrier != "none":
            from src.defense.carrier_mask import carrier_stats

            c = _carrier_of(x01, m, args)
            param.apply_where = carrier_apply_where(c, m).to(x01)
            run_extras.update(carrier_stats(c))
            run_extras["patch_carrier"] = args.patch_carrier
        else:
            param.apply_where = (1.0 - m).to(x01)
        run_extras["subject_mask_text"] = " | ".join(texts)
        run_extras.update(mask_stats(m))
    q_deliver = deliver_quality(args)
    # **在兩條路徑分岔之前包**：預算模式（`fit_to_budget`）內層自己呼叫
    # `run_param_pgd`，包在分岔之後那一支就會靜默少掉正則項，而報表上的
    # `flow_tau` 仍然寫著一個非零值。`--flow-tau 0`（預設）時原樣回傳，
    # 呼叫路徑逐位元不變。
    loss_fn = _with_flow_tv(param, x01, args, loss_fn)
    loss_fn = _with_color_tv(param, x01, args, loss_fn)
    loss_fn = _with_patch_tv(param, x01, args, loss_fn)
    loss_fn = _with_patch_tint(param, x01, args, loss_fn)
    if args.attn_weight:
        # **加項不是取代**：單獨最佳化「注意力落在補丁上」可以靠把補丁變得
        # 極端顯眼拿到高分，而那不保證人沒被改。
        from src.defense.attention_loss import make_attention_term

        # **檢查屬性存不存在，不檢查它的值**（`_with_consistency` 記過同一個
        # 坑）。`PatchParam.support` 在 `reset()` 之前是 `None`，而 `reset()`
        # 在 `run_param_pgd` 內部才發生——原本這裡寫的是
        # `getattr(param, "support", None) is None`，於是 `attn` 那個臂
        # **從來沒有跑起來過**：守門每次都在自己身上觸發。
        if not hasattr(param, "support"):
            raise SystemExit(
                "--attn-weight 只接在補丁族的參數化上（要有 support），"
                f"收到條件 {args.conditions}")
        _attn_term = make_attention_term(
            # 傳 callable 而不是值：支撐要到 `reset()` 之後才存在。
            ip2p, region=lambda: (None if param.support is None
                                  else param.support.to(x01.dtype)),
            text_embeds=_attack_embeds(), weight=args.attn_weight,
            zt_mode=args.ig_zt, x_clean=x01,
            t_min=args.ig_t_min, t_max=args.ig_t_max, seed=args.seed)
        _attn_base = loss_fn

        def _with_attn(x_def):
            return _attn_base(x_def) + _attn_term(x_def)

        # 評估函數**不含**注意力項：收斂監看的是主損失，而注意力項每一步重抽
        # 指令與 t，把它放進固定評估會讓那個評估不再決定性。
        _fe = getattr(_attn_base, "_fixed_eval", None)
        if _fe is not None:
            _with_attn._fixed_eval = _fe
        _adv = getattr(_attn_base, "_advance", None)
        if _adv is not None:
            _with_attn._advance = _adv
        loss_fn = _with_attn
    # 只有 `--flow-tau > 0` 的列是「加了原文沒有的 eps、且鄰域由我方指定」的
    # 移植，那一欄必須說實話。
    modified = bool(args.flow_tau)

    def dists_of(a, b):
        return float(suite.pairwise(b, a)["dists"])

    if args.radius is not None:
        param.set_radius(args.radius)
        # **續跑的載入掛在 `run_param_pgd` 的 `post_reset` 上，不在這裡做。**
        # 此處的參數張量還不存在——相位族的 `param.module` 要到 `reset()` 才
        # 被建出來，`params()` 讀 `self.module.theta` 會是
        # `AttributeError: 'NoneType' object has no attribute 'theta'`
        # （實際發生過，整格沒有產出）；加性／明暗／位移場族的 `delta`／`m`／
        # `c` 同樣是 `None`。而 `reset()` 又在 `run_param_pgd` 內部，它**換掉
        # 張量本身**，所以就算在這裡先自己 `reset()` 再載，載進去的值也會被
        # 那一次 `reset()` 清成零——不拋錯、不留症狀，跑完看起來正常但其實
        # 是從零練的。掛在 `post_reset` 上是唯一同時滿足「張量已存在」與
        # 「此後不會再被 reset 清掉」的時間點。
        resumed_cell = [0]

        def _resume_into(p):
            resumed_cell[0] = _load_weights(p, x01, args, cond)

        loss_fn = _with_consistency(param, x01, args, loss_fn)
        res = run_param_pgd(x01, param, loss_fn, steps=args.steps,
                            post_reset=(None if args.resume_weights is None
                                        else _resume_into),
                            seed=args.seed,
                            eval_fn=getattr(loss_fn, "_fixed_eval", None),
                            eval_every=args.eval_every,
                            patience=args.patience, min_delta=args.min_delta,
                            saturate_at=args.saturate_at,
                            update=args.update, step_size=args.step_size,
                            log_every=max(1, args.steps // 40),
                            deliver=_deliver_projection(args, x01),
                            step_hook=getattr(loss_fn, "_advance", None),
                            transform=_forward_transform(args))
        x_raw, radius, unreachable = res.x_def, param.radius, False
        # **收斂軌跡逐圖寫成一份 CSV**，不只留在 stdout 的 log 裡：判收斂看的
        # 是 `eval` 那一欄，它必須能被重讀與畫圖。停止原因也逐列記下——
        # 「跑滿步數」與「早停」在結果 CSV 上分不出來的話，就不知道那一格
        # 到底收斂了沒有。
        run_extras["resumed"] = resumed_cell[0]
        # **重播模式必須真的載到權重。** `--steps 0` 的用途是「同一張防禦圖
        # 接不同的攻擊指令」——防禦與指令無關（`L_ig` 只吃空字串的文字嵌入），
        # 重跑等於把同一個最佳化再算一次。但 `_load_weights` 找不到檔案時
        # 回 0 而不拋錯，於是這一格會交出**未經最佳化的原圖**當防禦圖，
        # 而 CSV 的每一欄看起來都正常。這是整批唯一會讓「沒有防禦」偽裝成
        # 「防禦無效」的路徑，故擋在這裡。
        if args.steps == 0 and not resumed_cell[0]:
            raise SystemExit(
                f"--steps 0 是重播模式，但 {args.resume_weights} 底下沒有 "
                f"{args._cur_image} 的權重檔。這一格會交出未經最佳化的原圖，"
                "而報表上看不出來。先確認防禦那一批已經跑完並存了 __w.pt。")
        if hasattr(param, "content_stats"):
            # 兩個約束都在夾取之前成立，夾取之後不一定。**量出來寫進 CSV**，
            # 不靠 docstring 宣稱——夾取把它破壞掉時不會有任何症狀。
            run_extras.update(param.content_stats(x01))
        if args.save_weights:
            run_extras["_weights"] = [t.detach().cpu().clone()
                                      for t in param.params()]
        # 學出來的包絡參數。**這是結果不是設定**：旗標那三欄記的是「有沒有
        # 開、幾個凸包、乘在哪一半」，這裡記的是 PGD 把凸包放到了哪裡、收得
        # 多窄、beta 收到多少。關著時是空的 dict，欄位不出現。沒有它就只能
        # 從防禦圖用眼睛猜，而「包絡有沒有真的動」與「動了有沒有用」是兩件
        # 事，前者必須是數字。
        if hasattr(param, "envelope_state"):
            run_extras.update(param.envelope_state())
        run_extras["stop_reason"] = res.stop_reason
        run_extras["stopped_at"] = res.stopped_at
        run_extras["best_eval"] = ("" if res.best_eval is None
                                   else round(res.best_eval, 8))
        if args.flow_tau:
            # 最後那一個場的 `L_flow` 讀數（**未乘 τ**）。含 eps 帶進來的常數
            # 偏移 `鄰居對數 × sqrt(eps)`，跨 eps 比較時要扣掉，見
            # `src/defense/stadv_flow.py`。
            with torch.no_grad():
                run_extras["flow_loss"] = round(float(flow_tv_loss(
                    param.flow_field(x01), eps=args.flow_eps,
                    neighbourhood=args.flow_neighbourhood)), 6)
        if hasattr(param, "geometry"):
            # 要求的面積與**拿到的**面積不一定相同（邊長取整到偶數），而位置
            # 是搜出來的。故實際幾何逐列寫出，不從 radius 反推。
            run_extras.update(param.geometry())
        if args.eval_every and res.history:
            run_extras["_trace"] = [
                {"condition": cond, "radius": round(param.radius, 4), **h}
                for h in res.history]
        if args.stage2_steps:
            x_raw, run_extras = _run_stage2(x01, param, loss_fn, args)
        if cond in DCT_UNIFIED_CONDS:
            # 交出去的整數位移長什麼樣。**`delta_within_1` 決定新穎性怎麼寫**
            # ——比例高就代表我們動的幾乎全在 DCT-Shield 的 eps=1 球裡，論文
            # 只能主張「約束不同」不能主張「動作不同」。
            run_extras.update(param.delta_stats())
    else:
        # **二分搜的失真要量在交付的圖上**，否則預算欄講的是一張沒有人會拿到
        # 的圖。`transform` 在關閉交付自壓時維持 `None`，與加入這個旗標之前
        # 逐位元相同——此前 `fit_to_budget` 從來沒有拿過 `--purify-aware` 的
        # 算子，改成一律傳會靜默改掉既有的預算模式批次。
        def budget_dists(a, b):
            return dists_of(a if q_deliver is None
                            else jpeg_roundtrip(a, q_deliver), b)

        out = fit_to_budget(x01, param, loss_fn, budget_dists, args.budget,
                            lo=lo, hi=hi, steps=args.steps, seed=args.seed,
                            rounds=args.rounds,
                            transform=(None if q_deliver is None
                                       else _forward_transform(args)))
        x_raw = out.x_def
        radius = out.radius
        unreachable = bool(out.history[-1].get("unreachable", False))

    if q_deliver is None:
        return x_raw, radius, unreachable, modified, dict(run_extras)

    # **交付的是壓縮後的圖**，不是最佳化直接吐出來的那張。這一步是本實驗與
    # 已否決的 `--purify-aware` 唯一的差別，理由見 `_forward_transform`。
    x_def = jpeg_roundtrip(x_raw, q_deliver)
    d_raw = (x_raw - x01).detach()
    d_del = (x_def - x01).detach()
    # 第二個基準：把 JPEG 對**乾淨影像本身**的重建誤差扣掉。兩者在真實影像上
    # 差 0.3% 以內（實測），並列是為了讓「保留率」不必先講清楚拿哪一張當基準
    # 才能讀。
    d_base = (x_def - jpeg_roundtrip(x01, q_deliver)).detach()
    sq = float((d_raw * d_raw).sum())
    n_raw, n_del = sq ** 0.5, float((d_del * d_del).sum()) ** 0.5
    dot = float((d_del * d_raw).sum())
    extras = {
        # 交付殘差在「最佳化前擾動方向」上的分量。判準寫在
        # `runs/ip2p_deliver_jpeg/README.md`：隨機擾動在 QD=0.75 下是 0.22，
        # 沒有明顯高於它就代表最佳化沒有學會落在量化格點上。
        "deliver_retention": round(dot / sq, 5) if sq > 0 else 0.0,
        "deliver_cosine": round(dot / (n_raw * n_del), 5) if n_raw * n_del > 0 else 0.0,
        "deliver_retention_base": (round(float((d_base * d_raw).sum()) / sq, 5)
                                   if sq > 0 else 0.0),
        "deliver_rms_raw": round(n_raw / d_raw.numel() ** 0.5, 6),
        "deliver_rms_out": round(n_del / d_del.numel() ** 0.5, 6),
    }
    extras.update(run_extras)
    return x_def, radius, unreachable, modified, extras


def weights_path(root: Path, name: str, cond: str) -> Path:
    return root / f"{name}__{cond}__w.pt"


def _load_weights(param, x01, args, cond) -> int:
    """把存下來的參數載回 `param`，回傳 1（載到）或 0（沒有對應檔）。

    **形狀不合就拋錯，不靜默略過**：形狀不合代表這批的構造與存檔時不同
    （換了 block／hop／K），那時「續跑」載進去的是別的東西。
    """
    src = weights_path(args.resume_weights, args._cur_image, cond)
    if not src.exists():
        return 0
    saved = torch.load(src, map_location="cpu")
    ps = param.params()
    if len(saved) != len(ps):
        raise SystemExit(
            f"{src} 有 {len(saved)} 個張量，本次的參數化有 {len(ps)} 個"
            "——構造不同，不可續跑")
    with torch.no_grad():
        for t, v in zip(ps, saved):
            if tuple(t.shape) != tuple(v.shape):
                raise SystemExit(
                    f"{src} 的張量形狀 {tuple(v.shape)} 與本次的 "
                    f"{tuple(t.shape)} 不符——構造不同，不可續跑")
            t.copy_(v.to(device=t.device, dtype=t.dtype))
    param.project()
    return 1


def make_encoder_loss(ip2p, name: str, y_target):
    """只讀編碼器的三個損失。`image_guidance` 回傳 None（逐圖建立）。

    抽成函式而不是留在 `main` 裡，是為了讓 `latent_norm_max` 的**符號**測得到：
    符號寫反不會拋錯也不會有症狀，它只會安靜地退化成 `latent_norm`，而報表上
    的 `loss` 欄仍寫著 `latent_norm_max`。
    """
    if name == "latent_norm_max":
        # **符號相反**：PGD 一律最小化，故推大模長要回傳負值。
        # 不加上界——編碼器的輸入被 [0,1] 夾住，模長本來就有界。
        def loss_fn(x01_):
            return -ip2p.encode_image(x01_).flatten().norm(p=2)
        return loss_fn
    if name == "latent_norm":
        # DCT-Shield §4.2 的目標。與 `make_latent_norm_loss` 同式，這裡直接寫
        # 出來避免把 baseline 模組的預設綁進相位臂。
        def loss_fn(x01_):
            return ip2p.encode_image(x01_).flatten().norm(p=2)
        return loss_fn
    if name == "image_guidance":
        # 逐圖建立（見 `make_base_loss`），這裡不建，留 None 讓漏接當場拋錯
        # 而不是靜默用到別的損失。
        return None
    return make_encoder_target_loss(ip2p, y_target)


def build_parser() -> argparse.ArgumentParser:
    """CLI 的定義。

    與 `main()` 分開是為了讓測試能在不載入 IP2P 權重的情況下檢查旗標與
    預設值。此前 parser 埋在 `main()` 裡，於是這支驅動的 import 破損了也
    沒有任何測試會發現——`dct_wm` 那一支引用的 `WatermarkSpec` 早已改名為
    `DJSMASpec`，而整個檔案在被修好之前根本 import 不進來。
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--conditions", nargs="+", default=["phase", "dct_shield"])
    ap.add_argument("--images", nargs="+", default=None, help="分片用")
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="OmniEdit 有 90 張帶兩條指令；預設一律取第 0 條")
    ap.add_argument("--target", type=Path, default=Path("data/targets/gray.png"))
    ap.add_argument("--loss",
                    choices=("encoder_target", "latent_norm", "latent_norm_max",
                             "image_guidance", "edit_divergence", "identity",
                             "facelock"),
                    default="encoder_target",
                    help="encoder_target = ‖E(x_def) − E(y_target)‖²（本專案既有）；"
                         "latent_norm = ‖E(x_def)‖₂（DCT-Shield §4.2 的目標）。"
                         "後者只壓長度、單調且無方向要求，前者要同時對長度與"
                         "方向。2026-08-21 加這個旗標是為了把「輸給 DCT-Shield "
                         "的部分是損失函數還是參數化」分開量。"
                         "image_guidance = ‖ε(z_t, E_img(x'), ∅) − ε(z_t, 0, ∅)‖²，"
                         "即直接要求 IP2P 取樣式裡的影像引導項消失；"
                         "edit_divergence = −‖x̂₀(z_t; x_def) − x̂₀(z_t; x)‖²，"
                         "**直接以防禦效果為目標**：不指定要把條件推到哪裡，"
                         "只要求模型建出來的東西離原圖的結果遠。文字條件仍取"
                         "空字串——威脅模型的前提是攻擊指令未知。"
                         "latent_norm 是它的**逐點版本**（把影像條件推向零"
                         "張量），本項只要求 UNet 的**反應**相同，可行集大得多。"
                         "**這是本專案第一個讀 UNet 的損失**。"
                         "latent_norm_max = **−**‖E(x_def)‖₂，也就是把 latent "
                         "推**離**分布而不是壓到零。理由：IP2P 用 "
                         "classifier-free guidance 訓練，訓練時本來就會隨機丟掉"
                         "影像條件，所以 `c_I = 0` 是模型**受訓過**的狀態——零"
                         "不是怪異的點，是最熟悉的那一點。模長遠大於正常值的"
                         "條件則從未出現在訓練裡。**這個方向從未跑過**。"
                         "facelock = 身分損失接在 **VAE 的一次往返**上，"
                         "不經過 UNet（arXiv:2411.16832）。判準（臉）與模組"
                         "（走到哪裡）是兩個獨立的軸，本專案此前只填了對角線："
                         "latent_norm 只走 VAE 但沒有臉，identity 有臉但走了"
                         "UNet。**受保護主體在像素上凍結，故重建圖的臉只能經由"
                         "VAE 的感受野被支撐內的內容影響**——這一批問的就是"
                         "那件事做不做得到")
    # ---- image_guidance 的四個設定（見 src/defense/image_guidance_loss.py）----
    ap.add_argument("--ig-zt", choices=ZT_MODES, default=None,
                    help="`z_t` 的抽法。**必填，沒有預設**：IP2P 由純噪聲"
                         "起步，中間步的 z_t 分布依賴條件、無法解析，兩個"
                         "候選都是近似（diffuse_src = 原圖 latent 的前向"
                         "擴散；noise = 純噪聲）。按 CLAUDE.md「查不到的參數"
                         "設為必填，不要填看起來合理的預設」")
    ap.add_argument("--ig-t-min", type=int, default=1,
                    help="抽樣時間步的下界（含）")
    ap.add_argument("--ig-t-max", type=int, default=1000,
                    help="抽樣時間步的上界（含）。預設涵蓋整個排程，因為"
                         "影像條件在整條軌跡上都作用；不動點項那邊取小 t 是"
                         "因為它對齊的是淨化器實際走的噪聲尺度，兩者理由不同")
    ap.add_argument("--ig-samples", type=int, default=1,
                    help="每一步平均幾組 (t, eps)。代價線性增加")
    ap.add_argument("--ig-weight", choices=("uniform", "subject"),
                    default="uniform",
                    help="影像引導殘差的空間平均要不要加權。uniform 是主線，"
                         "與加上這個旗標之前逐位元相同。subject 只對**受保護"
                         "主體的 latent token** 負責——`runs/ig_probe/"
                         "by_region_*.csv` 量到原圖殘差的 64% 落在主體上而"
                         "主體只佔 53% 的面積，均勻平均等於把預算平均花在"
                         "整張畫面。這是消融")
    ap.add_argument("--ig-weight-catalogue", type=Path,
                    default=Path("data/decoy_catalogue.yaml"),
                    help="`--ig-weight subject` 的主體名詞來源。與 "
                         "--subject-mask 是**兩件事**：那個限制擾動長在哪裡，"
                         "這個只改損失的權重，擾動仍可長在全圖")
    # ---- 收斂與 early stop（所有損失共用）----
    ap.add_argument("--eval-every", type=int, default=0,
                    help="每幾步用一組**固定**的抽樣評估一次並寫進 trace.csv。"
                         "0 = 關閉，逐位元等於加這個旗標之前。隨機目標的逐步"
                         "損失是取樣變異不是收斂訊號，判收斂一律看這一欄")
    ap.add_argument("--eval-draws", type=int, default=8,
                    help="固定評估用幾組 (t, eps)。只影響評估的雜訊，不影響訓練")
    ap.add_argument("--eval-seed", type=int, default=99991,
                    help="固定評估的抽樣種子。與訓練的 --seed 分開，"
                         "免得評估點正好是訓練走過的那幾個")
    ap.add_argument("--patience", type=int, default=0,
                    help="連續幾次評估沒有比歷史最佳改善超過 --min-delta 就"
                         "停。0 = 不早停，跑滿 --steps")
    ap.add_argument("--save-weights", action="store_true",
                    help="把最佳化後的參數張量存成 `<影像>__<條件>__w.pt`。"
                         "存的是參數本身不是防禦圖——防禦圖不可逆推回參數"
                         "（重疊相加是有損投影，即 FND-049 的 amp_dev），"
                         "沒存就只能從零重跑")
    ap.add_argument("--patch-lowfreq", type=int, default=0, metavar="SIGMA",
                    help="把補丁內容的低頻**換成原圖的**（0 = 關）。σ 與 "
                         "--patch-tint-sigma 同一個尺度、共用同一個盒式模糊。"
                         "這是約束不是懲罰：不必調權重，也不與主損失搶預算。")
    ap.add_argument("--patch-chroma", action="store_true",
                    help="凍結亮度，只讓色度可學。BT.601 權重和為 1，故"
                         "夾取之前 Y 逐位元等於原圖，褶皺與陰影完整保留。")
    ap.add_argument("--carrier-min-area", type=float, default=None,
                    metavar="A",
                    help="載體面積下限。預設沿用 carrier_mask.MIN_AREA。")
    ap.add_argument("--carrier-max-area", type=float, default=None,
                    metavar="A",
                    help="載體面積上限。預設沿用 carrier_mask.MAX_AREA（0.60）"
                         "；--patch-carrier background 幾乎一定要放寬。")
    ap.add_argument("--carrier-ring", type=int, default=0, metavar="OUTER",
                    help="支撐換成緊貼受保護主體外緣的**環帶**，往外到 OUTER "
                         "像素（0 = 關）。賭的是「碰到**對的** token」而不是"
                         "「碰到所有 token」：VAE 是卷積的，臉部 token 的表示"
                         "受鄰近像素影響，而臉本身凍結、緊貼它的那一圈沒有。")
    ap.add_argument("--carrier-ring-inner", type=int, default=8, metavar="INNER",
                    help="環帶與主體之間留的間隙。**不可為 0**：主體遮罩已經"
                         "往外羽化過，貼著零邊界會讓支撐與羽化帶重疊，實際"
                         "拿到的環帶比要求的窄而報表上看不出來。")
    ap.add_argument("--prompt-eot", action="store_true",
                    help="影像引導損失對**攻擊指令的分布**取期望，而不是只用"
                         "空字串。取整份目錄裡所有句子去重，不是這張影像那三句"
                         "——只取那三句等於偷看這一格的攻擊。")
    ap.add_argument("--attn-weight", type=float, default=0.0, metavar="LAMBDA",
                    help="加一項「把編輯指令的 cross-attention 吸到補丁上」"
                         "（0 = 關）。這是**改道**而不是破壞：模型去改標記而"
                         "不是改人。探索性質——「注意力落在哪裡」與「輸出被改"
                         "在哪裡」的關聯本專案還沒量過（attention_probe.py）。")
    ap.add_argument("--id-layout-weight", type=float, default=0.0,
                    metavar="LAMBDA",
                    help="--loss identity 的場景保持項權重（0 = 只毀身分）。"
                         "要求「臉以外的場景與模型從原圖建出來的一樣」，產物"
                         "因此是「同一個場景、換一張臉」而不是整張圖被毀。")
    ap.add_argument("--id-box-margin", type=float, default=0.35,
                    help="--loss identity 的裁臉框往外放寬多少（比例）。"
                         "編輯後臉會移動，而框是由原圖算的固定框。")
    ap.add_argument("--carrier-lattice", type=int, default=0, metavar="PITCH",
                    help="把支撐換成**規則格點上的小圓斑**，間距 PITCH 像素"
                         "（0 = 關）。SD 的 VAE 降採樣 8 倍，故 PITCH 8 時每一個"
                         " latent 格恰好被碰到一次：token 覆蓋率 100% 而像素面積"
                         "只有 π·r²/PITCH²。實測效果由 token 覆蓋率決定而不是"
                         "像素面積，這是「攤平整張畫面」以外另一條達到滿覆蓋的"
                         "路，而且產物是看得出來的規則點陣。")
    ap.add_argument("--carrier-dot-radius", type=float, default=2.0,
                    metavar="R",
                    help="--carrier-lattice 的圓斑半徑（像素）。2·R 必須小於"
                         " PITCH，否則圓斑相連、支撐退化成一整片。")
    ap.add_argument("--carrier-match", choices=("erode", "scale"),
                    default="erode",
                    help="--carrier-target-area 用哪一種縮法。erode 把邊界往內"
                         "縮（保留輪廓、一小塊全改）；scale 把整片權重乘上一個"
                         "常數（形狀不變、一大片各改一點）。兩者的 mean(w) 相同"
                         "而視覺與機制不同，故是兩個工作點不是一個。")
    ap.add_argument("--carrier-target-area", type=float, default=0.0,
                    metavar="A",
                    help="把載體往內縮，使**合法面積**（載體 ∩ 主體補集，"
                         "也就是 patch_area 記的那個量）恰好等於 A。"
                         "0 = 不對齊。用途是讓「放在背景」與「放在衣服」"
                         "同面積，否則等於同時動了位置與大小兩個變因。")
    ap.add_argument("--resume-weights", type=Path, default=None,
                    help="從這個目錄載入同名的 `__w.pt` 當**起點**續跑。"
                         "找不到對應檔案的影像照常從零起步，並在 CSV 的 "
                         "`resumed` 欄記 0，不靜默假裝續跑過")
    ap.add_argument("--skip-existing", action="store_true",
                    help="已經有 `__def.png` 的影像直接跳過。用於中途加旗標"
                         "重啟時保住已完成的部分")
    ap.add_argument("--min-delta", type=float, default=0.0002,
                    help="早停的相對改善門檻（每次評估）。**這是收斂判準不是"
                         "效果判準**。0.002 太大：實測 latent_norm 在第 6500 步"
                         "仍以每 100 步 0.17%% 單調下降、參數還在成長，卻因為"
                         "構不到 0.2%% 而被判定停滯。門檻必須小於曲線真正變平"
                         "之前的改善率，否則停下來的是一條還在降的曲線")
    # 相位／加性
    ap.add_argument("--color-pieces", type=int, default=64,
                    help="color_curve 的分段數 K。AdvCF 的 ImageNet 設定是 64；"
                         "它同時是曲線的自由度（3K）與亮度解析度。")
    ap.add_argument("--color-bound-mode", choices=("symmetric", "advcf"),
                    default="symmetric",
                    help="color_curve 的投影盒。advcf 是原程式的 "
                         "clamp(theta, 1/K, (1+r)/K)，下界等於初始值，故 sign "
                         "更新在起點只有往上可行；symmetric 是本專案指定的對數"
                         "對稱盒 [1/(K(1+r)), (1+r)/K]。**這一欄要進報表。**")
    ap.add_argument("--color-grid", type=int, default=8,
                    help="color_grid 的空間網格邊長 G。頻寬由它決定："
                         "f_n <= G/H，G=8、H=512 時 0.016。G=1 時空間上是常數"
                         "場，對裁切精確等變。")
    ap.add_argument("--patch-tile", type=int, default=0,
                    help="只學一塊 N×N 的磚並整片重複。0（預設）= 不平舖。"
                         "規則的重複本身就是「這是刻意放上去的標記」的視覺"
                         "訊號，同時把參數量由 3·H·W 降到 3·N²")
    ap.add_argument("--patch-res", type=int, default=1,
                    help="補丁內容的參數化解析度倒數 S。1（預設）= 全解析度，"
                         "呼叫路徑逐位元不變。S > 1 時只學 1/S 大小的張量再"
                         "雙線性升取樣，於是內容**依構造**沒有細於 S 像素的"
                         "分量——那正是模糊、重取樣與 JPEG 量化抹掉的東西。"
                         "與 --patch-tile 是兩個獨立的軸：平舖給的是週期與"
                         "冗餘（梳狀譜，諧波仍到 Nyquist），這一個給的是帶限。"
                         "機制取自 IAM（arXiv:2402.16586），但那篇動的是更新"
                         "步驟，此處動的是參數化本身")
    ap.add_argument("--patch-palette", type=int, default=0,
                    help="把補丁內容限制在 K 個可學顏色上。0（預設）= 關閉，"
                         "呼叫路徑逐位元不變。逐像素的空間自由度完全保留，"
                         "砍掉的只有**色彩的基數**——與 --patch-lowfreq（頻帶）"
                         "、--patch-chroma（通道）正交。產物是幾個色塊構成的"
                         "圖樣，像網版印刷而不像彩色雜訊")
    ap.add_argument("--patch-palette-temp", type=float, default=0.05,
                    help="調色盤指派的 softmax 溫度。越小越接近硬指派；"
                         "0.05（預設）時最大與最小 logit 的機率比是 e^20")
    ap.add_argument("--patch-seeds", type=int, default=0,
                    help="把補丁內容換成 K 個可學種子點的 Voronoi 圖，一胞一色。"
                         "0（預設）= 關閉。參數量是 5·K——K=64 時 320 個，比自由"
                         "補丁少三個數量級，而產物依構造是多邊形色塊。"
                         "移植自 arXiv:2606.17711。不可與 --patch-palette／"
                         "--patch-tile／--patch-res 併用")
    ap.add_argument("--patch-seed-temp", type=float, default=4e-4,
                    help="Voronoi 胞邊界的柔軟度。越小邊界越硬")
    ap.add_argument("--patch-polar", choices=("", "radial", "angular"),
                    default="",
                    help="把補丁內容換成只依一個極座標變數的一維剖面。"
                         "radial（c = f(r)）對繞影像中心的**任意角度旋轉**逐點"
                         "不變；angular（c = g(φ)）對**任意倍率的中心縮放**"
                         "逐點不變（`crop_resize` 正是中心裁切再放大）。"
                         "不變性對整個群成立，不是對某一個參數值——這是它與"
                         "已否決的 log-periodic 候選的分界。參數量 3·bins。"
                         "不可與 --patch-palette／--patch-seeds／--patch-tile／"
                         "--patch-res 併用。讀數見 runs/polar_carrier/")
    ap.add_argument("--patch-polar-bins", type=int, default=32,
                    help="剖面的格數，**同時是不變性的頻寬上限**。格數太多時"
                         "剖面本身是高頻的、場在取樣上混疊，重取樣會把它毀掉："
                         "角度場對裁切的餘弦在 32 格是 0.997、256 格掉到 0.79。"
                         "預設 32（參數量 96）")
    ap.add_argument("--patch-alpha", type=float, default=1.0,
                    help="標記的不透明度。1.0（預設）= 完全蓋掉；小於 1 時"
                         "原圖從標記底下透出來，讀成疊上去的浮水印而不是破圖")
    ap.add_argument("--patch-tv", type=float, default=0.0,
                    help="補丁內容的總變差權重。0（預設）時呼叫路徑逐位元"
                         "不變。壓住它就從逐像素噪點變成柔和色塊")
    ap.add_argument("--patch-init", choices=("identity", "random"),
                    default="identity",
                    help="補丁內容的起點。identity 從原圖開始（第 0 步即恆等）；"
                         "random 在支撐內抽噪聲。**--loss edit_divergence 必須用 "
                         "random**：那個損失在 x_def=x 處值與梯度都恰為零，"
                         "恆等起點是駐點，PGD 一步也走不動而且沒有症狀")
    ap.add_argument("--patch-placement",
                    choices=("far", "near", "complement", "crop_safe"),
                    default="far",
                    help="同尺寸有多個合法位置時挑哪一個：far 離主體重心最遠、"
                         "near 最近、complement 整個補集、crop_safe 限制在裁切"
                         "後仍留存的中央方框內並取最靠畫面中心者")
    ap.add_argument("--patch-count", type=int, default=1,
                    help="放幾塊同尺寸方塊（預設 1）。--radius 是**加起來**的"
                         "面積，故兩塊時單塊面積減半——「一塊 4%」與「兩塊各 "
                         "2%」在同一個總預算下可以直接對照。不可配 complement "
                         "或載體，那兩者的支撐不是方塊")
    ap.add_argument("--patch-crop-keep", type=float, default=0.8,
                    help="crop_safe 的留存比例：裁切算子每邊裁掉多少之後中央"
                         "還剩多少。0.8 對上 crop_resize0.1（每邊 10%）。"
                         "**這個值要與要防的裁切算子對齊**，故是 CSV 欄位")
    ap.add_argument("--attack-prompts", type=Path, default=None,
                    help="三類攻擊指令的目錄（data/attack_prompts.yaml）。"
                         "給定時覆寫資料集自帶的句子")
    ap.add_argument("--attack-category", default="",
                    choices=("", "clothing", "accessory", "background"),
                    help="用目錄裡的哪一類：改變衣著或顏色／加上配件／"
                         "變換背景與背景物品")
    ap.add_argument("--subject-source", default="clipseg",
                    choices=("clipseg", "face"),
                    help="受保護主體怎麼來。clipseg（預設）依登記的名詞；"
                         "face 由 ATR 的 Face+Hair 類別直接給——少一個模型、"
                         "少一組文字，也少一個「文字反應溢出」的失效面")
    ap.add_argument("--carrier-refine", type=int, default=0,
                    help="導引濾波的半徑，把 ATR 的粗邊界吸附到影像真實的"
                         "衣物邊緣。0（預設）時呼叫路徑逐位元不變")
    ap.add_argument("--carrier-erode", type=int, default=0,
                    help="載體邊界往**內**縮幾個像素，花紋永不溢出到皮膚或背景")
    ap.add_argument("--carrier-feather", type=int, default=0,
                    help="載體**往內**羽化幾個像素。支撐由 0/1 變成 [0,1] 的"
                         "軟權重，花紋在邊緣淡入衣服而不是硬切。方向與 "
                         "--subject-mask-feather 相反")
    ap.add_argument("--carrier-scatter", type=int, default=0,
                    help="不鋪滿載體，改成在載體內取 N 個分散斑塊，總面積由"
                         " --radius 控。0（預設）= 鋪滿")
    ap.add_argument("--patch-tint", type=float, default=0.0,
                    help="把花紋的**局部色調**拉向原本那件衣服的權重。"
                         "0（預設）時呼叫路徑逐位元不變。與 --patch-tv 針對"
                         "不同的東西：tv 罰掉所有局部變化、花紋也被壓平，"
                         "tint 只罰低頻、花紋的結構完全保留")
    ap.add_argument("--patch-tint-sigma", type=int, default=16,
                    help="--patch-tint 的鄰域邊長（像素）。「多大的範圍算同一"
                         "個色調」。與 --patch-tile 的磚長可並排比較")
    # **選項表由模組給，不在這裡另抄一份。** 抄一份的症狀是新增載體之後 CLI
    # 說「invalid choice」，而錯誤訊息指向的是使用者而不是這一行。
    from src.defense.carrier_mask import CARRIER_CLASSES as _CARRIERS

    ap.add_argument("--patch-carrier",
                    choices=("none",) + tuple(_CARRIERS),
                    default="none",
                    help="把支撐再交集一層**語意載體**（ATR 人體解析的衣物"
                         "類別，`src/defense/carrier_mask.py`）。none（預設）時"
                         "呼叫路徑逐位元不變。補集的支撐包含牆面與地板，花紋"
                         "貼在牆上讀起來是一塊壞掉的區域；限制在衣物上則止於"
                         "衣物輪廓，讀起來是布料印花。**只能配 "
                         "--patch-placement complement**")
    ap.add_argument("--color-tv", type=float, default=0.0,
                    help="色彩網格係數的空間總變差權重。0（預設）時呼叫路徑"
                         "逐位元不變。存在理由：容量放大會讓最佳化器去畫高頻"
                         "色彩噪點，產物不自然；這一項把場壓回平滑。"
                         "**只罰空間兩維，不罰亮度維**")
    ap.add_argument("--color-tv-luma", type=float, default=0.0,
                    help="色彩網格係數沿**亮度維**的總變差權重。0（預設）時"
                         "呼叫路徑逐位元不變。與 --color-tv 針對不同的東西："
                         "空間平滑管的是「顏色隨位置變化的快慢」，亮度平滑"
                         "管的是「相鄰色調會不會被映射到差很多的顏色」，"
                         "而後者才是產物上彩虹邊紋的來源")
    ap.add_argument("--color-luma-bins", type=int, default=8,
                    help="color_grid 的亮度格數 D。沿亮度軸是 D 段的分段線性。")
    ap.add_argument("--subject-mask", type=Path, default=None,
                    help="含 `objects` 對照表的 YAML（如 "
                         "data/decoy_catalogue.yaml）。給定時把擾動限制在"
                         "**主體之外**：遮罩由 CLIPSeg 依該表登記的主體名稱"
                         "產生，`apply_where = 1 − 遮罩`。主體名稱是防禦方"
                         "本來就知道的，攻擊指令不是。只有色彩族支援。")
    ap.add_argument("--subject-mask-threshold", type=float, default=0.30,
                    help="CLIPSeg sigmoid 的門檻。**刻意取低**：漏掉主體的"
                         "一部分比多保留一點背景嚴重得多")
    ap.add_argument("--subject-mask-dilate", type=int, default=16,
                    help="門檻之後往外膨脹的像素數")
    ap.add_argument("--subject-mask-feather", type=int, default=24,
                    help="膨脹之後往外羽化的像素數。**只往外**，故膨脹範圍內"
                         "恆為 1、主體逐位元保留")
    ap.add_argument("--color-rand-draw", choices=("corner", "uniform"),
                    default="corner",
                    help="隨機對照怎麼抽。corner 抽盒子的角點，與 sign 更新"
                         "的可達集合相同；uniform 抽盒內均勻。**預設 corner"
                         "是刻意的**：曲線族的 64 個係數獨立均勻抽會互相"
                         "抵消，同半徑下 DISTS 只到角點的三分之一，兩條曲線"
                         "的失真範圍幾乎不重疊、等失真內插整片 "
                         "out_of_range。")
    ap.add_argument("--radius", type=float, default=None,
                    help="直接指定半徑（掃描曲線用）。不給則二分搜到 --budget")
    ap.add_argument("--budget", type=float, default=0.0349, help="DISTS 預算")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--hop", type=int, default=None,
                    help="重疊步長。不給則取 block//2（逐位元等於加這個"
                         "旗標之前）。更小的 hop 讓每個像素被更多區塊覆蓋，"
                         "相鄰區塊各自獨立旋轉留下的接縫因此被平均掉——"
                         "防禦圖的紋理偏粗就是那個接縫。NOLA 條件在 hop "
                         "更小時只會更寬鬆，恆等保證不受影響")
    ap.add_argument("--dct-mode", default="plane",
                    choices=("plane", "shared_plane", "gain"),
                    help="dct_nonadd 的非加性形式。plane = 逐區塊學一個二維"
                         "平面與一個角度（保長）；shared_plane = 整張共用一個"
                         "平面（參數量少兩個數量級）；gain = 逐係數乘 exp(g)"
                         "（非加性但**不保長**，是非正交的對照點）")
    ap.add_argument("--update", default="sign",
                    choices=("sign", "adam", "lbfgs"),
                    help="PGD 的更新規則。sign 是本專案既有的 sign-PGD；"
                         "lbfgs 是 stAdv（arXiv:1801.02612）用的那一個"
                         "（`torch.optim.LBFGS`，line search 換成 PyTorch "
                         "唯一提供的 strong Wolfe，原文寫的是 backtracking，"
                         "屬 modified_from_paper；原文未載明迭代次數，故"
                         "**不寫死步數**：--steps 只是上限，實際停止由 "
                         "--patience 的收斂判定決定）。約束的處理方式與"
                         "代價見 run_param_pgd 的 docstring；"
                         "adam 在**帶折點的局部極小**上不會像 sign 那樣形成"
                         "週期 2 的振盪，用來分辨「損失不行」與「更新規則不行」。"
                         "**GOAL.md 把 Adam 列在更早期已否決的方向裡**，但那批"
                         "證據已刪除且是在相位參數化上做的；本旗標只為位移場的"
                         "探索批次而加，結果不可用來推翻相位臂上的那個否決")
    ap.add_argument("--step-size", type=float, default=None,
                    help="直接指定步長，取代 radius/(steps·saturate_at)。"
                         "不給時行為與加這個旗標之前逐位元相同。存在理由是"
                         "步長綁在半徑上會讓「放寬預算」同時「放大步長」")
    ap.add_argument("--saturate-at", type=float, default=0.25,
                    help="步長公式的分母係數，見 run_param_pgd 的 docstring")
    ap.add_argument("--warp-init-std", type=float, default=0.0,
                    help="位移場的隨機起點標準差（px）。0 = 全零起點，逐位元"
                         "等於加這個旗標之前。非零時由 --seed 決定抽樣。"
                         "理由：latent_norm 對位移場在零位移處有帶折點的局部"
                         "極小，從那裡起步量到的是 sign PGD 的性質")
    ap.add_argument("--dct-qd", type=float, default=0.85,
                    help="dct_rotate 的交付品質（參數作用在這個品質的量化"
                         "係數上，解碼即輸出）。與 --q-alg／--deliver-jpeg "
                         "走同一條 normalize_quality 換算")
    ap.add_argument("--dct-pairing", default="transpose",
                    choices=("transpose", "zigzag"),
                    help="係數配對規則。transpose 的兩格徑向頻率相同、量化階"
                         "幾乎相同，旋轉交換的是橫紋與直紋的方向；zigzag 的"
                         "兩格頻率與價錢都不同，是**對照組**，用來檢定"
                         "「保長只有在兩軸價錢相同時才有感知意義」這句話")
    ap.add_argument("--dct-gate", default="texture", choices=("texture", "band"),
                    help="dct_rotate 的閘。texture 走 pixel_texture_mask 再"
                         "avg_pool 到編解碼器的 8×8 格點；band 是全開")
    ap.add_argument("--warp-grid", type=int, default=WARP_GRID,
                    help="位移場的粗網格邊長（`warp`／`warp_rand`／"
                         "`warp_roundtrip` 三格用）。16 是本專案指定，與本機"
                         "量過的失真對照表同一個構造——換掉它那張表就作廢。"
                         "**位移場刻意不是 stAdv 的逐像素稠密場＋TV 正則**，"
                         "理由見 `WarpParam` 的 docstring")
    # ---- stAdv（arXiv:1801.02612）的流場正則，見 src/defense/stadv_flow.py ----
    ap.add_argument("--flow-tau", type=float, default=0.0,
                    help="流場總變差項的權重 τ，總目標是 L_adv + τ·L_flow。"
                         "0 = 關閉，逐位元等於加這一組旗標之前。**原文的 "
                         "0.05 不可沿用**：那是在分類器 logits 的 CW 損失上"
                         "網格搜尋（0.0005–0.05）得到的，本專案的 L_adv 是"
                         "擴散模型上的損失，量級不同。只接在 warp 上")
    ap.add_argument("--flow-eps", type=float, default=None,
                    help="加在 L_flow 每一項根號**裡面**的常數。**原文的式子"
                         "沒有這一項，故沒有預設值，開 --flow-tau 就必填**"
                         "（CLAUDE.md：查不到的參數設為必填，不要填看起來"
                         "合理的預設）。存在理由是 f ≡ 0 時 sqrt(0) 的梯度是"
                         "NaN，而位移場的預設起點正是全零。代價是 L_flow 多"
                         "一個常數偏移 `鄰居對數 × sqrt(eps)`")
    ap.add_argument("--flow-neighbourhood", default=None,
                    choices=tuple(sorted(NEIGHBOURHOODS)),
                    help="L_flow 的鄰域 N(p)。**原文只寫 q ∈ N(p)、沒有寫是"
                         "哪一種，故由本專案指定且沒有預設值**，開 --flow-tau "
                         "就必填。right_down = 每對相鄰像素只算一次；four = "
                         "四鄰域，恰為前者的兩倍；eight = 再加四條對角線")
    ap.add_argument("--disp-field-grid", type=int, default=16,
                    help="色散變形那個場的空間粗糙度：先在 grid×grid 上抽再"
                         "雙三次上採樣。**固定它，K 才是唯一的變因**——逐視窗"
                         "獨立抽（0）會讓 K=1 在空間上也最粗，色散度的效果就"
                         "與空間粗糙度混在一起。16 對齊 WarpParam 的粗網格")
    ap.add_argument("--r-min", type=float, default=0.12)
    ap.add_argument("--r-max", type=float, default=float("inf"),
                    help="徑向頻率閘的上界。預設無窮大即維持原本的高通行為。"
                         "穩健浮水印的標準作法是中頻帶嵌入——低頻可見、高頻"
                         "被壓縮與模糊抹掉；本模組原本只有下界，上界從未測過")
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="紋理閘的梯度能量參考分位數。閘的第二個因子是 "
                         "clamp(energy / 該分位數, 0, 1)，調低等於放行更多"
                         "低能量的區塊；0 使該因子恆為 1，即閘全開。"
                         "本值無出處，是本專案指定")
    ap.add_argument("--gate-edge-power", type=float, default=1.0,
                    help="紋理閘壓制邊緣那個因子的指數："
                         "(1 - coherence^2) ** 本值。1.0 = 現行行為（逐位元"
                         "相同），0 = 完全不壓制邊緣。邊緣是導向濾波、雙邊"
                         "濾波、TV 去噪的不變集，把擾動趕出邊緣等於放棄那幾"
                         "個算子底下唯一活得下來的位置。本值無出處，是本專案"
                         "指定")
    ap.add_argument("--freq-weight", choices=tuple(sorted(FREQ_WEIGHTS)),
                    default="binary",
                    help="頻率閘的知覺權重。binary = 二值帶通（逐位元等於加"
                         "這個旗標之前）；jpeg_luma = ITU-T T.81 Annex K 的"
                         "亮度量化表，雙線性內插到本模組的 rfft2 格點後正規"
                         "化到最大值 1。存在的理由是二值閘對 r=0.15 與 r=0.9 "
                         "開同一個價，而人眼對前者的敏感度高一個數量級；"
                         "RESULTS 已把 DCT-Shield 那 2 倍的失真效率優勢歸因給"
                         "JPEG 量化階的約束，並註明那不是加性本身帶來的")
    ap.add_argument("--freq-weight-power", type=float, default=1.0,
                    help="知覺定價的力道：權重取 power 次方。0 = 退回二值閘"
                         "（逐位元等於 --freq-weight binary），1 = 量化表的"
                         "原始定價。兩端都不是操作點：二值閘的位移／DISTS "
                         "只有 3.3-4.3，完整加權拉到 8-14.5 但通帶有效容量"
                         "掉到 0.544，要摸到會擋下的強度就得把半徑推過 theta "
                         "的封頂，之後只有增益在長而增益是振幅。本值無出處，"
                         "是本專案指定")
    ap.add_argument("--survival-weight", default="none",
                    choices=tuple(sorted(SURVIVAL_WEIGHTS)),
                    help="期望存活振幅，乘在頻率閘上（預設 none = 全 1，"
                         "逐位元等於加它之前）。"
                         "`jpeg_luma` 定價的是人眼看不看得見，而量化階隨頻率"
                         "遞增，所以它把預算往高頻推；但最佳化迴圈跑的是"
                         "未淨化的前向，看不到「模糊會把高頻整個拿掉」。"
                         "這一項只補最佳化看不到的那一半："
                         "w = (1 + Σ_σ exp(-2π²σ²f²)) / (1 + |S|)。"
                         "blur12 取 σ∈{1,2}（評測用的兩個），blur1 只取 σ=1。"
                         "**編碼器對哪一帶敏感不寫進來**——那由最佳化自己找")
    ap.add_argument("--gain-weight", choices=("shared", "jnd"),
                    default="shared",
                    help="增益的閘。shared = 與相位同一個閘（逐位元等於"
                         "加這個旗標之前）；jnd 另乘知覺權重，把振幅的"
                         "創造推到人眼看不見的頻帶。理由：自然影像的功率"
                         "譜按 1/f^2 掉，高頻幾乎沒有能量可以旋轉，相位在"
                         "那裡無事可做，而 exp(g)·|spec| 造得出容量。逐帶"
                         "量測見 runs/encoder_frequency_response")
    ap.add_argument("--phase-channels", choices=("rgb", "y"),
                    default="rgb",
                    help="本算子動哪些通道。rgb = 三通道各自做同一件事"
                         "（逐位元等於加這個旗標之前）；y = 只動亮度，"
                         "色差原樣送回。理由：增益在色度上累積成全域色偏，"
                         "而色偏屬於「單純劣化」不算擋下；RESULTS 的 "
                         "DCT-Shield 重現也記著「真正把失真砍半的是只動 Y "
                         "通道」，該篇的 Y-only 變體正是它在本失真帶內最好"
                         "的一格")
    ap.add_argument("--spectral-floor", type=float, default=0.0,
                    help="頻譜加性下限的強度。0 = 關閉（逐位元等於加"
                         "這個旗標之前）。相位與增益都是乘法，平坦區的"
                         "|spec| 接近零所以乘什麼都沒用；這一項在頻譜上"
                         "**加**一個由 JPEG 亮度量化表定價的量，且只乘"
                         "徑向帶通、不乘紋理閘，才進得去平坦區。"
                         "**開啟後方法不再是純粹的非加性重參數化**，"
                         "兩個設定都是主線、分開報（docs/METHOD.md）")
    ap.add_argument("--theta-budget", type=float, default=0.0,
                    help="幅度相依的相位上限，單位與係數同（Perturbing the "
                         "Phase, arXiv:2602.06577）：|theta| <= 2·arcsin("
                         "eps/(2|X|))，2|X| <= eps 的頻格相位自由。0 = 關閉。"
                         "它處理的是 FND-038——固定的 theta 不等於固定的失真")
    ap.add_argument("--coarsen", type=int, default=1,
                    help="三個空間場（theta／gain／floor）的視窗網格解析度"
                         "倍率。**1 = 關閉，逐位元等於加這個旗標之前。** "
                         "k > 1 時參數只存在 ceil(side/k) 見方的粗網格上，"
                         "前向雙線性升取樣回逐視窗。理由：hop=8、block=32 下"
                         "每個像素被 16 個視窗覆蓋，相鄰視窗的角度互相獨立時"
                         "那 16 份貢獻不同調，重疊相加會在選定頻格之外攤出"
                         "一層寬頻能量，而那正是 JPEG 最先丟掉的部分。"
                         "動機取自 IAM（arXiv:2402.16586）的內插平滑，"
                         "但**機制不同**：IAM 降的是影像解析度，本旗標平滑的"
                         "是視窗網格上的角度，不改變視窗內的頻率成分"
                         "（後者由 --r-min／--r-max 決定）")
    ap.add_argument("--floor-gate", choices=tuple(sorted(FLOOR_GATES)),
                    default="uniform",
                    help="加法項的價目表要不要隨區塊變。uniform（預設）只看"
                         "頻格，跨區塊是常數——那正是 DCT-Shield 的形狀。"
                         "complement 把加法限制在紋理閘的補集上（乘法那一半"
                         "動不了的地方）；watson 換成 Watson (1993) 的亮度"
                         "遮蔽 × 對比遮蔽。三者的**總預算相同**，改的是分配")
    ap.add_argument("--floor-r-min", type=float, default=None,
                    help="加法項**自己的**徑向帶下界。預設 None = 沿用 "
                         "--r-min，逐位元等於加這個旗標之前。"
                         "**--spectral-floor 0 時本旗標沒有作用**（價目表"
                         "根本不會被建出來）")
    ap.add_argument("--floor-r-max", type=float, default=None,
                    help="加法項**自己的**徑向帶上界。預設 None = 沿用 "
                         "--r-max，逐位元等於加這個旗標之前。"
                         "存在的理由：乘性那一半（相位 exp(iθ)、增益 exp(g)）"
                         "能改動的量正比於原圖自己的振幅 |S_b(ω)|，所以它得"
                         "待在有能量的地方；加法項不受該限制。而高斯模糊在"
                         "頻域乘 exp(-2π²σ²f²)，把高頻的振幅整個拿走——"
                         "「編碼器對哪一帶敏感」與「哪一帶活得過模糊」方向"
                         "相反。兩半共用一個帶通時壓低上界會連乘性那一半的"
                         "未淨化強度一起削掉；分開之後乘性留在高頻、加性放到"
                         "低頻，各買各的。**總預算不隨本旗標改變**——價目表"
                         "會被縮放回同一個平均值，改的是花在哪裡不是花多少"
                         "（見 PhaseResidual._build_floor_price）。"
                         "**--spectral-floor 0 時本旗標沒有作用**")
    ap.add_argument("--floor-survival", default="none",
                    choices=tuple(sorted(SURVIVAL_WEIGHTS)),
                    help="加法項**自己的**期望存活振幅，乘進價目表"
                         "（預設 none = 全 1，逐位元等於加它之前）。"
                         "--survival-weight 只乘在相位／增益的閘上，加法項的"
                         "價目表看不到它，而加法項正是不受 |S_b(ω)| 限制、"
                         "最適合被搬到活得過模糊的低頻的那一半。"
                         "式子與 --survival-weight 相同："
                         "w = (1 + Σ_σ exp(-2π²σ²f²)) / (1 + |S|)。"
                         "**總預算不隨本旗標改變**，改的是分配。"
                         "**--spectral-floor 0 時本旗標沒有作用**")
    ap.add_argument("--floor-envelope", default="none",
                    choices=list(FLOOR_ENVELOPES),
                    help="可學的**空間包絡**。none（預設）= 關閉，前向連張量"
                         "都不建，逐位元等於加這個旗標之前；gauss = 由 K 組"
                         "中心 (cy, cx)、K 個空間尺度 s、一個徑向低通截止 "
                         "f_c、一個強度 beta 參數化的平滑窗，五個純量與 "
                         "theta／gain／floor 走同一條 PGD。"
                         "**本模組原本沒有任何可學的空間定位**：紋理閘會定位"
                         "但由原圖決定、不可學，--floor-survival 已在頻率上"
                         "做軟性挑選但空間上均勻。"
                         "動機：裁切（crop_resize0.1 是繞中心放大 1.2488x，"
                         "落在中央 80%% 以內的才留下）與模糊（頻域乘 "
                         "exp(-2π²σ²f²)，與擾動放在哪裡無關，但空間上壓得越窄"
                         "頻譜攤得越寬，單位預算下被拿走的更多）兩者的交集是"
                         "「靠近中心的、大尺度的、平滑的結構」。"
                         "**總預算不隨本旗標改變**——價目表會被縮放回同一個"
                         "平均值（PhaseResidual.envelope_price），所以"
                         "「集中」在這裡的意思是「同樣的總量堆到少數位置、"
                         "那些位置的定價因此被抬高」。"
                         "beta = 0 時包絡逐位元是 1，故零初始化即恆等")
    ap.add_argument("--floor-envelope-k", type=int, default=1,
                    help="包絡由幾個高斯凸包的**軟聯集**組成："
                         "S = 1 - Π_k (1 - exp(-|p-c_k|²/(2 s_k²)))。"
                         "1（預設）時特判直接取單一凸包，與「只有一個凸包」"
                         "逐位元一致。取軟聯集而不是相加：相加會在重疊處"
                         "超過 1，beta 的內插語意就壞了")
    ap.add_argument("--floor-envelope-scope", default="floor",
                    choices=list(FLOOR_ENVELOPE_SCOPES),
                    help="包絡乘在哪一半上。floor（預設）= 只乘加法項的"
                         "價目表——加法項是唯一能自由選擇放在哪裡的那一半"
                         "（不受 |S_b(ω)| 限制），乘性那一半的位置已由紋理閘"
                         "決定。all = 連相位與增益的閘也乘。"
                         "**兩者的總量處理不同**：加法項一律做總量正規化"
                         "（radius 碰不到它，不補回總量的話等失真對齊會把"
                         "「加性變少」與「加性換位置」混在同一列）；乘性那"
                         "一半不做，它的強度旋鈕就是被二分搜尋的 radius，"
                         "而 --survival-weight 也是直接乘進 freq_gate 不補"
                         "總量，兩者一致")
    ap.add_argument("--pixel-gate-sigma", type=float, default=0.0,
                    help="逐像素紋理閘的高斯 sigma（像素）。0 = 關閉，逐位元與"
                         "加這個選項之前相同。要能分辨鬍鬚與臉頰就必須遠小於 "
                         "block=32；本值無出處，是本專案指定")
    ap.add_argument("--purify-aware",
                    choices=("none", "curriculum", "fixed75", "eot_jpeg",
                             "eot_ops", "eot_geometry", "eot_broad"),
                    default="none",
                    help="把可微分的淨化算子放進最佳化迴圈（改動三）。"
                         "curriculum = JPEG 品質 95→50 線性；fixed75 = 固定 75；"
                         "eot_jpeg = 每步抽一個品質；eot_ops = 每步抽一個算子"
                         "（其中的裁切是**固定**的中心 0.10）；"
                         "eot_geometry = 每步抽一個裁切比例**與位置**，這是"
                         "唯一會產生對一族幾何的不變性的一支"
                         "（identity／模糊／裁切縮放／JPEG75）。"
                         "eot_broad = 先抽算子類別（identity／JPEG／模糊／"
                         "裁切），再在該類的族內抽一個參數——eot_ops 的模糊 "
                         "sigma 與裁切比例都是固定值，固定的變換會被 co-adapt。"
                         "**回傳的防禦圖仍是未經算子處理的那一張**")
    # `eot_broad` 的兩個族。品質族沿用 `--eot-qualities`，故只需再加兩個。
    # **預設值就是模組裡的預設**，不給旗標時與模組的行為一致。
    ap.add_argument("--eot-sigmas", type=float, nargs="+",
                    default=list(BROAD_SIGMAS),
                    help="--purify-aware eot_broad 的高斯模糊 sigma 族")
    ap.add_argument("--eot-fractions", type=float, nargs="+",
                    default=list(BROAD_FRACTIONS),
                    help="--purify-aware eot_broad 的裁切比例族")
    # 拿掉整個類別。裁切那兩欄是結構性不可贏的
    # （`runs/ip2p_eot_geom_purify/README.md`），而每一步有四分之一的機率
    # 被花在它上面。**預設含全部四類，不給時逐位元不變。**
    ap.add_argument("--eot-classes", nargs="+", default=list(BROAD_CLASSES),
                    choices=BROAD_CLASSES,
                    help="--purify-aware eot_broad 抽的算子類別。必須含 identity")
    # ---- 交付前的像素域 L∞ 投影 ----
    # **0（預設）時逐位元等於加這個旗標之前。** 存在的理由是一個量到的縫：
    # 最佳化只被約束在 θ 的半徑球上，像素域沒有約束，於是同樣的 DISTS 下解
    # 可以是稀疏高振幅尖峰（`runs/ip2p_ig_converge` 實測 L∞ 0.79–0.93，主線
    # 相位族在相近 DISTS 只有 0.42），而人眼看得到、DISTS 與 LPIPS 看不到。
    # ---- 一致性懲罰 ----
    # **0（預設）時逐位元等於加這個旗標之前。** 存在的理由是一個量到的事實：
    # 最佳化要求的頻譜有 18–48% 的幅度是重疊相加交不出來的
    # （`runs/stft_consistency/`），而投影前的像素值域是 1.09–6.03——要求的
    # 那個東西根本不是一張影像，看到的刻紋就是投影誤差。
    ap.add_argument("--consistency-weight", type=float, default=0.0,
                    help="一致性懲罰的權重。損失加上 w × 相對幅度偏差，"
                         "使同樣損失值的解裡偏好可實現的那些。0 = 關閉。"
                         "與 --gl-iters 不同：那是在前向硬投影，這是軟偏好")
    # 退火：**0（預設）時權重恆定，逐位元等於加這個旗標之前。**
    # 固定權重會把失真封頂（`runs/arch_cons_matched`：w=0.30 的半徑拉 2.4 倍，
    # DISTS 只由 0.0835 走到 0.1145，到不了基準線的 0.153），而重畫與否看起來
    # 由失真水準決定。退火問的是「懲罰是不是只在早期需要」。
    # 把舊損失按權重加進新損失。**0（預設）時逐位元等於加這個旗標之前。**
    # 兩者實測互補：`latent_norm` 未淨化 0.6843 但 JPEG30 掉到地板底下
    # （0.1232 對 0.2547），`image_guidance` 未淨化 0.5850 但 JPEG30 是
    # 0.2604。舊項先除以它在乾淨影像上的值，故權重 1 真的是等權。
    ap.add_argument("--latent-norm-weight", type=float, default=0.0,
                    help="加到 --loss image_guidance 上的 latent_norm 權重"
                         "（已正規化，1 = 等權）。0 = 關閉")
    ap.add_argument("--consistency-decay", type=float, default=0.0,
                    help="一致性權重線性歸零的位置佔總步數的比例。"
                         "0.5 = 半程歸零，之後與不加懲罰相同。0 = 不退火")
    ap.add_argument("--linf-deliver", type=float, default=0.0,
                    help="交付前把像素域殘差夾到 [-eps, eps]（0–1 值域）。"
                         "0 = 關閉。迴圈的前向套直通估計版、交付與存檔套真正的"
                         "夾取，故最佳化的對象與交出去的對象是同一張圖")
    # ---- 分階段訓練（階段二：在階段一的解附近做受約束的再最佳化）----
    # **`--stage2-steps 0`（預設）時逐位元等於加這一組旗標之前**，由
    # `tests/test_stage2_training.py` 釘住。與已否決的 `--purify-aware` 的
    # 差別是「不從零開始」加上「不准把階段一的成果賠掉」，後者正是那批被否決
    # 的直接病灶（未淨化強度掉 10–25%，見 `docs/RESULTS.md`）。
    ap.add_argument("--stage2-steps", type=int, default=0,
                    help="階段二的步數。0 = 關閉")
    ap.add_argument("--stage2-ops", nargs="+",
                    default=["identity", "blur1", "blur2",
                             "crop05", "crop10", "crop15"],
                    help=f"階段二輪替的淨化算子，可用：{sorted(STAGE2_OPS)}。"
                         "identity 必須留著，否則最佳化會為了耐洗而放掉"
                         "未淨化時的效果")
    ap.add_argument("--stage2-order", choices=STAGE2_ORDERS, default="shuffle",
                    help="算子的給法：每輪洗牌後依序走完／固定輪替／每步隨機")
    ap.add_argument("--stage2-step-scale", type=float, default=0.2,
                    help="階段二步長對階段一步長的比例")
    ap.add_argument("--stage2-trust", type=float, default=0.95,
                    help="信賴域：未淨化增益不得低於階段一終值的這個比例")
    ap.add_argument("--stage2-check-every", type=int, default=20,
                    help="每幾步檢查一次信賴域")
    ap.add_argument("--stage2-ramp", type=int, default=0,
                    help="1 = 前半段只用弱算子（由弱到強）。0 = 全程同一池")
    # 多品質集成損失的品質集合（Shin & Song 2017：單一品質會過度特化）。
    # **預設 (95, 75, 50) 逐位元等於加這個旗鈕之前。** 與 `--deliver-jpeg` 併用
    # 時順序是「先自壓到交付格點、再讓攻擊方以抽到的品質壓一次」，也就是
    # **集成放在損失上、交付仍是單一格點**——放到交付上就是已否決的
    # `--purify-aware` 那三個變體。
    ap.add_argument("--eot-qualities", type=int, nargs="+",
                    default=[95, 75, 50],
                    help="--purify-aware eot_jpeg 每步隨機抽的品質集合")

    # ---- 不動點項（`runs/fixedpoint_framework/README.md` 第五節）----
    # **`--manifold-weight 0` 且 `--manifold-only` 關著時逐位元等於加這組旗標
    # 之前**，由 `tests/test_fixedpoint_loss.py` 釘住。
    ap.add_argument("--manifold-weight", type=float, default=0.0,
                    help="不動點項的權重。0 = 關閉。項本身已用乾淨影像上的值"
                         "正規化，故起點恰為 1，權重可跨影像比較")
    ap.add_argument("--manifold-t", type=int, default=100,
                    help="不動點項抽樣的時間步上限（總共 1000 步）。取小段是"
                         "為了對齊淨化器實際走的噪聲尺度")
    ap.add_argument("--manifold-balance", choices=("raw", "normalised"),
                    default="raw",
                    help="raw = 直接相加（權重 1 實際只佔約 1/70）；"
                         "normalised = 主項也除以乾淨影像上的值，兩項"
                         "都由 1 起步，權重 1 才是等權")
    ap.add_argument("--manifold-only", action="store_true",
                    help="判準 F3 的歸因對照：只留不動點項、拿掉對抗項")
    ap.add_argument("--dct-plane-weight", choices=("uniform", "priced"),
                    default="uniform",
                    help="整併版旋轉的目標方向要不要依 JPEG 量化表定價。"
                         "uniform 逐位元等於加這個旗鈕之前")
    ap.add_argument("--deliver-jpeg", type=float, default=0.0,
                    help="交付自壓的 JPEG 品質。**0 = 關閉，逐位元等於加這個"
                         "旗標之前。** 吃論文式的小數（0.85）也吃整數（85），"
                         "與 --q-alg 共用 normalize_quality。開著時做兩件事，"
                         "缺一不可：(1) 最佳化迴圈的前向套 jpeg_roundtrip_ste"
                         "（可微，前向值逐位元等於真實往返）；(2) **交付與存檔"
                         "的是 jpeg_roundtrip 的輸出**，即壓縮過的圖。"
                         "第二件事是本旗標與已否決的 --purify-aware 的唯一差別"
                         "——那三個變體把 JPEG 放進迴圈卻交付未壓縮的圖，"
                         "擾動一離開迴圈就不在量化格點上了。約束在格點上之後，"
                         "攻擊方以同品質或更高品質重壓近似恆等，"
                         "那正是 DCT-Shield 抗 JPEG 的全部來源。"
                         "只接受本方法的參數化條件，其餘條件直接拒絕")
    ap.add_argument("--gain-ratio", type=float, default=0.0,
                    help="可學幅度增益的上界對半徑的比例：gain_max = radius × "
                         "此值。`phase_gain`／`gain_only` 兩個條件需要它 > 0。"
                         "綁成單一旋鈕是為了讓既有的掃描與二分搜尋不必改成"
                         "二維搜尋；本值無出處，是本專案指定")
    ap.add_argument("--gl-iters", type=int, default=0,
                    help="Griffin-Lim 迭代投影的輪數。逐區塊轉相位後的係數一般"
                         "不一致，重疊相加會部分抵銷；這個投影把它拉回一致集"
                         "合。0 = 關閉，逐位元與加這個選項之前相同")
    ap.add_argument("--seed", type=int, default=0)
    # DCT-Shield
    ap.add_argument("--mode", choices=("paper",), default="paper")
    ap.add_argument("--eps", type=float, default=PAPER_EPS)
    ap.add_argument("--q-alg", type=float, default=PAPER_DEFAULT_QUALITY)
    ap.add_argument("--dct-loss", choices=("paper", "project"), default="paper",
                    help="DCT-Shield 用哪個損失。paper（預設）是該篇自己的 "
                         "‖E(x')‖₂，跑 baseline 必須用它。project 換成 "
                         "--loss 指定的那一個，是**消融**，會強制標記 "
                         "modified_from_paper")
    ap.add_argument("--dct-steps", type=int, default=PAPER_STEPS)
    # AdvDrop
    # DJSMA（DCT 反對角帶上的貪婪 JSMA）
    ap.add_argument("--wm-tau", type=int, default=PAPER_TAU,
                    help="迭代上限，同時是 l0 界。論文定案 1500")
    ap.add_argument("--wm-mu", type=int, default=PAPER_MU,
                    help="同一個係數最多改幾階，即 l∞ 界。論文定案 1")
    ap.add_argument("--wm-diagonals", type=int, nargs="+",
                    default=list(PAPER_ADV_DIAGONALS),
                    help="對抗擾動落在哪幾條 8×8 反對角帶（1-based）。論文用"
                         "第 3–5 條；第 6–8 條是它留給隱形浮水印的，本專案"
                         "未實作浮水印那兩個階段")
    ap.add_argument("--wm-q-embed", type=float, default=0.75,
                    help="嵌入端的 JPEG 品質。**論文未載**——論文只說評測時"
                         "再壓 Q=75。本值是本專案指定")
    ap.add_argument("--advdrop-eps", type=float, default=100.0,
                    help="量化表的可動上界 q_init+eps。論文掃 20/60/100")
    ap.add_argument("--advdrop-steps", type=int, default=50)
    ap.add_argument("--advdrop-step-size", type=float, default=4.0,
                    help="論文式 (7) 隱含 1，但本專案在原生威脅模型上實測要 4 "
                         "才重現得到它報的成功率；見 runs/advdrop_repro")
    ap.add_argument("--skip-dc", action="store_true",
                    help="DCT-Shield 不動 DC 係數。**論文沒有這一步**，用來檢定"
                         "「失真比論文差 1.76 倍」是不是 DC 的整階平移造成的")
    # 攻擊方（論文未載，本專案指定）
    ap.add_argument("--fl-w-latent", type=float, default=None,
                    help="--loss facelock 的 latent 推離項權重。預設取官方 "
                         "methods.py 的 0.2（論文正文只給符號、未給數值）")
    ap.add_argument("--fl-w-lpips", type=float, default=None,
                    help="--loss facelock 的重建圖 LPIPS 項權重。預設 1.0")
    ap.add_argument("--fl-start-fr", type=float, default=None,
                    help="--loss facelock 的身分項啟動比例。預設 0.35，"
                         "**論文正文未載，只在官方程式碼裡**")
    ap.add_argument("--fl-start-lpips", type=float, default=None,
                    help="--loss facelock 的 LPIPS 項啟動比例。預設 0.25，同上")
    ap.add_argument("--edit-steps", type=int, default=IP2P_STEPS)
    ap.add_argument("--text-guidance", type=float, default=IP2P_TEXT_GUIDANCE)
    ap.add_argument("--image-guidance", type=float, default=IP2P_IMAGE_GUIDANCE)
    ap.add_argument("--edit-seed", type=int, default=IP2P_SEED)
    ap.add_argument("--check-only", action="store_true",
                    help="只跑未防禦的編輯並報語意對齊，驗收 DEC-022 的前提")
    return ap


def validate_loss_args(args) -> None:
    """`--loss` 的相依旗標守門。**在載入權重之前**呼叫。

    存在理由：`--ig-zt` 沒有預設值，而缺了它的失效方式是「跑完一整個工作點
    才發現損失不是預期的那一個」。這一族的坑專案踩過——漏傳一支驅動讓三個
    工作點在 argparse 被擋下、卡空轉半小時（`docs/OPERATIONS.md`）。守門要
    擋在派工前面，不是印出來就算。
    """
    if args.loss not in ("image_guidance", "edit_divergence"):
        # `--ig-weight` 只在讀 UNet 的兩個損失底下有意義。給了卻用在別的損失上，
        # 靜默忽略的症狀是「報表上寫著 subject 而實際跑的是 uniform」。
        if args.ig_weight != "uniform":
            raise SystemExit(
                f"--ig-weight {args.ig_weight} 只能配 --loss image_guidance "
                f"或 edit_divergence，收到 --loss {args.loss}。")
        return
    if (args.loss == "edit_divergence"
            and any(c in PATCH_CONDS for c in args.conditions)
            and args.patch_init != "random"):
        raise SystemExit(
            "--loss edit_divergence 配 patch 時必須給 --patch-init random："
            "該損失是 −‖f(x_def) − f(x)‖²，在 x_def = x 處值與梯度**都恰為零**，"
            "恆等起點是駐點。實測會安靜地一步都不走（loss -0.000000、"
            "參數不動），而 trace 看起來只是「損失很小」。")
    if args.ig_weight == "subject" and not args.ig_weight_catalogue.exists():
        raise SystemExit(
            f"--ig-weight subject 需要主體名詞目錄，找不到 "
            f"{args.ig_weight_catalogue}")
    if args.ig_zt is None:
        raise SystemExit(
            f"--loss {args.loss} 必須同時給 --ig-zt。"
            "**不可以填一個看起來合理的預設**：IP2P 由純噪聲起步，中間步的"
            f" z_t 分布依賴條件、無法解析，{ZT_MODES} 兩者都只是近似。")
    if not 1 <= args.ig_t_min <= args.ig_t_max:
        raise SystemExit(
            f"需要 1 <= --ig-t-min <= --ig-t-max，"
            f"收到 {args.ig_t_min}／{args.ig_t_max}")
    if args.ig_samples < 1:
        raise SystemExit(f"--ig-samples 必須為正整數，收到 {args.ig_samples}")


def select_images(dataset, names, data_dir):
    """依 `--images` 挑影像。**指名而資料集裡沒有的一律拋錯。**

    選圖原本是集合交集，於是指名 150 張、資料集只有 147 張時少的那三張被
    靜默丟掉，唯一的守門是「一張都不剩才拋錯」。印出來的那行寫的是實際張數，
    沒有人會回頭跟要求的張數對照——`runs/ip2p_fair_comparison/images150.txt`
    指名了三張不在 `data/omniedit150/` 底下的影像，而每一個用該清單跑的批次
    都安靜地少跑三張。

    回傳的順序照 `dataset`，不照 `names`：逐圖相減依賴的是名字不是位置，
    但保持資料集自己的順序才與不給 `--images` 的批次可並列。
    """
    if names:
        keep = list(dict.fromkeys(names))
        have = {d["name"] for d in dataset}
        missing = [n for n in keep if n not in have]
        if missing:
            raise SystemExit(
                f"--images 指名了 {len(missing)} 張 {data_dir} 底下沒有的影像："
                f"{missing}。**不靜默略過**——少跑幾張在報表上看不出來，"
                "而張數會直接改變 FID 這類與樣本數相關的讀數。")
        dataset = [d for d in dataset if d["name"] in set(keep)]
    if not dataset:
        raise SystemExit(f"{data_dir} 底下沒有符合 --images 的影像")
    return dataset


def validate_patch_args(args) -> None:
    """補丁族的旗標守門。**在載入權重之前**呼叫。

    為什麼不放在 `validate_loss_args` 裡：那一支在 `--loss` 不是
    `image_guidance`／`edit_divergence` 時**提前 return**，於是補丁的守門在
    別的損失底下整段跳過而不會有任何症狀。補丁的幾何與損失無關，兩者不該
    共用一個入口。
    """
    is_patch = any(c in PATCH_CONDS for c in args.conditions)
    # 補集補丁的面積由遮罩決定，`radius` 完全不使用；不給 --radius 會落進
    # 預算模式，去二分搜尋一個不起作用的參數。實測會用滿 GPU 兩小時而
    # **一步 PGD 的紀錄都沒有**，看起來像卡住而不是設定錯。
    if is_patch and args.patch_placement == "complement" and args.radius is None:
        raise SystemExit(
            "--patch-placement complement 必須同時給 --radius："
            "補集模式的面積由主體遮罩決定、radius 不使用，但不給它會落進"
            "預算模式（二分搜尋 radius），那在這個組合下是在搜一個不起作用的"
            "參數，只會空轉。給任意值即可，該值會被忽略並記在 CSV。")
    # 載體只在補集模式下有意義。配矩形擺放模式時 `PatchParam.reset` 也會拋錯，
    # 但擋在派工前面才省得下載入權重與排隊的時間；配非補丁條件時則根本沒有
    # 支撐這個概念，靜默忽略的症狀是「報表寫著 clothes 而那一列與載體無關」。
    if args.patch_count != 1:
        if not is_patch:
            raise SystemExit(
                f"--patch-count {args.patch_count} 只接在補丁條件上，"
                f"收到條件 {args.conditions}。")
        if args.patch_count < 1:
            raise SystemExit(
                f"--patch-count 必須 >= 1，收到 {args.patch_count}。")
        if args.patch_placement == "complement":
            raise SystemExit(
                "--patch-count 不能配 --patch-placement complement："
                "補集是單一個不規則區域，「幾塊」在那個構造下沒有意義。")
        if args.patch_carrier != "none":
            raise SystemExit(
                f"--patch-count 不能配 --patch-carrier {args.patch_carrier}："
                "載體是不規則區域、方塊擺放是固定形狀，兩種支撐構造互斥。")
    # 載體有兩個用途：補丁族拿它當**支撐**，色彩族拿它當 `apply_where`。
    # 後者不需要 complement——那個旗標在色彩族上根本沒有意義。
    if (args.patch_carrier != "none" and not is_patch
            and args.subject_mask is None and args.subject_source != "face"):
        raise SystemExit(
            f"--patch-carrier {args.patch_carrier} 用在 {args.conditions} 上時"
            "必須給 --subject-mask 或 --subject-source face：色彩族的載體是 "
            "`apply_where = 載體 ∧ 主體補集`，那個交集要有主體遮罩才算得出來。"
            "不給的話載體會被靜默忽略，而報表上的 patch_carrier 欄仍然寫著它。")
    # ── 兩個內容約束只接在補丁族上 ──────────────────────────────────
    # 色彩族沒有「補丁內容」這個物件，靜默忽略的症狀是 CSV 寫著 lowfreq 16
    # 而實際跑的是完全自由的色彩場。
    for flag, val in (("--patch-lowfreq", args.patch_lowfreq),
                      ("--patch-chroma", args.patch_chroma),
                      # `--patch-res` 的關閉值是 1 不是 0，故比較的是 `> 1`。
                      ("--patch-res", args.patch_res > 1),
                      ("--patch-palette", args.patch_palette),
                      ("--patch-seeds", args.patch_seeds),
                      ("--patch-polar", args.patch_polar)):
        if val and not is_patch:
            raise SystemExit(
                f"{flag} 只接在補丁條件（patch／patch_rand）上，"
                f"收到條件 {args.conditions}。")
    if args.patch_seeds and args.patch_palette:
        raise SystemExit(
            "--patch-seeds 與 --patch-palette 不可同時給：Voronoi 本來就是"
            "一胞一色，K 由 --patch-seeds 決定。同時給會讓 CSV 上兩欄都有值，"
            "而實際只有一個生效。")
    if args.patch_seeds and (args.patch_tile or args.patch_res > 1):
        raise SystemExit(
            "--patch-seeds 不可與 --patch-tile／--patch-res 併用："
            "那兩個旋鈕作用在可學張量的空間解析度上，而 Voronoi 的可學張量"
            "是座標不是場。")
    if args.patch_polar and (args.patch_palette or args.patch_seeds
                             or args.patch_tile or args.patch_res > 1):
        raise SystemExit(
            "--patch-polar 不可與 --patch-palette／--patch-seeds／"
            "--patch-tile／--patch-res 併用：那些旋鈕作用在二維可學張量上，"
            "而極座標參數化的可學張量是一條一維剖面。同時給的話 CSV 上兩欄"
            "都有值，而實際只有一個生效。")
    if args.patch_polar and args.patch_polar_bins < 8:
        raise SystemExit(
            f"--patch-polar-bins {args.patch_polar_bins} 太小（至少 8）。")
    if args.patch_lowfreq and args.patch_tint:
        raise SystemExit(
            "--patch-lowfreq 與 --patch-tint 不可同時給：兩者針對的是**同一個**"
            "低頻帶，一個用替換一個用懲罰。同時開的話「低頻誤差還剩多少」"
            "是兩個機制的合成，拆不出各自的貢獻，而 CSV 上兩欄都有值、"
            "看起來像是可以拆的。要比較就分成兩格跑。")
    # ── 重播模式 ────────────────────────────────────────────────────
    # `--steps 0` 不做任何最佳化，交出的就是載回來的那張防禦圖。沒有
    # `--resume-weights` 時載不到東西，交出去的會是**原圖**，而 CSV 的每一欄
    # 看起來都正常——那是「沒有防禦」偽裝成「防禦無效」。
    if args.steps == 0 and args.resume_weights is None:
        raise SystemExit(
            "--steps 0 必須同時給 --resume-weights：零步最佳化交出的是載回來"
            "的防禦圖，沒有來源就等於交出原圖，而報表上看不出來。")
    # ── 載體的面積旗標 ──────────────────────────────────────────────
    if args.carrier_target_area and args.patch_carrier == "none":
        raise SystemExit(
            "--carrier-target-area 要有載體才有意義：沒有載體時支撐就是主體"
            "補集，面積由遮罩決定、縮不了。")
    if args.carrier_target_area and not 0.0 < args.carrier_target_area <= 1.0:
        raise SystemExit(
            f"--carrier-target-area 必須落在 (0,1]，收到 {args.carrier_target_area}")
    # 背景幾乎一定超過 `MAX_AREA`（實測十張全部）。擋在這裡而不是等
    # `carrier_from_seg` 拋錯，是為了省下載入 4 GB 權重與排隊的時間。
    # ── 環帶 ────────────────────────────────────────────────────────
    if args.carrier_ring:
        if not 0 < args.carrier_ring_inner < args.carrier_ring:
            raise SystemExit(
                f"需要 0 < --carrier-ring-inner < --carrier-ring，收到 "
                f"{args.carrier_ring_inner} 與 {args.carrier_ring}。inner 為 0 "
                "時支撐會與主體的羽化帶重疊，實際面積比要求的窄而看不出來。")
        if args.subject_source != "face":
            raise SystemExit(
                "--carrier-ring 需要 --subject-source face：環帶是繞著**受保護"
                "主體**長出來的，而那個主體由 ATR 的 Face+Hair 給。")
        for flag, val in (("--carrier-lattice", args.carrier_lattice),
                          ("--carrier-scatter", args.carrier_scatter),
                          ("--carrier-target-area", args.carrier_target_area)):
            if val:
                raise SystemExit(
                    f"--carrier-ring 不可與 {flag} 併用：環帶的形狀與面積由 "
                    "inner／outer 決定，再套一層別的支撐會讓兩個構造疊起來，"
                    "而 CSV 上兩欄都有值、看起來像是可以拆的。")
    # ── identity 損失 ───────────────────────────────────────────────
    if args.loss == "identity":
        if args.subject_source != "face":
            raise SystemExit(
                "--loss identity 需要 --subject-source face：裁臉框由受保護"
                "主體的遮罩算出來，而那個主體由 ATR 的 Face+Hair 給。")
        if args.id_box_margin < 0:
            raise SystemExit(
                f"--id-box-margin 不可為負，收到 {args.id_box_margin}")
        if args.id_layout_weight < 0:
            raise SystemExit(
                f"--id-layout-weight 不可為負，收到 {args.id_layout_weight}")
    elif args.id_layout_weight or args.id_box_margin != 0.35:
        raise SystemExit(
            f"--id-layout-weight／--id-box-margin 只接在 --loss identity 上，"
            f"收到 --loss {args.loss}。靜默忽略的症狀是 CSV 寫著一個權重而"
            "實際跑的損失裡沒有那一項。")
    # ── facelock 損失 ──────────────────────────────────────────────
    _FL = (("--fl-w-latent", args.fl_w_latent), ("--fl-w-lpips", args.fl_w_lpips),
           ("--fl-start-fr", args.fl_start_fr),
           ("--fl-start-lpips", args.fl_start_lpips))
    if args.loss == "facelock":
        if args.subject_source != "face":
            raise SystemExit(
                "--loss facelock 需要 --subject-source face：裁臉框由受保護"
                "主體的遮罩算出來，而那個主體由 ATR 的 Face+Hair 給。")
        for flag, v in _FL:
            if v is not None and v < 0:
                raise SystemExit(f"{flag} 不可為負，收到 {v}")
        for flag, v in _FL[2:]:
            if v is not None and v > 1:
                raise SystemExit(f"{flag} 必須落在 [0,1]，收到 {v}")
    else:
        for flag, v in _FL:
            if v is not None:
                raise SystemExit(
                    f"{flag} 只接在 --loss facelock 上，收到 --loss {args.loss}。"
                    "靜默忽略的症狀是 CSV 寫著一個權重而實際跑的損失裡沒有那一項。")
    # ── 注意力項與指令 EOT ──────────────────────────────────────────
    if args.attn_weight < 0:
        raise SystemExit(f"--attn-weight 不可為負，收到 {args.attn_weight}")
    if (args.attn_weight or args.prompt_eot) and not args.attack_prompts:
        raise SystemExit(
            "--attn-weight 與 --prompt-eot 都需要 --attack-prompts："
            "兩者都要一組**可能的**攻擊指令當文字條件。")
    if args.prompt_eot and args.loss != "image_guidance":
        raise SystemExit(
            f"--prompt-eot 目前只接在 --loss image_guidance 上，收到 "
            f"{args.loss}。其餘損失沒有文字條件那一支，靜默忽略時 CSV 的 "
            "prompt_eot 欄仍然寫著 1。")
    if args.carrier_lattice:
        if args.carrier_lattice < 2:
            raise SystemExit(
                f"--carrier-lattice 必須 >= 2，收到 {args.carrier_lattice}")
        if 2.0 * args.carrier_dot_radius >= args.carrier_lattice:
            raise SystemExit(
                f"--carrier-dot-radius {args.carrier_dot_radius} 對 "
                f"--carrier-lattice {args.carrier_lattice} 太大：圓斑會相連、"
                "支撐退化成一整片，那就不是點陣了。要滿版請用 "
                "--carrier-match scale。")
        if args.carrier_scatter:
            raise SystemExit(
                "--carrier-lattice 與 --carrier-scatter 不可同時給："
                "前者是很多個小點（位置由格點決定），後者是少數幾個大斑"
                "（位置由最遠點取樣決定），兩種支撐構造互斥。同時給會先散成"
                "大斑再打成點陣，而 CSV 上兩欄都有值、看起來像是可以拆的。")
        if args.carrier_target_area:
            raise SystemExit(
                "--carrier-lattice 與 --carrier-target-area 不可同時給："
                "點陣的面積是 pitch 與 radius 的函數，再對齊一次會把點陣"
                "整片調淡（scale）或侵蝕掉（erode），兩者都會破壞"
                "「每個 latent 格恰好被碰到一次」這個構造。")
    if args.carrier_match == "scale" and not args.carrier_target_area:
        raise SystemExit(
            "--carrier-match scale 必須同時給 --carrier-target-area："
            "縮放模式的整個作用就是把權重調到那個面積，沒有目標就等於沒有作用，"
            "而 CSV 上仍然寫著 scale。")
    if args.patch_carrier == "frame" and (args.carrier_max_area is None
                                          or args.carrier_max_area < 1.0):
        raise SystemExit(
            "--patch-carrier frame 必須給 --carrier-max-area 1.0："
            "整張畫面的面積恆為 1.0，而載體的面積上限本來是為了擋「解析器把整張"
            "圖都算成衣服」——那道守門在這個載體上沒有意義，但要明寫出來，"
            "不可以靜默放行。")
    if args.patch_carrier == "background" and args.carrier_max_area is None:
        raise SystemExit(
            "--patch-carrier background 必須同時給 --carrier-max-area："
            "背景的面積幾乎一定超過載體的預設上限 0.60，而那道守門本來是"
            "為了擋「解析器把整張圖都算成衣服」。背景要放寬到 0.95 左右。")
    if args.patch_carrier != "none" and is_patch:
        if args.patch_placement != "complement":
            raise SystemExit(
                f"--patch-carrier {args.patch_carrier} 必須配 "
                f"--patch-placement complement，收到 {args.patch_placement}："
                "矩形擺放模式會把載體忽略掉，症狀是報表上寫著載體名稱而實際"
                "跑的是一塊方塊。")


def validate_deliver_args(args) -> None:
    """`--linf-deliver` 的守門。**在載入權重之前**呼叫。

    預算模式（不給 `--radius`）走 `fit_to_budget`，它內層自己呼叫
    `run_param_pgd` 而**沒有把 `deliver` 傳下去**。靜默照跑的話，二分搜出來
    的半徑講的是一張沒有投影的圖，存檔的卻是投影過的——預算欄會變成謊話。
    **寧可拒絕**。
    """
    if not args.linf_deliver:
        return
    if not 0.0 < args.linf_deliver <= 1.0:
        raise SystemExit(
            f"--linf-deliver 必須落在 (0, 1]（影像值域 0–1），"
            f"收到 {args.linf_deliver}")
    if args.radius is None:
        raise SystemExit(
            "--linf-deliver 不可與預算模式併用：fit_to_budget 的內層迴圈"
            "沒有這條投影，二分搜的失真會量在一張不會被交付的圖上。"
            "請明給 --radius。")
    if args.stage2_steps:
        raise SystemExit(
            "--linf-deliver 不可與 --stage2-steps 併用："
            "階段二走 run_stage2_pgd，那一支同樣沒有這條投影。")


def validate_flow_args(args) -> None:
    """stAdv 流場正則的守門。**在載入權重之前**呼叫。

    兩個參數是必填不是有預設：`eps` 原文的式子裡沒有、`neighbourhood` 原文
    沒有寫是哪一種。填一個看起來合理的預設會讓報表上的 `flow_eps`／
    `flow_neighbourhood` 兩欄看起來像論文給的值（CLAUDE.md「移植他人的方法」）。

    只接 `warp`：`warp_rand`／`warp_roundtrip` 不最佳化（`params()` 為空、
    `run_param_pgd` 直接回傳），對它們加一個正則項不會有任何作用，但
    `flow_tau` 那一欄照樣寫著非零值——那是一列會騙人的紀錄。
    """
    if not args.flow_tau:
        if args.flow_eps is not None or args.flow_neighbourhood is not None:
            raise SystemExit(
                "--flow-eps／--flow-neighbourhood 只有在 --flow-tau > 0 時"
                "才有作用。給了其中之一卻沒開 τ，CSV 會寫著一組沒有被用到的"
                "設定。請一併給 --flow-tau，或把這兩個旗標拿掉。")
        return
    if args.flow_tau < 0:
        raise SystemExit(f"--flow-tau 不可為負，收到 {args.flow_tau}")
    if args.flow_eps is None:
        raise SystemExit(
            "開了 --flow-tau 就必須明給 --flow-eps：原文的 L_flow 式子裡"
            "沒有這個常數，它是本專案為了讓 f ≡ 0 處的梯度存在而加的"
            "（modified_from_paper），不可以有預設值。")
    if args.flow_eps <= 0:
        raise SystemExit(
            f"--flow-eps 必須為正，收到 {args.flow_eps}："
            "填 0 等於把 f ≡ 0 處的 NaN 梯度放回來。")
    if args.flow_neighbourhood is None:
        raise SystemExit(
            "開了 --flow-tau 就必須明給 --flow-neighbourhood：原文只寫 "
            "q ∈ N(p)，沒有寫是四鄰域、八鄰域，還是只取右與下。三種選法差"
            "一個倍率與一個方向偏好，故由本專案指定，不可以有預設值。")
    bad = [c for c in args.conditions if c != "warp"]
    if bad:
        raise SystemExit(
            f"--flow-tau 只接在 warp 這一格上，收到 {bad}："
            "隨機與往返兩格不最佳化，加了正則項不會有任何作用，"
            "CSV 卻會寫著一個非零的 flow_tau。")


def validate_convergence_args(args) -> None:
    """`--update lbfgs` 必須配上收斂判定。**在載入權重之前**呼叫。

    stAdv 原文沒有載明 L-BFGS 的迭代次數，故本專案不寫死步數：`--steps` 只是
    上限，實際停止由 `--eval-every` ＋ `--patience` 決定。兩者缺一時跑出來的
    是「停在某個人挑的步數」，而報表上的 `stopped_at` 會被讀成收斂點——那是
    一個沒有症狀的誤讀，**寧可拒絕**。

    這是「如何量得準」不是「有沒有效」：判的是曲線平了沒有，不是方法成不成立。
    """
    if args.update != "lbfgs":
        return
    if not args.eval_every or not args.patience:
        raise SystemExit(
            "--update lbfgs 必須同時給 --eval-every 與 --patience："
            "原文未載明迭代次數，本專案因此不寫死步數，--steps 只是上限，"
            f"實際停止由收斂判定決定（收到 --eval-every {args.eval_every}、"
            f"--patience {args.patience}）。")


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    # **擋在載入 4 GB 權重之前**：缺旗標的失效方式是跑完才發現損失不對。
    validate_loss_args(args)
    validate_patch_args(args)
    validate_attack_args(args)
    validate_deliver_args(args)
    validate_flow_args(args)
    validate_convergence_args(args)
    args.out.mkdir(parents=True, exist_ok=True)

    ip2p = IP2PWrapper(dtype=torch.float32)
    suite = MetricSuite(device=ip2p.device)
    dataset = load_dataset(args.data, prompt_index=args.prompt_index)
    dataset = select_images(dataset, args.images, args.data)
    # 三類攻擊指令。**就地覆寫 dataset 的 prompt**，於是下游（編輯、CSV 的
    # instruction 欄）全部自動跟著換，不必逐處改。
    if args.attack_category:
        import yaml as _y
        spec = _y.safe_load(args.attack_prompts.read_text(encoding="utf-8"))
        for d in dataset:
            d["prompt"] = attack_instruction(spec, d["name"], args.attack_category)
    print(f"IP2P 線：{len(dataset)} 張、條件 {args.conditions}、"
          f"steps={args.edit_steps} s_T={args.text_guidance} "
          f"s_I={args.image_guidance} seed={args.edit_seed}", flush=True)

    y_target = load_image_tensor(args.target, ip2p.device, size=RESOLUTION)
    loss_fn = make_encoder_loss(ip2p, args.loss, y_target)

    def _ig_weight_mask(x01):
        """`--ig-weight subject` 的權重遮罩，逐圖建一次。

        回傳 None 代表均勻平均，**呼叫路徑逐位元不變**。

        與 `--subject-mask` 的差別要講清楚：那一個把 `apply_where` 設成
        `1 − 遮罩`，限制擾動只能長在主體之外；這一個完全不限制擾動長在哪裡，
        只是把損失的空間平均改成加權平均。兩者可以並用，也可以只用其中一個，
        故各自一個旗標、各自一個 CSV 欄位。
        """
        args._ig_weight_text = ""
        if args.ig_weight == "uniform":
            return None
        import yaml as _yaml

        from src.defense.subject_mask import subject_mask

        spec = _yaml.safe_load(
            args.ig_weight_catalogue.read_text(encoding="utf-8"))
        texts = (spec.get("objects") or {}).get(args._cur_image)
        if not texts:
            raise SystemExit(
                f"{args.ig_weight_catalogue} 的 objects 裡沒有 "
                f"{args._cur_image} 的主體名稱。**不猜**——沒有主體名稱就沒有"
                f"「只對主體負責」這個概念。")
        texts = [texts] if isinstance(texts, str) else list(texts)
        m = subject_mask(x01, texts, threshold=args.subject_mask_threshold,
                         dilate=args.subject_mask_dilate,
                         feather=args.subject_mask_feather)
        args._ig_weight_text = " | ".join(texts)
        return m.to(x01)

    def _attack_embeds():
        """目錄檔裡所有指令的文字嵌入，(K,L,D)。

        **威脅模型沒有變**：這是一組**可能的**指令，不是這一格實際會被攻擊的
        那一句（那一句到編輯那一步才進來）。故取整份目錄裡的所有句子去重，
        不是這張影像那三句——只取那三句就等於偷看了這一格的攻擊。
        """
        import yaml as _y

        spec = _y.safe_load(args.attack_prompts.read_text(encoding="utf-8"))
        seen, texts = set(), []
        for rec in (spec.get("images") or {}).values():
            for t in (rec.get("prompts") or {}).values():
                if t not in seen:
                    seen.add(t)
                    texts.append(t)
        if not texts:
            raise SystemExit(f"{args.attack_prompts} 裡沒有任何指令")
        with torch.no_grad():
            embs = [ip2p.pipe._encode_prompt(t, ip2p.device, 1, False, None)
                    for t in texts]
        return torch.cat([e[-1:] for e in embs], dim=0).detach()

    def make_base_loss(x01):
        """逐圖取得主損失。

        `encoder_target` 與 `latent_norm` 與影像無關，回傳同一個物件，
        **呼叫路徑逐位元不變**；`image_guidance` 的 `z_t` 錨在原圖上，必須
        逐圖重建——用防禦圖當錨會讓取樣軌跡隨最佳化漂移，而且不會有症狀。
        """
        if args.loss not in ("image_guidance", "edit_divergence"):
            # `latent_norm`／`encoder_target` 本來就是決定性的，評估函數就是
            # 它自己——對照組因此與新損失走**同一條**收斂判定，不是兩套標準。
            if args.eval_every and not hasattr(loss_fn, "_fixed_eval"):
                loss_fn._fixed_eval = loss_fn
            return loss_fn
        if args.loss == "identity":
            from src.defense.carrier_mask import face_subject_mask
            from src.defense.identity_loss import make_identity_loss

            fm = face_subject_mask(x01, dilate=args.subject_mask_dilate,
                                   feather=args.subject_mask_feather)
            fn = make_identity_loss(
                ip2p, zt_mode=args.ig_zt, x_clean=x01, face_mask=fm,
                t_min=args.ig_t_min, t_max=args.ig_t_max,
                samples=args.ig_samples, seed=args.seed,
                box_margin=args.id_box_margin,
                layout_weight=args.id_layout_weight)
            if args.eval_every:
                fn._fixed_eval = fn.make_fixed(args.eval_draws, args.eval_seed)
            return fn
        if args.loss == "facelock":
            from src.defense.carrier_mask import face_subject_mask
            from src.defense.facelock_loss import (FACELOCK_START_FR,
                                                   FACELOCK_START_LPIPS,
                                                   FACELOCK_W_LATENT,
                                                   FACELOCK_W_LPIPS,
                                                   make_facelock_loss)

            fm = face_subject_mask(x01, dilate=args.subject_mask_dilate,
                                   feather=args.subject_mask_feather)
            fn = make_facelock_loss(
                ip2p, x_clean=x01, face_mask=fm, steps=args.steps,
                box_margin=args.id_box_margin,
                w_latent=(FACELOCK_W_LATENT if args.fl_w_latent is None
                          else args.fl_w_latent),
                w_lpips=(FACELOCK_W_LPIPS if args.fl_w_lpips is None
                         else args.fl_w_lpips),
                start_fr=(FACELOCK_START_FR if args.fl_start_fr is None
                          else args.fl_start_fr),
                start_lpips=(FACELOCK_START_LPIPS if args.fl_start_lpips is None
                             else args.fl_start_lpips))
            # 這個損失是**決定性**的（沒有 z_t 抽樣），故評估函數就是它自己；
            # 不包一層固定抽樣的話收斂判定會判到另一個量。
            if args.eval_every:
                fn._fixed_eval = fn
            return fn
        if args.loss == "edit_divergence":
            from src.defense.edit_divergence_loss import make_edit_divergence_loss
            fn = make_edit_divergence_loss(
                ip2p, zt_mode=args.ig_zt, x_clean=x01,
                t_min=args.ig_t_min, t_max=args.ig_t_max,
                samples=args.ig_samples, seed=args.seed,
                weight=_ig_weight_mask(x01))
            if args.eval_every:
                fn._fixed_eval = fn.make_fixed(args.eval_draws, args.eval_seed)
            return fn
        fn = make_image_guidance_loss(
            ip2p, zt_mode=args.ig_zt,
            x_clean=x01 if args.ig_zt == "diffuse_src" else None,
            t_min=args.ig_t_min, t_max=args.ig_t_max,
            samples=args.ig_samples, seed=args.seed,
            weight=_ig_weight_mask(x01),
            text_embeds=_attack_embeds() if args.prompt_eot else None)
        if args.eval_every:
            fn._fixed_eval = fn.make_fixed(args.eval_draws, args.eval_seed)
        if args.latent_norm_weight:
            fn = _blend_latent_norm(ip2p, x01, fn, args.latent_norm_weight)
        return fn

    def _blend_latent_norm(ip2p, x01, ig_fn, weight):
        """把舊損失按權重加到影像引導損失上。

        為什麼值得試
        ────────────────────────────────────────────────────────────
        兩個損失實測是**互補**的，不是誰取代誰（同兩張影像、同設定、只換
        `--loss`）：

        | | 未淨化 | JPEG 75 | JPEG 30 | 模糊 σ1 |
        |---|---|---|---|---|
        | `latent_norm` | **0.6843** | 0.2691 | 0.1232（**低於地板**） | 0.2154 |
        | `image_guidance` | 0.5850 | **0.4209** | **0.2604** | 0.1787 |

        舊的未淨化最強但一被壓縮就塌，新的未淨化較弱但每一級 JPEG 都撐得住。
        兩者從未加在一起過。

        **兩項的量級差很多**：`latent_norm` 在乾淨影像上是 70–80，影像引導的
        逐步值是 0.1–0.6。直接相加的話權重 1 實際只佔約 1/150。故把舊項除以
        它在**乾淨影像**上的值先正規化，兩項都由 1 起步，**權重 1 才真的是
        等權**——這與不動點項的 `normalised` 是同一個作法，理由也相同。

        評估函數也要一起包，否則收斂判定判的是另一個量。
        """
        with torch.no_grad():
            ref = float(ip2p.encode_image(x01).flatten().norm(p=2))
        ref = ref if ref > 0 else 1.0

        def ln(x_def):
            return ip2p.encode_image(x_def).flatten().norm(p=2) / ref

        def combined(x_def):
            return ig_fn(x_def) + weight * ln(x_def)

        base_eval = getattr(ig_fn, "_fixed_eval", None)
        if base_eval is not None:
            def combined_eval(x_def):
                return base_eval(x_def) + weight * ln(x_def)
            combined._fixed_eval = combined_eval
        combined._latent_norm_ref = ref
        return combined

    def wrap_manifold(x01, base_loss_fn):
        """把不動點項接到防禦損失上。**逐圖重建**：正規化的分母是該張乾淨
        影像上的殘差，換了影像就要重算，共用會讓權重的意義隨影像漂移。

        `--manifold-only` 是判準 F3 要求的歸因對照：只留不動點項、拿掉對抗項，
        用來分辨改善來自「迎合淨化器」還是來自失真型態改變。
        """
        term = make_normalised_term(
            ip2p, x01, t_max=args.manifold_t, seed=args.seed)
        # **兩項的量級差兩個數量級**：`latent_norm` 在乾淨影像上約 70–80，
        # 不動點項已正規化成 1。`raw` 直接相加，於是權重 1 實際只佔約 1/70；
        # `normalised` 把主項也除以它在乾淨影像上的值，兩項都由 1 起步，
        # **權重 1 才真的是等權**。預設 `raw` 是為了讓先跑的批次維持可解讀。
        base_ref = 1.0
        if args.manifold_balance == "normalised":
            with torch.no_grad():
                base_ref = abs(float(base_loss_fn(x01)))
            if not base_ref > 0:
                raise SystemExit(
                    "主損失在乾淨影像上為零，無法做等權正規化——"
                    "**不可以靜默退回 raw**，那會讓權重的意義隨影像而變")

        def fn(x_def):
            fix = term(x_def)
            if args.manifold_only:
                return fix
            return base_loss_fn(x_def) / base_ref + args.manifold_weight * fix

        fn.term = term
        fn.base_ref = base_ref
        return fn

    def edit(x01, item):
        return ip2p.edit(x01, item["prompt"], seed=args.edit_seed,
                         steps=args.edit_steps, s_t=args.text_guidance,
                         s_i=args.image_guidance)

    rows, trace_rows = [], []
    # **跳過的影像要把舊列帶回來。** 每張寫檔是整份重寫，只把這一輪跑過的
    # 列寫出去會把先前的量測**截掉**——PNG 還在、數字沒了，而報表上只會看到
    # 列數變少，不會拋錯。實際踩過一次（5 張防禦圖只剩 2 列）。
    if args.skip_existing:
        prev = args.out / "results.csv"
        if prev.exists():
            with prev.open(encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
            print(f"帶回 {len(rows)} 筆既有的列（--skip-existing）", flush=True)
    for item in dataset:
        x01 = load_image_tensor(item["path"], ip2p.device, size=RESOLUTION)
        item["path01"] = x01
        # `defend` 要靠它組出權重檔名。放在 args 上而不是多傳一層參數，
        # 是為了不動 `defend` 的簽名（測試釘住了它）。
        args._cur_image = item["name"]
        if args.skip_existing and all(
                (args.out / f"{item['name']}__{c}__def.png").exists()
                for c in args.conditions):
            print(f"{item['name']:32s} 已完成，跳過", flush=True)
            continue
        t0 = time.time()
        e_orig = edit(x01, item)
        vutils.save_image(x01, args.out / f"{item['name']}__orig.png")

        if args.check_only:
            # DEC-022 的前提檢查：編輯有沒有真的往指令走。
            so = suite.semantic(e_orig, item["prompt"])
            base = suite.semantic(x01, item["prompt"])
            vutils.save_image(e_orig, args.out / f"{item['name']}__check_edit.png")
            rows.append({
                "image": item["name"], "instruction": item["prompt"],
                "clip_orig": round(base["clip"], 5),
                "clip_edit": round(so["clip"], 5),
                "clip_gain": round(so["clip"] - base["clip"], 5),
                "siglip_orig": round(base["siglip"], 5),
                "siglip_edit": round(so["siglip"], 5),
                "siglip_gain": round(so["siglip"] - base["siglip"], 5),
                "pixel_lpips": round(float(suite.pairwise(x01, e_orig)["lpips"]), 5),
                "edit_steps": args.edit_steps, "s_t": args.text_guidance,
                "s_i": args.image_guidance, "edit_seed": args.edit_seed,
                "seconds": round(time.time() - t0, 1),
            })
            write_csv(args.out / "check.csv", rows)
            print(f"{item['name']:32s} clip_gain="
                  f"{rows[-1]['clip_gain']:+.4f} lpips={rows[-1]['pixel_lpips']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            continue

        # 不動點項的正規化分母逐圖算一次；關著時 `use_loss` 就是主損失
        # 本身，**呼叫路徑逐位元不變**。
        use_loss = make_base_loss(x01)
        if args.manifold_weight or args.manifold_only:
            use_loss = wrap_manifold(x01, use_loss)
            print(f"  不動點項：乾淨影像上的殘差 {use_loss.term.reference:.5f}"
                  f"、主項基準 {use_loss.base_ref:.4f}"
                  f"（權重 {args.manifold_weight}／{args.manifold_balance}"
                  f"{'，只留這一項' if args.manifold_only else ''}）", flush=True)

        for cond in args.conditions:
            t1 = time.time()
            x_def, radius, unreachable, modified, extras = defend(
                ip2p, suite, cond, x01, args, use_loss)
            for h in extras.pop("_trace", []):
                trace_rows.append({"image": item["name"], **h})
            fid = suite.pairwise(x01, x_def)
            e_def = edit(x_def, item)
            prot = suite.pairwise(e_orig, e_def)
            # 主讀數：兩張編輯輸出在 SigLIP 影像空間的距離。低於門檻即
            # 「攻擊方拿不到可用輸出」。在這裡算而不是事後補，是因為事後補
            # 依賴防禦圖還留在磁碟上，而影像不入版控。
            sim = suite.image_similarity(e_orig, e_def)
            for sub, img in (("def", x_def), ("edit_orig", e_orig),
                             ("edit_def", e_def)):
                vutils.save_image(img.clamp(0, 1),
                                  args.out / f"{item['name']}__{cond}__{sub}.png")
            w = extras.pop("_weights", None)
            if w is not None:
                torch.save(w, weights_path(args.out, item["name"], cond))
            rows.append({
                "image": item["name"],
                "condition": cond + ("_nodc" if args.skip_dc else ""),
                "attacker": "instruct-pix2pix",
                "instruction": item["prompt"], "task": item.get("class", ""),
                "radius": round(float(radius), 6), "unreachable": unreachable,
                "pixel_gate_sigma": args.pixel_gate_sigma,
                # r_min 決定放行哪些頻帶，是 2026-08-20 起在掃的變因。
                # 不逐列記下的話，同一個 θ 在不同 r_min 下的列長得一模一樣，
                # 合併分片之後就分不出來了。
                "r_min": args.r_min,
                "r_max": args.r_max,
                "gl_iters": args.gl_iters,
                "block": args.block,
                # hop 目前恆為 block//2，但那是 `PhaseResidual` 的預設而不是
                # 這裡的常數；不逐列記下的話，將來改動它會讓新舊列長得一樣。
                "hop": args.block // 2 if args.hop is None else args.hop,
                # 紋理閘的兩個設定。兩者都是本專案指定、無出處的值，按
                # CLAUDE.md 的規則必須是欄位而不是註解。此前只寫在 CLI 的
                # 預設值裡，掃描它們的批次在報表上分不出來。
                "quantile": args.quantile,
                "gate_edge_power": args.gate_edge_power,
                # 頻率閘的知覺權重。二值閘與加權閘跑出的列在其餘欄位上一模
                # 一樣，不記下來就無法在合併之後分辨。
                "freq_weight": args.freq_weight,
                "freq_weight_power": args.freq_weight_power,
                "survival_weight": args.survival_weight,
                "gain_weight": args.gain_weight,
                "phase_channels": args.phase_channels,
                # 色彩重映射的四個設定。**關著時仍然寫出來**——同一批裡
                # color_curve 與 color_grid 的列在其餘欄位上一模一樣，
                # 不記下來合併分片之後就分不出來。`color_bound_mode` 另有
                # 移植上的意義：advcf 是原程式的盒、symmetric 是本專案指定的。
                # 防禦端的種子。**此前沒有任何一欄記它**，於是「不最佳化」
                # 那一族（`*_rand`）的參數無法由 CSV 重建——它們的
                # `params()` 是空的，`--save-weights` 存不到東西，只能靠
                # 種子重抽。共防禦參照要重新套用同一個 D，缺這一欄就辦不到。
                "defense_seed": args.seed,
                "color_pieces": args.color_pieces,
                "color_bound_mode": args.color_bound_mode,
                "color_grid": args.color_grid,
                "color_luma_bins": args.color_luma_bins,
                "color_rand_draw": args.color_rand_draw,
                "spectral_floor": args.spectral_floor,
                # 加法項的價目分配。三個變體的總預算相同，跑出來的列
                # 在其餘欄位上一模一樣，不記下來合併之後就分不出來。
                "floor_gate": args.floor_gate,
                # 加法項自己的頻帶與存活加權。三者都不改變加法項的總預算
                # （價目表被縮放回同一個平均值），只改變它花在哪些頻格上，
                # 所以開著與關著跑出的列在其餘欄位上一模一樣，不記下來
                # 合併分片之後就分不出來。留空表示沿用 r_min／r_max。
                "floor_r_min": args.floor_r_min,
                "floor_r_max": args.floor_r_max,
                "floor_survival": args.floor_survival,
                # 可學的空間包絡。**關著時三欄仍然寫出來**，合併分片之後
                # 才分得出哪些列帶著它跑——三者都不改變加法項的總預算
                # （價目表被縮放回同一個平均值），只改變它花在哪些位置與
                # 哪些頻格上，所以開著與關著跑出的列在其餘欄位上一模一樣。
                # 學出來的五個純量在 `extras`（`env_beta`／`env_fc`／
                # `env_cy0` …），那是**結果**不是設定，只有開著時才有。
                "floor_envelope": args.floor_envelope,
                "floor_envelope_k": args.floor_envelope_k,
                "floor_envelope_scope": args.floor_envelope_scope,
                # 幅度相依的相位上限。關著與開著跑出的列在其餘欄位上
                # 一模一樣，不記下來合併之後就分不出來。
                "theta_budget": args.theta_budget,
                "coarsen": args.coarsen,
                # 位移場的粗網格邊長。本專案指定、論文未載，按 CLAUDE.md
                # 的規則必須是欄位而不是註解；換了它，本機量過的失真對照表
                # 就不再適用於這些列。
                "warp_grid": args.warp_grid,
                # stAdv 的流場正則。**關著時三欄仍然寫出來**，合併分片之後
                # 才分得出哪些列帶著它跑。`flow_eps` 是原文式子裡沒有的常數、
                # `flow_neighbourhood` 是原文未載明而由本專案指定的選擇，
                # 按 CLAUDE.md 兩者都必須是欄位而不是註解。關著時留空，
                # 表示該列與這一支無關。
                "flow_tau": args.flow_tau,
                "flow_eps": "" if args.flow_eps is None else args.flow_eps,
                "flow_neighbourhood": args.flow_neighbourhood or "",
                "update": args.update,
                "step_size": args.step_size,
                "saturate_at": args.saturate_at,
                "warp_init_std": args.warp_init_std,
                "disp_field_grid": args.disp_field_grid,
                "dct_mode": args.dct_mode,
                "dct_qd": args.dct_qd,
                "dct_pairing": args.dct_pairing,
                "dct_gate": args.dct_gate,
                "dct_plane_weight": args.dct_plane_weight,
                # 不動點項的設定。**關著時這些欄位仍然寫出來**，合併分片後才
                # 分得出哪些列帶著它跑。
                "manifold_weight": args.manifold_weight,
                "manifold_t": args.manifold_t,
                "manifold_only": int(args.manifold_only),
                "manifold_balance": args.manifold_balance,
                # 防禦端的 PGD 步數。**本方法預設 100，DCT-Shield 是 1000**
                # （該篇 §5.4），頭對頭表上這個差異從未被控制過，故逐列記下。
                "defense_steps": defense_steps(args, cond),
                "loss": args.loss,
                # image_guidance 的四個設定。**別的損失底下也照樣寫出來**，
                # 合併分片之後才分得出哪些列是它跑的；`ig_zt` 在別的損失底下
                # 是 None，欄位留空即該列與這一支無關。
                "ig_zt": args.ig_zt or "",
                # 影像引導殘差的空間權重。只活在 argparse 預設值裡的設定，
                # 合併分片之後在報表上分不出來（DEF「只活在 CLI 預設值裡的
                # 設定」），故逐列寫出，主體名詞也一併寫出。
                "patch_placement": args.patch_placement,
                # 載體那三欄：**每一列都寫出來**，非補丁的列也寫。只在開著時
                # 才有欄位會讓合併分片之後同一份表裡共存兩個 schema
                # （DEF「共存六個 CSV schema」）。補丁那些列的 `patch_carrier`
                # 會被 `param.geometry()` 從 `**extras` 覆蓋掉——那一份由實際
                # 跑的物件給，兩者分岔時它才會說實話。
                "patch_carrier": args.patch_carrier,
                # 攻擊指令的三類與主體的來源。每一列都寫——只活在 CLI 預設值
                # 裡的設定，合併分片之後在報表上分不出來。
                "attack_category": args.attack_category,
                "attack_prompts": ("" if args.attack_prompts is None
                                   else str(args.attack_prompts)),
                "subject_source": args.subject_source,
                "carrier_refine": args.carrier_refine,
                "carrier_erode": args.carrier_erode,
                "carrier_feather": args.carrier_feather,
                "carrier_scatter": args.carrier_scatter,
                "carrier_ring": args.carrier_ring,
                "carrier_ring_inner": args.carrier_ring_inner,
                "prompt_eot": int(args.prompt_eot),
                "attn_weight": args.attn_weight,
                "id_layout_weight": args.id_layout_weight,
                "id_box_margin": args.id_box_margin,
                "carrier_lattice": args.carrier_lattice,
                "carrier_dot_radius": args.carrier_dot_radius,
                "patch_tint": args.patch_tint,
                "patch_tint_sigma": args.patch_tint_sigma,
                # 固定形狀浮水印的三個幾何。每一列都寫，非補丁的列也寫。
                "patch_count": args.patch_count,
                "patch_crop_keep": args.patch_crop_keep,
                "patch_rects": "",
                "carrier_area": "",
                "carrier_source": "",
                "patch_init": args.patch_init,
                # `patch_lowfreq`／`patch_chroma` 由參數化物件在 geometry()
                # 裡給（與 patch_carrier 同一條理由），這裡只記載體那三個。
                "carrier_min_area": args.carrier_min_area,
                "carrier_max_area": args.carrier_max_area,
                "carrier_target_area": args.carrier_target_area,
                "carrier_match": args.carrier_match,
                "patch_tile": args.patch_tile,
                "patch_res": args.patch_res,
                "patch_palette": args.patch_palette,
                "patch_palette_temp": args.patch_palette_temp,
                "patch_seeds": args.patch_seeds,
                "patch_seed_temp": args.patch_seed_temp,
                "patch_polar": args.patch_polar,
                "patch_polar_bins": args.patch_polar_bins,
                # 四個都由 argparse 給，None 代表用了移植預設值；出表時要能
                # 分出「用了預設」與「明給了同一個數」。
                "fl_w_latent": ("" if args.fl_w_latent is None
                                else args.fl_w_latent),
                "fl_w_lpips": ("" if args.fl_w_lpips is None
                               else args.fl_w_lpips),
                "fl_start_fr": ("" if args.fl_start_fr is None
                                else args.fl_start_fr),
                "fl_start_lpips": ("" if args.fl_start_lpips is None
                                   else args.fl_start_lpips),
                "patch_alpha": args.patch_alpha,
                "patch_tv": args.patch_tv,
                "color_tv": args.color_tv,
                "color_tv_luma": args.color_tv_luma,
                "dct_loss": args.dct_loss,
                "ig_weight": args.ig_weight,
                "ig_weight_text": getattr(args, "_ig_weight_text", ""),
                "ig_t_min": args.ig_t_min,
                "ig_t_max": args.ig_t_max,
                "ig_samples": args.ig_samples,
                "gain_ratio": args.gain_ratio,
                "purify_aware": args.purify_aware,
                # **未載的參數要成為欄位不是註解**：集成的品質集合決定了
                # 這一列在哪一段壓縮上被訓練過，合併分片後必須分得出來。
                "eot_qualities": " ".join(str(q) for q in args.eot_qualities),
                # eot_broad 的另外兩族，與交付前的 L∞ 投影。**關著時也寫出來**，
                # 合併分片之後才分得出哪些列帶著它跑。
                "eot_sigmas": " ".join(str(v) for v in args.eot_sigmas),
                "eot_fractions": " ".join(str(v) for v in args.eot_fractions),
                "eot_classes": " ".join(args.eot_classes),
                "linf_deliver": args.linf_deliver,
                "consistency_weight": args.consistency_weight,
                "consistency_decay": args.consistency_decay,
                "latent_norm_weight": args.latent_norm_weight,
                # 交付自壓的品質。0 = 關閉。**這一欄不記下來，開著與關著跑出
                # 的列在其餘欄位上一模一樣**，合併分片之後就分不出來——而它
                # 決定了存檔的防禦圖在不在量化格點上，也就決定了抗淨化那一輪
                # 讀到的是什麼。`extras` 的四欄（保留率、餘弦、兩個 RMS）只有
                # 開著時才有。
                "deliver_jpeg": args.deliver_jpeg,
                # 分階段訓練的設定。**每一個未載的參數都要成為欄位不是註解**
                # ——關著時這些欄位仍然寫出來，合併分片後才分得出哪些列是
                # 兩段式跑出來的。逐圖的結果欄（退了幾次、守住多少）在
                # `extras` 裡，只有開著時才有。
                "stage2_steps": args.stage2_steps,
                "stage2_ops": " ".join(args.stage2_ops),
                "stage2_order": args.stage2_order,
                "stage2_step_scale": args.stage2_step_scale,
                "stage2_trust": args.stage2_trust,
                "stage2_check_every": args.stage2_check_every,
                "stage2_ramp": args.stage2_ramp,
                **extras,
                # DCT-Shield 的量化表由 `q_alg` 決定，base 是論文 §5.4 的
                # 0.95、Y-only 是 §6.3 的 0.85。此前它只在 CLI 預設值裡，
                # 兩個品質因子跑出的列在報表上分不出來（FND-058）。
                # `gamma` 是 §5.4 的步長係數，同理。
                "dct_q_alg": args.q_alg,
                "dct_gamma": PAPER_GAMMA,
                "wm_tau": args.wm_tau,
                "wm_mu": args.wm_mu,
                "wm_diagonals": "-".join(str(d) for d in args.wm_diagonals),
                "wm_q_embed": args.wm_q_embed,
                # 論文未載、本專案指定的三個推論參數逐列記下（DEC-031）
                "edit_steps": args.edit_steps, "s_t": args.text_guidance,
                "s_i": args.image_guidance, "edit_seed": args.edit_seed,
                "modified_from_paper": modified,
                **standard_row("fid_", fid),
                # CIEDE2000 的平均色差。**`standard_row` 只取定案清單的五項**，
                # 這一欄要自己接上去。色彩族沒有它就只剩結構度量，而 LPIPS 與
                # DISTS 對全域色偏的懲罰偏輕，等失真對齊會系統性偏袒色彩方法。
                "fid_deltaE00": round(fid["deltaE00"], 4),
                **standard_row("edit_", prot),
                "fid_linf": round(fid["linf"], 5),
                "fid_rms": round(fid["rms"], 5),
                "edit_lpips": round(float(prot["lpips"]), 5),
                # 擋下率的三欄。門檻逐列寫下的理由見
                # `src.metrics.standard.SIGLIP_BLOCKED_THRESHOLD`：它是本專案
                # 指定的值，改動之後舊列仍要可解讀。
                "edit_clip_sim": round(float(sim["clip"]), 5),
                "edit_siglip_sim": round(float(sim["siglip"]), 5),
                "blocked": blocked_by_siglip(sim["siglip"]),
                "siglip_blocked_threshold": SIGLIP_BLOCKED_THRESHOLD,
                "total_seconds": round(time.time() - t1, 1),
            })
            write_csv(args.out / "results.csv", rows)
            # 收斂軌跡累積後整份重寫。**不逐列 append**：欄位會隨條件變動
            # （只有開了 --eval-every 的列有 `eval`），append 會讓後來多出來
            # 的欄位靜默消失，而 CSV 看起來仍然完整。
            if trace_rows:
                write_csv(args.out / "trace.csv", trace_rows)
            keep = (f" keep={extras['deliver_retention']:.3f}"
                    if "deliver_retention" in extras else "")
            print(f"{item['name']:32s} {cond:14s} r={radius:.4f} "
                  f"dists={fid['dists']:.4f} lpips={fid['lpips']:.4f} "
                  f"effect={prot['lpips']:.4f}{keep} ({time.time() - t1:.0f}s)",
                  flush=True)

    out = args.out / ("check.csv" if args.check_only else "results.csv")
    print(f"\n表：{out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
