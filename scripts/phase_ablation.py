"""像素臂：參數化消融——加性 δ 對紋理重相位 θ 對隨機相位，同一個損失。

規格：`docs/superpowers/specs/2026-08-13-texture-rephasing-design.md` §3。

損失、更新規則、步數、種子、預算對齊程序全部固定，**唯一的變因是參數化**：

    add          φ = δ，L∞ 投影
    phase        φ = θ，紋理重相位
    phase_rand   同幅度的隨機 θ，即 RPN 本身，不最佳化

每個條件對半徑二分搜尋，使最終的 DISTS 落在目標上，故三者在同一個預算軸
的同一點上比較。相位有構造上的天花板（|θ| ≤ π），達不到的預算點標成
`unreachable` 而非 failed（與 FND-001 同型）。

評測直接沿用 `apa_baseline.evaluate`，不另寫一份——兩份評測會慢慢分岔而
沒有症狀，既有 runs/ 會靜默變得不可比。

用法：
    python scripts/phase_ablation.py --out runs/phaseA --data data/lo_aligned \
        --images horse_00 man_00 bird_03
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from apa_baseline import (  # noqa: E402
    EDIT_STRENGTH, MODEL_NAME, RESOLUTION, TARGET_IMAGE, evaluate, load_dataset,
)
from src.baselines.encoder_target import make_encoder_target_loss  # noqa: E402
from src.defense.param_pgd import (  # noqa: F401
    ShadingParam, ShadingRandomParam,  # noqa: E402
    WarpParam, WarpRandomParam, WarpRoundTripParam,  # noqa: E402
    AdditiveParam, PhaseParam, RandomPhaseParam, fit_to_budget, run_param_pgd,
)
from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

CONDITIONS = ("add", "phase", "phase_rand")

# 兩個預算點。0.075 是現行非加性條件的水位、0.04 是加性的（FND-026 的表）。
# 單點容易是巧合，故取兩點看趨勢。
BUDGETS = (0.04, 0.075)

# 人眼門檻（使用者 2026-08-13 於失真掃描頁上劃定）。給 --human-threshold 時
# 不做預算對齊，直接用這兩個半徑——**每個條件各自在自己的可接受上限上**，
# 這才是「匹配人眼可辨失真」的字面意思。以任何單一指標對齊都做不到這件事：
# 兩個門檻在十六項指標上沒有一項落在同一個值（最接近的 LPIPS 也差 1.28 倍）。
HUMAN_RADIUS = {"phase": 1.30, "phase_rand": 1.30, "add": 1.2 / 255.0}

# 加性的半徑上界取 32/255：Mist 在 [0,1] 上的等價 eps 是 16/255，留一倍
# 餘裕讓二分搜尋在上界內找得到 DISTS 0.075。
ADD_RADIUS_HI = 32.0 / 255.0
ADD_RADIUS_LO = 0.5 / 255.0
PHASE_RADIUS_LO = 0.05
# 候選二的半徑界：粗網格上 log 增益的 L-infinity 界。0.30 時亮部最多變
# exp(0.30) = 1.35 倍；再大就必然把亮部整片推到飽和，clamp 之後多出來的
# 預算不會變成擾動，只會變成一塊死白。下界 0.02 對應 ±2% 的明暗變化。
SHADING_RADIUS_LO = 0.02
SHADING_RADIUS_HI = 0.30
# WaNet 式三元對照的半徑界。**單位是最大位移像素數**，夾在 16x16 粗網格的
# 係數上（上採樣會過衝）。上界 48 px 是本專案指定：本機量過的失真對照表
# （`WarpParam` 的 docstring）在 24 px 就到 DISTS 0.16、PSNR 17.4，已經超出
# 失真帶；留到 48 是為了讓 `warp_roundtrip` 也掃得到帶內——它的幾何幾乎抵消，
# 同一個半徑上的失真遠低於單次 warp。下界 1 px 是取樣格點的量級，再小就只是
# 內插誤差。
WARP_RADIUS_LO = 1.0
WARP_RADIUS_HI = 48.0

# 旋轉角上界的二分搜區間。上界封頂在 pi（`theta = pi` 是聯合翻號，再大繞回去）。
DCT_ROTATE_RADIUS_LO = 0.2
DCT_ROTATE_RADIUS_HI = math.pi
# 粗網格邊長。16 是本專案指定，與本機量過的失真對照表同一個構造；
# 換掉它那張表就作廢，故它是 CSV 的欄位（`warp_grid`）而不是註解。
WARP_GRID = 16


def build(name: str, seed: int, block: int = 32, r_min: float = 0.12,
          hop=None,
          quantile: float = 0.5, gl_iters: int = 0, pixel_gate_sigma: float = 0.0,
          gain_ratio: float = 0.0, r_max: float = float("inf"),
          gate_edge_power: float = 1.0, freq_weight: str = "binary",
          freq_weight_power: float = 1.0,
          survival_weight: str = "none",
          gain_weight: str = "shared", channels: str = "rgb",
          spectral_floor: float = 0.0, floor_gate: str = "uniform",
          theta_budget: float = 0.0, warp_grid: int = WARP_GRID,
          coarsen: int = 1,
          dct_qd: float = 0.85, dct_pairing: str = "transpose",
          dct_gate: str = "texture", warp_init_std: float = 0.0,
          dct_mode: str = "plane", dct_plane_weight: str = "uniform",
          disp_field_grid: int = 16):
    """`block`／`r_min`／`quantile` 是相位算子的三個構造設定。

    預設值是 現行定案（`docs/METHOD.md` §4）。開放成參數是為了掃描
    「約束落在哪個頻帶、哪些區塊」對效果與失真的取捨——三者都改變**閘**，
    也就是改變擾動被允許出現的位置，不改變損失或更新規則。
    """
    if name == "add":
        return AdditiveParam(radius=ADD_RADIUS_HI), ADD_RADIUS_LO, ADD_RADIUS_HI
    if name.startswith("disp_k"):
        # 色散變形（`src/defense/dispersion.py`）：K 個頻帶各自一個隨機位移。
        # K=1 就是古典位移場、`disp_kfull` 是逐頻格獨立的隨機相位，也就是
        # 現行家族的隨機對照。**兩族的半徑單位不同**——逐頻帶位移是像素、
        # 逐頻格相位是弧度，故二分搜的區間分開給。
        from src.defense.dispersion import DispersionParam
        tail = name[len("disp_k"):]
        # `_opt` 後綴＝**可學＋接上兩個閘**（`runs/ip2p_dispersion` 那一批是
        # 隨機且不接閘，殘差因此是滿幅度尖峰、L∞ 0.98）。
        learn = tail.endswith("_opt")
        if learn:
            tail = tail[:-4]
        n_bands = None if tail == "full" else int(tail)
        lo, hi = ((0.05, math.pi) if n_bands is None else (0.05, 16.0))
        return (DispersionParam(radius=hi, n_bands=n_bands, block=block,
                                hop=hop if hop else block // 2, r_min=r_min,
                                field_grid=disp_field_grid,
                                learnable=learn, gate=learn,
                                energy_quantile=quantile,
                                gate_edge_power=gate_edge_power,
                                freq_weight_power=freq_weight_power),
                lo, hi)
    if name in ("dct_nonadd", "dct_nonadd_rand"):
        # DCT 域的**非加性**擾動（`src/defense/dct_nonadditive.py`）。
        # 半徑即角度（plane／shared_plane）或 log 增益（gain）的上界。
        # 帶內工作點未知，故二分搜區間取得寬：[0.1, pi]。
        from src.defense.dct_nonadditive import (
            DctNonAdditiveParam, DctNonAdditiveRandomParam)
        cls = (DctNonAdditiveParam if name == "dct_nonadd"
               else DctNonAdditiveRandomParam)
        return (cls(radius=DCT_ROTATE_RADIUS_HI, mode=dct_mode,
                    r_min=r_min, gate=dct_gate,
                    gate_edge_power=gate_edge_power),
                0.1, DCT_ROTATE_RADIUS_HI)
    if name in ("dct_unified", "dct_unified_rand"):
        # **整併版**：學出來的旋轉平面直接作用在量化後的整數係數上，交付即
        # 參數（`src/defense/dct_unified.py`）。半徑即角度上界，封頂意義同下。
        # 帶內工作點由 `nd_plane` 家族外推：浮點版 theta 2.5 是 DISTS 0.1285、
        # 事後投影版是 0.1617，故三點取 1.8／2.2／2.5 夾住失真帶
        # 0.1286–0.1447。二分搜區間沿用 [0.2, pi]。
        from src.defense.dct_unified import (
            DctUnifiedParam, DctUnifiedRandomParam)
        cls = (DctUnifiedParam if name == "dct_unified"
               else DctUnifiedRandomParam)
        return (cls(radius=DCT_ROTATE_RADIUS_HI, qd=dct_qd,
                    r_min=r_min, gate=dct_gate,
                    gate_edge_power=gate_edge_power,
                    plane_weight=dct_plane_weight),
                DCT_ROTATE_RADIUS_LO, DCT_ROTATE_RADIUS_HI)
    if name in ("dct_rotate", "dct_rotate_rand"):
        # DCT 域的保長配對旋轉（`runs/dct_phase_design/README.md`）。
        # 半徑就是旋轉角上界 theta_max，封頂在 pi——`theta = pi` 恰好是聯合
        # 翻號，也就是 DCT 上的離散相位操作，再大只是繞回去。
        # 帶內工作點實測在 theta 約 1.04–1.13（紋理閘、十張，`ceiling.csv`），
        # 故二分搜的區間取 [0.2, pi]。
        from src.defense.dct_rotation import (
            DctRotationParam, DctRotationRandomParam)
        cls = (DctRotationParam if name == "dct_rotate"
               else DctRotationRandomParam)
        return (cls(radius=DCT_ROTATE_RADIUS_HI, qd=dct_qd,
                    pairing=dct_pairing, gate=dct_gate),
                DCT_ROTATE_RADIUS_LO, DCT_ROTATE_RADIUS_HI)
    if name in ("shading", "shading_rand"):
        # 候選二：極低頻的乘性明暗場。**與相位算子的頻帶不相交**
        # （r_min = 0.12 以上 對 f_n < 0.03 以下），所以它可以疊加而不是取代。
        # 半徑是粗網格上 log 增益的 L-infinity 界：0.30 時亮部最多變 exp(0.3)
        # = 1.35 倍。上界取 0.30 是因為再大就必然把亮部整片推到飽和，
        # clamp 之後多出來的預算不會變成擾動，只會變成一塊死白。
        cls = ShadingParam if name == "shading" else ShadingRandomParam
        return cls(radius=SHADING_RADIUS_HI), SHADING_RADIUS_LO, SHADING_RADIUS_HI
    if name in ("warp", "warp_rand", "warp_roundtrip"):
        # WaNet 式三元對照（`runs/ip2p_warp/`）。三格共用同一個參數化，
        # 差別只在最佳化與幾何：
        #   warp            最佳化的位移場
        #   warp_rand       同半徑的隨機場，不最佳化
        #   warp_roundtrip  與 warp_rand **同一個**隨機場，先 f 再 −f
        # `warp_rand` vs `warp_roundtrip` 問「幾何本身有沒有貢獻」，
        # `warp` vs `warp_rand` 問「最佳化有沒有買到東西」（＝ FND-004）。
        # 半徑的單位是最大位移像素數，強度旗鈕是 `--radius`。
        cls = {"warp": WarpParam, "warp_rand": WarpRandomParam,
               "warp_roundtrip": WarpRoundTripParam}[name]
        kw = {"init_std": warp_init_std} if name == "warp" else {}
        return (cls(radius=WARP_RADIUS_HI, grid=warp_grid, **kw),
                WARP_RADIUS_LO, WARP_RADIUS_HI)
    if name == "phase":
        return (PhaseParam(size=RESOLUTION, block=block, r_min=r_min,
                           hop=hop, r_max=r_max,
                           energy_quantile=quantile, gl_iters=gl_iters,
                           pixel_gate_sigma=pixel_gate_sigma,
                           gate_edge_power=gate_edge_power,
                           freq_weight=freq_weight,
                           freq_weight_power=freq_weight_power,
                           survival_weight=survival_weight,
                           gain_weight=gain_weight,
                           channels=channels,
                           spectral_floor=spectral_floor,
                           floor_gate=floor_gate,
                           theta_budget=theta_budget,
                           coarsen=coarsen),
                PHASE_RADIUS_LO, math.pi)
    if name == "floor_only":
        # 相位與幅度都不動，只留頻譜加性下限。**強度旗鈕是 `--spectral-floor`
        # 而不是 radius**——radius 在這一格上完全沒有作用（theta 凍結在 0，
        # gain_max = radius × 0 = 0），CSV 上那一欄照樣寫出來但不可解讀。
        #
        # 為什麼要這一格：`DECISIONS.md` 撤回「不做加性項」時，唯一站得住的
        # 理由是「非加性那一半買的是感知代價」，而那句話的證據是一次性探針裡
        # `radius 0.1` 的近似（theta_max 與 gain_max 都還有 0.1，不是真的關掉），
        # 且來自已刪除的程式。真正的「只有加性」從未在主線程式上跑過。
        # 今日又量到 `--spectral-floor 0.04` 時加法項佔可用預算的 67.6%
        # （`runs/ip2p_residual_signature/allowed_budget_gini.csv`），這一格
        # 因此是整個加性裁決底下最該補的對照。
        if spectral_floor <= 0:
            raise ValueError(
                f"floor_only 需要 spectral_floor > 0，收到 {spectral_floor}")
        return (PhaseParam(size=RESOLUTION, block=block, r_min=r_min,
                           hop=hop, r_max=r_max,
                           energy_quantile=quantile, gl_iters=gl_iters,
                           pixel_gate_sigma=pixel_gate_sigma,
                           gain_ratio=0.0, phase_on=False,
                           gate_edge_power=gate_edge_power,
                           freq_weight=freq_weight,
                           freq_weight_power=freq_weight_power,
                           survival_weight=survival_weight,
                           gain_weight=gain_weight,
                           channels=channels,
                           spectral_floor=spectral_floor,
                           floor_gate=floor_gate,
                           theta_budget=theta_budget,
                           coarsen=coarsen),
                PHASE_RADIUS_LO, math.pi)
    if name in ("phase_gain", "gain_only"):
        # 2026-08-21 的改動一：幅度譜也可學。`gain_only` 把 theta 凍結在 0，
        # 用來分辨「幅度單獨有沒有用」與「兩者是否相加」。
        #
        # **上界不是 pi**：相位是週期量所以封頂在 pi，增益不是，這正是加它的
        # 主要理由。上界取 8.0 是本專案指定的掃描上限，沒有出處——超過那裡
        # exp(8) ~ 3000 倍，影像早就毀了，掃上去只是浪費機時。
        if gain_ratio <= 0:
            raise ValueError(f"{name} 需要 gain_ratio > 0，收到 {gain_ratio}")
        return (PhaseParam(size=RESOLUTION, block=block, r_min=r_min,
                           hop=hop, r_max=r_max,
                           energy_quantile=quantile, gl_iters=gl_iters,
                           pixel_gate_sigma=pixel_gate_sigma,
                           gain_ratio=gain_ratio,
                           phase_on=(name == "phase_gain"),
                           gate_edge_power=gate_edge_power,
                           freq_weight=freq_weight,
                           freq_weight_power=freq_weight_power,
                           survival_weight=survival_weight,
                           gain_weight=gain_weight,
                           channels=channels,
                           spectral_floor=spectral_floor,
                           floor_gate=floor_gate,
                           theta_budget=theta_budget,
                           coarsen=coarsen),
                PHASE_RADIUS_LO, 8.0)
    if name == "phase_rand":
        return (RandomPhaseParam(size=RESOLUTION, block=block, r_min=r_min,
                                 hop=hop, r_max=r_max,
                                 energy_quantile=quantile, gl_iters=gl_iters,
                                 gate_edge_power=gate_edge_power,
                                 freq_weight=freq_weight,
                                 freq_weight_power=freq_weight_power,
                           survival_weight=survival_weight,
                                 gain_weight=gain_weight,
                                 channels=channels,
                                 spectral_floor=spectral_floor,
                                 floor_gate=floor_gate,
                           theta_budget=theta_budget,
                           coarsen=coarsen),
                PHASE_RADIUS_LO, math.pi)
    raise ValueError(f"未知條件 {name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--budgets", nargs="+", type=float, default=list(BUDGETS))
    ap.add_argument("--human-threshold", action="store_true",
                    help="不做預算對齊，直接用 HUMAN_RADIUS 的人眼門檻半徑")
    ap.add_argument("--target", type=Path, default=Path(TARGET_IMAGE))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="用 prompts.yaml 的第幾個編輯 prompt（0 改內容、1 改場景）")
    ap.add_argument("--block", type=int, default=32, help="重疊區塊邊長")
    ap.add_argument("--r-min", type=float, default=0.12, help="徑向頻率閘的下限")
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="紋理閘的梯度能量參考分位數")
    ap.add_argument("--gate-edge-power", type=float, default=1.0,
                    help="紋理閘壓制邊緣那個因子的指數："
                         "(1 - coherence^2) ** 本值。1.0 = 現行行為（逐位元"
                         "相同），0 = 完全不壓制邊緣。本值無出處，是本專案指定")
    ap.add_argument("--gl-iters", type=int, default=0,
                    help="Griffin-Lim 迭代投影的輪數。>0 時把 STFT 一致性投影"
                         "誤差壓下去，用來判別效果來自相位重排還是新造的能量"
                         "（FND-040／049）。0 = 關閉，與既有批次逐位相同")
    ap.add_argument("--phase-radius", type=float, default=None,
                    help="覆寫人眼門檻的相位半徑（只在 --human-threshold 下有效）")
    ap.add_argument("--add-radius", type=float, default=None,
                    help="覆寫人眼門檻的加性半徑 eps，單位是 [0,1]（只在 "
                         "--human-threshold 下有效）。人眼門檻是 1.2/255 = 0.0047")
    ap.add_argument("--edit-strength", type=float, default=EDIT_STRENGTH,
                    help="SDEdit 的 strength。像素臂的攻擊與強度無關，這個旗標"
                         "只影響評測，供強度掃描使用")
    ap.add_argument("--purify-aware", choices=("none", "jpeg"), default="none",
                    help="DEC-027：把可微分 JPEG 放進最佳化迴圈的前向，讓擾動"
                         "自己找壓縮活得下來的位置。品質沿 95→50 的課程排程。"
                         "**改變的是防禦圖，不是評測**——交出去的仍是未經淨化的"
                         "防禦圖，且條件標籤會加上 `_pa` 以免與既有批次混淆")
    ap.add_argument("--tag-suffix", type=str, default="",
                    help="附加在條件標籤後，讓同一個 --out 下的多組設定不互相覆寫檔名")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)
    y_target = load_image_tensor(args.target, sd.device, size=RESOLUTION)
    loss_fn = make_encoder_target_loss(sd, y_target)

    def dists_of(a, b):
        return float(suite.pairwise(a.clamp(0, 1), b)["dists"])

    transform = None
    if args.purify_aware == "jpeg":
        from src.defense.purify_aware import make_jpeg_transform

        transform = make_jpeg_transform(args.steps)
        args.tag_suffix = args.tag_suffix + "_pa"
        print(f"[purify-aware] JPEG 課程排程進入最佳化迴圈，"
              f"標籤加上 _pa（{args.steps} 步）", flush=True)

    dataset = load_dataset(args.data, prompt_index=args.prompt_index)
    if args.images:
        dataset = [d for d in dataset if d["name"] in args.images]

    rows = []
    for item in dataset:
        item["path01"] = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        save_image(item["path01"], args.out / f"{item['name']}__orig.png")
        print(f"\n########## {item['name']} ({item['class']}) ##########", flush=True)

        budgets = ["human"] if args.human_threshold else args.budgets
        for budget in budgets:
            for cond in args.conditions:
                tag = (f"{cond}__human" if budget == "human"
                       else f"{cond}__d{budget:g}") + args.tag_suffix
                print(f"=== {item['name']} / {tag} ===", flush=True)
                t0 = time.time()
                param, lo, hi = build(cond, args.seed, block=args.block,
                                      r_min=args.r_min, quantile=args.quantile,
                                      gl_iters=args.gl_iters,
                                      gate_edge_power=args.gate_edge_power)
                if budget == "human":
                    r_human = HUMAN_RADIUS[cond]
                    if args.phase_radius is not None and cond in ("phase", "phase_rand"):
                        r_human = args.phase_radius
                    if args.add_radius is not None and cond == "add":
                        r_human = args.add_radius
                    param.set_radius(r_human)
                    res = run_param_pgd(item["path01"], param, loss_fn,
                                        steps=args.steps, seed=args.seed,
                                        transform=transform)
                    fit = {"unreachable": False, "target": r_human,
                           "reached": dists_of(res.x_def, item["path01"]),
                           "radius": param.radius}
                else:
                    res = fit_to_budget(
                        item["path01"], param, loss_fn, dists_of, budget,
                        lo=lo, hi=hi, steps=args.steps, seed=args.seed,
                        rounds=args.rounds, transform=transform,
                    )
                    fit = res.history[-1]
                metrics, eo, ed = evaluate(sd, suite, aes, item, res.x_def,
                                           strength=args.edit_strength)
                for sub, img in (("def", res.x_def), ("edit_orig", eo),
                                 ("edit_def", ed)):
                    save_image(img, args.out / f"{item['name']}__{tag}__{sub}.png")

                row = {
                    "image": item["name"], "condition": cond,
                    "prompt_index": args.prompt_index, "prompt": item["prompt"],
                    "block": args.block, "r_min": args.r_min,
                    "quantile": args.quantile, "target_image": str(args.target),
                    "budget_target": budget,
                    "budget_mode": "human" if budget == "human" else "dists",
                    "budget_reached": round(float(fit["reached"]), 5),
                    "unreachable": bool(fit["unreachable"]),
                    "radius": round(float(fit["radius"]), 5),
                    "total_seconds": round(time.time() - t0, 1),
                    **metrics,
                }
                if cond in ("phase", "phase_rand"):
                    row["amp_dev"] = round(
                        param.module.amplitude_deviation(item["path01"]), 5)
                    row["active_fraction"] = round(param.module.active_fraction(), 4)
                rows.append(row)
                print(row, flush=True)
                write_csv(args.out / "results.csv", rows)

    print(f"\n表：{args.out / 'results.csv'}")


if __name__ == "__main__":
    main()
