"""把本專案的殘差模塊放進 Lo et al. 的評測路徑，取得可與基準並排的數字。

    python scripts/run_ours_lo_eval.py --data data/lo_aligned \
        --out runs/ours_lo_PF --sites PF --eval_seeds 20

為什麼不直接用 run_defense.py
─────────────────────────────────────────────────────────────────────

`run_defense.py` 有自己的評測路徑（淨化掃描、對齊階段、泛化掃描），且

1. 它讀 `prompts.yaml` 的格式是 `{類別: [prompt, ...]}`，而 `data/lo_aligned`
   是 `{類別: {content: ..., prompts: [...]}}`。`list(dict)` 會把 `content`
   與 `prompts` 兩個**鍵名**當成 prompt 餵進去，沒有任何症狀。
2. 它每格只評一個噪聲種子。基準是 20 個種子的平均，SDEdit 對種子高度敏感，
   n=1 對 n=20 的比較讀不出東西——那正是 E29 之前一連串判定問題的來源。

本腳本改為直接沿用 `run_lo_baseline` 的 `reference_edits` 與 `evaluate`，
確保「未防禦的編輯」與「防禦後的編輯」兩端的取樣、種子偏移、步數、guidance
與基準逐字相同。只有產生 `x_def` 的那一步不同，那正是要比較的東西。

失真並未匹配
─────────────────────────────────────────────────────────────────────

本專案以 LPIPS 綁定約束（τ_lpips），基準以 L∞ ≤ κ 硬投影。實測基準在
κ = 0.06 上的擾動 LPIPS 為 0.49–0.54，是本專案 τ = 0.10 的 4–6 倍。
**兩者的數字因此不是同一條軸上的點**，本腳本不做匹配、也不假裝有匹配：
`pert_lpips` 與 `pert_linf` 逐格寫出，讀表時必須併看。匹配失真的掃描是
後續工作（見規格 §7）。

超參數
─────────────────────────────────────────────────────────────────────

沿用 `runs/e29c_P_tau0.10/env.json`：本專案唯一在有效威脅模型（攻擊方
guidance = 7.5）下校準過的一組。唯一的更動是 strength 0.5 → 0.3，理由是
必須與 `run_lo_baseline` 的評測設定相同，否則量到的是兩個不同的攻擊。
"""

import argparse
import csv
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.defense.linf_attack import LinfAttackConfig  # noqa: E402
from src.defense.objective import LossConfig  # noqa: E402
from src.defense.optimize import OptimConfig, optimize  # noqa: E402
from src.purify.ops import default_train_set  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image, save_json  # noqa: E402
from src.utils.device import get_device, peak_memory_mb, reset_peak_memory  # noqa: E402
from scripts.run_defense import build_module  # noqa: E402
from scripts.run_lo_baseline import (  # noqa: E402
    TABLE1,
    append_csv,
    completed_pairs,
    evaluate,
    load_dataset,
    reference_edits,
)


# 每個 site 各自校準過的學習率。**不可共用一個值。**
#
# φ 的量綱逐 site 不同：site P／PF 的 φ 是像素值（[0,1] 尺度），site S 的 φ 是
# 位移（像素尺度），site C 的 φ 是色度矩陣的偏離量。Adam 每步的位移量約為
# lr，故同一個 lr 在三個位置代表三種不同的實際步長。這正是 LEDGER 6.1
# （E21–E23 §5.4）記錄的「固定步數的網格是錯的協議」的同一個根因。
#
# 出處逐項可查：
#   S  → 0.1    E14 在 τ=0.05、beta_linf=0 下的掃描（0.03／0.1／0.3／1.0），
#               tail5_shift 0.3297 為最大且 LPIPS 在預算內。docs/RESULTS_E13-E23.md §2
#   P  → 0.008  同一張表，tail5_shift 0.3584。
#   PF → 0.008  唯一跑過全秩的 e8 用的值；LEDGER 6.14 實測 0.03 會震盪
#               （保真罰項 29→76→27→43→54）而必撞步數上限。
#   C  → 0.3    e27d 定出的兩個條件之一（runs/e27d_C_lr0.3/）。
#
# 2026-08-04 修正（before/after 見 docs/LEDGER.md 6.16）：本腳本原本只有一個
# `--lr`（預設 0.03），runs/ours_lo/ 那批以 `--lr 0.008` 執行，**site S 因此
# 跑在其校準值的 1/12.5**。後果由該批的 history.json 直接讀得到：man_00 的
# site S 在 150 步內三道 hinge 一次都沒有啟動（pen_lpips／pen_acut／pen_chroma
# 全為 0/150），末端 LPIPS 只有 0.0163，即預算 0.10 的 16%。
SITE_LR = {"P": 0.008, "PF": 0.008, "S": 0.1, "C": 0.3}


def parse_lr(spec: str) -> dict:
    """`--lr` 接受單一數值或 `site=值` 的逗號清單，回傳 {site: lr}。

    單一數值會套用到全部 site。這是本腳本唯一允許把單一 lr 套用到全部
    site 的入口，且必須是呼叫端明寫的——預設值不是它，理由見 SITE_LR。
    """
    spec = spec.strip()
    if "=" not in spec:
        v = float(spec)
        return {s: v for s in SITE_LR}
    out = dict(SITE_LR)
    for part in spec.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in SITE_LR:
            raise ValueError(
                f"--lr 指定了未知的 site {k!r}；已知的是 {sorted(SITE_LR)}"
            )
        out[k] = float(v)
    return out


def budget_thresholds(tau_lpips: float, root: Path) -> tuple:
    """依 LPIPS 預算查出該預算的 τ_acut 與 τ_chroma。

    `LossConfig` 的 `tau_acut = 0.04` 與 `tau_chroma = 0.8` 是**絕對值**，
    而且是在 τ_lpips = 0.05 的量級上由人眼判讀出來的（objective.py 的修訂
    之三與之五）。兩道副約束的職責是相對的——擋住最佳化去換 LPIPS 收費不足
    的兩種失真（模糊、空間連貫色偏）——故門檻必須隨預算而變，否則預算放大
    時它們會變成真正的有效約束，量到的就不是「該方法在此 LPIPS 預算下的能力」。
    這是 LEDGER 2.9／2.10 的判定，逐預算的值由 p14 實測（LEDGER 2.11／2.12）。

    2026-08-04 修正：runs/ours_lo/ 那批跑在 τ_lpips = 0.10，卻沿用了 0.04 與
    0.8。site PF 的 man_00 因此有 31/48 步被鈍化 hinge 綁住，而該預算的值
    應為 0.0598（高 49%）。before/after 見 docs/LEDGER.md 6.17。

    查表而非寫死係數：`thresholds.csv` 是量測結果，讓程式讀它可保證兩者
    不會各自漂移。預算不在表上時拋出，不內插——p14 §12 已量到 acut 軸的
    分離度隨預算塌掉（0.05 時 5.12 倍、0.28 時 1.39 倍），線性內插沒有依據。
    """
    f = root / "runs/p14_budget_thresholds/thresholds.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} 不存在。逐預算門檻是 τ_acut／τ_chroma 的唯一依據，"
            "缺少時不可退回 LossConfig 的預設值（那是 τ_lpips=0.05 的值）"
        )
    table = {}
    with open(f, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            table[round(float(row["budget"]), 6)] = (
                float(row["tau_acut"]), float(row["tau_chroma"]))
    key = round(float(tau_lpips), 6)
    if key not in table:
        raise KeyError(
            f"τ_lpips = {tau_lpips} 不在 {f} 的預算清單 {sorted(table)} 上。"
            "不內插：acut 軸的分離度隨預算塌掉（0.05 時 5.12 倍、0.28 時 "
            "1.39 倍），內插沒有依據。請先跑 scripts/p14_budget_thresholds.py "
            "補上該預算，或明確以 --tau_acut／--tau_chroma 指定"
        )
    return table[key]


def main():
    ap = argparse.ArgumentParser(
        description="本專案的殘差模塊，走 Lo et al. 的評測路徑"
    )
    ap.add_argument("--data", default="data/lo_aligned")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--sites", default="PF",
                    help="逗號分隔。P 加性低秩／PF 加性全秩／S 空間變形／C 色度")
    ap.add_argument("--rank", type=int, default=16,
                    help="site S 與 C 會把它重新解讀為控制網格邊長")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt_index", type=int, default=0)
    # ---- 沿用 e29c 的校準值 ----
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument(
        "--lr", default="",
        help="留空表示每個 site 各用自己校準過的值（見 SITE_LR）。"
             "可寫成單一數值（套用到全部 site，須自行負責）或 "
             "`PF=0.008,S=0.1` 這種逐 site 的清單。"
             "**不要共用一個值**：φ 的量綱逐 site 不同，同一個 lr 在三個位置"
             "代表三種不同的實際步長",
    )
    ap.add_argument("--tau_lpips", type=float, default=0.10)
    ap.add_argument(
        "--tau_acut", type=float, default=None,
        help="鈍化 hinge 的門檻。留空表示依 --tau_lpips 由 "
             "runs/p14_budget_thresholds/thresholds.csv 查逐預算的值。"
             "LossConfig 的預設 0.04 是 τ_lpips=0.05 的量級上定出來的絕對值，"
             "在更大的預算上會變成真正的有效約束（LEDGER 2.9／2.10）",
    )
    ap.add_argument(
        "--tau_chroma", type=float, default=None,
        help="色度偏壓 hinge 的門檻。留空表示同上查表。"
             "LossConfig 的預設 0.8 同樣是 τ_lpips=0.05 的值",
    )
    ap.add_argument("--margin", type=float, default=1.0)
    # 以下兩個係數的 dataclass 預設值都不是本專案要的，必須明確覆寫。
    #
    # beta_linf：預設 100.0，配 tau_linf = 0.06。留著它，綁定的就是 L∞ 而不是
    #   LPIPS。對 site S 是致命的——把一條邊緣移動不到一個像素，L∞ 可接近 1
    #   而人眼幾乎看不出來，那正是本專案主張 L∞ 不是好失真尺的理由。2026-08-03
    #   實測後果：site S 停在 pert_linf 0.0750（= tau_linf + 罰項），pert_lpips
    #   只有 0.0034，離 τ = 0.10 的預算還有 29 倍，等於完全沒有施展空間。
    # alpha_lpips：預設 1.0，即 LPIPS 在 τ 以內也要收費，最佳化因此不會把
    #   預算用滿，τ 就不是真正的匹配軸（見 objective.py line 244-252）。
    #
    # 兩者在 runs/e29c_P_tau0.10/env.json 裡都是 0.0。
    ap.add_argument("--beta_linf", type=float, default=0.0)
    ap.add_argument("--alpha_lpips", type=float, default=0.0)
    ap.add_argument("--defense_mode", default="untargeted",
                    choices=["untargeted", "targeted"])
    # 訓練期的淨化 EOT。**預設維持既有行為，但這是一個有記錄的不對稱。**
    #
    # `default_train_set()` 是 identity／blur σ=1.0／JPEG q=75 三個算子，
    # `OptimConfig.purify_mode` 預設 "rotate" 且 `n_eot = 1`，故每一步只用其中
    # 一個、逐步輪替：2/3 的梯度步是在「被淨化過的輸入」上求的。
    #
    # 而本腳本的評測路徑（沿用 run_lo_baseline.evaluate）**完全不做淨化**，
    # 基準的三個 PGD 攻擊也不做。也就是說本專案的條件在訓練時多背了一個評測
    # 從不量的目標，兩邊不在同一個問題上。抗淨化是本專案相對基準論文的加項
    # （規格 §3.2），不是缺陷，但它必須是呼叫端明選的而不是預設值帶進來的。
    ap.add_argument(
        "--no_purify_train", action="store_true",
        help="訓練期只用 identity，不做淨化 EOT。基準的三個攻擊都不做淨化，"
             "本腳本的評測也不量淨化，故加上此旗標時兩個條件在同一個問題上；"
             "不加則本專案的條件多背一個評測不量的目標（見規格 §3.2）",
    )
    ap.add_argument("--target_image", default="",
                    help="有目標模式的目標影像；基準的兩個 PhotoGuard 變體用灰圖")
    # site S 專用。預設值改為 bicubic：OptimConfig 的預設 bilinear 只為了讓
    # E13–E19 可重現，E20 §5.2 量出 bicubic 把銳利度保留率由 85.0% 拉到
    # 99.9%，e21 的實測也顯示同一 τ 下 bicubic 的編輯 LPIPS 0.2473 對
    # bilinear 的 0.0931。拿 bilinear 跑等於自願讓非加性位置變弱。
    ap.add_argument("--warp_resample", default="bicubic",
                    choices=["bilinear", "bicubic"])
    ap.add_argument("--warp_max_disp", type=float, default=1.5,
                    help="site S 的位移硬上界，單位像素。與 tau_lpips 併列"
                         "記錄：本位置的失真預算是位移量而非 L∞")
    # ---- 必須與 run_lo_baseline 相同，否則量到的是另一個攻擊 ----
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--eval_seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.defense_mode == "targeted" and not args.target_image:
        raise SystemExit(
            "--defense_mode targeted 需要 --target_image。"
            "有目標模式的損失是 d(編輯結果, 目標)，沒有目標就沒有目標函數"
        )

    root = Path(__file__).resolve().parents[1]
    lr_map = parse_lr(args.lr) if args.lr else dict(SITE_LR)
    if args.tau_acut is None or args.tau_chroma is None:
        t_acut, t_chroma = budget_thresholds(args.tau_lpips, root)
        if args.tau_acut is None:
            args.tau_acut = t_acut
        if args.tau_chroma is None:
            args.tau_chroma = t_chroma
    # 逐 site 的 lr 與逐預算的兩道門檻都寫進 protocol.json，不只印出來：
    # 這三個值決定了每一格實際被什麼綁住，事後無從由 summary.csv 反推。
    args.lr_map = lr_map

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)

    data = load_dataset(Path(args.data), args.size, device, args.limit)
    if not data:
        raise SystemExit(f"{args.data} 底下沒有任何 PNG")

    y_target = None
    if args.target_image:
        from PIL import Image
        import torchvision.transforms as T

        img = Image.open(args.target_image).convert("RGB").resize(
            (args.size, args.size), Image.LANCZOS)
        y_target = T.ToTensor()(img).unsqueeze(0).to(device)

    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    unknown = [s for s in sites if s not in lr_map]
    if unknown:
        raise SystemExit(
            f"這些 site 沒有校準過的學習率：{unknown}。"
            f"已校準的是 {sorted(lr_map)}。共用別的 site 的 lr 不是保守作法"
            "——φ 的量綱不同，量到的會是「這個步長走到哪裡」而非該位置的能力"
        )
    purifiers = default_train_set()
    if args.no_purify_train and args.defense_mode == "untargeted":
        # 這個組合會讓最佳化永遠停在 φ = 0，而且沒有任何症狀：損失是
        # max(0, margin − 0) = margin，看起來「有損失可以下降」，但 LPIPS
        # 在 y_def = y_orig 處是最小值，梯度精確為零（LEDGER 5.10）。
        # `untargeted` 至今能跑起來，靠的正是輪替到的 blur 打破這個對稱
        # （5.11）。此處直接拒絕，不靜默接受。
        raise SystemExit(
            "--no_purify_train 不可與 --defense_mode untargeted 併用。\n"
            "  原因：untargeted 的防禦項在兩條分支逐元素相同時梯度精確為零"
            "（LEDGER 5.10），而 φ=0 且淨化為 identity 時正是這個情形。\n"
            "  實測：runs/ours_lo/man_00__PF__history.json 的 step 0 是"
            " grad_norm = 0.000e+00，step 1 換成 blur 之後才是 4.457e-01。\n"
            "  要讓兩個條件在同一個問題上，請改用起點梯度非零的目標函數"
            "（--defense_mode targeted，或 crossattn 的 suppress），見 5.13"
        )
    if args.no_purify_train:
        # identity 一定在訓練集裡（strength = 0.0 那個）。取它而不是自己造一個
        # 新的物件，是為了讓「關掉淨化」與「淨化集只剩 identity」是同一件事。
        purifiers = [p for p in purifiers if p.strength == 0.0]
        if len(purifiers) != 1:
            raise SystemExit(
                f"default_train_set() 的 identity 算子不是恰好一個"
                f"（找到 {len(purifiers)} 個），--no_purify_train 無從定義"
            )
    print(f"[訓練期淨化] {len(purifiers)} 個算子："
          f"{[getattr(p, 'kind', type(p).__name__) for p in purifiers]}",
          flush=True)
    res_path, sum_path = out / "results.csv", out / "summary.csv"
    done = completed_pairs(sum_path)
    if done and not args.resume:
        raise SystemExit(
            f"{sum_path} 已有 {len(done)} 格結果。要接續請加 --resume；"
            "要重跑請換一個 --out。此處不自動覆寫"
        )
    save_json(vars(args), out / "protocol.json")
    if done:
        print(f"接續模式：已完成 {len(done)} 格，將略過", flush=True)

    for name, x01, prompts, content in data:
        prompt = prompts[args.prompt_index]
        todo = [s for s in sites if (name, s) not in done]
        if not todo:
            print(f"=== {name}：全部 site 已完成，略過 ===", flush=True)
            continue

        # 評測用的設定物件。與 run_lo_baseline 的欄位逐一對應——這是兩邊
        # 數字可比的唯一依據，改動任何一項都會讓比較失效。
        eval_cfg = LinfAttackConfig(
            strength=args.strength, guidance_scale=args.guidance,
            n_edit=args.n_edit, prompt_edit=prompt, seed=args.seed,
        )
        t_ref = time.perf_counter()
        refs = reference_edits(sd, x01, prompt, eval_cfg, args.eval_seeds)
        print(f"\n=== {name}：{len(refs)} 個種子的未防禦編輯，"
              f"{time.perf_counter()-t_ref:.0f}s ===", flush=True)

        for site in todo:
            cfg = OptimConfig(
                steps=args.steps, lr=lr_map[site], n_edit=args.n_edit,
                strength=args.strength, guidance_scale=args.guidance,
                prompt_edit=prompt, seed=args.seed,
                stop_on_plateau=True,
                stop_require_feasible=True,
                warp_resample=args.warp_resample,
                warp_max_disp=args.warp_max_disp,
            )
            loss_cfg = LossConfig(
                margin=args.margin, defense_mode=args.defense_mode,
                tau_lpips=args.tau_lpips,
                tau_acut=args.tau_acut, tau_chroma=args.tau_chroma,
                beta_linf=args.beta_linf, alpha_lpips=args.alpha_lpips,
            )
            # 綁定的必須是 LPIPS。若 L∞ 的係數非零，量到的是「在 L∞ 預算下
            # 能做什麼」，那是基準的約束不是本專案的，兩邊的條件會被不同的東西
            # 綁住，比較不成立。三道 hinge 的門檻與該 site 的 lr 一起印出來：
            # 事後只看 summary.csv 反推不出來哪一道實際綁住了這一格。
            print(f"  [約束] gamma_lpips={loss_cfg.gamma_lpips} "
                  f"tau_lpips={loss_cfg.tau_lpips} "
                  f"tau_acut={loss_cfg.tau_acut:.4f} "
                  f"tau_chroma={loss_cfg.tau_chroma:.4f} "
                  f"alpha_lpips={loss_cfg.alpha_lpips} "
                  f"beta_linf={loss_cfg.beta_linf} lr={cfg.lr}", flush=True)
            print(f"\n=== {name} / site {site} / τ_lpips={args.tau_lpips} / "
                  f"{prompt!r} ===", flush=True)
            reset_peak_memory()
            module = build_module(
                site, args.rank, cfg, sd, args.size, args.seed).to(device)
            try:
                res = optimize(sd, module, x01, cfg, loss_cfg, purifiers,
                               y_target=y_target)
            finally:
                # site W 把 hook 掛在共用的 UNet 上，不卸除會滲進下一格，
                # 症狀是「另一個 site 的結果莫名被改動」。
                if hasattr(module, "remove"):
                    module.remove()

            x_def = res.x_def.detach().clamp(0, 1)
            save_image(x_def, out / f"{name}__{site}__def.png")
            save_json(res.history, out / f"{name}__{site}__history.json")

            pert = suite.pairwise(x01, x_def)
            ev = evaluate(sd, suite, refs, x_def, prompt, eval_cfg,
                          out, f"{name}__{site}")
            rows = [{
                "image": name, "attack": site, "content": content,
                "prompt": prompt, "tau_lpips": args.tau_lpips,
                # 三道門檻與該 site 的 lr 逐格記錄：它們決定了這一格實際被
                # 什麼綁住，而 summary.csv 的指標欄反推不出來
                "tau_acut": args.tau_acut, "tau_chroma": args.tau_chroma,
                "lr": cfg.lr,
                "steps": args.steps, "steps_done": res.steps_done,
                "stop_reason": res.stop_reason,
                "defense_mode": args.defense_mode, "rank": args.rank,
                # site S 的失真預算是位移量，不是 L∞ 或 τ_lpips，不記就
                # 無從得知該格實際被綁在哪裡
                "warp_resample": args.warp_resample,
                "warp_max_disp": args.warp_max_disp,
                "strength": args.strength, "guidance_scale": args.guidance,
                "attack_seconds": round(res.seconds, 1),
                "peak_mb": round(peak_memory_mb(), 1),
                **{f"pert_{k}": v for k, v in pert.items()},
                **r,
            } for r in ev]
            n = len(ev)
            srow = {
                "image": name, "attack": site, "n_seeds": n,
                **{f"edit_{k}": sum(r[f"edit_{k}"] for r in ev) / n
                   for k in TABLE1},
                "pert_linf": pert["linf"], "pert_lpips": pert["lpips"],
                "steps_done": res.steps_done,
                "seconds": round(res.seconds, 1),
                "peak_mb": round(peak_memory_mb(), 1),
            }
            append_csv(res_path, rows)
            append_csv(sum_path, [srow])
            print(f"  [{site}] 擾動 LPIPS {pert['lpips']:.4f} / L∞ "
                  f"{pert['linf']:.4f}  →  編輯 LPIPS "
                  f"{srow['edit_lpips']:.4f}  PSNR {srow['edit_psnr']:.2f}"
                  f"  （{res.steps_done} 步，{res.seconds:.0f}s）", flush=True)

    print(f"\n完成。{sum_path}", flush=True)


if __name__ == "__main__":
    main()
