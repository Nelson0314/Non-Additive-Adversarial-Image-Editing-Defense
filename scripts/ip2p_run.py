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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torchvision.utils as vutils  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from phase_ablation import build  # noqa: E402
from src.baselines.advdrop import (
    PAPER_Q_INIT, AdvDropSpec, run_advdrop,
)
from src.baselines.dct_watermark import (  # noqa: E402
    WatermarkSpec, run_dct_watermark,
)
from src.baselines.dct_shield import (  # noqa: E402
    PAPER_DEFAULT_QUALITY, PAPER_EPS, PAPER_GAMMA, PAPER_STEPS,
    DCTShieldSpec, run_dct_shield,
)
from src.baselines.encoder_target import make_encoder_target_loss  # noqa: E402
from src.defense.purify_aware import (  # noqa: E402
    make_eot_jpeg_transform, make_eot_ops_transform, make_fixed_jpeg_transform,
    make_jpeg_transform,
)
from src.defense.param_pgd import fit_to_budget, run_param_pgd  # noqa: E402
from src.metrics.standard import standard_row  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.ip2p import (  # noqa: E402
    IP2P_IMAGE_GUIDANCE, IP2P_SEED, IP2P_STEPS, IP2P_TEXT_GUIDANCE, IP2PWrapper,
)
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
PHASE_CONDS = ("phase", "phase_rand", "add", "phase_gain", "gain_only")
DCT_CONDS = ("dct_shield", "dct_shield_y")
ADVDROP_CONDS = ("advdrop",)
WM_CONDS = ("dct_wm",)


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
        return make_eot_jpeg_transform((95, 75, 50), seed=args.seed)
    return make_eot_ops_transform((75,), seed=args.seed)


def defend(ip2p, suite, cond, x01, args, loss_fn):
    """回傳 `(x_def, radius, unreachable, modified_from_paper)`。

    兩條路徑：相位／加性走本專案共用的 sign PGD；DCT-Shield 走它自己論文的
    Algorithm 1。**兩者都不因為換了攻擊方而改動**——改的只有 `loss_fn` 裡的
    那個 `E`。
    """
    if cond in WM_CONDS:
        # **摘要重建，不是重現**（付費牆、無公開程式碼）。三個機制照摘要做，
        # 全部超參數為本專案指定，見 `src/baselines/dct_watermark.py`。
        spec = WatermarkSpec(
            name=cond, q_embed=args.q_alg, q_attack=args.wm_q_attack,
            block_frac=args.wm_block_frac, topk=args.wm_topk,
            eps=args.wm_eps, steps=args.wm_steps)
        res = run_dct_watermark(ip2p, x01, spec, loss_fn=loss_fn, log_every=0)
        return res.x_def, spec.eps, False, True

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
        return res.x_def, spec.eps, False, True

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
            # **不傳 loss_fn**：`run_dct_shield` 預設用的是該篇自己的損失
            # `‖E(x')‖₂`（§4.2 末段）。傳我們的 encoder-target 進去等於把別人
            # 的方法換掉一半，那是消融不是 baseline。兩者都只經過 VAE 編碼器，
            # 換掉不會拋錯也看不出來，故在此明寫。
            res = run_dct_shield(ip2p, x01, spec, log_every=250)
            return res.x_def, spec.eps, False, spec.modified_from_paper
        raise SystemExit(
            "DCT-Shield 的預算對齊模式尚未接到 IP2P 線上。曲線協定（DEC-029）"
            "是掃 eps 畫取捨曲線，錨點由 tradeoff_curve.py 內插求得，"
            "不需要在這裡二分搜尋")

    param, lo, hi = build(cond, args.seed, block=args.block, r_min=args.r_min,
                          r_max=args.r_max,
                          quantile=args.quantile, gl_iters=args.gl_iters,
                          pixel_gate_sigma=args.pixel_gate_sigma,
                          gain_ratio=args.gain_ratio)
    if args.radius is not None:
        param.set_radius(args.radius)
        res = run_param_pgd(x01, param, loss_fn, steps=args.steps,
                            seed=args.seed,
                            transform=_purify_transform(args))
        return res.x_def, param.radius, False, False

    def dists_of(a, b):
        return float(suite.pairwise(b, a)["dists"])

    out = fit_to_budget(x01, param, loss_fn, dists_of, args.budget,
                        lo=lo, hi=hi, steps=args.steps, seed=args.seed,
                        rounds=args.rounds)
    return (out.x_def, out.radius,
            bool(out.history[-1].get("unreachable", False)), False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--conditions", nargs="+", default=["phase", "dct_shield"])
    ap.add_argument("--images", nargs="+", default=None, help="分片用")
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="OmniEdit 有 90 張帶兩條指令；預設一律取第 0 條")
    ap.add_argument("--target", type=Path, default=Path("data/targets/gray.png"))
    ap.add_argument("--loss", choices=("encoder_target", "latent_norm"),
                    default="encoder_target",
                    help="encoder_target = ‖E(x_def) − E(y_target)‖²（本專案既有）；"
                         "latent_norm = ‖E(x_def)‖₂（DCT-Shield §4.2 的目標）。"
                         "後者只壓長度、單調且無方向要求，前者要同時對長度與"
                         "方向。2026-08-21 加這個旗標是為了把「輸給 DCT-Shield "
                         "的部分是損失函數還是參數化」分開量")
    # 相位／加性
    ap.add_argument("--radius", type=float, default=None,
                    help="直接指定半徑（掃描曲線用）。不給則二分搜到 --budget")
    ap.add_argument("--budget", type=float, default=0.0349, help="DISTS 預算")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--r-min", type=float, default=0.12)
    ap.add_argument("--r-max", type=float, default=float("inf"),
                    help="徑向頻率閘的上界。預設無窮大即維持原本的高通行為。"
                         "穩健浮水印的標準作法是中頻帶嵌入——低頻可見、高頻"
                         "被壓縮與模糊抹掉；本模組原本只有下界，上界從未測過")
    ap.add_argument("--quantile", type=float, default=0.5)
    ap.add_argument("--pixel-gate-sigma", type=float, default=0.0,
                    help="逐像素紋理閘的高斯 sigma（像素）。0 = 關閉，逐位元與"
                         "加這個選項之前相同。要能分辨鬍鬚與臉頰就必須遠小於 "
                         "block=32；本值無出處，是本專案指定")
    ap.add_argument("--purify-aware",
                    choices=("none", "curriculum", "fixed75", "eot_jpeg", "eot_ops"),
                    default="none",
                    help="把可微分的淨化算子放進最佳化迴圈（改動三）。"
                         "curriculum = JPEG 品質 95→50 線性；fixed75 = 固定 75；"
                         "eot_jpeg = 每步抽一個品質；eot_ops = 每步抽一個算子"
                         "（identity／模糊／裁切縮放／JPEG75）。"
                         "**回傳的防禦圖仍是未經算子處理的那一張**")
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
    ap.add_argument("--dct-steps", type=int, default=PAPER_STEPS)
    # AdvDrop
    # DCT 對抗浮水印（摘要重建）
    ap.add_argument("--wm-q-attack", type=int, default=75,
                    help="迴圈內模擬的重壓品質。摘要未載，本專案指定")
    ap.add_argument("--wm-block-frac", type=float, default=0.5,
                    help="依顯著度保留多少比例的區塊。摘要未載")
    ap.add_argument("--wm-topk", type=int, default=8,
                    help="每個區塊最多改幾個係數（8×8 共 64）。摘要未載")
    ap.add_argument("--wm-eps", type=float, default=2.0,
                    help="delta 的逐元素上界，單位是量化階。摘要未載")
    ap.add_argument("--wm-steps", type=int, default=300)
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
    ap.add_argument("--edit-steps", type=int, default=IP2P_STEPS)
    ap.add_argument("--text-guidance", type=float, default=IP2P_TEXT_GUIDANCE)
    ap.add_argument("--image-guidance", type=float, default=IP2P_IMAGE_GUIDANCE)
    ap.add_argument("--edit-seed", type=int, default=IP2P_SEED)
    ap.add_argument("--check-only", action="store_true",
                    help="只跑未防禦的編輯並報語意對齊，驗收 DEC-022 的前提")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ip2p = IP2PWrapper(dtype=torch.float32)
    suite = MetricSuite(device=ip2p.device)
    dataset = load_dataset(args.data, prompt_index=args.prompt_index)
    if args.images:
        keep = set(args.images)
        dataset = [d for d in dataset if d["name"] in keep]
    if not dataset:
        raise SystemExit(f"{args.data} 底下沒有符合 --images 的影像")
    print(f"IP2P 線：{len(dataset)} 張、條件 {args.conditions}、"
          f"steps={args.edit_steps} s_T={args.text_guidance} "
          f"s_I={args.image_guidance} seed={args.edit_seed}", flush=True)

    y_target = load_image_tensor(args.target, ip2p.device, size=RESOLUTION)
    if args.loss == "latent_norm":
        # DCT-Shield §4.2 的目標。與 `make_latent_norm_loss` 同式，這裡直接寫
        # 出來避免把 baseline 模組的預設綁進相位臂。
        def loss_fn(x01_):
            return ip2p.encode_image(x01_).flatten().norm(p=2)
    else:
        loss_fn = make_encoder_target_loss(ip2p, y_target)

    def edit(x01, item):
        return ip2p.edit(x01, item["prompt"], seed=args.edit_seed,
                         steps=args.edit_steps, s_t=args.text_guidance,
                         s_i=args.image_guidance)

    rows = []
    for item in dataset:
        x01 = load_image_tensor(item["path"], ip2p.device, size=RESOLUTION)
        item["path01"] = x01
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

        for cond in args.conditions:
            t1 = time.time()
            x_def, radius, unreachable, modified = defend(
                ip2p, suite, cond, x01, args, loss_fn)
            fid = suite.pairwise(x01, x_def)
            e_def = edit(x_def, item)
            prot = suite.pairwise(e_orig, e_def)
            for sub, img in (("def", x_def), ("edit_orig", e_orig),
                             ("edit_def", e_def)):
                vutils.save_image(img.clamp(0, 1),
                                  args.out / f"{item['name']}__{cond}__{sub}.png")
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
                "loss": args.loss,
                "gain_ratio": args.gain_ratio,
                "purify_aware": args.purify_aware,
                "wm_q_attack": args.wm_q_attack,
                "wm_block_frac": args.wm_block_frac,
                "wm_topk": args.wm_topk,
                "wm_eps": args.wm_eps,
                # 論文未載、本專案指定的三個推論參數逐列記下（DEC-031）
                "edit_steps": args.edit_steps, "s_t": args.text_guidance,
                "s_i": args.image_guidance, "edit_seed": args.edit_seed,
                "modified_from_paper": modified,
                **standard_row("fid_", fid),
                **standard_row("edit_", prot),
                "fid_linf": round(fid["linf"], 5),
                "fid_rms": round(fid["rms"], 5),
                "edit_lpips": round(float(prot["lpips"]), 5),
                "total_seconds": round(time.time() - t1, 1),
            })
            write_csv(args.out / "results.csv", rows)
            print(f"{item['name']:32s} {cond:14s} r={radius:.4f} "
                  f"dists={fid['dists']:.4f} lpips={fid['lpips']:.4f} "
                  f"effect={prot['lpips']:.4f} ({time.time() - t1:.0f}s)",
                  flush=True)

    out = args.out / ("check.csv" if args.check_only else "results.csv")
    print(f"\n表：{out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
