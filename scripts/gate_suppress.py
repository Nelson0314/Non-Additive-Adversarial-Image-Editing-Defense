"""閘門：把失真預算放寬到文獻標準，cross-attention 目標在語意軸上會不會動？

    python scripts/gate_suppress.py --images horse_00,horse_01 --tau_lpips 0.55

## 這一格在回答什麼

`docs/CONVERGENCE.md` §3 的閘門。L3 量到：文獻最強的基準在 **5.4 倍**於本專案
的失真預算上，語意失敗也只有 7/72 格（LEDGER 1.15）。本專案在 1/5.4 的失真上
預期是零。直接跑六格的匹配失真比較會量到兩個零，那是重演 E29
（LEDGER §8 已把「在 τ ≤ 0.10 下用 untargeted 再跑網格」列為死路）。

所以先問一個**單變因**的問題：

> 把預算一路放寬到與 κ = 0.06 相同的感知失真（LPIPS ≈ 0.55，見 2.18），
> `suppress` + site PF 在語意軸上會不會動？

- 會動 → 匹配失真的六格比較有意義。
- 不會動 → 第二層在這個威脅模型下不可答，該結論本身就是產出。

## 為什麼是這個設定

**只有一道約束。** `gamma_acut` 與 `gamma_chroma` 一律設 0。那兩道的職責是
「讓兩個 site 在同一種可辨失真上比較」，而本閘門只有一個 site，沒有要比較的
對象。留著它們等於多兩個可能綁住這一格的東西（LEDGER 6.2 已出現過八次）。

**目標函數是被證據釘死的，不是選的。** `untargeted` 在起點的梯度精確為零
（5.10）；`targeted` 起點梯度非零但在 L1 的語意軸上只有 1/24；`suppress`
兩項都通過，且是基準論文自己那一類（5.13、5.7）。

**影像刻意挑最有利的。** `horse_00`／`horse_01` 是 L1 中 `semantic` 的語意
失敗最強的兩張（Δsiglip −0.0722 與 −0.0678），且 `a zebra` 是沿用論文原文的
prompt。閘門是 go／no-go，挑最有利的случай是對的：**在最容易的影像上都不動，
就不必再花雲端時間**。反過來若它動了，也不能宣稱一般性——那要靠後續的網格。

**256² 不能取代 512² 的正式結果。** VAE 下採樣倍率固定為 8，256² 的 latent
是 32²，cross-attention 的空間解析度全部減半，綁定結構不同。閘門問的是
「會不會動」不是「動多少」。

**`suppress` 的對象是攻擊方的 prompt，不是 c_a**（LEDGER 5.7(d)）。這是比
基準論文更強的白盒假設，故本閘門量到的是該線的**上界**。

## 判準

語意軸：`Δsiglip = SigLIP(y_def, prompt) − SigLIP(y_ref, prompt)`，負值代表
免疫後的編輯較不服從 prompt。依 E25 的規則，判定要求 `|mean| > sd`，故評測
種子數必須 ≥ 2（LEDGER 1.4）。Table 1 五指標一併報，供與 L1 對讀。
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.defense.objective import LossConfig  # noqa: E402
from src.defense.optimize import OptimConfig, optimize_crossattn  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.purify.ops import Purifier  # noqa: E402
from src.residual.site_pixel_full import FullRankPixelResidual  # noqa: E402
from src.residual.site_warp import WarpResidual  # noqa: E402
from src.utils.artifacts import save_image, save_json  # noqa: E402
from src.utils.device import (  # noqa: E402
    get_device, peak_memory_mb, reset_peak_memory,
)
from src.metrics.ray_scale import (  # noqa: E402
    gaussian_control, lpips_against,
)
from scripts.run_ours_lo_eval import SITE_LR, parse_lr  # noqa: E402

EVAL_SEED_OFFSET = 10_000
TABLE1 = ("psnr", "ssim", "vif_p", "fsim", "lpips")

# site S 的位移硬上界。**閘門用的值與正式實驗不同，這是刻意的。**
#
# 正式實驗用 1.5 px（E20／E22 校準）。但閘門要在 τ_lpips = 0.55 上比較，
# 而 1.5 px 的平滑位移場達不到那個感知失真——L4 在 τ = 0.10 上只做到 0.025
# （LEDGER 3.23）。若沿用 1.5，綁住這一格的會是位移上界而不是 LPIPS，
# 那就不是匹配失真的比較，而是第九次「量到別的東西」（CONVERGENCE §1）。
#
# 故閘門把上界放寬，並**逐格記錄位移統計**，讓「到底是誰綁住這一格」
# 直接看得到。若放寬後仍達不到 τ，那本身就是關於該參數化的結果。
GATE_WARP_MAX_DISP = 8.0


def build(site, size, cfg, seed, grid_size=32):
    """`grid_size` 是 site S 的**容量**參數：位移場控制點的邊長。

    掃過全庫 `env.json`：24 次跑過 site S 的實驗**全部**用 32
    （e13 4 次、e14 4、e15 3、e16 1、e21 6、e22 3、e23 1、L4、本閘門），
    即「非加性比較弱」這個判斷完全建立在容量軸的單一點上（LEDGER 3.29）。
    而 3.28 已把 site S 的上限歸因到參數化的擾動能力本身，故這個參數是
    目前對研究問題最直接的變因。

    `None` 表示逐像素自由位移（stAdv 的原始設定）。該設定下平滑度不再由
    粗網格保證，`WarpResidual.tv()` 只是診斷量而非損失項，會出現撕裂狀的
    可見瑕疵——要用它必須先把 TV 加進損失。
    """
    if site == "PF":
        return FullRankPixelResidual(size=size, channels=3, seed=seed)
    if site == "S":
        return WarpResidual(size=size, grid_size=grid_size,
                            max_disp=cfg.warp_max_disp,
                            seed=seed, resample=cfg.warp_resample)
    raise ValueError(f"閘門只支援 PF 與 S，收到 {site!r}")


def load_image(data: Path, name: str, size: int, device):
    """回傳 (張量, 編輯 prompt, c_a)。名稱形如 `horse_00`。"""
    import yaml
    from PIL import Image
    import torchvision.transforms as T

    spec = yaml.safe_load((data / "prompts.yaml").read_text(encoding="utf-8"))
    cls = name.rsplit("_", 1)[0]
    if cls not in spec:
        raise SystemExit(f"{name} 的類別 {cls!r} 不在 {data/'prompts.yaml'} 裡")
    p = data / cls / f"{name}.png"
    if not p.exists():
        raise SystemExit(f"{p} 不存在")
    img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
    x = T.ToTensor()(img).unsqueeze(0).to(device)
    return x, spec[cls]["prompts"][0], spec[cls]["content"]


def append_csv(path: Path, rows: list):
    """附加到 CSV，並在欄位與既有表頭不符時**拋出**。

    `csv.DictWriter` 的 `fieldnames` 若取自新資料，而檔案已有一個欄位不同的
    表頭，附加出來的列會**按新順序寫進舊表頭底下**——沒有例外、沒有警告，
    只有錯位的資料。本專案的 `runs/` 是唯一的證據來源且實驗無法重跑，
    一次靜默錯位就毀掉整批。故此處明確比對。
    """
    if not rows:
        return
    names = list(rows[0].keys())
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != names:
            raise SystemExit(
                f"{path} 既有的表頭與要寫入的欄位不符，拒絕附加。\n"
                f"  既有：{header}\n  要寫：{names}\n"
                "  附加下去會讓資料按新順序寫進舊表頭底下而沒有任何症狀。"
                "請改用一個新的 --out 目錄")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=names)
        if not path.stat().st_size:
            w.writeheader()
        w.writerows(rows)


def random_control(suite, x01, target_lpips, seed, tol=0.005, iters=24):
    """同一個失真量、但沒有經過最佳化的隨機擾動。回傳 (x_ctrl, 實際 LPIPS)。

    薄包裝，實作與理由見 `src/metrics/ray_scale.py`。保留這個名字是因為
    `runs/gate_suppress/` 與 LEDGER 1.18 都以它為出處。
    """
    x, got, _ = gaussian_control(
        lpips_against(suite, x01), x01, target_lpips, seed,
        tol=tol, iters=iters)
    return x, got


@torch.no_grad()
def evaluate(sd, suite, x01, x_def, prompt, cfg, n_seeds):
    """逐種子比較未防禦編輯與防禦後編輯。兩條分支共用逐元素相同的 ε。"""
    emb = sd.encode_text(prompt).detach()
    emb_u = sd.encode_text("").detach()
    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    rows = []
    for i in range(n_seeds):
        seed = cfg.seed + EVAL_SEED_OFFSET + i
        nz = sd.sample_edit_noise(torch.empty(lat, device=x01.device), seed=seed)
        kw = dict(strength=cfg.strength, guidance_scale=cfg.guidance_scale,
                  emb_uncond=emb_u)
        y_ref = sd.sdedit(x01, emb, nz, cfg.n_edit, **kw)
        y_def = sd.sdedit(x_def, emb, nz, cfg.n_edit, **kw)
        m = suite.full(y_ref, y_def, prompt=prompt)
        rows.append({"eval_seed": seed,
                     **{f"edit_{k}": v for k, v in m.items()}})
    return rows


def main():
    ap = argparse.ArgumentParser(description="CONVERGENCE §3 的閘門")
    ap.add_argument("--data", default="data/lo_aligned")
    ap.add_argument("--out", default="runs/gate_suppress")
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--images", default="horse_00,horse_01")
    ap.add_argument("--sites", default="PF")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument(
        "--lr", default="",
        help="留空表示每個 site 各取自己校準過的值（`run_ours_lo_eval.SITE_LR`："
             "PF 0.008、S 0.1，相差 12.5 倍）。**不要傳單一數值**：φ 的量綱"
             "逐 site 不同，同一個 lr 代表不同的實際步長，那正是 "
             "runs/ours_lo/ 失效的原因（LEDGER 6.16）")
    ap.add_argument(
        "--warp_max_disp", type=float, default=GATE_WARP_MAX_DISP,
        help=f"site S 的位移硬上界，單位像素。閘門預設 {GATE_WARP_MAX_DISP}"
             "（正式實驗是 1.5）：1.5 px 的平滑位移場達不到 τ_lpips = 0.55，"
             "沿用會讓上界而非 LPIPS 成為有效約束。逐格記錄位移統計以供核對")
    ap.add_argument(
        "--grid_size", default="32",
        help="site S 位移場的控制點邊長，即該位置的**容量**。`none` 表示"
             "逐像素自由位移（stAdv 的原始設定，需自行加 TV 懲罰）。"
             "**24 次 site S 實驗全部用 32，這個軸從未被掃過**（LEDGER 3.29），"
             "而 3.28 已把 site S 的上限歸因到參數化的擾動能力本身")
    ap.add_argument(
        "--warp_resample", default="bicubic", choices=["bilinear", "bicubic"],
        help="site S 的 grid_sample 插值模式。E20 §5.2 量出 bicubic 把銳利度"
             "保留率由 85.0% 拉到 99.9%。**寫進 protocol.json**：事後要沿"
             "射線縮放位移場時必須知道當初用的是哪一種，猜錯會得到另一個解")
    ap.add_argument("--tau_lpips", type=float, default=0.55,
                    help="預設 0.55 對應 κ = 0.06 的感知失真（LEDGER 2.18）")
    ap.add_argument("--attn_timesteps", type=int, default=4)
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--eval_seeds", type=int, default=5,
                    help="判定要求 |mean| > sd，故必須 ≥ 2（LEDGER 1.4）")
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    lr_map = parse_lr(args.lr) if args.lr else dict(SITE_LR)

    if args.eval_seeds < 2:
        raise SystemExit(
            "--eval_seeds 必須 ≥ 2：n = 1 時 sd 恆為 0，"
            "任何「|mean| > sd」的判定自動成立（LEDGER 1.4）")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out / "protocol.json")
    device = get_device()
    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)
    res_path, sum_path = out / "results.csv", out / "summary.csv"

    for name in [s.strip() for s in args.images.split(",") if s.strip()]:
        x01, prompt, content = load_image(ROOT / args.data, name,
                                          args.size, device)
        for site in [s.strip() for s in args.sites.split(",") if s.strip()]:
            cfg = OptimConfig(
                steps=args.steps, lr=lr_map[site], n_edit=args.n_edit,
                strength=args.strength, guidance_scale=args.guidance,
                prompt_edit=prompt, seed=args.seed,
                attn_mode="suppress", attn_timesteps=args.attn_timesteps,
                attn_content_only=True,
                warp_resample=args.warp_resample,
                warp_max_disp=args.warp_max_disp,
                stop_on_plateau=False,   # 閘門要完整軌跡，不提早停
                unet_ckpt=True, vae_ckpt=True, log_every=10,
            )
            # 只有一道約束。兩道副 hinge 的職責是「讓兩個 site 在同一種可辨
            # 失真上比較」，本閘門只有一個 site，留著它們等於多兩個可能綁住
            # 這一格的東西（LEDGER 6.2 已出現過八次）。
            loss_cfg = LossConfig(
                margin=1.0, defense_mode="untargeted",   # crossattn 不讀這一項
                tau_lpips=args.tau_lpips, gamma_lpips=100.0,
                alpha_lpips=0.0, beta_linf=0.0,
                gamma_acut=0.0, gamma_chroma=0.0, gamma_psnr=0.0,
            )
            print(f"\n=== {name} / site {site} / τ_lpips={args.tau_lpips} / "
                  f"{prompt!r} / c_a={content!r} ===", flush=True)
            print(f"  [約束] 只有 LPIPS：gamma_lpips={loss_cfg.gamma_lpips} "
                  f"tau={loss_cfg.tau_lpips}；acut/chroma/linf/psnr 全為 0；"
                  f"lr={cfg.lr}"
                  + (f"　max_disp={cfg.warp_max_disp}" if site == "S" else ""),
                  flush=True)
            reset_peak_memory()
            gs = None if args.grid_size.lower() == "none" else int(args.grid_size)
            module = build(site, args.size, cfg, args.seed,
                           grid_size=gs).to(device)
            t0 = time.perf_counter()
            res = optimize_crossattn(sd, module, x01, cfg, loss_cfg,
                                     [Purifier("identity")])
            train_s = time.perf_counter() - t0

            # site S 的失真預算是位移量。逐格印出位移統計，否則「綁住這一格
            # 的是 LPIPS 還是位移上界」事後無從判斷（LEDGER 6.2 的第九次）。
            disp = module.disp_stats() if hasattr(module, "disp_stats") else {}
            if disp:
                print(f"  [位移] 平均 {disp['disp_mean_px']:.3f} px　"
                      f"最大 {disp['disp_max_px']:.3f} px　"
                      f"p99 {disp['disp_p99_px']:.3f} px　"
                      f"上界 {cfg.warp_max_disp}", flush=True)

            x_def = res.x_def.detach().clamp(0, 1)
            save_image(x_def, out / f"{name}__{site}__def.png")
            save_json(res.history, out / f"{name}__{site}__history.json")
            # φ 本身也存下來。只存 PNG 的話，事後要把解沿射線縮放就只能在
            # 影像空間做（`x + k·(x_def − x)`），而那對 site S 是**交叉淡入**
            # 不是「更大的位移」——兩者只在一階近似下相同。存了 state_dict
            # 之後就能直接縮放位移場本身。
            torch.save(module.state_dict(), out / f"{name}__{site}__phi.pt")
            pert = suite.pairwise(x01, x_def)

            save_image(x01, out / f"{name}__orig.png")

            # 兩個位置走**完全相同**的評測：最佳化解，以及同一個 LPIPS 上的
            # 隨機擾動。後者是 e2_phi0 那一課的直接落實（LEDGER 7.1）。
            arms = [("opt", x_def, pert["lpips"])]
            x_ctrl, ctrl_lpips = random_control(
                suite, x01, pert["lpips"], seed=args.seed)
            save_image(x_ctrl, out / f"{name}__{site}__rand.png")
            arms.append(("rand", x_ctrl, ctrl_lpips))
            print(f"  [對照] 隨機擾動對到 LPIPS {ctrl_lpips:.4f}"
                  f"（目標 {pert['lpips']:.4f}）", flush=True)

            for arm, xa, arm_lpips in arms:
                t1 = time.perf_counter()
                ev = evaluate(sd, suite, x01, xa, prompt, cfg, args.eval_seeds)
                n = len(ev)
                mean = lambda k: sum(r[k] for r in ev) / n      # noqa: E731
                dsig = [r["edit_siglip_b"] - r["edit_siglip_a"] for r in ev]
                dniq = [r["edit_niqe_b"] - r["edit_niqe_a"] for r in ev]
                m_s = sum(dsig) / n
                sd_s = (sum((v - m_s) ** 2 for v in dsig) / n) ** 0.5
                m_n = sum(dniq) / n
                sd_n = (sum((v - m_n) ** 2 for v in dniq) / n) ** 0.5
                pa = suite.pairwise(x01, xa)
                srow = {
                    "image": name, "site": site, "arm": arm,
                    "prompt": prompt, "content": content,
                    "tau_lpips": args.tau_lpips,
                    "steps_done": res.steps_done if arm == "opt" else 0,
                    "lr": cfg.lr, "warp_max_disp": cfg.warp_max_disp,
                    "disp_mean_px": disp.get("disp_mean_px", ""),
                    "disp_max_px": disp.get("disp_max_px", ""),
                    "n_seeds": n,
                    "pert_lpips": pa["lpips"], "pert_linf": pa["linf"],
                    "pert_psnr": pa["psnr"],
                    "attn_div_first": res.history[0]["attn_div"],
                    "attn_div_last": res.history[-1]["attn_div"],
                    "grad_norm_first": res.history[0]["grad_norm"],
                    "dsiglip_mean": m_s, "dsiglip_sd": sd_s,
                    "dniqe_mean": m_n, "dniqe_sd": sd_n,
                    # E25 的規則：平均為負且絕對值大於標準差才算語意失敗
                    "semantic_fail": bool(m_s < 0 and abs(m_s) > sd_s),
                    **{f"edit_{k}": mean(f"edit_{k}") for k in TABLE1},
                    "train_seconds": round(train_s, 1) if arm == "opt" else 0.0,
                    "eval_seconds": round(time.perf_counter() - t1, 1),
                    "peak_mb": round(peak_memory_mb(), 1),
                }
                append_csv(res_path, [{"image": name, "site": site,
                                       "arm": arm, **r} for r in ev])
                append_csv(sum_path, [srow])
                print(f"  [{site}/{arm}] 擾動 LPIPS {pa['lpips']:.4f} / L∞ "
                      f"{pa['linf']:.4f}   Δsiglip {m_s:+.5f} ± {sd_s:.5f}"
                      f"   Δniqe {m_n:+.4f} ± {sd_n:.4f}"
                      f"   語意失敗={srow['semantic_fail']}", flush=True)
                print(f"  [{site}/{arm}] 編輯 LPIPS {srow['edit_lpips']:.4f}  "
                      f"PSNR {srow['edit_psnr']:.2f}", flush=True)
            print(f"  [{site}] attn_div {res.history[0]['attn_div']:.4e}"
                  f" → {res.history[-1]['attn_div']:.4e}"
                  f"（{res.steps_done} 步，{train_s:.0f}s）", flush=True)
            del module, res
            torch.cuda.empty_cache()

    print(f"\n完成。{sum_path}", flush=True)
    print("讀法：semantic_fail 全為 False 且 Δsiglip 為正或接近零，"
          "即閘門未過——第二層在此威脅模型下不可答，見 docs/CONVERGENCE.md §3")


if __name__ == "__main__":
    main()
