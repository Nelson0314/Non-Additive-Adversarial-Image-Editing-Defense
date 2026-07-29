"""E2/E3 驅動腳本 — spec §7.3、§7.4。

對 (影像 × site × 秩) 的每一格優化一組 φ，然後在 E3 的淨化強度掃描下以
spec §8.1 的全部八項指標評測，並依 §8.3 留存全部影像產出。

**兩軸因子設計的用途**（spec §6.4）：site P 是低秩**加性**、site L 是低秩
**非加性**。兩者的對比切割「秩」與「非加性」這兩個對耐淨化性的競爭解釋：

- P 亦耐淨化 ⟹ 機制是秩結構
- P 不耐而 L 耐 ⟹ 機制是非加性

三種結果都是可發表的發現，故本腳本不對結果方向作任何假設。

執行：
    python scripts/run_defense.py --sites P,L --ranks 1,4,16 --steps 60 --out runs/e2
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.defense.objective import LossConfig
from src.defense.optimize import OptimConfig, optimize
from src.metrics.spectrum import analyze, low_freq_fraction
from src.metrics.suite import MetricSuite
from src.models.sd import SDWrapper
from src.purify.ops import Purifier, default_train_set, eval_sweep
from src.residual.site_latent import LatentResidual
from src.residual.site_latent_anchored import AnchoredLatentResidual
from src.residual.site_pixel import PixelResidual
from src.utils.artifacts import (
    save_history_plot,
    save_image,
    save_json,
    save_residual,
    save_spectrum_plot,
    save_x0_trace,
)
from src.utils.device import get_device, peak_memory_mb, reset_peak_memory


def load_images(root: Path, size: int, device, limit=None):
    """回傳 [(名稱, 張量, 編輯 prompt)]。prompt 取 prompts.yaml 的第一個。"""
    from PIL import Image
    import torchvision.transforms as T

    prompts = {}
    pf = root / "prompts.yaml"
    if pf.exists():
        import yaml

        prompts = yaml.safe_load(pf.read_text(encoding="utf-8")) or {}

    out = []
    for p in sorted(root.rglob("*.png")):
        cls = p.parent.name
        img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        x = T.ToTensor()(img).unsqueeze(0).to(device)
        plist = prompts.get(cls) or ["a photo"]
        out.append((p.stem, x, plist[0]))
    return out[:limit] if limit else out


def build_module(
    site: str, rank: int, cfg: OptimConfig, sd, size: int, seed: int,
    scale: float = 1.0,
):
    """依 site 建立殘差模塊。max_rank 直接取 rank：每格獨立訓練，不共用參數。

    `scale` 是注入殘差的固定倍率，不參與優化。site L/LA 的殘差加在 ε̂ 上，
    而 ε̂ 的量級約為 1；實測 scale=1 時 φ 造成的像素改變比重建誤差小
    20~36 倍（RMS），完全量不到。此參數即為 §3.7 建議 0 要掃的旋鈕。
    """
    if site == "P":
        return PixelResidual(
            size=size, channels=3, max_rank=rank, const_rank=rank,
            seed=seed, scale=scale,
        )
    if site in ("L", "LA"):
        lat = sd.latent_shape(size, size)
        cls = LatentResidual if site == "L" else AnchoredLatentResidual
        return cls(
            steps=cfg.k_inv, channels=lat[1], size=lat[-1],
            max_rank=rank, const_rank=rank, seed=seed, scale=scale,
        )
    raise ValueError(f"未實作的 site {site!r}（site W 見 spec §4.3，尚未實作）")


# 評測用的噪聲種子必須與訓練不同。φ 是針對訓練用的那一組 ε 優化出來的
# （n_eot=1 時尤其如此），若評測沿用同一組 ε，量到的是訓練集表現而非防禦
# 效果。偏移量會被系統性高估，且高估的幅度未知。
EVAL_SEED_OFFSET = 10_000


@torch.no_grad()
def evaluate(sd, suite, x01, x_def, cfg, prompt, out_dir, save_images=True):
    """E3 淨化強度掃描 + spec §8.1 全指標。

    兩條分支共用同一個 ε（spec §5.1）：否則量到的偏移主要來自噪聲差異。
    評測階段一律使用淨化的**真實實作**，不用訓練時的可微代理。

    **噪聲以未見過的種子取樣**（見 EVAL_SEED_OFFSET）。另外在 identity
    淨化下額外量一次訓練用的種子，兩者之差即為對特定噪聲的過擬合幅度，
    以 `noise_split` 欄位區分，報告中必須併列。
    """
    device = x01.device
    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    emb = sd.encode_text(prompt)

    def branch(seed):
        n = sd.sample_edit_noise(torch.empty(lat, device=device), seed=seed)
        return n, sd.sdedit(x01, emb, n, cfg.n_edit, strength=cfg.strength)

    noise, y_orig = branch(cfg.seed + EVAL_SEED_OFFSET)
    noise_tr, y_orig_tr = branch(cfg.seed)
    if save_images:
        save_image(y_orig, out_dir / "edit_orig.png")
        save_image(y_orig_tr, out_dir / "edit_orig_trainnoise.png")

    def measure(xp, nz, y_ref, kind, strength, split, x_ctrl=None):
        """`x_ctrl` 為**未防禦**的對照輸入（同一淨化施加於原圖）。

        必要性：spec §5.1 的 `d(E(P(x_def)), E(x))` 把淨化本身造成的偏移
        也算成防禦效果。`P(x) ≠ x`，故即使 φ=0，模糊或 JPEG 也會讓編輯結果
        偏離 `E(x)`。實測 site P r=1 在 identity 下 shift=0.095、在 blur 下
        0.347，高的那個是淨化自己造成的，不是防禦變強。不減掉對照就會讓
        E3 的每個數字被系統性高估。
        """
        y_def = sd.sdedit(xp, emb, nz, cfg.n_edit, strength=cfg.strength)
        row = {
            "purify": kind,
            "strength": strength,
            "noise_split": split,
            "proxy_gap": 0.0,
            **{f"edit_{k}": v for k, v in suite.full(y_ref, y_def, prompt=prompt).items()},
            **{f"defimg_{k}": v for k, v in suite.pairwise(x01, xp).items()},
        }
        if x_ctrl is not None:
            y_ctrl = sd.sdedit(x_ctrl, emb, nz, cfg.n_edit, strength=cfg.strength)
            m = suite.pairwise(y_ref, y_ctrl)
            row["ctrl_lpips"] = m["lpips"]
            row["ctrl_psnr"] = m["psnr"]
            # 防禦淨額：扣掉淨化本身造成的偏移後，還剩多少歸因於防禦
            row["net_lpips"] = row["edit_lpips"] - m["lpips"]
        return y_def, row

    rows = []
    # 過擬合幅度：同一張防禦圖、無淨化，只換噪聲種子
    _, row_tr = measure(x_def, noise_tr, y_orig_tr, "identity", 0.0, "train", x01)
    rows.append(row_tr)

    for kind, plist in eval_sweep().items():
        for pur in plist:
            xp = pur.evaluate(x_def)
            # 對照輸入：同一個淨化算子施加於**原圖**，φ 完全沒有參與
            y_def, row = measure(
                xp, noise, y_orig, kind, pur.strength, "heldout",
                x_ctrl=pur.evaluate(x01),
            )
            if not pur.differentiable:
                row["proxy_gap"] = pur.proxy_gap(x_def)
            rows.append(row)

            if save_images and pur.strength in (plist[0].strength, plist[-1].strength):
                tag = f"{kind}_{pur.strength}"
                save_image(xp, out_dir / f"purified_{tag}.png")
                save_image(y_def, out_dir / f"edit_def_{tag}.png")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--data", default="data/dayn_testset")
    ap.add_argument("--out", default="runs/e2")
    ap.add_argument("--sites", default="P,L")
    ap.add_argument("--ranks", default="1,4,16")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--k_inv", type=int, default=10)
    ap.add_argument(
        "--t_max", type=int, default=None,
        help="inversion timestep 上限。依 E0c 的重建地板量測結果指定",
    )
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--n_eot", type=int, default=1)
    ap.add_argument(
        "--scales", default="1.0",
        help="殘差注入倍率，逗號分隔可掃描。site L/LA 於 scale=1 時 φ 的"
             "效果比重建誤差小 20~36 倍（見 NIGHT_RUN §3.7）",
    )
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 張影像")
    ap.add_argument("--prompt_def", default="", help="防禦生成 prompt，預設空字串")
    ap.add_argument("--no_eval", action="store_true", help="只優化，跳過淨化掃描")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    sites = [s.strip() for s in args.sites.split(",")]
    ranks = [int(r) for r in args.ranks.split(",")]
    scales = [float(v) for v in args.scales.split(",")]

    print(f"[run] device={device} sites={sites} ranks={ranks} steps={args.steps}")
    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)
    images = load_images(Path(args.data), args.size, device, args.limit)
    print(f"[run] 影像 {len(images)} 張：{[n for n, _, _ in images]}")

    loss_cfg = LossConfig()
    purifiers = default_train_set()
    print(f"[run] 訓練期淨化集 {[p.kind for p in purifiers]}")

    all_rows = []
    summary = []
    t_start = time.perf_counter()

    for name, x01, prompt in images:
        for site in sites:
          for scale in scales:
            for rank in ranks:
                tag = f"{name}__{site}__r{rank}"
                if len(scales) > 1:
                    tag += f"__s{scale:g}"
                cell = out / tag
                cell.mkdir(parents=True, exist_ok=True)
                print(f"\n[run] {tag}  prompt={prompt!r}", flush=True)

                cfg = OptimConfig(
                    steps=args.steps, lr=args.lr, k_inv=args.k_inv,
                    t_max=args.t_max,
                    n_edit=args.n_edit, n_eot=args.n_eot, strength=args.strength,
                    prompt_def=args.prompt_def, prompt_edit=prompt, seed=args.seed,
                )
                module = build_module(
                    site, rank, cfg, sd, args.size, args.seed, scale=scale
                ).to(device)

                reset_peak_memory()
                res = optimize(sd, module, x01, cfg, loss_cfg, purifiers)
                peak = peak_memory_mb()

                # ---- spec §8.3 產出留存 ----
                save_image(x01, cell / "orig.png")
                save_image(res.x_def, cell / "defended.png")
                # x_base = G(x; φ=0)：該 site 未施加防禦時就已產生的圖。
                # 留存它，讀者才分得清哪些失真來自防禦、哪些來自重建。
                if res.x_base is not None:
                    save_image(res.x_base, cell / "baseline_phi0.png")
                    save_residual(res.x_def - res.x_base, cell / "residual_phi.png")
                delta = res.x_def - x01
                gain = save_residual(delta, cell / "residual.png")
                spec_an = analyze(delta)
                save_spectrum_plot(spec_an, cell / "spectrum.png", title=tag)

                # clamp 前的殘差另行分析。site P 的 x_def−x 已過 clamp，其
                # 數值秩不等於設定值（spec §7.2 修訂紀錄）；兩者分開量測才
                # 能區分「秩約束失效」與「clamp 造成的稀疏擾動」。
                raw_an, clamp_frac = None, None
                raw = module.raw_residual()
                if raw is not None:
                    raw_an = analyze(raw)
                    save_spectrum_plot(
                        raw_an, cell / "spectrum_raw.png", title=f"{tag} (pre-clamp)"
                    )
                    save_json(
                        {k: v for k, v in raw_an.items() if k != "per_channel"},
                        cell / "spectrum_rank_raw.json",
                    )
                    if hasattr(module, "clamped_fraction"):
                        clamp_frac = module.clamped_fraction(x01)
                save_history_plot(res.history, cell / "history.png", title=tag)
                save_json(
                    {k: v for k, v in spec_an.items() if k != "per_channel"},
                    cell / "spectrum_rank.json",
                )
                save_json(res.history, cell / "history.json")
                if res.x0_trace:
                    save_x0_trace(res.x0_trace, sd, cell / "x0_trace")

                last = res.history[-1]
                base = {
                    "image": name, "site": site, "rank": rank, "scale": scale,
                    "prompt": prompt,
                    "steps": cfg.steps, "k_inv": cfg.k_inv, "n_edit": cfg.n_edit,
                    "n_eot": cfg.n_eot, "seconds": round(res.seconds, 1),
                    "peak_mb": round(peak, 1), "residual_gain": round(gain, 2),
                    "eff_rank_mean": _mean(spec_an["effective_rank"]),
                    "energy_rank_99_mean": _mean(spec_an["energy_rank_99"]),
                    "energy_rank_90_mean": _mean(spec_an["energy_rank_90"]),
                    # 低頻能量比例：攻擊者的 SDEdit 先加噪到 t0，抹除高頻，
                    # 故只有低頻/結構性的擾動可能存活（見 docs/E4_SCALE_SWEEP.md）
                    "low_freq_frac": low_freq_fraction(delta),
                    "raw_eff_rank_mean": _mean(raw_an["effective_rank"]) if raw_an else "",
                    "raw_energy_rank_99_mean": (
                        _mean(raw_an["energy_rank_99"]) if raw_an else ""
                    ),
                    "clamped_fraction": clamp_frac if clamp_frac is not None else "",
                    "final_loss": last["loss"], "final_L_def": last["L_def"],
                    "final_L_fid": last["L_fid"], "final_shift": last["edit_shift"],
                    # final_* 為相對 x_base（防禦造成的改變），
                    # final_*_total 為相對原圖的絕對值。前緣圖用後者。
                    "final_psnr": last["fid_psnr"], "final_linf": last["fid_linf"],
                    "final_psnr_total": last["fid_psnr_total"],
                    "final_linf_total": last["fid_linf_total"],
                    "final_ssim": last["fid_ssim"],
                    "final_lpips": last["fid_lpips"],
                }
                summary.append(base)
                print(
                    f"[run] {tag} 完成 {res.seconds:.0f}s peak={peak:.0f}MB "
                    f"shift={last['edit_shift']:.4f} psnr={last['fid_psnr']:.2f} "
                    f"eff_rank={base['eff_rank_mean']:.1f}",
                    flush=True,
                )

                if not args.no_eval:
                    rows = evaluate(sd, suite, x01, res.x_def, cfg, prompt, cell)
                    for r in rows:
                        all_rows.append({**base, **r})

                del module, res
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 每格結束就落盤，中途中斷也保得住已完成的結果
                _write_csv(out / "summary.csv", summary)
                if all_rows:
                    _write_csv(out / "results.csv", all_rows)

    env = {
        "model": args.model, "size": args.size, "sites": sites, "ranks": ranks,
        "steps": args.steps, "lr": args.lr, "k_inv": args.k_inv,
        "scales": scales,
        "t_max": args.t_max,
        "n_edit": args.n_edit, "n_eot": args.n_eot, "strength": args.strength,
        "seed": args.seed, "prompt_def": args.prompt_def,
        "n_images": len(images), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_purifiers": [
            {"kind": p.kind, "strength": p.strength, "differentiable": p.differentiable}
            for p in purifiers
        ],
        "loss": loss_cfg.__dict__,
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    save_json(env, out / "env.json")
    print(f"\n[run] 全部完成，共 {len(summary)} 格，{env['total_seconds']:.0f}s")


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _write_csv(path: Path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
