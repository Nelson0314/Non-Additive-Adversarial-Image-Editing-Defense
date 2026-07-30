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

import numpy as np
import torch

from src.defense.objective import LossConfig
from src.defense.optimize import (
    OptimConfig, optimize, optimize_crossattn, optimize_encoder,
)
from src.metrics.spectrum import analyze
from src.metrics.suite import MetricSuite
from src.models.sd import SDWrapper
from src.purify.ops import Purifier, default_train_set, eval_sweep
from src.residual.site_embedding import EmbeddingResidual
from src.residual.site_latent import LatentResidual
from src.residual.site_pixel import PixelResidual
from src.residual.site_pixel_full import FullRankPixelResidual
from src.residual.site_warp import WarpResidual
from src.residual.site_weight import WeightResidual
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
    """回傳 [(名稱, 張量, prompt 清單)]。

    回傳整個清單而非只有第一個：prompts.yaml 每類已有兩個惡意編輯 prompt，
    訓練只用第 [0] 個，第 [1] 個是現成的 held-out。φ 是針對特定 prompt
    優化出來的，只在訓練用的那個上評測量到的是訓練集表現。
    """
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
        plist = list(prompts.get(cls) or ["a photo"])
        out.append((p.stem, x, plist))
    return out[:limit] if limit else out


def build_module(site: str, rank: int, cfg: OptimConfig, sd, size: int, seed: int):
    """依 site 建立殘差模塊。max_rank 直接取 rank：每格獨立訓練，不共用參數。"""
    if site == "P":
        return PixelResidual(
            size=size, channels=3, max_rank=rank, const_rank=rank, seed=seed
        )
    if site == "PF":
        # 全秩對照。rank 引數在此無意義（架構上不設限），仍照收以維持
        # 呼叫端介面一致；掃描時以 --ranks 0 表示「不適用」較不易誤讀。
        return FullRankPixelResidual(size=size, channels=3, seed=seed)
    if site == "W":
        # 掛在 SD 的 UNet 上，故呼叫端**必須**在該格結束後呼叫 module.remove()，
        # 否則 hook 會累積到下一格，症狀是「另一個 site 的結果莫名被改動」。
        return WeightResidual(sd.unet, rank=rank, seed=seed)
    if site == "E":
        # 形狀由 text encoder 的實際輸出決定，不寫死 77×768：tiny-SD 的
        # 維度與 SD v1.4 不同，寫死會讓本機煙霧測試跑不起來。
        emb = sd.encode_text(cfg.prompt_def)
        return EmbeddingResidual(
            tokens=emb.shape[-2], dim=emb.shape[-1],
            max_rank=rank, const_rank=rank, seed=seed,
        )
    if site == "L":
        lat = sd.latent_shape(size, size)
        return LatentResidual(
            steps=cfg.k_inv, channels=lat[1], size=lat[-1],
            max_rank=rank, const_rank=rank, seed=seed,
        )
    if site == "S":
        # 空間變形。此處 rank 引數被重新解釋為**位移場的控制網格邊長**：
        # 掃描介面沿用 --ranks 不另開旗標，但報告中必須寫成 grid_size，
        # 因為本位置沒有低秩結構，寫成「秩」會誤導。
        return WarpResidual(
            size=size, grid_size=(rank if rank > 0 else None),
            max_disp=cfg.warp_max_disp, seed=seed,
        )
    raise ValueError(
        f"未知的 site {site!r}；目前支援 "
        "P（像素低秩）、PF（像素全秩對照）、L（latent ε 注入）、"
        "E（文字嵌入）、W（權重空間 LoRA）、S（空間變形）"
    )


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


@torch.no_grad()
def evaluate_generalization(sd, suite, x01, x_def, cfg, prompts, strengths):
    """φ 對「訓練時沒見過的攻擊設定」還剩多少效果。

    訓練時固定了三件事：一個編輯 prompt、強度 0.5、10 步編輯。三者都是
    攻擊者可以自由更換的，任何一項換掉就失效的防禦沒有意義。此處量前兩項
    （prompt 與強度），第三項（編輯步數）由 cfg.n_edit 控制、另行掃描。

    **只在無淨化下量。** 完整的 (prompt × 強度 × 23 個淨化設定) 是 138 條
    編輯鏈，成本不成比例；泛化性與耐淨化性是兩個獨立的問題，先分開回答。

    每個組合仍然扣掉未防禦對照：換 prompt 或強度本身就會改變編輯結果，
    不減掉的話量到的是「換設定造成的差異」而非防禦效果。
    """
    device = x01.device
    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    rows = []

    for pi, prompt in enumerate(prompts):
        emb = sd.encode_text(prompt)
        for s in strengths:
            n = sd.sample_edit_noise(
                torch.empty(lat, device=device), seed=cfg.seed + EVAL_SEED_OFFSET
            )
            y_orig = sd.sdedit(x01, emb, n, cfg.n_edit, strength=s)
            y_def = sd.sdedit(x_def, emb, n, cfg.n_edit, strength=s)
            m_def = suite.pairwise(y_orig, y_def)
            rows.append({
                "prompt_idx": pi,
                # pi == 0 是訓練時用的那一個，其餘為 held-out
                "prompt_split": "train" if pi == 0 else "heldout",
                "eval_prompt": prompt,
                "eval_strength": s,
                "strength_split": "train" if s == cfg.strength else "heldout",
                "edit_lpips": m_def["lpips"],
                "edit_psnr": m_def["psnr"],
                # 未防禦對照在無淨化下恆為 0（x_ctrl = x01，兩條鏈完全相同），
                # 故 net 等於 edit。仍明寫出來，避免與有淨化的表格混淆時誤讀。
                "ctrl_lpips": 0.0,
                "net_lpips": m_def["lpips"],
            })
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
        "--purify_mode", default="rotate", choices=["rotate", "all"],
        help="rotate=每步一個淨化算子輪替（原始行為）；"
             "all=每步對全部算子求梯度後平均，成本乘以算子數",
    )
    ap.add_argument(
        "--align_steps", type=int, default=0,
        help="階段一（保真對齊）的步數。先訓練 φ 使 G(x;φ) 逼近 x，"
             "再以該 φ 熱啟動防禦訓練。0 表示不執行（既有行為）",
    )
    ap.add_argument("--align_lr", type=float, default=0.008)
    ap.add_argument(
        "--align_gamma_psnr", type=float, default=1.0,
        help="階段一專用的 PSNR 係數，覆蓋防禦階段的 0.0。"
             "重建對齊是逐像素準確度確實重要的場合（見 E9）",
    )
    ap.add_argument(
        "--tau_lpips", type=float, default=LossConfig.tau_lpips,
        help="保真度綁定約束：LPIPS(x_def, x_base) 的上限。"
             "全秩與低秩的比較以此為匹配軸，須掃描而非取單一值",
    )
    ap.add_argument(
        "--defense_mode", default="untargeted",
        choices=["untargeted", "targeted", "encoder", "crossattn"],
        help="untargeted=把編輯結果推離原編輯（既有行為）；"
             "targeted=推向 --target_image；"
             "encoder=改攻擊 VAE 編碼器，完全不走去噪鏈，每步成本降一個數量級；"
             "crossattn=破壞 prompt token 與影像位置的 cross-attention 綁定，"
             "每步只做 --attn_timesteps 次單步 UNet 前向",
    )
    ap.add_argument(
        "--attn_mode", default=OptimConfig.attn_mode,
        choices=["divergence", "entropy"],
        help="divergence=把注意力分佈推離原圖的；entropy=把它推向均勻",
    )
    ap.add_argument(
        "--attn_timesteps", type=int, default=OptimConfig.attn_timesteps,
        help="cross-attention 目標每步取樣的 timestep 數，均分於 [0, t_edit]",
    )
    ap.add_argument(
        "--target_image", default="",
        help="defense_mode=targeted 時的目標影像路徑",
    )
    ap.add_argument(
        "--warp_max_disp", type=float, default=OptimConfig.warp_max_disp,
        help="site S 的位移場硬上界，單位為像素。空間變形的失真預算是位移量"
             "而非 L∞，故此值與 --tau_lpips 同等重要，兩者都會寫入 env.json",
    )
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 張影像")
    ap.add_argument("--prompt_def", default="", help="防禦生成 prompt，預設空字串")
    ap.add_argument("--no_eval", action="store_true", help="只優化，跳過淨化掃描")
    ap.add_argument(
        "--eval_strengths", default="",
        help="泛化性評測的 SDEdit 強度清單，逗號分隔（例：0.3,0.5,0.7）。"
             "會與 prompts.yaml 的全部 prompt 交叉，在無淨化下量測，"
             "結果寫入 generalization.csv。留空表示不跑",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    sites = [s.strip() for s in args.sites.split(",")]
    ranks = [int(r) for r in args.ranks.split(",")]

    print(f"[run] device={device} sites={sites} ranks={ranks} steps={args.steps}")
    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)
    images = load_images(Path(args.data), args.size, device, args.limit)
    print(f"[run] 影像 {len(images)} 張：{[n for n, _, _ in images]}")

    # tau_lpips 是全秩與低秩比較的匹配軸：兩個 arm 在同一組 τ 下各跑一次，
    # 比較的是兩條曲線而非單點，結論不受匹配點的選擇左右。故它必須是 CLI
    # 參數，不能寫死在 LossConfig 的預設值裡。
    loss_cfg = LossConfig(tau_lpips=args.tau_lpips,
                          defense_mode=args.defense_mode)
    # 有目標模式的目標影像。載入時機在迴圈外：它對 φ 與影像都是常數。
    y_target = None
    if args.target_image:
        from PIL import Image
        import torchvision.transforms as T
        _t = Image.open(args.target_image).convert('RGB').resize(
            (args.size, args.size), Image.LANCZOS)
        y_target = T.ToTensor()(_t).unsqueeze(0).to(device)
    purifiers = default_train_set()
    print(f"[run] 訓練期淨化集 {[p.kind for p in purifiers]}")

    all_rows = []
    summary = []
    t_start = time.perf_counter()

    gen_rows = []

    for name, x01, plist in images:
        prompt = plist[0]        # 訓練一律用第 [0] 個，其餘保留為 held-out
        for site in sites:
            for rank in ranks:
                tag = f"{name}__{site}__r{rank}"
                cell = out / tag
                cell.mkdir(parents=True, exist_ok=True)
                print(f"\n[run] {tag}  prompt={prompt!r}", flush=True)

                cfg = OptimConfig(
                    steps=args.steps, lr=args.lr, k_inv=args.k_inv,
                    t_max=args.t_max,
                    n_edit=args.n_edit, n_eot=args.n_eot, strength=args.strength,
                    purify_mode=args.purify_mode,
                    align_steps=args.align_steps, align_lr=args.align_lr,
                    align_gamma_psnr=args.align_gamma_psnr,
                    warp_max_disp=args.warp_max_disp,
                    attn_mode=args.attn_mode,
                    attn_timesteps=args.attn_timesteps,
                    prompt_def=args.prompt_def, prompt_edit=prompt, seed=args.seed,
                )
                module = build_module(site, rank, cfg, sd, args.size, args.seed).to(device)

                reset_peak_memory()
                # encoder 模式是不同的方法而非 optimize 的選項：完全沒有
                # SDEdit、沒有 y_orig、沒有編輯 prompt，故走獨立的迴圈。
                if args.defense_mode == "encoder":
                    res = optimize_encoder(
                        sd, module, x01, cfg, loss_cfg, purifiers)
                elif args.defense_mode == "crossattn":
                    res = optimize_crossattn(
                        sd, module, x01, cfg, loss_cfg, purifiers)
                else:
                    res = optimize(sd, module, x01, cfg, loss_cfg,
                                   purifiers, y_target=y_target)
                peak = peak_memory_mb()

                # ---- spec §8.3 產出留存 ----
                save_image(x01, cell / "orig.png")
                save_image(res.x_def, cell / "defended.png")
                # baseline_phi0.png 恆為 G(x; φ=0)，即該位置未施加任何防禦時
                # 就已產生的圖；留存它，讀者才分得清哪些失真來自防禦、哪些來自
                # 重建。跑過階段一後 res.x_base 是 G(x; φ_align)，與 φ=0 是
                # 兩張不同的圖，故分開存，檔名各自對應其實際內容。
                if res.x_base0 is not None:
                    save_image(res.x_base0, cell / "baseline_phi0.png")
                if res.x_base is not None:
                    # 防禦本身造成的殘差：相對於防禦訓練時實際採用的保真基準
                    save_residual(res.x_def - res.x_base, cell / "residual_phi.png")
                delta = res.x_def - x01
                gain = save_residual(delta, cell / "residual.png")
                # 殘差另存 float32 陣列。residual.png 是 8-bit 且經過正規化
                # 放大，只能看不能算：像素注入 r=16 的殘差 RMS 約 2.6/255
                # （PSNR 39.8 dB），量化後細分徑向頻譜拿不到可用訊號。
                # .npy 不進 git（.gitignore 只收 csv/json/md/png），留在
                # 持久儲存供後續分析。
                np.save(cell / "residual.npy",
                        delta.detach().float().cpu().numpy())
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

                # 階段一的產物與逐步記錄。x_base 在執行階段一後是
                # G(x; φ_align)，與 x_base0 = G(x; φ=0) 是兩張不同的圖，
                # 兩者都留存，否則無法判斷階段一到底吸收掉多少重建誤差。
                if res.align_history:
                    save_json(res.align_history, cell / "align_history.json")
                    save_image(res.x_base, cell / "aligned_phi.png")

                a_last = res.align_history[-1] if res.align_history else None
                last = res.history[-1]
                base = {
                    "image": name, "site": site, "rank": rank, "prompt": prompt,
                    "steps": cfg.steps, "k_inv": cfg.k_inv, "n_edit": cfg.n_edit,
                    "n_eot": cfg.n_eot, "seconds": round(res.seconds, 1),
                    "purify_mode": cfg.purify_mode,
                    "align_steps": cfg.align_steps,
                    "align_seconds": round(res.align_seconds, 1),
                    # 階段一結束時 G(x; φ_align) 相對原圖的重建品質。這兩個
                    # 數字就是「低秩注入有沒有足夠容量吸收重建誤差」的答案。
                    "align_lpips": a_last["fid_lpips"] if a_last else "",
                    "align_psnr": a_last["fid_psnr_total"] if a_last else "",
                    "peak_mb": round(peak, 1), "residual_gain": round(gain, 2),
                    "eff_rank_mean": _mean(spec_an["effective_rank"]),
                    "energy_rank_99_mean": _mean(spec_an["energy_rank_99"]),
                    "energy_rank_90_mean": _mean(spec_an["energy_rank_90"]),
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
                # site S 的位移統計。空間變形的失真預算是位移量而非像素差值，
                # 少了這幾欄，「該格實際用掉多少預算」在 csv 裡完全看不出來。
                if hasattr(module, "disp_stats"):
                    base.update(module.disp_stats())
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

                if args.eval_strengths:
                    for r in evaluate_generalization(
                        sd, suite, x01, res.x_def, cfg, plist,
                        [float(s) for s in args.eval_strengths.split(",")],
                    ):
                        gen_rows.append({**base, **r})

                # site W 把 forward hook 註冊在 SD 的模組上；模塊被垃圾回收
                # 不會移除它們。不卸除的話 hook 會累積到後續每一格，而症狀是
                # 「別的 site 的結果莫名被改動」，極難追。
                if hasattr(module, "remove"):
                    module.remove()
                del module, res
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 每格結束就落盤，中途中斷也保得住已完成的結果
                _write_csv(out / "summary.csv", summary)
                if all_rows:
                    _write_csv(out / "results.csv", all_rows)
                if gen_rows:
                    _write_csv(out / "generalization.csv", gen_rows)

    env = {
        "model": args.model, "size": args.size, "sites": sites, "ranks": ranks,
        "steps": args.steps, "lr": args.lr, "k_inv": args.k_inv,
        "t_max": args.t_max,
        "n_edit": args.n_edit, "n_eot": args.n_eot, "strength": args.strength,
        "seed": args.seed, "prompt_def": args.prompt_def,
        # 這五項原本沒有記錄，導致 E11/E12 跑完後無法從 env.json 判斷實際
        # 用了哪個 align_lr、階段一的 PSNR 項有沒有生效——`loss` 子物件記的
        # 是防禦階段的 gamma_psnr（0.0），階段一是另一組係數。當時只能靠
        # 曲線與獨立探測結果反推，那不是可接受的可重現性。
        "purify_mode": args.purify_mode,
        "defense_mode": args.defense_mode,
        "target_image": args.target_image,
        "align_steps": args.align_steps, "align_lr": args.align_lr,
        "align_gamma_psnr": args.align_gamma_psnr,
        "warp_max_disp": args.warp_max_disp,
        "attn_mode": args.attn_mode, "attn_timesteps": args.attn_timesteps,
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
