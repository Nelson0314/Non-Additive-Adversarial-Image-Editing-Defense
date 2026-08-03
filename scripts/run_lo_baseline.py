"""在 L∞ ≤ κ 上重現 Lo et al. (CVPR 2024) Table 1 的三根柱子。

    python scripts/run_lo_baseline.py --data data/lo_aligned \
        --out runs/lo_baseline --attacks pg_encoder,pg_diffusion,semantic \
        --eval_seeds 20

三個攻擊共用同一個 PGD 迴圈、同一個 κ 與同一個步數，差別只在損失——這是
論文 §4.1 的公平比較設定（「we set the perturbation budget κ = 0.06,
number of iterations N = 100 for all attacks」）。

與 `run_defense.py` 的分工
─────────────────────────────────────────────────────────────────────

`run_defense.py` 跑的是本專案的方法：residual module + Adam + LPIPS／鈍化／
色度三道 hinge。本腳本跑的是文獻基準：像素 δ + PGD sign + L∞ 硬投影。
兩者的保真約束不同型，**同一張表裡的數字不可互相比較**，除非另外把失真
量對齊——那是後續的工作，不在本腳本內偷偷做。

多種子平均
─────────────────────────────────────────────────────────────────────

論文「averaging the editing results over 20 random seeds」。SDEdit 的輸出
對噪聲種子高度敏感，單種子的 Table 1 數字沒有意義：本專案 E29 之前的
n = 1 是後來一連串判定問題的來源（見 docs/LEDGER.md）。評測種子與攻擊
用的種子錯開 `EVAL_SEED_OFFSET`，否則量到的是訓練集表現。
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.defense.linf_attack import LinfAttackConfig, build_attack, pgd_linf
from src.metrics.suite import MetricSuite
from src.models.sd import SDWrapper
from src.utils.artifacts import save_image, save_json
from src.utils.device import get_device, peak_memory_mb, reset_peak_memory

# 與 run_defense.py 同一個常數、同一個理由：評測噪聲必須與攻擊時用的錯開，
# 否則量到的是「在訓練用的那一組 ε 上表現多好」。
EVAL_SEED_OFFSET = 10_000

# Table 1 的五個指標，附「防禦成功的方向」。注意這與 suite.HIGHER_IS_BETTER
# 相反：後者描述的是影像相似度，此處描述的是免疫效果。
TABLE1 = {
    "psnr": "lower", "ssim": "lower", "vif_p": "lower",
    "fsim": "lower", "lpips": "higher",
}


def load_dataset(root: Path, size: int, device, limit=None):
    """回傳 [(名稱, 張量, prompt, 要保護的內容 c_a)]。

    與 `run_defense.load_images` 的差別是多回傳 c_a。Lo 的 semantic attack
    需要指定「要保護哪個詞」，那是他的方法的核心輸入，不能由 prompt 猜——
    prompt 是攻擊方寫的，c_a 是防禦方選的，兩者在威脅模型裡屬於不同的人。
    """
    import yaml
    from PIL import Image
    import torchvision.transforms as T

    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for p in sorted(root.rglob("*.png")):
        cls = p.parent.name
        if cls not in spec:
            raise KeyError(
                f"{p} 所屬的類別 {cls!r} 不在 {root/'prompts.yaml'} 裡。"
                "每一類都必須明確宣告 prompts 與 content，不接受預設值——"
                "c_a 猜錯會讓 semantic attack 攻擊到別的東西而毫無症狀"
            )
        entry = spec[cls]
        img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        x = T.ToTensor()(img).unsqueeze(0).to(device)
        out.append((p.stem, x, list(entry["prompts"]), entry["content"]))
    return out[:limit] if limit else out


@torch.no_grad()
def reference_edits(sd, x01, prompt, cfg, n_seeds):
    """未防禦的編輯，逐種子各一張。回傳 [(seed, noise, y_ref)]。

    對同一張影像與同一個 prompt，`y_ref` 與攻擊方法無關。三個攻擊各自重算
    一次是把雲端時間白燒三分之一——每張影像 20 個種子的編輯鏈是本協定
    評測側的主要成本。故在攻擊迴圈之外算一次、三個攻擊共用。

    順帶保證了三個攻擊比的是**同一個**參照。分開算雖然種子相同、結果也應
    相同，但那是靠實作巧合而非結構保證。
    """
    emb = sd.encode_text(prompt).detach()
    emb_un = sd.encode_text("").detach()
    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    out = []
    for i in range(n_seeds):
        seed = cfg.seed + EVAL_SEED_OFFSET + i
        nz = sd.sample_edit_noise(torch.empty(lat, device=x01.device), seed=seed)
        y = sd.sdedit(x01, emb, nz, cfg.n_edit, strength=cfg.strength,
                      guidance_scale=cfg.guidance_scale, emb_uncond=emb_un)
        out.append((seed, nz, y))
    return out


def evaluate(sd, suite, refs, x_adv, prompt, cfg, out_dir, name):
    """逐種子比較「未防禦的編輯」與「免疫後的編輯」，回傳每個種子一列。

    兩條分支必須共用同一個 ε（spec §5.1），否則量到的差異主要來自噪聲不同。
    此處由 `refs` 帶入該 ε，故共用是結構上的而非約定上的。
    """
    emb = sd.encode_text(prompt).detach()
    emb_un = sd.encode_text("").detach()
    rows = []
    with torch.no_grad():
        for i, (seed, nz, y_ref) in enumerate(refs):
            y_def = sd.sdedit(x_adv, emb, nz, cfg.n_edit,
                              strength=cfg.strength,
                              guidance_scale=cfg.guidance_scale,
                              emb_uncond=emb_un)
            m = suite.full(y_ref, y_def, prompt=prompt)
            rows.append({"eval_seed": seed, **{f"edit_{k}": v for k, v in m.items()}})
            if i == 0:
                save_image(y_ref, out_dir / f"{name}_edit_ref.png")
                save_image(y_def, out_dir / f"{name}_edit_def.png")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="重現 Lo et al. (CVPR 2024) Table 1 的 L-infinity 協定"
    )
    ap.add_argument("--data", default="data/lo_aligned")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--attacks", default="pg_encoder,pg_diffusion,semantic")
    ap.add_argument("--target", default="data/targets/gray.png",
                    help="PhotoGuard 兩個變體的目標影像")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt_index", type=int, default=0,
                    help="用 prompts.yaml 的第幾個編輯 prompt")
    # 論文寫死的三個數字
    ap.add_argument("--kappa", type=float, default=0.06)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=10)
    # 論文未公布，本專案指定
    ap.add_argument("--step_size", type=float, default=None)
    ap.add_argument("--mask_tau", type=float, default=0.5)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--eval_seeds", type=int, default=20,
                    help="論文為 20；降低會讓 Table 1 的數字不可比")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)

    data = load_dataset(Path(args.data), args.size, device, args.limit)
    if not data:
        raise SystemExit(f"{args.data} 底下沒有任何 PNG")

    from PIL import Image
    import torchvision.transforms as T

    tgt_img = Image.open(args.target).convert("RGB").resize(
        (args.size, args.size), Image.LANCZOS)
    x_target = T.ToTensor()(tgt_img).unsqueeze(0).to(device)

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    rows, summary = [], []
    for name, x01, prompts, content in data:
        prompt = prompts[args.prompt_index]
        # 未防禦的編輯只算一次，三個攻擊共用。見 reference_edits 的 docstring。
        t_ref = time.perf_counter()
        refs = reference_edits(
            sd, x01,
            prompt,
            LinfAttackConfig(strength=args.strength, guidance_scale=args.guidance,
                             n_edit=args.n_edit, seed=args.seed),
            args.eval_seeds,
        )
        print(f"\n=== {name}：{len(refs)} 個種子的未防禦編輯，"
              f"{time.perf_counter()-t_ref:.0f}s ===", flush=True)
        for atk in attacks:
            cfg = LinfAttackConfig(
                kappa=args.kappa, steps=args.steps, step_size=args.step_size,
                timesteps=args.timesteps, content=content,
                mask_tau=args.mask_tau, strength=args.strength,
                guidance_scale=args.guidance, n_edit=args.n_edit,
                prompt_edit=prompt, seed=args.seed,
            )
            print(f"\n=== {name} / {atk} / c_a={content!r} / {prompt!r} ===",
                  flush=True)
            reset_peak_memory()
            t0 = time.perf_counter()
            terms = build_attack(atk, sd, cfg, x01, x_target)
            res = pgd_linf(x01, terms, cfg, tag=atk)
            save_image(res.x_adv, out / f"{name}__{atk}__adv.png")
            save_json(res.history, out / f"{name}__{atk}__history.json")

            # 擾動本身的失真：這是「他的約束」與「我們的約束」之間的橋。
            pert = suite.pairwise(x01, res.x_adv)
            ev = evaluate(sd, suite, refs, res.x_adv, prompt, cfg,
                          out, f"{name}__{atk}")
            for r in ev:
                rows.append({
                    "image": name, "attack": atk, "content": content,
                    "prompt": prompt, "kappa": args.kappa, "steps": args.steps,
                    "strength": args.strength, "guidance_scale": args.guidance,
                    "attack_seconds": round(res.seconds, 1),
                    "peak_mb": round(peak_memory_mb(), 1),
                    **{f"pert_{k}": v for k, v in pert.items()},
                    **r,
                })
            n = len(ev)
            summary.append({
                "image": name, "attack": atk,
                "n_seeds": n,
                **{f"edit_{k}": sum(r[f"edit_{k}"] for r in ev) / n
                   for k in TABLE1},
                "pert_linf": pert["linf"], "pert_lpips": pert["lpips"],
                "seconds": round(res.seconds, 1),
                "peak_mb": round(peak_memory_mb(), 1),
            })
            print(f"  [{atk}] 完成 {time.perf_counter()-t0:.0f}s  "
                  + "  ".join(f"{k}={summary[-1]['edit_'+k]:.4f}"
                              for k in TABLE1), flush=True)

    _write_csv(out / "results.csv", rows)
    _write_csv(out / "summary.csv", summary)
    print(f"\n寫出 {out/'results.csv'}（{len(rows)} 列）"
          f"與 {out/'summary.csv'}（{len(summary)} 列）")


def _write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"沒有任何列可以寫入 {path}；不寫出空檔案掩蓋失敗")
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
