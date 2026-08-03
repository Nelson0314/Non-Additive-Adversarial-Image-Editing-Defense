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

**維持 20，不要降。** 曾考慮降到 5，實測的取捨是：只省 12% 的總時間，因為
三個攻擊占 95% 的成本、評測本來就便宜；代價卻是平均值的標準誤差變成兩倍
（∝ 1/√n），而 SDEdit 對種子的變異本來就大。真正的成本槓桿是 `pg_diffusion`
（占 54%），但它是必要項。
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

    # 只走 prompts.yaml 宣告過的類別目錄，不用 rglob 掃整棵樹。
    #
    # rglob 會把根目錄下的任何 PNG 也算進來——實測 `overview.png`（資料集
    # 的總覽圖）就這樣被當成一張待攻擊的影像。防呆有擋下來，但正確的作法是
    # 「收錄的內容恰好等於宣告的內容」，而不是掃到什麼算什麼。
    out = []
    for cls, entry in spec.items():
        d = root / cls
        if not d.is_dir():
            raise FileNotFoundError(
                f"{root/'prompts.yaml'} 宣告了類別 {cls!r}，但 {d} 不存在"
            )
        for p in sorted(d.glob("*.png")):
            img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
            x = T.ToTensor()(img).unsqueeze(0).to(device)
            out.append((p.stem, x, list(entry["prompts"]), entry["content"]))

    # 未宣告的子目錄若放了 PNG，會被靜默忽略——那與「忘了宣告」無法區分，
    # 故明確拒絕。根目錄下的 PNG 不算（總覽圖之類的說明用檔案）。
    stray = [
        p for p in root.iterdir()
        if p.is_dir() and p.name not in spec and any(p.glob("*.png"))
    ]
    if stray:
        raise KeyError(
            f"這些目錄有 PNG 但沒有在 {root/'prompts.yaml'} 裡宣告："
            f"{[p.name for p in stray]}。每一類都必須明確宣告 prompts 與"
            " content，不接受預設值——c_a 猜錯會讓 semantic attack 攻擊到"
            "別的東西而毫無症狀"
        )
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
    ap.add_argument("--strength", type=float, default=0.3,
                help="論文未公布；0.3 對齊 PhotoGuard 的 img2img 評測")
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--eval_seeds", type=int, default=20,
                    help="論文為 20；降低只省約 12%% 的時間（攻擊占 95%% 成本），"
                         "卻讓平均值的標準誤差變大，不划算")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--resume", action="store_true",
                    help="接續既有的 --out：略過 summary.csv 裡已完成的格。"
                         "不加此旗標而目錄已有結果時會拒絕執行，不覆寫")
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

    res_path, sum_path = out / "results.csv", out / "summary.csv"
    done = completed_pairs(sum_path)
    if done and not args.resume:
        raise SystemExit(
            f"{sum_path} 已有 {len(done)} 格結果。要接續請加 --resume；"
            "要重跑請換一個 --out 或先自行移走舊目錄。"
            "此處不自動覆寫——`runs/` 是唯一的證據來源且實驗無法重跑"
        )
    check_protocol(out, args)
    if done:
        print(f"接續模式：已完成 {len(done)} 格，將略過", flush=True)

    n_done = 0
    for name, x01, prompts, content in data:
        prompt = prompts[args.prompt_index]
        todo = [a for a in attacks if (name, a) not in done]
        if not todo:
            print(f"=== {name}：三格皆已完成，略過 ===", flush=True)
            continue
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
        for atk in todo:
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
            coverage = getattr(terms, "mask_coverage", "")
            res = pgd_linf(x01, terms, cfg, tag=atk)
            save_image(res.x_adv, out / f"{name}__{atk}__adv.png")
            save_json(res.history, out / f"{name}__{atk}__history.json")

            # 擾動本身的失真：這是「他的約束」與「我們的約束」之間的橋。
            pert = suite.pairwise(x01, res.x_adv)
            ev = evaluate(sd, suite, refs, res.x_adv, prompt, cfg,
                          out, f"{name}__{atk}")
            rows = []
            for r in ev:
                rows.append({
                    "image": name, "attack": atk, "content": content,
                    "prompt": prompt, "kappa": args.kappa, "steps": args.steps,
                    "strength": args.strength, "guidance_scale": args.guidance,
                    "attack_seconds": round(res.seconds, 1),
                    "mask_coverage": coverage,
                    "step_size": cfg.step_size or cfg.kappa / 10,
                    "mask_tau": cfg.mask_tau,
                    "peak_mb": round(peak_memory_mb(), 1),
                    **{f"pert_{k}": v for k, v in pert.items()},
                    **r,
                })
            n = len(ev)
            srow = {
                "image": name, "attack": atk,
                "n_seeds": n,
                **{f"edit_{k}": sum(r[f"edit_{k}"] for r in ev) / n
                   for k in TABLE1},
                "pert_linf": pert["linf"], "pert_lpips": pert["lpips"],
                "mask_coverage": coverage,
                "seconds": round(res.seconds, 1),
                "peak_mb": round(peak_memory_mb(), 1),
            }
            # 每一格完成就落地，不是全部跑完才寫。24 張 × 3 攻擊要跑約
            # 三小時，中途斷線或 OOM 時把已算好的結果一起丟掉是不可接受的
            # ——`runs/` 是唯一的證據來源。
            append_csv(res_path, rows)
            append_csv(sum_path, [srow])
            n_done += 1
            print(f"  [{atk}] 完成 {time.perf_counter()-t0:.0f}s  "
                  + "  ".join(f"{k}={srow['edit_'+k]:.4f}" for k in TABLE1)
                  + f"   （本次第 {n_done} 格）", flush=True)

    if n_done == 0:
        print("沒有任何待跑的格；全部已完成。")
    else:
        print(f"\n本次完成 {n_done} 格，累計寫入 {sum_path}")


def completed_pairs(sum_path: Path) -> set:
    """讀 summary.csv，回傳已完成的 (影像, 攻擊) 組合。

    以 summary 而非 results 判定：summary 每格恰好一列，是在該格**全部**
    種子都評測完之後才寫入的。用 results 判定會把「寫到一半」的格子誤判為
    已完成，於是接續時跳過一個實際上不完整的格。
    """
    if not sum_path.exists():
        return set()
    with sum_path.open(encoding="utf-8", newline="") as f:
        return {(r["image"], r["attack"]) for r in csv.DictReader(f)}


# 決定「同一個 --out 底下的列彼此可比」的參數。改動其中任何一個，先前算的
# 格子就不再是同一個實驗。`--attacks` 不在此列：分批補跑攻擊是正當的接續。
# `--limit` 也不在此列：補進更多影像同樣正當。
PROTOCOL_KEYS = [
    "data", "size", "model", "prompt_index", "kappa", "steps", "timesteps",
    "step_size", "mask_tau", "strength", "guidance", "n_edit", "eval_seeds",
    "seed",
]


def check_protocol(out: Path, args) -> None:
    """首次執行時寫下協定參數；接續時比對，不符就拒絕。

    `--resume` 已是常態用法（見 scripts/drivers/lo_l1.sh），而接續與否只看
    summary.csv 的 (影像, 攻擊)，不看參數。少打一個 `--prompt_index 1` 就會
    把兩個編輯 prompt 的結果混進同一個平均，而且沒有任何症狀——summary.csv
    不記 prompt，事後也看不出來。這道守衛就是為了讓那種錯誤變成當場失敗。
    """
    now = {k: getattr(args, k) for k in PROTOCOL_KEYS}
    path = out / "protocol.json"
    if not path.exists():
        save_json(now, path)
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    diff = {k: (old.get(k), now[k]) for k in PROTOCOL_KEYS if old.get(k) != now[k]}
    if diff:
        raise SystemExit(
            f"{path} 記錄的協定與本次不同，拒絕接續：\n"
            + "\n".join(f"  {k}：既有 {o!r} → 本次 {n!r}" for k, (o, n) in diff.items())
            + "\n把不同協定的結果寫進同一個目錄會讓平均值無聲混雜。請換一個 --out。"
        )


def append_csv(path: Path, rows):
    """附加到 CSV；檔案不存在時連同表頭一起寫。

    表頭不符時拋出而非默默附加：欄位集合變了代表程式改過，把兩種 schema
    混在同一個檔裡會讓後續的判讀無聲地錯位。
    """
    if not rows:
        raise RuntimeError(f"沒有任何列可以寫入 {path}；不寫出空檔案掩蓋失敗")
    keys = list(rows[0].keys())
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            old = next(csv.reader(f), None)
        if old != keys:
            raise RuntimeError(
                f"{path} 既有的表頭與本次的欄位不同，拒絕附加。\n"
                f"  既有：{old}\n  本次：{keys}\n"
                "欄位變了表示程式改過；混在同一個檔裡會讓判讀無聲錯位。"
                "請換一個 --out。"
            )
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=keys).writerows(rows)
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
