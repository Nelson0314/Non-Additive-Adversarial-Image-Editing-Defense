"""把既有的解沿射線縮放，量出「可辨代價 vs 防禦效果」的整條曲線。

    python scripts/ray_curve.py --run runs/gate_suppress --site PF

不需要訓練，只有 SDEdit 與指標。

## 為什麼要改成量曲線

「匹配失真」至今**四次**被證明是假的（LEDGER 2.8、1.25）。最後一次最乾淨：
同一個 site、同一個參數化、同一個 LPIPS，最佳化解與隨機解的觀感與 NIQE
差 6 倍。也就是說**用單一純量把兩個方法對到同一點，這件事本身做不到**。

而且對本專案還多一層困難：**兩個 site 根本到不了同一個失真**。site PF 在
τ_lpips = 0.55 上收斂到 LPIPS 0.535，site S 在同一設定下 60 步只走到約 0.11
（`runs/gate_S/`）。要求它們落在同一點是要求一件不可能的事。

標準的替代作法是**量曲線再比曲線**，而不是求匹配點：對每個方法掃過幾個
擾動強度，把 (可辨代價, 防禦效果) 畫出來。兩條曲線只要在某個區間重疊就能
比較，不需要任何一次執行剛好落在指定的點上。

## 縮放在哪個空間做

`δ = x_def − x`，輸出 `clip(x + k·δ, 0, 1)`。

**對 site PF（加性）這就是精確的射線縮放。** 對 site S（空間變形）不是：
影像空間的線性內插是**交叉淡入**，不是「更大的位移」。兩者只在一階近似下
相同（`x_warp(k·f) ≈ x + k·(x_warp(f) − x)`，對小位移成立）。

故本腳本對 site S 的結果必須標明是一階近似。`gate_suppress.py` 自
2026-08-04 起會存 `__phi.pt`，之後可改為直接縮放位移場本身；
既有的 `runs/gate_suppress/` 沒有那個檔，只能用影像空間。

## 每一級都要有同失真的隨機對照

理由見 LEDGER 1.18：同一個 LPIPS 上的隨機高斯雜訊就取得最佳化解 60–74%
的效果。沒有這個條件，曲線的高度無從解讀。隨機條件在**每一級**各自匹配該級
實際達到的 LPIPS。

## 報什麼

不把「可辨代價」壓成一個純量。逐級同時報 LPIPS、NIQE 差、L∞、PSNR，
因為 1.25 已經證明它們不是同一件事。
"""

import argparse
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gate_suppress import append_csv, random_control  # noqa: E402
from src.metrics.ray_scale import lpips_against, solve_k  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image, save_json  # noqa: E402
from src.utils.device import get_device  # noqa: E402

EVAL_SEED_OFFSET = 10_000
TABLE1 = ("psnr", "ssim", "vif_p", "fsim", "lpips")
# 曲線要取值的 LPIPS。**指定目標而不是指定縮放係數**，兩個 site 的曲線
# 才會落在完全相同的 x 座標上、可以直接對讀。實測 site PF 的射線覆蓋
# LPIPS 0.088（k=0.10）到 0.764（k=4.0），故這個梯子在 PF 上全部可達。
# 某個 site 到不了某一級時，腳本會明確跳過並印出——**那個「到不了」本身
# 就是關於該參數化的結果**，不是錯誤。
TARGETS = (0.10, 0.20, 0.35, 0.50)


def exact_ray(run: Path, image: str, site: str, x01, device, resample=None):
    """若存了 φ，回傳一個「把 φ 乘上 k 之後重新生成防禦圖」的函式。

    影像空間的縮放 `x + k·(x_def − x)` 對加性位置是精確的，對 site S 不是
    ——線性內插是**交叉淡入**，不是「更大的位移」。兩者只在一階近似下相同
    （`x_warp(k·f) ≈ x + k·(x_warp(f) − x)`，對小位移成立），而本腳本要走到
    k = 4 以上，一階近似在那裡已經不成立：大 k 會讓內插出來的影像被 clamp
    成振鈴狀的假影，那不是「更大的位移」。

    `gate_suppress.py` 自 2026-08-04 起會存 `__phi.pt`。有它就直接縮放
    位移場本身，沒有就回傳 None 讓呼叫端退回影像空間並標明。
    """
    import torch

    from src.residual.site_warp import WarpResidual

    f = run / f"{image}__{site}__phi.pt"
    if site != "S" or not f.exists():
        return None
    state = torch.load(f, map_location=device)
    if "flow" not in state:
        raise SystemExit(
            f"{f} 沒有 `flow` 參數，不是 WarpResidual 的 state_dict")
    proto = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    rs = resample or proto.get("warp_resample")
    if rs is None:
        raise SystemExit(
            f"{run/'protocol.json'} 沒有 `warp_resample`，而重建位移場必須"
            "知道當初用的插值模式——bilinear 與 bicubic 的銳利度保留率差 15 "
            "個百分點（E20 §5.2），猜錯會得到另一個解。"
            "請以 --warp_resample 明確指定")
    md = proto.get("warp_max_disp")
    if md is None:
        raise SystemExit(
            f"{run/'protocol.json'} 沒有 `warp_max_disp`，無法重建位移場")
    flow = state["flow"].to(device)

    def build(k: float):
        m = WarpResidual(
            size=x01.shape[-1], grid_size=flow.shape[-1],
            max_disp=md, resample=rs,
        ).to(device)
        with torch.no_grad():
            m.flow.copy_(flow * k)
            return m.pixel_residual(x01).clamp(0, 1)

    return build


def load_png(path: Path, device):
    from PIL import Image
    import torchvision.transforms as T

    if not path.exists():
        raise SystemExit(f"{path} 不存在")
    return T.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


@torch.no_grad()
def reference_edits(sd, x01, prompt, cfg, n_seeds):
    """未防禦的編輯，逐種子各一張。回傳 [(seed, noise, y_ref)]。

    `y_ref` 只依賴 (影像, 種子)，與縮放等級和條件都無關。整個 run 算一次、
    全部重用：本腳本有 4 個等級 × 2 個條件，逐次重算會多花 44% 的評測時間，
    而評測正是本腳本的全部成本。`run_lo_baseline` 出於同一理由讓三個攻擊
    共用同一份參照。

    共用也順帶讓比較是**結構上**成立的：每一個條件、每一個等級都對著逐位元
    相同的 `y_ref` 與逐元素相同的 ε，而不是靠「種子相同所以結果應該相同」
    這個實作巧合。
    """
    emb = sd.encode_text(prompt).detach()
    emb_u = sd.encode_text("").detach()
    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    kw = dict(strength=cfg["strength"], guidance_scale=cfg["guidance"],
              emb_uncond=emb_u)
    out = []
    for i in range(n_seeds):
        s = cfg["seed"] + EVAL_SEED_OFFSET + i
        nz = sd.sample_edit_noise(torch.empty(lat, device=x01.device), seed=s)
        out.append((s, nz, sd.sdedit(x01, emb, nz, cfg["n_edit"], **kw)))
    return out


@torch.no_grad()
def evaluate(sd, suite, refs, xa, prompt, cfg):
    """對已算好的參照，逐種子評一個條件。"""
    emb = sd.encode_text(prompt).detach()
    emb_u = sd.encode_text("").detach()
    kw = dict(strength=cfg["strength"], guidance_scale=cfg["guidance"],
              emb_uncond=emb_u)
    rows = []
    for s, nz, y_ref in refs:
        y_def = sd.sdedit(xa, emb, nz, cfg["n_edit"], **kw)
        m = suite.full(y_ref, y_def, prompt=prompt)
        rows.append({"eval_seed": s,
                     **{f"edit_{k}": v for k, v in m.items()}})
    return rows


def main():
    ap = argparse.ArgumentParser(description="沿射線縮放，量代價-效果曲線")
    ap.add_argument("--run", default="runs/gate_suppress")
    ap.add_argument("--image", default="horse_00")
    ap.add_argument("--site", default="PF")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--targets", default=",".join(str(v) for v in TARGETS),
                    help="曲線要取值的 LPIPS。指定目標而非縮放係數，"
                         "兩個 site 的曲線才落在相同的 x 座標上")
    ap.add_argument("--eval_seeds", type=int, default=8,
                    help="配對分析說要分辨最佳化與隨機需要 n ≈ 7–10"
                         "（LEDGER 1.23），故預設 8 而非閘門的 5")
    ap.add_argument("--warp_resample", default=None,
                    choices=[None, "bilinear", "bicubic"],
                    help="覆寫 protocol.json 的值。舊的 run 沒有記這一欄時"
                         "必須明確指定，腳本不猜")
    ap.add_argument("--no_random", action="store_true",
                    help="不跑隨機對照條件。**只在時間不足時用**——沒有它，"
                         "曲線的高度無從解讀（LEDGER 1.18）")
    args = ap.parse_args()

    run = ROOT / args.run
    out = ROOT / (args.out or f"{args.run}_ray")
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    sd = SDWrapper(args.model)
    suite = MetricSuite(device=device)

    # 攻擊方設定由該 run 的 protocol.json 帶入，不重新指定——重新指定就會
    # 有兩份設定，而它們遲早會不一致
    proto = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    cfg = {k: proto[k] for k in ("strength", "guidance", "n_edit", "seed")}
    sm = {r["arm"]: r for r in csv.DictReader(
        open(run / "summary.csv", encoding="utf-8"))
        if r["image"] == args.image and r["site"] == args.site}
    if "opt" not in sm:
        raise SystemExit(f"{run/'summary.csv'} 沒有 {args.image}/{args.site} 的 opt 列")
    prompt = sm["opt"]["prompt"]

    x01 = load_png(run / f"{args.image}__orig.png", device)
    x_def = load_png(run / f"{args.image}__{args.site}__def.png", device)
    delta = x_def - x01
    build = exact_ray(run, args.image, args.site, x01, device,
                      resample=args.warp_resample)
    if build is None:
        def build(k, _d=delta, _x=x01):
            return (_x + k * _d).clamp(0, 1)
    save_json({**vars(args), **cfg, "prompt": prompt}, out / "protocol.json")
    print(f"=== {args.image} / site {args.site} / {prompt!r} ===", flush=True)
    print(f"  原解 LPIPS {float(suite.pairwise(x01, x_def)['lpips']):.4f}"
          f"　目標 {args.targets}", flush=True)
    exact = (run / f"{args.image}__{args.site}__phi.pt").exists() and args.site == "S"
    print(f"  縮放方式：{'精確（直接縮放位移場 φ）' if exact else '影像空間'}",
          flush=True)
    if args.site != "PF" and not exact:
        print("  [注意] site 非加性且沒有 __phi.pt，影像空間的縮放只是一階"
              "近似（見模組 docstring）", flush=True)

    t_ref = time.perf_counter()
    refs = reference_edits(sd, x01, prompt, cfg, args.eval_seeds)
    print(f"  {len(refs)} 個參照編輯（全部等級與條件共用），"
          f"{time.perf_counter() - t_ref:.0f}s", flush=True)

    res_path, sum_path = out / "results.csv", out / "summary.csv"
    unreachable = []
    for target in [float(v) for v in args.targets.split(",")]:
        try:
            _, got, k = solve_k(lpips_against(suite, x01), build, target)
        except ValueError as e:
            # 到不了是結果不是錯誤：某個參數化就是產生不出那麼大的可辨失真
            print(f"  [LPIPS {target}] 跳過——{e}", flush=True)
            unreachable.append(target)
            continue
        print(f"  [LPIPS {target}] k = {k:.4f} → 實際 {got:.4f}", flush=True)
        xk = build(k)
        arms = [("opt", xk)]
        if not args.no_random:
            pk = suite.pairwise(x01, xk)["lpips"]
            xr, _ = random_control(suite, x01, pk,
                                   seed=cfg["seed"] + int(k * 100))
            arms.append(("rand", xr))
        save_image(xk, out / f"{args.image}__{args.site}__lpips{target:g}.png")
        for arm, xa in arms:
            t0 = time.perf_counter()
            ev = evaluate(sd, suite, refs, xa, prompt, cfg)
            n = len(ev)
            dsig = [r["edit_siglip_b"] - r["edit_siglip_a"] for r in ev]
            dniq = [r["edit_niqe_b"] - r["edit_niqe_a"] for r in ev]
            ms, ss = st.mean(dsig), st.pstdev(dsig)
            mn, sn = st.mean(dniq), st.pstdev(dniq)
            p = suite.pairwise(x01, xa)
            srow = {
                "image": args.image, "site": args.site, "arm": arm,
                "target_lpips": target, "scale": round(k, 5),
                "prompt": prompt, "n_seeds": n,
                "pert_lpips": p["lpips"], "pert_linf": p["linf"],
                "pert_psnr": p["psnr"],
                "dsiglip_mean": ms, "dsiglip_sd": ss,
                "dniqe_mean": mn, "dniqe_sd": sn,
                "semantic_fail": bool(ms < 0 and abs(ms) > ss),
                **{f"edit_{c}": st.mean(r[f"edit_{c}"] for r in ev)
                   for c in TABLE1},
                "eval_seconds": round(time.perf_counter() - t0, 1),
            }
            append_csv(res_path, [{"image": args.image, "site": args.site,
                                   "arm": arm, "target_lpips": target, **r}
                                  for r in ev])
            append_csv(sum_path, [srow])
            print(f"    {arm:<5} LPIPS {p['lpips']:.4f}  "
                  f"Δsiglip {ms:+.5f} ± {ss:.5f}  Δniqe {mn:+.3f}  "
                  f"編輯LPIPS {srow['edit_lpips']:.4f}  "
                  f"語意失敗={srow['semantic_fail']}", flush=True)

    print(f"\n完成。{sum_path}", flush=True)
    print("讀法：把 (pert_lpips, dsiglip_mean) 畫成曲線，opt 與 rand 兩條。"
          "兩個 site 的曲線只要在某個 LPIPS 區間重疊就能比較，"
          "不需要任何一次執行剛好落在指定的點上。")


if __name__ == "__main__":
    main()
