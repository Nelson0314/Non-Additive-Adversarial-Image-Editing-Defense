"""各族的防禦圖實際把**影像引導項**壓到哪裡。**不訓練，只讀已存的圖。**

問的是什麼
────────────────────────────────────────────────────────────────────
`scripts/latent_norm_probe.py` 已經量過各族到達的 `‖E(x_def)‖₂`，結果是
**位移不是它的函數**——同一批裡隨機色彩網格的範數比原圖還高（1.067×），
位移卻高於把範數壓到 0.667× 的最佳化版。所以「損失只差一點點就跨過界線」
這個讀法沒有立足點。

本探針換一個量：`image_guidance` 損失

    L_ig(x) = E_{t,eps} || eps(z_t, E_img(x), null) - eps(z_t, 0, null) ||^2

它量的是 IP2P 取樣式裡 `s_I * [eps(z_t, c_I, null) - eps(z_t, 0, null)]`
這一項還剩多少，也就是**機制本身**。兩族都用同一組固定抽樣評估過，但
`results.csv` 的 `best_eval` 只存在於「用該損失訓練過」的格；隨機對照、
`latent_norm` 的格、DCT-Shield 都沒有這一欄。要把它們放進同一張圖上，
只能對**已存的防禦圖**直接量一次。

三件必須對齊訓練端的事
────────────────────────────────────────────────────────────────────
1. **固定抽樣的設定要與訓練端逐項相同**，否則量到的不是同一條軸。訓練端是
   `scripts/ip2p_run.py:1471` 的 `fn.make_fixed(args.eval_draws, args.eval_seed)`，
   預設 8 抽樣、seed 99991。本檔的預設值相同，且逐列寫進 CSV。
2. **`--ig-zt` 沒有預設值。** 兩個 `z_t` 的抽法都是近似，沒有一個是「對的」，
   按 CLAUDE.md「查不到的參數設為必填」。已存的批次全部是 `diffuse_src`。
3. **`x_clean` 一律是原圖。** `diffuse_src` 的 `z_t` 錨在乾淨影像上；改用
   防禦圖當錨會讓每一族各自站在不同的軌跡上，跨族比較立刻失效。

順帶把 `‖E(x_def)‖₂` 也量出來（一次 VAE 編碼，幾乎免費），使同一張表上
兩個候選代理量可以直接對照。

用法：

    python scripts/ig_probe.py --out runs/ig_probe/results.csv --ig-zt diffuse_src
        --images task_attr_mod_color_11699 task_attr_mod_color_6205
        --entry phase_ig=runs/ip2p_ig_loss/ig_eot
        --entry color_rand=runs/ip2p_color/grid_rand_r010
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.defense.image_guidance_loss import ZT_MODES, make_image_guidance_loss  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", action="append", required=True,
                    metavar="LABEL=DIR",
                    help="一族的防禦圖目錄。條件名由該目錄的 results.csv 讀出")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ig-zt", choices=ZT_MODES, default=None,
                    help="**必填**。兩個 z_t 的抽法都是近似，故不設預設值")
    ap.add_argument("--ig-t-min", type=int, default=1)
    ap.add_argument("--ig-t-max", type=int, default=1000)
    ap.add_argument("--eval-draws", type=int, default=8,
                    help="固定抽樣的組數。預設與 ip2p_run.py 相同")
    ap.add_argument("--eval-seed", type=int, default=99991,
                    help="固定抽樣的種子。預設與 ip2p_run.py 相同")
    return ap


def check_args(args) -> None:
    """把「補錯了不會報錯」的四種情形擋在載入權重之前。"""
    if args.ig_zt is None:
        raise SystemExit(
            "--ig-zt 必填：兩個 z_t 的抽法都是近似，填錯不會有症狀，"
            f"只會量到另一條軸。已存的批次全部是 diffuse_src。{ZT_MODES}")
    if not 1 <= args.ig_t_min <= args.ig_t_max:
        raise SystemExit(
            f"需要 1 <= --ig-t-min <= --ig-t-max，收到 "
            f"{args.ig_t_min}／{args.ig_t_max}")
    if args.eval_draws < 1:
        raise SystemExit(f"--eval-draws 必須為正整數，收到 {args.eval_draws}")
    for spec in args.entry:
        label, sep, d = spec.partition("=")
        if not sep or not label or not d:
            raise SystemExit(f"--entry 的格式是 LABEL=DIR，收到 {spec!r}")
        if not (Path(d) / "results.csv").exists():
            raise SystemExit(f"{d} 底下沒有 results.csv，無法取得條件名")


def defense_images(d: Path, name: str):
    """回傳該目錄裡這張影像的 (條件名, 防禦圖路徑, results 那一列)。

    一個目錄可能有多個條件（例如同一批跑了 `color_grid` 與
    `color_grid_rand`），全部回傳；缺圖的條件印出來後跳過，**不靜默略過**。
    """
    with (d / "results.csv").open(encoding="utf-8-sig", newline="") as fh:
        recs = [r for r in csv.DictReader(fh) if r["image"] == name]
    out = []
    for r in recs:
        png = d / f"{name}__{r['condition']}__def.png"
        if not png.exists():
            print(f"[skip] 缺 {png}", flush=True)
            continue
        out.append((r["condition"], png, r))
    return out


def main() -> None:
    args = build_parser().parse_args()
    check_args(args)

    from src.models.ip2p import IP2PWrapper

    ip2p = IP2PWrapper(dtype=torch.float32)
    dev = ip2p.device

    rows = []
    for name in args.images:
        orig = args.data / name / f"{name}.png"
        x = load_image_tensor(orig, dev, size=RESOLUTION)

        # 每張影像各建一次：`diffuse_src` 的 z_src 錨在**這張原圖**上。
        loss = make_image_guidance_loss(
            ip2p, zt_mode=args.ig_zt, x_clean=x,
            t_min=args.ig_t_min, t_max=args.ig_t_max)
        fixed = loss.make_fixed(args.eval_draws, args.eval_seed)

        with torch.no_grad():
            l0 = float(fixed(x))
            n0 = float(ip2p.encode_image(x).flatten().norm(p=2))
        common = {"image": name, "ig_zt": args.ig_zt,
                  "ig_t_min": args.ig_t_min, "ig_t_max": args.ig_t_max,
                  "eval_draws": args.eval_draws, "eval_seed": args.eval_seed}
        rows.append({**common, "label": "original", "condition": "",
                     "ig_loss": round(l0, 8), "ig_ratio_to_original": 1.0,
                     "latent_norm": round(n0, 4),
                     "latent_ratio_to_original": 1.0,
                     "fid_dists": "", "fid_lpips": "", "edit_lpips": "",
                     "best_eval_recorded": "", "loss_trained": ""})
        print(f"{name[15:]:26s}{'original':26s}L_ig={l0:.6f}  ‖E‖={n0:.2f}",
              flush=True)

        for spec in args.entry:
            label, _, d = spec.partition("=")
            for cond, png, r in defense_images(Path(d), name):
                xd = load_image_tensor(png, dev, size=RESOLUTION)
                with torch.no_grad():
                    lv = float(fixed(xd))
                    nv = float(ip2p.encode_image(xd).flatten().norm(p=2))
                rows.append({
                    **common, "label": label, "condition": cond,
                    "ig_loss": round(lv, 8),
                    "ig_ratio_to_original": round(lv / l0, 6),
                    "latent_norm": round(nv, 4),
                    "latent_ratio_to_original": round(nv / n0, 4),
                    "fid_dists": r.get("fid_dists", ""),
                    "fid_lpips": r.get("fid_lpips", ""),
                    "edit_lpips": r.get("edit_lpips", ""),
                    # 訓練端記下的固定評估值。與本欄的 `ig_loss` 不同是預期
                    # 的：`best_eval` 是評估序列的最小值，而存下的 `x_def`
                    # 是**最後一步**的迭代（`param_pgd.py:853`）。收斂的格上
                    # 兩者一致，沒收斂的格不一定。
                    "best_eval_recorded": r.get("best_eval", ""),
                    "loss_trained": r.get("loss", ""),
                })
                print(f"{name[15:]:26s}{label + '/' + cond:26s}"
                      f"L_ig={lv:.6f} ({lv / l0:.3f}× 原圖)  "
                      f"‖E‖={nv:.2f} ({nv / n0:.3f}×)  "
                      f"DISTS={r.get('fid_dists', '')} "
                      f"位移={r.get('edit_lpips', '')}", flush=True)
                write_csv(args.out, rows)

    write_csv(args.out, rows)
    print(f"\n寫出 {args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
