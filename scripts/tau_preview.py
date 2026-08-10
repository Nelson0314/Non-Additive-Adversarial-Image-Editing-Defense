"""τ 掃描：把同一個方向縮放到各個失真預算，逐 τ 存圖供人眼挑工作點。

為什麼需要它
──────────────────────────────────────────────────────────────────────
τ_train 是**由人眼定的**（DEC-001、DEC-007 都是這樣定的），因為 LPIPS 的
數值與「看不看得出來」之間沒有跨參數化、跨威脅模型的固定換算。inpainting
的失真型態與 img2img 不同，故 DEC-010 明訂不可沿用 0.25／0.30，要重新掃。

本腳本不重跑訓練。它讀段 1 已經產生的 φ（本批用 `Ra`——同一個 apa 參數化、
方向取高斯隨機、**同樣跑過階段一的保真對齊**，見 MTH-Ra），再以
`ray_scale.solve_k` 對縮放係數二分，使 `LPIPS(x_def, x)` 命中每一個目標 τ。
用 Ra 而不是 apa 的理由：要看的是「τ 這個量在這條參數化上長什麼樣」，
那與損失選什麼無關，而 Ra 不必最佳化，一次訓練就能掃完整條軸。

τ 的下界不是選的，是量出來的
──────────────────────────────────────────────────────────────────────
site apa 走生成路徑，`x_def = decode(...encode(x)...)`，故其最小失真是
**該影像的 VAE 重建下限**（`calib/recon_floor.csv`）。低於下限的 τ 在結構上
不可達，`solve_k` 會拋出——那是關於該方向的結果，不是錯誤，本腳本照實記錄
為 `unreachable` 而不是靜默取最接近的值。

用法
──────────────────────────────────────────────────────────────────────
    python scripts/tau_preview.py --batch runs/ip4_taupreview \\
        --images horse_00 man_00 bird_03 --condition Ra \\
        --taus 0.16 0.20 0.25 0.30 0.35 0.40 0.50 \\
        --masks data/lo_masks --prompt-index 1 --device cuda:0

產出 `<batch>/tau_preview/`：逐 (影像, τ) 一張 `x_def`、一張與原圖的差分放大圖、
`tau_preview.csv`（LPIPS／PSNR／銳利度比／k／是否可達）與 `compare.html`。
**每一格都要有影像可看**——`compare.html` 是主要產出物（CLAUDE.md）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch                                                 # noqa: E402

from src.experiment import executors                         # noqa: E402
from src.metrics.ray_scale import solve_k                    # noqa: E402


def diff_map(a: torch.Tensor, b: torch.Tensor, gain: float = 6.0
             ) -> torch.Tensor:
    """差分放大圖。逐通道取絕對差後乘 `gain` 截斷。

    非加性方法的擾動在原圖上常常看不出來，卻在差分圖上呈現結構（位移場是
    邊緣的雙線、apa 是低頻的塊）。那個結構本身就是要給人看的證據——只報
    LPIPS 會漏掉「同一個數值下兩種方法的失真型態完全不同」。
    """
    return ((a - b).abs() * gain).clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--condition", default="Ra",
                    help="讀哪一個條件的 φ。預設 Ra（同參數化隨機對照）")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    ap.add_argument("--metric", default="lpips", choices=["lpips", "dists"],
                    help="射線縮放要命中哪一個量。**LPIPS 排不出人眼順序**"
                         "（2026-08-10 實測：毀掉的 horse_00 其 LPIPS 0.179 "
                         "反而低於可接受的 horse_03 0.198），DISTS 排得出"
                         "（0.150 對 0.091）。理由是 DISTS 把結構與紋理分開量，"
                         "而本專案的失效模式是結構被改寫")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="二分的容差，與 rayscale 段一致")
    ap.add_argument("--out", type=Path, default=None)
    args, rest = ap.parse_known_args()

    # `build_resources` 需要完整的批次設定（模型、遮罩、prompt 索引……）。
    # 這裡不重建一份平行的參數表——那是兩份會分岔的清單——而是把未知旗標
    # 原樣轉交給 `run_stage` 的 parser。
    import run_stage as rs

    argv = ["calib", "--batch", args.batch.name,
            "--runs-root", str(args.batch.parent),
            "--images", *args.images, *rest]
    rs_args = rs.build_parser().parse_args(argv)
    rs.resolve_thresholds(rs_args, verbose=False)
    res = rs.build_resources(rs_args, args.batch)

    out_dir = args.out or (args.batch / "tau_preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_id in args.images:
        entry = res.image(image_id)
        phi_path = args.batch / args.condition / image_id / "phi.pt"
        if not phi_path.exists():
            raise SystemExit(
                f"{phi_path} 不存在。先跑 "
                f"`run_stage.py train --batch {args.batch.name} "
                f"--conditions {args.condition}`")
        payload = executors.load_phi(phi_path)

        executors.save_image(entry.x01, out_dir / f"{image_id}__orig.png")
        base = executors.materialize(payload, res, entry, k=0.0)
        m0 = res.suite.pairwise(entry.x01, base)
        print(f"[{image_id}] φ=0 的重建下限 {args.metric}="
              f"{m0[args.metric]:.4f}  LPIPS={m0['lpips']:.4f} "
              f"PSNR={m0['psnr']:.2f} dB", flush=True)
        executors.save_image(base, out_dir / f"{image_id}__floor.png")
        rows.append({"image_id": image_id, "tau": 0.0, "reachable": True,
                     "k": 0.0, "lpips": round(m0["lpips"], 4),
                     "psnr": round(m0["psnr"], 2),
                     "note": "φ=0，即 VAE 重建下限"})

        for tau in args.taus:
            if tau < m0[args.metric] - args.tol:
                # 低於下限：不是最佳化失敗，是這條參數化構不到。
                print(f"[{image_id}] 目標={tau:.4f} 低於重建下限 "
                      f"{m0[args.metric]:.4f}，結構上不可達", flush=True)
                rows.append({"image_id": image_id, "tau": tau,
                             "reachable": False, "k": "", "lpips": "",
                             "psnr": "", "note": "低於 VAE 重建下限"})
                continue
            try:
                x_def, got, k = solve_k(
                    lpips_fn=lambda y: res.suite.pairwise(entry.x01, y)[args.metric],
                    build=lambda kk: executors.materialize(
                        payload, res, entry, k=kk),
                    target=tau, tol=args.tol)
            except ValueError as exc:
                print(f"[{image_id}] τ={tau:.2f} 二分未達標：{exc}", flush=True)
                rows.append({"image_id": image_id, "tau": tau,
                             "reachable": False, "k": "", "lpips": "",
                             "psnr": "", "note": str(exc)[:120]})
                continue

            m = res.suite.pairwise(entry.x01, x_def)
            executors.save_image(x_def, out_dir / f"{image_id}__tau{tau:g}.png")
            executors.save_image(diff_map(x_def, entry.x01),
                                 out_dir / f"{image_id}__tau{tau:g}_diff.png")
            rows.append({"image_id": image_id, "tau": tau, "reachable": True,
                         "k": round(k, 4), "lpips": round(got, 4),
                         "psnr": round(m["psnr"], 2),
                         "acut": round(m.get("acutance_ratio", float("nan")), 4),
                         "dists": round(m["dists"], 4),
                         "metric": args.metric,
                         "note": ""})
            print(f"[{image_id}] 目標 {args.metric}={tau:.4f}  k={k:.3f}  "
                  f"達成={got:.4f}  LPIPS={m['lpips']:.4f}  "
                  f"DISTS={m['dists']:.4f}  PSNR={m['psnr']:.2f} dB",
                  flush=True)

    executors.write_csv(out_dir / "tau_preview.csv", rows)
    page = render(rows, args.images, args.taus, out_dir)
    print(f"\n表：{out_dir / 'tau_preview.csv'}\n比對頁：{page}")


def render(rows, images, taus, out_dir: Path) -> Path:
    """逐影像一列、逐 τ 一欄的比對頁。原圖與 φ=0 各自佔一欄作為錨點。"""
    by = {(r["image_id"], r["tau"]): r for r in rows}
    css = """body{background:#16171b;color:#e8e8ea;font:13px/1.5 system-ui,
sans-serif;margin:24px}h1{font-size:18px}table{border-collapse:collapse}
td,th{border:1px solid #2e3038;padding:6px;text-align:center;vertical-align:top}
th{background:#1e2027;position:sticky;top:0}img{display:block;width:190px;
height:190px;object-fit:contain;background:#000}
.m{font:11px ui-monospace,monospace;color:#9aa0aa}.bad{color:#ff8f6b}
.hint{color:#9aa0aa;max-width:60em}"""
    h = ["<!-- 由 scripts/tau_preview.py 產生 -->",
         f"<style>{css}</style>", "<h1>τ 掃描（inpainting，site apa）</h1>",
         "<p class=hint>挑一個<strong>看不太出來、但差分圖上確實有東西</strong>"
         "的 τ 作為 τ_train。上排是防禦圖，下排是與原圖的差分（×6）。"
         "φ=0 那一欄是 VAE 重建下限——低於它的 τ 在這條參數化上不可達，"
         "不是最佳化不夠好。</p>", "<table><tr><th>影像</th><th>原圖</th>",
         "<th>φ=0<br><span class=m>重建下限</span></th>"]
    for t in taus:
        h.append(f"<th>τ = {t:g}</th>")
    h.append("</tr>")
    for img in images:
        h.append(f"<tr><td>{img}</td>")
        h.append(f"<td><img src='{img}__orig.png'></td>")
        f = by.get((img, 0.0), {})
        h.append(f"<td><img src='{img}__floor.png'>"
                 f"<span class=m>LPIPS {f.get('lpips','—')}<br>"
                 f"PSNR {f.get('psnr','—')}</span></td>")
        for t in taus:
            r = by.get((img, t))
            if r is None or not r["reachable"]:
                note = (r or {}).get("note", "—")
                h.append(f"<td class=bad>不可達<br><span class=m>{note}</span></td>")
                continue
            h.append(
                f"<td><img src='{img}__tau{t:g}.png'>"
                f"<img src='{img}__tau{t:g}_diff.png'>"
                f"<span class=m>LPIPS {r['lpips']}<br>PSNR {r['psnr']}<br>"
                f"k {r['k']}</span></td>")
        h.append("</tr>")
    h.append("</table>")
    p = out_dir / "compare.html"
    p.write_text("\n".join(h), encoding="utf-8")
    return p


if __name__ == "__main__":
    main()
