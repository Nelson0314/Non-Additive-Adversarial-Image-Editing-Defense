"""τ=0.28 的可行域預估：把既有擾動放大到文獻預算，量另外兩道約束。

E31 的 gate（規格 §8）要求每一格的綁定者都是 LPIPS hinge。τ=0.10 時
`local_acutance_dev` 實測 0.0185（門檻 0.04）、`local_chroma_bias` 0.6475
（門檻 0.8），都還沒啟動。τ 拉到 0.28 之後它們會不會翻上來，決定 R1 的校準
會不會通過，而 R1 要花雲端 GPU。

本腳本用零 GPU 成本先給出數量級答案：取 `runs/e29c_P_tau0.10` 已有的擾動
δ = x_def − x_base，等比放大 k 倍直到 LPIPS(x_base + kδ, x) ≈ 目標，在那一點
量 acut、chroma、RMS、L∞ 與超過 16/255 的像素比例。

**這是預估不是量測。** 最佳化在 τ=0.28 下會找到與「把 τ=0.10 的解放大」不同
的解——放大是沿著既有解的射線走，最佳化可以離開那條射線。本腳本只回答
「三道約束在該量級上是否還有可行域」，不取代 R1 的實測。

執行：python scripts/p13_budget_probe.py
成本：本機，一分鐘內。不需要 SD 權重。
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.chroma import local_chroma_bias
from src.metrics.local_acutance import local_acutance_dev

# 兩道門檻的現行值，與 LossConfig 的預設一致。寫在此處是為了讓 CSV 能直接
# 帶出「有沒有超過」的判定，不留給讀者心算。
TAU_ACUT = 0.04
TAU_CHROMA = 0.8


def find_scale(measure, target, lo=1.0, hi=64.0, tol=1e-3, max_iter=60):
    """對單調遞增的 `measure` 二分搜尋，使其達到 `target`。

    上界不足時拋出而非回傳 hi：靜默回傳會讓呼叫端以為達到了 target，
    於是後續量到的 acut／chroma 對應的其實是別的 LPIPS，整份預估失效。
    """
    if measure(lo) >= target:
        return lo
    if measure(hi) < target:
        raise ValueError(
            f"上界 hi={hi} 只達到 {measure(hi):.4f}，未達 target={target}。"
            "請提高 hi，不可回傳上界充數"
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if measure(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="runs/e29c_P_tau0.10/car_00__P__r16,"
                                       "runs/e29c_C_tau0.10/car_00__C__r32")
    ap.add_argument("--out", default="runs/p13_budget_probe")
    ap.add_argument("--targets", default="0.10,0.15,0.20,0.28")
    args = ap.parse_args()

    import piq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    (out / "img").mkdir(parents=True, exist_ok=True)
    lpips = piq.LPIPS().to(device)

    def load(cell: Path, name):
        img = Image.open(cell / name).convert("RGB")
        return T.ToTensor()(img).unsqueeze(0).to(device)

    rows = []
    cell_specs = [(c.strip(), False) for c in args.cells.split(",")]
    # 參照臂：等 LPIPS 的純加性高斯雜訊。沒有它就無法分辨兩件事——
    # 「這個解在該預算上買了鈍化／色偏」與「τ_acut 與 τ_chroma 是在低預算下
    # 定的，任何達到該 LPIPS 的擾動都會超標」。E20 的四臂探針就是為了同一個
    # 區別而做的；此處把「預算」這一軸加進去。
    cell_specs.append((cell_specs[0][0], True))

    for cell_rel, as_noise in cell_specs:
        cell = ROOT / cell_rel.strip()
        tag = cell.parent.name + ("__isonoise" if as_noise else "")
        x = load(cell, "orig.png")
        x_base = load(cell, "baseline_phi0.png")
        if as_noise:
            # 白高斯，逐像素獨立、無空間結構。單位標準差，實際強度由 k 決定。
            g = torch.Generator(device="cpu").manual_seed(20260802)
            delta = torch.randn(x_base.shape, generator=g).to(device)
        else:
            delta = load(cell, "defended.png") - x_base
        print(f"[p13] {tag}  δ 的 L∞={float(delta.abs().max()):.4f}", flush=True)

        def build(k):
            return (x_base + k * delta).clamp(0, 1)

        # 純雜訊臂的 δ 是單位標準差，達到相同 LPIPS 所需的 k 比防禦解小得多，
        # 故下界要放到 0 而非 1——1.0 就已經遠超過目標，會直接回傳下界。
        lo = 0.0 if as_noise else 1.0
        for target in [float(t) for t in args.targets.split(",")]:
            k = find_scale(lambda k: float(lpips(build(k), x)), target, lo=lo)
            xd = build(k)
            d = (xd - x_base).abs()
            acut = float(local_acutance_dev(x_base, xd))
            chroma = float(local_chroma_bias(x_base, xd))
            rows.append({
                "cell": tag,
                "target_lpips": target,
                "scale": round(k, 4),
                "lpips": float(lpips(xd, x)),
                "acut": acut,
                "chroma": chroma,
                "rms": float((d ** 2).mean().sqrt()),
                "linf": float(d.max()),
                "frac_gt_16_255": float((d > 16 / 255).float().mean()),
                "psnr": float(piq.psnr(xd, x, data_range=1.0)),
                "acut_over_tau": acut > TAU_ACUT,
                "chroma_over_tau": chroma > TAU_CHROMA,
            })
            save_image(xd, out / "img" / f"{tag}__{target}.png")
            r = rows[-1]
            print(f"  target={target}  k={k:.3f}  acut={acut:.4f}"
                  f"{' ⚠' if r['acut_over_tau'] else ''}  chroma={chroma:.4f}"
                  f"{' ⚠' if r['chroma_over_tau'] else ''}  "
                  f"rms={r['rms']:.4f}  psnr={r['psnr']:.2f}", flush=True)
        save_image(x, out / "img" / f"{tag}__orig.png")

    with open(out / "probe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tags = sorted({r["cell"] for r in rows})
    html = [
        "<!doctype html><meta charset='utf-8'><title>E31 預算探針</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;"
        "padding:16px}img{width:260px;display:block}"
        "td{padding:6px;text-align:center;vertical-align:top}"
        "small{font-size:11px;color:#aaa}th{color:#8cf;text-align:left}</style>",
        "<h1>把 τ=0.10 的解沿射線放大到各個 LPIPS 預算</h1>",
        "<p>這是沿既有解的射線放大，<b>不是</b>重新最佳化。用途是判斷 "
        f"acut（門檻 {TAU_ACUT}）與 chroma（門檻 {TAU_CHROMA}）在該量級是否"
        "翻上來成為綁定者，以及 τ=0.28 在人眼上是什麼樣子。</p>",
    ]
    for tag in tags:
        html.append(f"<h2>{tag}</h2><table><tr>")
        html.append("<td><img src='img/%s__orig.png'><small>原圖</small></td>"
                    % tag)
        for r in [r for r in rows if r["cell"] == tag]:
            html.append(
                "<td><img src='img/%s__%s.png'><small>LPIPS=%.3f (k=%.2f)<br>"
                "acut=%.4f%s<br>chroma=%.4f%s<br>PSNR=%.2f dB<br>"
                "RMS=%.4f</small></td>"
                % (tag, r["target_lpips"], r["lpips"], r["scale"], r["acut"],
                   " ⚠" if r["acut_over_tau"] else "", r["chroma"],
                   " ⚠" if r["chroma_over_tau"] else "", r["psnr"], r["rms"]))
        html.append("</tr></table>")
    (out / "compare.html").write_text("\n".join(html), encoding="utf-8")
    print(f"[p13] 寫出 {out / 'probe.csv'} 與 {out / 'compare.html'}")


if __name__ == "__main__":
    main()
