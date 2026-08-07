"""ST-LPIPS 能不能當加性與非加性的共同約束。

問題
──────────────────────────────────────────────────────────────────────
第一階段以 `LPIPS(x_def, x) = τ` 作為共同貨幣，把每個條件縮放到同一個
失真預算再比較。但在 τ=0.20 上實測到：

    非加性（位移場）  PSNR 21.7–23.8 dB   SSIM 0.762–0.857   DISTS 0.068–0.074
    加性（三篇）      PSNR 38.1–43.8 dB   SSIM 0.984–0.994   DISTS 0.017–0.028

**每一個其他指標都把非加性判重數倍，且零重疊。** 也就是說「同一個 LPIPS」
並不對應「同樣的可辨失真」，該軸對非加性不利。

Ghildyal & Liu（ECCV 2022，arXiv:2207.13686）指出 LPIPS 對人眼看不見的
微小錯位過度敏感，並以 anti-aliasing 濾波與 pooling 的改造做出對位移寬容、
同時仍與人類判斷一致的版本（ST-LPIPS）。本腳本檢查它在**我們自己的資料**
上會不會把兩類拉到同一個尺度。

判準（事先宣告）
──────────────────────────────────────────────────────────────────────
在原本 LPIPS 固定為 0.20 的那一批 `x_def` 上重算 ST-LPIPS：

- **可採用**：非加性與加性的區間**重疊**，或跨類別倍率 < 1.5×。
  那表示它把兩類放在同一個尺度上，可以取代 LPIPS 作為共同約束。
- **不可採用**：仍分開 ≥ 2×。那表示它解的不是我們這個問題
  （它針對的是**不可察覺**的位移，而 τ=0.20 的位移場是看得見的），
  共同約束只能走人眼 2AFC。

一併輸出原版 LPIPS 作為對照——若兩者在同一批影像上給出相同的分離型態，
差異就不是來自 ST-LPIPS 的改造而是來自這兩類失真本身。

用法
──────────────────────────────────────────────────────────────────────
    python scripts/diag_stlpips.py --batch runs/v14_merged
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch
from PIL import Image

NONADD = ("N1", "N2", "N3", "R")
ADD = ("photoguard_c", "mist", "dia_r")
CONDS = NONADD + ADD
TAUS = ("0.05", "0.1", "0.2", "0.35")


def load(p: Path, device) -> torch.Tensor:
    """讀成 (1,3,H,W)、值域 [-1,1]——ST-LPIPS 與 LPIPS 的慣例。"""
    a = torch.frombuffer(
        bytearray(Image.open(p).convert("RGB").tobytes()), dtype=torch.uint8)
    w, h = Image.open(p).size
    x = a.reshape(1, h, w, 3).permute(0, 3, 1, 2).float() / 255.0
    return (x * 2.0 - 1.0).to(device)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, default=Path("runs/v14_merged"))
    ap.add_argument("--tau", default="0.2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from stlpips_pytorch import stlpips

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # `variant="shift_tolerant"` 是論文的模型；alex 是其主要骨幹。
    st = stlpips.LPIPS(net="alex", variant="shift_tolerant",
                       verbose=False).to(dev).eval()
    # 對照取**專案段 2 實際使用的那一個** LPIPS（`piq`，VGG 骨幹），
    # 而不是 stlpips 套件附的 vanilla——後者根本沒有隨套件發布
    # （`weights/vST0.0/` 只有 alex/vgg 的 shift_tolerant）。用實際在跑的
    # 那一個當對照，比較的才是「換掉共同貨幣會怎樣」而不是兩個第三方實作。
    import piq
    _piq = piq.LPIPS().to(dev)

    def base(a, b):
        return _piq((a + 1) / 2, (b + 1) / 2)

    imgs = sorted({p.parent.name for c in CONDS
                   for p in (args.batch / c).glob("*/orig.png")})
    if not imgs:
        raise SystemExit(f"{args.batch} 底下找不到任何 orig.png")
    print(f"影像 {imgs}　τ={args.tau}　裝置 {dev}\n")

    rows = []
    for cond in CONDS:
        for im in imgs:
            d = args.batch / cond / im
            xo, xd = d / "orig.png", d / f"x_def_tau{args.tau}.png"
            if not xd.exists():
                continue
            a, b = load(xo, dev), load(xd, dev)
            with torch.no_grad():
                v_st = float(st(a, b))
                v_ln = float(base(a, b))
            rows.append({"condition": cond, "image_id": im, "tau": args.tau,
                         "kind": "非加性" if cond in NONADD else "加性",
                         "stlpips": round(v_st, 5),
                         "lpips_alex": round(v_ln, 5)})
            print(f"  {cond:<14}{im:<10}ST-LPIPS={v_st:.4f}  "
                  f"LPIPS(alex)={v_ln:.4f}", flush=True)

    if not rows:
        raise SystemExit("沒有任何 x_def 可讀——段 2 的產物是否已回收？")

    out = args.out or (args.batch / f"stlpips_tau{args.tau}.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n寫出 {out}（{len(rows)} 列）\n")
    print("=== 判定 ===")
    for key, name in (("stlpips", "ST-LPIPS"), ("lpips_alex", "LPIPS(alex)")):
        na = [r[key] for r in rows if r["kind"] == "非加性"]
        ad = [r[key] for r in rows if r["kind"] == "加性"]
        if not na or not ad:
            continue
        overlap = not (max(na) < min(ad) or max(ad) < min(na))
        ratio = statistics.fmean(na) / max(statistics.fmean(ad), 1e-9)
        verdict = ("可採用" if (overlap or ratio < 1.5) else "不可採用")
        print(f"  {name:<12} 非加性 [{min(na):.4f}, {max(na):.4f}]  "
              f"加性 [{min(ad):.4f}, {max(ad):.4f}]  "
              f"均值比 {ratio:.2f}×  區間{'重疊' if overlap else '分離'}  "
              f"→ {verdict}")
    print("\n判準見本檔 docstring：重疊或 < 1.5× 可採用，≥ 2× 不可採用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
