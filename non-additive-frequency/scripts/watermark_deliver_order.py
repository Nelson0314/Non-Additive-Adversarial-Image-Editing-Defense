"""兩層防禦交付時誰先誰後：QIM 脆弱浮水印 × 量化交付。

要回答的問題
────────────────────────────────────────────────────────────────────
pipeline 是兩層的：

    干擾層   衣物載體上的補丁，`--deliver-jpeg QD` 交出去的**就是壓縮圖**，
             機制是「把解放在 JPEG 的量化格點上，攻擊方再壓一次接近恆等」。
    舉證層   `src/watermark/qim.py` 的脆弱浮水印，把訊息寫在 8×8 DCT 係數的
             QIM 格點上（定案工作點 Δ=64、`mid4`、`repeat=64`），
             用途是判斷「這張圖進過擴散自編碼器」。

**兩層都住在 8×8 DCT 係數上，而且都靠「把係數放到某個格點上」運作。**
那兩個格點不是同一個：JPEG 的格點由品質表決定、逐頻率不同，QIM 的格點是
固定的 Δ。誰先誰後因此不是排版問題，是兩個量化互相干涉的問題，而**目前
沒有任何資料**。

三個順序，各自壞在不同的地方（都要量，不能推論）：

| 順序 | 做什麼 | 先驗上會壞的地方 |
|---|---|---|
| `wm_then_jpeg` | 先嵌浮水印，再壓成 QD 交付 | JPEG 的量化把 QIM 的係數推離它的格點 → 位元讀不回來 |
| `jpeg_then_wm` | 先壓 QD，再嵌浮水印，**存 PNG** | QIM 把係數推離 JPEG 的格點 → 干擾層「攻擊方再壓一次接近恆等」的前提被破壞 |
| `jpeg_wm_jpeg` | 先壓 QD，嵌浮水印，再壓一次 QD 交付 | 第二次壓縮同時對付兩個格點，兩層都可能掉 |

四個讀數
────────────────────────────────────────────────────────────────────
| 欄 | 量什麼 |
|---|---|
| `bits_deliver` | 交付圖上的位元正確率——**舉證層還在不在** |
| `bits_jpeg75`／`bits_jpeg30`／`bits_blur1` | 攻擊方再處理一次之後的位元正確率 |
| `pert_cos`／`pert_norm` | 交付圖的殘差對**干擾層原本那個擾動方向**的餘弦與長度比——干擾層還剩多少 |
| `idem_qd`／`idem_j75` | **交付圖是不是攻擊方 JPEG 的不動點**——把交付圖再壓一次的 PSNR |
| `psnr_vs_def` | 這一整套交付相對於干擾層輸出多付了多少失真 |

`idem_*` 才是干擾層真正依賴的那個性質。量化交付的機制寫在
`runs/ip2p_deliver_jpeg/README.md`：「**JPEG 幾乎冪等**，同品質重壓只改
0.02–0.17% 的係數，所以交付壓縮圖等於把解放到量化格點上，攻擊方再壓一次
接近恆等」。QIM 把係數推到**它自己的** Δ 格點上，那與 JPEG 的品質表格點
不是同一組——所以「先壓再嵌」交出去的圖已經不在 JPEG 的格點上了，
而 `pert_cos` 這種對原方向的投影**看不出這件事**（那個位移相對於補丁本身
很小）。`idem_*` 直接問「再壓一次會變多少」。

**這一支不量「防禦有沒有效」。** 干擾層的內容在這裡是**隨機補丁**而不是
最佳化過的解——要問的是兩個量化格點互相干涉的程度，那與補丁裡長什麼樣子
無關，而用隨機內容讓這一支在 CPU 上幾秒鐘就跑得完、也不必動到卡。
效果的讀數在 `runs/ip2p_deliver_jpeg_patch/`。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.jpeg_codec import jpeg_roundtrip  # noqa: E402
from src.defense.patch_param import PatchRandomParam  # noqa: E402
from src.purify import ops  # noqa: E402
from src.watermark import qim  # noqa: E402

# `runs/watermark_qim/README.md` 的定案工作點。**不設預設值是 `qim.embed` 的
# 契約**（填錯的症狀與「浮水印被打掉」一模一樣），故在這裡明寫一次。
WM_DELTA = 64.0
WM_COEFFS = ((1, 2), (2, 1), (2, 2), (1, 3))
WM_REPEAT = 64
WM_BITS = 64


def load_image(path: Path) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    arr = np.asarray(Image.open(path).convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def make_defence(x01: torch.Tensor, seed: int) -> torch.Tensor:
    """干擾層的輸出：衣物大小的隨機補丁。**內容是隨機的，理由見檔頭。**"""
    h, w = x01.shape[-2:]
    m = torch.zeros(1, 1, h, w)
    m[..., : h // 3, :] = 1.0                 # 受保護主體：上三分之一
    p = PatchRandomParam(placement="complement", mask=m, init="random")
    p.reset(x01, seed)
    return p.render(x01).detach().clamp(0, 1)


def to_uint8_grid(x01: torch.Tensor) -> torch.Tensor:
    """存成 PNG 再讀回來會發生的事。**QIM 的契約是輸出落在 uint8 網格上**，
    不套這一步的話量到的是浮點中間值，而那不是交付出去的東西。"""
    return (x01.clamp(0, 1) * 255.0).round() / 255.0


def order_wm_then_jpeg(x_def, bits, qd):
    return jpeg_roundtrip(qim.embed(x_def, bits, delta=WM_DELTA,
                                    coeffs=WM_COEFFS, repeat=WM_REPEAT,
                                    verify=False), qd)


def order_jpeg_then_wm(x_def, bits, qd):
    return qim.embed(jpeg_roundtrip(x_def, qd), bits, delta=WM_DELTA,
                     coeffs=WM_COEFFS, repeat=WM_REPEAT, verify=False)


def order_jpeg_wm_jpeg(x_def, bits, qd):
    return jpeg_roundtrip(order_jpeg_then_wm(x_def, bits, qd), qd)


def order_wm_only(x_def, bits, qd):
    """對照：只有舉證層，不做量化交付。`qd` 不使用。"""
    return qim.embed(x_def, bits, delta=WM_DELTA, coeffs=WM_COEFFS,
                     repeat=WM_REPEAT, verify=False)


def order_jpeg_only(x_def, bits, qd):
    """對照：只有干擾層的量化交付，不嵌浮水印。位元欄由構造無意義。"""
    return jpeg_roundtrip(x_def, qd)


def order_jpeg_twice(x_def, bits, qd):
    """對照：壓兩次但**不**嵌浮水印。位元欄由構造無意義。

    **這一臂非有不可。** `jpeg_wm_jpeg` 走了兩次壓縮，而「壓兩次比壓一次更
    接近格點」是 JPEG 自己的性質，與浮水印無關。沒有這個對照的話，
    `jpeg_wm_jpeg` 在 `idem_qd` 上的優勢會被誤讀成「嵌浮水印讓交付更穩」。
    """
    return jpeg_roundtrip(jpeg_roundtrip(x_def, qd), qd)


ORDERS = {
    "wm_then_jpeg": order_wm_then_jpeg,
    "jpeg_then_wm": order_jpeg_then_wm,
    "jpeg_wm_jpeg": order_jpeg_wm_jpeg,
    "wm_only": order_wm_only,
    "jpeg_only": order_jpeg_only,
    "jpeg_twice": order_jpeg_twice,
}
# 沒有嵌入的臂：位元欄留空而不是填 0.5——填了看起來像「讀出來剛好一半對」，
# 那與「根本沒嵌」是兩件事。
NO_EMBED = ("jpeg_only", "jpeg_twice")


def read_bits(y01: torch.Tensor) -> torch.Tensor:
    return qim.extract(y01, WM_BITS, delta=WM_DELTA, coeffs=WM_COEFFS,
                       repeat=WM_REPEAT)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * torch.log10(
        torch.tensor(1.0 / mse)).item()


def run_cell(x01, x_def, name, qd, seed) -> Dict:
    bits = qim.random_bits(WM_BITS, seed)
    y = to_uint8_grid(ORDERS[name](x_def, bits, qd))

    # 干擾層還在不在：交付殘差對**原本那個擾動方向**的餘弦與長度比。
    d_ref = (x_def - x01).reshape(-1)
    d_got = (y - x01).reshape(-1)
    n_ref = float(d_ref.norm())
    row = {
        "order": name, "qd": qd,
        "pert_cos": round(float(torch.dot(d_got, d_ref) / (d_got.norm() * n_ref)), 5),
        "pert_norm": round(float(d_got.norm() / n_ref), 5),
        # 交付圖再壓一次會變多少。**這是干擾層依賴的性質**，見檔頭。
        "idem_qd": round(psnr(to_uint8_grid(jpeg_roundtrip(y, qd)), y), 3),
        "idem_j75": round(psnr(to_uint8_grid(jpeg_roundtrip(y, 0.75)), y), 3),
        "psnr_vs_def": round(psnr(y, x_def), 3),
    }
    if name in NO_EMBED:
        for k in ("bits_deliver", "bits_jpeg75", "bits_jpeg30", "bits_blur1"):
            row[k] = ""
        return row
    row["bits_deliver"] = round(qim.bit_accuracy(read_bits(y), bits), 5)
    for label, fn in (("bits_jpeg75", lambda t: jpeg_roundtrip(t, 0.75)),
                      ("bits_jpeg30", lambda t: jpeg_roundtrip(t, 0.30)),
                      ("bits_blur1", lambda t: ops.gaussian_blur(t, 1.0))):
        row[label] = round(qim.bit_accuracy(read_bits(to_uint8_grid(fn(y))), bits), 5)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--data-root", default="data/omniedit150")
    ap.add_argument("--qd", nargs="+", type=float, default=[0.85, 0.65, 0.45])
    ap.add_argument("--orders", nargs="+", default=list(ORDERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/watermark_deliver_order")
    args = ap.parse_args()

    rows: List[Dict] = []
    for name in args.images:
        path = ROOT / args.data_root / name / f"{name}.png"
        if not path.exists():
            raise FileNotFoundError(f"找不到影像 {path}")
        x01 = load_image(path)
        x_def = make_defence(x01, args.seed)
        print(f"[{name}] 干擾層 PSNR={psnr(x_def, x01):.2f}", flush=True)
        for qd in args.qd:
            for order in args.orders:
                r = run_cell(x01, x_def, order, qd, args.seed)
                r["image"] = name
                rows.append(r)
                print(f"  qd={qd:.2f} {order:14s} "
                      f"bits={r['bits_deliver'] or '—':>7} "
                      f"j75={r['bits_jpeg75'] or '—':>7} "
                      f"j30={r['bits_jpeg30'] or '—':>7} "
                      f"blur={r['bits_blur1'] or '—':>7} "
                      f"pert_cos={r['pert_cos']:.4f} "
                      f"idem_qd={r['idem_qd']:.1f} "
                      f"idem_j75={r['idem_j75']:.1f}", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["image", "order", "qd", "bits_deliver", "bits_jpeg75", "bits_jpeg30",
            "bits_blur1", "pert_cos", "pert_norm", "idem_qd", "idem_j75",
            "psnr_vs_def"]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n寫出 {len(rows)} 列到 {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
