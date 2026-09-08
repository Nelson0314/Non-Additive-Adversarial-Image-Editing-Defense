"""幾何類算子上的登記為什麼會飽和：殘差是支撐的**邊界**在動，不是內容在動。

`runs/carrier_purify_response/` 第二節量到一件事：把帶限的 S 由 8 加粗到 128，
旋轉的 `align` 只由 +0.62 爬到 +0.80、裁切由 +0.49 爬到 +0.59，兩者都明顯
飽和在遠低於 +1.0 的地方，而低通那三欄在 S=8 就已經飽和。

**這一支問那個上限是誰給的。** 兩個候選：

    內容    支撐內的圖樣被搬走之後與原來的圖樣對不上
    邊界    支撐本身被搬走，邊界帶上原本有擾動的像素變成沒有（反之亦然）

分得開，因為兩者**住在不同的像素上**：把殘差 `P(x+δ) − P(x)` 依「離支撐邊界
多遠」切成兩堆，各自算能量佔比與對齊，就知道是哪一個。

取 `res = 512`（可學張量是 1×1，內容是**單一顏色**）當極限：那時支撐內沒有
任何圖樣可以對不上，`align` 若仍然不到 +1.0，剩下的就只能是邊界。

**這一支不判斷任何形式成不成立**，它只回答「飽和的上限是誰給的」。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for q in (ROOT, ROOT / "scripts"):
    if str(q) not in sys.path:
        sys.path.insert(0, str(q))

from carrier_purify_response import CARRIER_FLAGS, FORMS, build_param  # noqa: E402
from purify_identity import build_purifier  # noqa: E402

#: 幾何類的四個算子，加兩個低通當對照——低通不搬東西，故它的殘差**不應該**
#: 集中在邊界上。沒有這兩欄就分不出「邊界佔比高」是幾何的性質還是遮罩的
#: 面積效應（邊界帶本來就佔一定比例的像素）。
PURIFIERS = ["rotate10", "crop_resize0.1", "resize_only",
             "jpeg_then_resize75", "blur1.5", "jpeg75"]

#: 邊界帶的寬度（像素）。`rotate10` 的角度是 5.15°，512 px 影像最遠角落位移
#: 約 32 px，故取到 32 才涵蓋得住最大位移。
BAND_PX = (4, 8, 16, 32)


def boundary_band(sup: torch.Tensor, px: int) -> torch.Tensor:
    """支撐邊界兩側各 `px` 像素的帶。膨脹減侵蝕，兩者用同一個方形核。"""
    m = (sup > 0.0).to(torch.float32)
    k = 2 * px + 1
    dil = (F.max_pool2d(m, k, stride=1, padding=px) > 0.5)
    ero = (-F.max_pool2d(-m, k, stride=1, padding=px) > 0.5)
    return dil & (~ero)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/omniedit150")
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--forms", nargs="+",
                    default=["free", "res_s08", "res_s32", "res_s128", "res_s512"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/registration_ceiling")
    args = ap.parse_args()

    from PIL import Image
    import numpy as np
    from src.defense.carrier_mask import face_subject_mask
    import ip2p_run

    if "res_s512" not in FORMS:
        FORMS["res_s512"] = dict(kind="plain", res=512)

    images = args.images
    if images is None:
        base = ROOT / "runs/ip2p_face_defence"
        images = sorted(p.name.replace("clothing_plain_", "")
                        for p in base.glob("clothing_plain_task_*") if p.is_dir())

    purs = [(lbl, build_purifier(lbl, seed=args.seed)) for lbl in PURIFIERS]
    cargs = SimpleNamespace(**CARRIER_FLAGS)
    rows: List[dict] = []

    for img in images:
        src = args.data / img / "input.png"
        if not src.exists():
            src = sorted((args.data / img).glob("*.png"))[0]
        arr = np.asarray(Image.open(src).convert("RGB").resize((512, 512),
                                                              Image.BICUBIC))
        x = torch.from_numpy(arr.copy()).permute(2, 0, 1)[None].float() / 255.0
        mask = face_subject_mask(x)
        carrier = ip2p_run._carrier_of(x, mask, cargs)
        print(f"\n[{img}]", flush=True)

        for form in args.forms:
            obj = build_param(FORMS[form], mask, carrier, args.seed)
            obj.reset(x, args.seed)
            xd = obj.render(x).clamp(0, 1)
            d = xd - x
            sup = obj.support.to(torch.float32)
            bands = {px: boundary_band(sup, px) for px in BAND_PX}

            for lbl, pur in purs:
                with torch.no_grad():
                    r = pur.evaluate(xd.clone()) - pur.evaluate(x.clone())
                row = {"image": img, "form": form, "purifier": lbl,
                       "align_all": round(float(
                           torch.dot(r.reshape(-1), d.reshape(-1))
                           / (r.norm() * d.norm())), 5)}
                for px, b in bands.items():
                    m3 = b.expand_as(r)
                    rb, ri = r[m3], r[~m3]
                    db, di = d[m3], d[~m3]
                    row[f"band{px}_pix_frac"] = round(float(b.to(torch.float32).mean()), 5)
                    row[f"band{px}_res_frac"] = round(
                        float(rb.pow(2).sum() / r.pow(2).sum()), 5)
                    row[f"band{px}_align_in"] = round(
                        float(torch.dot(ri, di) / (ri.norm() * di.norm())), 5)
                rows.append(row)
            print(f"  {form:10s} " + "  ".join(
                f"{r_['purifier'][:9]}={r_['align_all']:+.2f}→{r_['band32_align_in']:+.2f}"
                for r_ in rows[-len(purs):]), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n寫出 {len(rows)} 列到 {args.out / 'results.csv'}")


if __name__ == "__main__":
    main()
