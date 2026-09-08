"""支撐邊界的羽化寬度：買得到幾何對齊，代價是把預算羽化掉一半以上。

為什麼有這一支
────────────────────────────────────────────────────────────────────
`runs/registration_ceiling/` 與 `runs/polar_carrier/` 兩份表都收在同一句話上：
幾何那一欄剩下的損失**在支撐邊界上**——支撐本身被算子搬走，那與內容場的
參數化無關。而支撐邊界是**載體的幾何**，是可調的：`--carrier-feather` 就是
往內羽化的寬度。

羽化寬度變大時，邊界附近的權重逐漸趨零，被搬走的那一圈本來就沒有多少擾動，
邊界項因此縮小。**但同一個動作也把可用面積與擾動振幅一起壓掉**，而
`runs/purifier_transfer/` 第二節量過振幅對 JPEG 存活的影響比頻帶還大。
兩邊都要報，只報一邊會得到「羽化是免費的」這個錯誤印象。

量什麼
────────────────────────────────────────────────────────────────────
    加權面積   支撐權重的平均——**不是 `w > 0` 的像素數**，羽化帶會被後者
               整條算成滿的
    RMS        交付擾動的均方根
    align      cos( P(x+δ) − P(x), δ )，未對位那一種讀法

分割只跑一次，之後只換羽化寬度：分割每張圖跑一次要幾秒，而且同一張圖不同
寬度必須來自**同一個**分割結果，否則比較的是兩個載體不是兩個寬度。

**這一支不判斷該用哪個寬度。** 依 `CLAUDE.md`「訓練方法的實驗：不設判準」，
數與圖擺出來為止。
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
for _q in (ROOT, ROOT / "scripts"):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

from purify_identity import build_purifier  # noqa: E402

FEATHERS = (0, 8, 16, 32, 48, 64)
PURIFIERS = ("rotate10", "crop_resize0.1", "blur1.5", "jpeg75")


def arms():
    """三個內容形式。`res_s32` 是對照，兩個極座標場是 `runs/polar_carrier/`
    的構造。三者共用同一個支撐，差別只在內容。"""
    from src.defense.patch_param import PatchPolarRandomParam, PatchRandomParam
    return (("res_s32", PatchRandomParam, dict(res=32)),
            ("angular", PatchPolarRandomParam, dict(polar="angular", bins=32)),
            ("radial", PatchPolarRandomParam, dict(polar="radial", bins=32)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/omniedit150")
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--feathers", nargs="+", type=int, default=list(FEATHERS))
    ap.add_argument("--purifiers", nargs="+", default=list(PURIFIERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs/carrier_feather_tradeoff")
    args = ap.parse_args()

    import numpy as np
    from PIL import Image
    from src.defense.carrier_mask import (MAX_AREA, MIN_AREA, carrier_mask,
                                          erode_mask, exclude_boxes,
                                          face_subject_mask, feather_inward,
                                          guided_refine)
    from src.metrics.identity import face_boxes

    images = args.images
    if images is None:
        base = ROOT / "runs/ip2p_face_defence"
        images = sorted(p.name.replace("clothing_plain_", "")
                        for p in base.glob("clothing_plain_task_*") if p.is_dir())
    purs = [(l, build_purifier(l, seed=args.seed)) for l in args.purifiers]
    acc: Dict = {}
    area: Dict = {}
    rms: Dict = {}

    for img in images:
        src = args.data / img / "input.png"
        if not src.exists():
            src = sorted((args.data / img).glob("*.png"))[0]
        a = np.asarray(Image.open(src).convert("RGB").resize((512, 512),
                                                             Image.BICUBIC))
        x = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float() / 255.0
        mask = face_subject_mask(x)
        # 分割與前三步只跑一次；羽化是最後一步，故換寬度不必重跑分割。
        c0 = carrier_mask(x, "clothes", min_area=MIN_AREA, max_area=MAX_AREA)
        c0 = (guided_refine(c0, x, 4, 1e-3) > 0.5).to(c0.dtype)
        c0 = exclude_boxes(c0, face_boxes(x), margin=8)
        c0 = erode_mask(c0, 3)
        print(f"[{img}] 羽化前的載體 {float(c0.mean()):.3f}", flush=True)

        for fw in args.feathers:
            car = feather_inward(c0.clone(), fw) if fw else c0.clone()
            # **保留軟權重**：轉成布林會把整條羽化帶算成滿的，那時掃描完全
            # 沒有作用而每一列看起來都正常（實測過一次）。
            sup = car.to(torch.float32) * (mask <= 0).to(torch.float32)
            for name, cls, kw in arms():
                o = cls(radius=0.05, mask=mask, placement="complement", **kw)
                o.reset(x, args.seed)
                o.support = sup
                xd = o.render(x).clamp(0, 1)
                d = xd - x
                area.setdefault((name, fw), []).append(float(sup.mean()))
                rms.setdefault((name, fw), []).append(
                    float(d.pow(2).mean().sqrt()))
                for lbl, pur in purs:
                    with torch.no_grad():
                        r = pur.evaluate(xd.clone()) - pur.evaluate(x.clone())
                    acc.setdefault((name, fw, lbl), []).append(float(
                        torch.dot(r.reshape(-1), d.reshape(-1))
                        / (r.norm() * d.norm())))

    rows: List[Dict] = []
    print(f"\n{'形式':>8}{'羽化':>6}{'加權面積':>9}{'RMS':>7}"
          + "".join(f"{l:>15}" for l, _ in purs), flush=True)
    for name, _, _ in arms():
        for fw in args.feathers:
            line = (f"{name:>8}{fw:>6}{st.median(area[(name, fw)]):>9.3f}"
                    f"{st.median(rms[(name, fw)]):>7.3f}")
            row = {"form": name, "feather": fw,
                   "support_area": round(st.median(area[(name, fw)]), 5),
                   "delta_rms": round(st.median(rms[(name, fw)]), 5),
                   "n": len(area[(name, fw)])}
            for lbl, _ in purs:
                m = st.median(acc[(name, fw, lbl)])
                line += f"{m:>+15.2f}"
                row[f"align_{lbl}"] = round(m, 5)
            print(line, flush=True)
            rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n寫出 {len(rows)} 列到 {args.out / 'results.csv'}")


if __name__ == "__main__":
    main()
