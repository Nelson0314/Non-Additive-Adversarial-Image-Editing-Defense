"""極座標可分離的內容場：對**旋轉**或對**中心縮放**成立的不變性。

為什麼再問一次「幾何不變的載體」
────────────────────────────────────────────────────────────────────
`runs/registration_ceiling/` 的結論是：內容場的**粗細**這個軸解決低通、
解決不了未對位那一種讀法，而且已經走到頭（上限由支撐邊界與 `δ` 裡那份 −x 給）。
要動幾何那一欄必須換軸——讓擾動**對變換本身不變**，而不是讓它更平滑。

`runs/log_periodic_probe/` 試過一次並否決：在 `log r` 上以 `ln s` 為週期的場
對 `s = 1.2488` 的中心放大不變（方向存活 0.97），但那是一個**寬度為零的窄峰**
——相鄰的 1.15 與 1.30 全部塌到 0。那等於對評測算子的一個特定參數 co-adapt。

**這一支問的是不同的東西。** 極座標下把場寫成只依一個變數：

    c = f(r)      只依半徑    繞中心旋轉任意角度 → **逐點不變**
    c = g(φ)      只依角度    繞中心縮放任意倍率 → **逐點不變**

兩者的不變性都是**對整個群**成立的，不是對某一個參數值。`f(r)` 在旋轉下
不變是因為旋轉不改變 `r`；`g(φ)` 在中心縮放下不變是因為縮放不改變 `φ`。
**沒有峰寬這個問題可談**——log-periodic 的窄峰來自它要求 `log r` 上的週期性，
而這兩個構造把該變數整個拿掉。

代價寫在同一句話裡：`f(r)` 對縮放只是等變不是不變（半徑被拉伸），
`g(φ)` 對旋轉只是等變（角度被平移）。**兩者換掉的是不同的一半。**

本專案的評測算子恰好兩種都用：`crop_resize` 是 `CROP_MODE = "center"` 的
中心裁切再放大（純中心縮放），`rotate10` 是繞影像中心的旋轉。故兩個構造
各自對上其中一個。

三件必須先驗證、否則後面全部沒有意義的事
────────────────────────────────────────────────────────────────────
1. **構造本身對不對**：把場自己送進算子，餘弦要接近 1。寫錯不會拋錯。
2. **不變性有沒有峰寬問題**：掃角度與掃裁切比例，曲線要是**平的**，
   不是一個尖峰。這是與 log-periodic 唯一的分界。
3. **貼到真實載體上還剩多少**：支撐是不規則的衣物區域、不是整張圖，
   而且 `δ = c − x` 帶著一份 −x（`runs/registration_ceiling/` 第二節）。
   **上面兩點成立不蘊含這一點成立。**

**這一支不最佳化、不碰 GPU，也不判斷任何形式成不成立。**
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for q in (ROOT, ROOT / "scripts"):
    if str(q) not in sys.path:
        sys.path.insert(0, str(q))

from src.purify import ops  # noqa: E402


def polar(side: int, device=None):
    """以**影像中心**為原點的 `(r, φ)`。中心必須與算子的中心一致——
    `rotate_random` 走 `affine_grid`（繞影像中心）、`crop_resize` 是中心裁切，
    兩者的原點都是影像中心，不是支撐的質心。原點取錯時不變性完全不成立，
    而且沒有任何症狀。"""
    ys = torch.linspace(-1.0, 1.0, side, device=device)[:, None]
    xs = torch.linspace(-1.0, 1.0, side, device=device)[None, :]
    return (ys ** 2 + xs ** 2).sqrt(), torch.atan2(ys.expand(side, side),
                                                   xs.expand(side, side))


def profile(n: int, seed: int, periodic: bool) -> torch.Tensor:
    """平滑的一維隨機剖面，`(3, n)`。頻寬由 `n` 決定。

    `periodic=True` 用實數 FFT 造，故 `g(φ)` 在 `φ = ±π` 的接縫上連續——
    不連續會在畫面上留一條半徑方向的硬邊，而那條邊是高頻的，正好是要避開的。
    """
    g = torch.Generator().manual_seed(seed)
    if periodic:
        k = 6                       # 保留的諧波數，決定角向的平滑度
        a = torch.randn(3, k, generator=g)
        b = torch.randn(3, k, generator=g)
        t = torch.linspace(0.0, 2.0 * torch.pi, n + 1)[:-1]
        out = sum(a[:, i: i + 1] * torch.cos((i + 1) * t)
                  + b[:, i: i + 1] * torch.sin((i + 1) * t) for i in range(k))
    else:
        out = F.interpolate(torch.rand(1, 3, 12, generator=g),
                            size=n, mode="linear", align_corners=True)[0]
    out = out - out.amin(dim=1, keepdim=True)
    return out / out.amax(dim=1, keepdim=True).clamp_min(1e-8)


def field(kind: str, side: int, seed: int) -> torch.Tensor:
    """`(1,3,side,side)`，值域 [0,1]。"""
    r, phi = polar(side)
    n = 256
    if kind == "radial":
        p = profile(n, seed, periodic=False)
        idx = (r / r.max() * (n - 1)).clamp(0, n - 1)
    elif kind == "angular":
        p = profile(n, seed, periodic=True)
        idx = ((phi + torch.pi) / (2 * torch.pi) * n) % n
    else:
        raise ValueError(kind)
    lo = idx.floor().long().clamp(0, n - 1)
    hi = (lo + 1) % n if kind == "angular" else (lo + 1).clamp(max=n - 1)
    w = (idx - lo.to(idx.dtype)).unsqueeze(0)
    return (p[:, lo] * (1 - w) + p[:, hi] * w).unsqueeze(0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/polar_carrier")
    ap.add_argument("--data", type=Path, default=ROOT / "data/omniedit150")
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--skip-carrier", action="store_true",
                    help="只跑前兩關（構造與峰寬），不載分割模型")
    args = ap.parse_args()
    rows: List[Dict] = []

    # ── 一、構造：把場自己送進算子 ─────────────────────────────
    print("一、場自己過算子（扣掉直流之後的餘弦；1.00 = 逐點不變）", flush=True)
    print(f"{'場':>10}{'旋轉 5.15°':>12}{'裁切 10%':>11}{'裁切 5%':>10}"
          f"{'裁切 15%':>11}{'模糊 1.5':>11}", flush=True)
    tests = [("rotate5.15", lambda t: ops.rotate_random(t, 10.0, seed=1)),
             ("crop0.10", lambda t: ops.crop_resize(t, 0.10)),
             ("crop0.05", lambda t: ops.crop_resize(t, 0.05)),
             ("crop0.15", lambda t: ops.crop_resize(t, 0.15)),
             ("blur1.5", lambda t: ops.gaussian_blur(t, 1.5))]
    for kind in ("radial", "angular"):
        line = f"{kind:>10}"
        for lbl, fn in tests:
            vs = []
            for s in range(args.seeds):
                c = field(kind, args.side, s)
                c = c - c.mean()
                vs.append(cosine(fn(c + 0.5) - 0.5, c))
            v = sum(vs) / len(vs)
            line += f"{v:>11.4f} "
            rows.append({"stage": "construction", "field": kind, "op": lbl,
                         "cos": round(v, 5)})
        print(line, flush=True)

    # ── 二、峰寬：掃角度、掃裁切比例 ───────────────────────────
    print("\n二、峰寬（log-periodic 在這裡塌成一個尖峰；平的才算對整個群不變）",
          flush=True)
    angles = [1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 25.0, 45.0]
    fracs = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    for kind in ("radial", "angular"):
        for name, xs, mk in (("rotate", angles, lambda a: (
                lambda t: ops.rotate_random(t, 0.0, seed=0) if a == 0 else
                _rot(t, a))),
                ("crop", fracs, lambda f: (lambda t: ops.crop_resize(t, f)))):
            line = f"{kind:>10} {name:>7}"
            for v in xs:
                fn = mk(v)
                cs = []
                for s in range(args.seeds):
                    c = field(kind, args.side, s)
                    c = c - c.mean()
                    cs.append(cosine(fn(c + 0.5) - 0.5, c))
                m = sum(cs) / len(cs)
                line += f"{m:>8.3f}"
                rows.append({"stage": "width", "field": kind, "op": name,
                             "param": v, "cos": round(m, 5)})
            print(f"{line}    ({'角度°' if name == 'rotate' else '裁切比例'}: "
                  + " ".join(str(v) for v in xs) + ")", flush=True)

    # ── 三、貼到真實載體上 ────────────────────────────────────
    # 前兩關量的是**場自己**。真正要交付的是 `δ = c − x`，而且支撐是不規則
    # 的衣物區域、不含影像中心。兩件事都會把不變性打掉一部分：
    #   支撐被算子搬走（邊界項，見 runs/registration_ceiling/ 第一節）
    #   δ 裡那份 −x 隨算子被搬走（同節第二節）
    # **前兩關成立不蘊含這一關成立。**
    if not args.skip_carrier:
        rows += carrier_stage(args)

    args.out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(args.out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n寫出 {len(rows)} 列到 {args.out / 'results.csv'}")


def carrier_stage(args) -> List[Dict]:
    """第三關：把場貼到真實衣物支撐上，量 `δ = c − x` 過算子之後的存活。

    對照組是 `帶限 S=32`——`runs/carrier_purify_response/` 裡幾何欄最好的
    那一族之一。沒有它就不知道 0.9 是好還是普通。
    """
    import statistics as st
    from types import SimpleNamespace as NS
    import numpy as np
    from PIL import Image
    from carrier_purify_response import CARRIER_FLAGS, FORMS, build_param
    from purify_identity import build_purifier
    from src.defense.carrier_mask import face_subject_mask
    import ip2p_run

    images = args.images
    if images is None:
        base = ROOT / "runs/ip2p_face_defence"
        images = sorted(p.name.replace("clothing_plain_", "")
                        for p in base.glob("clothing_plain_task_*") if p.is_dir())
    purs = [(l, build_purifier(l, seed=0)) for l in
            ("rotate10", "crop_resize0.1", "blur1.5", "jpeg75", "resize_only")]
    cargs = NS(**CARRIER_FLAGS)
    acc: Dict = {}
    rms: Dict = {}
    print("\n三、貼到真實載體上（delta 的 align 中位）", flush=True)

    for img in images:
        src = args.data / img / "input.png"
        if not src.exists():
            src = sorted((args.data / img).glob("*.png"))[0]
        a = np.asarray(Image.open(src).convert("RGB").resize((512, 512),
                                                             Image.BICUBIC))
        x = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float() / 255.0
        mask = face_subject_mask(x)
        car = ip2p_run._carrier_of(x, mask, cargs)
        sup = (car.to(torch.float32) * (mask <= 0).to(torch.float32))

        cands = {"radial": field("radial", 512, 0),
                 "angular": field("angular", 512, 0)}
        obj = build_param(FORMS["res_s32"], mask, car, 0)
        obj.reset(x, 0)
        cands["res_s32 ◂ 對照"] = obj._field(x).detach()

        for name, c in cands.items():
            # **軟支撐要原樣用**：轉成布林（`sup > 0`）會把整條羽化帶算成滿的，
            # 那時量到的不是 `render` 實際交付的那張圖，而每一列看起來都正常。
            # 這與 `PatchParam.render` 的混色式子相同，`w = 0` 處逐位元等於原圖。
            w = sup.to(x.dtype)
            xd = ((1.0 - w) * x + w * c.clamp(0, 1)).clamp(0, 1)
            d = xd - x
            rms.setdefault(name, []).append(float(d.pow(2).mean().sqrt()))
            for lbl, pur in purs:
                with torch.no_grad():
                    r = pur.evaluate(xd.clone()) - pur.evaluate(x.clone())
                acc.setdefault((name, lbl), []).append(cosine(r, d))

    out: List[Dict] = []
    hdr = f"{'':16}{'RMS':>7}" + "".join(f"{l[:11]:>13}" for l, _ in purs)
    print(hdr, flush=True)
    for name in ("res_s32 ◂ 對照", "radial", "angular"):
        line = f"{name:16}{st.median(rms[name]):>7.3f}"
        for lbl, _ in purs:
            m = st.median(acc[(name, lbl)])
            line += f"{m:>+13.2f}"
            out.append({"stage": "carrier", "field": name, "op": lbl,
                        "cos": round(m, 5), "n": len(acc[(name, lbl)])})
        print(line, flush=True)
    return out


def _rot(t: torch.Tensor, deg: float) -> torch.Tensor:
    """繞影像中心轉**指定**角度。`rotate_random` 的角度由種子抽出，
    掃描時需要直接給角度，故這裡用同一組設定（雙線性、補零、affine_grid
    的反向映射）重寫一次。**兩式必須同形**，否則掃出來的曲線與正式算子
    量的不是同一件事。"""
    import math
    rad = math.radians(deg)
    cos, sin = math.cos(rad), math.sin(rad)
    theta = torch.tensor([[cos, sin, 0.0], [-sin, cos, 0.0]],
                         dtype=t.dtype)[None].expand(t.shape[0], -1, -1)
    grid = F.affine_grid(theta, list(t.shape), align_corners=False)
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


if __name__ == "__main__":
    main()
