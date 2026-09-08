"""四種內容形式的擾動撐不撐得過十二個淨化算子——**純 CPU，不需要訓練**。

為什麼這一支可以在沒有 GPU 的情況下回答一部分問題
────────────────────────────────────────────────────────────────────
`runs/purifier_transfer/` 量的是**算子**：一個放在頻率 f、振幅 RMS 的擾動，
過了算子之後還剩多少。那份表的自變數是人造的帶限雜訊，不是真的載體內容。

這一支換掉自變數：把**真正的參數化在真正的載體支撐上渲染出來**的那一張圖
拿去過同樣的算子。中間不需要擴散模型，因為問的仍然是訊號處理的問題——
「這個形狀的擾動被算子吃掉多少」，與它對 InstructPix2Pix 有沒有效無關。

    energy   ‖P(x+δ) − P(x)‖ / ‖δ‖      擾動還剩多少
    align    cos( P(x+δ) − P(x), δ )     它還在不在原來的位置

兩欄分開報的理由與 `purifier_transfer` 相同，且已有實例：裁切之後殘差對
原格點的餘弦是 0.000，對算子自己搬過的那一份卻是 0.995——只看 energy 會把
「被破壞」與「被搬走」讀成同一件事。

**這一支不判斷任何形式成不成立。**
────────────────────────────────────────────────────────────────────
它量的是**隨機起點、未最佳化**的場。最佳化會改變場的頻譜，故這裡的數字是
「這個參數化典型會產生的內容有多耐算子」，不是「訓練出來的解有多耐算子」。
兩者可能不同，而差多少要等 GPU 那批回來才知道——那正是這份表的用途：
它給出一個**在訓練之前就存在的預期**，訓練後的實測可以與它對照。

依 `CLAUDE.md`「訓練方法的實驗：不設判準」，本表不得用來決定任何一批跑不跑。

隨機臂而不是最佳化臂的起點
────────────────────────────────────────────────────────────────────
四種形式都有對應的 `*RandomParam`，它們與最佳化臂共用同一個渲染路徑、
同一組色票、同一個支撐，只有可學的那一份是隨機的。用它們可以讓四種形式
在完全相同的處置下並排，不必先跑任何最佳化。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.purify import ops  # noqa: E402
from purify_identity import LITERATURE_SET, build_purifier  # noqa: E402

#: 載體的六個旗標，逐字取自 `scripts/patch_res_round.sh` 的 `COMMON`。
#: **不要在這裡另填一組看起來合理的值**——兩處分岔時這份表量的就不是
#: 那一批實際用的支撐，而兩邊都不會有症狀。
CARRIER_FLAGS = dict(
    patch_carrier="clothes", carrier_refine=4, carrier_erode=3,
    carrier_feather=8, patch_placement="complement", radius=0.05,
    carrier_min_area=None, carrier_max_area=None, carrier_ring=0,
    carrier_ring_inner=0, carrier_lattice=0, carrier_dot_radius=0.0,
    carrier_scatter=0, carrier_target_area=0.0, carrier_match="erode", seed=0,
)

#: 四種內容形式。名稱與 `docs/DIRECTION.md` §3.8、派工腳本的臂名一致。
FORMS: Dict[str, dict] = {
    "free":      dict(kind="plain", res=1),
    "res_s08":   dict(kind="plain", res=8),
    "res_s16":   dict(kind="plain", res=16),
    "res_s24":   dict(kind="plain", res=24),
    "res_s32":   dict(kind="plain", res=32),
    # 以下四個**不在派工的格點上**，只用來看幾何類那一欄的趨勢會不會續漲：
    # S=8→32 的旋轉 align 是 +0.62→+0.73，單調而且沒有轉平。
    "res_s48":   dict(kind="plain", res=48),
    "res_s64":   dict(kind="plain", res=64),
    "res_s96":   dict(kind="plain", res=96),
    "res_s128":  dict(kind="plain", res=128),
    "palette_k08": dict(kind="palette", palette=8),
    "voronoi_k24": dict(kind="voronoi", seeds=24),
    "voronoi_k64": dict(kind="voronoi", seeds=64),
}


def build_param(spec: dict, mask, carrier, seed: int):
    """建**隨機臂**的參數化物件。四種形式共用同一個渲染路徑。"""
    from src.defense.patch_param import (PatchPaletteRandomParam,
                                         PatchRandomParam,
                                         PatchVoronoiRandomParam)
    common = dict(radius=CARRIER_FLAGS["radius"], mask=mask,
                  placement="complement")
    if spec["kind"] == "voronoi":
        obj = PatchVoronoiRandomParam(seeds=spec["seeds"], **common)
    elif spec["kind"] == "palette":
        obj = PatchPaletteRandomParam(palette=spec["palette"], **common)
    else:
        obj = PatchRandomParam(res=spec["res"], **common)
    obj.carrier = carrier
    obj.carrier_kind = CARRIER_FLAGS["patch_carrier"]
    return obj


def measure(fn, x: torch.Tensor, xd: torch.Tensor) -> tuple:
    """`(energy, align)`，在**整張圖**上量。

    不限制在支撐內：幾何類算子會把支撐搬走，把統計限制在原支撐的位置上
    等於預先假設了「沒有人替它對回去」，而那正是 `align` 那一欄要回答的事。
    """
    with torch.no_grad():
        a = fn(x.clone())
        b = fn(xd.clone())
    r = (b - a).reshape(-1)
    d = (xd - x).reshape(-1)
    nr, nd = float(r.norm()), float(d.norm())
    if nd == 0:
        raise ValueError("擾動的能量為零——渲染沒有改動任何像素")
    return nr / nd, (float(torch.dot(r, d) / (nr * nd)) if nr > 0 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/omniedit150")
    ap.add_argument("--images", nargs="+", default=None,
                    help="預設由 runs/ip2p_face_defence 的 clothing_plain_* 推導")
    ap.add_argument("--forms", nargs="+", default=list(FORMS))
    ap.add_argument("--purifiers", nargs="+", default=list(LITERATURE_SET))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/carrier_purify_response")
    args = ap.parse_args()

    import torchvision.utils as _  # noqa: F401  提早暴露相依缺失
    from PIL import Image
    import numpy as np
    from src.defense.carrier_mask import face_subject_mask
    from src.metrics.identity import face_boxes  # noqa: F401  給 _carrier_of 用
    import ip2p_run

    images = args.images
    if images is None:
        base = ROOT / "runs/ip2p_face_defence"
        images = sorted(p.name.replace("clothing_plain_", "")
                        for p in base.glob("clothing_plain_task_*") if p.is_dir())
    if not images:
        raise SystemExit("推導不出影像清單；用 --images 明給")

    purs = [(lbl, build_purifier(lbl, seed=args.seed)) for lbl in args.purifiers]
    missing = [lbl for lbl, p in purs if not p.available]
    if missing:
        raise SystemExit(f"相依不齊的算子：{missing}。**不靜默跳過**"
                         "——少一個算子表上就少一欄")

    cargs = SimpleNamespace(**CARRIER_FLAGS)
    rows: List[dict] = []
    print(f"{len(images)} 張 × {len(args.forms)} 形式 × {len(purs)} 算子",
          flush=True)

    for img in images:
        src = args.data / img / "input.png"
        if not src.exists():
            cand = sorted((args.data / img).glob("*.png"))
            if not cand:
                raise SystemExit(f"{args.data / img} 底下找不到影像")
            src = cand[0]
        arr = np.asarray(Image.open(src).convert("RGB").resize((512, 512),
                                                              Image.BICUBIC))
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 255.0
        mask = face_subject_mask(x)
        carrier = ip2p_run._carrier_of(x, mask, cargs)
        print(f"\n[{img}] 主體 {float(mask.mean()):.3f} "
              f"載體 {float(carrier.mean()):.3f}", flush=True)

        for form in args.forms:
            obj = build_param(FORMS[form], mask, carrier, args.seed)
            obj.reset(x, args.seed)
            xd = obj.render(x).clamp(0, 1)
            d = xd - x
            area = float(obj.support.to(torch.float32).mean())
            rms = float(d.pow(2).mean().sqrt())
            line = f"  {form:13s} 支撐 {area:.3f} RMS {rms:.4f} |"
            for lbl, pur in purs:
                e, a = measure(pur.evaluate, x, xd)
                rows.append({"image": img, "form": form,
                             "support_area": round(area, 5),
                             "delta_rms": round(rms, 5),
                             "purifier": lbl, "purifier_kind": pur.kind,
                             "geometric": pur.kind in ops.GEOMETRIC_KINDS,
                             "energy": round(e, 5), "align": round(a, 5)})
                if lbl in ("jpeg75", "blur1.5", "rotate10", "resize_only"):
                    line += f" {lbl}={e:.2f}/{a:+.2f}"
            print(line, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / "results.csv"
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n寫出 {len(rows)} 列到 {dst}")


if __name__ == "__main__":
    main()
