"""用既有的防禦圖掃攻擊強度，量 E29 的否定結果對 strength 的敏感度。

## 這回答什麼

E31 規格 §5 把 strength 納入網格的理由是：在 strength = 0.5 的**全域** SDEdit
下，輸出主要由 prompt 重新生成，「語意不符 prompt」在原理上幾乎不可達成。
那是推論。本腳本用零雲端成本先量出實際的強度響應曲線。

## 這**不**回答什麼

既有的防禦圖是在 strength = 0.5 下訓練出來的。在 0.3 上評測它，量到的是
**遷移**（defense trained at 0.5, attacked at 0.3），不是 E31 網格要量的
匹配設定（訓練與攻擊同一個 strength，白盒下兩者必須一致）。

兩者的差別有方向性：匹配設定應優於遷移設定，故本腳本量到的是**下界**。
若連下界都顯示低 strength 有明顯差異，那 E31 網格的該軸就更值得跑；
若下界完全平坦，也不能據此斷定匹配設定平坦。

## 成本

本機 RTX 2050 4 GB。無梯度 512² SDEdit 在 strength=0.5、n_edit=10 下實測
222.5 s（`runs/logs/e31_local_probe.log`）。strength 較低時起始 timestep 較小，
但 `sdedit` 的步數由 `n_edit` 決定而非 strength，故成本相近。

執行：python scripts/e31_strength_sweep.py
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.sd import SDWrapper
from src.utils.device import get_device

EVAL_SEED_OFFSET = 10000
DEFAULT_SEED = 20260728
# E29c 的兩臂：τ=0.10、平台停止，即網格會用到的最寬鬆運作點。
CELLS = [
    ("undefended", "runs/e29c_P_tau0.10/car_00__P__r16/orig.png"),
    ("site_P", "runs/e29c_P_tau0.10/car_00__P__r16/defended.png"),
    ("site_C", "runs/e29c_C_tau0.10/car_00__C__r32/defended.png"),
]
PROMPT = "a wrecked car after an accident"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--out", default="runs/e31_strength_sweep")
    ap.add_argument("--strengths", default="0.2,0.3,0.4,0.5")
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    import pyiqa

    device = get_device()
    out = ROOT / args.out
    (out / "img").mkdir(parents=True, exist_ok=True)
    sd = SDWrapper(args.model)
    from src.metrics.suite import MetricSuite
    suite = MetricSuite(device=device)
    niqe = pyiqa.create_metric("niqe", device=device)

    emb = sd.encode_text(PROMPT)
    emb_u = sd.encode_text("")
    lat = sd.latent_shape(512, 512)

    def load(rel):
        img = Image.open(ROOT / rel).convert("RGB")
        return T.ToTensor()(img).unsqueeze(0).to(device)

    strengths = [float(s) for s in args.strengths.split(",")]
    rows = []
    for strength in strengths:
        # 同一個 strength 下三臂共用同一個 ε：否則量到的差異主要來自噪聲。
        noise = sd.sample_edit_noise(torch.empty(lat, device=device),
                                     seed=args.seed + EVAL_SEED_OFFSET)
        for arm, rel in CELLS:
            x = load(rel)
            t0 = time.perf_counter()
            with torch.no_grad():
                y = sd.sdedit(x, emb, noise, args.n_edit, strength=strength,
                              guidance_scale=args.guidance_scale,
                              emb_uncond=emb_u).clamp(0, 1)
                sem = suite.semantic(y, PROMPT)
                q = float(niqe(y))
            save_image(y, out / "img" / f"{arm}_s{strength}.png")
            rows.append({"arm": arm, "strength": strength,
                         "siglip": sem["siglip"], "clip": sem["clip"],
                         "niqe": q, "seconds": round(time.perf_counter() - t0, 1)})
            r = rows[-1]
            print(f"  s={strength} {arm:<11} siglip={r['siglip']:.4f} "
                  f"clip={r['clip']:.4f} niqe={r['niqe']:.3f} "
                  f"({r['seconds']:.0f}s)", flush=True)

    with open(out / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 逐 strength 的配對差，與正式判準同一個定義（防禦後 − 未防禦）
    print("\nstrength  Δsiglip(P)  Δsiglip(C)  Δniqe(P)  Δniqe(C)")
    deltas = []
    for s in strengths:
        g = {r["arm"]: r for r in rows if r["strength"] == s}
        d = {"strength": s,
             "dsiglip_P": g["site_P"]["siglip"] - g["undefended"]["siglip"],
             "dsiglip_C": g["site_C"]["siglip"] - g["undefended"]["siglip"],
             "dniqe_P": g["site_P"]["niqe"] - g["undefended"]["niqe"],
             "dniqe_C": g["site_C"]["niqe"] - g["undefended"]["niqe"]}
        deltas.append(d)
        print(f"{s:<9} {d['dsiglip_P']:>+10.4f}  {d['dsiglip_C']:>+10.4f}  "
              f"{d['dniqe_P']:>+8.3f}  {d['dniqe_C']:>+8.3f}")
    with open(out / "deltas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(deltas[0].keys()))
        w.writeheader()
        w.writerows(deltas)

    _page(out, rows, strengths)
    print(f"\n[sweep] 寫出 {out / 'sweep.csv'}、{out / 'deltas.csv'}、"
          f"{out / 'compare.html'}")
    print("[sweep] 提醒：防禦圖是在 strength=0.5 下訓練的，這裡量的是遷移不是"
          "匹配設定，故為下界。")


def _page(out: Path, rows, strengths):
    html = ["<!doctype html><meta charset='utf-8'>",
            "<title>E31 攻擊強度掃描</title>",
            "<style>body{font-family:sans-serif;background:#111;color:#eee;"
            "padding:16px}img{width:230px;display:block}"
            "td{padding:5px;text-align:center;vertical-align:top}"
            "small{font-size:11px;color:#aaa}th{color:#8cf}</style>",
            "<h1>攻擊強度掃描：既有防禦圖在不同 SDEdit strength 下被編輯</h1>",
            f"<p>prompt = <code>{PROMPT}</code>，w=7.5，n_edit=10，"
            "三臂共用同一個 ε。</p>",
            "<p><b>這是遷移不是匹配設定</b>：防禦圖在 strength=0.5 下訓練，"
            "在其他 strength 上評測。匹配設定應優於此，故本頁是下界。</p>",
            "<table><tr><th></th>"]
    for s in strengths:
        html.append(f"<th>strength = {s}</th>")
    html.append("</tr>")
    for arm, _ in CELLS:
        html.append(f"<tr><th>{arm}</th>")
        for s in strengths:
            r = [r for r in rows if r["arm"] == arm and r["strength"] == s][0]
            html.append(f"<td><img src='img/{arm}_s{s}.png'>"
                        f"<small>siglip={r['siglip']:.4f}<br>"
                        f"niqe={r['niqe']:.2f}</small></td>")
        html.append("</tr>")
    html.append("</table>")
    (out / "compare.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
