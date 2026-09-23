"""現行的色調曲線載體，但**臉與背景吃不同的色偏預算**，且不產生生硬接縫。

問題
────────────────────────────────────────────────────────────────────
現行交付的操作點 `colour_curve_ours` 是一條**全域**單調分段線性 RGB tone
curve（`pieces 64`、`radius 5.0`、900 步、ΔE00 ≤ 16）。全域的意思是同一個
輸入色階在畫面任何位置都映到同一個輸出——這正是它沒有接縫的原因，也是它
**無法**把預算在臉與背景之間重新分配的原因：給臉一道較緊的上限，只會讓整條
曲線被那道上限綁住，背景並不會因此拿到更多振幅。

把預算分開，勢必要讓映射帶有空間相依性，而空間相依性正是接縫的來源。本檔
用兩種互斥的方式處理這件事，各自成為一個臂。

`chroma`：預算分在**色度**上，映射仍然全域
────────────────────────────────────────────────────────────────────
載體不變（單一全域曲線），只是上限分兩道：整圖 ΔE00 ≤ `--frame-cap`，人臉框
內 ΔE00 ≤ `--face-cap`（較緊）。因為映射是全域的，「把臉的色偏壓下來」實際
發生在**臉的顏色所落到的那幾段曲線**上；背景中恰好同色的像素一起被壓，
其餘色階仍可用滿整圖的預算。

**接縫在構造上不存在**：同一個輸入色階的輸出唯一。本專案已在交付的八張上
逐像素反查驗證過這件事（同一輸入色階的輸出寬度最大值為 0）。
代價是「臉」被定義成一組顏色而不是一塊區域。

`spatial`：兩條曲線 ＋ 平滑混合場
────────────────────────────────────────────────────────────────────

    y = w·F_face(x) + (1 − w)·F_bg(x)

`F_face` 與 `F_bg` 各自是一條單調分段線性曲線（同一族、同一個 `radius`），
兩者各自求解；`w` 是由人臉框長出的**固定**權重場，不是可學參數。

接縫的控制有三層，缺一層都會回到已知的失敗形狀（遮罩帶窄出斑塊、帶寬出臉霧）：

1. **`w` 用 smoothstep，不是二值遮罩也不是高斯模糊的二值遮罩。** 橢圓距離
   `d` 在 `[1, 1+feather]` 上以 `3t²−2t³` 由 1 降到 0，兩端一階導數為零，
   故 `w` 是 C¹ 的，沒有任何硬邊。
2. **過渡帶上的位移場 TV 有上限**（`--band-tv-cap`）。接縫的定義就是位移場在
   過渡帶上的梯度，直接把它當一道約束。本專案已量到均勻度與防禦強度直接
   對立，轉折落在位移場 TV 1.0 與 2.0 之間，故預設取 2.0。
3. **兩條曲線的差本身沒有被獎勵。** 目標只看編輯端的代理，曲線差只是達成
   手段；配上第 2 點，解會自動避開「兩條曲線在同一個輸入色階上差很多」的
   區域——那正是過渡帶上 TV 的來源。

求解端（兩個臂共用，與 `colour_curve_ours` 同一組設定）
────────────────────────────────────────────────────────────────────
`FreeObjective`（`src/defense/instruction_free.py`）：不含指令的目標，權重
`id 1.0`／`enc 0.5`／`cond 1.0`，`timesteps 4`、`chain_steps 6`、
`grad_steps 1`、`s_i 1.5`、`resample`。約束走增廣 Lagrange
（`optimise_carrier`），**不做事後投影**；可行性在量化後的 PNG 上檢查，
最佳可行 checkpoint 由固定驗證抽樣的 `eval_score` 挑。

**求解端不含任何文字。** 三個項都不經過 text encoder。

產出（版面與 `defence_run.py` 相同）
────────────────────────────────────────────────────────────────────
    {out}/{image}__orig.png
    {out}/{image}__{arm}__def.png
    {out}/results.csv                逐列寫入

用法
    python code/curve_budget_defence.py --arm curve_dual_spatial \\
        --out lab/runs/defence/curve_dual_spatial --budget-mode spatial
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402
import yaml  # noqa: E402

from colour_support import (  # noqa: E402
    apply_slopes, chroma_p95, skin_colour_support, skin_tone_slopes,
)
from src.defense.color_amplitude import delta_e00  # noqa: E402
from src.defense.color_param import ColorCurveParam  # noqa: E402
from src.defense.delta_e_torch import delta_e00_torch  # noqa: E402
from src.defense.immunise import Cap, optimise_carrier, quantise  # noqa: E402
from src.defense.instruction_free import FreeObjective  # noqa: E402
from src.defense.uniformity import lab_offset, tv_offset  # noqa: E402
from src.metrics.identity import (  # noqa: E402
    DETECTOR_IMAGE_SIZE, DETECTOR_MARGIN, face_boxes,
)
from src.models.ip2p import IP2PWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512

#: 求解端看到的文字條件。整條路徑不含文字，這一列只是為了讓 CSV 自己說得出
#: 「防禦方看到了什麼」，與其餘條件的欄位對得起來。
SOLVER_PROMPT = ("", "無文字條件：三個項都不經過 text encoder")


# ---- 載體 ----

class ColorCurveDual:
    """兩條單調曲線 ＋ 固定的平滑混合場。`w = 1` 的地方完全由 `a` 決定。

    `w` 不是參數。讓混合場可學會把「接縫該放哪裡」變成可最佳化的東西，
    而最佳化會把它推到任何能換到分數的地方——本專案已量過三道自然度門檻
    都被鑽過，可行域不是靠事後門檻守得住的。
    """

    name = "color_curve_dual"

    def __init__(self, weight: torch.Tensor, radius: float = 5.0,
                 pieces: int = 64, bound_mode: str = "advcf"):
        self.a = ColorCurveParam(radius=radius, pieces=pieces,
                                 bound_mode=bound_mode)
        self.b = ColorCurveParam(radius=radius, pieces=pieces,
                                 bound_mode=bound_mode)
        self.weight = weight
        self.radius = radius
        self.pieces = pieces

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self.a.reset(x01, seed)
        self.b.reset(x01, seed + 7919)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(device=x01.device, dtype=x01.dtype)
        return w * self.a.render(x01) + (1.0 - w) * self.b.render(x01)

    def params(self):
        return self.a.params() + self.b.params()

    def project(self) -> None:
        self.a.project()
        self.b.project()

    def state_dict(self):
        return {"a": self.a.state_dict(), "b": self.b.state_dict()}

    def load_state_dict(self, state):
        self.a.load_state_dict(state["a"])
        self.b.load_state_dict(state["b"])

    @property
    def stages(self):
        return [self.a, self.b]

    def set_amplitude(self, a):
        if not isinstance(a, (list, tuple)):
            a = [a, a]
        self.a.set_amplitude(a[0])
        self.b.set_amplitude(a[1])

    def step_scale(self) -> float:
        return self.a.step_scale()

    def set_radius(self, r: float) -> None:
        self.a.set_radius(r)
        self.b.set_radius(r)


# ---- 遮罩與權重場 ----

def expanded_box(x01: torch.Tensor, box):
    """`metrics.identity.embed_box` 實際裁下來的範圍，不是偵測器給的原框。

    身分嵌入是在擴框之後的區域上取的；上限若只算原框，最佳化可以把高色差
    推到「嵌入看得到、上限沒算到」的那一圈邊緣。
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    mx = DETECTOR_MARGIN * (x1 - x0) / (DETECTOR_IMAGE_SIZE - DETECTOR_MARGIN)
    my = DETECTOR_MARGIN * (y1 - y0) / (DETECTOR_IMAGE_SIZE - DETECTOR_MARGIN)
    h, w = x01.shape[-2:]
    return (max(0.0, x0 - mx / 2), max(0.0, y0 - my / 2),
            min(float(w), x1 + mx / 2), min(float(h), y1 + my / 2))


def box_support(x01: torch.Tensor, box) -> torch.Tensor:
    """硬遮罩，只給**量測**用（上限算在哪一塊）。不進 render。"""
    m = torch.zeros_like(x01[:, :1])
    a, b, c, d = (int(round(v)) for v in box)
    m[..., b:d, a:c] = 1.0
    if float(m.sum()) < 4:
        raise ValueError(f"主體框退化：{box}")
    return m


def smooth_field(x01: torch.Tensor, box, feather: float) -> torch.Tensor:
    """由人臉框長出的 C¹ 權重場。框內為 1，框外經 smoothstep 降到 0。

    `feather` 是過渡帶的寬度，單位是框半徑的倍數（0.35 即框半徑的 35%）。
    用 smoothstep 而不是「二值遮罩再高斯模糊」：後者在遮罩邊界上一階導數
    不連續，模糊只是把不連續攤開，仍會在強色差下看得出一圈。
    """
    h, w = x01.shape[-2:]
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = max((x1 - x0) / 2.0, 1.0), max((y1 - y0) / 2.0, 1.0)
    yy = torch.arange(h, device=x01.device, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(w, device=x01.device, dtype=torch.float32).view(1, -1)
    d = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2).sqrt()
    t = ((d - 1.0) / max(feather, 1e-6)).clamp(0.0, 1.0)
    field = 1.0 - (3.0 * t ** 2 - 2.0 * t ** 3)
    return field.view(1, 1, h, w)


def transition_band(field: torch.Tensor, lo: float = 0.05,
                    hi: float = 0.95) -> torch.Tensor:
    """過渡帶：`w` 既不是 0 也不是 1 的那一圈。接縫只可能長在這裡。"""
    band = ((field > lo) & (field < hi)).to(field.dtype)
    if float(band.sum()) < 16:
        raise ValueError("過渡帶太窄，接縫的上限量不到；把 --feather 調大")
    return band


# ---- 資料 ----

def load_images(root: Path, only):
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for cls in sorted(k for k in spec if k != "edits"):
        for img in sorted((root / cls).glob("*.png")):
            if only and img.stem not in only:
                continue
            out.append({"name": img.stem, "class": cls, "path": img})
    if not out:
        raise SystemExit(f"{root} 找不到影像（--images 過濾後為空）")
    return out


def write_rows(path: Path, rows) -> None:
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--budget-mode", required=True, choices=("chroma", "spatial"))
    ap.add_argument("--frame-cap", type=float, default=16.0,
                    help="整圖 ΔE00 上限。與 colour_curve_ours 相同，不要改")
    ap.add_argument("--face-cap", type=float, default=8.0,
                    help="人臉框內 ΔE00 上限。比整圖緊，這就是「不同預算」")
    ap.add_argument("--band-tv-cap", type=float, default=2.0,
                    help="過渡帶上位移場的 TV 上限（只有 spatial 用得到）")
    ap.add_argument("--feather", type=float, default=0.35,
                    help="過渡帶寬度，框半徑的倍數")
    ap.add_argument("--init", default="identity",
                    choices=("identity", "skin_tone"),
                    help="最佳化的起點。skin_tone 用該張自己的膚色色調（整圖"
                         "累積分布對到膚色區累積分布）當起點，而不是恆等")
    ap.add_argument("--init-strength", type=float, default=1.0,
                    help="起點在恆等與膚色色調之間的插值")
    ap.add_argument("--skin-radius", type=float, default=0.0,
                    help="> 0 時加一道 skin_colour 上限：原圖裡與膚色同色的"
                         "像素（不限位置）的 ΔE00 不得超過 --face-cap")
    ap.add_argument("--chroma-gain", type=float, default=0.0,
                    help="> 0 時加一道彩度上限：輸出 C* 的 p95 不得超過原圖的"
                         "p95 乘上這個倍率")
    ap.add_argument("--pieces", type=int, default=64)
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr-final-ratio", type=float, default=0.2)
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--lam-every", type=int, default=5)
    ap.add_argument("--check-every", type=int, default=10)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--attack-steps", type=int, default=50)
    ap.add_argument("--s-i", type=float, default=1.5)
    ap.add_argument("--noise-seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ip2p = IP2PWrapper(dtype=torch.float32)
    items = load_images(args.data, set(args.images) if args.images else None)

    rows = []
    for item in items:
        started = time.time()
        x = load_image_tensor(item["path"], device, size=RESOLUTION)
        boxes = face_boxes(x, device)
        if not boxes:
            raise SystemExit(f"{item['name']} 偵測不到臉；預算要分在臉與背景之間，"
                             "偵不到臉就沒有可分的界線，不要靜默退回單一預算")
        box = expanded_box(x, max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])))
        frame = torch.ones_like(x[:, :1])
        face = box_support(x, box)

        skin = (skin_colour_support(x, face, args.skin_radius)
                if args.skin_radius > 0 else None)
        c95 = float(chroma_p95(x))
        chroma_cap = c95 * args.chroma_gain if args.chroma_gain > 0 else 0.0

        if args.budget_mode == "spatial":
            field = smooth_field(x, box, args.feather)
            band = transition_band(field)
            carrier = ColorCurveDual(field, radius=args.radius,
                                     pieces=args.pieces, bound_mode="advcf")
        else:
            field = None
            band = None
            carrier = ColorCurveParam(radius=args.radius, pieces=args.pieces,
                                      bound_mode="advcf")
        carrier.reset(x, args.noise_seed)

        init_clipped = ""
        if args.init == "skin_tone":
            # 起點用的膚色支撐一律是顏色定義的那一塊；沒有給 --skin-radius
            # 時也要有一個，否則「膚色色調」沒有來源。
            base = skin if skin is not None else skin_colour_support(x, face, 12.0)
            slopes = skin_tone_slopes(x, base, args.pieces, args.init_strength)
            clipped = [apply_slopes(stage, slopes) for stage in carrier.stages]
            init_clipped = "|".join(f"{v:.4f}" for v in clipped)

        caps = [
            Cap("frame", lambda y: delta_e00_torch(x, y, frame),
                lambda y: delta_e00(x, y, frame), args.frame_cap),
            Cap("face_box", lambda y: delta_e00_torch(x, y, face),
                lambda y: delta_e00(x, y, face), args.face_cap),
        ]
        if args.budget_mode == "spatial" and args.band_tv_cap > 0:
            caps.append(
                Cap("band_tv", lambda y: tv_offset(lab_offset(x, y), band),
                    lambda y: float(tv_offset(lab_offset(x, y), band)),
                    args.band_tv_cap))
        if skin is not None:
            caps.append(
                Cap("skin_colour", lambda y: delta_e00_torch(x, y, skin),
                    lambda y: delta_e00(x, y, skin), args.face_cap))
        if chroma_cap > 0:
            caps.append(
                Cap("chroma_p95", lambda y: chroma_p95(y),
                    lambda y: float(chroma_p95(y)), chroma_cap))

        objective = FreeObjective(
            ip2p, x, box=box, k=4, steps=args.attack_steps,
            seed=args.noise_seed,
            weights={"id": 1.0, "enc": 0.5, "cond": 1.0},
            chain_steps=6, grad_steps=1, s_i=args.s_i, resample=True)

        stats = optimise_carrier(
            carrier, x, objective, steps=args.steps, lr=args.lr,
            caps=caps, rho=args.rho, lam_every=args.lam_every,
            check_every=args.check_every, log_every=args.log_every,
            lr_final_ratio=args.lr_final_ratio, probe_every=args.probe_every)

        with torch.no_grad():
            y = quantise(carrier.render(x))
            offset = lab_offset(x, y)
            row = {
                "image": item["name"], "class": item["class"], "arm": args.arm,
                "budget_mode": args.budget_mode,
                "carrier": carrier.name, "pieces": args.pieces,
                "radius": args.radius, "solver_steps": args.steps,
                "lr": args.lr, "frame_cap": args.frame_cap,
                "face_cap": args.face_cap,
                "band_tv_cap": args.band_tv_cap if args.budget_mode == "spatial" else "",
                "feather": args.feather if args.budget_mode == "spatial" else "",
                "init": args.init, "init_strength": args.init_strength,
                "init_clipped_frac": init_clipped,
                "skin_radius": args.skin_radius, "chroma_gain": args.chroma_gain,
                "chroma_p95_orig": round(c95, 4),
                "chroma_p95_cap": round(chroma_cap, 4) if chroma_cap else "",
                "chroma_p95_out": round(float(chroma_p95(y)), 4),
                "deltaE00_skin_colour": (round(float(delta_e00(x, y, skin)), 4)
                                         if skin is not None else ""),
                "skin_pixels_frac": (round(float(skin.mean()), 5)
                                     if skin is not None else ""),
                "solver_prompt": SOLVER_PROMPT[0],
                "solver_prompt_source": SOLVER_PROMPT[1],
                "deltaE00_frame": round(float(delta_e00(x, y, frame)), 4),
                "deltaE00_face_box": round(float(delta_e00(x, y, face)), 4),
                "tv_frame": round(float(tv_offset(offset, frame)), 5),
                "tv_band": (round(float(tv_offset(offset, band)), 5)
                            if band is not None else ""),
                "psnr": round(float(10 * torch.log10(
                    1.0 / (y - x).pow(2).mean())), 4),
                "linf": round(float((y - x).abs().max()), 5),
                "seconds": round(time.time() - started, 1),
                # `optimise_carrier` 的欄位原樣帶進來（`free_*`），
                # 好與顏色線既有的 CSV 對得起來。
                **{k: v for k, v in stats.items()},
            }
        save_image(x, args.out / f"{item['name']}__orig.png")
        save_image(y, args.out / f"{item['name']}__{args.arm}__def.png")
        if field is not None:
            save_image(field.repeat(1, 3, 1, 1),
                       args.out / f"{item['name']}__field.png")
        rows.append(row)
        write_rows(args.out / "results.csv", rows)
        print(f"[{args.arm}] {item['name']}  ΔE frame {row['deltaE00_frame']} "
              f"face {row['deltaE00_face_box']}  TV band {row['tv_band']}  "
              f"score {row['free_score_start']} → {row['free_score_end']}  "
              f"違反 {row['free_cap_violations']}  {row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 張 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
