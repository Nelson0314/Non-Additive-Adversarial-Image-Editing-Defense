"""色度平面的全域扭曲：保住膚色所在的那一塊顏色，其餘色相大幅移動。

為什麼往這個方向走
────────────────────────────────────────────────────────────────────
`curve_dual_chroma` 的產物自然（溫和暖粉色調、膚色保住、構造上零接縫），
但防禦強度只與 `colour_curve_ours` 打平。瓶頸看得出來：它的載體是**逐 RGB
通道**的單調曲線，而膚色與背景在 RGB 通道值上大量重疊——壓低膚色的色差，
就等於壓低所有共用那幾段通道值的顏色。`man_00` 整圖只到 ΔE00 7.44 而上限
是 16，就是這件事。

**分開膚色與其他顏色的正確軸是色相，不是通道強度。** 本檔把映射改成定義在
CIELAB 的 `(a, b)` 平面上：

    (a, b) → (a, b) + d(a, b)          d 由 G×G 個錨點的位移場平滑內插
    L      → 100 · F(L/100)            F 是單調分段線性曲線

膚色在 `(a, b)` 上是一塊緊緻的區域，所以臉那道上限只釘住它附近的錨點，
其餘錨點仍可用滿整圖的預算。

三件事由構造保證
────────────────────────────────────────────────────────────────────
1. **映射仍然全域。** 輸出只依賴該像素自己的顏色，與位置無關，所以同一個
   輸入色一定映到同一個輸出色——**接縫在構造上不存在**，與
   `curve_dual_chroma` 同一個保證。
2. **沒有色帶。** 位移場是 Gaussian RBF 的單位分割內插（Shepard 形式），
   在 `(a, b)` 平面上處處 C^∞；亮度那一條是單調的，不會出現色調反轉。
3. **振幅有硬上界。** 單位分割使 `|d| ≤ max|w|`，而 `w` 每一步投影回
   `--warp-radius`，所以位移場的幅度不會靠最佳化跑掉。

約束
────────────────────────────────────────────────────────────────────
除了整圖與臉框的 ΔE00，另外兩道是本專案已經量到的「色偏難看的兩種情形」：

| 上限 | 擋什麼 |
|---|---|
| `frame` ΔE00 ≤ `--frame-cap` | 整體改動量 |
| `face_box` ΔE00 ≤ `--face-cap` | 臉框內的色偏 |
| `skin_colour` ΔE00 ≤ `--face-cap` | **與原膚色同色**的像素，不限位置 |
| `chroma_p95` ≤ 原圖 p95 × `--chroma-gain` | 高飽和 |

`skin_colour` 是**顏色定義**的支撐，不是遮罩：原圖裡 `(a, b)` 落在膚色群心
半徑 `--skin-radius` 內的像素都算。這樣「背景裡與膚色同色的東西也一起被
保護」，而不是只有臉框那一塊。

約束一律進損失（增廣 Lagrange），不做事後投影；可行性在量化後的 PNG 上檢查。

產出（版面與 `defence_run.py` 相同）
────────────────────────────────────────────────────────────────────
    {out}/{image}__orig.png
    {out}/{image}__{arm}__def.png
    {out}/{image}__warp.png      (a,b) 平面的位移場視覺化
    {out}/results.csv
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

from colour_support import chroma_p95, skin_colour_support  # noqa: E402
from curve_budget_defence import (  # noqa: E402
    box_support, expanded_box, load_images, write_rows,
)
from src.defense.color_amplitude import delta_e00  # noqa: E402
from src.defense.delta_e_torch import delta_e00_torch  # noqa: E402
from src.defense.immunise import Cap, optimise_carrier, quantise  # noqa: E402
from src.defense.instruction_free import FreeObjective  # noqa: E402
from src.defense.ncf_param import lab_to_rgb, rgb_to_lab  # noqa: E402
from src.defense.uniformity import lab_offset, tv_offset  # noqa: E402
from src.metrics.identity import face_boxes  # noqa: E402
from src.models.ip2p import IP2PWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512
SOLVER_PROMPT = ("", "無文字條件：三個項都不經過 text encoder")


class AbWarpParam:
    """`(a,b)` 平面的 RBF 位移場 ＋ 單調亮度曲線。全域映射，與位置無關。

    兩組參數都以**無量綱的原始量**存放，再乘回各自的尺度：

    | 原始量 | 形狀 | 可行域 | 實際量 |
    |---|---|---|---|
    | `w_raw` | (G·G, 2) | 範數 ≤ 1 | `w = warp_radius · w_raw`（Lab 單位） |
    | `th_raw` | (K,) | `[-1, 1]` | 斜率 `= span^th_raw / K`，`span = 1 + l_radius` |

    **這樣寫是為了讓兩組參數共用同一個學習率。** `w` 的自然尺度是幾十個 Lab
    單位、斜率的自然尺度是 `1/K ≈ 0.06`，直接丟進同一個 Adam 的話其中一組
    幾乎不動、另一組每一步都撞邊界，而兩種症狀在逐步損失上都看不出來。
    """

    name = "ab_warp"

    def __init__(self, grid: int = 7, extent: float = 90.0,
                 warp_radius: float = 30.0, pieces: int = 16,
                 l_radius: float = 0.6):
        self.grid = int(grid)
        self.extent = float(extent)
        self.warp_radius = float(warp_radius)
        self.pieces = int(pieces)
        self.l_radius = float(l_radius)
        axis = torch.linspace(-extent, extent, self.grid)
        aa, bb = torch.meshgrid(axis, axis, indexing="ij")
        self.anchors = torch.stack([aa.reshape(-1), bb.reshape(-1)], -1)  # (G²,2)
        # σ 取格距，讓相鄰錨點的核有足夠重疊；太小會在錨點之間出現起伏。
        self.sigma = float(2 * extent / max(self.grid - 1, 1))
        self.w_raw = None
        self.th_raw = None

    # ---- 介面 ----

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        dev, dt = x01.device, torch.float32
        self.anchors = self.anchors.to(dev, dt)
        self.w_raw = torch.zeros((self.anchors.shape[0], 2), device=dev, dtype=dt,
                                 requires_grad=True)
        self.th_raw = torch.zeros((self.pieces,), device=dev, dtype=dt,
                                  requires_grad=True)

    def params(self):
        return [self.w_raw, self.th_raw]

    @torch.no_grad()
    def project(self) -> None:
        norm = self.w_raw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self.w_raw.mul_((1.0 / norm).clamp(max=1.0))
        self.th_raw.clamp_(-1.0, 1.0)

    def state_dict(self):
        return {"w_raw": self.w_raw.detach().clone(),
                "th_raw": self.th_raw.detach().clone()}

    def load_state_dict(self, state):
        self.w_raw = state["w_raw"].clone().requires_grad_(True)
        self.th_raw = state["th_raw"].clone().requires_grad_(True)

    @property
    def stages(self):
        return [self]

    def set_amplitude(self, a):
        if isinstance(a, (list, tuple)):
            a = a[0]
        self.amplitude = float(a)

    def step_scale(self) -> float:
        return 2.0          # 兩組原始量的可行域寬度都是 2

    def set_radius(self, r: float) -> None:
        self.warp_radius = float(r)

    # ---- 構造 ----

    @property
    def w(self) -> torch.Tensor:
        return self.warp_radius * self.w_raw

    @property
    def theta(self) -> torch.Tensor:
        span = 1.0 + max(0.0, self.l_radius)
        return torch.pow(torch.tensor(span, device=self.th_raw.device,
                                      dtype=self.th_raw.dtype),
                         self.th_raw) / self.pieces

    def displacement(self, ab: torch.Tensor) -> torch.Tensor:
        """`ab` (N,2) → (N,2)。Shepard 形式的單位分割，故 `|d| ≤ max|w|`。"""
        d2 = torch.cdist(ab, self.anchors).pow(2)                 # (N, G²)
        phi = torch.softmax(-d2 / (2.0 * self.sigma ** 2), dim=-1)
        return phi @ self.w

    def lightness(self, l01: torch.Tensor) -> torch.Tensor:
        """單調分段線性曲線，值域固定 `[0,1]`——不會出現色調反轉。"""
        k = self.pieces
        th = self.theta.to(l01.dtype)
        idx = (l01 * k).floor().clamp_(0, k - 1).long()
        cum = torch.cat([th.new_zeros(1), th.cumsum(0)[:-1]])
        out = cum[idx] / k + (l01 - idx.to(l01.dtype) / k) * th[idx]
        return out * (k / th.sum())

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        lab = rgb_to_lab(x01.clamp(0, 1).float())
        n, _, h, wd = lab.shape
        L = lab[:, 0]
        ab = lab[:, 1:].permute(0, 2, 3, 1).reshape(-1, 2)
        moved = ab + self.displacement(ab)
        moved = moved.reshape(n, h, wd, 2).permute(0, 3, 1, 2)
        L2 = 100.0 * self.lightness((L / 100.0).clamp(0, 1)).unsqueeze(1)
        out = lab_to_rgb(torch.cat([L2, moved], 1))
        return out.clamp(0, 1).to(x01.dtype)


# ---- 分通道預算 ----

#: (名稱, Lab 通道, 方向)。方向 +1 量正向位移、−1 量負向、0 量絕對值。
#: 依據：偏藍的改動最容易被歸給光源（Winkler et al. 2015, Curr. Biol.;
#: Pearce et al. 2014, PLoS One），而紅／粉／洋紅／黃／暖黃落在膚色所在或
#: 緊鄰的象限（偏好膚色中心 a*b* ≈ (21, 24)，Zeng & Luo），與膚色記憶色衝突。
CHANNEL_CAPS = (("a_pos", 1, 1), ("a_neg", 1, -1), ("b_pos", 2, 1),
                ("b_neg", 2, -1), ("l_abs", 0, 0))


def channel_shift_p95(x01, y01, channel, sign, q=0.95):
    """逐像素 Lab 位移在某通道、某方向的 p95。沒有往該方向動的像素記 0。"""
    d = rgb_to_lab(y01.clamp(0, 1).float()) - rgb_to_lab(x01.clamp(0, 1).float())
    v = d[:, channel].reshape(-1)
    v = v.abs() if sign == 0 else torch.relu(sign * v)
    return torch.quantile(v, q)


# ---- 視覺化 ----


def warp_picture(carrier: AbWarpParam, device) -> torch.Tensor:
    """把 `(a,b)` 平面畫成一張圖：底色是該點的顏色，箭頭長度用亮度表示位移。"""
    n = 256
    axis = torch.linspace(-carrier.extent, carrier.extent, n, device=device)
    bb, aa = torch.meshgrid(axis, axis, indexing="ij")
    ab = torch.stack([aa.reshape(-1), bb.reshape(-1)], -1)
    with torch.no_grad():
        mag = carrier.displacement(ab).norm(dim=-1).reshape(1, 1, n, n)
        lab = torch.cat([torch.full((1, 1, n, n), 70.0, device=device),
                         aa.reshape(1, 1, n, n), bb.reshape(1, 1, n, n)], 1)
        base = lab_to_rgb(lab).clamp(0, 1)
        shade = (mag / max(carrier.warp_radius, 1e-6)).clamp(0, 1)
        return (base * (0.35 + 0.65 * shade)).clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--extent", type=float, default=90.0)
    ap.add_argument("--warp-radius", type=float, default=30.0)
    ap.add_argument("--pieces", type=int, default=16)
    ap.add_argument("--l-radius", type=float, default=0.6)
    ap.add_argument("--frame-cap", type=float, default=16.0)
    ap.add_argument("--face-cap", type=float, default=8.0)
    ap.add_argument("--skin-radius", type=float, default=12.0,
                    help="膚色群心的半徑（Lab 單位），定義 skin_colour 支撐")
    ap.add_argument("--chroma-gain", type=float, default=1.15,
                    help="輸出彩度 p95 相對原圖的上限倍率。高飽和是本專案量到的"
                         "兩種難看情形之一")
    for name, _, _ in CHANNEL_CAPS:
        ap.add_argument(f"--{name.replace('_', '-')}-cap", type=float, default=0.0,
                        help=f"> 0 時加一道上限：逐像素 Lab 位移 {name} 的 p95")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lr", type=float, default=0.02,
                    help="兩組參數都是無量綱原始量，可行域寬度都是 2，共用一個 lr")
    ap.add_argument("--lr-final-ratio", type=float, default=0.2)
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--lam-every", type=int, default=5)
    ap.add_argument("--check-every", type=int, default=10)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--attack-steps", type=int, default=50)
    ap.add_argument("--s-i", type=float, default=1.5)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--objective", default="free", choices=("free", "comm"),
                    help="free：現行 FreeObjective；comm：等變殘差（comm_objective.py）")
    ap.add_argument("--init", default="zero", choices=("zero", "random"),
                    help="random：w_raw ~ N(0, init_std²) 後投影、th_raw ~ U(−init_std, init_std)")
    ap.add_argument("--init-std", type=float, default=0.1)
    ap.add_argument("--lpips-ref-arm", default="",
                    help="非空時加一道逐張輸入 LPIPS 上限：該臂防禦圖對原圖的 piq LPIPS ＋ 容差")
    ap.add_argument("--lpips-tolerance", type=float, default=0.0025)
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
            raise SystemExit(f"{item['name']} 偵測不到臉；膚色群心定不出來，"
                             "不要靜默退回單一預算")
        box = expanded_box(x, max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])))
        frame = torch.ones_like(x[:, :1])
        face = box_support(x, box)
        skin = skin_colour_support(x, face, args.skin_radius)
        c95 = float(chroma_p95(x))
        chroma_cap = c95 * args.chroma_gain

        carrier = AbWarpParam(grid=args.grid, extent=args.extent,
                              warp_radius=args.warp_radius, pieces=args.pieces,
                              l_radius=args.l_radius)
        carrier.reset(x, args.noise_seed)
        if args.init == "random":
            g = torch.Generator(device="cpu").manual_seed(args.noise_seed)
            with torch.no_grad():
                carrier.w_raw.copy_((args.init_std * torch.randn(
                    carrier.w_raw.shape, generator=g)).to(device))
                carrier.th_raw.copy_(((torch.rand(carrier.th_raw.shape, generator=g) * 2 - 1)
                                      * args.init_std).to(device))
            carrier.project()

        caps = [
            Cap("frame", lambda y: delta_e00_torch(x, y, frame),
                lambda y: delta_e00(x, y, frame), args.frame_cap),
            Cap("face_box", lambda y: delta_e00_torch(x, y, face),
                lambda y: delta_e00(x, y, face), args.face_cap),
            Cap("skin_colour", lambda y: delta_e00_torch(x, y, skin),
                lambda y: delta_e00(x, y, skin), args.face_cap),
            *[Cap(f"shift_{n}", lambda y, c=c, sg=sg: channel_shift_p95(x, y, c, sg),
                  lambda y, c=c, sg=sg: float(channel_shift_p95(x, y, c, sg)),
                  getattr(args, f"{n}_cap"))
              for n, c, sg in CHANNEL_CAPS if getattr(args, f"{n}_cap") > 0],
            Cap("chroma_p95", lambda y: chroma_p95(y),
                lambda y: float(chroma_p95(y)), chroma_cap),
        ]

        extra = {"objective": args.objective, "init": args.init,
                 "init_std": args.init_std if args.init == "random" else ""}
        lp = None
        if args.lpips_ref_arm or args.objective == "comm":
            import piq
            lp = piq.LPIPS().to(device).eval()
            lp.requires_grad_(False)
        if args.lpips_ref_arm:
            import hashlib
            ref_png = Path("lab/runs/defence") / args.lpips_ref_arm /                 f"{item['name']}__{args.lpips_ref_arm}__def.png"
            with torch.no_grad():
                t_i = float(lp(load_image_tensor(ref_png, device, size=RESOLUTION), x).mean())
            caps.append(Cap("input_lpips", lambda y: lp(y, x).mean(),
                            lambda y: float(lp(y, x).mean()), t_i + args.lpips_tolerance))
            extra.update(lpips_ref_arm=args.lpips_ref_arm, lpips_ref=round(t_i, 6),
                         lpips_ref_sha256=hashlib.sha256(ref_png.read_bytes()).hexdigest())

        free = FreeObjective(
            ip2p, x, box=box, k=4, steps=args.attack_steps, seed=args.noise_seed,
            weights={"id": 1.0, "enc": 0.5, "cond": 1.0},
            chain_steps=6, grad_steps=1, s_i=args.s_i, resample=True)
        from comm_objective import CommObjective, align_weight, assert_nonzero_grad
        diag = CommObjective(free, carrier, x, lp) if lp is not None else None
        if args.objective == "comm":
            objective = CommObjective(free, carrier, x, lp)
            assert_nonzero_grad(carrier, lambda: objective.eval_terms(carrier.render(x))["comm"])
            extra.update({k: round(v, 6) for k, v in
                          align_weight(objective, free, carrier, x).items()})
        else:
            objective = free
        if diag is not None:
            with torch.no_grad():
                extra["comm_val_start"] = round(float(diag.eval_terms(carrier.render(x))["comm"]), 6)

        stats = optimise_carrier(
            carrier, x, objective, steps=args.steps, lr=args.lr, caps=caps,
            rho=args.rho, lam_every=args.lam_every, check_every=args.check_every,
            log_every=args.log_every, lr_final_ratio=args.lr_final_ratio,
            probe_every=args.probe_every)
        if diag is not None:
            with torch.no_grad():
                yq = quantise(carrier.render(x))
                extra["comm_val_end"] = round(float(diag.eval_terms(yq)["comm"]), 6)
                extra["lpips_out"] = round(float(lp(yq, x).mean()), 6)

        with torch.no_grad():
            y = quantise(carrier.render(x))
            off = lab_offset(x, y)
            row = {
                "image": item["name"], "class": item["class"], "arm": args.arm,
                "carrier": carrier.name, "grid": args.grid, "extent": args.extent,
                "warp_radius": args.warp_radius, "pieces": args.pieces,
                "l_radius": args.l_radius, "solver_steps": args.steps, "lr": args.lr,
                "frame_cap": args.frame_cap, "face_cap": args.face_cap,
                "skin_radius": args.skin_radius, "chroma_gain": args.chroma_gain,
                "chroma_p95_orig": round(c95, 4),
                "chroma_p95_cap": round(chroma_cap, 4),
                "chroma_p95_out": round(float(chroma_p95(y)), 4),
                "skin_pixels_frac": round(float(skin.mean()), 5),
                **{f"shift_{n}_p95": round(float(channel_shift_p95(x, y, c, sg)), 4)
                   for n, c, sg in CHANNEL_CAPS},
                **{f"shift_{n}_cap": getattr(args, f"{n}_cap") for n, _, _ in CHANNEL_CAPS},
                "solver_prompt": SOLVER_PROMPT[0],
                "solver_prompt_source": SOLVER_PROMPT[1],
                "deltaE00_frame": round(float(delta_e00(x, y, frame)), 4),
                "deltaE00_face_box": round(float(delta_e00(x, y, face)), 4),
                "deltaE00_skin_colour": round(float(delta_e00(x, y, skin)), 4),
                "tv_frame": round(float(tv_offset(off, frame)), 5),
                "warp_max": round(float(carrier.w.norm(dim=-1).max()), 3),
                "l_slope_min": round(float(carrier.theta.min() * args.pieces), 4),
                "l_slope_max": round(float(carrier.theta.max() * args.pieces), 4),
                "psnr": round(float(10 * torch.log10(1.0 / (y - x).pow(2).mean())), 4),
                "linf": round(float((y - x).abs().max()), 5),
                "seconds": round(time.time() - started, 1),
                **extra,
                **{k: v for k, v in stats.items()},
            }
        save_image(x, args.out / f"{item['name']}__orig.png")
        save_image(y, args.out / f"{item['name']}__{args.arm}__def.png")
        torch.save(carrier.state_dict(), args.out / f"{item['name']}__carrier.pt")
        save_image(warp_picture(carrier, device),
                   args.out / f"{item['name']}__warp.png")
        rows.append(row)
        write_rows(args.out / "results.csv", rows)
        print(f"[{args.arm}] {item['name']}  ΔE frame {row['deltaE00_frame']} "
              f"face {row['deltaE00_face_box']} skin {row['deltaE00_skin_colour']}  "
              f"C95 {row['chroma_p95_out']}/{row['chroma_p95_cap']}  "
              f"違反 {row['free_cap_violations']}  {row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 張 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
