"""color：CIELAB 色度平面的全域映射，以等變殘差目標對 ip2p 最佳化。

載體（全域映射，輸出只依賴該像素自己的顏色，與位置無關）

    (a, b) → (a, b) + d(a, b)     d 為 7×7 個錨點位移的 Gaussian RBF 單位分割內插，|d| ≤ warp_radius
    L      → 100 · F(L/100)       F 為 16 段單調分段線性曲線

目標（等變殘差）

    comm(θ; ξ) = LPIPS( N_ξ(T_θ(x)), T_θ(N_ξ(x)) )
    score(θ)   = − w_comm · comm / c0

N_ξ 為無文字的 ip2p 短鏈（`FreeObjective.null_edit`，6 步、最後 1 步回傳梯度、s_i 1.5），噪聲每步重抽。
編輯器對 T 等變時 comm = 0；comm 量的是色調穿過編輯之後剩下的差。c0 為起點在固定驗證抽樣上的值，
w_comm 使起點梯度範數等於 FreeObjective。恆等起點的 comm 與梯度皆為 0，因此用非恆等起點。

上限（增廣 Lagrange，可行性在量化後的 PNG 上檢查）

| 上限 | 值 |
|---|---|
| 整圖 ΔE00 | ≤ 32 |
| 臉框 ΔE00、與原膚色同色像素的 ΔE00 | ≤ 16 |
| 彩度 p95 | ≤ 原圖 × 2.0 |
| 逐像素 Lab 位移（p95 與精確最大值） | a*＋ ≤ 4、a*－ ≤ 15、b*＋ ≤ 4、b*－ ≤ 25、\\|ΔL*\\| ≤ 15 |
| 對原圖的 LPIPS | ≤ `data/color_lpips_ref.csv` 的逐張值 ＋ 0.0025 |

產出：{out}/{image}__orig.png、{image}__color__def.png、{image}__warp.png、{image}__carrier.pt、results.csv
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

from src.defense.color_amplitude import delta_e00  # noqa: E402
from src.defense.delta_e_torch import delta_e00_torch  # noqa: E402
from src.defense.immunise import Cap, optimise_carrier, quantise  # noqa: E402
from src.defense.instruction_free import FreeObjective  # noqa: E402
from src.defense.ncf_param import lab_to_rgb, rgb_to_lab  # noqa: E402
from src.defense.uniformity import lab_offset, tv_offset  # noqa: E402
from src.metrics.identity import DETECTOR_IMAGE_SIZE, DETECTOR_MARGIN, face_boxes  # noqa: E402
from src.models.ip2p import IP2PWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512
ARM = "color"


# ---- 載體 ----

class ColorMap:
    """`(a,b)` 平面的 RBF 位移場 ＋ 單調亮度曲線。

    兩組參數以無量綱原始量存放（`w_raw` 範數 ≤ 1、`th_raw` ∈ [−1, 1]），再乘回各自的尺度，
    使兩組共用同一個學習率：`w = warp_radius · w_raw`，斜率 `= (1 + l_radius)^th_raw / K`。
    """

    name = ARM

    def __init__(self, grid=7, extent=90.0, warp_radius=80.0, pieces=16, l_radius=0.6):
        self.grid, self.extent, self.warp_radius = int(grid), float(extent), float(warp_radius)
        self.pieces, self.l_radius = int(pieces), float(l_radius)
        axis = torch.linspace(-extent, extent, self.grid)
        aa, bb = torch.meshgrid(axis, axis, indexing="ij")
        self.anchors = torch.stack([aa.reshape(-1), bb.reshape(-1)], -1)
        self.sigma = float(2 * extent / max(self.grid - 1, 1))  # 格距，相鄰核充分重疊
        self.w_raw = self.th_raw = None
        self.box = None  # (G², 4)：每個錨點位移的 [a 下限, a 上限, b 下限, b 上限]（Lab 單位）

    def set_box(self, lo_a, hi_a, lo_b, hi_b):
        """把每個錨點的位移夾在方框內。像素位移是錨點位移的凸組合，所以逐像素位移由構造落在
        各錨點方框的聯集內；方框全部相同時即逐像素精確上限。參數為長度 G² 的張量或純量。"""
        n = self.anchors.shape[0]
        cols = [torch.as_tensor(v, dtype=torch.float32).expand(n) for v in (lo_a, hi_a, lo_b, hi_b)]
        self.box = torch.stack(cols, -1)

    def reset(self, x01, seed):
        dev = x01.device
        self.anchors = self.anchors.to(dev, torch.float32)
        self.w_raw = torch.zeros((self.anchors.shape[0], 2), device=dev, requires_grad=True)
        self.th_raw = torch.zeros((self.pieces,), device=dev, requires_grad=True)

    def params(self):
        return [self.w_raw, self.th_raw]

    @torch.no_grad()
    def project(self):
        self.w_raw.mul_((1.0 / self.w_raw.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0))
        if self.box is not None:
            b = self.box.to(self.w_raw.device) / self.warp_radius
            self.w_raw[:, 0].copy_(torch.maximum(torch.minimum(self.w_raw[:, 0], b[:, 1]), b[:, 0]))
            self.w_raw[:, 1].copy_(torch.maximum(torch.minimum(self.w_raw[:, 1], b[:, 3]), b[:, 2]))
        self.th_raw.clamp_(-1.0, 1.0)

    def state_dict(self):
        return {"w_raw": self.w_raw.detach().clone(), "th_raw": self.th_raw.detach().clone()}

    def load_state_dict(self, state):
        self.w_raw = state["w_raw"].clone().requires_grad_(True)
        self.th_raw = state["th_raw"].clone().requires_grad_(True)

    # optimise_carrier 的介面
    @property
    def stages(self):
        return [self]

    def set_amplitude(self, a):
        self.amplitude = float(a[0] if isinstance(a, (list, tuple)) else a)

    def step_scale(self):
        return 2.0

    def set_radius(self, r):
        self.warp_radius = float(r)

    @property
    def w(self):
        return self.warp_radius * self.w_raw

    @property
    def theta(self):
        span = torch.tensor(1.0 + max(0.0, self.l_radius), device=self.th_raw.device)
        return torch.pow(span, self.th_raw) / self.pieces

    def displacement(self, ab):
        phi = torch.softmax(-torch.cdist(ab, self.anchors).pow(2) / (2.0 * self.sigma ** 2), dim=-1)
        return phi @ self.w

    def lightness(self, l01):
        k, th = self.pieces, self.theta.to(l01.dtype)
        idx = (l01 * k).floor().clamp_(0, k - 1).long()
        cum = torch.cat([th.new_zeros(1), th.cumsum(0)[:-1]])
        return (cum[idx] / k + (l01 - idx.to(l01.dtype) / k) * th[idx]) * (k / th.sum())

    def render(self, x01):
        lab = rgb_to_lab(x01.clamp(0, 1).float())
        n, _, h, wd = lab.shape
        ab = lab[:, 1:].permute(0, 2, 3, 1).reshape(-1, 2)
        moved = (ab + self.displacement(ab)).reshape(n, h, wd, 2).permute(0, 3, 1, 2)
        L2 = 100.0 * self.lightness((lab[:, 0] / 100.0).clamp(0, 1)).unsqueeze(1)
        return lab_to_rgb(torch.cat([L2, moved], 1)).clamp(0, 1).to(x01.dtype)


def warp_picture(carrier, device):
    """`(a,b)` 平面：底色為該點顏色，亮度表示位移幅度。"""
    n = 256
    axis = torch.linspace(-carrier.extent, carrier.extent, n, device=device)
    bb, aa = torch.meshgrid(axis, axis, indexing="ij")
    ab = torch.stack([aa.reshape(-1), bb.reshape(-1)], -1)
    with torch.no_grad():
        mag = carrier.displacement(ab).norm(dim=-1).reshape(1, 1, n, n)
        lab = torch.cat([torch.full((1, 1, n, n), 70.0, device=device),
                         aa.reshape(1, 1, n, n), bb.reshape(1, 1, n, n)], 1)
        shade = (mag / max(carrier.warp_radius, 1e-6)).clamp(0, 1)
        return (lab_to_rgb(lab).clamp(0, 1) * (0.35 + 0.65 * shade)).clamp(0, 1)


# ---- 量測 ----

#: (名稱, Lab 通道, 方向)：+1 正向位移、−1 負向、0 絕對值
CHANNEL_CAPS = (("a_pos", 1, 1), ("a_neg", 1, -1), ("b_pos", 2, 1), ("b_neg", 2, -1), ("l_abs", 0, 0))


def _shift(x01, y01, channel, sign):
    d = rgb_to_lab(y01.clamp(0, 1).float()) - rgb_to_lab(x01.clamp(0, 1).float())
    v = d[:, channel].reshape(-1)
    return v.abs() if sign == 0 else torch.relu(sign * v)


def channel_shift_p95(x01, y01, channel, sign, q=0.95):
    """逐像素 Lab 位移在某通道、某方向的分位數；沒有往該方向動的像素記 0。"""
    return torch.quantile(_shift(x01, y01, channel, sign), q)


def channel_shift_max(x01, y01, channel, sign):
    """同上的精確最大值：p95 管不到佔比 1–3% 的小區域（如嘴唇）。"""
    return _shift(x01, y01, channel, sign).max()


def chroma_p95(x01, q=0.95):
    lab = rgb_to_lab(x01.clamp(0, 1).float())
    return torch.quantile(lab[:, 1:].pow(2).sum(1).clamp_min(1e-12).sqrt().reshape(-1), q)


def skin_centre(x01, face):
    """臉框內 `(a,b)` 的中位數（平均會被頭髮與背景的離群值拉走）。"""
    ab = rgb_to_lab(x01.clamp(0, 1).float())[:, 1:]
    sel = ab[0].permute(1, 2, 0)[face[0, 0] > 0.5]
    if sel.numel() == 0:
        raise ValueError("臉框內沒有像素，膚色群心定不出來")
    return sel.median(0).values


def skin_colour_support(x01, face, radius=12.0):
    """原圖中 `(a,b)` 落在臉框膚色中位數半徑 `radius` 內的像素，不限位置。"""
    ab = rgb_to_lab(x01.clamp(0, 1).float())[:, 1:]
    centre = skin_centre(x01, face).view(1, 2, 1, 1)
    return ((ab - centre).pow(2).sum(1, keepdim=True).clamp_min(1e-12).sqrt() <= radius).to(ab.dtype)


def skin_scaled_box(anchors, centre, base, gain=2.0, r0=20.0, r1=60.0):
    """色偏上限依錨點離膚色中心的距離放大：r ≤ r0 維持 base，r ≥ r1 為 base × gain，其間線性。
    只放大冷色方向（a*－、b*－）；暖色方向（a*＋、b*＋）維持 base。回傳 (lo_a, hi_a, lo_b, hi_b)。"""
    r = (anchors - centre.to(anchors.device).view(1, 2)).norm(dim=-1)
    s = 1.0 + (gain - 1.0) * ((r - r0) / (r1 - r0)).clamp(0.0, 1.0)
    return (-base["a_neg"] * s, torch.full_like(s, base["a_pos"]),
            -base["b_neg"] * s, torch.full_like(s, base["b_pos"]))


def expanded_box(x01, box):
    """身分嵌入實際裁切的範圍（偵測框外擴 DETECTOR_MARGIN）。"""
    x0, y0, x1, y1 = (float(v) for v in box)
    mx = DETECTOR_MARGIN * (x1 - x0) / (DETECTOR_IMAGE_SIZE - DETECTOR_MARGIN)
    my = DETECTOR_MARGIN * (y1 - y0) / (DETECTOR_IMAGE_SIZE - DETECTOR_MARGIN)
    h, w = x01.shape[-2:]
    return (max(0.0, x0 - mx / 2), max(0.0, y0 - my / 2), min(float(w), x1 + mx / 2), min(float(h), y1 + my / 2))


def box_support(x01, box):
    m = torch.zeros_like(x01[:, :1])
    a, b, c, d = (int(round(v)) for v in box)
    m[..., b:d, a:c] = 1.0
    if float(m.sum()) < 4:
        raise ValueError(f"主體框退化：{box}")
    return m


# ---- 目標 ----

VAL_DRAWS = (90001, 90002)


class CommObjective:
    def __init__(self, free, carrier, x01, lpips):
        self.free, self.carrier, self.x, self.lpips = free, carrier, x01, lpips
        self.w_comm, self.c0 = 1.0, 1.0

    def _comm_at(self, y, draw):
        keep, self.free.draws = self.free.draws, draw
        try:
            with torch.no_grad():
                nx = self.free.null_edit(self.x).float().clamp(0, 1)
            ny = self.free.null_edit(y).float().clamp(0, 1)
            return self.lpips(ny, self.carrier.render(nx).float()).mean()
        finally:
            self.free.draws = keep

    def terms(self, y):
        self.free.draws += 1
        return {"comm": self._comm_at(y, self.free.draws)}

    def score(self, y):
        return -self.w_comm * self.terms(y)["comm"] / self.c0

    def eval_terms(self, y):
        return {"comm": torch.stack([self._comm_at(y, d) for d in VAL_DRAWS]).mean()}

    def eval_score(self, y):
        return -self.w_comm * self.eval_terms(y)["comm"] / self.c0


class XAttnScore:
    """攻擊端 UNet 對類別詞（man／woman）的交叉注意力質量，相對原圖的比值，最小化。

    文字條件只有類別詞；每個 `attn2` 層取該詞 token 的注意力機率在像素與 head 上的平均，
    除以原圖在同一組 (t, ε) 下的值後跨層平均（起點約 1）。每步 2 組隨機 (t, ε)，t ∈ [100, 900)；
    驗證用固定 4 組。實作沿用 `style_prompt_defence.CrossAttnObjective` 的量測。
    """

    def __init__(self, ip2p, x01, class_word, seed):
        from style_prompt_defence import CrossAttnObjective
        ns = argparse.Namespace(class_word=class_word, attn_tmin=100, attn_tmax=900, attn_k=2,
                                attn_val_k=4, val_seed=777, noise_seed=seed, a_adv=1.0)
        self.o = CrossAttnObjective(ip2p, x01, ns)
        self.ip2p, self.w = ip2p, 1.0

    def _ratio(self, y, draws):
        vals = []
        for t, eps in draws:
            with torch.no_grad():
                ref = [m.detach() for m in self.o._mass(self.o.z_ref, self.o.c_ref, t, eps)]
            post = self.ip2p.posterior_mean(y, use_ckpt=True).float()
            cur = self.o._mass(post * self.ip2p.scaling_factor, post, t, eps)
            vals.append(torch.stack([a / b for a, b in zip(cur, ref)]).mean())
            self.o.maps = []
        return torch.stack(vals).mean()

    def terms(self, y):
        return {"xattn": self._ratio(y, [self.o._draw(self.o.gen) for _ in range(2)])}

    def score(self, y):
        return self.w * self.terms(y)["xattn"]

    def eval_terms(self, y):
        return {"xattn": self._ratio(y, self.o.val_draws)}

    def eval_score(self, y):
        return self.w * self.eval_terms(y)["xattn"]


def _grad_norm(carrier, fn):
    for p in carrier.params():
        p.grad = None
    fn().backward()
    grads = [p.grad for p in carrier.params()]
    for p in carrier.params():
        p.grad = None
    if any(g is None or float(g.abs().max()) <= 0 for g in grads):
        raise RuntimeError("起點處有一組參數梯度為零")
    return float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads)))


def align_weight(comm, free, carrier, x01):
    with torch.no_grad():
        comm.c0 = float(comm.eval_terms(carrier.render(x01))["comm"])
    if not comm.c0 > 0:
        raise RuntimeError(f"起點 comm = {comm.c0}，不能正規化")
    comm.w_comm = 1.0
    g_comm = _grad_norm(carrier, lambda: comm.eval_score(carrier.render(x01)))
    g_free = _grad_norm(carrier, lambda: free.eval_score(carrier.render(x01)))
    comm.w_comm = g_free / g_comm
    return {"comm_c0": comm.c0, "comm_grad_norm_start": g_comm,
            "free_grad_norm_start": g_free, "comm_weight": comm.w_comm}


# ---- 資料 ----

def load_images(root: Path, only):
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    out = [{"name": img.stem, "class": cls, "path": img}
           for cls in sorted(k for k in spec if k != "edits")
           for img in sorted((root / cls).glob("*.png")) if not only or img.stem in only]
    if not out:
        raise SystemExit(f"{root} 找不到影像（--images 過濾後為空）")
    return out


def write_rows(path: Path, rows) -> None:
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---- 主程式 ----

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--lpips-ref", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "color_lpips_ref.csv")
    ap.add_argument("--lpips-tolerance", type=float, default=0.0025)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--arm", default=ARM, help="產出檔名與 CSV 的臂名")
    ap.add_argument("--objective", default="comm", choices=("comm", "xattn"),
                    help="comm：等變殘差；xattn：降低攻擊端對類別詞（man／woman）的整圖交叉注意力")
    ap.add_argument("--lr-final-ratio", type=float, default=0.2)
    ap.add_argument("--lam-every", type=int, default=5)
    ap.add_argument("--caps", default="full", choices=("full", "simple"),
                    help="full：現行 15 道上限；simple：五個方向的色偏上限改為錨點方框（構造保證）、"
                         "亮度斜率範圍 1.35（|ΔL| ≤ 15），只保留膚色同色 ΔE00 ≤ 16 與暖色（a*＋、b*＋）逐像素最大值 3 道")
    ap.add_argument("--carrier", default="ab", choices=("ab",),
                    help="ab：(a,b) RBF 位移 ＋ 亮度曲線")
    ap.add_argument("--box", default="uniform", choices=("uniform", "skin"),
                    help="simple 的方框：uniform 各錨點相同；skin 冷色方向依離膚色中心距離放大到 2 倍")
    args = ap.parse_args()

    caps_cfg = {"frame": 32.0, "face": 16.0, "skin_radius": 12.0, "chroma_gain": 2.0,
                "shift": {"a_pos": 4, "a_neg": 15, "b_pos": 4, "b_neg": 25, "l_abs": 15}}
    lpips_ref = {r["image"]: float(r["lpips_ref"]) for r in csv.DictReader(args.lpips_ref.open(encoding="utf-8"))}

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ip2p = IP2PWrapper(dtype=torch.float32)
    import piq
    lp = piq.LPIPS().to(device).eval()
    lp.requires_grad_(False)

    rows = []
    for item in load_images(args.data, set(args.images) if args.images else None):
        started = time.time()
        name = item["name"]
        if name not in lpips_ref:
            raise SystemExit(f"{args.lpips_ref} 沒有 {name} 的 LPIPS 參考值")
        x = load_image_tensor(item["path"], device, size=RESOLUTION)
        boxes = face_boxes(x, device)
        if not boxes:
            raise SystemExit(f"{name} 偵測不到臉；膚色群心定不出來")
        box = expanded_box(x, max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])))
        frame, face = torch.ones_like(x[:, :1]), box_support(x, box)
        skin = skin_colour_support(x, face, caps_cfg["skin_radius"])
        c95 = float(chroma_p95(x))

        simple = args.caps == "simple"
        # simple：斜率範圍 1.35 使 |F(L) − L| ≤ 0.15（任意 16 段單調曲線的最壞情形）
        carrier = ColorMap(l_radius=0.35 if simple else 0.6)
        carrier.reset(x, args.noise_seed)
        g = torch.Generator(device="cpu").manual_seed(args.noise_seed)
        with torch.no_grad():  # 非恆等起點：恆等映射下 comm 與其梯度皆為 0
            carrier.w_raw.copy_((0.1 * torch.randn(carrier.w_raw.shape, generator=g)).to(device))
            carrier.th_raw.copy_(((torch.rand(carrier.th_raw.shape, generator=g) * 2 - 1) * 0.1).to(device))
        sh = caps_cfg["shift"]
        if simple:
            if args.box == "skin":
                carrier.set_box(*skin_scaled_box(carrier.anchors, skin_centre(x, face), sh))
            else:
                carrier.set_box(-sh["a_neg"], sh["a_pos"], -sh["b_neg"], sh["b_pos"])
        carrier.project()

        caps = [
            Cap("frame", lambda y: delta_e00_torch(x, y, frame), lambda y: delta_e00(x, y, frame), caps_cfg["frame"]),
            Cap("face_box", lambda y: delta_e00_torch(x, y, face), lambda y: delta_e00(x, y, face), caps_cfg["face"]),
            Cap("skin_colour", lambda y: delta_e00_torch(x, y, skin), lambda y: delta_e00(x, y, skin), caps_cfg["face"]),
            *[Cap(f"shift_{n}", lambda y, c=c, s=s: channel_shift_p95(x, y, c, s),
                  lambda y, c=c, s=s: float(channel_shift_p95(x, y, c, s)), sh[n]) for n, c, s in CHANNEL_CAPS],
            *[Cap(f"shift_{n}_max", lambda y, c=c, s=s: channel_shift_max(x, y, c, s),
                  lambda y, c=c, s=s: float(channel_shift_max(x, y, c, s)), sh[n]) for n, c, s in CHANNEL_CAPS],
            Cap("chroma_p95", lambda y: chroma_p95(y), lambda y: float(chroma_p95(y)), c95 * caps_cfg["chroma_gain"]),
            Cap("input_lpips", lambda y: lp(y, x).mean(), lambda y: float(lp(y, x).mean()),
                lpips_ref[name] + args.lpips_tolerance),
        ]
        if simple:
            # 方框只管映射本身；轉回 RGB 的色域裁切仍可能產生暖色位移，暖色兩個方向保留逐像素最大值
            caps = [c for c in caps if c.name in ("skin_colour", "shift_a_pos_max", "shift_b_pos_max")]
            if args.objective == "xattn":
                # 注意力目標的權重把解推到暖色上限邊界，量化後超出 0.4–0.5；訓練端留 0.5 餘量，可行性仍以 4 檢查
                caps = [c._replace(soft=lambda y, f=c.soft: f(y) + 0.5) if c.name.startswith("shift_") else c
                        for c in caps]

        free = FreeObjective(ip2p, x, box=box, k=4, steps=50, seed=args.noise_seed,
                             weights={"id": 1.0, "enc": 0.5, "cond": 1.0},
                             chain_steps=6, grad_steps=1, s_i=1.5, resample=True)
        if args.objective == "comm":
            objective = CommObjective(free, carrier, x, lp)
            extra = {k: round(v, 6) for k, v in align_weight(objective, free, carrier, x).items()}
        else:
            objective = XAttnScore(ip2p, x, item["class"], args.noise_seed)
            g_obj = _grad_norm(carrier, lambda: objective.eval_score(carrier.render(x)))
            g_free = _grad_norm(carrier, lambda: free.eval_score(carrier.render(x)))
            objective.w = g_free / g_obj  # 起點梯度範數對齊 FreeObjective
            extra = {"xattn_weight": round(objective.w, 6), "class_word": item["class"]}
        term = args.objective
        extra.update(objective=args.objective, caps_mode=args.caps, box=args.box if simple else "", carrier=carrier.name)
        with torch.no_grad():
            extra[f"{term}_val_start"] = round(float(objective.eval_terms(carrier.render(x))[term]), 6)

        stats = optimise_carrier(carrier, x, objective, steps=args.steps, lr=args.lr, caps=caps,
                                 rho=10.0, lam_every=args.lam_every, check_every=10, log_every=100,
                                 lr_final_ratio=args.lr_final_ratio, probe_every=50)
        # optimise_carrier 的回傳鍵一律以 free_ 開頭
        if stats.get("free_feasible_step", -1) == -1:
            raise SystemExit(f"{name}：{args.steps} 步內找不到可行 checkpoint"
                             f"（違反：{stats.get('free_caps_unprojected')}），拒絕存檔")

        with torch.no_grad():
            y = quantise(carrier.render(x))
            off = lab_offset(x, y)
            row = {
                "image": name, "class": item["class"], "arm": args.arm, "solver_steps": args.steps, "lr": args.lr,
                "lpips_cap": round(lpips_ref[name] + args.lpips_tolerance, 6), "lpips_out": round(float(lp(y, x).mean()), 6),
                f"{term}_val_end": round(float(objective.eval_terms(y)[term]), 6),
                "chroma_p95_orig": round(c95, 4), "chroma_p95_out": round(float(chroma_p95(y)), 4),
                **{f"shift_{n}_p95": round(float(channel_shift_p95(x, y, c, s)), 4) for n, c, s in CHANNEL_CAPS},
                **{f"shift_{n}_max_out": round(float(channel_shift_max(x, y, c, s)), 4) for n, c, s in CHANNEL_CAPS},
                "deltaE00_frame": round(float(delta_e00(x, y, frame)), 4),
                "deltaE00_face_box": round(float(delta_e00(x, y, face)), 4),
                "deltaE00_skin_colour": round(float(delta_e00(x, y, skin)), 4),
                "tv_frame": round(float(tv_offset(off, frame)), 5),
                "warp_max": round(float(carrier.w.norm(dim=-1).max()), 3),
                "psnr": round(float(10 * torch.log10(1.0 / (y - x).pow(2).mean())), 4),
                "seconds": round(time.time() - started, 1), **extra, **stats,
            }
        save_image(x, args.out / f"{name}__orig.png")
        save_image(y, args.out / f"{name}__{args.arm}__def.png")
        torch.save(carrier.state_dict(), args.out / f"{name}__carrier.pt")
        save_image(warp_picture(carrier, device), args.out / f"{name}__warp.png")
        rows.append(row)
        write_rows(args.out / "results.csv", rows)
        print(f"[{args.arm}] {name}  ΔE frame {row['deltaE00_frame']} face {row['deltaE00_face_box']} "
              f"LPIPS {row['lpips_out']}/{row['lpips_cap']}  {row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 張 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
