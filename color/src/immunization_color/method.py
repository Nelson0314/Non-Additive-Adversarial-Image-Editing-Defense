"""color 方法：CIELAB 色度平面的全域映射載體、等變殘差與交叉注意力目標、臉框與膚色支撐。

命令列入口為 `immunization_color.cli.generate_color_defenses`，方法說明見其模組 docstring。
"""

from __future__ import annotations

import argparse

import torch

from immunization_core.color.space import lab_to_rgb, rgb_to_lab
from immunization_core.metrics.identity import DETECTOR_IMAGE_SIZE, DETECTOR_MARGIN
from immunization_core.optimization.attention import CrossAttnObjective

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

    # optimize_carrier 的介面
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
    驗證用固定 4 組。量測為 `immunization_core.optimization.attention.CrossAttnObjective`。
    """

    def __init__(self, ip2p, x01, class_word, seed):
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
