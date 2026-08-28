"""色散變形：把「位移」與「相位」放在同一條軸上。

要回答什麼
────────────────────────────────────────────────────────────────────
位移場（FND-004）在本專案上死了兩次：`ip2p_warp/DIAGNOSIS.md` 證明預算從頭
到尾就夠、是最佳化沒有走；`ip2p_warp_hard` 補上兩個端點——grid 16 到不了失真
帶，grid 64 到得了但付 1.92 倍失真只換到 65% 的位移。

本模組補的是**歸因**：單值位移場是微分同胚，它把一張自然影像映成另一張自然
影像，編碼器與 UNet 面對的仍是一張合理的照片。真正動得到編碼器的只有場的
**粗糙度**（重取樣走樣），而那與加性高頻雜訊同族。這不是推論——
`runs/ip2p_warp/matched_geometry.csv` 已直接量到 `warp_roundtrip`（先 f 再 −f，
幾何抵銷、只留內插 artifact）與 `warp_rand` 在等失真上只差 **8.6%**。

若歸因成立，病灶是「**單值**」而不是「位移」：單值場強迫所有空間頻率一起搬，
這正是它留在自然影像流形上的原因。

構造
────────────────────────────────────────────────────────────────────
讓每個頻帶各自有自己的位移。加窗區塊的複數頻譜上，「把第 k 帶平移 u_k 像素」
就是「該帶的相位加 −2π<f, u_k>」——這是相位式運動處理（Wadhwa et al.,
Phase-Based Video Motion Processing, SIGGRAPH 2013）所用的同一條恆等式，
在那裡它被用來放大影片裡看不見的運動。

    theta_b[omega] = -2*pi * ( fy_omega * uy_k(b) + fx_omega * ux_k(b) )

於是 K（獨立位移的頻帶數）成為一個「色散度」旋鈕：

    K = 1              古典位移場（逐視窗一個位移），**已知失敗，是內建對照**
    K = 2..8           色散變形，未測
    K = 每個頻格獨立    現行的紋理重相位

參數量由 59 萬降到 K × 視窗數 × 2。

三個實作上的選擇
────────────────────────────────────────────────────────────────────
1. **不走 `PhaseResidual` 的夾取路徑。** `_rephase` 會把 theta 夾在 ±theta_max
   （π）之內，而位移的相位斜坡本來就會超過 π——相位是週期量，超過 π 應該
   **繞回去**而不是被夾平。夾平會把斜坡削成別的東西，且不會有症狀。本模組
   因此直接用 `analyze`／`synthesize` 這一對（Griffin & Lim 的最小平方重建，
   `theta = 0` 時逐位元恆等由它保證），自己乘 `exp(i·theta)`。
2. **閘只取二值帶通**（`radial_gate`）。知覺定價 `q(omega)^gamma` 會逐格改變
   斜坡的斜率，那樣做出來的東西就不再是「把某一帶平移 u 像素」。要比的是
   色散度，不是定價。
3. **頻帶以 log2 半徑等分**（八度帶）。自然影像的功率譜按 1/f² 掉，等寬的
   線性帶會讓最外圈那一帶佔掉絕大多數的格子而幾乎沒有能量。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from src.residual.perceptual_weight import freq_weight
from src.residual.texture_rephase import (
    PhaseResidual, radial_gate, texture_gate,
)

# 角落頻格的歸一化半徑是 sqrt(2)（fy 與 fx 各到 1），與
# `scripts/encoder_frequency_response.py` 同一套座標。
R_CORNER = math.sqrt(2.0)


def freq_axes(block: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """rfft2 格點的**每取樣週期數**頻率（cycles/pixel），(block,1) 與 (1,nb)。

    `radial_gate` 用的是乘過 2 的歸一化半徑（Nyquist = 1）；位移的相位斜坡
    要的是物理頻率（Nyquist = 0.5），兩者差一個因子 2。**混用會讓位移的
    單位差一倍且不會有症狀**，故兩個座標各自取一次，不互相換算。
    """
    fy = torch.fft.fftfreq(block, device=device, dtype=dtype)[:, None]
    fx = torch.fft.rfftfreq(block, device=device, dtype=dtype)[None, :]
    return fy, fx


def band_index(block: int, n_bands: int, r_min: float, device,
               r_max: float = R_CORNER,
               dtype=torch.float32) -> torch.Tensor:
    """(block, block//2+1) 的頻帶編號。通帶外一律 −1。

    `n_bands = 1` 時整個通帶是一帶，也就是古典位移場那一格。
    帶界以 log2 半徑等分；`r_min` 之下、`r_max` 之上、以及 `fx = 0` 與
    `fx = block//2` 兩行（rfft2 的共軛對稱依賴它們）都給 −1。
    """
    if n_bands < 1:
        raise ValueError(f"n_bands 必須為正整數，收到 {n_bands}")
    if r_max <= r_min:
        raise ValueError(f"r_max={r_max} 不大於 r_min={r_min}，通帶是空的")
    mask = radial_gate(block, r_min, device, dtype, r_max) > 0
    fy = torch.fft.fftfreq(block, device=device, dtype=dtype)[:, None] * 2.0
    fx = torch.fft.rfftfreq(block, device=device, dtype=dtype)[None, :] * 2.0
    r = torch.sqrt(fy ** 2 + fx ** 2).clamp_min(1e-12)
    lo, hi = math.log2(r_min), math.log2(r_max)
    pos = (torch.log2(r) - lo) / (hi - lo)
    idx = torch.clamp((pos * n_bands).floor().long(), 0, n_bands - 1)
    return torch.where(mask, idx, torch.full_like(idx, -1))


def displacement_theta(block: int, bands: torch.Tensor,
                       u: torch.Tensor) -> torch.Tensor:
    """把逐（視窗, 頻帶）的位移展開成逐（視窗, 頻格）的相位。

    `u` 形狀 (L, K, 2)，最後一維是 (ux, uy)，單位是**像素**。
    回傳 (1, L, block, block//2+1)，通帶外為 0。

    `theta = -2*pi*(fx*ux + fy*uy)`：這就是平移定理，**沒有近似**。
    """
    if u.dim() != 3 or u.shape[-1] != 2:
        raise ValueError(f"u 必須是 (L, K, 2)，收到 {tuple(u.shape)}")
    device, dtype = u.device, u.dtype
    fy, fx = freq_axes(block, device, dtype)
    n_bands = int(u.shape[1])
    safe = bands.clamp_min(0)                                   # (n, nb)
    ux = u[:, :, 0]                                             # (L, K)
    uy = u[:, :, 1]
    # 逐頻格取它所屬頻帶的位移：(L, n, nb)
    gx = ux[:, safe.reshape(-1)].reshape(-1, *bands.shape)
    gy = uy[:, safe.reshape(-1)].reshape(-1, *bands.shape)
    theta = -2.0 * math.pi * (fx[None] * gx + fy[None] * gy)
    theta = torch.where(bands[None] >= 0, theta, torch.zeros_like(theta))
    return theta.unsqueeze(0)


def random_field(side: int, tail: Tuple[int, ...], amp: float, seed: int,
                 grid: int, device, dtype=torch.float32) -> torch.Tensor:
    """視窗格點上的隨機場，形狀 `(side*side, *tail)`，值域 [−amp, amp]。

    `grid > 0` 時先在 `grid × grid` 的粗網格上抽，再雙三次上採樣到視窗格點，
    **空間上因此是平滑的**；`grid = 0` 則逐視窗獨立抽。

    為什麼要有這個旋鈕
    ────────────────────────────────────────────────────────────
    色散度那條軸要問的是「同一個位置上，不同頻率搬得一不一致」。若逐視窗
    獨立抽，K=1 的場在**空間上**也是最粗的，於是它與 `WarpParam` 的
    16×16 粗網格＋雙三次（刻意平滑）之間差了兩件事，K 的效果就與空間粗糙度
    混在一起。`grid` 把空間粗糙度固定住，讓 K 成為唯一的變因。

    預設對齊 `WarpParam` 的 16——本專案量過的位移—失真對照表就是在那個構造
    上做的。上採樣之後夾回 ±amp：雙三次會過衝，不夾的話「強度」沒有定義
    （與 `WarpParam.project` 夾粗網格係數同一條理由的另一半）。

    取均勻而非高斯：sign PGD 會把 L∞ 球用滿，同族的隨機對照一律取用滿的
    分布（與 `WarpRandomParam`、`RandomPhaseParam` 同一條理由）。
    """
    if grid < 0:
        raise ValueError(f"grid 不可為負，收到 {grid}")
    n_tail = 1
    for t in tail:
        n_tail *= int(t)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    if grid == 0:
        r = torch.rand((side * side, n_tail), generator=g) * 2.0 - 1.0
    else:
        coarse = torch.rand((1, n_tail, grid, grid), generator=g) * 2.0 - 1.0
        up = torch.nn.functional.interpolate(
            coarse, size=(side, side), mode="bicubic", align_corners=False)
        r = up.clamp(-1.0, 1.0).reshape(n_tail, side * side).transpose(0, 1)
    return (r * amp).reshape(side * side, *tail).to(device=device, dtype=dtype)


def random_displacements(side: int, n_bands: int, amp: float, seed: int,
                         device, grid: int = 16,
                         dtype=torch.float32) -> torch.Tensor:
    """(L, K, 2) 的隨機逐頻帶位移，單位是像素。空間平滑度由 `grid` 控制。"""
    return random_field(side, (n_bands, 2), amp, seed, grid, device, dtype)


def random_phase_theta(side: int, block: int, amp: float, seed: int,
                       bands: torch.Tensor, device, grid: int = 16,
                       dtype=torch.float32) -> torch.Tensor:
    """逐頻格獨立的隨機相位，(1, L, block, block//2+1)。色散度的另一個端點。

    空間上與 `random_displacements` 走同一個 `grid`，**頻率上則完全獨立**
    ——這正是「色散度最大」的意思。強度單位是弧度。
    """
    nb = block // 2 + 1
    theta = random_field(side, (block, nb), amp, seed, grid, device,
                         dtype).unsqueeze(0)
    return torch.where(bands[None, None] >= 0, theta, torch.zeros_like(theta))


def make_operator(x01: torch.Tensor, block: int, hop: int, r_min: float,
                  r_max: float = float("inf")) -> PhaseResidual:
    """只借 `analyze`／`synthesize` 那一對的 STFT 算子，已備好窗。

    閘全開（`energy_quantile=0`、`gate_edge_power=0`、`freq_weight="binary"`）
    ——本模組自己乘相位、不走 `_rephase`，故兩個閘不參與前向。

    **但 `prepare_gates` 仍然必須呼叫**：`window` 在 `__init__` 只註冊成一個
    零緩衝，真正的 Hann 窗是在 `prepare_gates` 裡填的（`texture_rephase.py`
    第 587 與 613 行）。少了這一步 `analyze` 會回傳整片零，而**輸出不會拋錯、
    只會變成全黑**——`tests/test_dispersion.py` 的逐位元恆等那一條就是為了
    擋這個。尺寸、裝置與 dtype 一律由 `x01` 決定，不另外收參數，免得三者與
    影像不一致。
    """
    size = int(x01.shape[-1])
    if x01.shape[-2] != size:
        raise ValueError(f"只支援正方形影像，收到 {tuple(x01.shape[-2:])}")
    op = PhaseResidual(
        size=size, block=block, hop=hop, r_min=r_min, r_max=r_max,
        theta_max=math.pi, energy_quantile=0.0, gate_edge_power=0.0,
        freq_weight="binary", freq_weight_power=1.0,
    ).to(device=x01.device, dtype=x01.dtype)
    op.prepare_gates(x01)
    return op


def apply_theta(op: PhaseResidual, x01: torch.Tensor,
                theta: torch.Tensor) -> torch.Tensor:
    """`x' = clamp(synthesize(analyze(x) * exp(i*theta)), 0, 1)`。

    **不夾 theta**：相位是週期量，超過 π 應該繞回去。`exp(i*theta)` 自然
    如此，而 `PhaseResidual._rephase` 的 `clamp` 不是——這正是本模組不走
    那條路徑的理由。
    """
    spec = op.analyze(x01)
    rot = torch.polar(torch.ones_like(theta), theta).unsqueeze(1)
    return op.synthesize(spec * rot).clamp(0.0, 1.0)


def bandpass(op: PhaseResidual, x01: torch.Tensor,
             bands: torch.Tensor) -> torch.Tensor:
    """只留通帶內容的那一份，供診斷與檢定用。

    **這不是前向路徑。** `apply_theta` 只旋轉通帶內的相位，通帶外的係數
    原樣通過，故 `theta = 0` 時它是逐位元恆等而不是一個帶通濾波器。要檢定
    「K=1 的相位斜坡真的是一個平移」就必須先把通帶那一份抽出來比。
    """
    mask = (bands >= 0).to(x01.dtype)
    return op.synthesize(op.analyze(x01) * mask)


def fold_fraction(disp: torch.Tensor) -> float:
    """位移場 `disp`（1,2,H,W，單位像素）造成的**折疊比例**。

    映射是 `T(p) = p + d(p)`，其 Jacobian 行列式

        det = (1 + d_x,x)(1 + d_y,y) - d_x,y * d_y,x

    小於等於 0 的位置代表局部翻面／壓成零測度，也就是那裡的映射**不是**
    微分同胚。這一欄存在的理由是：「平滑就是微分同胚就是留在流形上」這句
    歸因必須有一個量，不能用嘴講。差分取前向差，邊界少一行一列。
    """
    if disp.dim() != 4 or disp.shape[1] != 2:
        raise ValueError(f"disp 必須是 (1,2,H,W)，收到 {tuple(disp.shape)}")
    dx, dy = disp[:, 0], disp[:, 1]
    dxx = dx[:, :-1, 1:] - dx[:, :-1, :-1]
    dxy = dx[:, 1:, :-1] - dx[:, :-1, :-1]
    dyx = dy[:, :-1, 1:] - dy[:, :-1, :-1]
    dyy = dy[:, 1:, :-1] - dy[:, :-1, :-1]
    det = (1.0 + dxx) * (1.0 + dyy) - dxy * dyx
    return float((det <= 0).to(torch.float32).mean())


def band_price(block: int, bands: torch.Tensor, n_bands: int, device,
               dtype=torch.float32, power: float = 0.25,
               name: str = "jpeg_luma") -> torch.Tensor:
    """每個頻帶一個知覺價錢：`q(ω)^power` 在該帶上的平均，形狀 `(K,)`。

    **取帶內平均而不是逐格定價**，這是整個構造的關鍵取捨。逐格的
    `q(ω)^0.25` 會逐格改變相位斜坡的斜率，做出來的東西就不再是「把這一帶
    平移 u 像素」——那是本模組唯一的賣點。取帶平均之後，第 k 帶仍然是一個
    純平移，只是幅度被它自己的價錢縮放過，於是**知覺定價與平移語意兩者都
    保得住**。代價是定價的解析度只到帶，不到格。
    """
    q = freq_weight(name, block, device, dtype, power)
    out = torch.zeros(n_bands, device=device, dtype=dtype)
    for k in range(n_bands):
        m = bands == k
        out[k] = q[m].mean() if bool(m.any()) else 0.0
    return out


class DispersionParam:
    """逐頻帶位移。`learnable=True` 時 `u` 是 PGD 參數，否則抽一次就凍結。

    存在的理由是把色散度那條軸接進主線管線：`scripts/dispersion_probe.py`
    只寫 CSV、不存圖，而報告頁要的是防禦圖、淨化圖與編輯圖。介面與
    `WarpRandomParam`／`RandomPhaseParam` 相同（`params()` 回傳空清單，
    `run_param_pgd` 因此不會更新任何東西）。

    `n_bands = None` 表示逐頻格獨立的隨機相位，也就是色散度的另一個端點
    （即現行家族的隨機對照）。**兩者的半徑單位不同**——逐頻帶位移是像素，
    逐頻格相位是弧度，故不可跨族比同一個 radius，一律在等失真上比。
    """

    name = "disp"

    def __init__(self, radius: float, n_bands: Optional[int] = None,
                 block: int = 32, hop: int = 8, r_min: float = 0.12,
                 field_grid: int = 16, learnable: bool = False,
                 gate: bool = False, energy_quantile: float = 0.0,
                 gate_edge_power: float = 1.0,
                 freq_weight_power: float = 0.25):
        if n_bands is not None and n_bands < 1:
            raise ValueError(f"n_bands 必須為正整數或 None，收到 {n_bands}")
        if learnable and n_bands is None:
            raise ValueError(
                "learnable=True 需要有限的 n_bands：逐頻格獨立相位沒有『位移』"
                "這個物件，那一格是相位族不是位移族")
        self.radius = float(radius)
        self.n_bands = n_bands
        self.block, self.hop, self.r_min = block, hop, r_min
        self.field_grid = field_grid
        # 可學：`u` 由**零**起步（零位移即帶通恆等），走 sign PGD、夾在 ±radius。
        self.learnable = bool(learnable)
        # 兩個閘。`gate=False` 時逐位元等於加這組參數之前。
        self.gate = bool(gate)
        self.energy_quantile = energy_quantile
        self.gate_edge_power = gate_edge_power
        self.freq_weight_power = freq_weight_power
        self._u = None
        self._tex = None
        self._price = None
        self._bands = None
        self._op = None
        self._theta = None
        self._seed = 0
        self._device = None
        self._dtype = None

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self._seed = int(seed)
        self._device, self._dtype = x01.device, x01.dtype
        self._op = make_operator(x01, self.block, self.hop, self.r_min)
        k = 1 if self.n_bands is None else self.n_bands
        self._bands = band_index(self.block, k, self.r_min, x01.device,
                                 dtype=x01.dtype)
        if self.gate:
            # **兩個閘都由原圖算一次就凍結**，與現行主線同一條規則：閘決定
            # 擾動被允許出現在哪裡，不參與最佳化。
            self._tex = texture_gate(
                x01, self.block, self.hop,
                energy_quantile=self.energy_quantile,
                edge_power=self.gate_edge_power).reshape(-1).to(x01.dtype)
            self._price = band_price(self.block, self._bands, k, x01.device,
                                     x01.dtype, self.freq_weight_power)
        if self.learnable:
            self._u = torch.zeros(self._op.side ** 2, self.n_bands, 2,
                                  device=x01.device, dtype=x01.dtype,
                                  requires_grad=True)
        else:
            self._build()

    def _build(self) -> None:
        k = 1 if self.n_bands is None else self.n_bands
        bands = band_index(self.block, k, self.r_min, self._device,
                           dtype=self._dtype)
        if self.n_bands is None:
            self._theta = random_phase_theta(
                self._op.side, self.block, self.radius, self._seed, bands,
                self._device, grid=self.field_grid, dtype=self._dtype)
        else:
            u = random_displacements(
                self._op.side, self.n_bands, self.radius, self._seed,
                self._device, grid=self.field_grid, dtype=self._dtype)
            self._theta = displacement_theta(self.block, bands, u)

    def _scaled(self, u: torch.Tensor) -> torch.Tensor:
        """套上兩個閘。紋理閘是逐視窗的純量、帶價是逐帶的純量，**兩者都不
        隨頻格變**，所以第 k 帶在第 b 個視窗上仍然是一個純平移。"""
        if not self.gate:
            return u
        return u * self._tex[:, None, None] * self._price[None, :, None]

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        if self._op is None:
            raise RuntimeError("reset() 未呼叫")
        if self.learnable:
            theta = displacement_theta(self.block, self._bands,
                                       self._scaled(self._u))
        else:
            theta = self._theta
        return apply_theta(self._op, x01, theta)

    def params(self) -> list:
        return [self._u] if self.learnable else []

    @torch.no_grad()
    def project(self) -> None:
        if self.learnable:
            self._u.clamp_(-self.radius, self.radius)

    def set_radius(self, r: float) -> None:
        """半徑改變要**重抽**：場的振幅是抽樣的尺度，不是事後夾的界。

        與 `WarpRandomParam` 同一條理由——那邊也是在 `reset` 裡按半徑抽。
        二分搜（`fit_to_budget`）會反覆呼叫這一支，故重建必須便宜，而這裡
        只是重抽一個粗網格的隨機場再上採樣。
        """
        self.radius = float(r)
        if self._op is not None and not self.learnable:
            self._build()
