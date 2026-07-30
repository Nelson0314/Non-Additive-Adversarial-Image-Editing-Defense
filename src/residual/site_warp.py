"""site S — 空間變形 —— 非加性、且不經生成路徑。

    x_def = Warp(x; f_φ) = grid_sample(x, identity_grid + f_φ)

φ 是一個位移場（flow field），每個像素被移動一小段距離；輸出是**重新取樣**
的結果，不是 `x + Δ`。構造出自 stAdv（spatially transformed adversarial
examples, Xiao et al., ICLR 2018）與「Generalizing Universal Adversarial
Attacks Beyond Additive Perturbations」的非加性分支。

**為什麼需要這個位置**

先前實作的三個非加性位置（latent 注入、文字嵌入、權重 LoRA）**全部經過
VAE 解碼**，因此全部繼承同一個重建誤差地板：φ=0 時 `G(x;0)` 與原圖已相差
LPIPS 0.194 / PSNR 26.56 dB，保真度預算在 φ 起作用之前就用光了。E9 與 E12
花了 5 GPU 小時嘗試用更大的容量吸收該誤差，最好也只到 LPIPS 0.095，
仍遠不及加性位置實際運作的 0.063。

**非加性不需要生成。** 空間變形是非加性的，卻停在像素空間：

| 性質 | latent / 嵌入 / 權重 | **空間變形** |
|---|---|---|
| 非加性 | 是 | **是** |
| 經過 VAE 解碼 | 是 | **否** |
| φ=0 時等於原圖 | 否（LPIPS 0.194） | **是（構造保證）** |
| 每步成本（k_inv=20） | 約 11.8 秒 | **約 4.1 秒** |
| 需要階段一保真對齊 | 是 | **不需要** |

**保真度的自然預算是位移量，不是像素差值。**

這一點必須在報告中講清楚：把一條邊緣移動一個像素，其 L∞ 可以接近 1.0，
但人眼幾乎看不出來。加性擾動用 L∞ 當上界是合理的（像素值改了多少就是
看到了多少），空間變形不是——stAdv 界定的是 flow 的大小而非 L∞。故本模塊
以 `max_disp`（單位：像素）為硬上界，並回報位移的平均與最大值供報告使用。
既有的 L∞ hinge 對本位置可能過嚴，該係數須另行檢視。

**平滑度**：不平滑的位移場會產生撕裂狀的可見瑕疵。此處以**粗網格 + 雙線性
上採樣**取得平滑性（`grid_size < size` 時），而非事後加懲罰項——由構造保證
比由懲罰項近似更可靠，且 `grid_size` 直接就是一個乾淨的容量旋鈕。
`grid_size=None` 表示逐像素自由位移（等同 stAdv 的原始設定），此時平滑度
只能靠 `tv()` 診斷觀察。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.residual.base import ResidualModule


def identity_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    """(1, H, W, 2) 的恆等取樣網格，座標順序為 (x, y)，與 grid_sample 一致。

    以 `align_corners=True` 的約定建構：座標 −1 與 +1 分別對應第一個與最後
    一個像素的中心。此約定下恆等網格落在像素中心上，`grid_sample` 的雙線性
    插值權重為 (1, 0)，理論上應回傳原值。**該性質由測試實測，不假設。**
    """
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)


class WarpResidual(ResidualModule):
    site = "S"

    def __init__(
        self,
        size: int = 512,
        grid_size: Optional[int] = 32,
        max_disp: float = 1.5,
        init_std: float = 0.0,
        seed: int = None,
        force_resample: bool = False,
    ):
        """`max_disp` 的單位是**像素**；`grid_size=None` 表示逐像素自由位移。

        `init_std=0` 使初始位移場恆為零，即 x_def = x。位移場是自由參數，
        在零點的梯度不為零（∂x_def/∂f 由影像梯度決定，一般非零），故不需要
        像外積參數化那樣的「一半高斯、一半零」安排。

        `force_resample=True` 取消零位移短路，即使 φ=0 也實際跑一次
        `grid_sample`。這只是為了做對照：量測「純數值重取樣誤差」本身
        造成多少編輯偏移，不是正常訓練該用的設定。
        """
        super().__init__()
        g = size if grid_size is None else grid_size
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        if init_std > 0:
            f0 = torch.randn(1, 2, g, g, generator=gen) * init_std
        else:
            f0 = torch.zeros(1, 2, g, g)
        self.flow = nn.Parameter(f0)
        self.size = size
        self.grid_size = g
        self.max_disp = float(max_disp)
        self.force_resample = bool(force_resample)

    # ---- 位移場 ----

    def displacement(self, h: int, w: int) -> torch.Tensor:
        """回傳 (1, 2, H, W) 的位移場，單位為**像素**，已套用 max_disp 上界。

        粗網格以雙線性上採樣到全解析度，藉此取得平滑性；`align_corners=True`
        與 `identity_grid` 的約定一致，否則兩者的座標系統會錯開半個像素。
        """
        f = self.flow
        if f.shape[-2] != h or f.shape[-1] != w:
            f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=True)
        if self.max_disp > 0:
            # 硬上界而非懲罰項：位移量是本位置的保真度預算，須確實不被超過
            f = f.clamp(-self.max_disp, self.max_disp)
        return f

    def pixel_residual(self, x01: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return x01
        b, c, h, w = x01.shape
        disp = self.displacement(h, w).to(x01.dtype)

        # 位移恆為零時直接回傳原圖。這不是為了規避問題，而是因為**恆等變形
        # 的正確結果就是原圖**，而經過 grid_sample 反而會引入純數值誤差。
        #
        # 實測（512²、float32）：恆等網格經 grid_sample 後與原圖最大差
        # 5.78e-05。根因是 align_corners=True 下的座標 −1 + 2i/(N−1) 在
        # float32 無法精確表示，實測偏離整數像素位置最多 3.05e-05，雙線性
        # 插值再依比例混入鄰居像素。全程改用 float64 可降到 1.10e-13，但
        # 把 float64 網格降回 float32 無效（5.78e-05 不變）——誤差來自表示
        # 極限而非計算順序。整條管線是 fp32，為此改用 float64 不值得。
        #
        # 短路使「模塊停用或 φ=0 時不改變任何計算結果」與其他位置一樣是
        # 逐位元成立的。φ 非零時仍有該數值底線，其量級為 5.78e-05，
        # 相對 site P 實測的 L∞ 0.07 小 1200 倍；但**它對編輯偏移的貢獻
        # 必須實測一次**（見 force_resample）——site L 的教訓正是一個
        # 「看起來很小」的 φ 無關效應最後主導了全部數字。
        #
        # **只在不建計算圖時短路。** 直接回傳 x01 會切斷 φ 與輸出的連結，
        # 訓練第一步（此時位移恰為零）會以
        # `element 0 of tensors does not require grad` 失敗——實測如此。
        # 短路要保證的是「回報與評測的數值在 φ=0 時精確」，而評測、
        # x_base0 = G(x;0) 與所有留存影像都在 torch.no_grad() 下計算，
        # 故以 grad 模式為判準即涵蓋全部需要精確的場合；訓練期第一步多
        # 承受 5.78e-05 的偏差（PSNR 約 85 dB），且優化器一動位移即非零，
        # 之後兩條路徑本來就相同。
        needs_graph = torch.is_grad_enabled() and self.flow.requires_grad
        if not self.force_resample and not needs_graph and not disp.any():
            return x01

        # 像素單位換算為 grid_sample 的正規化單位。align_corners=True 下
        # 座標跨距為 2，涵蓋 (w−1) 個像素間隔，故一個像素等於 2/(w−1)。
        sx = 2.0 / max(w - 1, 1)
        sy = 2.0 / max(h - 1, 1)
        off = torch.stack((disp[:, 0] * sx, disp[:, 1] * sy), dim=-1)

        grid = identity_grid(h, w, x01.device, x01.dtype) + off
        return F.grid_sample(
            x01, grid.expand(b, -1, -1, -1), mode="bilinear",
            padding_mode="border", align_corners=True,
        )

    # ---- 診斷 ----

    def raw_residual(self) -> None:
        """空間變形沒有「注入前的像素殘差」這個量：φ 是位移而非像素值。

        回傳 None 而非 `x_def − x`：後者是變形造成的**結果**，把它當成
        殘差會讓報告誤以為這是一個加性方法。位移場本身以 disp_stats() 回報。
        """
        return None

    @torch.no_grad()
    def disp_stats(self) -> dict:
        """位移場的統計。這才是本位置的保真度預算，不是 L∞。"""
        d = self.displacement(self.size, self.size)
        mag = d.pow(2).sum(dim=1).sqrt()          # 逐像素位移長度（像素）
        return {
            "disp_mean_px": float(mag.mean()),
            "disp_max_px": float(mag.max()),
            "disp_p99_px": float(mag.flatten().quantile(0.99)),
            "flow_tv": float(self.tv()),
            "grid_size": self.grid_size,
        }

    def tv(self) -> torch.Tensor:
        """位移場的 total variation，即不平滑程度。

        本模塊以粗網格取得平滑性，故此值僅作診斷，不進損失。若改用
        `grid_size=None` 的逐像素自由位移，則需要把它加進損失，否則會出現
        撕裂狀的可見瑕疵。
        """
        f = self.flow
        dh = (f[..., 1:, :] - f[..., :-1, :]).abs().mean()
        dw = (f[..., :, 1:] - f[..., :, :-1]).abs().mean()
        return dh + dw

    def rank_trace(self, ts=None, steps=None) -> list:
        """回報位移場的控制點數，作為容量的度量。

        刻意不叫「秩」：本位置沒有低秩結構，硬套會誤導。
        """
        return [self.grid_size]
