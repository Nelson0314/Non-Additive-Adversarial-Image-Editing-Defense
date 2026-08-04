"""site C — 色度矩陣場 —— 非加性、不經生成路徑、且結構上不鈍化。

    x_def = YUV⁻¹( Y(x), M_φ · [U(x), V(x)]ᵀ )

φ 是一個定義在粗網格上的 2×2 矩陣場，逐像素作用在 Rec.601 色度平面
(U, V) 上；亮度 Y 原封不動地通過。輸出是原影像色度被重新混合的結果，
不是 `x + Δ`。

為什麼需要這個位置：site S 的死因在這裡結構上不存在。

E20/E21 量出 site S（空間變形）的鈍化幾乎全部來自重取樣本身——雙線性
grid_sample 的銳利度保留率只有 85.0%，而真實 site S 是 85.2%。改用雙三次
可回到 99.9%，但 E23 把步數提到 100 之後 site S 對加性基準反轉為 0.85×。

鈍化約束 `local_acutance_dev`（`src/metrics/local_acutance.py`）量的是
Rec.601 亮度的逐區塊梯度能量比（見該檔 `_grad_sq` → `_luma`）。本位置
不動 Y，故亮度梯度逐像素不變，`local_acutance_dev` 在數學上為 0。

| 性質 | site P（加性） | site S（變形） | site C（色度） |
|---|---|---|---|
| 非加性 | 否 | 是 | 是 |
| 經過 VAE 解碼 | 否 | 否 | 否 |
| φ=0 時等於原圖 | 是 | 是（構造保證） | 是（構造保證） |
| 鈍化約束是否啟動 | 幾乎不（0.009） | 是（0.15） | 結構上為 0 |

「約束不對它收費」是雙面的，不可當成優點宣稱。E20 的教訓正是「指標
實際在收什麼費」必須逐個位置量過：現行約束集（LPIPS ∩ 鈍化）對本位置的
特徵失真——色偏、色度串音、飽和區的假色——是否收費尚未量測。加入本位置
之後必須為它跑一次四條件等 LPIPS 探針，否則就是重蹈 site S 用 LPIPS 換到的是模糊
的覆轍，只是換成用由色度換來。

保真度的自然預算是矩陣偏離單位陣的量，不是 L∞。與 site S 以位移量
（像素）為預算同理：把一片天空的色調整體轉幾度，其 L∞ 可以不小，人眼卻
只覺得白平衡不同。故本模塊以 `max_dev`（矩陣元素偏離單位陣的上界）為硬
上界，並回報實際的色度位移量供報告使用。

已知限制：近乎無彩的區域沒有容量。M 作用在 (U, V) 上是乘性的，
故 U = V = 0 的像素無論 M 為何都不動。灰階或高度去飽和的影像在本位置的
可用容量趨近於零。這是參數化的性質不是缺陷，但必須在報告中載明，且是
「本位置對哪些影像有效」這個問題的直接答案。加偏移項可以解掉它，但偏移項
是加性的（色度上的常數位移），會破壞本位置「非加性」的定位，故不加。

平滑度由粗網格 + 雙線性上採樣的構造保證，與 site S 同一手法：由構造
保證比由懲罰項近似可靠，且 `grid_size` 就是一個乾淨的容量參數。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.residual.base import ResidualModule

# Rec.601。與 `src/metrics/acutance.py` 的 `_luma` 同一組亮度係數——兩者
# 必須一致，否則「本位置不改變亮度」這個設計性質對該指標並不成立。
_KR, _KG, _KB = 0.299, 0.587, 0.114        # Rec.601 亮度係數
_UMAX, _VMAX = 0.436, 0.615                # Rec.601 的色度歸一化上限


def _rgb2yuv_matrix() -> torch.Tensor:
    """由亮度係數推導 RGB→YUV，不用文獻上那組四捨五入的常數。

    色度的定義是 U ∝ (B − Y)、V ∝ (R − Y)，故 U 列與 V 列的元素和必須
    精確為零——那正是「灰階像素的色度為零」這件事。教科書列的
    [-0.14713, -0.28886, 0.436] 是各自捨入後的結果，其和為 1e-05 而非 0，
    於是 R=G=B 的像素會帶著 1e-05·R 的假色度。後果不是數值上的小瑕疵而是
    設計性質失效：本模塊「無彩區域沒有容量」這條已知限制會變成「幾乎沒有
    容量」，梯度實測為 3.3e-04 而非零（未修正前 `test_無彩影像上色度變換
    沒有容量` 即以此失敗）。由 Y 推導後該和精確為 0，殘餘只剩浮點精度。
    """
    y = torch.tensor([_KR, _KG, _KB], dtype=torch.float64)
    u = (torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) - y) * (_UMAX / (1 - _KB))
    v = (torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64) - y) * (_VMAX / (1 - _KR))
    return torch.stack((y, u, v))


_RGB2YUV = _rgb2yuv_matrix().float()
# 精確求逆，不用文獻上那組四捨五入的常數。教科書列的
# [[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]] 是各自捨入的近似值，
# 與正向矩陣互為反矩陣只到約 1.8e-05——實測 RGB→YUV→RGB 來回誤差即為該量級，
# 而本位置的短路要保證的是「φ=0 時逐位元等於原圖」。一個 1.8e-05 的系統性
# 偏差會讓該性質只是「近似成立」。在 float64 求逆後轉回 float32，來回誤差
# 降到 3e-08。
_YUV2RGB = torch.linalg.inv(_rgb2yuv_matrix()).float()


def rgb_to_yuv(x: torch.Tensor) -> torch.Tensor:
    m = _RGB2YUV.to(x.device, x.dtype)
    return torch.einsum("ij,njhw->nihw", m, x)


def yuv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    m = _YUV2RGB.to(x.device, x.dtype)
    return torch.einsum("ij,njhw->nihw", m, x)


class ColorResidual(ResidualModule):
    site = "C"

    def __init__(
        self,
        size: int = 512,
        grid_size: Optional[int] = 32,
        max_dev: float = 0.15,
        init_std: float = 0.0,
        seed: int = None,
        force_transform: bool = False,
    ):
        """`max_dev` 是矩陣元素偏離單位陣的上界；`grid_size=None` 表示逐像素。

        `init_std=0` 使初始 ΔM 恆為零，即 M = I、x_def = x。ΔM 是自由參數，
        在零點的梯度不為零（∂x_def/∂ΔM 由該像素的色度決定，只要 U、V 不同時
        為零就非零），故不需要像外積參數化那樣的對稱破壞安排。但無彩像素
        的梯度確實為零，見模組 docstring 的已知限制。

        `force_transform=True` 取消 φ=0 的短路，用來單獨量測「純 RGB↔YUV
        來回的數值誤差」本身造成多少編輯偏移。這是對照用，不是訓練設定。
        """
        super().__init__()
        g = size if grid_size is None else grid_size
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        if init_std > 0:
            d0 = torch.randn(1, 4, g, g, generator=gen) * init_std
        else:
            d0 = torch.zeros(1, 4, g, g)
        # 參數是 ΔM（偏離單位陣的量）而非 M：φ=0 必須精確對應恆等變換，
        # 若直接參數化 M 則需要把初值設成 I，而「零初值等於恆等」這個性質
        # 對所有位置一致比較不容易出錯。
        self.delta = nn.Parameter(d0)
        self.size = size
        self.grid_size = g
        self.max_dev = float(max_dev)
        self.force_transform = bool(force_transform)

    # ---- 矩陣場 ----

    def matrix_field(self, h: int, w: int) -> torch.Tensor:
        """回傳 (1, 4, H, W) 的 ΔM，已套用 max_dev 上界。

        粗網格以雙線性上採樣到全解析度取得平滑性；`align_corners=True` 與
        site S 的約定一致，兩個位置的容量參數才可直接對照。
        """
        d = self.delta
        if d.shape[-2] != h or d.shape[-1] != w:
            d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=True)
        if self.max_dev > 0:
            # 硬上界而非懲罰項：偏離量是本位置的保真度預算，須確實不被超過
            d = d.clamp(-self.max_dev, self.max_dev)
        return d

    def pixel_residual(self, x01: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return x01
        b, c, h, w = x01.shape
        d = self.matrix_field(h, w).to(x01.dtype)

        # ΔM 恆為零時直接回傳原圖。與 site S 的短路同一理由：恆等變換的正確
        # 結果就是原圖，而經過 RGB→YUV→RGB 反而引入純數值誤差（兩個矩陣互為
        # 反矩陣只到浮點精度）。只在不建計算圖時短路——直接回傳 x01 會切斷
        # φ 與輸出的連結，訓練第一步（此時 ΔM 恰為零）會以
        # `element 0 of tensors does not require grad` 失敗，site S 已實測過。
        needs_graph = torch.is_grad_enabled() and self.delta.requires_grad
        if not self.force_transform and not needs_graph and not d.any():
            return x01

        yuv = rgb_to_yuv(x01)
        y, u, v = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
        # M = I + ΔM，逐像素作用在 (U, V) 上
        u2 = (1.0 + d[:, 0:1]) * u + d[:, 1:2] * v
        v2 = d[:, 2:3] * u + (1.0 + d[:, 3:4]) * v
        out = yuv_to_rgb(torch.cat((y, u2, v2), dim=1))

        # 色度被重新混合後可能落到 RGB 立方體之外。夾回是「可顯示影像」的
        # 定義，不是為了讓數字好看：不夾的話後續的 LPIPS 與銳利度都在量一張
        # 不存在的影像。夾回會微幅改變亮度，故「本位置不改變 Y」這個性質
        # 在飽和處並非精確成立；其量級由 tests/test_site_color.py 實測釘住。
        return out.clamp(0, 1)

    # ---- 診斷 ----

    def raw_residual(self) -> None:
        """色度變換沒有「注入前的像素殘差」這個量：φ 是矩陣而非像素值。

        回傳 None 而非 `x_def − x`：後者是變換造成的結果，把它當成殘差
        會讓報告誤以為這是一個加性方法。矩陣場本身以 `color_stats()` 回報。
        """
        return None

    @torch.no_grad()
    def color_stats(self) -> dict:
        """矩陣場的統計。這才是本位置的保真度預算，不是 L∞。"""
        d = self.matrix_field(self.size, self.size)
        # 逐像素取 ΔM 的 Frobenius 範數：它是「色度被混得多凶」的直接度量
        fro = d.pow(2).sum(dim=1).sqrt()
        return {
            "cdev_mean": float(fro.mean()),
            "cdev_max": float(fro.max()),
            "cdev_p99": float(fro.flatten().quantile(0.99)),
            "matrix_tv": float(self.tv()),
            "grid_size": self.grid_size,
        }

    def tv(self) -> torch.Tensor:
        """矩陣場的 total variation，即不平滑程度。僅作診斷，不進損失。

        與 site S 同：平滑性由粗網格構造保證。改用 `grid_size=None` 的逐像素
        自由矩陣時才需要把它加進損失，否則會出現色塊狀的可見瑕疵。
        """
        d = self.delta
        dh = (d[..., 1:, :] - d[..., :-1, :]).abs().mean()
        dw = (d[..., :, 1:] - d[..., :, :-1]).abs().mean()
        return dh + dw

    def rank_trace(self, ts=None, steps=None) -> list:
        """回報矩陣場的控制點數，作為容量的度量。

        刻意不叫「秩」：本位置沒有低秩結構，硬套會誤導（同 site S）。
        """
        return [self.grid_size]
