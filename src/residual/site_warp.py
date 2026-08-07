"""site S — 空間變形 —— 非加性、且不經生成路徑。

    x_def = Warp(x; f_φ) = grid_sample(x, identity_grid + f_φ)

φ 是一個位移場（flow field），每個像素被移動一小段距離；輸出是重新取樣
的結果，不是 `x + Δ`。構造出自 stAdv（spatially transformed adversarial
examples, Xiao et al., ICLR 2018）與「Generalizing Universal Adversarial
Attacks Beyond Additive Perturbations」的非加性分支。

為什麼需要這個位置

先前實作的三個非加性位置（latent 注入、文字嵌入、權重 LoRA）全部經過
VAE 解碼，因此全部繼承同一個重建誤差下限：φ=0 時 `G(x;0)` 與原圖已相差
LPIPS 0.194 / PSNR 26.56 dB，保真度預算在 φ 起作用之前就用光了。E9 與 E12
花了 5 GPU 小時嘗試用更大的容量吸收該誤差，最好也只到 LPIPS 0.095，
仍遠不及加性位置實際運作的 0.063。

非加性不需要生成。空間變形是非加性的，卻停在像素空間：

| 性質 | latent / 嵌入 / 權重 | 空間變形 |
|---|---|---|
| 非加性 | 是 | 是 |
| 經過 VAE 解碼 | 是 | 否 |
| φ=0 時等於原圖 | 否（LPIPS 0.194） | 是（構造保證） |
| 每步成本（k_inv=20） | 約 11.8 秒 | 約 4.1 秒 |
| 需要階段一保真對齊 | 是 | 不需要 |

保真度的自然預算是位移量，不是像素差值。

這一點必須在報告中講清楚：把一條邊緣移動一個像素，其 L∞ 可以接近 1.0，
但人眼幾乎看不出來。加性擾動用 L∞ 當上界是合理的（像素值改了多少就是
看到了多少），空間變形不是——stAdv 界定的是 flow 的大小而非 L∞。故本模塊
以 `max_disp`（單位：像素）為硬上界，並回報位移的平均與最大值供報告使用。
既有的 L∞ hinge 對本位置可能過嚴，該係數須另行檢視。

平滑度：不平滑的位移場會產生撕裂狀的可見瑕疵。此處以粗網格 + 雙線性
上採樣取得平滑性（`grid_size < size` 時），而非事後加懲罰項——由構造保證
比由懲罰項近似更可靠，且 `grid_size` 直接就是一個乾淨的容量參數。
`grid_size=None` 表示逐像素自由位移（等同 stAdv 的原始設定），此時平滑度
只能靠 `tv()` 診斷觀察。

遮罩閘（2026-08-08 新增，使用者裁決的處置 B）

inpainting 威脅模型下攻擊方會把遮罩內整片重畫，防禦方在該區域的擾動
**沒有任何防禦價值，卻照樣被 `L_fid` 全額收費**。PhotoGuard-c 與 PromptFlare
的原始碼本來就把梯度乘 `(1 − mask)`（本專案以 `grad_outside_mask` 忠實
實作），不加閘等於我方先丟掉「遮罩涵蓋率」比例的預算才開始比較。

閘作用在**粗網格上的 `flow` 參數**而非上採樣後的位移場：後者會讓位移量在
遮罩邊界一個像素內由 `max_disp` 跳到 0，而 `attention_box` 產生的是硬邊
矩形，那道跳變正是本模塊靠粗網格避掉的撕裂瑕疵。乘在粗網格上再上採樣，
過渡帶自然展開為 `size / grid_size` 個像素（512²、`grid_size=32` 下為 16）。
閘值取該格**未被遮罩覆蓋的面積比例**（`mode="area"` 降採樣），故完全落在
遮罩內的格恰為 0、完全在外的恰為 1。
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
    插值權重為 (1, 0)，理論上應回傳原值。該性質由測試實測，不假設。
    """
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)


class WarpResidual(ResidualModule):
    site = "S"

    # 重取樣模式。E20 實測：在同一個 LPIPS 上，雙線性重取樣的銳利度保留率
    # 只有 85.0%，雙三次為 99.9%（見 docs/RESULTS_E13-E23.md §5.2）。
    # E19 量到的真實 site S 為 85.2%，與雙線性條件幾乎相同——**site S 的鈍化
    # 幾乎全部來自重取樣本身，不是最佳化學到的低通**。
    RESAMPLE_MODES = ("bilinear", "bicubic")

    def __init__(
        self,
        size: int = 512,
        grid_size: Optional[int] = 32,
        max_disp: float = 1.5,
        init_std: float = 0.0,
        seed: int = None,
        force_resample: bool = False,
        resample: str = "bilinear",
        gate: Optional[torch.Tensor] = None,
    ):
        """`max_disp` 的單位是像素；`grid_size=None` 表示逐像素自由位移。

        `gate` 是 `coarse_gate()` 產生的 (1,1,g,g) 遮罩閘，`None` 表示不加閘
        （img2img 威脅模型）。它不是可訓練參數也不進 `state_dict`
        （`persistent=False`）——它由遮罩決定，重建時與 `grid_size` 一樣
        由 `phi.pt` 的 `build` 欄還原。

        `resample` 選 `grid_sample` 的插值模式。預設維持 `"bilinear"`，使
        E13–E19 的既有結果可重現；`"bicubic"` 是 E20 之後的建議值，理由見
        `RESAMPLE_MODES` 的說明。改這個值會改變所有既有數字，故不改預設。

        `init_std=0` 使初始位移場恆為零，即 x_def = x。位移場是自由參數，
        在零點的梯度不為零（∂x_def/∂f 由影像梯度決定，一般非零），故不需要
        像外積參數化那樣的「一半高斯、一半零」安排。

        `force_resample=True` 取消零位移短路，即使 φ=0 也實際跑一次
        `grid_sample`。這只是為了做對照：量測「純數值重取樣誤差」本身
        造成多少編輯偏移，不是正常訓練該用的設定。
        """
        super().__init__()
        if resample not in self.RESAMPLE_MODES:
            raise ValueError(
                f"未知的 resample: {resample!r}，只接受 {self.RESAMPLE_MODES}。"
                "靜默退回預設會讓實驗跑完才發現設定沒生效"
            )
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
        self.resample = resample
        if gate is not None:
            if tuple(gate.shape) != (1, 1, g, g):
                raise ValueError(
                    f"gate 的形狀為 {tuple(gate.shape)}，應為 (1, 1, {g}, {g})"
                    "，即與 flow 同一個粗網格。形狀不符時廣播會靜默給出一個"
                    "看似合理但閘錯位置的位移場"
                )
            gate = gate.detach().clone().float()
        self.register_buffer("gate", gate, persistent=False)

    # ---- 遮罩閘 ----

    @staticmethod
    def coarse_gate(mask: torch.Tensor, grid_size: int) -> torch.Tensor:
        """由 (1,1,H,W) 的遮罩產生 (1,1,g,g) 的閘，值域 [0,1]。

        `mask` 的約定與 `src/data/masks.py` 一致：**1 表示攻擊方要重畫的
        區域**。閘為 `1 − 該格被遮罩覆蓋的面積比例`，故遮罩內為 0。

        降採樣取 `mode="area"`（面積平均）而非 `nearest`：後者只看格中心，
        一個比格小的遮罩會整個消失，而症狀是「閘看起來全開、結果照樣被
        重畫的區域吃掉預算」。
        """
        if mask.dim() != 4 or mask.shape[:2] != (1, 1):
            raise ValueError(
                f"遮罩形狀為 {tuple(mask.shape)}，應為 (1, 1, H, W)"
            )
        m = F.interpolate(mask.float(), size=(grid_size, grid_size),
                          mode="area")
        return (1.0 - m).clamp(0.0, 1.0)

    # ---- 位移場 ----

    def displacement(self, h: int, w: int) -> torch.Tensor:
        """回傳 (1, 2, H, W) 的位移場，單位為像素，已套用 max_disp 上界。

        粗網格以雙線性上採樣到全解析度，藉此取得平滑性；`align_corners=True`
        與 `identity_grid` 的約定一致，否則兩者的座標系統會錯開半個像素。
        遮罩閘（若有）在上採樣**之前**乘上去，理由見模組 docstring。
        """
        f = self.flow
        if self.gate is not None:
            # 乘在粗網格上，過渡帶由後續的雙線性上採樣展開（見模組 docstring）
            f = f * self.gate.to(dtype=f.dtype, device=f.device)
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
        # 只在不建計算圖時短路。直接回傳 x01 會切斷 φ 與輸出的連結，
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
        out = F.grid_sample(
            x01, grid.expand(b, -1, -1, -1), mode=self.resample,
            padding_mode="border", align_corners=True,
        )
        # 雙三次會過衝到 [0,1] 之外（雙線性是凸組合故不會）。夾回是「可顯示
        # 影像」的定義，不是為了讓數字好看：不夾的話後續的 LPIPS 與銳利度
        # 都在量一張不存在的影像。夾回可微，梯度在飽和處為零。
        return out.clamp(0, 1) if self.resample == "bicubic" else out

    # ---- 診斷 ----

    def raw_residual(self) -> None:
        """空間變形沒有「注入前的像素殘差」這個量：φ 是位移而非像素值。

        回傳 None 而非 `x_def − x`：後者是變形造成的結果，把它當成
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
            # 有沒有加閘必須逐格留在證據裡：兩種形態的位移場統計看起來一樣，
            # 事後只憑 disp_max 分不出這一格是不是只動了脈絡。
            "mask_gated": self.gate is not None,
            "gate_open_frac": (1.0 if self.gate is None
                               else float(self.gate.mean())),
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
