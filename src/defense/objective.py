"""損失函數 — spec §5。

    L(φ) = λ_def · L_def + λ_fid · L_fid

**無低秩懲罰項。** 秩是外積參數化帶來的架構硬約束（spec §4.1），不以懲罰
項近似，故此處看不到任何與秩有關的項，這是設計而非遺漏。

防禦項（spec §5.1）：

    L_def = E_{P∼𝒫, ε∼N(0,I)} [ max(0, m − d( E(P(x_def), ε), E(x, ε) )) ]

hinge 形式是必要設計，不是調參選擇：無界的最大化會發散。偏移超過 margin
`m` 之後不再施力，優化轉而改善保真項。

保真項（spec §5.2）：

    L_fid = LPIPS(x_def, x)
          + α · (1 − SSIM(x_def, x))
          + β · max(0, ‖Δ‖_∞ − τ)
          + γ · max(0, PSNR_floor − PSNR(x_def, x))

後兩項是硬地板，其存在理由是 v2 的實測結果：apa 的 LPIPS 與 pg_enc 幾乎
相同，PSNR 卻差 12.7 dB、L∞ 差 28 倍。保真項若只用 LPIPS，優化會利用
LPIPS 的量測盲區，產生 LPIPS 數值良好但人眼可見的失真。兩道地板封閉此路徑。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class LossConfig:
    """全部係數集中於此，避免散落在呼叫端。

    數值為初始設定，須由 E2 的前緣掃描檢驗，不是已驗證的最佳值。
    """

    lam_def: float = 1.0
    lam_fid: float = 1.0

    # 防禦項：margin 以 d 的尺度為準，d 預設為 LPIPS
    margin: float = 0.5

    # 保真項
    alpha_ssim: float = 1.0
    beta_linf: float = 100.0
    gamma_psnr: float = 1.0
    tau_linf: float = 0.06     # ≈ 15/255，對抗擾動文獻的常見上限量級

    # psnr_floor 由 E0c 實測決定，不是憑經驗填的。site L 在 φ=0 時的
    # x_def 已是 inversion + VAE 來回的重建，其 PSNR 有一個 φ 無法消除的
    # 地板：t_max=500、k_inv=20 下實測平均 26.56 dB（VAE 單獨來回的
    # 不可約地板為 27.51 dB，n=6）。若把 psnr_floor 設在地板之上（原值
    # 30 dB），PSNR hinge 對 site L 將永遠處於啟動狀態，保真項變成一個
    # 恆定且無法改善的懲罰並壓過防禦項。故取 26.0，略低於實測地板。
    psnr_floor: float = 26.0   # dB，見 runs/e0c_tmax/recon_floor.csv


class DefenseObjective:
    """L(φ) 的計算。持有 LPIPS 與 SSIM 的可微實作。

    `y_orig` 對 φ 為常數（spec §5.1），由呼叫端算好並傳入；本類別不負責
    快取，以免把「哪些量對 φ 為常數」這個關鍵前提藏在實作細節裡。
    """

    def __init__(self, cfg: LossConfig, device: torch.device):
        import piq

        self.cfg = cfg
        self.device = device
        # piq.LPIPS 可微，訓練與評測共用同一實作，避免兩者定義不一致
        self._lpips = piq.LPIPS().to(device)

    # ---- 距離 d(·,·)：兩張編輯結果的差異 ----

    def distance(self, y_a: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        """spec §8.1 的 d。取 LPIPS，因其為感知距離的標準指標且可微。

        評測階段仍會報全部八項指標；此處單取一項是因為訓練目標必須是純量
        且可微，與評測的「不得只報單一指標」是兩件事。
        """
        return self._lpips(y_a.clamp(0, 1), y_b.clamp(0, 1))

    # ---- 防禦項 ----

    def defense_term(
        self, y_def_list: List[torch.Tensor], y_orig_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """對 (淨化算子, 噪聲) 的取樣求平均，即 spec §5.1 的期望值估計。

        兩條分支的噪聲必須逐元素相同，否則量到的偏移主要來自噪聲差異。
        此不變量由呼叫端以共用 `noise` 保證（見 SDWrapper.sdedit 的介面）。
        """
        if len(y_def_list) != len(y_orig_list):
            raise ValueError(
                f"兩側取樣數不符：{len(y_def_list)} vs {len(y_orig_list)}，"
                "每個防禦分支必須對上使用相同噪聲的原圖分支"
            )
        terms = [
            torch.clamp(self.cfg.margin - self.distance(yd, yo), min=0.0)
            for yd, yo in zip(y_def_list, y_orig_list)
        ]
        return torch.stack(terms).mean()

    # ---- 保真項 ----

    def fidelity_term(
        self, x_def: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """回傳 (L_fid, 各分項的純量值)。分項一併回傳供逐步診斷。"""
        import piq

        c = self.cfg
        xd = x_def.clamp(0, 1)
        xr = x.clamp(0, 1)

        lpips = self._lpips(xd, xr)
        ssim = piq.ssim(xd, xr, data_range=1.0)
        linf = (xd - xr).abs().max()
        # 手寫 PSNR 而非用 piq.psnr：需要對 x_def 可微，且要能處理 mse=0
        mse = torch.nn.functional.mse_loss(xd, xr)
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))

        pen_linf = torch.clamp(linf - c.tau_linf, min=0.0)
        pen_psnr = torch.clamp(c.psnr_floor - psnr, min=0.0)

        total = (
            lpips
            + c.alpha_ssim * (1.0 - ssim)
            + c.beta_linf * pen_linf
            + c.gamma_psnr * pen_psnr
        )
        parts = {
            "fid_lpips": float(lpips),
            "fid_ssim": float(ssim),
            "fid_linf": float(linf),
            "fid_psnr": float(psnr),
            "fid_pen_linf": float(pen_linf),
            "fid_pen_psnr": float(pen_psnr),
        }
        return total, parts

    # ---- 總損失 ----

    def __call__(
        self,
        x_def: torch.Tensor,
        x: torch.Tensor,
        y_def_list: List[torch.Tensor],
        y_orig_list: List[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        c = self.cfg
        l_def = self.defense_term(y_def_list, y_orig_list)
        l_fid, parts = self.fidelity_term(x_def, x)
        total = c.lam_def * l_def + c.lam_fid * l_fid

        # 平均編輯偏移是最直接的進展指標，與 hinge 後的 L_def 分開記錄：
        # hinge 飽和後 L_def 恆為 0，看不出偏移還在不在增加
        with torch.no_grad():
            shift = torch.stack(
                [self.distance(yd, yo) for yd, yo in zip(y_def_list, y_orig_list)]
            ).mean()

        log = {
            "loss": float(total),
            "L_def": float(l_def),
            "L_fid": float(l_fid),
            "edit_shift": float(shift),
            **parts,
        }
        return total, log
