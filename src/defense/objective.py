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
          + β · max(0, ‖x_def − x_base‖_∞ − τ)   ← 見下方修訂
          + γ · max(0, PSNR_floor − PSNR(x_def, x_base))   ← 見下方修訂

後兩項是硬地板，其存在理由是 v2 的實測結果：apa 的 LPIPS 與 pg_enc 幾乎
相同，PSNR 卻差 12.7 dB、L∞ 差 28 倍。保真項若只用 LPIPS，優化會利用
LPIPS 的量測盲區，產生 LPIPS 數值良好但人眼可見的失真。兩道地板封閉此路徑。

**相對 spec §5.2 的修訂（2026-07-29）**：**兩道 hinge 的對象**由對原圖
`x` 改為對 `x_base = G(x; φ=0)`，即該 site 在未施加防禦時就已產生的圖。原式對 site L 不可用：
E0c 實測 VAE 單獨來回的 L∞ 平均為 0.707，φ=0 時實測 1.0000，遠高於
τ = 0.06，hinge 恆為啟動且 φ 無法改善，L_fid 被一個約 94 的常數主導，
而防禦項的上限僅 margin = 0.5，防禦完全無法發展。改為相對 `x_base` 後，
兩個 site 量的都是「防禦本身加了多少」，彼此可比；`x_def` 與原圖的總體
保真仍由 LPIPS、SSIM、PSNR 三項把關，且 `fid_linf_total` 仍逐步記錄，
報告中兩者都會列出。

PSNR hinge 同樣改為相對，理由更強：E0c 實測六張影像的重建地板由
19.61 dB（car_00）到 31.01 dB（person_00），相差 11.4 dB。任何**全域
固定**的絕對門檻都不可能同時適用 —— 原值 30 dB 對六張全部不可達，
改成 26 dB 後對 car_00（19.61）與 car_01（22.28）仍不可達，對
person_00（31.01）又完全不施力。改以 `PSNR(x_def, x_base)` 為對象後，
門檻自動隨每張影像與每個 site 的地板調整，兩個 site 量到的都是「防禦
讓 PSNR 掉了多少」。對原圖的絕對值以 `fid_psnr_total` 逐步記錄。

值得一併記下的觀察：同一組量測中 LPIPS 的地板僅由 0.157 到 0.210，
遠比 PSNR 穩定。這與 v2 的發現方向相反但不矛盾——v2 說的是 LPIPS 會
**低估**非加性失真，此處說的是 LPIPS 對重建誤差的**跨影像變異**較小。
兩者都支持「不得只報單一指標」。

**相對 spec §5.2 的修訂之二（2026-07-29，改用感知指標作為綁定約束）**：

    修訂前（本檔 line 173-180）
        pen_psnr = max(0, psnr_floor − PSNR(x_def, x_base))    psnr_floor = 34 dB
        total    = lpips + α(1−ssim) + β·pen_linf + γ·pen_psnr  γ = 1.0

    修訂後
        pen_lpips = max(0, LPIPS(x_def, x_base) − τ_lpips)      τ_lpips = 0.05
        total     = lpips + α(1−ssim) + β·pen_linf + γ_l·pen_lpips + γ·pen_psnr
                                                                γ_l = 100.0, γ = 0.0

理由：PSNR 是逐像素平方誤差，與人眼可辨性的關聯薄弱，作為「防禦不得
被看出來」的綁定約束並不適當。上一段記錄的量測正是直接證據——同一組
影像的 PSNR 重建地板跨影像相差 11.4 dB，LPIPS 地板卻只由 0.157 到
0.210。改以 LPIPS(x_def, x_base) 為綁定 hinge，量的是「防禦本身造成
多少感知可辨的改變」。

**L∞ hinge 不移除**：v2 的實測（本檔 line 22-24）顯示 LPIPS 會低估
非加性失真，單以 LPIPS 為約束會留下與當初導入硬地板時相同的漏洞。
L∞ 抓得到 LPIPS 平均掉的局部爆點，兩者並用。

PSNR hinge 改為 γ = 0.0，即**保留計算與記錄但不參與梯度**。刪掉會失去
與既有 36 格結果的對照基準；留著係數則可一行復原。

γ_l = 100.0 與 β_linf 同量級：兩者的對象都在 [0,1] 尺度，而防禦項的
上限僅 margin = 0.5，故超出 τ 後的懲罰必須足以壓過防禦項才構成地板。
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
    tau_linf: float = 0.06     # ≈ 15/255，對抗擾動文獻的常見上限量級

    # 綁定的保真度約束：LPIPS(x_def, x_base)，即「防禦本身造成多少感知
    # 可辨的改變」。取代原本的 PSNR 地板，理由見模組 docstring 的修訂之二。
    # τ 預設 0.05：主網格的像素注入 r=16 實測落在 LPIPS 0.0626，故此值
    # 比現況略緊；正式實驗以 {0.02, 0.05, 0.10} 掃描，得到曲線而非單點。
    gamma_lpips: float = 100.0
    tau_lpips: float = 0.05

    # PSNR 地板保留計算與記錄，但預設不參與梯度（gamma_psnr = 0）。
    # 保留係數是為了能一行復原，並維持與既有 36 格結果的對照基準。
    gamma_psnr: float = 0.0
    psnr_floor: float = 34.0   # dB，相對 x_base；見 runs/e0c_tmax/


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
        self,
        x_def: torch.Tensor,
        x: torch.Tensor,
        x_base: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """回傳 (L_fid, 各分項的純量值)。分項一併回傳供逐步診斷。

        `x_base` 是 φ=0 時的輸出，即該 site 在**未施加任何防禦**時就已經
        產生的圖。site P 為 `x` 本身；site L 為 inversion + VAE 來回的重建。
        L∞ hinge 以 `x_def − x_base` 為對象，其餘三項仍以 `x_def − x` 為
        對象。理由見下方 spec 修訂說明。
        """
        import piq

        c = self.cfg
        xd = x_def.clamp(0, 1)
        xr = x.clamp(0, 1)
        xb = xr if x_base is None else x_base.clamp(0, 1)

        lpips = self._lpips(xd, xr)
        ssim = piq.ssim(xd, xr, data_range=1.0)
        # L∞ 取「φ 造成的改變」而非「與原圖的總差距」。E0c 實測 VAE 單獨
        # 來回的 L∞ 平均已達 0.707：單一個飽和像素就能讓 L∞ 飽和，該值
        # 幾乎完全由重建誤差決定，與防禦強度無關。若以總差距為對象，
        # τ=0.06 對 site L 不可達，hinge 恆為啟動，L_fid 被一個 φ 無法
        # 改善的常數（實測約 94）主導，防禦項（上限 margin=0.5）完全失效。
        # 改以 x_def − x_base 為對象後，兩個 site 量的都是「防禦加了多少」，
        # 彼此可比；總保真度仍由 LPIPS/SSIM/PSNR 三項對 x 把關。
        linf = (xd - xb).abs().max()

        # PSNR hinge 同樣以 x_base 為對象，理由與 L∞ 相同且更強：E0c 實測
        # 各影像的重建地板由 19.61 dB（car_00）到 31.01 dB（person_00），
        # 相差 11.4 dB。任何**全域固定**的 psnr_floor 都不可能同時適用：
        # 取 26 時對 car_00 與 car_01 不可達（hinge 恆啟動），對 person_00
        # 又完全不施力。改以 PSNR(x_def, x_base) 為對象後，門檻自動隨每張
        # 影像與每個 site 的地板調整，量的一律是「防禦讓 PSNR 掉了多少」。
        # 對原圖的絕對 PSNR 仍以 fid_psnr_total 記錄並在報告中列出。
        mse = torch.nn.functional.mse_loss(xd, xb)
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        mse_total = torch.nn.functional.mse_loss(xd, xr)
        psnr_total = 10.0 * torch.log10(1.0 / mse_total.clamp_min(1e-12))

        # 綁定的感知約束，對象與 L∞/PSNR 一致為 x_base：量的是「防禦加了
        # 多少」而非「與原圖的總差距」。site P 的 x_base 即 x，此時本項與
        # 上方的 lpips 相同，多算一次是為了讓兩個 site 的 hinge 定義一致，
        # 不因 site 而分支。
        lpips_rel = lpips if x_base is None else self._lpips(xd, xb)

        pen_linf = torch.clamp(linf - c.tau_linf, min=0.0)
        pen_lpips = torch.clamp(lpips_rel - c.tau_lpips, min=0.0)
        pen_psnr = torch.clamp(c.psnr_floor - psnr, min=0.0)

        total = (
            lpips
            + c.alpha_ssim * (1.0 - ssim)
            + c.beta_linf * pen_linf
            + c.gamma_lpips * pen_lpips
            + c.gamma_psnr * pen_psnr
        )
        parts = {
            "fid_linf_total": float((xd - xr).abs().max()),
            "fid_psnr_total": float(psnr_total),
            "fid_lpips": float(lpips),
            "fid_lpips_rel": float(lpips_rel),
            "fid_ssim": float(ssim),
            "fid_linf": float(linf),
            "fid_psnr": float(psnr),
            "fid_pen_linf": float(pen_linf),
            "fid_pen_lpips": float(pen_lpips),
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
        x_base: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        c = self.cfg
        l_def = self.defense_term(y_def_list, y_orig_list)
        l_fid, parts = self.fidelity_term(x_def, x, x_base=x_base)
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
