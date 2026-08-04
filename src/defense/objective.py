"""損失函數 — spec §5。

    L(φ) = λ_def · L_def + λ_fid · L_fid

## 防禦項 L_def

由 `LossConfig.defense_mode` 決定：

    untargeted   max(0, m − d(y_def, y_orig))     把編輯結果推離原編輯
    targeted     d(y_def, y_target)               把編輯結果推向固定目標

`d` 取 LPIPS。hinge 是必要設計不是調參：無界的最大化會發散，偏移超過
margin `m` 之後不再施力，優化轉而改善保真項。

**`untargeted` 在 y_def 與 y_orig 逐元素相同時梯度精確為零**，因為 LPIPS
在該點取到最小值。φ = 0 且淨化算子為 identity 時正是這個情形，故該模式
實際上是靠淨化 EOT 輪替到非 identity 算子才脫離起點。見 LEDGER 5.10、
5.11 與 `tests/test_zero_gradient.py`。

## 保真項 L_fid

四道 hinge 取**交集**，加上兩個可關閉的加權項：

    L_fid = α_l·LPIPS(x_def, x)          α_l = alpha_lpips，預設 1.0
          + α·(1 − SSIM(x_def, x))       α  = alpha_ssim，預設 0.0
          + β  ·max(0, L∞      − τ_linf)   β   = beta_linf
          + γ_l·max(0, LPIPS   − τ_lpips)  γ_l = gamma_lpips
          + γ_a·max(0, acut    − τ_acut)   γ_a = gamma_acut
          + γ_c·max(0, chroma  − τ_chroma) γ_c = gamma_chroma
          + γ  ·max(0, τ_psnr  − PSNR)     γ   = gamma_psnr，預設 0.0

取交集而非加權和的理由：加權和永遠可以用便宜的軸換貴的軸，交集式的
可行域不允許這種交換。

四道 hinge 的對象都是 `x_base = G(x; φ=0)`（該位置未施加防禦時就已產生的
圖），不是原圖 `x`。量的因此是「防禦本身加了多少」，兩個 site 彼此可比。
對原圖的絕對值仍以 `fid_lpips`、`fid_psnr_total`、`fid_linf_total` 記錄。

`acut` 是 `local_acutance_dev`（鈍化），`chroma` 是 `local_chroma_bias`
（空間連貫的色偏）。兩者都是由等 LPIPS 多條件探針判別出來的——ΔE 類與
SSIM／NLPD／VIF／GMSD／HaarPSI 都沒有通過。

## 預設值的注意事項

`alpha_lpips = 1.0` 與 `beta_linf = 100.0` **不是本專案要的值**，只為了讓
E2–E27 的既有數字可重現。正式實驗一律覆寫為 0，理由見各欄位的註解。
`tau_acut` 與 `tau_chroma` 是在 τ_lpips = 0.05 的量級上定出來的絕對值，
換預算必須改（LEDGER 2.10、6.17）。

## 修訂史

五次修訂的 before/after 與各自的實測依據在
`docs/OBJECTIVE_HISTORY.md`（2026-08-04 由本 docstring 原封搬出）。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from src.metrics.chroma import local_chroma_bias
from src.metrics.local_acutance import local_acutance_dev


@dataclass
class LossConfig:
    """全部係數集中於此，避免散落在呼叫端。

    數值為初始設定，須由 E2 的前緣掃描檢驗，不是已驗證的最佳值。
    """

    lam_def: float = 1.0
    lam_fid: float = 1.0

    # 防禦項：margin 以 d 的尺度為準，d 預設為 LPIPS
    margin: float = 0.5

    # 防禦項的形式：
    #   "untargeted" — max(0, m − d(E(P(x_def)), E(x)))，把編輯結果推離原編輯，
    #                  方向不限（既有行為）
    #   "targeted"   — d(E(P(x_def)), y_target)，把編輯結果推向一個固定目標
    #
    # 無目標最大化在文獻上一貫比有目標脆弱：損失地形沒有盆地，只有一個
    # 「往外走」的方向，而該方向高度依賴當下的噪聲與 prompt。本專案實測的
    # 過擬合倍率 3.3x（訓練種子 0.3735 對未見種子 0.1133，site P r=16）正是
    # 這個現象。有目標則有明確的收斂點。
    defense_mode: str = "untargeted"

    # 保真項
    #
    # alpha_ssim 由 1.0 改為 0.0（E20，見模組 docstring 的修訂之三）：SSIM 在
    # 等 LPIPS 下把雜訊判得比模糊貴約 2 倍，留在梯度裡等於補貼模糊。保留
    # 係數與逐步記錄，可一行復原。
    alpha_ssim: float = 0.0
    # 原始 LPIPS 項的係數（不是 hinge，是 L_fid 裡那個直接相加的 lpips）。
    #
    # 預設 1.0 只為了讓 E2–E27 的既有數字可重現，它與 E20 的設計意圖相衝突。
    # E20 導入交集式 hinge 的理由是「加權和永遠可以用便宜的軸換貴的軸，交集式
    # 的可行域不允許這種交換」，但當時只加了 hinge，沒有拿掉這個原始項，於是
    # 保真項仍有一半是加權和。後果由 E27 實測：該項是一個持續把失真往零拉的
    # 力，最佳化停在「邊際防禦收益 = 邊際 LPIPS 成本」的平衡點，而該點在 τ
    # 之下——site C 末端 LPIPS 0.031–0.045、site P 0.040，τ=0.05 的 hinge 在
    # 60 步中只啟動 0–8 步。τ 因此不是綁定的那道約束，兩個條件停在各自不同的失真
    # 上，「匹配失真的比較」不成立。
    #
    # 設為 0 之後，LPIPS 在 τ 以內完全免費，最佳化會把預算用滿，τ 才真正成為
    # 匹配軸。正式重跑一律用 0。
    alpha_lpips: float = 1.0
    beta_linf: float = 100.0
    tau_linf: float = 0.06     # ≈ 15/255，對抗擾動文獻的常見上限量級

    # 綁定的保真度約束：LPIPS(x_def, x_base)，即「防禦本身造成多少感知
    # 可辨的改變」。取代原本的 PSNR 下限，理由見模組 docstring 的修訂之二。
    # τ 預設 0.05：主網格的像素注入 r=16 實測落在 LPIPS 0.0626，故此值
    # 比現況略緊；正式實驗以 {0.02, 0.05, 0.10} 掃描，得到曲線而非單點。
    gamma_lpips: float = 100.0
    tau_lpips: float = 0.05

    # 第二道綁定約束：鈍化。與 LPIPS hinge 取交集而非加權和——加權和
    # 永遠可以用便宜的軸換貴的軸，那正是 E18/E19 觀察到的行為；兩道各自的
    # hinge 使可行域是兩者的交集，不允許這種交換。
    # τ_acut = 0.04 的來源見模組 docstring 的修訂之三；應與 τ_lpips 一樣掃描。
    gamma_acut: float = 100.0
    tau_acut: float = 0.04

    # 第三道綁定約束：色度偏壓。與前兩道同樣取交集。
    #
    # 預設開啟（與 gamma_acut 於 E20 導入時的處置相同，而非 alpha_lpips
    # 那種預設關閉）。理由是忘記開會重演 E27 的失效——site C 會再次
    # 用色調偏移換防禦效果，而報告裡沒有任何一欄看得出來。E2–E27 的既有數字
    # 本來就已因 E26 的 guidance 缺陷而失效，不再需要為它們保留預設值。
    #
    # τ = 0.8 由人眼判讀（2026-08-01）：使用者在 `runs/p10_chroma_ladder/`
    # 的階梯上判讀「0.3 還有 0.6 都看不出來，1.0 以上才開始有一些細微色調
    # 變化⋯⋯1.0 要很仔細看才看得出來，可以直接取 0.8 或 1.0」。
    #
    # 取 0.8 而非 1.0：1.0 是已確認看得見的那一級（即使需要仔細看），
    # 把門檻設在該處等於允許約束放行一個可見的失真，約束就失去意義。0.8 落在
    # 0.6（確認看不見）與 1.0 之間，偏向安全側。
    gamma_chroma: float = 100.0
    tau_chroma: float = 0.8

    # PSNR 下限保留計算與記錄，但預設不參與梯度（gamma_psnr = 0）。
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
        self,
        y_def_list: List[torch.Tensor],
        y_orig_list: List[torch.Tensor],
        y_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """對 (淨化算子, 噪聲) 的取樣求平均，即 spec §5.1 的期望值估計。

        兩條分支的噪聲必須逐元素相同，否則量到的偏移主要來自噪聲差異。
        此不變量由呼叫端以共用 `noise` 保證（見 SDWrapper.sdedit 的介面）。

        `defense_mode="targeted"` 時改為最小化與 `y_target` 的距離。此時
        `y_orig_list` 只用於長度檢查與 `edit_shift` 的記錄，不進入梯度——
        有目標的定義就是「往哪裡去」，而不是「離哪裡遠」。
        """
        if len(y_def_list) != len(y_orig_list):
            raise ValueError(
                f"兩側取樣數不符：{len(y_def_list)} vs {len(y_orig_list)}，"
                "每個防禦分支必須對上使用相同噪聲的原圖分支"
            )
        mode = self.cfg.defense_mode
        if mode == "targeted":
            if y_target is None:
                raise ValueError(
                    "defense_mode='targeted' 需要 y_target；"
                    "缺少時不可退回無目標，兩者是不同的目標函數"
                )
            terms = [self.distance(yd, y_target) for yd in y_def_list]
        elif mode == "untargeted":
            terms = [
                torch.clamp(self.cfg.margin - self.distance(yd, yo), min=0.0)
                for yd, yo in zip(y_def_list, y_orig_list)
            ]
        else:
            raise ValueError(
                f"未知的 defense_mode: {mode!r}，只接受 'untargeted' 或 'targeted'"
            )
        return torch.stack(terms).mean()

    # ---- VAE 編碼器目標（PhotoGuard 的 encoder attack 形式）----

    def encoder_term(
        self, z_def: torch.Tensor, z_target: torch.Tensor
    ) -> torch.Tensor:
        """‖E_vae(x_def) − z_target‖²，不經過 UNet。

        存在理由是成本與過擬合兩件事同時解決：

        1. 成本：完全不走去噪鏈。E0 的成本模型是
           `秒 ≈ 1.05 + 0.384·k_inv + 0.304·n_edit`，此式的兩個係數項在此
           都消失，每步只剩一次 VAE 編碼。可以跑 1000 步而非 25 步。
        2. 過擬合：目標不依賴任何特定的編輯 prompt 或噪聲取樣，直接
           消除本專案最大的兩個過擬合來源（實測噪聲過擬合 3.3 倍，
           prompt 過擬合從未量過）。

        代價是它不再針對「這個編輯」最佳化，泛化性換特異性。這是一個
        需要實測的取捨，不是必然更好。
        """
        return torch.nn.functional.mse_loss(z_def, z_target)

    # ---- 保真項 ----

    def fidelity_term(
        self,
        x_def: torch.Tensor,
        x: torch.Tensor,
        x_base: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """回傳 (L_fid, 各分項的純量值)。分項一併回傳供逐步診斷。

        `x_base` 是 φ=0 時的輸出，即該 site 在未施加任何防禦時就已經
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
        # 各影像的重建誤差下限由 19.61 dB（car_00）到 31.01 dB（person_00），
        # 相差 11.4 dB。任何全域固定的 psnr_floor 都不可能同時適用：
        # 取 26 時對 car_00 與 car_01 不可達（hinge 恆啟動），對 person_00
        # 又完全不施力。改以 PSNR(x_def, x_base) 為對象後，門檻自動隨每張
        # 影像與每個 site 的下限調整，量的一律是「防禦讓 PSNR 掉了多少」。
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

        # 鈍化：對象同為 x_base，量的是「防禦本身讓影像鈍化了多少」。site P
        # 的 x_base 即 x。此量對位移不敏感，故空間變形只要不模糊就不被罰。
        acut = local_acutance_dev(xb, xd)

        # 色度偏壓：對象同為 x_base。量的是空間上連貫的色偏，而非色度
        # 誤差的量值——P9 實測 ΔE 那一類分不出加性雜訊與可見色偏（等 LPIPS
        # 下 2.44 對 2.79），而人眼分得出來。此量對隨機擾動不敏感（區塊內
        # 相消），故加性位置只要不整片偏色就不被罰。
        chroma = local_chroma_bias(xb, xd)

        pen_linf = torch.clamp(linf - c.tau_linf, min=0.0)
        pen_lpips = torch.clamp(lpips_rel - c.tau_lpips, min=0.0)
        pen_acut = torch.clamp(acut - c.tau_acut, min=0.0)
        pen_chroma = torch.clamp(chroma - c.tau_chroma, min=0.0)
        pen_psnr = torch.clamp(c.psnr_floor - psnr, min=0.0)

        total = (
            c.alpha_lpips * lpips
            + c.alpha_ssim * (1.0 - ssim)
            + c.beta_linf * pen_linf
            + c.gamma_lpips * pen_lpips
            + c.gamma_acut * pen_acut
            + c.gamma_chroma * pen_chroma
            + c.gamma_psnr * pen_psnr
        )
        parts = {
            # acut 直接參與梯度，故轉純量前顯式 detach；其餘各項由 piq 回傳
            # 時已不帶圖。這是記錄用的轉換，不影響 total 的計算圖。
            "fid_acut": float(acut.detach()),
            "fid_pen_acut": float(pen_acut.detach()),
            "fid_chroma": float(chroma.detach()),
            "fid_pen_chroma": float(pen_chroma.detach()),
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
        y_target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        c = self.cfg
        l_def = self.defense_term(y_def_list, y_orig_list, y_target=y_target)
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
            # 有目標模式下 edit_shift 仍以「離原編輯多遠」記錄，與無目標
            # 模式可直接比較。它不是該模式的優化目標，但仍是我們評測的量。
            "edit_shift": float(shift),
            "defense_mode": c.defense_mode,
            **parts,
        }
        return total, log
