"""紋理重相位 —— 非加性、停在像素空間、由構造保證恆等。

    x_def = OLA( irfft2( rfft2(w*P_b) * exp(i * g_b * m_w * theta_b) ) * w ) / OLA(w^2)

`P_b` 是以 hop = block/2 取出的重疊區塊，`w` 是 2D 週期 Hann 窗。相位偏移
`theta_b` 是唯一的可學參數；`g_b`（紋理度）與 `m_w`（徑向頻率）是由原圖與
網格算出的固定閘，不參與最佳化。

為什麼是相位
────────────────────────────────────────────────────────────────────
Random Phase Noise（Galerne, Gousseau & Morel, IEEE TIP 2011）證明：把影像的
傅立葉相位隨機化，微紋理的知覺外觀不變——紋理由幅度譜刻畫，相位只決定紋素
的擺放。本模塊把「隨機」換成「最佳化」。

其動機是三筆既有結果的共同機制（FND-004／013／026）：加性擾動把能量放在
高頻，VAE encoder 敏感而人眼不敏感；既有的非加性形式（warp、latent 球）
產生自然的低頻結構改動，VAE 忠實編碼它，於是每單位可見失真換到的位移少。
相位擾動要押的落差是——DISTS 的設計目標明文包含 tolerant to texture
resampling（Ding et al. 2020），人眼判紋理亦靠統計量，而 VAE encoder 的
卷積特徵對相位敏感。

三個由構造而非懲罰項保證的性質
────────────────────────────────────────────────────────────────────
1. **theta = 0 時逐位等於原圖。** 分析窗與合成窗各乘一次、再除以 `OLA(w^2)`,
   theta=0 時分子恰為 `OLA(w^2 * x)`，與分母逐位相消。
   這條式子不是本模塊發明的：它是 Griffin & Lim (1984) 對「由被修改過的 STFT
   還原訊號」這個問題的**最小平方最佳解**——先乘分析窗再疊加相加，多出來的
   那次加窗以窗平方和正規化補償。
   **依賴的是 NOLA 而非 COLA。** COLA（相鄰窗和為常數）是完美重建的**充分**
   條件；必要的是更弱的 NOLA（`sum w^2 > 0` 處處成立），由 `clamp_min(1e-8)`
   保證。故換窗型或換 hop 都不會靜默破壞恆等——這一點由測試實測
   （`test_identity_holds_for_non_cola_hop` 用不滿足 COLA 的 hop=8）。
2. **幅度譜逐位保留。** 係數乘上單位模的複數 `exp(i*theta)` 而非走
   `abs`／`angle` 再重組：後者在零幅度處梯度未定義，前者處處可微且
   `|X * e^{i*theta}| = |X|` 是代數恆等式。
3. **輸出為實數。** `rfft2`／`irfft2` 只存半平面，實數性由型別保證。
   但半平面裡 `fx = 0` 與 `fx = N/2` 兩行自身必須對 `fy` 共軛，對它們逐格
   施加獨立相位會破壞該關係（輸出仍是實數，只是那兩行的幅度不再保留）。
   故 `m_w` 在這兩行取 0——寧可少 2/17 的自由度，也不要一個只在兩行上
   失效、且不會有任何症狀的近似。

已知的不精確之處：STFT 一致性投影誤差
────────────────────────────────────────────────────────────────────
逐區塊各自轉相位之後，那組係數**一般不是任何一張實影像的 STFT**——重疊的
區塊之間互相不一致。Griffin & Lim (1984) 稱這種輸入為 modified STFT，
而上面第 1 點的重建式正是把它**投影回一致集合**的最小平方解。投影會改動
係數，所以「局部幅度譜保留」在整張圖的層級是近似而非恆等。

`amplitude_deviation()` 回報該偏差。構造上它應該接近 0；實測偏大即代表本模塊
在造新能量而不是重排相位，那會讓它退化成被紋理遮蔽的加性高頻噪聲
（規格 §6 風險一）。實測 0.0065–0.0653，且與效果正相關 r = +0.449（FND-040）。

**要壓低它有現成的辦法，已於 2026-08-17 實作**：`gl_iters > 0` 時，前向在
標準輸出之後再跑 `gl_iters` 輪 Griffin-Lim 迭代投影——每輪重新分析、把幅度
換回原圖的、再合成。`gl_iters = 0`（預設）時整段不執行，行為與加這個選項
之前逐位相同。跑幾輪之後 `amp_dev` 應該下降，此時效果掉多少就直接回答了
「效果來自相位重排還是來自新造的能量」（FND-040）。

參考文獻
────────────────────────────────────────────────────────────────────
- Griffin & Lim. Signal Estimation from Modified Short-Time Fourier Transform.
  IEEE TASSP 32(2):236-243, 1984.  ——重建式與一致性投影
- Allen & Rabiner. A Unified Approach to Short-Time Fourier Analysis and
  Synthesis. Proc. IEEE 65(11):1558-1564, 1977.  ——STFT 分析/合成框架
- Galerne, Gousseau & Morel. Random Phase Textures. IEEE TIP 20(1):257-267,
  2011.  ——相位隨機化保留微紋理外觀（整張圖，未切塊）
- Weickert. Coherence-Enhancing Diffusion Filtering. IJCV 31:111-127, 1999.
  ——結構張量的 coherence（紋理閘）
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.residual.base import ResidualModule


def hann2d(n: int, device, dtype) -> torch.Tensor:
    """(n, n) 的 2D 週期 Hann 窗，可分離構造。

    取週期（periodic）而非對稱版本：週期 Hann 在 hop = n/2 下相鄰窗和恰為 1
    （即滿足 COLA），對稱版本不是。本模塊的恆等性由 `OLA(w^2)` 正規化保證，
    **只需要 NOLA 而不需要 COLA**；保留週期版本是為了讓 `OLA(w^2)` 不出現
    接近零的格，也就是讓 NOLA 有充裕的餘裕。

    Hann 窗與 STFT 的分析/合成框架見 Allen & Rabiner, Proc. IEEE 65(11), 1977。
    """
    k = torch.arange(n, device=device, dtype=dtype)
    w = 0.5 - 0.5 * torch.cos(2.0 * math.pi * k / n)
    return torch.outer(w, w)


from src.residual.perceptual_weight import (
    FREQ_WEIGHTS,
    freq_weight as perceptual_freq_weight,
)


# BT.601 的亮度／色差矩陣。**反矩陣由 `torch.linalg.inv` 算出**，不用
# JFIF 公布的那組常數：後者正逆各自四捨五入到小數第六位，往返只互逆到
# 1.2e-6，而本模組要保住「theta = 0 時輸出逐位元等於原圖」這條構造保證。
# `src/baselines/jpeg_codec.py` 沿用 JFIF 的常數是對的——那支在模擬 libjpeg，
# 這裡不是。
_RGB2YCC = torch.tensor([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312],
], dtype=torch.float64)
_YCC2RGB = torch.linalg.inv(_RGB2YCC)


def luma_split(x01: torch.Tensor):
    """(N,3,H,W) 的 RGB 拆成 (N,1,H,W) 的亮度與 (N,2,H,W) 的色差。

    色差**不加 128 的偏移**：本模組不寫進 JPEG 位元流，偏移只會讓
    `luma_join` 多一次加減而不影響任何結果。
    """
    if x01.shape[1] != 3:
        raise ValueError(f"亮度拆分需要 3 通道，收到 {x01.shape[1]}")
    m = _RGB2YCC.to(device=x01.device, dtype=x01.dtype)
    ycc = torch.einsum("ij,njhw->nihw", m, x01)
    return ycc[:, :1], ycc[:, 1:]


def luma_join(y: torch.Tensor, chroma: torch.Tensor) -> torch.Tensor:
    """`luma_split` 的逆，精確互逆到浮點精度。"""
    m = _YCC2RGB.to(device=y.device, dtype=y.dtype)
    return torch.einsum("ij,njhw->nihw", m, torch.cat([y, chroma], dim=1))


def radial_gate(block: int, r_min: float, device, dtype,
                r_max: float = float("inf")) -> torch.Tensor:
    """(block, block//2+1) 的徑向頻率閘。**帶通**：`r_min <= r <= r_max`。

    低頻格帶著區塊的位置與結構，動它會在重疊相加後產生接縫，故歸一化半徑
    低於 `r_min` 的格取 0。`fx = 0` 與 `fx = block//2` 兩行一律取 0，理由見
    模組 docstring 第 3 點。

    `r_max`（2026-08-21 新增，**預設無窮大即維持原本的高通行為**）：穩健
    浮水印的標準作法是把訊號放在**中頻帶**——低頻改動可見、高頻被壓縮與
    模糊抹掉。本模組原本只有下界，等於高通；上界從未測過。本專案在
    crop_resize 上只留 13% 的淨增益、模糊上 24%，兩者都是高頻先被抹掉的
    症狀，故補上這個旋鈕。
    """
    if r_max <= r_min:
        raise ValueError(f"r_max={r_max} 不大於 r_min={r_min}，通帶是空的")
    fy = torch.fft.fftfreq(block, device=device, dtype=dtype) * 2.0   # [-1, 1)
    fx = torch.fft.rfftfreq(block, device=device, dtype=dtype) * 2.0  # [0, 1]
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    m = ((r >= r_min) & (r <= r_max)).to(dtype)
    m[:, 0] = 0.0
    m[:, -1] = 0.0
    return m


def texture_gate(
    x01: torch.Tensor,
    block: int,
    hop: int,
    energy_quantile: float = 0.5,
    eps: float = 1e-8,
    edge_power: float = 1.0,
) -> torch.Tensor:
    """逐區塊的紋理度 g in [0,1]，形狀 (B, L)。固定不可學。

        g = (1 - coherence^2)^edge_power * clamp(energy / energy_ref, 0, 1)

    `coherence = (l1 - l2)/(l1 + l2)` 取自結構張量（Foerstner & Guelch 1987、
    Bigun & Granlund 1987；coherence 這個量與名稱見 Weickert, IJCV 31, 1999）。
    邊緣的梯度方向一致,
    coherence 接近 1；紋理的梯度方向雜亂，接近 0。平坦區兩者都小，靠第二個
    因子壓掉——那裡任何改動都直接可見，而 coherence 在該處是零除的雜訊。

    `energy_ref` 取該影像自己的分位數而非固定常數：梯度能量的絕對值隨影像
    內容差好幾個數量級，固定常數會讓不同影像拿到不同的有效閘，而那個差異
    不會有症狀（與 DEC-021 取第 0 次迭代作正規化常數同型）。
    """
    # 三通道時取亮度加權（結構張量要的是人眼看得到的那個梯度）；其餘通道數
    # 取平均。latent 是 4 通道且沒有亮度語意，套 RGB 權重會是憑空的假設。
    if x01.shape[1] == 3:
        lum = (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]).unsqueeze(1)
    else:
        lum = x01.mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x01.device,
        dtype=x01.dtype,
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), ky)

    jxx, jxy, jyy = (block_mean(t, block, hop) for t in (gx * gx, gx * gy, gy * gy))

    tr = jxx + jyy
    disc = torch.sqrt(torch.clamp(((jxx - jyy) * 0.5) ** 2 + jxy ** 2, min=0.0))
    coh = (2.0 * disc) / (tr + eps)
    ref = torch.quantile(tr, energy_quantile, dim=1, keepdim=True).clamp_min(eps)
    return ((1.0 - coh ** 2) ** edge_power) * torch.clamp(tr / ref, 0.0, 1.0)


def pixel_texture_mask(
    x01: torch.Tensor,
    sigma: float,
    energy_quantile: float = 0.5,
    eps: float = 1e-8,
    edge_power: float = 1.0,
) -> torch.Tensor:
    """逐**像素**的紋理度 m in [0,1]，形狀 (B,1,H,W)。

    公式與 `texture_gate` 完全相同——
    `(1 - coherence^2)^edge_power * clamp(energy/ref, 0, 1)`
    ——差別只在把結構張量的區塊平均換成**高斯平滑**，於是解析度由「一格 32×32
    的區塊」變成「一個像素」。

    存在的理由（2026-08-20）：`texture_gate` 回傳的是每個區塊一個純量，而擾動
    是逐像素可見的。一個 32×32 的區塊若同時蓋到鬍鬚（高紋理）與臉頰（平滑），
    整塊拿到同一個高閘值，平滑的臉頰也被全強度旋轉相位——實測人臉會出現抹開
    與波紋。模組原本的設計意圖（「平坦區任何改動都直接可見，靠第二個因子壓掉」）
    是對的，逐區塊的解析度沒有兌現它。

    `sigma` 是高斯的標準差，單位是像素。**本專案指定**，沒有出處：要能分辨
    鬍鬚與臉頰就必須遠小於 block=32。此函式不設預設值，由呼叫端明給。
    """
    if sigma <= 0:
        raise ValueError(f"sigma 必須為正，收到 {sigma}")
    if x01.shape[1] == 3:
        lum = (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]).unsqueeze(1)
    else:
        lum = x01.mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x01.device, dtype=x01.dtype).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), ky)

    rad = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-rad, rad + 1, device=x01.device, dtype=x01.dtype)
    k1 = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    k1 = k1 / k1.sum()

    def smooth(t):
        t = F.conv2d(F.pad(t, (rad, rad, 0, 0), mode="reflect"), k1.view(1, 1, 1, -1))
        return F.conv2d(F.pad(t, (0, 0, rad, rad), mode="reflect"), k1.view(1, 1, -1, 1))

    jxx, jxy, jyy = (smooth(t) for t in (gx * gx, gx * gy, gy * gy))
    tr = jxx + jyy
    disc = torch.sqrt(torch.clamp(((jxx - jyy) * 0.5) ** 2 + jxy ** 2, min=0.0))
    coh = (2.0 * disc) / (tr + eps)
    flat = tr.reshape(tr.shape[0], -1)
    ref = torch.quantile(flat, energy_quantile, dim=1).clamp_min(eps)
    ref = ref.view(-1, 1, 1, 1)
    return ((1.0 - coh ** 2) ** edge_power) * torch.clamp(tr / ref, 0.0, 1.0)


def block_mean(t: torch.Tensor, block: int, hop: int) -> torch.Tensor:
    """(1,1,H,W) → (1,L)，每個區塊內取平均。

    紋理閘與遮罩閘都必須走這一條：兩者若用不同的填補或步幅，就落在不同的
    區塊格點上，相乘之後對不齊而且沒有症狀。
    """
    pad = block // 2
    t = F.pad(t, (pad, pad, pad, pad), mode="reflect")
    return F.unfold(t, kernel_size=block, stride=hop).mean(dim=1)


def rotate_spectrum(spec: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """對頻譜乘上單位模的複數。`shift` 可廣播到 `spec` 的形狀。

    乘上 `exp(i*theta)` 而非拆 `abs`／`angle` 再重組：`|X * e^{i*theta}| = |X|`
    是代數恆等式，且 `angle` 在原點的梯度未定義。
    """
    return spec * torch.complex(torch.cos(shift), torch.sin(shift))


def replace_magnitude(spec: torch.Tensor, target_mag: torch.Tensor,
                      eps: float = 1e-12) -> torch.Tensor:
    """保留 `spec` 的相位、把幅度換成 `target_mag`。Griffin-Lim 迭代的投影步。

    寫成 `target_mag * spec / |spec|` 而非 `target_mag * exp(i*angle(spec))`：
    後者在 `|spec| = 0` 處梯度未定義。除法的 `clamp_min` 使該處的極限為 0，
    即「原本是零的係數維持為零」——這是有定義的合理取值，不是為了掩蓋
    例外而加的保護。
    """
    return target_mag.to(spec.dtype) * (spec / spec.abs().clamp_min(eps))


class PhaseResidual(ResidualModule):
    """紋理重相位。phi = theta，形狀 (1, L, block, block//2+1)，RGB 三通道共用。

    通道共用相位是依 2026-08-13 的顏色結論：等亮度色度擾動的位移比 RGB 獨立
    低 31%，而 RGB 獨立那組的效果全部來自它順帶改變的亮度。共用相位使擾動
    落在結構而非色度上，不白付色偏的代價。
    """

    name = "texture_rephase"

    def __init__(
        self,
        size: int = 512,
        block: int = 32,
        hop: Optional[int] = None,
        r_min: float = 0.25,
        r_max: float = float("inf"),
        theta_max: float = math.pi,
        energy_quantile: float = 0.5,
        init_std: float = 0.0,
        seed: Optional[int] = None,
        gl_iters: int = 0,
        pixel_gate_sigma: float = 0.0,
        gain_max: float = 0.0,
        gate_edge_power: float = 1.0,
        freq_weight: str = "binary",
        freq_weight_power: float = 1.0,
        gain_weight: str = "shared",
        channels: str = "rgb",
    ):
        super().__init__()
        if block % 2 != 0:
            raise ValueError(f"block 必須是偶數，收到 {block}")
        hop = block // 2 if hop is None else hop
        if hop > block:
            raise ValueError(f"hop ({hop}) 大於 block ({block}) 會留下未覆蓋的像素")
        if not (0.0 < theta_max <= math.pi):
            raise ValueError(f"theta_max 必須落在 (0, pi]，收到 {theta_max}")
        if gl_iters < 0:
            raise ValueError(f"gl_iters 不可為負，收到 {gl_iters}")
        if gate_edge_power < 0:
            raise ValueError(
                f"gate_edge_power 不可為負，收到 {gate_edge_power}")
        # 頻率閘的知覺權重。"binary" = 二值帶通，逐位元等於加這個選項之前。
        # 名字打錯在這裡就拋錯（`freq_weight()` 會檢查），不等到 prepare_gates
        # ——後者在防禦迴圈裡逐張呼叫，錯誤會被淹在批次輸出裡。
        if freq_weight not in FREQ_WEIGHTS:
            raise ValueError(
                f"未知的 freq_weight：{freq_weight!r}，"
                f"可用的是 {sorted(FREQ_WEIGHTS)}")
        self.freq_weight = freq_weight
        # 定價的力道，0 = 退回二值閘。合法性由 `freq_weight()` 檢查。
        self.freq_weight_power = freq_weight_power
        # 增益的閘。"shared" = 與相位同一個閘，逐位元等於加這個選項之前。
        # "jnd" 另外乘上知覺權重，把振幅的創造推到人眼看不見的頻帶——那裡
        # 相位無事可做（1/f^2 的功率譜使高頻幾乎沒有能量可以旋轉），而增益
        # 造得出容量。逐帶的量測見 `runs/encoder_frequency_response`。
        if gain_weight not in ("shared", "jnd"):
            raise ValueError(
                f"未知的 gain_weight：{gain_weight!r}，可用的是 shared／jnd")
        self.gain_weight = gain_weight
        # 動哪些通道。"rgb" = 三個通道各自做同一件事（預設，逐位元等於加
        # 這個選項之前）；"y" = 只動亮度，色差原樣送回。增益在色度上累積成
        # 的全域色偏因此消失，而 RESULTS 記載「真正把失真砍半的是只動 Y 通道」。
        if channels not in ("rgb", "y"):
            raise ValueError(
                f"未知的 channels：{channels!r}，可用的是 rgb／y")
        self.channels = channels

        self.size = size
        self.block = block
        self.hop = hop
        self.pad = block // 2
        self.r_min = r_min
        self.r_max = r_max
        self.theta_max = theta_max
        self.energy_quantile = energy_quantile
        # 紋理閘裡壓制邊緣那個因子的指數：`(1 - coherence^2) ** gate_edge_power`。
        # **預設 1.0 逐位元等於加這個旋鈕之前**（`x ** 1.0` 在 float32／float64
        # 上都是恆等，已由測試釘住）。0 表示完全不壓制邊緣，此時紋理閘退化成
        # 只看梯度能量。
        #
        # 為什麼要它：邊緣（coherence 高）目前被設為 0，而邊緣正是導向濾波、
        # 雙邊濾波、TV 去噪這一類算子的不變集——把擾動趕出邊緣，等於主動放棄
        # 那幾個淨化算子底下唯一活得下來的位置。另一方面 DISTS 量的是紋理
        # 統計量，把擾動全部集中在紋理區會使同一個 LPIPS 換到更高的 DISTS。
        # 兩個理由指向同一個實驗：把這個指數放開。
        self.gate_edge_power = gate_edge_power
        self.gl_iters = gl_iters
        # > 0 時在重疊相加之後再乘一層**逐像素**的紋理遮罩（見
        # `pixel_texture_mask`）。**預設 0 表示關閉**，此時整條路徑與加這個
        # 選項之前逐位元相同——SDEdit 那條凍結的線必須能逐位重跑。
        if pixel_gate_sigma < 0:
            raise ValueError(f"pixel_gate_sigma 不可為負，收到 {pixel_gate_sigma}")
        self.pixel_gate_sigma = pixel_gate_sigma
        # > 0 時幅度譜也可學：spec' = |spec|·exp(g)·exp(i(phi + theta))。
        # **預設 0 表示關閉**，此時整條路徑與加這個選項之前逐位元相同。
        #
        # 為什麼是**乘性**而不是加性：加一項 `+ a·exp(i·psi)` 等於在頻譜上做
        # 加性擾動，換個座標寫而已，會放棄非加性的主張。乘性增益仍然是對
        # 影像自己的能量做重參數化——平坦區 |spec| ~ 0，乘上 exp(g) 還是 0，
        # 故它**不會**把擾動鋪到平坦背景（2026-08-21 已向使用者說明）。
        #
        # 它換到的是另外兩件事：g 沒有週期性，**失真的天花板被拆掉**（相位
        # 加到 pi 就繞回去，這是 sigma=1 那條臂在 DISTS 0.049 卡住的原因）；
        # 而且它動的是能量大小，卷積編碼器對這一維比對相位敏感。
        if gain_max < 0:
            raise ValueError(f"gain_max 不可為負，收到 {gain_max}")
        self.gain_max = gain_max

        padded = size + 2 * self.pad
        if (padded - block) % hop != 0:
            raise ValueError(
                f"size={size}、block={block}、hop={hop} 下 unfold 無法整除，"
                f"邊緣會被靜默丟棄"
            )
        side = (padded - block) // hop + 1
        self.n_blocks = side * side
        self.padded = padded

        nbins = (block, block // 2 + 1)
        if init_std > 0:
            gen = torch.Generator(device="cpu")
            if seed is not None:
                gen.manual_seed(seed)
            init = torch.randn(1, self.n_blocks, *nbins, generator=gen) * init_std
            init = init.clamp(-theta_max, theta_max)
        else:
            init = torch.zeros(1, self.n_blocks, *nbins)
        self.theta = nn.Parameter(init)
        # 對數增益，與 theta 同形、同樣三通道共用。一律建立（讓 state_dict
        # 穩定），但 gain_max = 0 時前向完全不碰它。
        self.gain = nn.Parameter(torch.zeros(1, self.n_blocks, *nbins))

        self.register_buffer("window", torch.zeros(block, block), persistent=False)
        self.register_buffer("freq_gate", torch.zeros(*nbins), persistent=False)
        self.register_buffer("tex_gate", torch.zeros(1, self.n_blocks), persistent=False)
        self.pixel_mask = None
        self._gates_ready = False
        self._gain_freq = None

    # ---- 閘 ----

    def prepare_gates(self, x01: torch.Tensor,
                      keep: Optional[torch.Tensor] = None) -> None:
        """由原圖算出兩個固定閘。必須在第一次前向之前呼叫一次。

        閘取自**原圖**而非當前的防禦圖：閘若隨 phi 移動，`g_b` 會變成優化目標
        的一部分（把擾動搬到閘自己放寬的地方），那不是本模塊要量的東西。

        `keep` (1,1,H,W) 給定時再乘上每個區塊落在其中的比例。inpainting 威脅
        模型下這是必要的：`SDWrapper.mask_latents` 算的是
        `encode(x01 * (1 - mask))`，**落在重畫區的擾動在進入 UNet 之前就被
        歸零**，那部分容量是白付的失真。傳 `keep = 1 - mask` 把容量集中到
        存活得下來的區域。取比例而非二值：區塊有重疊又加了窗，部分落在界外
        的區塊仍有貢獻，二值化會把邊界上的容量整塊丟掉。
        """
        device, dtype = x01.device, x01.dtype
        self.window = hann2d(self.block, device, dtype)
        # 閘 = 帶通遮罩 x 知覺權重。相乘而不是相加：權重不得讓通帶外的
        # 頻格復活，尤其是 rfft2 共軛對稱依賴的 fx=0 與 fx=N/2 兩行。
        self.freq_gate = radial_gate(
            self.block, self.r_min, device, dtype, self.r_max
        ) * perceptual_freq_weight(self.freq_weight, self.block, device,
                           dtype, self.freq_weight_power)
        tex = texture_gate(x01, self.block, self.hop, self.energy_quantile,
                           edge_power=self.gate_edge_power)
        if keep is not None:
            tex = tex * block_mean(keep.to(device=device, dtype=dtype),
                                   self.block, self.hop)
        self.tex_gate = tex.detach()
        if self.pixel_gate_sigma > 0:
            m = pixel_texture_mask(x01, self.pixel_gate_sigma,
                                   self.energy_quantile,
                                   edge_power=self.gate_edge_power)
            if keep is not None:
                m = m * keep.to(device=device, dtype=dtype)
            self.pixel_mask = m.detach()
        else:
            self.pixel_mask = None
        if self.gain_weight == "jnd":
            self._gain_freq = perceptual_freq_weight(
                "jpeg_luma", self.block, device, dtype)
        else:
            self._gain_freq = None
        self._gates_ready = True

    def gain_gate(self) -> torch.Tensor:
        """增益被允許出現的位置。`shared` 時逐位元等於 `gate()`。"""
        g = self.gate()
        return g if self._gain_freq is None else g * self._gain_freq

    def gate(self) -> torch.Tensor:
        """(1, L, 1, 1) 與 (block, nbins) 廣播後的合成閘，供診斷與前向共用。"""
        return self.tex_gate[..., None, None] * self.freq_gate

    def active_fraction(self) -> float:
        """紋理閘的有效面積佔比（規格 §6 風險三的前置量測）。"""
        return float(self.tex_gate.mean())

    # ---- 前向 ----

    def _fold_norm(self, batch: int) -> torch.Tensor:
        w2 = (self.window ** 2).reshape(1, -1, 1).expand(batch, -1, self.n_blocks)
        out = F.fold(
            w2,
            output_size=(self.padded, self.padded),
            kernel_size=self.block,
            stride=self.hop,
        )
        return out[..., self.pad:self.pad + self.size, self.pad:self.pad + self.size]

    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) → 加窗區塊頻譜 (B,C,L,block,block//2+1)。STFT 的分析側。"""
        b, c = x.shape[0], x.shape[1]
        xp = F.pad(x, (self.pad,) * 4, mode="reflect")
        blocks = F.unfold(xp, kernel_size=self.block, stride=self.hop)
        blocks = blocks.view(b, c, self.block, self.block, self.n_blocks)
        blocks = blocks.permute(0, 1, 4, 2, 3)                       # (B,C,L,n,n)
        return torch.fft.rfft2(blocks * self.window, norm="ortho")

    def synthesize(self, spec: torch.Tensor) -> torch.Tensor:
        """區塊頻譜 → (B,C,size,size)。Griffin & Lim (1984) 的最小平方重建。

        與 `analyze` 是一對：`synthesize(analyze(x)) == x`（NOLA 下的恆等），
        整個模組只有這一條合成路徑，故主前向與 Griffin-Lim 迭代不可能分岔。
        """
        b, c = spec.shape[0], spec.shape[1]
        out = torch.fft.irfft2(
            spec, s=(self.block, self.block), norm="ortho") * self.window
        out = out.permute(0, 1, 3, 4, 2).reshape(
            b, c * self.block * self.block, self.n_blocks
        )
        folded = F.fold(
            out,
            output_size=(self.padded, self.padded),
            kernel_size=self.block,
            stride=self.hop,
        )
        folded = folded[..., self.pad:self.pad + self.size,
                        self.pad:self.pad + self.size]
        return folded / self._fold_norm(b).clamp_min(1e-8)

    def pixel_residual(self, x01: torch.Tensor) -> Optional[torch.Tensor]:
        """`channels = "y"` 時只在亮度上跑整條管線，色差原樣送回。

        拆／併用的是精確互逆的矩陣（`luma_split`），故 `theta = 0` 的逐位元
        恆等在這條路徑上仍然成立。閘由**亮度**算出——結構張量本來就取亮度
        加權，兩條路徑因此拿到同一組閘。
        """
        if self.channels == "y":
            y, chroma = luma_split(x01)
            return luma_join(self._rephase(y), chroma)
        return self._rephase(x01)

    def _rephase(self, x01: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return x01
        if not self._gates_ready:
            raise RuntimeError(
                "prepare_gates() 未呼叫。閘必須由原圖算出並固定，"
                "延後到第一次前向會讓它跟著防禦圖漂移"
            )
        h, w = x01.shape[-2:]
        if h != self.size or w != self.size:
            raise ValueError(f"影像尺寸 {h}x{w} 與建構時的 size={self.size} 不符")

        shift = torch.clamp(self.theta, -self.theta_max, self.theta_max)
        shift = (shift * self.gate()).unsqueeze(1)                   # (1,1,L,n,nb)

        spec = self.analyze(x01)
        rot = rotate_spectrum(spec, shift)
        if self.gain_max > 0:
            # (1,L,n,nb) → (1,1,L,n,nb)，與 spec 的 (B,C,L,n,nb) 廣播。
            # 閘由 `gain_weight` 決定。預設 `shared` 與相位同一個閘——那是
            # 歸因期間的約束，讓兩者的比較是乾淨的「動什麼」而不是「動哪裡」。
            # `jnd` 把增益的預算推到人眼看不見的頻帶，理由見 `__init__`。
            g = torch.clamp(self.gain, -self.gain_max, self.gain_max)
            rot = rot * torch.exp(g * self.gain_gate()).unsqueeze(1)
        x_def = self.synthesize(rot)

        # Griffin-Lim 的迭代投影：反覆把幅度換回**目標**幅度，逼近「相位被
        # 轉過、幅度為目標值」那個一般不存在的訊號。gl_iters = 0 時整段不
        # 執行，行為與加這個選項之前逐位相同。
        #
        # 增益開著時目標必須是**改過的**幅度 `rot.abs()`，用 `spec.abs()`
        # 會把增益整個投影掉。關著時仍用 `spec.abs()`——它與 rot.abs() 在
        # 浮點上未必逐位相同，而既有批次要能逐位重跑。
        target_mag = rot.abs() if self.gain_max > 0 else spec.abs()
        for _ in range(self.gl_iters):
            x_def = self.synthesize(
                replace_magnitude(self.analyze(x_def), target_mag))

        if self.pixel_mask is not None:
            # 逐像素混合：擾動只落在紋理遮罩允許的像素上。**這一步不改變
            # 相位旋轉本身**，只限制它的效果被允許出現在哪裡——即模組
            # docstring 原本就宣稱、但逐區塊的閘沒有兌現的那件事。
            x_def = x01 + self.pixel_mask * (x_def - x01)
        return x_def

    # ---- 診斷 ----

    @torch.no_grad()
    def amplitude_deviation(self, x01: torch.Tensor) -> float:
        """整張圖層級的局部幅度譜相對偏差。

        **只有 `gain_max == 0` 時構造上才應接近 0**（模組 docstring）。增益
        開著時幅度是被刻意改動的，這個量會是有限值而非誤差，不可當診斷用。
        """
        x_def = self.pixel_residual(x01)
        a = self.analyze(x01).abs()
        b = self.analyze(x_def).abs()
        return float((a - b).norm() / a.norm().clamp_min(1e-12))

    def linf(self) -> float:
        return float(self.theta.detach().abs().max())
