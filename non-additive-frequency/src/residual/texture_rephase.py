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
    SURVIVAL_WEIGHTS,
    survival_weight as expected_survival_weight,
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


# Watson (1993) 的兩個常數。亮度遮蔽的指數 a_T = 0.649、對比遮蔽的指數
# w = 0.7，兩者取自 Watson, "DCT quantization matrices visually optimized for
# individual images", SPIE 1913:202-216, 1993，並依 Podilchuk & Zeng,
# "Image-adaptive watermarking using visual models", IEEE JSAC 16(4):525-539,
# 1998 的用法搬到浮水印式的預算分配上。**兩個值都不是本專案指定的。**
WATSON_LUMINANCE_EXPONENT = 0.649
WATSON_CONTRAST_EXPONENT = 0.7


def _floor_price_complement(module, x01: torch.Tensor,
                            base: torch.Tensor) -> torch.Tensor:
    """加法項花在**乘法那一半可達量最少**的區塊上。

    存在理由：相位與增益都是乘法，能改動的量正比於 `|spec|`——通帶內幾乎沒有
    能量的區塊，強度開再大也動不了。加法項就是為了那些地方而存在的。均勻的
    價目表把預算同時發給乘法已經在動的區塊，那既是重複投資，也正是
    DCT-Shield 的形狀（逐係數的 eps·Q，跨區塊為常數）。

    「可達量」取乘法那一半在該區塊上能碰到的幅度總量：

        reach_b = || |S_b(w)| * g_b * m_w ||_2        （通帶內，逐區塊一個純量）

    再取 `1 - reach_b / max_b reach_b`。除以最大值使它落在 [0, 1] 且無單位，
    **不引入任何新的常數**。紋理閘與頻率閘都已經是固定的，故這個因子也固定，
    不參與最佳化。
    """
    spec = module.analyze(x01)
    mag = spec.abs().mean(dim=1)                       # (B, L, n, nb)
    reach = (mag * module.gate()).flatten(2).norm(dim=2)   # (B, L)
    w = 1.0 - reach / reach.max().clamp_min(1e-12)
    return base[None, None] * w[..., None, None]


def _floor_price_complement_rank(module, x01: torch.Tensor,
                                 base: torch.Tensor) -> torch.Tensor:
    """同 `complement`，但用**名次**而不是「1 − reach/max」。

    為什麼要這個變體：可達量的分布是重尾的，多數區塊都遠低於最大值，於是
    `1 − reach / max reach` 對它們幾乎都等於 1，預算實際上只比均勻偏了一兩個
    百分點（實測：可達量低的一半拿到 51.5%–62.3%，均勻是 50%）。名次轉換把
    分布拉平，低的一半固定拿到 75%，**且仍然沒有引入任何自由參數**。

        w_b = 1 − rank(reach_b) / (L − 1)

    名次由小到大，最大的那一格恰為 0，與 `complement` 的邊界一致。
    """
    spec = module.analyze(x01)
    mag = spec.abs().mean(dim=1)
    reach = (mag * module.gate()).flatten(2).norm(dim=2)          # (B, L)
    order = torch.argsort(reach, dim=1)
    rank = torch.empty_like(order)
    idx = torch.arange(reach.shape[1], device=reach.device)
    rank.scatter_(1, order, idx.expand_as(order))
    w = 1.0 - rank.to(base.dtype) / max(reach.shape[1] - 1, 1)
    return base[None, None] * w[..., None, None]


def _floor_price_watson(module, x01: torch.Tensor,
                        base: torch.Tensor) -> torch.Tensor:
    """逐區塊逐頻格的知覺門檻：亮度遮蔽 × 對比遮蔽。

        t_b(w)  = base(w) * floor * (DC_b / mean DC) ** a_T      亮度遮蔽
        s_b(w)  = max( t_b(w), |S_b(w)| ** w * t_b(w) ** (1-w) ) 對比遮蔽

    `base` 是 JPEG 亮度量化表，即 Watson 模型裡的 t_ij；DCT-Shield 的預算
    只到這一層（逐係數 eps·Q，跨區塊為常數）。多出來的兩項把價目變成
    **內容相依**——這是本方法與它在加法那一半上唯一的構造差異。

    零支撐自動保住：`base = 0` 處 `t = 0`，而 `0 ** (1-w) = 0`。
    """
    spec = module.analyze(x01)
    mag = spec.abs().mean(dim=1)                       # (B, L, n, nb)
    dc = mag[..., 0, 0]                                # (B, L)
    lum = (dc / dc.mean().clamp_min(1e-12)).clamp_min(1e-6)
    lum = lum ** WATSON_LUMINANCE_EXPONENT
    t = base[None, None] * lum[..., None, None] * module.spectral_floor
    w = WATSON_CONTRAST_EXPONENT
    s = torch.maximum(t, mag ** w * t ** (1.0 - w))
    return s / module.spectral_floor


FLOOR_GATES = {
    "uniform": None,          # 預設，價目只看頻格。`_build_floor_price` 直接回傳 base
    "complement": _floor_price_complement,
    "complement_rank": _floor_price_complement_rank,
    "watson": _floor_price_watson,
}


# 可學的空間包絡（`floor_envelope`）。"none" = 關閉，前向完全不建立它，
# 逐位元等於加這個旋鈕之前。"gauss" = 下面 `PhaseResidual.envelope()` 的
# 高斯凸包 x 徑向低通。
FLOOR_ENVELOPES = ("none", "gauss")

# 包絡乘在哪一半上。"floor" = 只乘加法項的價目表（預設）；"all" = 連相位與
# 增益的閘也乘。**兩者的總量處理不同**，見 `PhaseResidual.envelope()`。
FLOOR_ENVELOPE_SCOPES = ("floor", "all")


def window_centres(side: int, hop: int, size: int, device, dtype) -> torch.Tensor:
    """每個分析視窗的中心在**正規化影像座標**上的位置，形狀 (side*side, 2)。

    `analyze()` 對 `reflect` 填補過的影像做 `unfold(kernel=block, stride=hop)`，
    第 (row, col) 個視窗在填補座標上的左上角是 `(row*hop, col*hop)`，中心是
    `(row*hop + block/2, col*hop + block/2)`；扣掉 `pad = block//2` 之後，
    **中心在原圖像素座標上恰為 `(row*hop, col*hop)`**。

    正規化取 `u = 2*(row*hop)/size - 1`，於是四角是 (-1,-1) 與 (1,1)，
    與 `torch.nn.functional.grid_sample` 的 `align_corners=True` 同一套座標。
    回傳的列序與 `unfold` 的視窗序（row-major）相同——順序錯掉的話包絡會被
    轉置，而輸出仍然是一張合法的影像，不會有任何症狀。
    """
    idx = torch.arange(side, device=device, dtype=dtype)
    coord = 2.0 * (idx * hop) / size - 1.0                 # (side,)
    yy = coord[:, None].expand(side, side).reshape(-1)
    xx = coord[None, :].expand(side, side).reshape(-1)
    return torch.stack([yy, xx], dim=-1)                   # (L, 2)


def radial_radius(block: int, device, dtype) -> torch.Tensor:
    """(block, block//2+1) 的歸一化徑向頻率 r。與 `radial_gate` 同一套座標。

    抽成函式而不是在兩處各寫一次：`radial_gate` 與包絡的低通若用不同的
    座標約定（例如一邊 [-1,1) 一邊 [-0.5,0.5)），低通的截止就會對到別的地方，
    而輸出仍然是一張合法的影像。
    """
    fy = torch.fft.fftfreq(block, device=device, dtype=dtype) * 2.0
    fx = torch.fft.rfftfreq(block, device=device, dtype=dtype) * 2.0
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


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
        survival_weight: str = "none",
        gain_weight: str = "shared",
        channels: str = "rgb",
        spectral_floor: float = 0.0,
        floor_gate: str = "uniform",
        floor_r_min: Optional[float] = None,
        floor_r_max: Optional[float] = None,
        floor_survival: str = "none",
        floor_envelope: str = "none",
        floor_envelope_k: int = 1,
        floor_envelope_scope: str = "floor",
        theta_budget: float = 0.0,
        coarsen: int = 1,
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
        # 期望存活振幅。"none" = 全 1，逐位元等於加這一層之前。名字打錯
        # 在這裡就拋錯，理由同 `freq_weight`。
        if survival_weight not in SURVIVAL_WEIGHTS:
            raise ValueError(
                f"未知的 survival_weight：{survival_weight!r}，"
                f"可用的是 {sorted(SURVIVAL_WEIGHTS)}")
        self.survival_weight = survival_weight
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
        # 頻譜加性下限。相位與增益都是**乘法**，而平坦區的 |spec| 接近零，
        # 乘任何東西還是接近零——13 張裡失敗的那幾張全是大面積平滑主體。
        # 這一項在頻譜上**加**一個由 JPEG 亮度量化表定價的量，且**只乘徑向
        # 帶通、不乘紋理閘**（紋理閘在平坦區就是零，乘了它等於沒加）。
        # 0 = 關閉，逐位元等於加這個選項之前。代價：方法不再是純粹的非加性
        # 重參數化，故兩個設定都是主線、分開報（`docs/METHOD.md`）。
        if spectral_floor < 0:
            raise ValueError(
                f"spectral_floor 不可為負，收到 {spectral_floor}")
        self.spectral_floor = spectral_floor
        # 加法項的價目表要不要隨**區塊**變。`uniform`（預設）逐位元等於加這個
        # 旋鈕之前：價目只看頻格，跨區塊是一個常數——而那正是 DCT-Shield 的
        # 形狀（逐係數 ±eps·Q，Q 只看頻率）。兩個替代品各自對應一條可辯護的
        # 差異，說明見 `FLOOR_GATES`。
        if floor_gate not in FLOOR_GATES:
            raise ValueError(
                f"未知的 floor_gate：{floor_gate!r}，"
                f"可用的是 {sorted(FLOOR_GATES)}")
        self.floor_gate = floor_gate
        # 加法項**自己的**徑向帶。`None`（預設）表示沿用相位那一半的
        # `r_min`／`r_max`，逐位元等於加這兩個旋鈕之前。
        #
        # 為什麼兩半要能分開：乘性那一半（相位 exp(i*theta) 與增益 exp(g)）
        # 能改動的量正比於原圖自己的振幅 `|S_b(w)|`，所以它必須待在有能量的
        # 地方；加法項不受該限制。而高斯模糊在頻域乘的是實正數
        # `exp(-2*pi^2*sigma^2*f^2)`，把高頻的振幅整個拿走——「編碼器對哪一帶
        # 敏感」與「哪一帶活得過模糊」方向相反。兩半共用一個帶通時，把上界
        # 壓低會連乘性那一半的未淨化強度一起削掉。分開之後乘性可以留在高頻、
        # 加性可以放到低頻，各買各的。
        #
        # 合法性（`r_max > r_min`）由 `radial_gate` 檢查，不在此重複。
        self.floor_r_min = floor_r_min
        self.floor_r_max = floor_r_max
        # 加法項**自己的**期望存活振幅。`survival_weight` 只乘在相位／增益的
        # 閘上（`prepare_gates`），加法項的價目表看不到它。`"none"`（預設）
        # 是全 1，逐位元等於加這個旋鈕之前。名字打錯在這裡就拋錯，理由同
        # `survival_weight`：靜默回退會讓一整批掃描跑成基準的重複。
        if floor_survival not in SURVIVAL_WEIGHTS:
            raise ValueError(
                f"未知的 floor_survival：{floor_survival!r}，"
                f"可用的是 {sorted(SURVIVAL_WEIGHTS)}")
        self.floor_survival = floor_survival
        # 可學的**空間包絡**。本模組原本沒有任何空間定位的自由度：紋理閘會
        # 定位但由原圖決定、不可學；`floor_survival` 已在**頻率**上做軟性
        # 挑選，空間上仍是均勻的。
        #
        # 為什麼要它：裁切（`crop_resize0.1` 是繞中心放大 1.2488x，落在中央
        # 80% 以內的才留下來）與模糊（頻域乘 `exp(-2*pi^2*sigma^2*f^2)`，
        # 與擾動放在畫面哪裡無關，但空間上壓得越窄、頻譜攤得越寬，單位預算
        # 下被拿走的**更多**）兩者的交集是「靠近中心的、大尺度的、平滑的
        # 結構」。**空間定位是這個構造缺的那一個旋鈕**，而總量正規化之下
        # 「集中」等價於「在少數位置上把振幅放大」。
        #
        # 五個純量與 theta／gain／floor 走同一條 PGD：K 組中心 `(cy, cx)`、
        # K 個空間尺度 `s`、一個徑向低通截止 `f_c`、一個強度 `beta`。
        # `beta = 0` 時包絡**逐位元**是 1（`(1-0) + 0*bump = 1.0`），故零初始化
        # 即恆等，與 theta 的零初始化同性質。
        if floor_envelope not in FLOOR_ENVELOPES:
            raise ValueError(
                f"未知的 floor_envelope：{floor_envelope!r}，"
                f"可用的是 {list(FLOOR_ENVELOPES)}")
        self.floor_envelope = floor_envelope
        if floor_envelope_scope not in FLOOR_ENVELOPE_SCOPES:
            raise ValueError(
                f"未知的 floor_envelope_scope：{floor_envelope_scope!r}，"
                f"可用的是 {list(FLOOR_ENVELOPE_SCOPES)}")
        self.floor_envelope_scope = floor_envelope_scope
        if floor_envelope_k < 1 or floor_envelope_k != int(floor_envelope_k):
            raise ValueError(
                f"floor_envelope_k 必須是 >= 1 的整數，收到 {floor_envelope_k}")
        self.floor_envelope_k = int(floor_envelope_k)
        # 只有加法項、卻沒有加法項可乘：包絡的梯度恆為零，跑出來的列與關著
        # 的列一模一樣而 CSV 上 `floor_envelope` 那一欄照樣寫著 `gauss`。
        # 拒絕啟動而不是靜默空轉。
        if (floor_envelope != "none" and floor_envelope_scope == "floor"
                and spectral_floor <= 0):
            raise ValueError(
                "floor_envelope_scope='floor' 需要 spectral_floor > 0，"
                f"收到 {spectral_floor}——包絡會沒有東西可乘")
        # 幅度相依的相位上限（Perturbing the Phase, arXiv:2602.06577）。
        #
        # 把係數 X 的相位轉 theta，係數本身移動 `2|X|·sin(theta/2)`。要把那個
        # 位移界在 eps 以內，相位就必須滿足
        #
        #     |theta| <= 2·arcsin( eps / (2|X|) )        （2|X| > eps）
        #     相位自由                                   （2|X| <= eps）
        #
        # 這一條解決的是本專案記過的缺陷：**固定的 theta 不等於固定的失真**
        # （FND-038，同一個 theta 在 24 張圖上 PSNR 由 23.15 漂到 39.54）。
        # 上限由**原圖**的幅度算出並固定，與兩個閘同型，不參與最佳化。
        #
        # 0 = 關閉，逐位元等於加這個旋鈕之前。
        if theta_budget < 0:
            raise ValueError(f"theta_budget 不可為負，收到 {theta_budget}")
        self.theta_budget = theta_budget

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
        self.side = side
        self.n_blocks = side * side
        self.padded = padded

        # 三個空間場（theta／gain／floor）的**視窗網格解析度**。1 = 每個視窗
        # 一組獨立參數，逐位元等於加這個旋鈕之前。k > 1 時參數只存在
        # `ceil(side/k)` 見方的粗網格上，前向再雙線性升取樣回 `side` 見方。
        #
        # 為什麼：本方法的 hop = 8、block = 32，每個像素被 16 個視窗覆蓋。
        # 相鄰視窗的旋轉角互相獨立時，那 16 份貢獻彼此不同調，重疊相加會在
        # 選定的頻格之外攤出一層**寬頻**能量——那正是 JPEG 量化最先丟掉的
        # 部分。強制相鄰視窗的角度平滑變化使那 16 份貢獻同調，能量回到被選中
        # 的頻格上。
        #
        # **與 IAM（arXiv:2402.16586）的差別必須寫清楚。** IAM 是把**影像**
        # 降解析度、在該解析度上走梯度、再升回去，於是擾動本身沒有高頻。本處
        # 平滑的是**視窗網格上的旋轉角**，管的是擾動的空間包絡是否連續，
        # 而不是視窗**內**的頻率成分（後者由徑向帶通 `r_min`／`r_max` 決定，
        # 不受本旋鈕影響）。兩者的動機同源、機制不同，不可互相代稱。
        if coarsen < 1 or coarsen != int(coarsen):
            raise ValueError(f"coarsen 必須是 >= 1 的整數，收到 {coarsen}")
        coarsen = int(coarsen)
        if coarsen > 1 and theta_budget > 0:
            # `theta_cap` 是逐（視窗, 頻格）由該處幅度算出的上限，形狀在
            # **細**網格上；把它拿去夾粗網格的參數是拿兩個不同索引空間的東西
            # 相比。與其靜默夾錯，不如拒絕。
            raise ValueError(
                "coarsen > 1 與 theta_budget > 0 不可同時使用："
                "theta_cap 定義在細網格上，無法夾粗網格的參數")
        self.coarsen = coarsen
        self.side_c = -(-side // coarsen) if coarsen > 1 else side

        nbins = (block, block // 2 + 1)
        n_param = self.side_c * self.side_c
        if init_std > 0:
            gen = torch.Generator(device="cpu")
            if seed is not None:
                gen.manual_seed(seed)
            init = torch.randn(1, n_param, *nbins, generator=gen) * init_std
            init = init.clamp(-theta_max, theta_max)
        else:
            init = torch.zeros(1, n_param, *nbins)
        self.theta = nn.Parameter(init)
        # 對數增益，與 theta 同形、同樣三通道共用。一律建立（讓 state_dict
        # 穩定），但 gain_max = 0 時前向完全不碰它。
        self.gain = nn.Parameter(torch.zeros(1, n_param, *nbins))
        # 加性下限的係數。恆存在（形狀固定，state_dict 才不會分岔），
        # 但 `spectral_floor == 0` 時前向完全不碰它。
        self.floor = nn.Parameter(torch.zeros(1, n_param, *nbins))

        # ---- 可學的空間包絡：五個純量 ----
        #
        # **四個邊界全部由網格導出，沒有一個是憑感覺挑的常數。**
        #
        #   sigma_min = 2*hop/size   相鄰視窗在正規化座標上的間距。再窄就
        #                            落在單一視窗上，總量正規化會把它放大成
        #                            一個無界的尖峰，那不是「集中」是數值爆炸。
        #   sigma_max = 2*sqrt(2)    正規化座標下影像的對角線長。此時最遠的
        #                            角落仍有 exp(-1) = 0.368，再大只是更平。
        #   fc_min    = 2/block      `fftfreq(block)*2` 的格點間距，即一格。
        #   fc_max    = sqrt(2)      格點上存在的最大半徑（角落 (1,1)）。
        #
        # 初值同樣由座標導出：中心取影像中心 (0, 0)，`sigma = 1` 是影像
        # 半寬、`f_c = 1` 是單軸的 Nyquist 半徑，`beta = 0` 即恆等。
        self.sigma_min = 2.0 * hop / size
        self.sigma_max = 2.0 * math.sqrt(2.0)
        self.fc_min = 2.0 / block
        self.fc_max = math.sqrt(2.0)
        k = self.floor_envelope_k
        # 一律建立（讓 `state_dict` 的形狀不隨旗標分岔），但
        # `floor_envelope == "none"` 時前向完全不碰它們。
        self.env_center = nn.Parameter(torch.zeros(k, 2))
        self.env_sigma = nn.Parameter(torch.ones(k))
        self.env_fc = nn.Parameter(torch.ones(1))
        self.env_beta = nn.Parameter(torch.zeros(1))

        self.register_buffer("window", torch.zeros(block, block), persistent=False)
        self.register_buffer("freq_gate", torch.zeros(*nbins), persistent=False)
        self.register_buffer("tex_gate", torch.zeros(1, self.n_blocks), persistent=False)
        # 包絡的兩張座標表。與 `window`／`freq_gate` 同型：形狀在建構時固定，
        # 值在 `prepare_gates` 填（那時才知道 device 與 dtype）。
        self.register_buffer("win_xy", torch.zeros(self.n_blocks, 2),
                             persistent=False)
        self.register_buffer("freq_r", torch.zeros(*nbins), persistent=False)
        self.pixel_mask = None
        self.theta_cap = None
        self._gates_ready = False
        self._gain_freq = None
        self._floor_price = None

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
        # 包絡的座標表。**無條件填**，與 `floor_envelope` 是否開啟無關：
        # 它們只是網格，填了不會改變任何輸出，而只在開啟時才填會讓
        # 「先建再改旗標」這條路徑拿到一張全零的座標表。
        self.win_xy = window_centres(self.side, self.hop, self.size,
                                     device, dtype)
        self.freq_r = radial_radius(self.block, device, dtype)
        # 閘 = 帶通遮罩 x 知覺權重。相乘而不是相加：權重不得讓通帶外的
        # 頻格復活，尤其是 rfft2 共軛對稱依賴的 fx=0 與 fx=N/2 兩行。
        self.freq_gate = radial_gate(
            self.block, self.r_min, device, dtype, self.r_max
        ) * perceptual_freq_weight(self.freq_weight, self.block, device,
                           dtype, self.freq_weight_power)
        # 期望存活振幅：最佳化迴圈跑的是未淨化的前向，看不到「模糊會把
        # 高頻整個拿掉」，那一半必須寫進閘裡。none 時是全 1，逐位元等於
        # 加這一層之前。
        self.freq_gate = self.freq_gate * expected_survival_weight(
            self.survival_weight, self.block, device, dtype)
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
        # 幅度相依的相位上限。取通道平均的幅度：三個通道共用同一個 theta
        # （見類別 docstring），上限也必須共用，否則最緊的那個通道會被放行。
        if self.theta_budget > 0:
            mag = self.analyze(x01).abs().mean(dim=1)          # (B, L, n, nb)
            ratio = (self.theta_budget / (2.0 * mag).clamp_min(1e-12)).clamp(max=1.0)
            self.theta_cap = (2.0 * torch.asin(ratio)).detach()
        else:
            self.theta_cap = None
        # 加法項的價目表。基底是徑向帶通 × 知覺權重（**不含相位那一半的
        # 紋理閘**）；`floor_gate` 決定要不要再乘一層逐區塊的因子。
        self._floor_price = (
            self._build_floor_price(x01, device, dtype)
            if self.spectral_floor > 0 else None)
        if self.gain_weight == "jnd":
            self._gain_freq = perceptual_freq_weight(
                "jpeg_luma", self.block, device, dtype)
        else:
            self._gain_freq = None
        self._gates_ready = True

    def _build_floor_price(self, x01: torch.Tensor, device,
                           dtype) -> torch.Tensor:
        """加法項的價目表。`uniform` 回傳 (n, nb)，其餘回傳 (1, L, n, nb)。

        三層定價，由頻格到區塊：

            徑向帶通（`floor_r_min`／`floor_r_max`，None 時沿用相位那一半的
            `r_min`／`r_max`）
          × `jpeg_luma` 知覺權重（取一次方，見 `floor_price`）
          × 期望存活振幅（`floor_survival`，"none" 時全 1）
          × 逐區塊的因子（`floor_gate`，"uniform" 時沒有這一層）

        總預算正規化：**全部**縮放到同一個參考點
        ────────────────────────────────────────────────────────────
        參考點固定取「相位那一半的帶通 × `jpeg_luma`」的平均值，四個旋鈕
        （`floor_gate`／`floor_r_min`／`floor_r_max`／`floor_survival`）
        都不改變它。所以這四個旋鈕改的一律是「預算花在哪裡」，不是「花多少」。

        **為什麼新的兩個旋鈕也要納入同一條正規化。** 本模組的等失真對齊是
        對 `radius` 做二分搜尋，而 `radius` 只驅動 theta 與 gain，**碰不到
        加法項**（加法項的強度旗鈕是 `spectral_floor`）。若把加法項的帶壓窄
        或乘上一層 <= 1 的存活權重而不補回總量，加法項的預算就直接掉下去，
        二分搜尋會抬高 `radius` 來補失真——於是同一列同時混進了「加性變少、
        乘性變多」與「加性換了頻帶」兩件事，讀不出是哪一個在起作用。這正是
        `floor_gate` 當初要正規化的同一個理由，故沿用同一條規則。

        **代價要寫明。** `floor_gate` 的三個變體零支撐相同，比值只重分配；
        新的徑向帶會**改變零支撐**，把同樣的 L1 總量壓進更少的頻格，每一格
        的定價因此被抬高。這是刻意的：不抬高的話 `floor_r_max` 在中等值上
        就等於一個關閉開關（block=32 的格點裡 `r <= 0.4` 只佔約 12%，而
        `jpeg_luma` 在低頻又只有最大值的約 0.13），量到的會是「加法項被關掉」
        而不是「加法項換了頻帶」。同時，L1 相等不等於 L2 相等，集中之後實際
        失真會上升——那由量測協定的兩個失真軸照實回報，不在這裡預先修正。

        參考帶為空（該帶通內一格都沒有）時**拋錯而不是回傳全零**：全零的
        價目表會讓加法項悄悄失效，而 CSV 上 `spectral_floor` 那一欄照樣寫著
        非零值。
        """
        jpeg = perceptual_freq_weight("jpeg_luma", self.block, device, dtype)
        # 參考點：相位那一半的帶通 × jpeg_luma。**不含存活加權、不含逐區塊
        # 因子**，故四個旋鈕誰都動不了它。
        ref = radial_gate(self.block, self.r_min, device, dtype, self.r_max) * jpeg
        ref_mean = ref.mean()
        if float(ref_mean) <= 0.0:
            raise ValueError(
                f"r_min={self.r_min}、r_max={self.r_max} 的帶通在 "
                f"block={self.block} 的格點上是空的，加法項的總預算參考點為零")
        # 加法項自己的帶。None 時取相位那一半的值，此時下面這一行與參考點
        # 逐位元相同。
        band = radial_gate(
            self.block,
            self.r_min if self.floor_r_min is None else self.floor_r_min,
            device, dtype,
            self.r_max if self.floor_r_max is None else self.floor_r_max)
        # 存活加權："none" 時是全 1，乘上去逐位元不變（x * 1.0 == x）。
        base = band * jpeg * expected_survival_weight(
            self.floor_survival, self.block, device, dtype)
        price = (base if self.floor_gate == "uniform"
                 else FLOOR_GATES[self.floor_gate](self, x01, base))
        # 四個旋鈕全預設時 price 與 ref 逐位元相同，於是 scale 恰為 1.0，
        # 乘上去不改變任何一位——「預設關閉時逐位元等於現在」由此成立。
        scale = ref_mean / price.mean().clamp_min(1e-12)
        return (price * scale).detach()

    def floor_price(self) -> torch.Tensor:
        """加法項每個頻格值多少。三層定價與總預算正規化見 `_build_floor_price`；
        最裡面那一層是徑向帶通 × `jpeg_luma` 權重（取一次方）。

        相位與增益的閘取 `q ** 0.25`，這裡取原值：完整定價會把通帶有效容量
        壓到 0.544，對**乘法**那一半太保守，而加法項本來就要集中到人眼看不見
        的地方。
        """
        if self._floor_price is None:
            raise RuntimeError("spectral_floor == 0 時沒有價目表")
        return self._floor_price

    def envelope(self) -> torch.Tensor:
        """(1, L, n, nb) 的可學空間包絡。**`beta = 0` 時逐位元是 1。**

        構造
        ────────────────────────────────────────────────────────────
            S(l)   = 1 - prod_k [ 1 - exp( -|p_l - c_k|^2 / (2 s_k^2) ) ]
            F(w)   = exp( -(r_w / f_c)^2 )
            env    = [ (1-beta) + beta*S(l) ] * [ (1-beta) + beta*F(w) ]

        `p_l` 是視窗 l 的中心（`window_centres`，正規化影像座標），`r_w` 是
        頻格的歸一化徑向半徑（`radial_radius`，與 `radial_gate` 同座標）。

        三個構造上的選擇，各有理由
        ────────────────────────────────────────────────────────────
        - **`beta` 是一個真的參數而不是常數。** 它讓「包絡恆為 1」成為可行集
          裡的一個點，於是零初始化即恆等，PGD 自己決定要不要集中；也讓
          「逐位元退回現況」可以直接在跑得起來的設定上釘住，不必靠樁。
          `(1 - 0.0) + 0.0 * S = 1.0` 在 IEEE754 上是精確的，`x * 1.0 == x`
          也是，故 `beta = 0` 的前向與加這個旋鈕之前**逐位元**相同。
        - **K 個凸包取的是軟聯集 `1 - prod(1 - b_k)` 而不是相加。** 相加會
          在重疊處超過 1，`beta` 的內插語意就壞了。K = 1 時**特判**直接取
          `b_0`：`1 - (1 - b)` 在浮點上不等於 `b`（例如 b = 0.3 會變成
          0.30000000000000004），特判是為了讓 K = 1 與「只有一個凸包」
          這件事逐位元一致，不是為了效能。
        - **空間與頻率兩個因子各自帶自己的 `(1-beta) + beta*(.)`**，不是先
          相乘再內插。這樣 `beta` 同時是兩者的力道，且兩者都在 `beta = 0`
          時精確退回 1；先乘再內插的話 `beta` 只有一個混合係數，中間狀態是
          「一半的乘積」而不是「兩個一半的乘積」，那沒有可解釋的意義。

        四個邊界（`sigma_min`／`sigma_max`／`fc_min`／`fc_max`）全部由網格
        導出，見 `__init__`。**這裡與 `project()` 都夾一次**：可行集由後者
        維持，前向這一次是為了「沒呼叫 project 也不會超出邊界」，與
        `theta_cap` 的處理同型。
        """
        if self.floor_envelope == "none":
            raise RuntimeError("floor_envelope == 'none' 時沒有包絡")
        beta = self.env_beta.clamp(0.0, 1.0)                        # (1,)
        c = self.env_center.clamp(-1.0, 1.0)                        # (K, 2)
        s = self.env_sigma.clamp(self.sigma_min, self.sigma_max)    # (K,)
        fc = self.env_fc.clamp(self.fc_min, self.fc_max)            # (1,)

        d2 = ((self.win_xy[None] - c[:, None]) ** 2).sum(-1)        # (K, L)
        bump = torch.exp(-d2 / (2.0 * s[:, None] ** 2))             # (K, L)
        sp = bump[0] if self.floor_envelope_k == 1 else (
            1.0 - torch.prod(1.0 - bump, dim=0))                    # (L,)
        lp = torch.exp(-(self.freq_r / fc) ** 2)                    # (n, nb)

        env_s = (1.0 - beta) + beta * sp                            # (L,)
        env_f = (1.0 - beta) + beta * lp                            # (n, nb)
        return env_s[None, :, None, None] * env_f[None, None]

    def envelope_price(self, price: torch.Tensor,
                       env: torch.Tensor) -> torch.Tensor:
        """把包絡乘進加法項的價目表，**並把總預算縮放回原本的平均值**。

        沿用 `_build_floor_price` 已經立下的那條規則：加法項的四個旋鈕
        （`floor_gate`／`floor_r_min`／`floor_r_max`／`floor_survival`）都被
        縮放回同一個參考平均值，所以它們改的一律是「預算花在哪裡」不是
        「花多少」。理由在那裡寫得很完整：等失真對齊是對 `radius` 二分搜尋，
        而 `radius` 只驅動 theta 與 gain、碰不到加法項；不補回總量的話，
        加法項的預算直接掉下去，二分搜尋會抬高 `radius` 來補，同一列就同時
        混進「加性變少乘性變多」與「加性換了位置」兩件事。

        包絡與那四個旋鈕的差別是**它每一步都在動**，所以縮放必須在前向裡做
        且可微，不能像價目表那樣預先算好。

        **這正是「集中」在本構造裡的意思**：平均值固定，包絡把價目從別處
        收回來堆到凸包底下，於是那一塊的每一格定價被抬高——使用者要的
        「擾動超級大但只在一個位置」由這一條實現，不是由把 `spectral_floor`
        調大實現。L1 相等不等於 L2 相等，集中之後實際失真會上升，那由量測
        協定的兩個失真軸照實回報，不在這裡預先修正。

        參考量取 `(price * ones_like(env)).mean()` 而不是 `price.mean()`：
        `price` 在 `floor_gate = "uniform"` 時是 (n, nb)、其餘是 (1, L, n, nb)，
        兩者的元素個數差 L 倍，浮點平均的求和順序因此不同，`beta = 0` 時
        比值會落在 1 附近而不是精確的 1.0，逐位元恆等就破了。乘上
        `ones_like` 之後兩個平均是在**形狀完全相同**的張量上算的，
        `env` 全為 1 時兩者逐位元相等，比值精確為 1.0。
        """
        base = price * torch.ones_like(env)
        pe = price * env
        return pe * (base.mean() / pe.mean().clamp_min(1e-12))

    @torch.no_grad()
    def envelope_state(self) -> dict:
        """學出來的五個純量，供 CSV 逐列記下。關閉時回傳空的 dict。

        **這是結果不是旗標**：旗標（`floor_envelope`／`_k`／`_scope`）由
        驅動腳本寫，這裡寫的是 PGD 把凸包放到了哪裡、收得多窄。沒有它就
        只能從防禦圖用眼睛猜，而「包絡有沒有真的動」與「動了有沒有用」是
        兩件事，前者必須是數字。

        中心額外換算成像素座標：正規化座標讀不出「離主體多遠」，而裁切
        （繞中心放大 1.2488x）與主體位置都是以像素講的。
        """
        if self.floor_envelope == "none":
            return {}
        c = self.env_center.clamp(-1.0, 1.0)
        s = self.env_sigma.clamp(self.sigma_min, self.sigma_max)
        out = {
            "env_beta": round(float(self.env_beta.clamp(0.0, 1.0)), 6),
            "env_fc": round(float(self.env_fc.clamp(self.fc_min, self.fc_max)), 6),
        }
        for k in range(self.floor_envelope_k):
            cy, cx = float(c[k, 0]), float(c[k, 1])
            out[f"env_cy{k}"] = round(cy, 6)
            out[f"env_cx{k}"] = round(cx, 6)
            out[f"env_sigma{k}"] = round(float(s[k]), 6)
            # 正規化座標 u = 2*px/size - 1 的反解。
            out[f"env_py{k}"] = round((cy + 1.0) * self.size / 2.0, 2)
            out[f"env_px{k}"] = round((cx + 1.0) * self.size / 2.0, 2)
            out[f"env_sigma_px{k}"] = round(float(s[k]) * self.size / 2.0, 2)
        return out

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

    def expand(self, p: torch.Tensor) -> torch.Tensor:
        """把粗網格上的空間場升取樣成逐視窗的場。`coarsen == 1` 時原樣回傳。

        形狀 `(1, side_c*side_c, n, nb)` → `(1, side*side, n, nb)`。插值只作用
        在**視窗的空間索引**上；頻格那兩維原封不動搬運，因為相鄰頻格之間沒有
        「相鄰」的意義（它們是不同的基底函數，不是同一個場的取樣）。

        `align_corners=True`：角落的粗網格點精確對到角落的細網格點，插值全部
        落在四個鄰居的**凸包**內，不會外插。由此得到一條在 `project()` 上會用
        到的性質——`|expand(p)| <= max|p|`，所以夾住粗網格的參數就等於夾住了
        升取樣之後的每一個視窗，可行集不需要另外定義。這條由
        `tests/test_coarsen.py` 釘住。
        """
        if self.coarsen == 1:
            return p
        b, _, n, nb = p.shape
        grid = p.reshape(b, self.side_c, self.side_c, n * nb).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(self.side, self.side),
                             mode="bilinear", align_corners=True)
        return grid.permute(0, 2, 3, 1).reshape(b, self.n_blocks, n, nb)

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

    def requested_spectrum(self, x01: torch.Tensor) -> torch.Tensor:
        """最佳化**要求**的頻譜：相位轉過、增益乘過、下限加過，但**尚未經過
        重疊相加**。

        為什麼要把它抽出來
        ────────────────────────────────────────────────────────────
        `synthesize` 是一個投影。一組逐視窗的係數若互相矛盾——block 32、
        hop 8 表示每個像素由 **16 個視窗**加總，而 `theta` 在每個視窗上是
        自由的——就**不對應任何真實影像**，重疊相加只能給出最近的那一張。
        要量那個投影誤差、或把它當懲罰項放進損失，需要的是投影**之前**的這個
        張量，而 `_rephase` 只交出投影之後的影像。

        實測（`runs/stft_consistency/`）：主線工作點的相對幅度偏差 0.175，
        逐像素遮罩那批最高 0.480——要求的頻譜有 18–48% 交不出來，相位也丟掉
        11–17%。投影前的像素值域是 1.09–6.03，而影像的合法值域是 0–1。

        **只有一份實作**：`_rephase` 也走這一支，兩者不可能分岔。
        """
        # **先夾參數再升取樣**，與 `project()` 夾的是同一個東西。反過來做
        # （先升取樣再夾）在數值上不同，而且會讓「可行集定義在參數上」這件事
        # 失去意義。凸包性質保證夾過的粗網格升取樣後仍在界內。
        shift = self.expand(
            torch.clamp(self.theta, -self.theta_max, self.theta_max))
        if self.theta_cap is not None:
            # 與 `theta_max` 同型：前向與 `project()` 都夾一次。可行集由後者
            # 維持，故這裡在實務上是恆等，留著是為了「沒呼叫 project 也不會
            # 超出預算」。**夾的是參數本身而不是乘過閘之後的量**：閘只會把它
            # 再縮小，故預算仍然被遵守，只是在閘小的地方留有餘裕。
            shift = torch.clamp(shift, -self.theta_cap, self.theta_cap)
        # 可學的空間包絡。**`"none"` 時整段不執行**，連張量都不建，故逐位元
        # 等於加這個旋鈕之前——這與 `coarsen == 1` 不呼叫 `F.interpolate`
        # 是同一種關閉方式，測試也用同一種樁去釘。
        env = self.envelope() if self.floor_envelope != "none" else None
        # `scope == "all"` 時包絡也乘進相位／增益的閘。**這一半不做總量
        # 正規化**：乘性那一半從來就沒有那條約定（`survival_weight` 也是直接
        # 乘進 `freq_gate` 不補回總量），它的強度旋鈕是 `radius`，等失真對齊
        # 的二分搜尋搜的正是 `radius`，所以總量由那裡負責。加法項不同——
        # `radius` 碰不到它，故那一半必須自己守住總量（`envelope_price`）。
        gate = self.gate()
        if env is not None and self.floor_envelope_scope == "all":
            gate = gate * env
        shift = (shift * gate).unsqueeze(1)                          # (1,1,L,n,nb)

        spec = self.analyze(x01)
        rot = rotate_spectrum(spec, shift)
        if self.gain_max > 0:
            # (1,L,n,nb) → (1,1,L,n,nb)，與 spec 的 (B,C,L,n,nb) 廣播。
            # 閘由 `gain_weight` 決定。預設 `shared` 與相位同一個閘——那是
            # 歸因期間的約束，讓兩者的比較是乾淨的「動什麼」而不是「動哪裡」。
            # `jnd` 把增益的預算推到人眼看不見的頻帶，理由見 `__init__`。
            g = self.expand(
                torch.clamp(self.gain, -self.gain_max, self.gain_max))
            gg = self.gain_gate()
            if env is not None and self.floor_envelope_scope == "all":
                gg = gg * env
            rot = rot * torch.exp(g * gg).unsqueeze(1)
        if self.spectral_floor > 0:
            # 加在幅度上，方向沿用**已轉過的**相位。|rot| 為零時這一項也是
            # 零——自然影像的高頻雖小但非零，故實務上有效；真正的零值區
            # （合成的純色塊）動不了，這是構造上的邊界。
            a = self.expand(torch.clamp(self.floor, -1.0, 1.0))
            price = self.floor_price()
            if env is not None:
                # **兩個 scope 都走這一條**：`"all"` 的意思是「連乘性那一半
                # 也一起定位」，不是「加法項改為不正規化」。
                price = self.envelope_price(price, env)
            added = (a * price * self.spectral_floor).unsqueeze(1)
            rot = rot + added * (rot / (rot.abs() + 1e-12))
        return rot

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

        rot = self.requested_spectrum(x01)
        x_def = self.synthesize(rot)

        # Griffin-Lim 的迭代投影：反覆把幅度換回**目標**幅度，逼近「相位被
        # 轉過、幅度為目標值」那個一般不存在的訊號。gl_iters = 0 時整段不
        # 執行，行為與加這個選項之前逐位相同。
        #
        # 增益開著時目標必須是**改過的**幅度 `rot.abs()`，用原圖的幅度
        # 會把增益整個投影掉。關著時仍用**原圖**的幅度——它與 rot.abs() 在
        # 浮點上未必逐位相同，而既有批次要能逐位重跑。
        #
        # `analyze(x01)` 在這裡重算一次。`requested_spectrum` 內部算過同一個
        # 東西，但那是決定性的運算，重算逐位元相同；只有 `gl_iters > 0` 且
        # 增益關著時才會走到，故不影響主線的成本。
        target_mag = (rot.abs() if self.gain_max > 0
                      else self.analyze(x01).abs()) if self.gl_iters else None
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
