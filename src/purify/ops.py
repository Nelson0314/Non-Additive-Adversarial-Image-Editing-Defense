"""淨化算子 𝒫 — spec §5.1、§9.2。

spec §5.1 要求淨化寫進訓練目標而非事後量測，這需要淨化在訓練時可微。
但 JPEG 量化與 GrIDPure 都不可微，故每個算子分成兩個實作：

- `forward`      訓練用，可微（真實實作或其可微代理）
- `evaluate`     評測用，真實實作，不要求可微

代理與真實實作的差距必須在報告中明列，不得省略（spec §5.1 末段）。
`Purifier.proxy_gap` 提供直接量測此差距的方法，使該聲明有數字支撐而非
只是免責聲明。

本階段納入的算子與其可微性：

| 算子 | 訓練 | 評測 | 代理方式 |
|---|---|---|---|
| identity | 可微 | 同 | 無需代理（𝒫 必須含恆等算子，見 spec §5.1） |
| gaussian_blur | 可微 | 同 | 無需代理 |
| gaussian_noise | 可微 | 同 | 無需代理 |
| jpeg | 不可微 | 真實 | 直通估計（straight-through） |
| quantize | 不可微 | 真實 | 直通估計 |

GrIDPure 需要額外的擴散模型推論，成本遠高於上列各項，列為後續工作，
不在本階段的 𝒫 內。此為範圍限制，須在報告中載明。

---

## 主組（文獻共識的六個算子，`DESIGN.md` §3.3）

| 算子 | kind | 訓練 | 評測 | 出處與狀態 |
|---|---|---|---|---|
| JPEG | `jpeg` | 直通 | 真實 | DIA / DiffVax / PhotoGuard 系，q = 75、30 |
| Crop & Resize | `crop_resize` | 可微 | 同 | DIA 給「10%」，其餘我方指定（見 `crop_resize`） |
| Adverse Cleaner | `adverse_cleaner` | 直通 | 真實 | 上游 16 行原碼，需 opencv-contrib |
| CNN 去噪 | `cnn_denoise_substitute` | — | — | **非 NTIRE 2023 冠軍**，冠軍不可得，為我方替代；缺權重 |
| IMPRESS | `impress` | 直通 | 真實 | 官方 repo，PhotoGuard 情境參數；需 SDWrapper 與 LPIPS 後端 |
| DiffPure | `diffpure` | 直通 | 真實 | 官方 repo，t=150；**缺檢查點，目前拋出** |
| （對照）resize only | `resize_only` | 可微 | 同 | DiffPure 降升取樣的必要對照，見 `src/purify/diffpure.py` |

每個算子的逐項出處查證見 `docs/_audit_purify.md`，裁決見
`docs/reference/SOURCE_AUDIT.md` §9。凡標「我方指定」者不得寫成原論文設定。
"""

import io
import math
from typing import Dict, List

import torch
import torch.nn.functional as F

from src.purify import diffpure as _diffpure
from src.purify.adverse_cleaner import adverse_cleaner_real, has_guided_filter
from src.purify.diffpure import diffpure_real, resize_only
from src.purify.impress import has_impress_deps, impress_real


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs**2) / (2.0 * sigma**2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """可分離高斯模糊。sigma ≤ 0 時回傳原張量。"""
    if sigma <= 0:
        return x
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    c = x.shape[1]
    pad = (k.numel() - 1) // 2
    xh = F.conv2d(
        F.pad(x, (pad, pad, 0, 0), mode="reflect"),
        k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c,
    )
    return F.conv2d(
        F.pad(xh, (0, 0, pad, pad), mode="reflect"),
        k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c,
    )


def gaussian_noise(x: torch.Tensor, sigma: float, seed: int = None) -> torch.Tensor:
    """加性高斯噪聲。seed 固定時同一強度可重現，供淨化強度掃描使用。"""
    if sigma <= 0:
        return x
    g = None if seed is None else torch.Generator(x.device).manual_seed(seed)
    n = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    return (x + sigma * n).clamp(0, 1)


def quantize_real(x: torch.Tensor, levels: int) -> torch.Tensor:
    """真實量化，不可微（round 的導數幾乎處處為零）。"""
    q = float(levels - 1)
    return torch.round(x.clamp(0, 1) * q) / q


def straight_through(x: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """前向取 `hard`、反向視為對 `x` 的恆等映射。

    寫法必須是 `hard.detach() + (x - x.detach())` 而不是常見的
    `x + (hard - x).detach()`。兩者在實數上等價，在 float32 上不等價：
    後者要算 `x + hard - x`，兩次捨入後不保證等於 `hard`，實測 JPEG 上
    確實出現逐元素差異。前者的 `x - x.detach()` 逐元素精確為 0，故前向
    位元等同於 `hard`。
    """
    return hard.detach() + (x - x.detach())


def quantize_proxy(x: torch.Tensor, levels: int) -> torch.Tensor:
    """量化的直通估計：前向為真實量化，反向視為恆等。

    這是代理與真實實作唯一的差異來源：前向數值位元等同，只有梯度不同。
    故此代理不引入前向誤差，`proxy_gap` 對 quantize 必為 0。
    """
    return straight_through(x, quantize_real(x, levels))


def jpeg_real(x: torch.Tensor, quality: int) -> torch.Tensor:
    """真實 JPEG 編解碼。經 PIL，故不可微且必須離開計算圖。"""
    from PIL import Image
    import numpy as np

    out = []
    for i in range(x.shape[0]):
        arr = (x[i].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        dec = np.asarray(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
        out.append(torch.from_numpy(dec).permute(2, 0, 1))
    return torch.stack(out).to(x.device, x.dtype)


def jpeg_proxy(x: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG 的直通估計：前向呼叫真實編解碼，反向視為恆等。

    此代理的梯度是錯的，而非近似的：真實 JPEG 的區塊 DCT 量化在梯度
    上與恆等映射毫無關係。採用它的理由是前向數值完全正確，優化過程看到的
    是真實的 JPEG 輸出；代價是梯度方向只反映「淨化後的圖長什麼樣」而非
    「淨化本身如何反應擾動」。此限制須在報告中明列。
    """
    return straight_through(x, jpeg_real(x, quality))


# ------------------------------------------------- 真實世界變換串接（C&R）

# arXiv:2604.23688 §4：C = JPEG quality factor 75，R = 0.5× 降採樣、Lanczos
# 插值，**C&R 指先 C 再 R**（順序是協定的一部分）。該論文測到三格之下所有
# 防護方法大幅失效，且結論是「只以單獨變換評測會嚴重高估 robustness」。
CR_JPEG_QUALITY = 75
CR_RESIZE_FACTOR = 0.5
# 以下一項該論文未指定，為我方指定（與 `crop_resize` 的三項同性質）：降採樣
# 之後**升回原尺寸**。理由是本專案的評測端要把淨化後的圖餵回 512² 的 SDEdit，
# 停在 256² 會讓「淨化」與「換解析度」兩件事混在同一格裡。升取樣同樣用
# Lanczos，使降升兩端的插值核一致。
CR_UPSAMPLE_BACK = True


def jpeg_then_resize(
    x: torch.Tensor,
    quality: int = CR_JPEG_QUALITY,
    factor: float = CR_RESIZE_FACTOR,
    upsample_back: bool = CR_UPSAMPLE_BACK,
) -> torch.Tensor:
    """JPEG(q) → factor× Lanczos 降採樣 →（可選）Lanczos 升回原尺寸。

    Lanczos 走 PIL：`F.interpolate` 沒有 lanczos 核，用 bicubic 代替會是另一個
    算子而不是同一個算子的近似。本函式整條都不可微，故只能經 `straight_through`
    當代理——與 `jpeg` 相同的限制。
    """
    from PIL import Image
    import numpy as np

    h, w = x.shape[-2:]
    nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    out = []
    for i in range(x.shape[0]):
        arr = (x[i].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        im = Image.open(buf).convert("RGB").resize((nw, nh), Image.LANCZOS)
        if upsample_back:
            im = im.resize((w, h), Image.LANCZOS)
        dec = np.asarray(im).astype(np.float32) / 255.0
        out.append(torch.from_numpy(dec).permute(2, 0, 1))
    return torch.stack(out).to(x.device, x.dtype).clamp(0, 1)


# --------------------------------------------------------------- Crop & Resize

# DIA 補充材料 §B.2：`We cropped 10% of each image and then resized it to match
# the model's input requirements.` 只有「10%」有來源。
CROP_FRACTION_DIA = 0.10
# 以下三項 DIA 全文未指定，為我方指定（`reference/SOURCE_AUDIT.md` §9 第 5 項）：
# 中心裁切、每邊各裁 10%、bicubic 升回原尺寸。
CROP_MODE = "center"
CROP_INTERPOLATION = "bicubic"
CROP_ANTIALIAS = True


def crop_resize(
    x: torch.Tensor,
    fraction: float = CROP_FRACTION_DIA,
    mode: str = CROP_INTERPOLATION,
    antialias: bool = CROP_ANTIALIAS,
) -> torch.Tensor:
    """Crop & Resize。原生可微（切片 + `F.interpolate`）。

    來源切分（不得混為一談）：

    - **來自 DIA**：裁切比例 10%，裁完縮放回模型輸入尺寸。
    - **我方指定**：中心裁切（DIA 未寫中心或隨機）、10% 解讀為**每邊各裁邊長的
      10%**（故保留中央 80% 邊長，DIA 未寫是邊長還是面積）、以 bicubic 升回原尺寸
      並開 antialias（DIA 未寫插值方法）。

    bicubic 會過衝出 `[0,1]`，故最後 clamp；此為值域維護，非演算法內容。
    """
    if x.dim() != 4:
        raise ValueError(f"需要 (B,C,H,W) 張量，收到 {tuple(x.shape)}")
    h, w = x.shape[-2:]
    dh, dw = int(round(h * fraction)), int(round(w * fraction))
    if 2 * dh >= h or 2 * dw >= w:
        raise ValueError(f"裁切比例 {fraction} 對 {h}×{w} 的影像過大，裁完為空")
    cropped = x[..., dh : h - dh, dw : w - dw]
    return F.interpolate(cropped, size=(h, w), mode=mode, antialias=antialias).clamp(0, 1)


# ------------------------------------------------------------- CNN 去噪（替代）

# DiffVax 只引 NTIRE 2023 挑戰賽報告、未指名模型；冠軍 IPTV2（Team Apply AI）的
# 程式碼與權重皆未公開（`_audit_purify.md` §4.3）。依 `SOURCE_AUDIT` §9 第 3 項，
# 改用公開的強去噪器替代，暫定 Restormer。**不得稱為 NTIRE 2023 冠軍模型。**
CNN_DENOISE_SUBSTITUTE_ARCH = "Restormer (swz30/Restormer)"
CNN_DENOISE_SUBSTITUTE_CKPT = "gaussian_color_denoising_sigma50.pth"
CNN_DENOISE_SIGMA = 50  # NTIRE 2023 挑戰賽的雜訊等級（[0,255] 尺度），非盲


def has_cnn_denoise_weights(ckpt=None) -> bool:
    """替代去噪器的架構與權重是否到位。目前恆為 False。"""
    return False


def cnn_denoise_substitute_real(x: torch.Tensor, ckpt=None) -> torch.Tensor:
    """CNN 去噪（**我方替代，非 NTIRE 2023 冠軍模型**）。目前無法執行。

    NTIRE 2023 冠軍為 Team Apply AI 的 IPTV2（29.96 dB），其程式碼與權重皆未公開；
    挑戰賽官方 repo `ofsoundof/NTIRE2023_Dn50` 的 `model_zoo/` 只有主辦方 baseline
    SGN，非任何參賽方法。DiffVax 也未指名其所用的去噪器。因此本項一律標註為
    我方替代，不得聲稱重現 DiffVax 的該項評測。
    """
    raise NotImplementedError(
        "CNN 去噪（我方替代，非 NTIRE 2023 冠軍）無法執行，缺以下項目：\n"
        f"  1. 權重 {CNN_DENOISE_SUBSTITUTE_CKPT}（{CNN_DENOISE_SUBSTITUTE_ARCH} 的"
        f" σ={CNN_DENOISE_SIGMA} 彩色高斯去噪模型，官方以 Google Drive 發布，"
        "本機無此檔且環境無網路）。\n"
        "  2. 對應的架構定義（Restormer 的 `basicsr`-style arch），環境未安裝 basicsr。\n"
        "  另需主 session 裁決：替代對象確定為 Restormer 或改 NAFNet／SCUNet"
        "（SOURCE_AUDIT §9 第 3 項標為『需你確認替代對象』）。"
    )


KINDS = (
    # 既有（掃描組）
    "identity",
    "blur",
    "noise",
    "jpeg",
    "quantize",
    # 主組新增
    "crop_resize",
    "adverse_cleaner",
    "cnn_denoise_substitute",
    "impress",
    "diffpure",
    "resize_only",
    "jpeg_then_resize",
)

# 原生可微（`forward` 走真實實作並提供真實梯度）的算子。其餘一律經
# `straight_through`：前向為真實輸出、反向視為恆等。
_DIFFERENTIABLE = ("identity", "blur", "noise", "crop_resize", "resize_only")


class Purifier:
    """單一淨化設定。`forward` 供訓練、`evaluate` 供評測。

    `options` 供各算子的額外相依：

    | kind | 用到的 option |
    |---|---|
    | `crop_resize` | 無（`strength` 即裁切比例，預設 DIA 的 0.10） |
    | `adverse_cleaner` | 無（七個參數皆為上游固定值，`strength` 未使用） |
    | `impress` | `sd`（SDWrapper，必要）、`backend`、`iters`、`lr`、`alpha`、`noise`、`eps` |
    | `diffpure` | `ckpt`（`strength` 即 t，預設 150） |
    | `cnn_denoise_substitute` | `ckpt` |
    | `resize_only` | 無（降升取樣參數由 `src/purify/diffpure.py` 統一提供） |
    """

    def __init__(self, kind: str, strength: float = 0.0, seed: int = None, **options):
        self.kind = kind
        self.strength = strength
        self.seed = seed
        self.options = options
        if kind not in KINDS:
            raise ValueError(f"未知的淨化算子 {kind!r}")

    @property
    def differentiable(self) -> bool:
        """代理是否提供了真實梯度。走直通估計者（jpeg、quantize、
        adverse_cleaner、impress、diffpure）為 False。

        `cnn_denoise_substitute` 也是 False：替代對象未定案、權重未到位，
        在能實際跑之前不宣稱其梯度性質。
        """
        return self.kind in _DIFFERENTIABLE

    @property
    def available(self) -> bool:
        """相依是否齊備。False 表示呼叫 `forward`／`evaluate` 會拋出。"""
        if self.kind == "adverse_cleaner":
            return has_guided_filter()
        if self.kind == "impress":
            return has_impress_deps(self.options.get("sd"),
                                    self.options.get("backend", "lpips"))
        if self.kind == "diffpure":
            return _diffpure.has_diffpure_weights(self.options.get("ckpt"))
        if self.kind == "cnn_denoise_substitute":
            return has_cnn_denoise_weights(self.options.get("ckpt"))
        return True

    # ---- 真實實作（評測用），`forward` 與 `evaluate` 共用同一份 ----

    def _real(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "identity":
            return x
        if self.kind == "blur":
            return gaussian_blur(x, self.strength)
        if self.kind == "noise":
            return gaussian_noise(x, self.strength, self.seed)
        if self.kind == "jpeg":
            return jpeg_real(x, int(self.strength))
        if self.kind == "quantize":
            return quantize_real(x, int(self.strength))
        if self.kind == "crop_resize":
            frac = self.strength if self.strength else CROP_FRACTION_DIA
            return crop_resize(x, frac)
        if self.kind == "resize_only":
            return resize_only(x)
        if self.kind == "jpeg_then_resize":
            q = int(self.strength) if self.strength else CR_JPEG_QUALITY
            return jpeg_then_resize(x, quality=q)
        if self.kind == "adverse_cleaner":
            return adverse_cleaner_real(x)
        if self.kind == "impress":
            opts = dict(self.options)
            sd = opts.pop("sd", None)
            return impress_real(x, sd, seed=self.seed, **opts)
        if self.kind == "diffpure":
            t = int(self.strength) if self.strength else _diffpure.DIFFPURE_T_DEFAULT
            return diffpure_real(x, t=t, ckpt=self.options.get("ckpt"))
        return cnn_denoise_substitute_real(x, ckpt=self.options.get("ckpt"))

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """在 fp32 上執行算子，回傳時轉回輸入的 dtype。

        **算子一律在 fp32 上做。** jpeg、quantize、crop_resize、
        adverse_cleaner 與 CNN 去噪都要經 numpy 或 OpenCV，而那兩者沒有
        bfloat16：`x.cpu().numpy()` 直接以
        `TypeError: Got unsupported ScalarType BFloat16` 中止
        （2026-08-06 於段 0 的 N3 stage2 實測）。

        只有生成路徑會餵進半精度：warp 的輸出沿用 `x01` 的 fp32，而 apa
        走 `decode(...)`，在 bf16 下產出的就是 bf16。也就是說這條路徑
        只有 N3 會踩到，N1／N2 不會——差異來自注入位置，不是來自算子。

        轉回輸入的 dtype 是為了讓本類別對呼叫端**dtype 中性**：淨化不該
        改變管線的精度。fp32 進來時兩次轉型都是恆等，故評測路徑
        （輸入恆為 fp32）逐位元不變。
        """
        return self._real(x.float()).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.differentiable:
            return self._run(x)
        return straight_through(x, self._run(x))

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x)

    def proxy_gap(self, x: torch.Tensor) -> float:
        """代理與真實實作的前向最大絕對差，供報告引用。"""
        with torch.no_grad():
            return float((self.forward(x) - self.evaluate(x)).abs().max())

    def __repr__(self) -> str:
        return f"Purifier({self.kind}, strength={self.strength})"


def default_train_set() -> List[Purifier]:
    """訓練期的 𝒫。必須包含恆等算子（spec §5.1）。

    強度取各算子的中等值：訓練目標是耐受一般淨化，不是耐受某個極端設定。
    強度掃描留給 E3 的評測階段。
    """
    return [
        Purifier("identity"),
        Purifier("blur", 1.0),
        Purifier("jpeg", 75),
    ]


def main_set(sd=None, seed: int = 0) -> List[Purifier]:
    """主組：文獻共識的六個淨化算子（`DESIGN.md` §3.3 上半表）。

    JPEG 取 q = 75 與 30（DIA 報 70／80／90，DiffVax 與 PhotoGuard 系另有其值，
    本專案沿用既有的 75／30 兩點）。其餘五項各一個設定：

    - Crop & Resize：DIA 的 10%（其餘細節我方指定，見 `crop_resize`）
    - Adverse Cleaner：上游預設（DIA 未覆寫）
    - CNN 去噪：我方替代，非 NTIRE 2023 冠軍
    - IMPRESS：PhotoGuard 情境那組預設，需傳入 `sd`
    - DiffPure：t = 150（ImageNet）

    `resize_only` **不在主組六個之內**，它是 DiffPure 解析度處置的對照
    （`SOURCE_AUDIT` §9 第 4 項），列在 `eval_sweep` 中一併量測。
    """
    return [
        Purifier("jpeg", 75),
        Purifier("jpeg", 30),
        Purifier("crop_resize", CROP_FRACTION_DIA),
        Purifier("adverse_cleaner"),
        Purifier("cnn_denoise_substitute"),
        Purifier("impress", sd=sd, seed=seed),
        Purifier("diffpure", _diffpure.DIFFPURE_T_DEFAULT, seed=seed),
    ]


def eval_sweep(sd=None) -> Dict[str, List[Purifier]]:
    """E3 的淨化強度掃描（既有掃描組）＋ 主組的五個新算子。

    既有的四個鍵（blur／jpeg／noise／quantize）逐字未動：其中 jpeg 的 95…30
    已涵蓋主組要的 75 與 30 兩點，不另設重複的鍵。

    新增的鍵各只有一個設定（這些算子由來源固定參數，沒有強度軸可掃）。
    `diffpure` 與 `cnn_denoise_substitute` 目前缺權重、`impress` 需 `sd`、
    `adverse_cleaner` 需 opencv-contrib；呼叫端可先以 `Purifier.available`
    篩選，未篩選時會在該算子上明確拋出而不是靜默略過。
    """
    return {
        "blur": [Purifier("blur", s) for s in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)],
        "jpeg": [Purifier("jpeg", q) for q in (95, 85, 75, 60, 45, 30)],
        "noise": [Purifier("noise", s, seed=0) for s in (0.0, 0.01, 0.02, 0.04, 0.08)],
        "quantize": [Purifier("quantize", n) for n in (256, 64, 32, 16, 8)],
        "crop_resize": [Purifier("crop_resize", CROP_FRACTION_DIA)],
        "adverse_cleaner": [Purifier("adverse_cleaner")],
        "cnn_denoise_substitute": [Purifier("cnn_denoise_substitute")],
        "impress": [Purifier("impress", sd=sd, seed=0)],
        "diffpure": [Purifier("diffpure", _diffpure.DIFFPURE_T_DEFAULT, seed=0)],
        "resize_only": [Purifier("resize_only")],
    }
