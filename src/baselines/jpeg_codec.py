"""可微的 JPEG 編解碼管線 — DCT-Shield（Bala et al., ICCV 2025）的底層。

出處
────────────────────────────────────────────────────────────────────
arXiv:2504.17894 §3.3 與 §4.2。論文把 JPEG 編碼寫成 `JPEG_E`、解碼寫成
`JPEG_D`，並在式 7–9 把防禦擾動 δ 加在 **`JPEG_E` 的輸出**（量化後的整數
係數）上：

    α  = JPEG_E(x; Q_alg)                 α(c) ∈ Z^{n_p(c)×8×8}
    x' = JPEG_D(α + δ; Q_alg)

**為什麼一定要拆成兩半**：量化含四捨五入，其導數幾乎處處為零。若把 δ 加在
像素上，梯度反傳穿過四捨五入之後會被歸零，最佳化一步都動不了（論文 §4.2
明講這是設計動機）。因此本檔的 `jpeg_encode` **不需要可微**（α 只算一次、
之後當常數），`jpeg_decode` **必須完全可微**。

與 libjpeg 的對齊
────────────────────────────────────────────────────────────────────
量化表與品質縮放照 libjpeg 的 `jpeg_quality_scaling` 與 `jpeg_add_quant_table`
（Independent JPEG Group, `jcparam.c`）：

    scale = 5000/Q        (Q < 50)
    scale = 200 - 2Q      (Q >= 50)
    table = clamp((base * scale + 50) / 100, 1, 255)    ← 整數除法

`tests/test_jpeg_codec.py` 用 PIL 實際存一張 JPEG、把 `img.quantization`
讀回來逐格比對，不是照抄公式就算數（實測八個品質全部逐格相同）。

色度次取樣取 4:2:0（論文的參數量 `O(3HW/2)` 只有在 4:2:0 下才成立：
`HW + 2·(HW/4) = 3HW/2`）。上取樣用 `bilinear` 且 `align_corners=False`
——這與 libjpeg 的 `h2v2_fancy_upsample` 是同一個濾波器：輸出像素落在輸入
座標的 ±0.25 處，可分離的權重恰為 9/16、3/16、3/16、1/16。

**往返不會逐位等於 PIL**：libjpeg 用整數近似 IDCT（islow）、PIL 會把輸出
四捨五入成 uint8，而本檔走浮點。實測差在 32 dB 以上，遠優於兩者對原圖的
重建誤差，足以抓出色彩矩陣／次取樣／量化表寫錯這類系統性錯誤。

值域
────────────────────────────────────────────────────────────────────
對外介面是本專案慣用的 `[0,1]`，內部一律轉成 JPEG 的 `[0,255]` 再做 level
shift（減 128）。轉換發生在頭尾各一次。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 量化表（ITU-T T.81 Annex K，libjpeg 的 std_luminance/chrominance_quant_tbl）
# ---------------------------------------------------------------------------

LUMA_BASE = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)

CHROMA_BASE = (
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
)


def quality_to_scale(quality: int) -> int:
    """libjpeg 的 `jpeg_quality_scaling`。回傳百分比尺度的整數。"""
    q = int(quality)
    if q <= 0:
        q = 1
    if q > 100:
        q = 100
    return 5000 // q if q < 50 else 200 - q * 2


def quant_table(quality: int, chroma: bool = False, *,
                device=None, dtype=torch.float32) -> torch.Tensor:
    """(8,8) 的量化步長表。**整數除法與夾到 [1,255] 都照 libjpeg。**"""
    base = CHROMA_BASE if chroma else LUMA_BASE
    scale = quality_to_scale(quality)
    vals = [min(255, max(1, (b * scale + 50) // 100)) for b in base]
    return torch.tensor(vals, device=device, dtype=dtype).reshape(8, 8)


def normalize_quality(q_alg: float) -> int:
    """論文用 `Q_alg = 0.95` 這種小數寫法，本專案內部一律用 1–100 的整數。"""
    if q_alg <= 0:
        raise ValueError(f"品質必須為正，收到 {q_alg}")
    return int(round(q_alg * 100)) if q_alg <= 1.0 else int(round(q_alg))


# ---------------------------------------------------------------------------
# 色彩空間（JFIF full-range BT.601，即 libjpeg 的 rgb_ycc_convert）
# ---------------------------------------------------------------------------

_RGB2YCC = ((0.299, 0.587, 0.114),
            (-0.168736, -0.331264, 0.5),
            (0.5, -0.418688, -0.081312))
_YCC2RGB = ((1.0, 0.0, 1.402),
            (1.0, -0.344136, -0.714136),
            (1.0, 1.772, 0.0))


def _offset(x):
    return torch.tensor([0.0, 128.0, 128.0], device=x.device,
                        dtype=x.dtype).view(1, 3, 1, 1)


def rgb_to_ycbcr(x255: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) `[0,255]` 的 RGB → YCbCr。Cb/Cr 已加上 128 的偏移。"""
    m = torch.tensor(_RGB2YCC, device=x255.device, dtype=x255.dtype)
    return torch.einsum("ij,njhw->nihw", m, x255) + _offset(x255)


def ycbcr_to_rgb(y255: torch.Tensor) -> torch.Tensor:
    """`rgb_to_ycbcr` 的逆。**不是精確互逆**：JFIF 公布的正逆常數各自四捨五入
    到小數第六位，兩者只互逆到 1.2e-6（`[0,255]` 上約 3e-4）。改用
    `inv(forward)` 可以讓往返精確，但那樣就偏離 libjpeg 了。"""
    m = torch.tensor(_YCC2RGB, device=y255.device, dtype=y255.dtype)
    return torch.einsum("ij,njhw->nihw", m, y255 - _offset(y255))


def subsample_420(c: torch.Tensor) -> torch.Tensor:
    """2×2 盒狀平均，即 libjpeg 預設的 `h2v2_downsample`。"""
    return F.avg_pool2d(c, kernel_size=2, stride=2)


def upsample_420(c: torch.Tensor) -> torch.Tensor:
    """三角形內插，等同 libjpeg 的 `h2v2_fancy_upsample`（見模組 docstring）。"""
    return F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 8×8 區塊 DCT
# ---------------------------------------------------------------------------

def dct_matrix(device=None, dtype=torch.float32) -> torch.Tensor:
    """(8,8) 的一維 DCT-II 矩陣 `D`，使 `F = D P Dᵀ` 恰為論文式 3。

    論文式 3 的係數是 `(1/4)C(u)C(v)`，可分離成兩個 `(1/2)C(u)`。而
    `(1/2)C(0) = 1/(2√2) = √(1/8)`、`(1/2)C(u>0) = 1/2 = √(2/8)`，正是正交
    歸一 DCT-II 的係數。**所以 JPEG 的 DCT 就是正交歸一 DCT-II**，`D Dᵀ = I`。
    """
    u = torch.arange(8, device=device, dtype=dtype).view(8, 1)
    x = torch.arange(8, device=device, dtype=dtype).view(1, 8)
    d = torch.cos((2 * x + 1) * u * math.pi / 16.0)
    scale = torch.full((8, 1), math.sqrt(2.0 / 8.0), device=device, dtype=dtype)
    scale[0, 0] = math.sqrt(1.0 / 8.0)
    return d * scale


def _blockify(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    if c != 1:
        raise ValueError(f"_blockify 只吃單通道，收到 {c}")
    if h % 8 or w % 8:
        raise ValueError(f"高寬必須是 8 的倍數，收到 {h}×{w}")
    return x.view(n, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4).contiguous()


def _unblockify(b: torch.Tensor) -> torch.Tensor:
    n, hb, wb = b.shape[0], b.shape[1], b.shape[2]
    return b.permute(0, 1, 3, 2, 4).contiguous().view(n, 1, hb * 8, wb * 8)


def block_dct(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """(N,1,H,W) 的空間值 → (N,hb,wb,8,8) 的 DCT 係數。輸入須已做 level shift。"""
    return torch.einsum("uy,nijyz,vz->nijuv", d, _blockify(x), d)


def block_idct(coef: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """`block_dct` 的逆。"""
    return _unblockify(torch.einsum("uy,nijuv,vz->nijyz", d, coef, d))


# ---------------------------------------------------------------------------
# JPEG_E / JPEG_D
# ---------------------------------------------------------------------------

CHANNEL_NAMES = ("Y", "Cb", "Cr")


def jpeg_encode(x01: torch.Tensor, quality: float) -> Dict[str, torch.Tensor]:
    """論文的 `JPEG_E`：`[0,1]` 影像 → 三通道的**整數**量化係數。

    **不可微**（含 `round`），也不需要可微：α 在最佳化中是常數。
    每個張量形狀 (N, hb, wb, 8, 8)；Cb/Cr 因 4:2:0 是 Y 的一半邊長。
    """
    if x01.dim() != 4 or x01.shape[1] != 3:
        raise ValueError(f"需要 (N,3,H,W)，收到 {tuple(x01.shape)}")
    h, w = x01.shape[-2:]
    if h % 16 or w % 16:
        raise ValueError(f"4:2:0 需要高寬是 16 的倍數，收到 {h}×{w}")

    q = normalize_quality(quality)
    d = dct_matrix(x01.device, x01.dtype)
    ycc = rgb_to_ycbcr(x01 * 255.0)
    planes = (ycc[:, 0:1], subsample_420(ycc[:, 1:2]), subsample_420(ycc[:, 2:3]))

    out: Dict[str, torch.Tensor] = {}
    for name, plane in zip(CHANNEL_NAMES, planes):
        tbl = quant_table(q, chroma=(name != "Y"), device=x01.device, dtype=x01.dtype)
        out[name] = torch.round(block_dct(plane - 128.0, d) / tbl)
    return out


def jpeg_decode(coef: Dict[str, torch.Tensor], quality: float, *,
                clamp: bool = True) -> torch.Tensor:
    """論文的 `JPEG_D`：量化係數 → `[0,1]` 影像。**完全可微**。"""
    q = normalize_quality(quality)
    ref = coef["Y"]
    d = dct_matrix(ref.device, ref.dtype)
    planes = []
    for name in CHANNEL_NAMES:
        tbl = quant_table(q, chroma=(name != "Y"), device=ref.device, dtype=ref.dtype)
        planes.append(block_idct(coef[name] * tbl, d) + 128.0)
    ycc = torch.cat([planes[0], upsample_420(planes[1]), upsample_420(planes[2])], dim=1)
    x01 = ycbcr_to_rgb(ycc) / 255.0
    return x01.clamp(0.0, 1.0) if clamp else x01


def jpeg_roundtrip(x01: torch.Tensor, quality: float) -> torch.Tensor:
    """`JPEG_D(JPEG_E(x))`。**這就是 DCT-Shield 在 δ=0 時的輸出**，也就是它的
    失真地板——與紋理重相位 θ=0 時逐位等於原圖不同。實測七張平均在
    Q=0.95 上 DISTS 0.0022、LPIPS 0.0299、PSNR 42.25。"""
    return jpeg_decode(jpeg_encode(x01, quality), quality)


def coefficient_count(h: int, w: int, channels: Tuple[str, ...] = CHANNEL_NAMES) -> int:
    """指定通道集合下的係數個數。用來核對論文的 `O(3HW/2)` 與 `O(HW)`。"""
    return sum(h * w if c == "Y" else (h // 2) * (w // 2) for c in channels)
