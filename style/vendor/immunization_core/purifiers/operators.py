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
"""

import io
import math
from typing import Dict, List

import torch
import torch.nn.functional as F

from immunization_core.purifiers import diffpure as _diffpure
from immunization_core.purifiers.adverse_cleaner import adverse_cleaner_real, has_guided_filter
from immunization_core.purifiers.diffpure import diffpure_real, resize_only
from immunization_core.purifiers.impress import has_impress_deps, impress_real


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
    """前向取 `hard`、反向視為對 `x` 的恆等映射。"""
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
    """JPEG 的直通估計：前向呼叫真實編解碼，反向視為恆等。"""
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


# ---------------------------------------------------- crop_resize 的兩個分解對照


def resample_roundtrip(
    x: torch.Tensor,
    inner: int = 410,
    mode: str = CROP_INTERPOLATION,
    antialias: bool = CROP_ANTIALIAS,
) -> torch.Tensor:
    """降到 `inner` 再升回原尺寸。**視野不變、幾何不變**，只有重取樣的損失。

    `inner` 預設 410 使它與 `crop_resize(0.10)` 的取樣率相同（512 → 410），
    差別只在後者換了視野而這一支沒有。兩者相減即幾何那一份。
    """
    if x.dim() != 4:
        raise ValueError(f"需要 (B,C,H,W) 張量，收到 {tuple(x.shape)}")
    h, w = x.shape[-2:]
    if inner < 2 or inner > min(h, w):
        raise ValueError(f"inner={inner} 超出 [2, {min(h, w)}]")
    small = F.interpolate(x, size=(inner, inner), mode=mode, antialias=antialias)
    return F.interpolate(small, size=(h, w), mode=mode,
                         antialias=antialias).clamp(0, 1)


def shift_only(x: torch.Tensor, pixels: int = 51) -> torch.Tensor:
    """平移 `pixels` 像素，**不重取樣、不縮放**，邊界以反射填補。

    中心裁切本身**不含平移**（中心是不動點），所以這一支不是 `crop_resize`
    的分解項；它回答的是另一個問題：攻擊方若不置中裁切，我們掉多少。
    """
    if x.dim() != 4:
        raise ValueError(f"需要 (B,C,H,W) 張量，收到 {tuple(x.shape)}")
    if pixels == 0:
        return x
    h, w = x.shape[-2:]
    if abs(pixels) >= min(h, w):
        raise ValueError(f"平移 {pixels} 超過影像尺寸 {h}×{w}")
    k = abs(int(pixels))
    padded = F.pad(x, (k, k, k, k), mode="reflect")
    top = k + int(pixels)
    left = k + int(pixels)
    return padded[..., top:top + h, left:left + w]


ROTATE_DEGREES_FACELOCK = 10.0


#: 角度小於此值時，512 px 影像的最遠角落位移不到 5 px，各處都在次像素量級，
#: 旋轉與恆等映射在讀數上分不開。見 `rotate_angle` 的說明。
ROTATE_DEGENERATE_DEGREES = 1.0


ROTATE_FIXED = True


def rotate_angle(degrees: float = ROTATE_DEGREES_FACELOCK,
                 seed: int = None) -> float:
    """實際會套用的角度。**不轉圖，只回報角度。**"""
    if ROTATE_FIXED:
        return float(degrees)
    g = torch.Generator(device="cpu")
    g.manual_seed(0 if seed is None else int(seed))
    return float((torch.rand(1, generator=g).item() * 2.0 - 1.0) * degrees)


def rotate_random(x: torch.Tensor, degrees: float = ROTATE_DEGREES_FACELOCK,
                  seed: int = None) -> torch.Tensor:
    """繞影像中心隨機旋轉 `U(−degrees, +degrees)`，雙線性重取樣、邊界補零。

    **兩個設定是本專案指定的，論文沒有寫**：FaceLock 只給角度區間，未載插值
    核與邊界填補方式。這裡取雙線性（與 `crop_resize`、`resize_only` 一致，
    使幾何類算子彼此可比）與補零（旋轉後四角必然離開原畫面，補零是
    `torchvision.RandomRotation` 的預設行為）。移植報表上必須標
    `modified_from_paper`。
    """
    if x.dim() != 4:
        raise ValueError(f"需要 (B,C,H,W) 張量，收到 {tuple(x.shape)}")
    angle = rotate_angle(degrees, seed)
    if angle == 0.0:
        return x
    rad = math.radians(angle)
    cos, sin = math.cos(rad), math.sin(rad)
    # `affine_grid` 吃的是「輸出座標 → 輸入座標」的反向映射，故用 −angle 的
    # 旋轉矩陣；正負號寫錯不會拋錯，只會轉到另一邊。
    theta = torch.tensor([[cos, sin, 0.0], [-sin, cos, 0.0]],
                         device=x.device, dtype=x.dtype)[None].expand(x.shape[0], -1, -1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)

# ------------------------------------------------------------- CNN 去噪（替代）

CNN_DENOISE_SUBSTITUTE_ARCH = "Restormer (swz30/Restormer)"
CNN_DENOISE_SUBSTITUTE_CKPT = "gaussian_color_denoising_sigma50.pth"
CNN_DENOISE_SIGMA = 50  # NTIRE 2023 挑戰賽的雜訊等級（[0,255] 尺度），非盲


# ── 色彩類淨化算子 ────────────────────────────────────────────────────
#
# 為什麼要有這一族：色彩重映射的防禦（`src/defense/color_param.py`）繞開的是
# 空間性的失效機制，它的代價是**多開了一個攻擊面**。AdvCF（arXiv:2011.06690）
# 圖 10 量到色彩攻擊在 JPEG q30／中值濾波／resize&pad 上存活 75–82%，
# 但**灰階轉換只剩約 18%**——那是它唯一的死穴，不測它主張就不成立。
#
# 四個算子涵蓋攻擊方在**沒有乾淨參照**時能做的全部色彩正規化：
# 丟掉色度（grayscale）、對齊白平衡（gray_world）、對齊逐通道的動態範圍
# （auto_levels）、對齊局部對比（clahe）。有參照的手段（直方圖匹配到原圖）
# 不在威脅模型內——攻擊方拿到的就只有防禦圖。
#
# **這四個都不是幾何類**（不改取景也不改像素格點），故參照照舊，
# 不進 `GEOMETRIC_KINDS`。


def grayscale_real(x: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 亮度，複製回三通道。`strength` 未使用。

    係數與 `src/defense/color_param.LUMA_WEIGHTS`、`src/metrics/acutance._luma`
    同一組——三處若不一致，「防禦把能量放在哪個亮度上」與「淨化拿走哪個
    亮度」講的就不是同一件事。
    """
    w = torch.tensor((0.299, 0.587, 0.114), device=x.device, dtype=x.dtype)
    y = (x * w.view(1, 3, 1, 1)).sum(1, keepdim=True)
    return y.expand(-1, 3, -1, -1).contiguous().clamp(0.0, 1.0)


def gray_world_real(x: torch.Tensor) -> torch.Tensor:
    """灰世界白平衡：逐通道增益，使三個通道的均值相等。

    `gain_c = mean(x) / mean(x_c)`。分母夾在 `1e-6`：全黑通道的增益無定義，
    這是數值邊界不是症狀掩蓋——原始碼裡不夾就是 inf 傳到整張圖。
    """
    m = x.mean(dim=(0, 2, 3), keepdim=True)
    gain = m.mean() / m.clamp_min(1e-6)
    return (x * gain).clamp(0.0, 1.0)


def auto_levels_real(x: torch.Tensor, frac: float = 0.01) -> torch.Tensor:
    """逐通道百分位拉伸：把 `frac` 與 `1 - frac` 分位映到 0 與 1。

    `frac` 預設 0.01，**本專案指定**（無出處）；它是 `Purifier.strength`，
    故逐列進 CSV。分母夾在 `1e-6`，理由同 `gray_world_real`。
    """
    n, c = x.shape[0], x.shape[1]
    flat = x.reshape(n, c, -1)
    q = torch.tensor([frac, 1.0 - frac], device=x.device, dtype=x.dtype)
    lo, hi = torch.quantile(flat, q, dim=-1)          # 各 (N,C)
    lo = lo.view(n, c, 1, 1)
    hi = hi.view(n, c, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def clahe_real(x: torch.Tensor, clip_limit: float = 2.0) -> torch.Tensor:
    """CLAHE，作用在 Lab 的 L 通道上，8×8 tile。

    `clip_limit` 是 `Purifier.strength`（預設 2.0，OpenCV 的預設值）。
    走 OpenCV 故不可微，`Purifier.forward` 以直通估計接梯度。
    """
    import cv2
    import numpy as np

    out = []
    for i in range(x.shape[0]):
        arr = (x[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0)
        arr = arr.round().astype(np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        cl = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        out.append(torch.from_numpy(rgb).permute(2, 0, 1))
    return torch.stack(out).to(device=x.device, dtype=x.dtype)


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
    "gridpure",
    "fdpure",
    # crop_resize 的分解對照。**不是文獻裡的淨化算子**，只用於歸因，
    # 不得進入頭對頭的淨化器清單（`phase_retention.purifier_set` 沒有它們）。
    "resample_roundtrip",
    "shift_only",
    # FaceLock 與 EditShield 的旋轉欄。幾何類，見 `rotate_random`。
    "rotate",
    # 色彩類。針對色彩重映射防禦而加，見上方 `grayscale_real` 前的說明。
    # 都不是幾何類，參照照舊。
    "grayscale",
    "gray_world",
    "auto_levels",
    "clahe",
)

GEOMETRIC_KINDS = frozenset({
    "crop_resize",
    "resample_roundtrip",
    "resize_only",
    "shift_only",
    "jpeg_then_resize",
    #   rotate             繞中心旋轉。像素格點被重取樣（雙線性），且四角離開
    #                      原畫面、邊界補零，**取景改變**。理由與 shift_only
    #                      同型，故同屬幾何類。
    "rotate",
})
assert GEOMETRIC_KINDS <= set(KINDS)


def kind_of_label(name: str) -> str:
    """由 `phase_retention.label()` 的輸出還原 `Purifier.kind`。

    `label()` 是 `kind` 或 `f"{kind}{strength:g}"`，所以尾段若存在必定整段
    是一個數字。**這不是字首猜測**：候選一律取自 `KINDS`，而且尾段必須
    `float()` 得過。故 `jpeg_then_resize75` 不會被 `jpeg` 吃掉——它的尾段
    `_then_resize75` 不是數字；長的候選先試，`jpeg_then_resize` 才是答案。
    """
    for kind in sorted(KINDS, key=len, reverse=True):
        if name == kind:
            return kind
        if name.startswith(kind):
            rest = name[len(kind):]
            try:
                float(rest)
            except ValueError:
                continue
            return kind
    raise ValueError(f"無法由標籤 {name!r} 還原淨化算子；已知的 kind：{sorted(KINDS)}")


def label_is_geometric(name: str) -> bool:
    """標籤是否屬於幾何類。出表的程式手上只有 CSV 的 `purifier` 欄，

    沒有 `Purifier` 物件，故由標籤還原 `kind` 之後再判定。
    """
    return kind_of_label(name) in GEOMETRIC_KINDS


# 原生可微（`forward` 走真實實作並提供真實梯度）的算子。其餘一律經
# `straight_through`：前向為真實輸出、反向視為恆等。
_DIFFERENTIABLE = ("identity", "blur", "noise", "crop_resize", "resize_only",
                   "resample_roundtrip", "shift_only", "rotate",
                   # 三者都是逐點或逐通道的可微運算，原生有梯度。
                   # `clahe` 走 OpenCV，不在此列，由直通估計接。
                   "grayscale", "gray_world", "auto_levels")


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
    | `gridpure` | `t`、`gamma`、`iters` **皆必填**（論文正文未載）、`ckpt` |
    | `fdpure` | `t_star` **必填**（論文正文未載）、`d_a`、`d_p`、`delta`、`ckpt` |
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
        if self.kind in ("gridpure", "fdpure"):
            return False
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
        if self.kind == "resample_roundtrip":
            return resample_roundtrip(
                x, int(self.strength) if self.strength else 410)
        if self.kind == "shift_only":
            return shift_only(x, int(self.strength) if self.strength else 51)
        if self.kind == "rotate":
            deg = self.strength if self.strength else ROTATE_DEGREES_FACELOCK
            return rotate_random(x, deg, seed=self.seed)
        if self.kind == "jpeg_then_resize":
            q = int(self.strength) if self.strength else CR_JPEG_QUALITY
            return jpeg_then_resize(x, quality=q)
        if self.kind == "grayscale":
            return grayscale_real(x)
        if self.kind == "gray_world":
            return gray_world_real(x)
        if self.kind == "auto_levels":
            frac = self.strength if self.strength else 0.01
            return auto_levels_real(x, frac)
        if self.kind == "clahe":
            cl = self.strength if self.strength else 2.0
            return clahe_real(x, cl)
        if self.kind == "adverse_cleaner":
            return adverse_cleaner_real(x)
        if self.kind == "impress":
            opts = dict(self.options)
            sd = opts.pop("sd", None)
            return impress_real(x, sd, seed=self.seed, **opts)
        if self.kind == "diffpure":
            t = int(self.strength) if self.strength else _diffpure.DIFFPURE_T_DEFAULT
            return diffpure_real(x, t=t, ckpt=self.options.get("ckpt"))
        if self.kind == "gridpure":
            raise NotImplementedError("gridpure 未包含於 core；必須提供獨立、經驗證的實作，禁止改用近似算子")
        if self.kind == "fdpure":
            raise NotImplementedError("fdpure 未包含於 core；必須提供獨立、經驗證的實作，禁止改用近似算子")
        return cnn_denoise_substitute_real(x, ckpt=self.options.get("ckpt"))

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """在 fp32 上執行算子，回傳時轉回輸入的 dtype。

        只有生成路徑會餵進半精度：warp 的輸出沿用 `x01` 的 fp32，而走
        `decode(...)` 的那一條在 bf16 下產出的就是 bf16。也就是說這條路徑
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
