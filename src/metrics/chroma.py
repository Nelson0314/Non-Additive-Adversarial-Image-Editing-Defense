"""色度失真的候選度量 —— 由 E27 的否定結果直接指出的一軸。

為什麼需要這個模組。E27 在同一個 LPIPS 下量到 site C（色度矩陣場）的
防禦圖有人眼可見的色調偏移，而 site P（加性）幾乎看不出來；使用者判讀
`runs/e27d_C_lr0.3/compare.html` 的回報是「P 的那兩張防禦圖人眼看起來跟原圖
幾乎一樣，其他則有色調偏移一點點」。

現行的約束集擋不住它，而且是構造上擋不住：

- `local_acutance_dev` 只看 Rec.601 亮度（見該檔 `_grad_sq` → `_luma`），
  純色度變化不改變亮度梯度，故它恆為 0。這正是 site C 當初被設計出來的
  理由，也正是它現在的漏洞。
- LPIPS 對色度偏移低估——實測 car_00 在 LPIPS 0.0409 下 PSNR 只有 24.52 dB、
  L∞ 達 0.995，而同一 LPIPS 的 site P 是 34.01 dB / 0.225。

這是同一個失效的第三次（E15 的 site S 換到的是模糊、E27 的 site C 換到的是色調偏移），
故本模組的候選項一律要通過 E20 的等 LPIPS 探針判別，不得憑聲譽選用——
E20 §3.3(2) 的 NLPD 與 VIF 就是被文獻聲譽誤導的前例。

---

收錄的候選項

| 名稱 | 定義 | 收錄理由 |
|---|---|---|
| `de76` | CIELAB 中的歐氏距離 | 最簡單且無歧義，作為基準。已知弱點在高飽和的藍綠區 |
| `de00` | CIEDE2000 | 色差的現行標準，含亮度／彩度／色相的分項加權與藍區旋轉項 |
| `dchroma` | 只取 (a*, b*) 的位移量，丟掉 ΔL* | 直接對應「色調偏移」，不受亮度變化干擾 |
| `local_dchroma_dev` | 逐區塊 `dchroma` 的平均 | 已證明無效，保留為陰性對照：先取量值再池化等於沒有池化 |
| `local_chroma_bias` | 逐區塊有號色度位移的平均，再取量值 | 唯一能區分「連貫色偏」與「隨機色度雜訊」的構造 |

全部可微。P9 的實測結論：前四項都不合格，只有 `local_chroma_bias` 通過。

ΔE 那一類逐像素取量值，量到的是「色度誤差有多大」。P9 在等 LPIPS 下量到加性
高斯雜訊在該量上與明顯的色偏幾乎一樣高（`dchroma` 2.44 vs 2.79、`de76`
2.59 vs 2.80），因為雜訊也在每個像素上擾動 (a*, b*)。但人眼對兩者的反應差很多
——使用者判讀 E27 的比對頁時看得出 site C 的色調偏移、看不出 site P 的加性擾動。

人眼在意的是空間上連貫的色偏，不是色度誤差的量值。這與 E20 的型態相同：
當時 LPIPS 對模糊不收費，也是因為它量的軸與人眼在意的軸不是同一條。

CIEDE2000 已對 `scikit-image` 交叉驗證（`tests/test_chroma.py::
test_de00與skimage一致`）：CIELAB 最大絕對差 0.0048、ΔE00 最大絕對差 0.0071
（相對 0.016%），差距屬 float32 對 float64 的精度層級。故 `de00` 的絕對尺度
可信。

即便如此，本專案的 τ 仍一律由探針的實測值夾出（同 τ_acut 的 0.04 由四個實測
值決定），不引用「JND ≈ 1–2 ΔE 單位」這類文獻常數——E20 §3.3(2) 的 NLPD 與
VIF 就是憑文獻聲譽選指標而被實測推翻的前例。
"""

from typing import Dict

import torch
import torch.nn.functional as F

# sRGB → 線性 RGB → XYZ（D65）。列向量乘法的慣例與 site_color.py 一致。
_RGB2XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
# D65 白點（2° 觀察者）
_WHITE = torch.tensor([0.95047, 1.00000, 1.08883])

PATCH = 32     # 與 local_acutance 同一區塊大小，兩者的空間尺度才可對讀


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB 的傳輸函數反轉。分段點 0.04045 依標準。

    低段用線性式而非直接 `x ** 2.4`：後者在 0 附近的導數為 0，梯度會消失，
    而本模組的輸出要進損失。
    """
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(0) ** 2.4)


def rgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) 的 sRGB [0,1] → CIELAB。回傳 (N,3,H,W)，順序為 (L*, a*, b*)。"""
    lin = _srgb_to_linear(x.clamp(0, 1))
    m = _RGB2XYZ.to(x.device, x.dtype)
    xyz = torch.einsum("ij,njhw->nihw", m, lin)
    xyz = xyz / _WHITE.to(x.device, x.dtype).view(1, 3, 1, 1)

    # f(t) 的分段點 (6/29)^3。低段用線性式的理由同上：t^(1/3) 在 0 的導數發散
    d = 6.0 / 29.0
    t = xyz
    f = torch.where(t > d ** 3, t.clamp_min(1e-12) ** (1.0 / 3.0),
                    t / (3 * d * d) + 4.0 / 29.0)
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    return torch.cat([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=1)


def _lab_pair(a: torch.Tensor, b: torch.Tensor):
    return rgb_to_lab(a), rgb_to_lab(b)


def de76_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N,1,H,W) 的逐像素 CIE76 色差。"""
    la, lb = _lab_pair(a, b)
    return (la - lb).pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()


def dchroma_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N,1,H,W) 的逐像素色度位移量，即只取 (a*, b*) 而丟掉 ΔL*。

    丟掉亮度是刻意的：本模組要擋的是色調偏移，而亮度那一軸已由 LPIPS 與
    `local_acutance_dev` 各自把關。把 ΔL* 算進來會讓這道約束與既有兩道重疊，
    形成 E20 §5.3 批評過的「循環論證」的另一種版本——重複懲罰同一件事。
    """
    la, lb = _lab_pair(a, b)
    return (la[:, 1:] - lb[:, 1:]).pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()


def de2000_map(a: torch.Tensor, b: torch.Tensor,
               kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> torch.Tensor:
    """(N,1,H,W) 的逐像素 CIEDE2000 色差。

    依 CIE 的定義實作。彩度為零處 `atan2(0, 0)` 的色相無定義，此處回傳 0
    並由 `ΔH'` 前的 `sqrt(C1'·C2')` 因子把該項歸零——那正是定義所要求的
    行為（無彩色沒有色相差），不是為了迴避 NaN 而加的補丁。
    """
    la, lb = _lab_pair(a, b)
    L1, a1, b1 = la[:, 0:1], la[:, 1:2], la[:, 2:3]
    L2, a2, b2 = lb[:, 0:1], lb[:, 1:2], lb[:, 2:3]

    eps = 1e-12
    C1 = (a1 * a1 + b1 * b1).clamp_min(eps).sqrt()
    C2 = (a2 * a2 + b2 * b2).clamp_min(eps).sqrt()
    Cbar = (C1 + C2) / 2
    c7 = Cbar.pow(7)
    G = 0.5 * (1 - (c7 / (c7 + 25.0 ** 7)).clamp_min(0).sqrt())
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = (a1p * a1p + b1 * b1).clamp_min(eps).sqrt()
    C2p = (a2p * a2p + b2 * b2).clamp_min(eps).sqrt()

    deg = 180.0 / torch.pi
    h1p = torch.atan2(b1, a1p) * deg % 360.0
    h2p = torch.atan2(b2, a2p) * deg % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dh = h2p - h1p
    dhp = torch.where(dh > 180.0, dh - 360.0, torch.where(dh < -180.0, dh + 360.0, dh))
    dHp = 2 * (C1p * C2p).clamp_min(eps).sqrt() * torch.sin(dhp / 2 / deg)

    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum, hdiff = h1p + h2p, (h1p - h2p).abs()
    hbp = torch.where(
        hdiff <= 180.0, hsum / 2,
        torch.where(hsum < 360.0, (hsum + 360.0) / 2, (hsum - 360.0) / 2))

    T = (1
         - 0.17 * torch.cos((hbp - 30.0) / deg)
         + 0.24 * torch.cos((2 * hbp) / deg)
         + 0.32 * torch.cos((3 * hbp + 6.0) / deg)
         - 0.20 * torch.cos((4 * hbp - 63.0) / deg))
    dtheta = 30.0 * torch.exp(-(((hbp - 275.0) / 25.0) ** 2))
    cbp7 = Cbp.pow(7)
    RC = 2 * (cbp7 / (cbp7 + 25.0 ** 7)).clamp_min(0).sqrt()
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / (20 + (Lbp - 50) ** 2).clamp_min(eps).sqrt()
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -torch.sin((2 * dtheta) / deg) * RC

    tL, tC, tH = dLp / (kL * SL), dCp / (kC * SC), dHp / (kH * SH)
    return (tL * tL + tC * tC + tH * tH + RT * tC * tH).clamp_min(0).sqrt()


def local_dchroma_dev(orig: torch.Tensor, rec: torch.Tensor,
                      patch: int = PATCH) -> torch.Tensor:
    """已證明無效，保留為陰性對照，不要拿來當約束。

    原意是「先在區塊內取平均、再對區塊取平均」以防止互相抵銷。但 `dchroma_map`
    已經逐像素取過量值，之後再做區塊平均與全域平均，兩者合起來就等於
    全域平均——區塊結構完全沒有作用。P9 的實測直接顯示這一點：本項與
    `dchroma` 的數值在每一個條件上都完全相同（見 `runs/p9_chroma_probe/`）。

    取量值與取平均的順序寫反。要區分「連貫的色偏」與「隨機的色度雜訊」，必須先在區塊內對
    有號的 (Δa*, Δb*) 取平均、再取量值——那才是 `local_chroma_bias`。

    保留而不刪除，是因為它與 `dchroma` 的數值相同這件事本身就是「先取量值
    再池化等於沒有池化」的證據，刪除會使該推導需要重做。
    """
    d = dchroma_map(orig, rec)
    h = (d.shape[-2] // patch) * patch
    w = (d.shape[-1] // patch) * patch
    blocks = F.avg_pool2d(d[..., :h, :w], patch)
    return blocks.flatten(1).mean(1).mean()


def local_chroma_bias(orig: torch.Tensor, rec: torch.Tensor,
                      patch: int = PATCH) -> torch.Tensor:
    """可微的逐區塊色度偏壓，供 `objective.py` 當約束使用。

        bias = mean_p ‖ mean_{i∈p} (Δa*_i, Δb*_i) ‖

    與 `dchroma` 的差別只在取量值與取平均的順序，此差異即為關鍵。
    ΔE 那一類逐像素取量值，量到的是「色度誤差有多大」；P9 實測顯示加性高斯
    雜訊在這個量上與明顯的色偏一樣高（等 LPIPS 下 2.44 vs 2.79），因為雜訊
    也在每個像素上擾動 (a*, b*)。但人眼對兩者的反應差很多——使用者判讀
    E27 的比對頁時，看得出 site C 的色調偏移、看不出 site P 的加性擾動。

    人眼在意的是空間上連貫的色偏。先在區塊內對有號的色度位移取平均，
    隨機雜訊會相消而連貫的偏壓不會；之後才取量值，使「一處偏紅、他處偏綠」
    在區塊之間仍無法互相抵銷。

    區塊邊長取 32 與 `local_acutance_dev` 一致，兩者的空間尺度才可對讀。

    不以原圖能量加權（與 `local_acutance_dev` 不同）：銳利度比值在平坦區
    由極小的分母決定故需要加權；色度偏壓是絕對量，平坦區的偏色一樣看得見
    ——大片天空偏色其實更明顯——加權反而會把它壓掉。
    """
    la, lb = _lab_pair(orig, rec)
    d = lb[:, 1:] - la[:, 1:]                       # (N,2,H,W) 有號色度位移
    h = (d.shape[-2] // patch) * patch
    w = (d.shape[-1] // patch) * patch
    blocks = F.avg_pool2d(d[..., :h, :w], patch)    # 先在區塊內取有號平均
    mag = blocks.pow(2).sum(dim=1).clamp_min(1e-12).sqrt()   # 再取量值
    return mag.flatten(1).mean(1).mean()


@torch.no_grad()
def chroma_battery(orig: torch.Tensor, rec: torch.Tensor,
                   patch: int = PATCH) -> Dict[str, float]:
    """報告版：四個候選項一次算完。訓練與評測共用同一份定義。"""
    return {
        "de76": float(de76_map(orig, rec).mean()),
        "de00": float(de2000_map(orig, rec).mean()),
        "dchroma": float(dchroma_map(orig, rec).mean()),
        "local_dchroma_dev": float(local_dchroma_dev(orig, rec, patch)),
        "local_chroma_bias": float(local_chroma_bias(orig, rec, patch)),
    }
