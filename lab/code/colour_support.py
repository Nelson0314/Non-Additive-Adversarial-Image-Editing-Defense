"""顏色定義的支撐、彩度上界、以及「從原膚色的色調起步」的初始化。

三件事共用一個前提：**本專案的色偏只有兩種情形難看**——高飽和，以及與原膚色
的色差過大。兩者都是**全域上界**，不是遮罩；遮罩式的膚色保護已經量過，
帶窄出斑塊、帶寬出臉霧，而且臉框的色差量不到那個症狀。

所以這一檔提供的是：

| 名字 | 是什麼 | 擋什麼 |
|---|---|---|
| `skin_colour_support` | 原圖裡與膚色**同色**的像素（不限位置） | 「與原膚色色差過大」 |
| `chroma_p95` | 輸出彩度的 95 百分位 | 「高飽和」 |
| `skin_tone_slopes` | 把整圖色調拉向**該張自己的膚色色調**的曲線 | 最佳化的起點 |
"""

from __future__ import annotations

import torch

from src.defense.ncf_param import rgb_to_lab


def skin_colour_support(x01: torch.Tensor, face: torch.Tensor,
                        radius: float = 12.0) -> torch.Tensor:
    """原圖裡與膚色**同色**的像素，不限位置。回傳 (1,1,H,W) 的 0／1 遮罩。

    群心取臉框內 `(a, b)` 的**中位數**——平均會被頭髮與背景的離群值拉走。

    這一塊與臉框是兩個不同的支撐：臉框是位置定義的，這一塊是顏色定義的。
    全域映射動的是顏色，所以顏色定義的那一塊才是它真正的作用範圍：背景裡
    與膚色同色的東西會跟著臉一起被保護，而臉框上限完全看不到那些像素。
    """
    ab = rgb_to_lab(x01.clamp(0, 1).float())[:, 1:]
    sel = ab[0].permute(1, 2, 0)[face[0, 0] > 0.5]
    if sel.numel() == 0:
        raise ValueError("臉框內沒有像素，膚色群心定不出來")
    centre = sel.median(0).values.view(1, 2, 1, 1)
    dist = (ab - centre).pow(2).sum(1, keepdim=True).clamp_min(1e-12).sqrt()
    return (dist <= radius).to(ab.dtype)


def chroma_p95(x01: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """輸出彩度（Lab 的 C*）的 95 百分位。

    `sqrt` 先 `clamp_min`：`sqrt` 在 0 的導數是無限大，而純灰（`a = b = 0`）
    在人像上到處都是——白色天空整片就是。不夾的話第 0 步就拿到 NaN 梯度，
    而 `delta_e_torch` 那一份早就因為同一個理由夾過了。
    """
    lab = rgb_to_lab(x01.clamp(0, 1).float())
    c = lab[:, 1:].pow(2).sum(1).clamp_min(1e-12).sqrt().reshape(-1)
    return torch.quantile(c, q)


def _cdf(values: torch.Tensor, bins: int) -> torch.Tensor:
    hist = torch.histc(values.clamp(0, 1), bins=bins, min=0.0, max=1.0)
    total = hist.sum().clamp_min(1.0)
    return torch.cumsum(hist / total, 0)


@torch.no_grad()
def skin_tone_slopes(x01: torch.Tensor, skin: torch.Tensor, pieces: int,
                     strength: float = 1.0, bins: int = 256) -> torch.Tensor:
    """把整圖拉向**該張自己的膚色色調**的單調曲線，寫成分段斜率 `(1,3,K)`。

    為什麼起點要是這個而不是恆等
    ────────────────────────────────────────────────────────────────
    恆等起點上，最佳化只看代理分數，於是它往哪個方向離開恆等完全由梯度決定，
    而那個方向沒有理由落在「這張照片本來就有的顏色」附近——`curve_dual_spatial`
    交付的整片洋紅就是這麼來的：臉的預算守住了，但背景被推到與原圖差很遠的
    顏色去。

    這裡的起點是**用這張照片自己的膚色定出來的色調**：逐通道把整圖的累積
    分布對到膚色區的累積分布（直方圖匹配），得到的映射把整張照片的色調拉向
    它自己的膚色。起點因此是由這張照片自己的顏色構成的，最佳化從那裡往外走。

    `strength` 在恆等與該映射之間插值，1.0 為完全採用。

    **可行域會夾掉一部分。** `ColorCurveParam` 的斜率下界是 `1/K`（正規化後
    等於 `1/(1+radius)`），膚色累積分布平坦的那幾段要的斜率可能比它更小，
    夾了之後曲線與匹配映射不完全相同。夾的量寫進 CSV，不靜默。
    """
    k = int(pieces)
    dev, dt = x01.device, torch.float32
    out = torch.empty((1, 3, k), device=dev, dtype=dt)
    sel = skin[0, 0] > 0.5
    for c in range(3):
        chan = x01[0, c].to(dt)
        src = _cdf(chan.reshape(-1), bins)
        tgt = _cdf(chan[sel].reshape(-1), bins) if sel.any() else src
        # M(v) = G⁻¹(F(v))：對每個輸入分位找到目標分布上同樣分位的值。
        grid = torch.linspace(0.0, 1.0, k + 1, device=dev, dtype=dt)
        idx = (grid * (bins - 1)).round().long().clamp(0, bins - 1)
        p = src[idx]
        pos = torch.searchsorted(tgt.contiguous(), p.contiguous()).clamp(0, bins - 1)
        m = pos.to(dt) / (bins - 1)
        m = torch.cummax(m, 0).values                      # 保證單調
        m = (m - m[0]) / (m[-1] - m[0]).clamp_min(1e-6)    # 端點錨回 0 與 1
        slopes = (m[1:] - m[:-1]).clamp_min(0.0)
        flat = torch.full_like(slopes, 1.0 / k)
        out[0, c] = (1.0 - strength) * flat + strength * slopes
    return out


@torch.no_grad()
def apply_slopes(curve, slopes: torch.Tensor) -> float:
    """把斜率寫進 `ColorCurveParam` 並投影回可行域，回傳被夾掉的比例。"""
    lo, hi = curve.bounds()
    theta = slopes.to(curve.theta.device, curve.theta.dtype)
    clipped = float(((theta < lo) | (theta > hi)).to(torch.float32).mean())
    curve.theta = theta.clamp(lo, hi).clone().requires_grad_(True)
    return clipped
