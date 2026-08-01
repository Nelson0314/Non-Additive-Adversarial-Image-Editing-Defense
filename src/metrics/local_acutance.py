"""局部銳利度偏差 —— 由 P1/P1b 的實測直接指出的指標。

為什麼既有的兩個選項都不夠。P1b 在等 LPIPS 下量了四種失真（模糊、雜訊、
雙線性變形、雙三次變形），結果把候選指標分成兩群：

- 位移主導（GMSD、NLPD、VIF、HaarPSI、SSIM、DISTS、PSNR）：對變形收的費
  比對模糊高，且對更銳利的雙三次變形收得比雙線性更高。它們量的是幾何
  位移，不是高頻流失。拿它們當保真約束，會因為「site S 是一個變形」而懲罰
  它——那是循環論證，site S 本來就被允許是變形。
- 鈍化追蹤（`acutance`、stLPIPS、MUSIQ）：計價順序與實測銳利度一致。

於是需要的性質有兩條，而既有的兩個選項各缺一條：

| | 對位移不敏感 | 局部、不可抵銷 |
|---|---|---|
| `acutance`（全域能量比） | ✅ | ❌ 一處模糊、他處加噪可湊回 1 |
| GMSD（逐點梯度相似度） | ❌ | ✅ |
| 本模組 | ✅ | ✅ |

定義。把影像切成不重疊的 `patch`×`patch` 區塊，逐區塊計算梯度能量比
`r_p = E_p(rec) / E_p(orig)`，再以原圖的區塊能量為權重取加權平均偏差：

    local_acutance_dev = Σ_p w_p · |1 − r_p|,    w_p = E_p(orig) / Σ_q E_q(orig)

0 為完美、越大越糟，與其他失真量同向。

為何這樣就同時具備兩條性質。

- *對位移不敏感*：次像素位移把梯度在區塊內搬動，不會把能量搬出區塊
  （只要區塊邊長遠大於位移量），故 r_p ≈ 1。
- *不可抵銷*：取的是逐區塊偏差的絕對值再加權平均。一處 r_p < 1、他處
  r_p > 1 兩者都貢獻正值，無法互相抵銷，這正是全域能量比的漏洞所在。

權重取原圖能量而非等權，是因為近乎平坦的區塊其 r_p 由極小的分母決定，
數值不穩定。加權而非排除這些區塊：排除需要一個門檻，而門檻會變成另一個
可鑽的旋鈕；加權讓它們自然地趨近於零貢獻，不引入新參數。

一併回傳有號版本 `local_acutance_signed = Σ_p w_p (r_p − 1)`，用以區分「鈍化」
（負）與「過銳」（正）。約束應該用無號的 `dev`，報告則兩者都列。
"""

from typing import Dict

import torch
import torch.nn.functional as F

from src.metrics.acutance import _KX, _KY, _luma

# 區塊邊長。32 px 遠大於本專案關心的位移量級（site S 實測平均 0.39 px、
# max_disp 上界 1.5 px），故位移不會把梯度能量搬出區塊。
PATCH = 32


def _grad_sq(x: torch.Tensor) -> torch.Tensor:
    """(N,1,H,W) 的逐像素梯度平方和。Sobel 與 `acutance` 同一組核。"""
    y = _luma(x)
    kx = _KX.to(y.device, y.dtype).view(1, 1, 3, 3)
    ky = _KY.to(y.device, y.dtype).view(1, 1, 3, 3)
    y = F.pad(y, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(y, kx).pow(2) + F.conv2d(y, ky).pow(2)


def patch_energy(x: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    """(N, H/patch, W/patch) 的逐區塊梯度能量。

    邊長不整除時裁掉右下的殘餘，而非補零：補零會造出一個能量偏低的假區塊，
    它的 r_p 不對應任何真實區域。本專案的影像一律 512²，PATCH=32 整除。
    """
    g = _grad_sq(x)
    h = (g.shape[-2] // patch) * patch
    w = (g.shape[-1] // patch) * patch
    g = g[..., :h, :w]
    return F.avg_pool2d(g, patch).squeeze(1) * (patch * patch)


def _weights_and_ratio(orig: torch.Tensor, rec: torch.Tensor, patch: int):
    """回傳 (權重, 逐區塊能量比)。兩者都保留計算圖。

    `orig` 對 φ 為常數，故權重也是常數；梯度只經由 `rec` 的區塊能量流回。
    """
    eo = patch_energy(orig, patch)
    er = patch_energy(rec, patch)
    total = eo.flatten(1).sum(1)
    w = eo / total.view(-1, 1, 1)
    # 能量為零的區塊其權重亦為零，故此處的除法只影響權重為零的項；
    # clamp_min 只是避免產生 inf 後與 0 相乘得到 NaN。
    r = er / eo.clamp_min(1e-12)
    return w, r, total


def local_acutance_dev(
    orig: torch.Tensor, rec: torch.Tensor, patch: int = PATCH
) -> torch.Tensor:
    """可微的局部銳利度偏差，供 `src/defense/objective.py` 當約束使用。

    回傳整批的平均純量張量。`local_acutance` 是同一個量的報告版本（不建圖、
    回傳 float 並附兩個診斷欄位）；訓練與評測共用同一個定義，兩者不得分歧。

    原圖全平坦時分母為零。此處不回傳 NaN——那會讓損失變成 NaN 並靜默毀掉
    整次訓練——而是直接拋出：一張沒有任何梯度的原圖不是合法輸入。
    """
    w, r, total = _weights_and_ratio(orig, rec, patch)
    if bool((total <= 0).any()):
        raise ValueError(
            "原圖的梯度能量為零，局部銳利度偏差無定義。"
            "全平坦的影像不是合法輸入，此處不以 NaN 帶過"
        )
    return (w * (r - 1.0).abs()).flatten(1).sum(1).mean()


@torch.no_grad()
def local_acutance(
    orig: torch.Tensor, rec: torch.Tensor, patch: int = PATCH
) -> Dict[str, float]:
    """rec 相對 orig 的局部銳利度偏差（報告版）。

    orig 全平坦時回傳 NaN 而非拋出：報告端掃過整批影像，讓一張退化影像
    中斷整份報告是把限制升級成故障（同 `suite.py` 的 NIQE）。訓練端的
    `local_acutance_dev` 則相反，必須拋出。
    """
    w, r, total = _weights_and_ratio(orig, rec, patch)
    if bool((total <= 0).any()):
        nan = float("nan")
        return {"local_acutance_dev": nan, "local_acutance_signed": nan,
                "local_acutance_worst": nan}

    dev = (w * (r - 1.0).abs()).flatten(1).sum(1)
    signed = (w * (r - 1.0)).flatten(1).sum(1)
    # 最差區塊只取權重前 10% 的區塊，避免由不穩定的平坦區塊決定
    k = max(1, int(0.1 * w.shape[-1] * w.shape[-2]))
    idx = w.flatten(1).topk(k, dim=1).indices
    worst = (r - 1.0).abs().flatten(1).gather(1, idx).max(dim=1).values

    return {
        "local_acutance_dev": float(dev.mean()),
        "local_acutance_signed": float(signed.mean()),
        "local_acutance_worst": float(worst.mean()),
    }
