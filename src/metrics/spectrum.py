"""殘差奇異值譜診斷 — spec §8.2，圖表形式取自 LRDM Fig.3。

三項用途：
1. 驗證 site P 的秩確實等於設定值（此處秩有理論保證，量測是為了抓實作錯誤）；
2. **量測 site L 湧現的像素空間秩**。site L 的秩約束作用在注入的 latent 殘差，
   像素殘差還要經過 DDIM 更新與 VAE 解碼兩道非線性變換，其秩沒有理論保證，
   只能實測（spec §4.3）；
3. 作為論文中「本方法的殘差確實／並不具低秩結構」的直接證據。

**秩的零判準必須是相對的。** einsum 在 float32 下的捨入誤差正比於 σ₁ 的量級，
不是固定值：σ₁ ≈ 1e-3 時殘留奇異值約 1e-10，用絕對閾值 1e-10 會誤判。全部
判準一律取 σ_i/σ₁。
"""

from typing import Dict, List

import torch

# 相對閾值：σ_i/σ₁ 低於此值視為零。理由見模組說明。
RANK_RTOL = 1e-6


def channel_spectrum(delta: torch.Tensor) -> Dict[str, List[float]]:
    """單通道 (H,W) 殘差的奇異值、相對變化率、累積能量比。

    以 float64 計算，避免 float32 的捨入誤差污染尾端奇異值。
    """
    sv = torch.linalg.svdvals(delta.double())
    total = (sv**2).sum()
    cum = torch.cumsum(sv**2, dim=0) / total if total > 0 else torch.zeros_like(sv)
    # 變化率取相鄰比值 σ_{i+1}/σ_i，比一階差分更能顯示衰減速度
    ratio = torch.cat([torch.ones(1, dtype=sv.dtype), sv[1:] / sv[:-1].clamp_min(1e-300)])
    return {
        "singular_values": sv.tolist(),
        "decay_ratio": ratio.tolist(),
        "cumulative_energy": cum.tolist(),
    }


def effective_rank(delta: torch.Tensor, rtol: float = RANK_RTOL) -> int:
    """σ_i/σ₁ ≥ rtol 的個數。全零殘差回傳 0。"""
    sv = torch.linalg.svdvals(delta.double())
    if sv[0] <= 0:
        return 0
    return int((sv / sv[0] >= rtol).sum())


def energy_rank(delta: torch.Tensor, frac: float = 0.99) -> int:
    """涵蓋 `frac` 比例能量所需的最小秩。

    此為比 `effective_rank` 更接近 LRDM 低秩假設語意的量度：假設講的是
    「主要能量集中在低維子空間」，而非「尾端奇異值精確為零」。
    """
    sv = torch.linalg.svdvals(delta.double())
    total = (sv**2).sum()
    if total <= 0:
        return 0
    cum = torch.cumsum(sv**2, dim=0) / total
    return int((cum < frac).sum()) + 1


def analyze(delta: torch.Tensor) -> Dict:
    """逐通道分析 (C,H,W) 或 (1,C,H,W) 的殘差。

    逐通道而非把整張圖攤平，是因為秩約束本來就是逐通道施加的（外積參數化
    的 U/V 帶 channel 維度），把通道混在一起會量到不同的東西。
    """
    if delta.dim() == 4:
        if delta.shape[0] != 1:
            raise ValueError(f"僅支援單張影像，收到 batch={delta.shape[0]}")
        delta = delta[0]
    d = delta.detach().cpu()
    return {
        "per_channel": [channel_spectrum(d[c]) for c in range(d.shape[0])],
        "effective_rank": [effective_rank(d[c]) for c in range(d.shape[0])],
        "energy_rank_99": [energy_rank(d[c], 0.99) for c in range(d.shape[0])],
        "energy_rank_90": [energy_rank(d[c], 0.90) for c in range(d.shape[0])],
    }


# ---------------------------------------------------------------- 空間頻率

def radial_power(delta: torch.Tensor, n_bins: int = 32) -> Dict[str, List[float]]:
    """殘差的徑向平均功率譜，用於判斷能量集中在低頻還是瀰散於全頻帶。

    **用途**（E4 提出的可檢驗解釋）：攻擊者的 SDEdit 先把影像加噪到
    `t₀ = 500`，該操作抹除高頻。若防禦擾動的能量集中在高頻，再加噪就會
    把它洗掉；能存活的必須是低頻／結構性成分。此函數量測該分布。

    回傳三項，索引皆對應由低頻到高頻的 `n_bins` 個等寬環：
      `power`          各環的平均功率（已對環內樣本數正規化）
      `cumulative`     由低頻起算的累積能量比
      `freq`           各環的中心頻率，正規化至 [0, 1]（1 為 Nyquist）

    以 float64 計算。輸入 (C,H,W) 或 (1,C,H,W)，跨通道平均。
    """
    d = delta.detach().cpu().double()
    if d.dim() == 4:
        if d.shape[0] != 1:
            raise ValueError(f"僅支援單張影像，收到 batch={d.shape[0]}")
        d = d[0]

    h, w = d.shape[-2:]
    # fftshift 後把 DC 移到中心，再依到中心的距離分環
    f = torch.fft.fftshift(torch.fft.fft2(d), dim=(-2, -1))
    power = (f.real**2 + f.imag**2).mean(dim=0)   # 跨通道平均

    yy = torch.arange(h, dtype=torch.float64).view(-1, 1) - h / 2
    xx = torch.arange(w, dtype=torch.float64).view(1, -1) - w / 2
    # 正規化半徑：除以短邊的一半，使 1.0 對應 Nyquist
    r = torch.sqrt((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2) / (2**0.5)

    edges = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    vals, freqs = [], []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1] if i < n_bins - 1 else r <= edges[i + 1])
        n = int(m.sum())
        vals.append(float(power[m].mean()) if n > 0 else 0.0)
        freqs.append(float((edges[i] + edges[i + 1]) / 2))

    total = sum(vals)
    cum, acc = [], 0.0
    for v in vals:
        acc += v
        cum.append(acc / total if total > 0 else 0.0)
    return {"power": vals, "cumulative": cum, "freq": freqs}


def low_freq_fraction(delta: torch.Tensor, cutoff: float = 0.25) -> float:
    """正規化頻率低於 `cutoff` 的環所佔的能量比例。

    單一純量，便於進表比較。值越高代表殘差越集中於低頻、越可能存活於
    攻擊者的再加噪。
    """
    rp = radial_power(delta)
    total = sum(rp["power"])
    if total <= 0:
        return 0.0
    lo = sum(p for p, f in zip(rp["power"], rp["freq"]) if f < cutoff)
    return lo / total
