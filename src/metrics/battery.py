"""保真約束的候選指標組 —— 供 P1/P2 篩選使用。

這個模組的用途不是評測，是篩選。判準只有一個：把它加進約束後，
最佳化還能不能靠模糊換取防禦效果。故此處收錄的指標必須涵蓋「高頻能量
流失」這一軸——那正是 `suite.py` 的九項共同的盲區（見 E19 §3）。

收錄理由逐項如下：

| 指標 | 類型 | 方向 | 為何收錄 |
|---|---|---|---|
| `gmsd`      | 梯度   | 低者佳 | 逐點梯度強度相似度的空間標準差。取標準差是關鍵：它不像全域能量比那樣可以用「一處模糊、他處加噪」互相抵銷 |
| `nlpd`      | 多尺度 | 低者佳 | 正規化 Laplacian 金字塔，模擬局部亮度相減與對比增益控制。模糊直接抽掉各頻帶能量 |
| `vif_p`     | 資訊   | 高者佳 | 資訊論式保真度。模糊即資訊流失，是古典指標中對模糊最敏感者之一 |
| `haarpsi`   | 小波   | 高者佳 | Haar 小波高頻係數的相似度，直接量高頻 |
| `ms_ssim`   | 結構   | 高者佳 | 多尺度 SSIM，作為既有 SSIM 的多尺度對照 |
| `dists`     | 感知   | 低者佳 | 已知在此失效（E19 §3.2），保留為陰性對照 |
| `lpips`     | 感知   | 低者佳 | 現行綁定約束，保留為基準 |
| `stlpips`   | 感知   | 低者佳 | shift-tolerant LPIPS。site S 是空間位移，標準 LPIPS 對位移的計價方式本身可疑，需要一把對位移寬容的尺來對照 |
| `acutance`  | 頻域   | 趨近 1 | 現行的補丁。已知有結構性漏洞：全域梯度能量比，一處模糊他處加噪可湊回 1.0。收錄是為了量出該漏洞 |
| `niqe`      | 無參考 | 低者佳 | 不需原圖，故無法用「與原圖的總體差距」來鑽 |
| `musiq`     | 無參考 | 高者佳 | 學習式無參考品質，與 NIQE 的統計式互為對照 |

方向不一致是刻意的：篩選時要看的是「模糊 vs 雜訊哪個被判得比較貴」，
統一方向會把資訊壓掉。報告端一律查 `HIGHER_IS_BETTER`。
"""

from typing import Dict, Optional

import torch

# `acutance` 不列於此表：其最佳值為 1，既非越高越好也非越低越好。
HIGHER_IS_BETTER = {
    "gmsd": False,
    "nlpd": False,
    "vif_p": True,
    "haarpsi": True,
    "ms_ssim": True,
    "dists": False,
    "lpips": False,
    "stlpips": False,
    "niqe": False,
    "musiq": True,
    "psnr": True,
    "ssim": True,
}

# 無參考指標對「待評影像」單獨計算，不吃原圖。分開列出是因為篩選時
# 它們的解讀方式不同：其餘各項量的是「離原圖多遠」，這兩項量的是
# 「這張圖本身像不像自然影像」。
NO_REFERENCE = ("niqe", "musiq")

# MUSIQ 的骨幹在 224 以下的輸入會取不到多尺度區塊；512² 的正式資料
# 不會觸發，此常數只用於讓煙霧測試明確跳過而非拋出形狀錯誤。
MUSIQ_MIN_SIDE = 224
# NIQE 以 96×96 區塊統計，短邊小於此值時沒有任何完整區塊（同 suite.py）。
NIQE_MIN_SIDE = 96


class MetricBattery:
    """候選指標的統一介面。影像一律為 (N,3,H,W)、[0,1]。

    模型延遲載入並常駐，整個篩選實驗應共用同一個實例。
    """

    def __init__(self, device: Optional[torch.device] = None):
        import piq

        self.device = device or torch.device("cpu")
        self._lpips = piq.LPIPS().to(self.device)
        self._dists = piq.DISTS().to(self.device)
        self._pyiqa: Dict[str, object] = {}

    def _iqa(self, name: str):
        """pyiqa 模型的延遲載入與快取。"""
        if name not in self._pyiqa:
            import pyiqa

            self._pyiqa[name] = pyiqa.create_metric(name, device=self.device)
        return self._pyiqa[name]

    @torch.no_grad()
    def full_reference(self, orig: torch.Tensor, dist: torch.Tensor) -> Dict[str, float]:
        """全部有參考指標。`orig` 為參照、`dist` 為待評影像。

        `acutance_ratio` 不對稱（是 dist 相對 orig 的比值），其餘各項對調
        後不變或取倒數。故兩個引數的順序有意義，不可互換。
        """
        import piq

        from src.metrics.acutance import acutance

        a = orig.to(self.device).clamp(0, 1)
        b = dist.to(self.device).clamp(0, 1)

        out = {
            "psnr": float(piq.psnr(a, b, data_range=1.0)),
            "ssim": float(piq.ssim(a, b, data_range=1.0)),
            "ms_ssim": float(piq.multi_scale_ssim(a, b, data_range=1.0)),
            "lpips": float(self._lpips(a, b)),
            "dists": float(self._dists(a, b)),
            "gmsd": float(piq.gmsd(a, b, data_range=1.0)),
            "haarpsi": float(piq.haarpsi(a, b, data_range=1.0)),
            "vif_p": float(piq.vif_p(a, b, data_range=1.0)),
            "nlpd": float(self._iqa("nlpd")(b, a)),
            "stlpips": float(self._iqa("stlpips")(b, a)),
            "acutance_ratio": acutance(a, b)["acutance_ratio"],
        }
        return out

    @torch.no_grad()
    def no_reference(self, x: torch.Tensor) -> Dict[str, float]:
        """無參考指標。影像過小時回傳 NaN。

        回傳 NaN 而非拋例外，理由同 `suite.py` 的 NIQE：那是指標的定義域
        限制而非錯誤，讓它中斷整批篩選是把限制升級成故障。
        """
        x = x.to(self.device).clamp(0, 1)
        side = min(x.shape[-2], x.shape[-1])
        nan = float("nan")
        return {
            "niqe": float(self._iqa("niqe")(x)) if side >= NIQE_MIN_SIDE else nan,
            "musiq": float(self._iqa("musiq")(x)) if side >= MUSIQ_MIN_SIDE else nan,
        }

    def evaluate(self, orig: torch.Tensor, dist: torch.Tensor) -> Dict[str, float]:
        """有參考 + 待評影像的無參考指標。原圖的無參考值另以後綴記錄，
        供判讀「這張圖本身變得比原圖更不自然還是更自然」。"""
        out = self.full_reference(orig, dist)
        out.update(self.no_reference(dist))
        for k, v in self.no_reference(orig).items():
            out[f"{k}_orig"] = v
        return out
