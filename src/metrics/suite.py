"""spec §8.1 的八項指標。

**設計主張**：任何 `edit(orig)` vs `edit(defend)` 或 `x_def` vs `x` 的比較都
必須同時報出全部八項，不得只報單一指標。此要求來自 v2 的實測：apa 方法的
LPIPS 與 pg_enc 幾乎相同，PSNR 卻相差 12.7 dB、L∞ 相差 28 倍。單一 LPIPS
會低估非加性方法造成的失真，據以下結論會有系統性偏差。

| 指標 | 類型 | 方向 | 實作 |
|---|---|---|---|
| PSNR   | 像素   | 高者佳 | piq.psnr |
| L∞     | 像素   | 低者佳 | 直接計算 |
| SSIM   | 結構   | 高者佳 | piq.ssim |
| LPIPS  | 感知   | 低者佳 | piq.LPIPS (AlexNet) |
| DISTS  | 感知   | 低者佳 | piq.DISTS |
| NIQE   | 無參考 | 低者佳 | pyiqa |
| CLIP   | 語意   | 高者佳 | openai/clip-vit-base-patch32 |
| SigLIP | 語意   | 高者佳 | google/siglip-base-patch16-224 |

CLIP 與 SigLIP 兩個語意指標都納入，是為了避免單一視覺語言模型的偏誤主導
語意層面的結論。兩者的分數尺度不同（CLIP 為餘弦相似度、SigLIP 為 sigmoid
校準過的 logit），**不可互相比較絕對值**，只能各自比較組間差異。

模型權重載入一次後常駐，`MetricSuite` 應在整個實驗中共用同一個實例。
"""

from typing import Dict, Optional

import torch

CLIP_REPO = "openai/clip-vit-base-patch32"
SIGLIP_REPO = "google/siglip-base-patch16-224"

# 方向表：報告與繪圖依此決定「較好」的方向，不在各處各寫一次
HIGHER_IS_BETTER = {
    "psnr": True, "linf": False, "ssim": True, "lpips": False,
    "dists": False, "niqe": False, "clip": True, "siglip": True,
}


class MetricSuite:
    """八項指標的統一介面。影像一律為 (N,3,H,W)、[0,1]。"""

    def __init__(self, device: Optional[torch.device] = None, lazy: bool = True):
        import piq

        self.device = device or torch.device("cpu")
        self._lpips = piq.LPIPS().to(self.device)
        self._dists = piq.DISTS().to(self.device)
        self._niqe = None
        self._clip = None
        self._siglip = None
        if not lazy:
            self._ensure_niqe()
            self._ensure_vlm()

    # ---- 延遲載入：像素／感知指標常用，語意與無參考指標較少用 ----

    def _ensure_niqe(self):
        if self._niqe is None:
            import pyiqa

            self._niqe = pyiqa.create_metric("niqe", device=self.device)

    def _ensure_vlm(self):
        if self._clip is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._clip_proc = AutoProcessor.from_pretrained(CLIP_REPO)
        self._clip = AutoModel.from_pretrained(CLIP_REPO).to(self.device).eval()
        self._siglip_proc = AutoProcessor.from_pretrained(SIGLIP_REPO)
        self._siglip = AutoModel.from_pretrained(SIGLIP_REPO).to(self.device).eval()

    # ---- 成對指標 ----

    @torch.no_grad()
    def pairwise(self, a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
        """a 與 b 的六項成對指標。順序無關者對稱，NIQE 另計。"""
        import piq

        a = a.to(self.device).clamp(0, 1)
        b = b.to(self.device).clamp(0, 1)
        return {
            "psnr": float(piq.psnr(a, b, data_range=1.0)),
            "linf": float((a - b).abs().max()),
            "ssim": float(piq.ssim(a, b, data_range=1.0)),
            "lpips": float(self._lpips(a, b)),
            "dists": float(self._dists(a, b)),
        }

    # NIQE 把影像切成 96×96 區塊統計，短邊小於此值時沒有任何完整區塊，
    # pyiqa 會以 [1,1,0,0] 的空張量進入 F.pad 而拋出 RuntimeError。
    NIQE_MIN_SIDE = 96

    @torch.no_grad()
    def niqe(self, x: torch.Tensor) -> float:
        """無參考品質。影像短邊小於 96 時回傳 NaN。

        回傳 NaN 而非拋出例外，是因為這是 NIQE 的定義域限制而非錯誤：
        512² 的正式實驗不會觸發，只有 tiny-SD 的 64² 煙霧測試會。讓一個
        指標算不出來就中斷整批實驗，是把限制升級成故障。NaN 會原樣寫入
        CSV，分析時看得見它缺席，不會被誤當成 0。
        """
        side = min(x.shape[-2], x.shape[-1])
        if side < self.NIQE_MIN_SIDE:
            return float("nan")
        self._ensure_niqe()
        return float(self._niqe(x.to(self.device).clamp(0, 1)))

    @torch.no_grad()
    def semantic(self, x: torch.Tensor, prompt: str) -> Dict[str, float]:
        """影像與 prompt 的語意對齊。CLIP 取餘弦相似度、SigLIP 取其 logit。

        兩者尺度不同，只比較組間差異，不比較彼此的絕對值。
        """
        self._ensure_vlm()
        from torchvision.transforms.functional import resize

        out = {}
        for key, model, proc in (
            ("clip", self._clip, self._clip_proc),
            ("siglip", self._siglip, self._siglip_proc),
        ):
            size = proc.image_processor.size
            side = size.get("shortest_edge") or size["height"]
            img = resize(x.to(self.device).clamp(0, 1), [side, side], antialias=True)
            mean = torch.tensor(proc.image_processor.image_mean, device=self.device)
            std = torch.tensor(proc.image_processor.image_std, device=self.device)
            img = (img - mean[:, None, None]) / std[:, None, None]

            tok = proc.tokenizer(
                [prompt], return_tensors="pt",
                padding="max_length" if key == "siglip" else True,
                truncation=True,
            ).to(self.device)
            res = model(pixel_values=img, **tok)
            ie = res.image_embeds / res.image_embeds.norm(dim=-1, keepdim=True)
            te = res.text_embeds / res.text_embeds.norm(dim=-1, keepdim=True)
            out[key] = float((ie * te).sum(-1).mean())
        return out

    def full(
        self, a: torch.Tensor, b: torch.Tensor, prompt: Optional[str] = None
    ) -> Dict[str, float]:
        """八項全報。`prompt` 為 None 時略過語意指標。

        NIQE 是無參考指標，對 a 與 b 各報一次，故鍵名帶後綴。
        """
        out = self.pairwise(a, b)
        out["niqe_a"] = self.niqe(a)
        out["niqe_b"] = self.niqe(b)
        if prompt is not None:
            for k, v in self.semantic(a, prompt).items():
                out[f"{k}_a"] = v
            for k, v in self.semantic(b, prompt).items():
                out[f"{k}_b"] = v
        return out
