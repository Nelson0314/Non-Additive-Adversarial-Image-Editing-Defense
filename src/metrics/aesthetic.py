"""美學／影像品質指標 — 給 `scripts/apa_baseline.py` 的評測用。

NIMA-AVA、CNNIQA 出自指導者提供的 APA 官方消融表（Table 3，
`docs/reference/dia_apa.md` §5.1），本專案既有的 `MetricSuite` 沒有這兩項
——它們是單張影像的美學／品質評分，不比對原圖，與 NIQE／銳利度保留率
量的是不同的東西。透過 `pyiqa`（已於 conda env wacv 驗證，0.1.16）取得
官方預訓練權重，不重新訓練。

CLIP image-image cosine 同樣取自那張表的「CLIP Score」欄——那是重建影像
對原圖的 CLIP 影像-影像餘弦相似度，與本專案既有的 CLIP-T（影像-文字語意
對齊，`src/metrics/suite.py` 的 `pairwise`）是不同的量，故另開一個輕量
函式，不擴充 `MetricSuite`（那個類別的 CLIP 模型是為 image-text 對齊載入的，
混用會讓兩種用途共用同一個 processor 設定，日後改任一邊都會互相牽動）。
"""

from typing import Dict

import torch


class AestheticSuite:
    """惰性載入：三個模型只在第一次被用到時才建構，避免探針只跑一項指標時
    也要付全部模型的載入成本。
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._nima = None
        self._cnniqa = None
        self._clip_model = None
        self._clip_proc = None

    def _ensure_nima(self) -> None:
        if self._nima is None:
            import pyiqa
            self._nima = pyiqa.create_metric("nima", device=self.device)

    def _ensure_cnniqa(self) -> None:
        if self._cnniqa is None:
            import pyiqa
            self._cnniqa = pyiqa.create_metric("cnniqa", device=self.device)

    def _ensure_clip(self) -> None:
        if self._clip_model is None:
            from transformers import AutoModel, AutoProcessor
            repo = "openai/clip-vit-base-patch32"
            self._clip_proc = AutoProcessor.from_pretrained(repo)
            self._clip_model = (AutoModel.from_pretrained(repo)
                                .to(self.device).eval())

    @torch.no_grad()
    def nima(self, x01: torch.Tensor) -> float:
        """NIMA-AVA 美學分數，越高越好。輸入 (1,3,H,W)，[0,1]。"""
        self._ensure_nima()
        return float(self._nima(x01.to(self.device).clamp(0, 1)).mean())

    @torch.no_grad()
    def cnniqa(self, x01: torch.Tensor) -> float:
        """CNNIQA 品質分數，越高越好。"""
        self._ensure_cnniqa()
        return float(self._cnniqa(x01.to(self.device).clamp(0, 1)).mean())

    @torch.no_grad()
    def clip_image_similarity(self, a01: torch.Tensor, b01: torch.Tensor) -> float:
        """兩張影像的 CLIP 餘弦相似度，[-1,1]，越高越像。

        對應 APA Table 3 的「CLIP Score」欄——重建影像對原圖，不是
        `MetricSuite.pairwise` 的 image-text CLIP-T。
        """
        self._ensure_clip()
        imgs = [_to_pil(a01), _to_pil(b01)]
        inputs = self._clip_proc(images=imgs, return_tensors="pt").to(self.device)
        # transformers 5.14.1：`CLIPModel.__call__` 需要同時給 `input_ids`
        # （`MetricSuite.pairwise` 走的是那條路，見 `suite.py` 的
        # `semantic`：`model(pixel_values=img, **tok)`）。純影像場景改走
        # `get_image_features`，但該版本的回傳是 `BaseModelOutputWithPooling`，
        # 投影後的影像特徵在其 `.pooler_output`（見
        # `CLIPModel.get_image_features` 原始碼：
        # `vision_outputs.pooler_output = self.visual_projection(pooled_output)`），
        # 不是 `.image_embeds`。
        out = self._clip_model.get_image_features(
            pixel_values=inputs["pixel_values"])
        feats = out.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return float((feats[0] @ feats[1]).item())

    def measure(self, x01: torch.Tensor) -> Dict[str, float]:
        """回傳不比對原圖的兩項：NIMA、CNNIQA。"""
        return {"nima": self.nima(x01), "cnniqa": self.cnniqa(x01)}


def _to_pil(x01: torch.Tensor):
    from torchvision.transforms.functional import to_pil_image
    return to_pil_image(x01.detach().cpu().clamp(0, 1)[0])
