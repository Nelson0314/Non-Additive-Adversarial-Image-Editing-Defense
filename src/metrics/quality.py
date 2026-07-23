"""指標計算 — SPEC.md §2.6、STRUCTURE.md §2.5。

PSNR/SSIM/VIFp/FSIM/LPIPS 一律用 piq，不可自行實作、不可換套件
（APA 用 IQA-PyTorch，不同套件，勿混用）。FID、CLIP score 為補充。

衡量的是兩個「編輯結果」之間的差異，差異越大代表保護越成功。
FID 為資料集層級指標（需特徵集合），不在 compute_all 之內，另以
compute_fid() 提供。
"""

import piq
import torch

# 方向定義：True 表示「數值越高、防禦越成功」
# 採 DAYN 慣例（SPEC §2.6）。PhotoGuard 用相反慣例，勿混淆。
METRIC_HIGHER_IS_BETTER = {
    "psnr":  False,
    "ssim":  False,
    "vifp":  False,
    "fsim":  False,
    "lpips": True,
    "fid":   True,
    "clip":  False,
}

_lpips = None


def _get_lpips():
    """LPIPS 模型延遲載入（首次呼叫下載 VGG16 權重）。"""
    global _lpips
    if _lpips is None:
        _lpips = piq.LPIPS()
    return _lpips


def compute_all(
    edited_protected: torch.Tensor,
    edited_original: torch.Tensor,
    prompt: str = None,      # CLIP score 需要
    clip_scorer=None,        # ClipScorer；與 prompt 同時提供才計算 clip
) -> dict[str, float]:
    """計算指標。回傳 {"psnr": ..., "ssim": ..., "vifp": ..., "fsim": ..., "lpips": ...[, "clip": ...]}"""
    x = edited_protected.detach().float().clamp(0, 1)
    y = edited_original.detach().float().clamp(0, 1)
    out = {
        "psnr": piq.psnr(x, y, data_range=1.0).item(),
        "ssim": piq.ssim(x, y, data_range=1.0).item(),
        "vifp": piq.vif_p(x, y, data_range=1.0).item(),
        "fsim": piq.fsim(x, y, data_range=1.0).item(),
        "lpips": _get_lpips()(x, y).item(),
    }
    if prompt is not None and clip_scorer is not None:
        out["clip"] = clip_scorer.score(x, prompt)
    return out


def lpips_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """LPIPS(x, y)，兩張 (1,3,H,W) [0,1]。stage0 相似性校準用。"""
    return _get_lpips()(x.detach().float().clamp(0, 1), y.detach().float().clamp(0, 1)).item()


def compute_fid(feats_x: torch.Tensor, feats_y: torch.Tensor) -> float:
    """FID（資料集層級）。feats 為兩組影像之特徵（N, D），由呼叫端以
    InceptionV3 等抽取後傳入（piq.FID 之標準用法）。"""
    return piq.FID()(feats_x, feats_y).item()


class ClipScorer:
    """CLIP score：編輯結果與（惡意）prompt 之對齊程度（cosine similarity）。

    model_name 由 config 提供；本地測試可用 tiny 模型。
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        from src.utils.device import get_device

        self.device = get_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def score(self, image01: torch.Tensor, prompt: str) -> float:
        import torchvision.transforms as T

        pil = T.ToPILImage()(image01[0].clamp(0, 1).cpu())
        inputs = self.processor(
            text=[prompt], images=pil, return_tensors="pt", padding=True
        ).to(self.device)
        out = self.model(**inputs)
        sim = torch.cosine_similarity(out.image_embeds, out.text_embeds)
        return sim.item()
