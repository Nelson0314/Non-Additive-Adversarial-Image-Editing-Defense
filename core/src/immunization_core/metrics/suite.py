"""影像成對、分布與語意指標。

影像輸入為 (N,3,H,W)、[0,1]，指標輸出為 float 或具名字典。
MetricSuite 建構時載入 piq LPIPS／DISTS 權重，其餘後端依方法延遲載入；
匯入模組不載權重。各方法保留原有前處理、空輸入與最小樣本數契約。"""

from typing import Dict, Optional, Sequence

import torch

CLIP_REPO = "openai/clip-vit-base-patch32"
SIGLIP_REPO = "google/siglip-base-patch16-224"

HIGHER_IS_BETTER = {
    "psnr": True, "linf": False, "ssim": True, "lpips": False,
    "dists": False, "niqe": False, "clip": True, "siglip": True,
    "vif_p": True, "fsim": True,
    "rms": False, "frac_gt_16_255": False,
}


def _delta_e00(a: torch.Tensor, b: torch.Tensor) -> float:
    """兩張 (N,3,H,W)、[0,1] sRGB 影像的平均 CIEDE2000 色差。

    sRGB → CIELAB 的轉換與色差公式都取自 `skimage.color`，不自行實作——
    CIEDE2000 有六個分段條件與一個旋轉項，自己寫的版本很難驗。
    回傳的是逐像素色差的平均，量綱是 ΔE 單位（1 約為剛可辨的差異）。
    """
    from skimage.color import deltaE_ciede2000, rgb2lab

    x = a.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    y = b.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return float(deltaE_ciede2000(rgb2lab(x), rgb2lab(y)).mean())


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
        self._inception = None
        if not lazy:
            self._ensure_niqe()
            self._ensure_vlm()

    @property
    def lpips_module(self):
        """可微的 LPIPS 本體，給要拿它當損失的呼叫端。

        `pairwise` 是 `@torch.no_grad` 的量測介面，回傳的是 float，接不到
        梯度。重建對齊（`src/defense/recon.py`）要的是同一個度量的**可微**
        版本——同一份權重，量測與最佳化才是同一件事；各自 `piq.LPIPS()` 一份
        會多佔一份 VGG 的顯存，而且哪一份被用到看不出來。
        """
        return self._lpips

    @property
    def dists_module(self):
        """可微的 DISTS 本體。存在理由同 `lpips_module`。"""
        return self._dists

    # ---- 延遲載入：像素／感知指標常用，語意與無參考指標較少用 ----

    def _ensure_niqe(self):
        if self._niqe is None:
            import pyiqa

            self._niqe = pyiqa.create_metric("niqe", device=self.device)

    def _ensure_inception(self):
        """FID 用的 Inception-V3。

        `use_fid_inception=True` 取的是 FID 原論文那份權重（`pt_inception`），
        不是 torchvision 的 ImageNet 分類權重——兩者的數值不可比，換了就與
        文獻上任何一個 FID 數字都對不起來。`output_blocks=[3]` 是 2048 維的
        pool3 特徵，即標準設定。
        """
        if self._inception is None:
            from piq.feature_extractors import InceptionV3

            self._inception = InceptionV3(
                output_blocks=[3], resize_input=True, normalize_input=True,
                requires_grad=False, use_fid_inception=True).to(self.device).eval()

    def _ensure_vlm(self):
        if self._clip is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._clip_proc = AutoProcessor.from_pretrained(CLIP_REPO)
        self._clip = AutoModel.from_pretrained(CLIP_REPO).to(self.device).eval()
        self._siglip_proc = AutoProcessor.from_pretrained(SIGLIP_REPO)
        self._siglip = AutoModel.from_pretrained(SIGLIP_REPO).to(self.device).eval()

    def release_vlm(self) -> None:
        """把 CLIP 與 SigLIP 移出顯存。下次用到時 `_ensure_vlm` 會重新載入。"""
        self._clip = None
        self._siglip = None
        self._clip_proc = None
        self._siglip_proc = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 成對指標 ----

    @torch.no_grad()
    def pairwise(self, a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
        """a 與 b 的成對指標。NIQE 另計。

        銳利度不對稱：`acutance_ratio` 是 b 相對 a 的比值，故 a 必須是
        參照（原圖）、b 是待評影像。其餘各項對調不變，此項會變成倒數。

            before: psnr / linf / ssim / lpips / dists / acutance_ratio
                    / rms / frac_gt_16_255                        八項
            after:  上列八項 + vif_p + fsim                        十項

        兩個輸入一律轉 fp32。**指標量的是影像，不是產生它的計算精度**，
        而 `piq` 不做隱式轉型：混著餵會以
        `RuntimeError: expected scalar type BFloat16 but found Float` 中止。

        `objective.fidelity_term` 早已在自己內部做同一件事（見該函式
        「三者一律轉 fp32」）。同一個問題在兩處各修一次，代表轉型應該放在
        指標的邊界上，故此處補上——凡是進得了指標套件的張量都經過這裡。
        """
        import piq

        from immunization_core.metrics.acutance import acutance

        a = a.to(self.device).float().clamp(0, 1)
        b = b.to(self.device).float().clamp(0, 1)
        d = (a - b).abs()
        return {
            "psnr": float(piq.psnr(a, b, data_range=1.0)),
            "linf": float(d.max()),
            "ssim": float(piq.ssim(a, b, data_range=1.0)),
            "vif_p": float(piq.vif_p(a, b, data_range=1.0)),
            "fsim": float(piq.fsim(a, b, data_range=1.0)),
            "lpips": float(self._lpips(a, b)),
            "dists": float(self._dists(a, b)),
            "acutance_ratio": acutance(a, b)["acutance_ratio"],
            "rms": float((d ** 2).mean().sqrt()),
            "frac_gt_16_255": float((d > 16 / 255).float().mean()),
            # CIEDE2000 的平均色差。**為色彩重映射的防禦而加**：LPIPS 與
            # DISTS 都是結構／紋理度量，對全域色偏的懲罰偏輕，只用它們對齊
            # 失真會系統性偏袒色彩方法。這一欄是那個偏誤的對照軸。
            # 走 skimage 的 `deltaE_ciede2000`（實作已被廣泛驗證），
            # 不自行實作該公式。
            "deltaE00": _delta_e00(a, b),
        }

    FID_MIN_TRUSTED = 150

    @torch.no_grad()
    def fid(self, xs, ys, batch_size: int = 8) -> float:
        """兩組影像之間的 FID。`xs`／`ys` 為 (N,3,H,W)、[0,1] 或其可迭代形式。

        **與 `pairwise` 的差別是它吃分布不是吃一對。** 兩組的張數可以不同
        （FID 比的是兩個高斯的距離），但每組至少要有 2 張，否則協方差無定義；
        少於 `FID_MIN_TRUSTED` 時仍會算出來並不攔阻，是因為煙霧測試要跑得動，
        **但報表端有責任留空**——把它藏成例外反而會讓小樣本數字悄悄進表。

        `batch_size` 預設 8 是為了本機 4 GB 顯存的 512² 影像；遠端可調大。
        """
        feats = []
        for group in (xs, ys):
            if isinstance(group, torch.Tensor):
                imgs = group if group.dim() == 4 else group.unsqueeze(0)
            else:
                imgs = torch.cat([g if g.dim() == 4 else g.unsqueeze(0)
                                  for g in group], dim=0)
            if imgs.shape[0] < 2:
                raise ValueError(
                    f"FID 每組至少需要 2 張影像，收到 {imgs.shape[0]} 張")
            self._ensure_inception()
            out = []
            for i in range(0, imgs.shape[0], batch_size):
                chunk = imgs[i:i + batch_size].to(self.device).float().clamp(0, 1)
                f = self._inception(chunk)[0]
                out.append(f.flatten(start_dim=1).cpu())
            feats.append(torch.cat(out, dim=0))

        import piq

        return float(piq.FID()(feats[0], feats[1]))

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

        `pairwise` 的 fp32 轉型理由在此同樣適用：pyiqa 的 NIQE 權重是 fp32。
        段 2 的 `rayscale_executor` 對 N3 傳的 `x_tau` 直接來自
        `gen.generate`，即本批的計算精度。
        """
        side = min(x.shape[-2], x.shape[-1])
        if side < self.NIQE_MIN_SIDE:
            return float("nan")
        self._ensure_niqe()
        return float(self._niqe(x.to(self.device).float().clamp(0, 1)))

    @torch.no_grad()
    def semantic_multi(self, x: torch.Tensor,
                       prompts: Sequence[str]) -> Dict[str, Dict[str, float]]:
        """一張影像對多個 prompt 的語意對齊，回傳 `{prompt: {model: score}}`。

            margin(y) = SigLIP(y, 目標類) − SigLIP(y, 原類)

        `semantic` 改為呼叫本方法，故**兩者的數字不可能分歧**——那正是把
        多 prompt 版本做成同一條路徑而不是另寫一份前處理的理由。有測試釘住。

        影像只前向一次，文字逐 prompt 前向。全量重算 2100 張圖時，這使
        SigLIP／CLIP 的影像前向次數由 `圖 × prompt` 降到 `圖`。
        """
        self._ensure_vlm()
        from torchvision.transforms.functional import resize

        prompts = list(prompts)
        if not prompts:
            raise ValueError("prompts 不可為空：沒有 prompt 就沒有可算的對齊")
        x = x.to(self.device).float()
        out: Dict[str, Dict[str, float]] = {p: {} for p in prompts}
        for key, model, proc in (
            ("clip", self._clip, self._clip_proc),
            ("siglip", self._siglip, self._siglip_proc),
        ):
            size = proc.image_processor.size
            side = size.get("shortest_edge") or size["height"]
            img = resize(x.clamp(0, 1), [side, side], antialias=True)
            mean = torch.tensor(proc.image_processor.image_mean, device=self.device)
            std = torch.tensor(proc.image_processor.image_std, device=self.device)
            img = (img - mean[:, None, None]) / std[:, None, None]

            tok = proc.tokenizer(
                prompts, return_tensors="pt",
                padding="max_length" if key == "siglip" else True,
                truncation=True,
            ).to(self.device)
            res = model(pixel_values=img, **tok)
            ie = res.image_embeds / res.image_embeds.norm(dim=-1, keepdim=True)
            te = res.text_embeds / res.text_embeds.norm(dim=-1, keepdim=True)
            # ie 是 (1, D)、te 是 (P, D)。逐 prompt 取內積，`semantic` 的
            # 單 prompt 情形於是逐位元落回原本的 `(ie * te).sum(-1).mean()`。
            sims = (ie * te).sum(-1)
            for p, v in zip(prompts, sims.tolist()):
                out[p][key] = float(v)
        return out

    @torch.no_grad()
    def image_similarity(self, a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
        """兩張影像在 CLIP／SigLIP 影像空間裡的餘弦相似度。"""
        self._ensure_vlm()
        from torchvision.transforms.functional import resize

        out: Dict[str, float] = {}
        for key, model, proc in (
            ("clip", self._clip, self._clip_proc),
            ("siglip", self._siglip, self._siglip_proc),
        ):
            size = proc.image_processor.size
            side = size.get("shortest_edge") or size["height"]
            mean = torch.tensor(proc.image_processor.image_mean, device=self.device)
            std = torch.tensor(proc.image_processor.image_std, device=self.device)
            # 走完整前向再取 `image_embeds`，與 `semantic_multi` 同一條路徑。
            # `get_image_features` 在本環境的 transformers 版本回傳的是
            # `BaseModelOutputWithPooling` 而不是張量，且它少了投影層；
            # 共用同一條前向也保證兩個方法的影像側不可能分岔。文字側餵一個
            # 固定的空 prompt，其輸出不被使用。
            tok = proc.tokenizer(
                [""], return_tensors="pt",
                padding="max_length" if key == "siglip" else True,
                truncation=True,
            ).to(self.device)
            embs = []
            for x in (a, b):
                img = resize(x.to(self.device).float().clamp(0, 1),
                             [side, side], antialias=True)
                img = (img - mean[:, None, None]) / std[:, None, None]
                e = model(pixel_values=img, **tok).image_embeds
                embs.append(e / e.norm(dim=-1, keepdim=True))
            out[key] = float((embs[0] * embs[1]).sum(-1).mean())
        return out

    @torch.no_grad()
    def direction_similarity(self, src: torch.Tensor, edit: torch.Tensor,
                             instruction: str) -> Dict[str, float]:
        """            CLIP-S = cos( E_img(編輯) − E_img(原圖), E_txt(指令) )

        它與既有的兩個語意讀數量的不是同一件事：

        前處理與 `semantic_multi`／`image_similarity` 走同一段，故三者的影像側
        不可能分岔。SigLIP 一併回報：它的文字塔訓練目標不同（sigmoid 而非
        對比餘弦），差異本身是可報的，但**不與 CLIP 的值互比絕對大小**。
        """
        self._ensure_vlm()
        from torchvision.transforms.functional import resize

        out: Dict[str, float] = {}
        for key, model, proc in (
            ("clip", self._clip, self._clip_proc),
            ("siglip", self._siglip, self._siglip_proc),
        ):
            size = proc.image_processor.size
            side = size.get("shortest_edge") or size["height"]
            mean = torch.tensor(proc.image_processor.image_mean, device=self.device)
            std = torch.tensor(proc.image_processor.image_std, device=self.device)
            tok = proc.tokenizer(
                [instruction], return_tensors="pt",
                padding="max_length" if key == "siglip" else True,
                truncation=True,
            ).to(self.device)
            embs = []
            for x in (src, edit):
                img = resize(x.to(self.device).float().clamp(0, 1),
                             [side, side], antialias=True)
                img = (img - mean[:, None, None]) / std[:, None, None]
                embs.append(model(pixel_values=img, **tok))
            # **相減前不正規化**，與 FaceLock 的 `eval_clip_s.py` 一致：
            # 它對 `get_image_features` 的原始輸出相減。先正規化再相減會得到
            # 另一個量（球面上的弦向量），數值不同。
            delta = embs[1].image_embeds - embs[0].image_embeds
            te = embs[0].text_embeds
            out[key] = float(
                torch.nn.functional.cosine_similarity(delta, te, dim=-1).mean())
        return out

    def semantic(self, x: torch.Tensor, prompt: str) -> Dict[str, float]:
        """影像與 prompt 的語意對齊。CLIP 取餘弦相似度、SigLIP 取其 logit。

        兩者尺度不同，只比較組間差異，不比較彼此的絕對值。

        輸入轉 fp32 的理由同 `pairwise`：CLIP 與 SigLIP 的權重是 fp32。
        """
        return self.semantic_multi(x, [prompt])[prompt]

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
