"""spec §8.1 的八項指標。

設計主張：任何 `edit(orig)` vs `edit(defend)` 或 `x_def` vs `x` 的比較都
必須同時報出全部八項，不得只報單一指標。此要求來自 v2 的實測：apa 方法的
LPIPS 與 pg_enc 幾乎相同，PSNR 卻相差 12.7 dB、L∞ 相差 28 倍。單一 LPIPS
會低估非加性方法造成的失真，據以下結論會有系統性偏差。

| 指標 | 類型 | 方向 | 實作 | Lo Table 1 |
|---|---|---|---|---|
| PSNR   | 像素   | 高者佳 | piq.psnr | ✓ |
| L∞     | 像素   | 低者佳 | 直接計算 | 約束 κ |
| SSIM   | 結構   | 高者佳 | piq.ssim | ✓ |
| VIFp   | 資訊   | 高者佳 | piq.vif_p | ✓ |
| FSIM   | 特徵   | 高者佳 | piq.fsim | ✓ |
| LPIPS  | 感知   | 低者佳 | piq.LPIPS (AlexNet) | ✓ |
| DISTS  | 感知   | 低者佳 | piq.DISTS | |
| 銳利度 | 頻域   | 趨近 1 | src.metrics.acutance | |
| NIQE   | 無參考 | 低者佳 | pyiqa | |
| CLIP   | 語意   | 高者佳 | openai/clip-vit-base-patch32 | |
| SigLIP | 語意   | 高者佳 | google/siglip-base-patch16-224 | |

最右欄標示該項是否出現在 Lo et al., *Distraction is All You Need*（CVPR 2024）
Table 1。該表是本專案對齊的主判準：五項全部量「免疫後的編輯輸出」與
「原始編輯結果」之間的不相似度（PSNR↓ SSIM↓ VIFp↓ FSIM↓ LPIPS↑），
對應本模組 `full(y_ref, y_def)` 產生的 `edit_*` 欄位。

2026-07-31 新增銳利度保留率（第 9 項）。before：`pairwise` 回傳
psnr/linf/ssim/lpips/dists 五項。after：加上 `acutance_ratio`。原因是 E18 的
人眼比對發現一個兩個感知指標都沒抓到的現象——latent 最佳化後的影像「不差，
但比較鈍」。實測銳利度由下限的 95.7% 掉到 84.3%，而 LPIPS 六張全判改善、
DISTS 只對其中三張報退步且與鈍化程度不對應（dog_01 鈍化到 81.9% 仍判改善）。
上面那條「不得只報單一指標」的主張，這次是被自己的指標組打臉：八項裡沒有
一項直接量高頻能量流失。

CLIP 與 SigLIP 兩個語意指標都納入，是為了避免單一視覺語言模型的偏誤主導
語意層面的結論。兩者的分數尺度不同（CLIP 為餘弦相似度、SigLIP 為 sigmoid
校準過的 logit），不可互相比較絕對值，只能各自比較組間差異。

模型權重載入一次後常駐，`MetricSuite` 應在整個實驗中共用同一個實例。
"""

from typing import Dict, Optional, Sequence

import torch

CLIP_REPO = "openai/clip-vit-base-patch32"
SIGLIP_REPO = "google/siglip-base-patch16-224"

# 方向表：報告與繪圖依此決定「較好」的方向，不在各處各寫一次
HIGHER_IS_BETTER = {
    "psnr": True, "linf": False, "ssim": True, "lpips": False,
    "dists": False, "niqe": False, "clip": True, "siglip": True,
    # VIFp 與 FSIM（2026-08-03）：兩者皆為相似度，高者代表兩張影像較接近。
    # 注意本表描述的是「影像品質／相似度」的方向，不是「防禦成功」的方向。
    # 用於 Lo Table 1 的判準時，`edit_*` 前綴下的 vif_p、fsim 反而是越低
    # 代表防禦越成功——那個轉換由報表端負責，不在此表混入。
    "vif_p": True, "fsim": True,
    # 銳利度保留率的最佳值是 1（與原圖相同），不是越高越好也不是越低越好：
    # < 1 為鈍化、> 1 為過銳。故不列於此表，報告端須依「趨近 1」處理。
    #
    # 擾動的能量與尖峰比例（2026-08-02，E31）。文獻以 ε=16/255 的 L∞ 球為
    # 標準預算，而本專案的 beta_linf=0，擾動是稀疏尖峰型：τ=0.10 的實測是
    # 5.3% 的像素超過該球、中位數只有 5/255。只報 LPIPS 會讓與文獻的對比
    # 失真，故兩軸都要有欄位。
    "rms": False, "frac_gt_16_255": False,
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

    def release_vlm(self) -> None:
        """把 CLIP 與 SigLIP 移出顯存。下次用到時 `_ensure_vlm` 會重新載入。

        兩者合計 **1,352 MB**（CLIP 577 + SigLIP 775，2026-08-06 於 RTX 3090
        實測），而它們**只在語意指標上用得到**，訓練迴圈一次也不碰。

        為什麼需要顯式釋放：`_ensure_vlm` 是延遲載入，段 0 的
        `calibrate_strength` 為了做編輯有效性過濾而呼叫 `semantic()`，之後
        兩份權重就一直留在卡上。接著 `calibrate_lr` 要建 N1 的訓練圖——
        該條件因 attention hook 與 checkpoint 不相容而**不能開 UNet
        checkpoint**（見 `optimize._build_attn_step`），1024² 下需保留
        12 個完整的 UNet 計算圖。23.56 GB 的卡上實測差約 600 MB 而 OOM，
        閒置的 1,352 MB 正是可回收的部分。

        本方法只動裝置常駐，不改任何數值：重新載入的是同一個 repo 的同一份
        權重，`semantic()` 的結果逐位元不變。

        **不要在評測迴圈裡逐格呼叫**——那 4,000 格每格都要重載 1.35 GB。
        呼叫點限於「接下來要跑長時間最佳化」之處。

        2026-08-06 新增。before：無釋放路徑，`_ensure_vlm` 載入後即常駐。
        """
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

        2026-08-02（E31）加入 `rms` 與 `frac_gt_16_255`。原本只有 psnr /
        linf / ssim / lpips / dists / acutance_ratio 六項。理由：與文獻的
        預算對比需要 RMS 這一軸。本專案在 τ_lpips=0.10 的實測是 LPIPS
        0.0856、RMS 0.0319、L∞ 0.373，其中 L∞ 是文獻標準球 ε=16/255 的
        六倍而 LPIPS 只有文獻運作點（0.267–0.362）的三分之一——擾動是稀疏
        尖峰型（5.3% 的像素超過該球、中位數 5/255），只報單一軸會讓
        「本專案的預算比文獻低 5–8 倍」這個敘述在別的軸上不成立。

        2026-08-03 加入 `vif_p` 與 `fsim`。

            before: psnr / linf / ssim / lpips / dists / acutance_ratio
                    / rms / frac_gt_16_255                        八項
            after:  上列八項 + vif_p + fsim                        十項

        理由：Lo et al.（CVPR 2024）Table 1 的判準是 PSNR／SSIM／VIFp／
        FSIM／LPIPS 五項，本專案原本只有其中三項。缺 VIFp 與 FSIM 時，
        既有全部 run 都無法與該表逐欄對照——而該表是本專案的主判準
        （見模組 docstring 的表格最右欄）。`piq` 0.8.0 已內含 `vif_p` 與
        `fsim`，不需新增相依。

        兩個輸入一律轉 fp32。**指標量的是影像，不是產生它的計算精度**，
        而 `piq` 不做隱式轉型：混著餵會以
        `RuntimeError: expected scalar type BFloat16 but found Float` 中止。

        2026-08-06 修正。before：只有 `.to(self.device).clamp(0, 1)`，dtype
        原樣傳給 `piq`。走生成路徑的條件（N3／site apa）的 `x_def` 來自
        `gen.generate`，是本批的計算精度（bf16），而 `entry.x01` 是 fp32，
        於是 `_finish_train` 的 `pairwise(entry.x01, x_def)` 在**整格訓練
        跑完之後**中止（b2 的 N3 bird_03，第 117 步收斂後）。

        `objective.fidelity_term` 早已在自己內部做同一件事（見該函式
        「三者一律轉 fp32」）。同一個問題在兩處各修一次，代表轉型應該放在
        指標的邊界上，故此處補上——凡是進得了指標套件的張量都經過這裡。
        """
        import piq

        from src.metrics.acutance import acutance

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

        2026-08-08 新增。動機是**類別 margin** 這個讀出量：

            margin(y) = SigLIP(y, 目標類) − SigLIP(y, 原類)

        它比的是**同一張圖對兩個 prompt**，故畫質與風格的變化會同時影響兩項
        而抵消，剩下的才是類別訊息。既有的 `effect_siglip` 比的是**兩張不同
        的圖對同一個 prompt**，於是「圖變怪了」與「類別被改掉了」進到同一個
        數字裡（`RESULTS_2026-08-08` §9.2）。

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
