"""VAE-Trans 淨化：以微調後的 VAE encoder 編碼、ip2p 原 VAE 的 decoder 解碼，一次、取 `latent_dist.mean`、不加雜訊。

encoder 權重由 color 專案訓練（`color/trials/lut3d_defense_and_naturalness/tools/train_vae_trans.py`，
輸出 `artifacts/purification/vae_trans/model/encoder.pt`，鍵為 `encoder` 與 `quant_conv`）。
本模組的計算與該 trial 的 `measure/apply_vae_purification.py` 中 `vae_trans` 一支相同：
輸入以 `load_image_tensor(size=512)` 讀入，`decode(encode(2x−1).mean)` 後映回 [0,1] 並截斷。

淨化名 `LABEL` 與七道算子並列：淨化後的輸入放在
`<淨化根>/<條件>/vae_trans/ip2p_<條件>_vae_trans/<影像>__orig.png`，編輯器以 `--purifier vae_trans` 讀取。
"""

from __future__ import annotations

from pathlib import Path

import torch

LABEL = "vae_trans"
RESOLUTION = 512


class VAETrans:
    def __init__(self, encoder_path: Path, device):
        from diffusers import AutoencoderKL

        from immunization_core.editors.instruct_pix2pix import MODEL_NAME

        self.base = AutoencoderKL.from_pretrained(
            MODEL_NAME, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
        self.trans = AutoencoderKL.from_pretrained(
            MODEL_NAME, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
        state = torch.load(encoder_path, map_location=device)
        self.trans.encoder.load_state_dict(state["encoder"])
        self.trans.quant_conv.load_state_dict(state["quant_conv"])
        self.model_name = MODEL_NAME

    @torch.no_grad()
    def __call__(self, x01: torch.Tensor) -> torch.Tensor:
        z = self.trans.encode(x01 * 2 - 1).latent_dist.mean
        return ((self.base.decode(z).sample + 1) / 2).clamp(0, 1)
