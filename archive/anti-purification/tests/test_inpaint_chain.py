"""`SDWrapper.inpaint` 的取樣鏈契約。

釘住的是一條**曾經寫錯而且沒有症狀**的規則：9 通道的 inpainting 模型
**不在取樣迴圈裡把遮罩外的 latent 貼回原圖**。官方 pipeline 的那一行混合
被 `num_channels_unet == 4` 守住（`diffusers 0.39.0` 的
`pipeline_stable_diffusion_inpaint.py` 第 1294 行守衛、第 1307 行混合），
9 通道模型靠的是後 5 個條件通道。

寫錯的後果是：遮罩外的 latent 每一步被壓回原圖，而那一塊在本專案的人像上
佔 49–67%，模型幾乎沒有空間照 prompt 生成。輸出看起來仍然是一張合理的圖
（就是原圖加一點背景變化），所以**沒有例外、沒有 NaN、沒有任何症狀**，
只有逐張看圖才會發現指令沒有被執行。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.sd import SDWrapper  # noqa: E402

LATENT = 4
SIDE = 16


class _ConstUNet(torch.nn.Module):
    """回傳固定 eps 的替身。in_channels 宣告成 9 讓 wrapper 走 inpainting 路徑。"""

    def __init__(self, value: float = 0.1):
        super().__init__()
        self.value = value
        self.config = type("C", (), {"in_channels": 9})()
        self.calls = []

    def forward(self, z, t, emb, **kw):
        self.calls.append(tuple(z.shape))
        return type("O", (), {"sample": torch.full(
            (z.shape[0], LATENT, z.shape[-2], z.shape[-1]), self.value,
            dtype=z.dtype, device=z.device)})()


def _wrapper(monkeypatch):
    sd = SDWrapper.__new__(SDWrapper)
    sd.model_name = "stub-inpainting"
    sd.device = torch.device("cpu")
    sd.compute_dtype = torch.float32
    sd._inpaint_cond = None
    unet = _ConstUNet()
    sd._unet = unet

    def enc(x01, use_ckpt=False):
        return torch.nn.functional.avg_pool2d(x01, 8)[:, :LATENT] \
            if x01.shape[1] >= LATENT else torch.nn.functional.avg_pool2d(
                x01.repeat(1, 2, 1, 1), 8)[:, :LATENT]

    monkeypatch.setattr(type(sd), "unet", property(lambda s: unet))
    monkeypatch.setattr(type(sd), "is_inpainting", property(lambda s: True))
    monkeypatch.setattr(type(sd), "inpaint_in_channels",
                        property(lambda s: 9))
    monkeypatch.setattr(type(sd), "num_train_timesteps", property(lambda s: 1000))
    monkeypatch.setattr(type(sd), "latent_channels", property(lambda s: LATENT))
    monkeypatch.setattr(SDWrapper, "encode_image",
                        lambda s, x01, use_ckpt=False: enc(x01))
    monkeypatch.setattr(SDWrapper, "decode_latent",
                        lambda s, z, use_ckpt=False: z[:, :3])
    monkeypatch.setattr(SDWrapper, "alphas_cumprod",
                        lambda s, device=None: torch.linspace(
                            0.999, 1e-4, 1000))
    monkeypatch.setattr(
        SDWrapper, "_eps_cfg",
        lambda s, zin, t, emb, g, emb_u, use_ckpt=False: unet(
            zin, t, emb).sample)

    # `mask_latents` 會去問 VAE 的 `block_out_channels` 推下採樣倍率，替身沒有
    # 真的 VAE，故直接給它等價的結果：遮罩最近鄰縮到 latent 格點，
    # masked-image latent 取「重繪區歸零」的編碼。極性與本體一致。
    def _mask_latents(s, x01, mask, vae_ckpt=False):
        m = torch.nn.functional.interpolate(mask, size=(SIDE, SIDE),
                                            mode="nearest")
        z_masked = enc(x01 * (1.0 - mask) + 0.5 * mask)
        return m, z_masked

    monkeypatch.setattr(SDWrapper, "mask_latents", _mask_latents)
    return sd, unet


def _inputs():
    x01 = torch.rand(1, 3, SIDE * 8, SIDE * 8)
    mask = torch.zeros(1, 1, SIDE * 8, SIDE * 8)
    mask[..., : SIDE * 4, :] = 1.0          # 上半重繪、下半保留
    noise = torch.randn(1, LATENT, SIDE, SIDE)
    emb = torch.zeros(1, 77, 8)
    return x01, mask, noise, emb


def test_九通道輸入由四加一加四拼成(monkeypatch):
    sd, unet = _wrapper(monkeypatch)
    x01, mask, noise, emb = _inputs()
    sd.inpaint(x01, mask, emb, noise, num_steps=2)
    assert unet.calls, "UNet 沒有被呼叫"
    for shape in unet.calls:
        assert shape[1] == 9, f"UNet 收到 {shape[1]} 通道，應為 9"


def test_取樣迴圈裡不把遮罩外貼回原圖(monkeypatch):
    """這是本檔的主張。

    逐步貼回時，遮罩外的 latent 在最後一步會**恰好等於原圖的編碼**
    （t_prev = 0 那一支貼的是乾淨的 `z_keep`）。沒有貼回時它是自由演化的。
    所以「遮罩外是否逐位元等於原圖編碼」可以把兩種實作分開。
    """
    sd, _ = _wrapper(monkeypatch)
    x01, mask, noise, emb = _inputs()
    out = sd.inpaint(x01, mask, emb, noise, num_steps=3)

    z_orig = sd.encode_image(x01)
    m_lat = torch.nn.functional.interpolate(mask, size=(SIDE, SIDE),
                                            mode="nearest")
    keep = (m_lat < 0.5)
    # `decode_latent` 在替身裡是取前三個通道，故可直接比對。
    kept = out[:, :3][keep.expand(-1, 3, -1, -1)]
    ref = z_orig[:, :3][keep.expand(-1, 3, -1, -1)]
    assert not torch.allclose(kept, ref, atol=1e-6), (
        "遮罩外逐位元等於原圖的編碼，表示取樣迴圈裡又把它貼回去了。"
        "9 通道模型不該這樣做，見本模組 docstring")


def test_沒有遮罩就拒絕執行(monkeypatch):
    sd, _ = _wrapper(monkeypatch)
    x01, mask, noise, emb = _inputs()
    with pytest.raises(ValueError, match="尺寸不符"):
        sd.inpaint(x01, mask[..., :8, :8], emb, noise, num_steps=1)
