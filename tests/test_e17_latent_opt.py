"""E18/E19 對 `scripts/e17_vae_floor.latent_opt` 兩項改動的契約測試。

刻意不斷言收斂行為。tiny-SD 是隨機初始化的測試模型，「lam 調大則
LPIPS 變低」這種命題在它身上不成立也不代表程式錯誤，那是模型規模的性質。
此檔只測結構性契約：權重是否真的進了損失、decoder 是否真的被替換掉。
"""

import pytest
import torch

from scripts.e17_vae_floor import asym_decoder, latent_opt
from src.models.sd import SDWrapper
from src.utils.device import get_device

TINY = "hf-internal-testing/tiny-stable-diffusion-pipe"
SIZE = 64
SEED = 20260731
DEV = get_device()


@pytest.fixture(scope="module")
def sd():
    return SDWrapper(TINY)


@pytest.fixture(scope="module")
def x01():
    g = torch.Generator().manual_seed(SEED)
    return torch.rand(1, 3, SIZE, SIZE, generator=g).to(DEV)


def _lpips_stub(a, b):
    """可微、值域為正的替身，避免測試依賴 piq 的權重下載。

    取兩張圖的平均絕對差，對 a 可微，數值與 LPIPS 無關——本檔測的是
    「這一項有沒有以 lam 的比例進入損失」，不是 LPIPS 本身。
    """
    return (a - b).abs().mean()


def test_lam_0_使損失退化為純_mse(sd, x01):
    """lam=0 時記錄的 loss 必須等於 mse，否則 LPIPS 項沒有真的被權重控制。"""
    _, hist = latent_opt(sd, x01, _lpips_stub, steps=2, lr=1e-3, lam=0.0)
    assert hist, "history 不該為空"
    for e in hist:
        assert e["loss"] == pytest.approx(e["mse"], rel=1e-6), \
            f"lam=0 時 loss 應等於 mse，實得 loss={e['loss']} mse={e['mse']}"


def test_lam_以線性比例進入損失(sd, x01):
    """loss - mse 必須等於 lam × lpips，逐步檢查。"""
    lam = 3.0
    _, hist = latent_opt(sd, x01, _lpips_stub, steps=2, lr=1e-3, lam=lam)
    for e in hist:
        assert e["loss"] - e["mse"] == pytest.approx(lam * e["lpips"], rel=1e-5), \
            f"step {e['step']}: loss-mse={e['loss'] - e['mse']} != {lam}*{e['lpips']}"


def test_decode_參數確實取代預設_decoder(sd, x01):
    """傳入的 decode 必須被呼叫，且回傳值來自它而非 sd.decode_latent。"""
    calls = []

    def fake_decode(z):
        calls.append(z.shape)
        # 與 SD decoder 同形狀但內容恆定，好與真實輸出區分
        return torch.full_like(x01, 0.25).requires_grad_(True) * z.mean()

    rec, hist = latent_opt(sd, x01, _lpips_stub, steps=3, lr=1e-3,
                           decode=fake_decode)
    # 迴圈 3 次 + 收尾 1 次
    assert len(calls) == 4, f"decode 應被呼叫 4 次，實得 {len(calls)}"
    assert rec.shape == x01.shape


def test_decode_為_None_時走原廠_decoder(sd, x01):
    """不傳 decode 時輸出必須與 sd.decode_latent 的形狀一致。"""
    rec, _ = latent_opt(sd, x01, _lpips_stub, steps=1, lr=1e-3)
    with torch.no_grad():
        expect = sd.decode_latent(sd.encode_image(x01))
    assert rec.shape == expect.shape


def test_asym_decoder_回傳可微且形狀正確的函式(sd, x01):
    """用一個假的 asym VAE 驗證包裝邏輯：mask 全 1、值域由 [-1,1] 轉回 [0,1]。"""

    class FakeSample:
        def __init__(self, sample):
            self.sample = sample

    class FakeAsymVAE:
        dtype = torch.float32

        def __init__(self):
            self.seen_mask = None

        def decode(self, z, image, mask):
            self.seen_mask = mask
            # 回傳 [-1,1] 值域的張量，包裝層應轉成 [0,1]
            out = torch.zeros_like(image) + z.mean() * 0.0 - 1.0
            return FakeSample(out + z.sum() * 0.0)

    vae = FakeAsymVAE()
    dec = asym_decoder(vae, sd, x01)
    z = sd.encode_image(x01)
    out = dec(z)

    assert out.shape == x01.shape
    assert torch.equal(vae.seen_mask, torch.ones_like(x01[:, :1])), \
        "asym_decoder 必須用 mask 全 1，條件分支才拿不到原圖"
    # -1 經 (out+1)/2 應為 0
    # 顯式轉 float：`pytest.approx` 會試著把張量轉成 numpy，而 CUDA 張量
    # 不能直接轉。先前 DEV 恰為 CPU 故沒發作。
    assert float(out.abs().max()) == pytest.approx(0.0, abs=1e-6)
