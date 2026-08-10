"""A 段（DEC-016）接進五段流程的三個接點：起點、解碼器、雜湊。

這三件事各自都有「壞掉但沒有症狀」的形態，故各釘一條：

- 起點沒換成 `z*` → 重建仍是舊下限，看起來只是「A 段效果不好」
- 解碼器的微調值沒還原 → 上一張圖的過擬合污染下一張，看起來是「某些圖較好」
- 新旋鈕無條件進 `config_hash` → 既有批次每一格判為未完成而整批重跑
"""

import pytest
import torch
import torch.nn as nn

from src.defense.generator import DefenseGenerator
from src.defense.recon import ReconAdapter, decoder_tunable
from src.experiment.executors import RunConfig
from src.utils.cellid import config_hash


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(2, 4)
        self.conv = nn.Conv2d(4, 4, 3, padding=1)


def _adapter(dec, fill=3.0):
    named = dict(dec.named_parameters())
    return ReconAdapter(
        torch.zeros(1, 4, 8, 8),
        {k: torch.full_like(named[k], fill)
         for k, _ in decoder_tunable(dec)})


def test_applied_在區塊內生效離開即還原():
    dec = ToyDecoder()
    before = {k: v.detach().clone() for k, v in dec.named_parameters()}
    with _adapter(dec).applied(dec):
        assert torch.allclose(dec.norm.weight, torch.full_like(dec.norm.weight, 3.0))
        assert torch.allclose(dec.conv.bias, torch.full_like(dec.conv.bias, 3.0))
    for k, v in dec.named_parameters():
        assert torch.equal(v, before[k]), f"{k} 沒有還原"


def test_未被微調的參數不受影響():
    """只寫進 `decoder` 裡列到的名稱。多寫等於改動 A 段沒有量過的東西。"""
    dec = ToyDecoder()
    w = dec.conv.weight.detach().clone()
    with _adapter(dec).applied(dec):
        assert torch.equal(dec.conv.weight, w)


def test_名稱對不上直接拋出():
    """依名稱而非位置對齊：位置對齊會在結構改動時靜默寫到別的張量上。"""
    dec = ToyDecoder()
    bad = ReconAdapter(torch.zeros(1, 4, 8, 8),
                       {"不存在的層.bias": torch.zeros(4)})
    with pytest.raises(KeyError, match="解碼器沒有這些參數"):
        with bad.applied(dec):
            pass


def test_payload_來回不變():
    dec = ToyDecoder()
    a = _adapter(dec)
    b = ReconAdapter.from_payload(a.to_payload())
    assert torch.equal(a.z_star, b.z_star)
    assert set(a.decoder) == set(b.decoder)
    for k in a.decoder:
        assert torch.equal(a.decoder[k], b.decoder[k])


# ---------------------------------------------------------------------------
# 生成路徑的起點
# ---------------------------------------------------------------------------


class _StubModule:
    site = "apa"
    enabled = False

    def pixel_residual(self, x):
        return None

    def disable(self):
        pass

    def enable(self):
        pass


class _StubSD:
    """記下 `prepare()` 實際拿了什麼當起點。"""

    def __init__(self):
        self.encode_calls = 0
        self.seen = None

    def encode_text(self, p):
        return torch.zeros(1, 4, 4)

    def timesteps(self, k, t_max=None):
        return torch.arange(k)

    def conditioning_for(self, x, vae_ckpt=False):
        from contextlib import nullcontext
        return nullcontext()

    def encode_image(self, x, use_ckpt=False):
        self.encode_calls += 1
        return torch.zeros(1, 4, 8, 8)

    def ddim_inversion(self, z0, emb, ts, k):
        self.seen = z0.detach().clone()
        return z0


def test_有_adapter_時起點取_z_star_而不呼叫編碼器():
    sd = _StubSD()
    z = torch.full((1, 4, 8, 8), 0.7)
    gen = DefenseGenerator(sd, _StubModule(), k_inv=2,
                           recon=ReconAdapter(z, {}))
    gen.prepare(torch.rand(1, 3, 64, 64))
    assert sd.encode_calls == 0, "仍呼叫了 encode_image，A1 等於沒有接上"
    assert torch.equal(sd.seen, z)


def test_無_adapter_時走原本的編碼器():
    sd = _StubSD()
    gen = DefenseGenerator(sd, _StubModule(), k_inv=2)
    gen.prepare(torch.rand(1, 3, 64, 64))
    assert sd.encode_calls == 1


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------


def _cell_cfg(cfg: RunConfig):
    """`config_hash` 吃的必填鍵，其餘固定，只讓 `module_params` 變動。"""
    return {
        "spec_version": 1, "model": "sd", "resolution": 512, "guidance": 7.5,
        "steps": 50, "strength": 0.6, "gpu": "RTX-3090", "precision": "fp32",
        "condition": "apa", "loss_params": {}, "optim_params": {},
        "lr": None, "tau": None, "purify": None, "seed": 0,
        "image_id": "horse_00",
        "module_params": cfg.module_params(),
    }


def test_recon_關閉時雜湊與新增欄位之前逐位相同():
    """既有批次（`runs/s3t20_merged` 等）續跑時每一格必須仍判為已完成。
    這裡直接釘住 `module_params` 的鍵集合——新增鍵無條件出現就是那個缺陷。"""
    cfg = RunConfig()
    assert cfg.recon is False
    assert set(cfg.module_params()) == {
        "warp_grid_size", "warp_max_disp", "warp_resample",
        "apa_lora_rank", "apa_latent_max_rank", "apa_latent_const_rank",
        "random_init_std",
    }


def test_recon_開啟時雜湊必須改變():
    """換了重建起點就是換一組實驗，沿用舊產物會把兩條路徑的結果混在一起。"""
    off, on = RunConfig(), RunConfig(recon=True)
    assert config_hash(_cell_cfg(off)) != config_hash(_cell_cfg(on))


@pytest.mark.parametrize("field,value", [
    ("recon_objective", "dists"),
    ("recon_a1_steps", 400),
    ("recon_a1_lr", 0.04),
    ("recon_a2_steps", 100),
    ("recon_a2_lr", 1e-3),
    ("recon_floor_ratio", 0.7),
    ("recon_gamma_acut", 0.0),
    ("recon_acut_band", 0.1),
    ("recon_w_pixel", 0.15),
])
def test_每個_A_段旋鈕都改變雜湊(field, value):
    """漏掉任一個，改了它的批次會沿用前一批的產物而沒有症狀。"""
    base = RunConfig(recon=True)
    other = RunConfig(recon=True, **{field: value})
    assert config_hash(_cell_cfg(base)) != config_hash(_cell_cfg(other))
