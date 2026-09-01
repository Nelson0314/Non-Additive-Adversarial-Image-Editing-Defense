"""`src/defense/apa_stage1.py`：APA 官方階段一，由 `optimize.py` 抽出。

抽出的理由是依賴：主線只用 `align_apa_native`，而 `optimize.py` 匯入
`generator`／`objective`／`purify.ops`／`calibration`，一支帶進 16 個模組
（`docs/PLAN.md` §6.1a）。函式本身只用 `sd` 與 `lora`，與該檔其餘部分零耦合。

**與官方的偏離必須維持逐字不變**（FND-027 是拿它量出來的）：AdamW 而非
Adam、固定步數而非保留最佳步、noise_offset 逐通道。
"""

import pytest
import torch


class _FakeSD:
    """只提供 `align_apa_native` 用到的介面。用假的 SD 是因為本檔要驗的是
    最佳化器設定與迴圈結構，不是 UNet 的數值。"""

    num_train_timesteps = 1000

    def __init__(self):
        self.device = torch.device("cpu")
        self.unet = self._unet
        self.calls = []

    def encode_image(self, x01):
        return torch.zeros(1, 4, 8, 8)

    def encode_text(self, name):
        self.calls.append(name)
        return torch.zeros(1, 77, 32)

    def alphas_cumprod(self, device):
        return torch.linspace(0.99, 0.01, self.num_train_timesteps)

    def _unet(self, noisy, t, cond):
        class _Out:
            pass
        o = _Out()
        o.sample = noisy * self.w
        return o


class _Lora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1))
        self.enabled = False

    def enable(self):
        self.enabled = True


def _sd_with_param(lora):
    sd = _FakeSD()
    sd.w = lora.p
    return sd


def test_沒有可訓練參數時直接拒絕():
    """LoRA 忘了解凍時，訓練會跑完 200 步而什麼都沒學到，且不會報錯。"""
    from src.defense.apa_stage1 import align_apa_native

    lora = _Lora()
    lora.p.requires_grad_(False)
    with pytest.raises(ValueError, match="沒有可訓練參數"):
        align_apa_native(_sd_with_param(lora), lora, torch.zeros(1, 3, 64, 64),
                         "horse", steps=1, lr=1e-4, noise_offset=0.0)


def test_用AdamW而非Adam():
    """官方用 AdamW（weight_decay=1e-2）。換成 Adam 不會報錯，只會讓
    FND-027 的比較變成「本專案版本」而答不了原本的問題。"""
    from src.defense import apa_stage1

    seen = []
    orig = torch.optim.AdamW

    def spy(params, **kw):
        seen.append(kw)
        return orig(params, **kw)

    apa_stage1.torch.optim.AdamW = spy
    try:
        lora = _Lora()
        apa_stage1.align_apa_native(_sd_with_param(lora), lora,
                                    torch.zeros(1, 3, 64, 64), "horse",
                                    steps=1, lr=1e-4, noise_offset=0.0)
    finally:
        apa_stage1.torch.optim.AdamW = orig

    assert seen and seen[0]["weight_decay"] == 1e-2
    assert seen[0]["betas"] == (0.9, 0.999)


def test_跑滿指定步數且不保留最佳步():
    """官方是固定步數直接用最後一步。加上「保留最佳步」會製造第三個
    未查證的偏離。"""
    from src.defense.apa_stage1 import align_apa_native

    lora = _Lora()
    hist = align_apa_native(_sd_with_param(lora), lora,
                            torch.zeros(1, 3, 64, 64), "horse",
                            steps=5, lr=1e-4, noise_offset=0.0, log_every=99)
    assert [h["step"] for h in hist] == [0, 1, 2, 3, 4]
    assert all("loss" in h for h in hist)
