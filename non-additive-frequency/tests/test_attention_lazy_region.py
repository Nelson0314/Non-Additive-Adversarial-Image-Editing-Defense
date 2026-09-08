"""注意力項的支撐必須**延後解析**，否則那個臂永遠跑不起來。

`PatchParam.support` 是在 `reset()` 裡才建的，而 `reset()` 發生在
`run_param_pgd` **內部**——組損失的時候它還是 `None`。原本
`scripts/ip2p_run.py` 的守門寫成 `getattr(param, "support", None) is None`
就拒絕，於是 `attn` 那個臂**從來沒有跑起來過**：每一次派工都死在自己的守門上，
訊息是「收到的參數化沒有 support」，看起來像是旗標組錯了。

`_with_consistency` 的註解早就記過同一條規則——**檢查屬性存不存在，不檢查
它的值**（那次的症狀是「三個工作點全部被自己的守門擋下」）。這一支當初沒有
照做。本檔把兩件事釘住：

1. `make_attention_term` 的 `region` 接受 callable，且**在 `term()` 呼叫時
   才解析**——建構當下傳 `lambda: None` 不可以拋錯。
2. 解析出 `None` 時要**明確拋錯**，不可以回零。回零會讓這一項安靜地失效，
   而 CSV 上的 `attn_weight` 仍然寫著一個非零值。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.attention_loss import make_attention_term  # noqa: E402
from src.defense.patch_param import PatchParam  # noqa: E402


class _StubIP2P:
    """只提供 `make_attention_term` 建構期會碰到的東西。

    建構期讀 `unet`、`device`、排程器，並對乾淨影像取一次 latent；真正的
    UNet 前向要到 `term()` 才發生，而本檔的測試全部停在 `term()` 之前或在
    它拋錯的那一行。`encode_image` 的形狀取 512²／8 = 64 的四通道 latent。
    """

    def __init__(self):
        self.unet = torch.nn.Module()
        self.device = torch.device("cpu")
        self.scheduler = _StubScheduler()

    def encode_image(self, x01):
        h, w = x01.shape[-2:]
        return torch.zeros(1, 4, h // 8, w // 8)

    def image_latents(self, x01):
        return self.encode_image(x01)


class _StubScheduler:
    alphas_cumprod = torch.linspace(0.999, 0.001, 1000)


@pytest.fixture
def stub(monkeypatch):
    ip2p = _StubIP2P()
    monkeypatch.setattr("src.defense.fixedpoint_loss._scheduler_of",
                        lambda _: _StubScheduler())
    return ip2p


def _embeds():
    return torch.zeros(2, 77, 768)


def test_建構時傳_callable_不會解析它(stub):
    """支撐還沒建好時就要能組出損失——那正是 `run_param_pgd` 之前的狀態。

    若 `make_attention_term` 在建構期就呼叫 `region()`，這裡會拿到 None
    並在建構期爆掉，而那就是原本那個 bug 的形狀。
    """
    calls = []

    def region():
        calls.append(1)
        return None

    term = make_attention_term(
        stub, region=region, text_embeds=_embeds(), weight=1.0,
        zt_mode="noise", x_clean=torch.rand(1, 3, 64, 64), seed=0)
    assert callable(term)
    assert calls == [], "建構期不可以解析 region——那時支撐還沒建好"


def test_支撐仍為_None_時拋錯而不是回零(stub):
    """回零會讓這一項安靜地失效，而 CSV 上的 `attn_weight` 仍是非零值。"""
    term = make_attention_term(
        stub, region=lambda: None, text_embeds=_embeds(), weight=1.0,
        zt_mode="noise", x_clean=torch.rand(1, 3, 64, 64), seed=0)
    with pytest.raises(RuntimeError, match="support"):
        term(torch.rand(1, 3, 64, 64))


def test_傳張量仍然可以(stub):
    """既有呼叫端傳的是張量，不可以因為支援 callable 就壞掉。"""
    term = make_attention_term(
        stub, region=torch.ones(1, 1, 64, 64), text_embeds=_embeds(),
        weight=1.0, zt_mode="noise", x_clean=torch.rand(1, 3, 64, 64), seed=0)
    assert callable(term)


def test_support_在_reset_之前確實是_None():
    """這是整個 bug 的前提，直接釘住它——哪天 `PatchParam` 改成在
    `__init__` 就建支撐，這個測試會失敗並提醒上面那些延後解析可以簡化。"""
    p = PatchParam(radius=0.05, placement="complement")
    assert p.support is None
    assert hasattr(p, "support"), (
        "守門檢查的是屬性存不存在，故這個屬性必須恆存在")
