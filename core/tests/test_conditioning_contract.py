"""SDXL 拆分後保留序列與 pooled 嵌入配對，不載入編輯模型。"""
import pytest
import torch

from immunization_core.editors.conditioning import (
    SDXLPrompt, concatenate_conditioning, expand_conditioning,
)


def test_concatenation_keeps_each_pooled_embedding_with_its_sequence():
    a = SDXLPrompt(torch.ones(1, 2, 4), torch.full((1, 3), 10.0))
    b = SDXLPrompt(torch.full((1, 2, 4), 2.0), torch.full((1, 3), 20.0))
    combined = concatenate_conditioning([a, b])
    assert torch.equal(combined.embeds[:, 0, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(combined.pooled[:, 0], torch.tensor([10.0, 20.0]))


def test_expansion_and_residual_preserve_pooled_embedding():
    prompt = SDXLPrompt(torch.ones(1, 2, 4), torch.full((1, 3), 5.0))
    expanded = expand_conditioning(prompt, 3)
    moved = expanded + torch.full_like(expanded.embeds, 0.5)
    assert moved.embeds.shape == (3, 2, 4)
    assert torch.all(moved.embeds == 1.5)
    assert torch.equal(moved.pooled, torch.full((3, 3), 5.0))


def test_sd_tensor_conditioning_and_empty_input_contract():
    x = torch.ones(1, 2, 4)
    assert concatenate_conditioning([x, x]).shape == (2, 2, 4)
    assert expand_conditioning(x, 3).shape == (3, 2, 4)
    with pytest.raises(ValueError, match="空序列"):
        concatenate_conditioning([])
