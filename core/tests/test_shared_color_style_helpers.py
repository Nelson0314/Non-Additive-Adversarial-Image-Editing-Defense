"""color 與 style 共用的 Lab 位移、資料集與 CSV helpers，以及注意力目標的 CPU 契約。"""
import argparse
import csv

from PIL import Image
import pytest
import torch

from immunization_core.color.shift import channel_shift_max, channel_shift_p95
from immunization_core.color.space import lab_to_rgb, rgb_to_lab
from immunization_core.io import load_dataset_images, write_sorted_csv
from immunization_core.optimization.attention import CrossAttnObjective


def test_channel_shift_is_zero_on_identity_and_signed():
    x = torch.rand(1, 3, 16, 16) * 0.6 + 0.2
    assert float(channel_shift_p95(x, x, 1, 1)) == pytest.approx(0.0, abs=1e-4)
    lab = rgb_to_lab(x)
    lab[:, 1] += 5.0
    y = lab_to_rgb(lab).clamp(0, 1)
    assert float(channel_shift_p95(x, y, 1, 1)) > 3.0
    assert float(channel_shift_p95(x, y, 1, -1)) == pytest.approx(0.0, abs=1e-3)
    assert float(channel_shift_max(x, y, 1, 1)) >= float(channel_shift_p95(x, y, 1, 1))


def test_dataset_images_follow_prompt_classes_and_filter(tmp_path):
    (tmp_path / "prompts.yaml").write_text("man: {content: man}\nwoman: {content: woman}\n"
                                           "edits: {ip2p: [a]}\n", encoding="utf-8")
    for name in ("man/man_00", "man/man_01", "woman/woman_00"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        Image.new("RGB", (4, 4)).save(tmp_path / f"{name}.png")
    assert [d["name"] for d in load_dataset_images(tmp_path, None)] == ["man_00", "man_01", "woman_00"]
    assert [d["class"] for d in load_dataset_images(tmp_path, {"woman_00"})] == ["woman"]
    with pytest.raises(SystemExit):
        load_dataset_images(tmp_path, {"missing"})


def test_sorted_csv_uses_union_of_sorted_fields(tmp_path):
    path = tmp_path / "rows.csv"
    write_sorted_csv(path, [{"b": 1, "a": 2}, {"c": 3}])
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["a", "b", "c"]
        assert len(list(reader)) == 2


class FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = None

    def set_processor(self, processor):
        self.processor = processor

    def get_attention_scores(self, query, key, attention_mask=None):
        return (query @ key.transpose(-1, -2)).softmax(-1)


class FakeUNet(torch.nn.Module):
    """兩層交叉注意力；query 取自輸入的空間位置，key 取自文字嵌入。"""

    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.block = torch.nn.ModuleDict({"attn2": FakeAttention(), "attn1": FakeAttention()})
        self.proj = torch.nn.Linear(8, 4, bias=False)

    def forward(self, sample, t, encoder_hidden_states=None, return_dict=True):
        query = sample.flatten(2).transpose(1, 2)[..., :4]
        key = encoder_hidden_states
        self.block["attn2"].get_attention_scores(query, key)
        self.block["attn2"].get_attention_scores(2 * query, key)
        return (sample,)


class FakeTokenizer:
    model_max_length = 3

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                 return_tensors=None, add_special_tokens=True):
        ids = [5] if not add_special_tokens else [0, 5, 1][:max_length or 3]
        tensor = torch.tensor([ids])
        return argparse.Namespace(input_ids=tensor if return_tensors else ids)


class FakeIP2P:
    device = torch.device("cpu")
    scaling_factor = 0.5

    def __init__(self):
        self.unet = FakeUNet()
        self.pipe = argparse.Namespace(tokenizer=FakeTokenizer(),
                                       scheduler=argparse.Namespace(alphas_cumprod=torch.linspace(0.99, 0.1, 1000)))
        self.embedding = torch.nn.Embedding(8, 4)

    def text_encoder(self, ids):
        return (self.embedding(ids),)

    def posterior_mean(self, x, use_ckpt=False):
        return x.mean(1, keepdim=True).expand(-1, 4, -1, -1)


def test_cross_attention_mass_starts_at_one_and_has_gradient():
    ip2p = FakeIP2P()
    x = torch.rand(1, 3, 4, 4)
    args = argparse.Namespace(class_word="man", attn_tmin=100, attn_tmax=900, attn_k=2,
                              attn_val_k=2, val_seed=7, noise_seed=3, a_adv=1.0)
    objective = CrossAttnObjective(ip2p, x, args)
    assert objective.layers == ["block.attn2"]
    assert objective.value(x, validation=True) == pytest.approx(1.0, abs=1e-6)
    y = (x + 0.1 * torch.randn_like(x)).requires_grad_(True)
    objective.value(y, backward=True)
    assert y.grad is not None and torch.isfinite(y.grad).all()
