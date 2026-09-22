"""`src/defense/purify_aware.py` 與可微分 JPEG 往返的驗收。

要釘住三件事：直通估計的**前向值不能偏離真實 JPEG**（否則量到的失真與最終
存檔的不一致）、梯度要真的通得過（否則整個「針對淨化最佳化」是空的）、
以及把 `transform` 加進 `run_param_pgd` **沒有改變預設行為**。
"""

import pytest
import torch

from src.baselines.jpeg_codec import (
    jpeg_roundtrip, jpeg_roundtrip_ste, quant_table, quantize_ste,
)
from src.defense.param_pgd import AdditiveParam, run_param_pgd
from src.defense.purify_aware import (
    CURRICULUM_Q_HI, CURRICULUM_Q_LO, jpeg_quality_at, make_eot_jpeg_transform,
    make_fixed_jpeg_transform, make_jpeg_transform,
)


def _img(h=64, w=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    base = (torch.sin(xx / 5.0) + torch.cos(yy / 7.0)) * 0.25 + 0.5
    x = base.view(1, 1, h, w).repeat(1, 3, 1, 1).double()
    return (x + torch.randn(x.shape, generator=g).double() * 0.05).clamp(0, 1)


# ---- 直通估計 ----

def test_ste_forward_is_bitwise_equal_to_real_rounding():
    coef = torch.randn(1, 2, 2, 8, 8, dtype=torch.float64) * 40
    tbl = quant_table(75, device=coef.device, dtype=coef.dtype)
    assert torch.equal(quantize_ste(coef, tbl), torch.round(coef / tbl))


def test_ste_backward_is_the_identity_through_rounding():
    """`round` 的導數應被當成 1，故 `d(quantize)/d(coef) = 1/table`。"""
    coef = (torch.randn(1, 1, 1, 8, 8, dtype=torch.float64) * 40
            ).requires_grad_(True)
    tbl = quant_table(75, device=coef.device, dtype=coef.dtype)
    quantize_ste(coef, tbl).sum().backward()
    torch.testing.assert_close(coef.grad, (1.0 / tbl).expand_as(coef).clone(),
                               atol=1e-12, rtol=0)


@pytest.mark.parametrize("quality", [50, 75, 95])
def test_ste_roundtrip_matches_the_real_roundtrip_bitwise(quality):
    """**這一項是整個設計的前提**：可微版與真實版的前向必須逐位元相同，
    否則最佳化看到的圖與最終存檔的圖不是同一張。"""
    x = _img()
    assert torch.equal(jpeg_roundtrip(x, quality),
                       jpeg_roundtrip_ste(x, quality).detach())


def test_real_roundtrip_gradient_is_exactly_zero_and_ste_is_not():
    """**這就是 DCT-Shield 必須把 δ 加在量化之後的理由，值得釘住。**

    PyTorch 的 `torch.round` 並非「沒有梯度」——它有 `grad_fn`，只是導數恆為
    零。所以真實往返會安靜地回傳一個全零梯度，最佳化一步都不動而且不報錯。
    直通估計換掉的正是那個零。
    """
    y = _img().requires_grad_(True)
    out = jpeg_roundtrip(y, 75)
    assert out.requires_grad, "round 有 grad_fn，只是導數為零"
    out.pow(2).sum().backward()
    assert float(y.grad.abs().sum()) == 0.0, "真實往返的梯度必須恆為零"

    x = _img().requires_grad_(True)
    jpeg_roundtrip_ste(x, 75).pow(2).sum().backward()
    assert float(x.grad.abs().sum()) > 0, "直通估計必須讓梯度通過"


def test_ste_roundtrip_rejects_bad_shapes():
    with pytest.raises(ValueError):
        jpeg_roundtrip_ste(torch.rand(1, 3, 24, 64), 75)
    with pytest.raises(ValueError):
        jpeg_roundtrip_ste(torch.rand(3, 64, 64), 75)


# ---- 課程排程 ----

def test_curriculum_hits_both_endpoints():
    assert jpeg_quality_at(0, 100) == CURRICULUM_Q_HI
    assert jpeg_quality_at(99, 100) == CURRICULUM_Q_LO


def test_curriculum_is_monotone_non_increasing():
    seq = [jpeg_quality_at(i, 60) for i in range(60)]
    assert all(a >= b for a, b in zip(seq, seq[1:]))


def test_curriculum_single_step_degenerates_to_high_quality():
    assert jpeg_quality_at(0, 1) == CURRICULUM_Q_HI


def test_curriculum_rejects_out_of_range_and_inverted_endpoints():
    with pytest.raises(ValueError):
        jpeg_quality_at(5, 5)
    with pytest.raises(ValueError):
        jpeg_quality_at(-1, 5)
    with pytest.raises(ValueError):
        jpeg_quality_at(0, 10, q_hi=50, q_lo=95)
    with pytest.raises(ValueError):
        jpeg_quality_at(0, 0)


def test_transforms_change_the_image_and_keep_the_range():
    x = _img()
    for t in (make_jpeg_transform(10), make_fixed_jpeg_transform(75),
              make_eot_jpeg_transform([50, 75, 95])):
        out = t(x, 0)
        assert out.shape == x.shape
        assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
        assert float((out - x).abs().max()) > 1e-4


def test_eot_transform_is_reproducible_and_actually_varies():
    x = _img()
    # 不能用 mean 判別：JPEG 往返幾乎不動直流，品質 30 與 95 的平均值相同。
    # 改用「離原圖最遠的像素」，它隨量化步長變化。
    dev = lambda t: round(float((t - x).abs().max()), 9)   # noqa: E731
    ta = make_eot_jpeg_transform([30, 95], seed=1)
    tb = make_eot_jpeg_transform([30, 95], seed=1)
    a = [dev(ta(x, i)) for i in range(8)]
    b = [dev(tb(x, i)) for i in range(8)]
    assert a == b, "同一個 seed 必須給出同一串品質"
    assert len(set(a)) > 1, "抽樣應該真的在變"


def test_eot_transform_rejects_empty():
    with pytest.raises(ValueError):
        make_eot_jpeg_transform([])


# ---- 接進 run_param_pgd ----

def _loss(x):
    return x.pow(2).mean()


def test_default_behaviour_is_unchanged_without_transform():
    """加入 `transform` 參數不得改動既有結果——這是回歸測試。"""
    x = _img()
    a = run_param_pgd(x, AdditiveParam(radius=0.02), _loss, steps=4, seed=0)
    b = run_param_pgd(x, AdditiveParam(radius=0.02), _loss, steps=4, seed=0,
                      transform=None)
    assert torch.equal(a.x_def, b.x_def)


def test_transform_is_applied_and_receives_the_step_index():
    x = _img()
    seen = []

    def spy(t, step):
        seen.append(step)
        return t

    run_param_pgd(x, AdditiveParam(radius=0.02), _loss, steps=5, seed=0,
                  transform=spy)
    assert seen == [0, 1, 2, 3, 4]


def test_transform_changes_the_result():
    x = _img()
    plain = run_param_pgd(x, AdditiveParam(radius=0.02), _loss, steps=6, seed=0)
    jpeg = run_param_pgd(x, AdditiveParam(radius=0.02), _loss, steps=6, seed=0,
                         transform=make_fixed_jpeg_transform(50))
    assert not torch.equal(plain.x_def, jpeg.x_def)


def test_returned_defence_image_is_not_transformed():
    """**交出去的是防禦圖，不是被淨化過的防禦圖。** 若回傳被 transform 過的
    影像，等於我們自己先幫攻擊方壓縮了一次，抗淨化的讀數會失去意義。"""
    x = _img()
    res = run_param_pgd(x, AdditiveParam(radius=0.02), _loss, steps=3, seed=0,
                        transform=make_fixed_jpeg_transform(50))
    assert float((res.x_def - jpeg_roundtrip(res.x_def, 50)).abs().max()) > 1e-4
