"""`eot_broad`：兩層抽樣的機率、族的驗證、以及與既有兩支的區別。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.purify_aware import (
    BROAD_FRACTIONS, BROAD_SIGMAS, make_eot_broad_transform,
    make_eot_ops_transform,
)


def test_四類算子的機率各為四分之一():
    """類別要均勻，不可因為某一類的參數比較多就被放大。

    直接攤平成 (類別, 參數) 的清單會讓 identity 由 1/4 掉到 1/13，
    也就是把「未淨化時的效果」系統性地讓掉。
    """
    t = make_eot_broad_transform(seed=0)
    x = torch.rand(1, 3, 32, 32)
    n_identity = 0
    trials = 400
    for i in range(trials):
        if torch.equal(t(x, i), x):
            n_identity += 1
    assert 0.19 < n_identity / trials < 0.31


def test_模糊與裁切不是固定值():
    """`eot_ops` 的模糊恆為 sigma=1.0、裁切恆為 0.10，本支必須撐開。"""
    t = make_eot_broad_transform(qualities=(75,), seed=3)
    x = torch.rand(1, 3, 32, 32)
    outs = [t(x, i) for i in range(200)]
    uniq = {round(float((o - x).abs().mean()), 6) for o in outs}
    assert len(uniq) >= 6          # identity + 至少五種不同強度


def test_空的族一律報錯():
    for kw in ({"qualities": ()}, {"sigmas": ()}, {"fractions": ()}):
        with pytest.raises(ValueError):
            make_eot_broad_transform(**kw)


def test_不合法的裁切比例與_sigma_報錯():
    with pytest.raises(ValueError):
        make_eot_broad_transform(fractions=(0.6,))
    with pytest.raises(ValueError):
        make_eot_broad_transform(sigmas=(0.0,))


def test_同一顆種子可重現():
    x = torch.rand(1, 3, 32, 32)
    a = [make_eot_broad_transform(seed=7)(x, i) for i in range(20)]
    b = [make_eot_broad_transform(seed=7)(x, i) for i in range(20)]
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_預設族未被改動():
    assert BROAD_SIGMAS == (0.5, 1.0, 1.5, 2.0)
    assert BROAD_FRACTIONS == (0.05, 0.10, 0.15, 0.20)


def test_既有的_eot_ops_行為未變():
    """新增一支不得動到舊的那一支。"""
    x = torch.rand(1, 3, 32, 32)
    a = [make_eot_ops_transform((75,), seed=0)(x, i) for i in range(10)]
    b = [make_eot_ops_transform((75,), seed=0)(x, i) for i in range(10)]
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_拿掉裁切之後三類各佔三分之一():
    """裁切那兩欄結構性不可贏，把那四分之一的取樣預算還給其餘類別。"""
    from src.defense.purify_aware import BROAD_CLASSES
    t = make_eot_broad_transform(seed=0, classes=("identity", "jpeg", "blur"))
    x = torch.rand(1, 3, 32, 32)
    n_identity = sum(1 for i in range(600) if torch.equal(t(x, i), x))
    assert 0.28 < n_identity / 600 < 0.39
    assert BROAD_CLASSES == ("identity", "jpeg", "blur", "crop")


def test_classes_預設含全部四類且行為不變():
    x = torch.rand(1, 3, 32, 32)
    a = [make_eot_broad_transform(seed=11)(x, i) for i in range(30)]
    b = [make_eot_broad_transform(seed=11, classes=("identity", "jpeg", "blur",
                                                    "crop"))(x, i)
         for i in range(30)]
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_classes_必須含_identity():
    with pytest.raises(ValueError):
        make_eot_broad_transform(classes=("jpeg", "blur"))


def test_未知類別報錯而不是靜默略過():
    with pytest.raises(ValueError):
        make_eot_broad_transform(classes=("identity", "sharpen"))


def test_空的_classes_報錯():
    with pytest.raises(ValueError):
        make_eot_broad_transform(classes=())
