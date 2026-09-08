"""`--ig-weight` 的三件事：預設不改變行為、設定進得了 CSV、用錯就拋錯。

存在理由
────────────────────────────────────────────────────────────────────
這個旗標的失效方式全部是靜默的：

1. 預設值若不是逐位元等同原本的 `.mean()`，所有既有批次的 `best_eval` 與
   `trace.csv` 立刻不可與新批次比較，而曲線看起來還是曲線。
2. 設定若只活在 argparse 的預設值裡，合併分片之後兩組不同設定的列長得一模
   一樣（本專案已經因為 `quantile`／`hop`／`defense_steps` 犯過一次）。
3. `--ig-weight subject` 配到別的損失上時若靜默忽略，報表會寫著 subject
   而實際跑的是 uniform。

第四件是它與 `--subject-mask` 的分工：那一個限制擾動長在哪裡，這一個只改
損失的權重。兩者共用 CLIPSeg 的形狀參數但**是不同的旗標**，混為一談的症狀
是「以為擾動被限制住了，其實沒有」。

只用 CPU，不載任何權重。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ip2p_run  # noqa: E402
from src.defense.image_guidance_loss import _weighted_mean  # noqa: E402


def _args(*extra):
    return ip2p_run.build_parser().parse_args(
        ["--out", "o", "--images", "img", *extra])


def test_預設是uniform且不需要目錄():
    a = _args()
    assert a.ig_weight == "uniform"


def test_不加權時就是原本的mean():
    """歸約只有一個實作，`weight=None` 必須逐位元等同 `.mean()`。"""
    g = torch.Generator().manual_seed(4)
    sq = torch.rand(1, 4, 8, 8, generator=g)
    assert torch.equal(_weighted_mean(sq, None), sq.mean())


def test_全一的權重等於不加權():
    g = torch.Generator().manual_seed(5)
    sq = torch.rand(1, 4, 8, 8, generator=g)
    got = _weighted_mean(sq, torch.ones(1, 1, 32, 32))
    assert float(got) == pytest.approx(float(sq.mean()), rel=1e-6)


def test_權重乘上常數不改變結果():
    g = torch.Generator().manual_seed(6)
    sq = torch.rand(1, 4, 8, 8, generator=g)
    w = torch.zeros(1, 1, 32, 32)
    w[..., :16, :] = 1.0
    assert float(_weighted_mean(sq, w)) == pytest.approx(
        float(_weighted_mean(sq, w * 0.3)), rel=1e-6)


def test_空權重拋錯而不是回傳零():
    sq = torch.rand(1, 4, 8, 8)
    with pytest.raises(ValueError, match="總和為 0"):
        _weighted_mean(sq, torch.zeros(1, 1, 32, 32))


def test_配到別的損失上就拋錯而不是靜默忽略():
    a = _args("--loss", "latent_norm", "--ig-weight", "subject")
    with pytest.raises(SystemExit, match="只能配 --loss image_guidance"):
        ip2p_run.validate_loss_args(a)


def test_uniform配別的損失不受影響():
    a = _args("--loss", "latent_norm")
    ip2p_run.validate_loss_args(a)          # 不應拋錯


def test_目錄不存在就擋在載入權重之前(tmp_path):
    a = _args("--loss", "image_guidance", "--ig-zt", "diffuse_src",
              "--ig-weight", "subject",
              "--ig-weight-catalogue", str(tmp_path / "沒有這個檔.yaml"))
    with pytest.raises(SystemExit, match="主體名詞目錄"):
        ip2p_run.validate_loss_args(a)


def test_兩個欄位逐列寫出():
    """只活在 argparse 預設值裡的設定，合併分片之後分不出來。"""
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert '"ig_weight": args.ig_weight,' in src
    assert '"ig_weight_text": getattr(args, "_ig_weight_text", ""),' in src


def test_跨批分析的遷移表登記了新欄位():
    """未登記的欄位會讓 `distortion_axis_analysis` 直接拋錯。"""
    import distortion_axis_analysis as daa
    assert daa.SETTING_DEFAULTS["ig_weight"] == "uniform"
    assert daa.SETTING_DEFAULTS["ig_weight_text"] == ""


def test_與subject_mask是兩個不同的旗標():
    a = _args("--loss", "image_guidance", "--ig-zt", "diffuse_src",
              "--ig-weight", "subject")
    assert a.subject_mask is None          # 沒有被順便打開
    assert a.ig_weight == "subject"
