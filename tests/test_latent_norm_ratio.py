"""防禦把影像 latent 推了多遠。

存在理由
────────────────────────────────────────────────────────────────────
`acutance_ratio` 在使用者確認的判定上把擋下與失敗完全分開（0.704 對 0.119）：
**失敗的格子就是被糊掉的格子**。而 `latent_norm` 損失壓的正是 `‖E(x')‖`——
壓到夠低時 UNet 的影像條件失去資訊、IP2P 退化成純文生圖（重畫）；壓不夠時
模型仍跟著一個劣化的 latent 走（變糊）。

若逐圖的範數比值能預測擋下，它就是比 DISTS 好的**逐圖停止準則**。逐圖對齊
DISTS 已經測過並失敗（擋下率由 6/7 掉到 3/7），原因正是對齊了錯的量。

本模組只測欄位與比值的定義，不載 IP2P 權重（那需要 GPU）。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import latent_norm_ratio as lnr  # noqa: E402


def test_ratios_are_defined_against_the_original_latent():
    z0 = torch.tensor([[3.0, 4.0]])          # 範數 5
    z1 = torch.tensor([[0.0, 1.0]])          # 範數 1，差 (3,3) 範數 sqrt(18)
    out = lnr.ratios(z0, z1)
    assert abs(out["norm_ratio"] - 0.2) < 1e-6
    assert abs(out["move_ratio"] - (18 ** 0.5) / 5.0) < 1e-6


def test_zero_reference_norm_raises_rather_than_dividing():
    """原圖 latent 範數為零是不可能的；真的發生代表編碼器壞了，
    回傳 inf 會讓那一列看起來只是個極端值。"""
    try:
        lnr.ratios(torch.zeros(1, 2), torch.ones(1, 2))
    except ValueError as e:
        assert "範數" in str(e)
    else:
        raise AssertionError("沒有拋錯")


def test_required_columns_are_written_literally():
    src = (ROOT / "scripts" / "latent_norm_ratio.py").read_text(encoding="utf-8")
    for col in ("norm_ratio", "move_ratio", "norm_def", "norm_orig"):
        assert f'"{col}":' in src, col


def test_cli_requires_a_source_and_an_output():
    ns = lnr.build_parser().parse_args(["--src", "a", "--out", "b.csv"])
    assert ns.src.name == "a" and ns.out.name == "b.csv"
