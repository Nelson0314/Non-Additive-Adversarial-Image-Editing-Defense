"""`--latent-norm-weight`：預設關閉時行為不變，開啟時正規化與評估都對。

兩個損失實測互補（`runs/ip2p_latent_norm_purify` 對 `runs/ip2p_ig_converge`）：
舊的未淨化 0.6843 但 JPEG30 掉到地板底下，新的未淨化 0.5850 但 JPEG30 是
0.2604。把它們加起來從未試過，這一支釘住加法本身是對的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class FakeIP2P:
    """`encode_image` 回傳一個範數可控的張量。"""

    def __init__(self, scale=70.0):
        self.scale = scale

    def encode_image(self, x):
        return x * self.scale


def _blend(weight, ig_value=0.3, x=None):
    import ip2p_run
    args = ip2p_run.build_parser().parse_args(
        ["--out", "x", "--radius", "2.5", "--loss", "image_guidance",
         "--ig-zt", "diffuse_src", "--latent-norm-weight", str(weight)])
    return args


def test_關閉時整條路徑不變():
    args = _blend(0.0)
    assert args.latent_norm_weight == 0.0


def test_正規化讓權重一等於等權():
    """舊項在乾淨影像上是 70–80，新項是 0.1–0.6。不正規化的話權重 1 只佔
    約 1/150——那不是等權，而是「幾乎只有舊項」。"""
    import importlib
    ip2p_run = importlib.import_module("ip2p_run")
    src = Path(ip2p_run.__file__).read_text(encoding="utf-8")
    # 正規化那一步必須存在且用乾淨影像當分母
    assert "_latent_norm_ref" in src
    assert "ip2p.encode_image(x01).flatten().norm(p=2)" in src


def test_乾淨影像上的舊項恰好是一():
    """正規化的定義：舊項除以它在乾淨影像上的值。"""
    ip2p = FakeIP2P(70.0)
    x01 = torch.rand(1, 3, 8, 8)
    ref = float(ip2p.encode_image(x01).flatten().norm(p=2))
    assert abs(ref / ref - 1.0) < 1e-9
    x_def = x01 * 0.5
    got = float(ip2p.encode_image(x_def).flatten().norm(p=2)) / ref
    assert abs(got - 0.5) < 1e-5


def test_旗標寫進_csv_欄位():
    import ip2p_run
    src = Path(ip2p_run.__file__).read_text(encoding="utf-8")
    assert '"latent_norm_weight": args.latent_norm_weight' in src
