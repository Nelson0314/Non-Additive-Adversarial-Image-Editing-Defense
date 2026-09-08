# -*- coding: utf-8 -*-
"""五個新機制裡**不需要載權重**就測得到的部分。

這些東西今晚要無人看顧地跑，而每一個都有會靜默失效的面：符號寫反只會讓曲線
往錯的方向收斂、processor 沒還原只會讓後續變慢、臉框算錯只會讓損失量到別的
區域。測試釘住的是這些，不是「函式有回傳值」。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.attention_loss import MAX_SIDE, CrossAttnShare
from src.defense.carrier_mask import ring_support
from src.defense.identity_loss import (_FACENET_MEAN, _FACENET_STD, _FACE_SIDE,
                                       face_box)


# ── 環帶 ────────────────────────────────────────────────────────────

def _face(h=256, w=256):
    m = torch.zeros(1, 1, h, w)
    m[..., 90:150, 100:160] = 1.0
    return m


@pytest.mark.parametrize("inner,outer", [(8, 24), (8, 40), (16, 64)])
def test_環帶不碰主體(inner, outer):
    """「主體那一側恆為 0」是這個威脅模型的硬約束。環帶緊貼著臉長，
    是最容易踩到它的支撐，故逐組合檢查。"""
    m = _face()
    r = ring_support(m, inner, outer)
    assert float((r * (m > 0.5)).max()) == 0.0


def test_環帶面積隨外緣單調增加():
    m = _face()
    areas = [float(ring_support(m, 8, o).mean()) for o in (16, 24, 40, 56)]
    assert areas == sorted(areas)


def test_環帶的內緣不可為零():
    """主體遮罩已經往外羽化過；inner=0 時支撐與羽化帶重疊，實際拿到的環帶
    比要求的窄，而報表上的 carrier_ring 欄仍寫著原本的數字。"""
    with pytest.raises(ValueError, match="inner"):
        ring_support(_face(), 0, 24)
    with pytest.raises(ValueError, match="inner"):
        ring_support(_face(), 30, 24)


def test_環帶是空遮罩時面積為零而不是全圖():
    """沒有臉時環帶應該是空的。回退成全圖會讓損失變成另一個目標。"""
    empty = torch.zeros(1, 1, 128, 128)
    assert float(ring_support(empty, 8, 24).sum()) == 0.0


# ── 身分損失的臉框 ──────────────────────────────────────────────────

def test_臉框是方形且涵蓋整張臉():
    """嵌入網路吃 160×160 的方形輸入。框若不是方形，縮放會把臉壓扁，
    而嵌入仍然算得出來、餘弦仍然有值。"""
    m = _face()
    top, left, sh, sw = face_box(m, margin=0.35)
    assert sh == sw
    idx = torch.nonzero(m[0, 0] > 0.5)
    y0, x0 = (int(v) for v in idx.min(dim=0).values)
    y1, x1 = (int(v) for v in idx.max(dim=0).values)
    assert top <= y0 and left <= x0
    assert top + sh >= y1 and left + sw >= x1


def test_臉框放寬之後仍在畫面內():
    """臉貼著畫面邊緣時，放寬的框會出界；出界的裁切會靜默變短而不拋錯。"""
    m = torch.zeros(1, 1, 128, 128)
    m[..., :30, :30] = 1.0
    top, left, sh, sw = face_box(m, margin=1.0)
    assert top >= 0 and left >= 0
    assert top + sh <= 128 and left + sw <= 128


def test_臉框的放寬比例真的有作用():
    m = _face()
    small = face_box(m, margin=0.0)[2]
    big = face_box(m, margin=0.5)[2]
    assert big > small


def test_空遮罩時臉框拋錯而不是回退到整張畫面():
    """回退會讓損失變成「整張圖像不像那個人」，與旗標名稱說的不是同一件事。"""
    with pytest.raises(ValueError, match="沒有可用的臉"):
        face_box(torch.zeros(1, 1, 64, 64))


def test_facenet的正規化常數與該套件一致():
    """`post_process=True` 做的是 `(x*255 − 127.5)/128`。從 [0,1] 出發就是
    `(x − 0.5)/(128/255)`。差一點不會拋錯，只會讓嵌入落在訓練分布之外
    而餘弦全部貼近零——那看起來像「防禦很成功」。"""
    assert _FACENET_MEAN == 0.5
    assert abs(_FACENET_STD - 128.0 / 255.0) < 1e-12
    assert _FACE_SIDE == 160
    x = torch.tensor([[0.0, 0.5, 1.0]])
    got = (x - _FACENET_MEAN) / _FACENET_STD
    want = (x * 255.0 - 127.5) / 128.0
    assert torch.allclose(got, want, atol=1e-5)


# ── 注意力項 ────────────────────────────────────────────────────────

class _FakeUNet:
    """只提供 `attn_processors` 與 `set_attn_processor` 的殼。"""

    def __init__(self):
        self.attn_processors = {"a": object()}
        self.set_calls = []

    def set_attn_processor(self, p):
        self.set_calls.append(p)
        self.attn_processors = p


def test_processor_一定會被還原():
    """不還原的話後續的編輯會繼續走自己算 softmax 的慢路徑，**輸出不變、
    只有速度變慢**——這是最難發現的一種失效。"""
    unet = _FakeUNet()
    saved = unet.attn_processors
    col = CrossAttnShare(unet, torch.ones(1, 1, 64, 64)).install()
    assert unet.attn_processors is not saved
    col.restore()
    assert unet.attn_processors is saved


def test_重複還原不會把別的東西塞回去():
    unet = _FakeUNet()
    saved = unet.attn_processors
    col = CrossAttnShare(unet, torch.ones(1, 1, 64, 64)).install()
    col.restore()
    col.restore()
    assert unet.attn_processors is saved


def test_只收粗解析度的層():
    """64×64 那一層的注意力矩陣是 4096×77×heads，全部收下來會讓一步 PGD 的
    峰值記憶體翻倍。**跳過要靜默但可驗證**，故直接檢查 shares 的長度。"""
    region = torch.ones(1, 1, 64, 64)
    col = CrossAttnShare(_FakeUNet(), region)
    fine = MAX_SIDE * 2
    col._accumulate(torch.rand(4, fine * fine, 77))
    assert col.shares == []
    col._accumulate(torch.rand(4, MAX_SIDE * MAX_SIDE, 77))
    assert len(col.shares) == 1


def test_落在區域上的比例是機率質量而不是原始總和():
    """三塊區域的比例要相加為 1，否則「有多少注意力落在補丁上」讀不成比例。"""
    side = 16
    col_all = CrossAttnShare(_FakeUNet(), torch.ones(1, 1, side, side))
    probs = torch.rand(2, side * side, 77)
    col_all._accumulate(probs)
    assert abs(float(col_all.shares[0]) - 1.0) < 1e-4

    half = torch.zeros(1, 1, side, side)
    half[..., : side // 2, :] = 1.0
    col_half = CrossAttnShare(_FakeUNet(), half)
    col_half._accumulate(probs)
    other = torch.zeros(1, 1, side, side)
    other[..., side // 2:, :] = 1.0
    col_other = CrossAttnShare(_FakeUNet(), other)
    col_other._accumulate(probs)
    assert abs(float(col_half.shares[0]) + float(col_other.shares[0]) - 1.0) < 1e-4


def test_沒有收到任何層時mean_share回None():
    """回 0 會讓注意力項安靜地失效，而 CSV 的 attn_weight 仍寫著非零值。
    呼叫端據此拋錯。"""
    col = CrossAttnShare(_FakeUNet(), torch.ones(1, 1, 32, 32))
    assert col.mean_share() is None


def test_注意力比例不可以退化成區域面積():
    """softmax 是對 token 維正規化的，用 `probs.sum(dim=-1)` 會讓每個位置恆等
    於 1，比例就變成**區域的面積**——與注意力無關，而數字看起來完全正常。

    實測踩過一次：三個臂的「注意力比例」分別是 0.2719、0.1339、0.7131，正好
    等於它們的支撐面積，原圖與防禦圖逐位元相同、四個時間步也相同。

    這一條的作法是造一個**注意力沿位置變化**的矩陣，且讓變化與區域相反，
    於是「量到的是注意力」與「量到的是面積」會給出不同的值。
    """
    side = 8
    hw = side * side
    region = torch.zeros(1, 1, side, side)
    region[..., : side // 2, :] = 1.0          # 面積剛好 0.5

    probs = torch.zeros(1, hw, 4)
    # 上半：幾乎全部注意力落在 BOS（＝不受文字驅動）；下半：完全不落在 BOS。
    top = torch.arange(hw) < hw // 2
    probs[0, top, 0] = 0.9
    probs[0, top, 1:] = 0.1 / 3
    probs[0, ~top, 0] = 0.0
    probs[0, ~top, 1:] = 1.0 / 3

    col = CrossAttnShare(_FakeUNet(), region)
    col._accumulate(probs)
    got = float(col.shares[0])
    assert abs(got - 0.5) > 0.2, (
        f"比例 {got:.4f} 太接近區域面積 0.5——很可能又退化成面積了")
    # 上半是「不受文字驅動」的一半，故比例應該遠低於 0.5。
    assert got < 0.2
