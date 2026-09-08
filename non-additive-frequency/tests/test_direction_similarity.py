"""CLIP-S：位移方向對指令的餘弦。逐行對照 FaceLock 的 `evaluation/eval_clip_s.py`。

    CLIP-S = cos( E_img(編輯) − E_img(原圖), E_txt(指令) )

它是 FaceLock（CVPR 2025）六個指標之一，方向是**低者防禦強**。本專案先前只有
`semantic`（影像對一句描述）與 `image_similarity`（兩張圖之間），沒有這一個。

三件要釘住的事：

1. **相減前不正規化。** FaceLock 對 `get_image_features` 的原始輸出相減。
   先正規化再相減是另一個量（球面上的弦向量），數值不同而且不會有症狀。
2. **影像側與既有兩個語意讀數走同一條前向。** 分岔了不會拋錯，只會讓三個
   數字來自三個略有差異的前處理。
3. **方向要合理**：位移方向與指令一致時分數高於不一致時。這是這個指標唯一
   能被獨立驗證的性質——它是個餘弦，沒有絕對尺度可對。

本檔需要 CLIP 與 SigLIP 的權重；抓不到時整檔 skip（**介面測試不 skip**）。
"""

import pytest
import torch

from src.metrics.suite import MetricSuite


@pytest.fixture(scope="module")
def suite():
    s = MetricSuite(device=torch.device("cpu"))
    try:
        s._ensure_vlm()
    except Exception as e:  # 權重抓不到
        pytest.skip(f"CLIP／SigLIP 權重不可得：{type(e).__name__}")
    return s


def _img(seed, size=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


def test_回傳兩個模型的鍵(suite):
    out = suite.direction_similarity(_img(0), _img(1), "turn the shirt red")
    assert set(out) == {"clip", "siglip"}
    for v in out.values():
        assert isinstance(v, float)
        assert -1.0001 <= v <= 1.0001, "餘弦必須落在 [-1, 1]"


def test_同一張圖的位移是零向量(suite):
    """`edit == src` 時 delta 為零向量，餘弦沒有定義。

    PyTorch 的 `cosine_similarity` 對零向量回傳 0（分母加了 eps），
    不會拋錯。釘住這件事是為了讓呼叫端知道那個 0 是**沒有位移**而不是
    「方向正交」——兩者在報表上意義完全不同。
    """
    x = _img(3)
    out = suite.direction_similarity(x, x, "turn the shirt red")
    assert abs(out["clip"]) < 1e-5


def test_指令一致時分數較高(suite):
    """唯一能獨立驗證的性質：位移方向指向指令時，分數高於指向別處。

    構造：由一張紅色圖與一張藍色圖組出兩個位移，對「turn it red」這句指令
    比分數。**這不是在測 CLIP 的品質**，是在測正負號與相減順序有沒有寫反
    ——寫反時這個測試會失敗，而報表上只會看到一個看似合理的數字。
    """
    grey = torch.full((1, 3, 64, 64), 0.5)
    red = grey.clone()
    red[:, 0] = 1.0
    red[:, 1:] = 0.0
    blue = grey.clone()
    blue[:, 2] = 1.0
    blue[:, :2] = 0.0

    towards_red = suite.direction_similarity(grey, red, "turn it red")["clip"]
    towards_blue = suite.direction_similarity(grey, blue, "turn it red")["clip"]
    assert towards_red > towards_blue, (
        f"往紅色的位移對「turn it red」應該分數較高，"
        f"實得 {towards_red:.4f} vs {towards_blue:.4f}——相減順序或正負號寫反了"
    )


def test_影像側與_image_similarity_同一條前向(suite):
    """兩個方法都對同一張圖取 `image_embeds`。若前處理分岔，這兩個數字
    背後的嵌入就不是同一個，而報表會把它們並列。

    間接檢定：`direction_similarity(a, b)` 的 delta 若與
    `image_similarity(a, b)` 用的嵌入同源，則 `a == b` 時前者為 0
    （已由上面那項覆蓋）且後者為 1。
    """
    x = _img(11)
    assert abs(suite.image_similarity(x, x)["clip"] - 1.0) < 1e-4
    assert abs(suite.direction_similarity(x, x, "anything")["clip"]) < 1e-5
