"""    CLIP-S = cos( E_img(編輯) − E_img(原圖), E_txt(指令) )

三件要釘住的事：

本檔需要 CLIP 與 SigLIP 的權重；以 weights 標記明確選取；依賴或執行錯誤直接失敗。
"""

import pytest
import torch

from immunization_core.metrics.suite import MetricSuite


@pytest.fixture(scope="module")
def suite():
    s = MetricSuite(device=torch.device("cpu"))
    s._ensure_vlm()
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
    """`edit == src` 時 delta 為零向量，餘弦沒有定義。"""
    x = _img(3)
    out = suite.direction_similarity(x, x, "turn the shirt red")
    assert abs(out["clip"]) < 1e-5


def test_指令一致時分數較高(suite):
    """    構造：由一張紅色圖與一張藍色圖組出兩個位移，對「turn it red」這句指令

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

pytestmark = pytest.mark.weights
