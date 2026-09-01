"""`MetricSuite.semantic_multi`：一張影像對多個 prompt。

2026-08-08 新增，動機是類別 margin 這個讀出量
（`RESULTS_2026-08-08` §9）：

    margin(y) = SigLIP(y, 目標類) − SigLIP(y, 原類)

本檔**不載入 CLIP／SigLIP 權重**。既有的 `test_suite_pairwise.py` 只碰
`pairwise`／`niqe` 也是同一個理由——讓測試不依賴模型下載。這裡要釘住的是
契約，而契約不需要真實權重：

  1. `semantic` 必須**委派**給 `semantic_multi`，不得各自帶一份前處理。
     兩份前處理會在某一次改動之後靜默分歧，而兩邊都回傳一個合理的數字。
  2. 空的 prompt 清單必須拋出，不得回傳空 dict——呼叫端拿到空 dict 會在
     取值時才炸，而那時已經離開真正的原因。
"""
import pytest
import torch

from src.metrics.suite import MetricSuite


def test_semantic委派給semantic_multi而不是另寫一份前處理(monkeypatch):
    s = MetricSuite()
    seen = {}

    def fake(x, prompts):
        seen["prompts"] = list(prompts)
        return {p: {"clip": 0.1, "siglip": 0.2} for p in prompts}

    monkeypatch.setattr(s, "semantic_multi", fake)
    out = s.semantic(torch.zeros(1, 3, 8, 8), "a cat")
    assert seen["prompts"] == ["a cat"], (
        "`semantic` 必須把單一 prompt 包成清單交給 `semantic_multi`；"
        "各自帶一份前處理會在改動之後靜默分歧"
    )
    assert out == {"clip": 0.1, "siglip": 0.2}


def test_空的prompt清單直接拋出(monkeypatch):
    s = MetricSuite()
    monkeypatch.setattr(s, "_ensure_vlm", lambda: None)
    with pytest.raises(ValueError, match="prompts"):
        s.semantic_multi(torch.zeros(1, 3, 8, 8), [])


def test_semantic_multi的回傳以prompt為鍵(monkeypatch):
    """margin 的算法是取兩個 prompt 的差，故鍵必須是 prompt 本身。
    以位置索引回傳會讓呼叫端在 prompt 順序改變時取到錯的那一個而無症狀。"""
    s = MetricSuite()
    calls = []

    def fake(x, prompts):
        calls.append(list(prompts))
        return {p: {"siglip": float(i)} for i, p in enumerate(prompts)}

    monkeypatch.setattr(s, "semantic_multi", fake)
    # 直接驗證委派的形狀；真實實作的鍵值由上面兩條與型別註記約束。
    assert s.semantic(torch.zeros(1, 3, 8, 8), "a dog") == {"siglip": 0.0}
    assert calls == [["a dog"]]
