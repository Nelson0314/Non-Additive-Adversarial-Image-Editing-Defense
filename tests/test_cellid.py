"""設定雜湊：確認「改了設定卻沒改路徑」不可能靜默沿用舊結果。"""

import pytest

from src.utils.cellid import (REQUIRED_KEYS, ConfigIncomplete, canonical_json,
                              cell_id, config_hash)

BASE = {
    "spec_version": 1, "model": "SDXL-1.0-base", "resolution": 1024,
    "guidance": 7.5, "steps": 50, "strength": 0.6,
    "gpu": "Tesla V100-SXM2-32GB", "precision": "fp16",
    "condition": "N1", "loss_params": {"margin": 0.5, "lam_def": 1.0},
    "lr": 0.1, "tau": 0.20, "purify": None, "seed": 0, "image_id": "pie_0007",
}


def test_相同設定得相同雜湊():
    assert config_hash(BASE) == config_hash(dict(BASE))


def test_鍵順序不影響雜湊():
    shuffled = {k: BASE[k] for k in reversed(list(BASE))}
    assert config_hash(shuffled) == config_hash(BASE)


@pytest.mark.parametrize("key,new", [
    ("guidance", 1.0), ("precision", "bf16"), ("gpu", "RTX 5090"),
    ("tau", 0.35), ("lr", 0.008), ("seed", 1), ("spec_version", 2),
])
def test_任一變因改動即改變雜湊(key, new):
    assert config_hash(BASE | {key: new}) != config_hash(BASE)


def test_巢狀損失參數改動也改變雜湊():
    """loss_params 是字典，若只比對頂層鍵就會漏掉這一層。"""
    other = BASE | {"loss_params": {"margin": 0.9, "lam_def": 1.0}}
    assert config_hash(other) != config_hash(BASE)


@pytest.mark.parametrize("missing", REQUIRED_KEYS)
def test_缺任一必填鍵即拋出(missing):
    cfg = {k: v for k, v in BASE.items() if k != missing}
    with pytest.raises(ConfigIncomplete) as e:
        config_hash(cfg)
    assert missing in str(e.value)


def test_值為None與省略該鍵等價():
    """`{"purify": None}` 與省略指的是同一件事，不該產生兩個目錄。
    但必填檢查仍要求該鍵存在——強迫呼叫端對每個欄位表態。"""
    with_none = canonical_json(BASE)
    without = canonical_json({k: v for k, v in BASE.items() if v is not None})
    assert with_none == without


def test_布林不被當成整數():
    """bool 是 int 的子類別，處理順序寫錯會讓 True 變成 1。"""
    a = canonical_json(BASE | {"loss_params": {"flag": True}})
    b = canonical_json(BASE | {"loss_params": {"flag": 1}})
    assert a != b


def test_浮點以最短往返表示():
    """repr(0.1) 必須是 "0.1"，否則同一字面值在不同機器上可能得到不同字串。"""
    assert '"0.1"' in canonical_json(BASE | {"lr": 0.1})


def test_識別碼不含雜湊():
    """同一格改設定後應出現在同一個位置，由 meta.json 的雜湊指出它已過期，
    而不是散落成兩個目錄。"""
    cid = cell_id("eval", "N1", "pie_0007", tau=0.20, purify="jpeg30", seed=3)
    assert cid == "eval/N1/pie_0007/purifyjpeg30/seed3/tau0.2"
    assert config_hash(BASE) not in cid


def test_識別碼略過None欄位():
    assert cell_id("train", "N1", "pie_0007", tau=None) == "train/N1/pie_0007"
