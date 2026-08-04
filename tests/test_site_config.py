"""釘住「預設值不是本專案要的，而呼叫端沒有覆寫」這一類的缺陷。

這一類已經踩過六次，型態完全相同：一個為了讓舊 run 可重現而保留的預設值
被新的驅動沿用，實驗照常跑完、CSV 照常寫出、沒有任何症狀，只有結果不是
在回答原本要問的問題。已知的六個：

    beta_linf = 100.0        LEDGER 6.12   → L∞ 而非 LPIPS 成為綁定約束
    alpha_lpips = 1.0        LEDGER 6.12   → 預算永遠用不滿，τ 不是匹配軸
    plateau_stop 不檢查可行  LEDGER 6.13   → 停在超標的失真上
    lr = 0.03 用在 site PF   LEDGER 6.14   → 全秩震盪不收斂
    一個 lr 打全部 site      LEDGER 6.16   → site S 跑在校準值的 1/12.5
    τ_acut／τ_chroma 不隨預算 LEDGER 6.17  → 副約束變成真正的有效約束

前四個在別處已有測試（test_objective.py、test_plateau_stop.py）。本檔
釘住後兩個，並釘住 `run_defense.load_images` 的資料集格式辨認。
"""

import pytest


# ---------------------------------------------------------------------------
# 逐 site 的學習率（LEDGER 6.16）
# ---------------------------------------------------------------------------


def test_每個site的學習率取自己校準過的值():
    from scripts.run_ours_lo_eval import SITE_LR

    # E14（docs/RESULTS_E13-E23.md §2）在同一判準下掃出來的兩個值。
    # 相差 12.5 倍——這正是「不可共用一個 lr」的量化理由。
    assert SITE_LR["S"] == 0.1
    assert SITE_LR["P"] == 0.008
    assert SITE_LR["S"] / SITE_LR["P"] == pytest.approx(12.5)
    # PF 沿用唯一跑過全秩的 e8；0.03 是 LEDGER 6.14 判定會震盪的那個值
    assert SITE_LR["PF"] == 0.008


def test_未指定lr時逐site取校準值():
    from scripts.run_ours_lo_eval import SITE_LR, parse_lr

    assert parse_lr("PF=0.008") == {**SITE_LR, "PF": 0.008}
    # 只覆寫其中一個，其餘維持各自的校準值——不會被連坐改掉
    got = parse_lr("S=0.3")
    assert got["S"] == 0.3
    assert got["PF"] == SITE_LR["PF"]


def test_單一數值會套用到全部site():
    from scripts.run_ours_lo_eval import SITE_LR, parse_lr

    # 允許，但必須是呼叫端明寫的。預設值不是它。
    assert parse_lr("0.02") == {s: 0.02 for s in SITE_LR}


def test_未知的site必須報錯():
    from scripts.run_ours_lo_eval import parse_lr

    with pytest.raises(ValueError, match="未知的 site"):
        parse_lr("Z=0.1")


# ---------------------------------------------------------------------------
# 逐預算的兩道副門檻（LEDGER 6.17）
# ---------------------------------------------------------------------------


def test_門檻隨預算而變且大於_LossConfig_的預設值():
    from pathlib import Path

    from src.defense.objective import LossConfig
    from scripts.run_ours_lo_eval import budget_thresholds

    root = Path(__file__).resolve().parents[1]

    # τ_lpips = 0.05 是這兩個預設值當初判讀定出的那個預算，故應與預設值一致
    a05, c05 = budget_thresholds(0.05, root)
    assert a05 == pytest.approx(LossConfig.tau_acut)
    assert c05 == pytest.approx(LossConfig.tau_chroma)

    # runs/ours_lo/ 實際跑的預算。門檻必須比預設值鬆，否則副約束會變成
    # 真正的有效約束——實測 site PF 的 man_00 有 31/48 步被鈍化 hinge 綁住
    a10, c10 = budget_thresholds(0.10, root)
    assert a10 > LossConfig.tau_acut
    assert c10 > LossConfig.tau_chroma
    assert a10 == pytest.approx(0.0598)
    assert c10 == pytest.approx(1.2965)


def test_未列表的預算必須報錯而不是內插():
    from pathlib import Path

    from scripts.run_ours_lo_eval import budget_thresholds

    root = Path(__file__).resolve().parents[1]
    # 0.075 落在 0.05 與 0.10 之間。內插沒有依據：p14 已量到 acut 軸的
    # 分離度隨預算塌掉（0.05 時 5.12 倍、0.28 時 1.39 倍）。
    with pytest.raises(KeyError, match="不在"):
        budget_thresholds(0.075, root)


# ---------------------------------------------------------------------------
# 資料集格式的辨認（run_defense.load_images）
# ---------------------------------------------------------------------------


def test_兩種prompts格式都要讀對():
    from pathlib import Path

    from scripts.run_defense import class_prompts

    src = Path("prompts.yaml")
    # data/dayn_testset 的格式
    assert class_prompts(["a cat", "a dog"], "dog", src) == ["a cat", "a dog"]
    # data/lo_aligned 的格式。原本的 list(dict) 會回傳 ['content', 'prompts']
    entry = {"content": "dog", "prompts": ["a cat", "a dog in the park"]}
    assert class_prompts(entry, "dog", src) == ["a cat", "a dog in the park"]


def test_dict少了prompts鍵必須報錯():
    from pathlib import Path

    from scripts.run_defense import class_prompts

    with pytest.raises(KeyError, match="prompts"):
        class_prompts({"content": "dog"}, "dog", Path("prompts.yaml"))


def test_不認識的格式必須報錯而不是退回預設prompt():
    from pathlib import Path

    from scripts.run_defense import class_prompts

    # 靜默退回 ["a photo"] 與「宣告了 prompt 但沒被讀到」在外部分不出來
    with pytest.raises(TypeError):
        class_prompts("a cat", "dog", Path("prompts.yaml"))


def test_load_images不把根目錄的png當成資料集影像(tmp_path):
    import torch

    from scripts.run_defense import load_images

    (tmp_path / "dog").mkdir()
    _write_png(tmp_path / "dog" / "dog_00.png")
    # data/lo_aligned/overview.png 就是這樣被 rglob 掃進來的
    _write_png(tmp_path / "overview.png")
    (tmp_path / "prompts.yaml").write_text(
        "dog:\n  content: dog\n  prompts:\n    - a cat\n", encoding="utf-8")

    got = load_images(tmp_path, 32, torch.device("cpu"))
    assert [n for n, _, _ in got] == ["dog_00"]
    assert got[0][2] == ["a cat"]


def test_未宣告的類別目錄必須報錯(tmp_path):
    import torch

    from scripts.run_defense import load_images

    (tmp_path / "dog").mkdir()
    _write_png(tmp_path / "dog" / "dog_00.png")
    (tmp_path / "cat").mkdir()
    _write_png(tmp_path / "cat" / "cat_00.png")
    (tmp_path / "prompts.yaml").write_text(
        "dog:\n  - a cat\n", encoding="utf-8")

    with pytest.raises(KeyError, match="cat"):
        load_images(tmp_path, 32, torch.device("cpu"))


def test_缺少prompts_yaml必須報錯(tmp_path):
    import torch

    from scripts.run_defense import load_images

    (tmp_path / "dog").mkdir()
    _write_png(tmp_path / "dog" / "dog_00.png")
    with pytest.raises(FileNotFoundError, match="prompts.yaml"):
        load_images(tmp_path, 32, torch.device("cpu"))


# ---------------------------------------------------------------------------
# CSV 附加時的欄位比對（證據完整性）
# ---------------------------------------------------------------------------


def test_附加到欄位不同的CSV必須拒絕(tmp_path):
    """`csv.DictWriter` 在這種情況下不會報錯，只會把資料寫錯欄。

    `runs/` 是唯一的證據來源且實驗無法重跑，一次靜默錯位就毀掉整批。
    """
    from scripts.gate_suppress import append_csv

    p = tmp_path / "summary.csv"
    append_csv(p, [{"a": 1, "b": 2}])
    with pytest.raises(SystemExit, match="表頭與要寫入的欄位不符"):
        append_csv(p, [{"a": 1, "b": 2, "c": 3}])
    # 欄位相同但順序不同同樣要擋：順序決定資料落在哪一欄
    with pytest.raises(SystemExit, match="表頭與要寫入的欄位不符"):
        append_csv(p, [{"b": 2, "a": 1}])


def test_欄位相同時可以正常附加(tmp_path):
    import csv as _csv

    from scripts.gate_suppress import append_csv

    p = tmp_path / "summary.csv"
    append_csv(p, [{"a": 1, "b": 2}])
    append_csv(p, [{"a": 3, "b": 4}])
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def _write_png(path):
    from PIL import Image

    Image.new("RGB", (16, 16), (128, 64, 32)).save(path)


# ---------------------------------------------------------------------------
# 沿射線縮放到指定的感知失真（src/metrics/ray_scale.py）
# ---------------------------------------------------------------------------


def test_縮放到指定的LPIPS():
    import torch

    from src.metrics.ray_scale import scale_to_lpips

    x = torch.zeros(1, 3, 16, 16)
    # 單調且可解析：fn(x + k·d) = k，其中 d 是全 0.1 的常數場
    fn = lambda t: float((t - x).abs().mean() * 10)          # noqa: E731
    d = torch.ones_like(x) * 0.1
    _, got, k = scale_to_lpips(fn, x, d, 0.5)
    assert got == pytest.approx(0.5, abs=5e-3)
    assert k == pytest.approx(0.5, abs=5e-3)


def test_達不到目標時必須拋出而不是取最接近的值():
    """「達不到 0.5」與「達到了 0.5」在下游是完全不同的兩件事。"""
    import torch

    from src.metrics.ray_scale import scale_to_lpips

    x = torch.zeros(1, 3, 16, 16)
    fn = lambda t: float((t - x).abs().mean() * 10)          # noqa: E731
    with pytest.raises(ValueError, match="達不到目標"):
        scale_to_lpips(fn, x, torch.ones_like(x) * 0.1, 100.0, k_max=8)


def test_隨機對照條件會被縮放到指定的失真():
    import torch

    from src.metrics.ray_scale import gaussian_control

    x = torch.zeros(1, 3, 16, 16)
    fn = lambda t: float((t - x).abs().mean() * 10)          # noqa: E731
    _, got, _ = gaussian_control(fn, x, 0.3, seed=1)
    assert got == pytest.approx(0.3, abs=5e-3)


def test_三個腳本共用同一個實作():
    """2026-08-04 合併前有三份幾乎相同的二分搜尋。"""
    import inspect

    from scripts.gate_suppress import random_control
    from src.metrics import ray_scale

    src = inspect.getsource(random_control)
    assert "gaussian_control" in src          # 已改為薄包裝
    assert hasattr(ray_scale, "scale_to_lpips")
    assert hasattr(ray_scale, "gaussian_control")
