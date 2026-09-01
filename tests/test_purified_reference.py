"""幾何類淨化算子換參照，其餘算子不換（`docs/DECISIONS.md`）。

    effect(p) = LPIPS( 編輯(p(原圖)), 編輯(p(防禦圖)) )    幾何類
    effect(p) = LPIPS( 編輯(原圖),    編輯(p(防禦圖)) )    其餘

兩件事需要被釘住，而且都不會有症狀：

1. **非幾何類必須逐位元不變。** 既有的 `runs/` 全部是那個參照量的，換掉之後
   數字仍然長得很正常，只是不再可比。
2. **幾何類的參照不可跨算子共用。** `編輯(p(原圖))` 隨 p 而變；快取鍵少了算子
   標籤，第二個幾何算子會拿到第一個算子的取景當參照，而表上看不出來。

本檔不載入任何擴散模型權重：`SDWrapper` 與 `MetricSuite` 都換成確定性的替身，
淨化算子本身用真的（那正是要判定的東西）。
"""

import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import phase_retention  # noqa: E402
from apa_baseline import EDIT_SEED  # noqa: E402
from src.purify.ops import (  # noqa: E402
    GEOMETRIC_KINDS, kind_of_label, label_is_geometric,
)

SIZE = 32
NON_GEOMETRIC = ["identity", "blur1", "jpeg75", "noise0.05", "quantize16"]
GEOMETRIC = ["crop_resize0.1", "crop_resize0.15", "jpeg_then_resize75"]
SEEDS = 2


# --------------------------------------------------------------- 集合本身

def test_幾何類就是改變格點或取景的那五個():
    assert GEOMETRIC_KINDS == {
        "crop_resize", "resample_roundtrip", "resize_only", "shift_only",
        "jpeg_then_resize"}


@pytest.mark.parametrize("kind", [
    "identity", "blur", "noise", "jpeg", "quantize", "adverse_cleaner",
    "impress", "diffpure", "gridpure", "fdpure", "cnn_denoise_substitute"])
def test_其餘算子一律不在幾何類裡(kind):
    """它們的地板反映的是算子對影像內容的破壞，那正是要扣掉的東西。"""
    assert kind not in GEOMETRIC_KINDS


def test_標籤還原不會把_jpeg_then_resize_當成_jpeg():
    """字首比對會把 `jpeg_then_resize75` 讀成 `jpeg`，於是幾何欄被當成
    非幾何欄，而數字看起來一切正常。"""
    assert kind_of_label("jpeg_then_resize75") == "jpeg_then_resize"
    assert kind_of_label("jpeg75") == "jpeg"
    assert label_is_geometric("jpeg_then_resize75") is True
    assert label_is_geometric("jpeg75") is False


def test_標籤還原帶強度與不帶強度都認得():
    assert kind_of_label("identity") == "identity"
    assert kind_of_label("crop_resize0.1") == "crop_resize"
    assert kind_of_label("shift_only51") == "shift_only"
    assert kind_of_label("resize_only") == "resize_only"


def test_不認得的標籤直接拋錯():
    with pytest.raises(ValueError, match="無法由標籤"):
        kind_of_label("crop_resize_0.1")


# ------------------------------------------------------- 跑一遍 main() 的替身

def edit_of(x: torch.Tensor, seed: int) -> torch.Tensor:
    """替身編輯的閉式解：輸出只由輸入與 seed 決定，同輸入同 seed 位元相同。

    偏移取 `seed % 1000` 而不是 `seed` 本身。`EDIT_SEED` 是 20260812，float32
    在那個量級的 ulp 已經是 2，直接加上去會把整張圖壓成同一個數，於是每一個
    差都變成 0，這一組測試就永遠是綠的。
    """
    return x + 0.001 * (seed % 1000)


class FakeSD:
    """確定性的編輯：輸出只由輸入與 seed 決定，故同輸入同 seed 位元相同。"""

    device = "cpu"

    def __init__(self, *args, **kwargs):
        self.edits = []          # (輸入張量, seed)

    def encode_image(self, x):
        return x

    def sample_edit_noise(self, z_like, seed):
        # float64。`EDIT_SEED` 是 20260812，已經超過 float32 能逐一表示整數的
        # 範圍（2**24），在 float32 上 20260812 與 20260813 會塌成同一個數，
        # 於是兩顆種子的替身編輯變得無法分辨。
        return torch.full((1,), float(seed), dtype=torch.float64)

    def encode_text(self, prompt):
        return prompt

    def uncond_prompt(self):
        return None

    def sdedit(self, x01, emb, noise, steps, strength=None,
               guidance_scale=None, emb_uncond=None, keep01=None):
        seed = int(noise[0])
        self.edits.append((x01.clone(), seed))
        return edit_of(x01, seed)


class DriftingSD(FakeSD):
    """編輯**不是**確定性的：每呼叫一次就漂一點。用來檢查地板的守門會拋錯。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def sdedit(self, x01, emb, noise, steps, **kwargs):
        self.calls += 1
        return super().sdedit(x01, emb, noise, steps, **kwargs) + 0.01 * self.calls


class FakeSuite:
    def __init__(self, device=None):
        self.pairs = []          # (參照, 待評)
        self.sims = []

    def pairwise(self, a, b):
        self.pairs.append((a.clone(), b.clone()))
        return {"lpips": float((a - b).abs().mean())}

    def image_similarity(self, a, b):
        self.sims.append((a.clone(), b.clone()))
        return {"siglip": 0.5}


def _pattern(seed: int) -> torch.Tensor:
    """非常數、非線性的圖。常數圖過 `crop_resize` 之後不變、線性斜坡過高斯
    模糊之後幾乎不變，兩種參照都會意外相等，這一組測試就什麼也證明不了。"""
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, SIZE, SIZE), generator=g)


def _run(tmp_path, monkeypatch, purifiers, floor=False, sd_cls=FakeSD,
         geometric_extra=()):
    """跑一次 `phase_retention.main()`，回傳 (寫出的列, FakeSD, FakeSuite)。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "condition", "budget_target"])
        w.writeheader()
        w.writerow({"image": "img_a", "condition": "cond_x", "budget_target": ""})
    (run_dir / "img_a__cond_x__def.png").write_bytes(b"")

    x01, x_def = _pattern(11), _pattern(22)
    sd_holder, suite_holder, rows_holder = [], [], []

    def fake_load_image_tensor(path, device, size=None):
        return x_def.clone() if str(path).endswith("__def.png") else x01.clone()

    def make_sd(*a, **k):
        sd = sd_cls()
        sd_holder.append(sd)
        return sd

    def make_suite(*a, **k):
        suite = FakeSuite()
        suite_holder.append(suite)
        return suite

    monkeypatch.setattr(phase_retention, "SDWrapper", make_sd)
    monkeypatch.setattr(phase_retention, "MetricSuite", make_suite)
    monkeypatch.setattr(phase_retention, "load_image_tensor",
                        fake_load_image_tensor)
    monkeypatch.setattr(phase_retention, "load_dataset",
                        lambda p: [{"name": "img_a", "path": "orig.png",
                                    "prompt": "make it red"}])
    monkeypatch.setattr(phase_retention, "head_keep", lambda item, x01: None)
    monkeypatch.setattr(phase_retention, "write_csv",
                        lambda out, rows: rows_holder.append(list(rows)))
    if geometric_extra:
        monkeypatch.setattr(phase_retention, "GEOMETRIC_KINDS",
                            GEOMETRIC_KINDS | set(geometric_extra))

    argv = ["phase_retention.py", "--run", str(run_dir), "--data", str(tmp_path),
            "--seeds", str(SEEDS), "--purifiers", *purifiers,
            "--out", str(tmp_path / "out.csv")]
    if floor:
        argv.append("--floor")
    monkeypatch.setattr(sys, "argv", argv)
    phase_retention.main()
    return rows_holder[-1], sd_holder[0], suite_holder[0], x01, x_def


# ------------------------------------------------------------- 非幾何類不變

def test_非幾何類的參照仍是編輯原圖(tmp_path, monkeypatch):
    """`a` 這一側必須逐位元等於 `編輯(原圖, seed)`，也就是 `x01 + seed`。"""
    rows, sd, suite, x01, _ = _run(tmp_path, monkeypatch, NON_GEOMETRIC)
    expect = [edit_of(x01, EDIT_SEED + k) for k in range(SEEDS)]
    assert len(suite.pairs) == len(NON_GEOMETRIC) * SEEDS
    for i, (a, _b) in enumerate(suite.pairs):
        assert any(torch.equal(a, e) for e in expect), i


def test_非幾何類的參照快取以影像與種子為鍵(tmp_path, monkeypatch):
    """五個算子共用同一份 `編輯(原圖)`，故參照那一側只編輯 `|種子|` 次。
    總次數 = |種子| × (1 + |算子|)，與換參照之前相同。"""
    rows, sd, suite, x01, _ = _run(tmp_path, monkeypatch, NON_GEOMETRIC)
    assert len(sd.edits) == SEEDS * (1 + len(NON_GEOMETRIC))
    on_orig = [e for e in sd.edits if torch.equal(e[0], x01)]
    assert len(on_orig) == SEEDS


def test_非幾何類的_reference_欄是_orig(tmp_path, monkeypatch):
    rows, *_ = _run(tmp_path, monkeypatch, NON_GEOMETRIC)
    assert {r["purifier"]: r["reference"] for r in rows} == \
        {p: "orig" for p in NON_GEOMETRIC}


# --------------------------------------------------------------- 幾何類換參照

def test_幾何類的參照是編輯淨化後的原圖(tmp_path, monkeypatch):
    from src.purify import ops as purify_ops

    rows, sd, suite, x01, _ = _run(
        tmp_path, monkeypatch, ["identity", "crop_resize0.1"])
    want = purify_ops.Purifier("crop_resize", 0.1).evaluate(x01)
    refs = [a for a, _ in suite.pairs[-SEEDS:]]
    assert all(torch.equal(a, edit_of(want, EDIT_SEED + k))
               for k, a in enumerate(refs))


def test_幾何類的快取鍵含算子標籤(tmp_path, monkeypatch):
    """三個幾何算子各要一份自己的參照。少了標籤這一段，後兩個會拿第一個的
    取景當參照——數字照樣寫得出來，只是錯的。"""
    purs = ["identity", *GEOMETRIC]
    rows, sd, suite, x01, _ = _run(tmp_path, monkeypatch, purs)
    assert len(sd.edits) == SEEDS * (1 + len(purs) + len(GEOMETRIC))


def test_幾何類的_reference_欄是_purified_orig(tmp_path, monkeypatch):
    rows, *_ = _run(tmp_path, monkeypatch, ["identity", *GEOMETRIC])
    by = {r["purifier"]: r["reference"] for r in rows}
    assert by["identity"] == "orig"
    assert all(by[p] == "purified_orig" for p in GEOMETRIC)


def test_siglip_與_effect_踩同一個參照(tmp_path, monkeypatch):
    """一邊換一邊不換，兩個讀數會指向不同的基準而表上看不出來。"""
    rows, sd, suite, x01, _ = _run(
        tmp_path, monkeypatch, ["identity", "crop_resize0.1"])
    # 每個算子的 image_similarity 的 a，必須等於它第一顆種子的 pairwise 的 a。
    for k, name in enumerate(["identity", "crop_resize0.1"]):
        first_pair_a = suite.pairs[k * SEEDS][0]
        assert torch.equal(suite.sims[k][0], first_pair_a), name


def test_identity_走哪一條都得到同一個數(tmp_path, monkeypatch):
    """`identity(原圖) = 原圖`，故兩種參照對它是同一個張量。這一條說明分流
    本身沒有引入偏移，差異全部來自算子真的改了取景。"""
    a, *_ = _run(tmp_path, monkeypatch, ["identity", "blur1"])
    b, *_ = _run(tmp_path, monkeypatch, ["identity", "blur1"],
                 geometric_extra=("identity",))
    pick = lambda rows: {r["purifier"]: r["effect_mean"] for r in rows}
    assert pick(a)["identity"] == pick(b)["identity"]


# ------------------------------------------------------------------- 空白地板

def test_幾何類在地板下恰為零(tmp_path, monkeypatch):
    """地板那一格的防禦圖就是原圖，兩側同算子同種子，故位元相同。"""
    rows, *_ = _run(tmp_path, monkeypatch, ["identity", *GEOMETRIC], floor=True)
    by = {r["purifier"]: r["effect_mean"] for r in rows}
    assert all(by[p] == 0.0 for p in GEOMETRIC)


def test_非幾何類的地板照舊非零(tmp_path, monkeypatch):
    """扣地板的機制原樣保留：這幾欄的地板反映算子對影像內容的破壞。"""
    rows, *_ = _run(tmp_path, monkeypatch, NON_GEOMETRIC, floor=True)
    by = {r["purifier"]: r["effect_mean"] for r in rows}
    assert by["identity"] == 0.0          # 同輸入同種子
    assert by["blur1"] > 0.0
    assert by["jpeg75"] > 0.0


def test_地板下幾何類非零就拋錯(tmp_path, monkeypatch):
    """編輯不是確定性的、或種子沒對齊時，這一格會漂。那是量測壞了，
    不是一個可以解讀的小數字。"""
    with pytest.raises(RuntimeError, match="應恰為 0"):
        _run(tmp_path, monkeypatch, ["identity", "crop_resize0.1"],
             floor=True, sd_cls=DriftingSD)


# ------------------------------------------------------------------- 出表程式

def _floor_rows(crop_floor):
    from retention_table import net_gain
    rows = [
        {"_file": "ours_all.csv", "image": "a", "condition": "phase_gain",
         "purifier": "crop_resize0.1", "effect_mean": "0.30"},
        {"_file": "floor_all.csv", "image": "a", "condition": "none",
         "purifier": "crop_resize0.1", "effect_mean": str(crop_floor)},
        {"_file": "ours_all.csv", "image": "a", "condition": "phase_gain",
         "purifier": "blur1", "effect_mean": "0.30"},
        {"_file": "floor_all.csv", "image": "a", "condition": "none",
         "purifier": "blur1", "effect_mean": "0.20"},
    ]
    return net_gain(rows)


def test_retention_table_幾何地板為零時扣地板是恆等():
    table, dropped = _floor_rows(0.0)
    by = {r["purifier"]: r for r in table}
    assert by["crop_resize0.1"]["net_gain"] == pytest.approx(0.30)
    assert by["blur1"]["net_gain"] == pytest.approx(0.10)


def test_retention_table_幾何地板非零就拋錯():
    with pytest.raises(ValueError, match="舊參照"):
        _floor_rows(0.5193)


def _band_src(tmp_path, crop_floor):
    src = tmp_path / "purify"
    src.mkdir()
    fields = ["image", "condition", "purifier", "effect_mean"]

    def dump(name, rows):
        with (src / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    dump("floor_all.csv", [
        {"image": "a", "condition": "none", "purifier": "blur1",
         "effect_mean": "0.2"},
        {"image": "a", "condition": "none", "purifier": "crop_resize0.1",
         "effect_mean": str(crop_floor)},
    ])
    dump("ours_all.csv", [
        {"image": "a", "condition": "phase_gain", "purifier": "blur1",
         "effect_mean": "0.3"},
        {"image": "a", "condition": "phase_gain", "purifier": "crop_resize0.1",
         "effect_mean": "0.3"},
    ])
    return src


def _band_out(tmp_path, monkeypatch, capsys, crop_floor=0.0):
    import band_allocation_table as bat

    src = _band_src(tmp_path, crop_floor)
    monkeypatch.setattr(sys, "argv", ["band_allocation_table.py",
                                      "--src", str(src)])
    bat.main()
    return capsys.readouterr().out


def test_band_allocation_總增益與淨增益並列(tmp_path, monkeypatch, capsys):
    """兩個絕對值都要在表上，讀者才看得到差額。

    `_band_src` 的資料：blur1 的 effect 0.3、地板 0.2（非幾何）；
    crop_resize0.1 的 effect 0.3、地板 0（幾何）。
    """
    out = _band_out(tmp_path, monkeypatch, capsys)
    lines = out.splitlines()
    gross = [ln for ln in lines if ln.startswith("總增益")]
    net = [ln for ln in lines if ln.startswith("淨增益")]
    assert len(gross) == 1 and len(net) == 1
    # 欄序由 PURIFIER_ORDER 決定：blur1 在 crop_resize0.1 之前。
    rows = [ln for ln in lines if ln.startswith("ours")]
    assert len(rows) == 2                       # 兩張表各一列
    assert rows[0].split() == ["ours", "0.3000", "0.3000"]   # 總增益
    assert rows[1].split() == ["ours", "0.1000", "0.3000"]   # 淨增益
    # 幾何欄：總增益與淨增益相等。非幾何欄：兩者恰差一個地板。
    assert float(rows[0].split()[2]) == float(rows[1].split()[2])
    assert (float(rows[0].split()[1]) - float(rows[1].split()[1])
            == pytest.approx(0.2))


def test_band_allocation_表尾有地板與參照(tmp_path, monkeypatch, capsys):
    out = _band_out(tmp_path, monkeypatch, capsys)
    floors = [ln for ln in out.splitlines() if ln.startswith("空白地板")]
    refs = [ln for ln in out.splitlines() if ln.startswith("參照")]
    assert len(floors) == 2 and len(refs) == 2      # 兩張表各印一列
    for ln in floors:
        assert ln.split()[1:] == ["0.2000", "0.0000"]
    for ln in refs:
        assert ln.split()[1:] == ["orig", "purified_orig"]


def test_band_allocation_不出現佔比讀數(tmp_path, monkeypatch, capsys):
    """主讀數是兩個絕對值。任何「佔可達範圍的比例」都不得回到表上。"""
    out = _band_out(tmp_path, monkeypatch, capsys)
    assert "可達" not in out and "%" not in out
    # 飽和值只保留為表尾的一行說明，不進算式。
    assert "0.772" in out and "飽和" in out


def test_band_allocation_幾何地板非零就拋錯(tmp_path, monkeypatch):
    import band_allocation_table as bat

    src = _band_src(tmp_path, 0.5193)
    monkeypatch.setattr(sys, "argv", ["band_allocation_table.py",
                                      "--src", str(src)])
    with pytest.raises(SystemExit, match="舊參照"):
        bat.main()


# ----------------------------------------------------- 報表頁（HTML）

def _ig_src(tmp_path, crop_floor=0.0):
    """只鋪 `purify/*_all.csv`；影像缺席時 `thumb()` 回 None，頁面照樣生得出來。"""
    src = tmp_path / "ip2p_ig_loss"
    gal = src / "purify"
    gal.mkdir(parents=True)
    fields = ["image", "condition", "purifier", "effect_mean", "reference"]

    def dump(name, rows):
        with (gal / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def row(img, cond, pur, eff):
        return {"image": img, "condition": cond, "purifier": pur,
                "effect_mean": str(eff),
                "reference": ("purified_orig" if label_is_geometric(pur)
                              else "orig")}

    dump("floor_all.csv", [row("a", "none", "blur1", 0.2),
                           row("a", "none", "crop_resize0.1", crop_floor)])
    dump("ig_f08_eot_all.csv", [row("a", "phase_gain", "blur1", 0.5),
                                row("a", "phase_gain", "crop_resize0.1", 0.36)])
    return src, gal


def _ig_page(tmp_path, monkeypatch, crop_floor=0.0):
    import build_ig_loss_report as rep

    src, gal = _ig_src(tmp_path, crop_floor)
    monkeypatch.setattr(rep, "SRC", src)
    monkeypatch.setattr(rep, "GAL", gal)
    out = tmp_path / "report.html"
    monkeypatch.setattr(sys, "argv", ["build_ig_loss_report.py",
                                      "--out", str(out)])
    rep.main()
    return out.read_text(encoding="utf-8")


def test_報表頁同時給總增益與淨增益(tmp_path, monkeypatch):
    page = _ig_page(tmp_path, monkeypatch)
    assert "<h3>總增益＝effect(算子)</h3>" in page
    assert "<h3>淨增益＝effect(算子) − 空白地板</h3>" in page
    # blur1（非幾何，地板 0.2）：總 0.5、淨 0.3。
    assert "<td>0.5000</td>" in page and "<td>0.3000</td>" in page
    # crop_resize0.1（幾何，地板 0）：兩張表都是 0.3600。
    assert page.count("<td>0.3600</td>") == 2


def test_報表頁兩張表都印出地板列(tmp_path, monkeypatch):
    page = _ig_page(tmp_path, monkeypatch)
    assert page.count("<td>空白地板（絕對位移）</td>") == 2
    assert "<td>0.2000</td>" in page


def test_報表頁標出每一欄的參照(tmp_path, monkeypatch):
    page = _ig_page(tmp_path, monkeypatch)
    assert "參照＝編輯(算子(原圖))" in page          # 幾何
    assert "參照＝編輯(原圖)" in page                # 其餘


def test_報表頁不含佔比讀數(tmp_path, monkeypatch):
    """`可達範圍` 那一段與 `…%` 的儲存格都不得回到頁面上。"""
    page = _ig_page(tmp_path, monkeypatch)
    assert "可達" not in page
    assert "%</td>" not in page and "%）" not in page
    # 飽和值只留為說明的一行，不進算式。
    assert "飽和於 0.772" in page


def test_報表頁在幾何地板非零時拋錯(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="舊參照"):
        _ig_page(tmp_path, monkeypatch, crop_floor=0.5193)
