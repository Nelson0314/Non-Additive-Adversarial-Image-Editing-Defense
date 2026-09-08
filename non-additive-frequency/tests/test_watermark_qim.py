"""`src/watermark/qim.py` 的驗收。

這一支釘住的是「補錯了不會報錯、只會量到 0.5」的那幾件事。QIM 的失敗模式
幾乎全部長成同一個樣子——位元正確率掉到隨機猜測附近——而那個讀數同時也是
「浮水印確實被擴散編輯打掉了」的長相。兩者無法由讀數本身分辨，所以：

1. **無攻擊時必須逐位元精確**，不是「接近 1.0」。留了餘裕就等於留了一個
   分不出來源的誤差項。
2. **經 uint8 PNG 存檔再讀回也必須逐位元精確。** 本專案的產物一律存 PNG，
   若嵌入端輸出的是浮點而萃取端讀的是 uint8，中間那道量化就是一道沒有被
   量測的攻擊。
3. **必填參數要真的必填。** `delta`／`coeffs`／`repeat` 有預設值的話，
   最容易發生的錯誤（兩端填了不同的值）會靜默地量成「編輯很成功」。

裁切不在本檔驗收：區塊 DCT 綁在固定的 8×8 格點上，裁切必然把讀數打到亂猜，
那是本方案的已知限制，由 `scripts/watermark_probe.py` 量出來記錄。
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.purify.ops import Purifier
from src.watermark.qim import (
    band_coeffs, bit_accuracy, embed, extract, random_bits, slot_count,
)

# 四個中頻格。JPEG q=75 的 luma 量化步長在這四格是 7／7／8／10，
# q=30 是 23／22／27／32；選它們的理由見模組 docstring。
MID4 = ((1, 2), (2, 1), (2, 2), (1, 3))


def _natural(size=64, seed=0):
    """低通的合成影像，統計上接近自然影像（中頻能量不像白雜訊那樣滿）。

    **不用 `torch.rand` 當測試影像**：均勻白雜訊的中頻能量遠高於任何自然
    影像，而 QIM 對振幅縮放沒有不變性（模糊造成的偏移正比於載體大小），
    白雜訊會讓抗模糊那幾條測試量到一個沒有代表性的極端。
    """
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(1, 3, size // 8, size // 8, generator=g)
    return torch.nn.functional.interpolate(
        low, size=(size, size), mode="bilinear", align_corners=False)


def _round_to_uint8_grid(x):
    return torch.round(x * 255.0) / 255.0


# ---------------------------------------------------------------- 無攻擊的往返


def test_無攻擊時往返逐位元精確():
    x, bits = _natural(), random_bits(32, seed=1)
    w = embed(x, bits, delta=32.0, coeffs=MID4, repeat=8)
    got = extract(w, 32, delta=32.0, coeffs=MID4, repeat=8)
    assert torch.equal(got, bits)


@pytest.mark.parametrize("repeat", [1, 2, 3, 16])
def test_各種重複倍數都精確往返(repeat):
    x, bits = _natural(seed=2), random_bits(16, seed=repeat)
    w = embed(x, bits, delta=32.0, coeffs=MID4, repeat=repeat)
    got = extract(w, 16, delta=32.0, coeffs=MID4, repeat=repeat)
    assert torch.equal(got, bits)


def test_嵌入的輸出已落在uint8網格上():
    """`embed` 自己先做 uint8 量化，故存 PNG 這一步不再引入誤差。"""
    x, bits = _natural(seed=3), random_bits(24, seed=3)
    w = embed(x, bits, delta=32.0, coeffs=MID4, repeat=8)
    assert torch.equal(w, _round_to_uint8_grid(w))


def test_經PNG存檔讀回仍逐位元精確(tmp_path):
    x, bits = _natural(seed=4), random_bits(32, seed=4)
    w = embed(x, bits, delta=32.0, coeffs=MID4, repeat=8)

    arr = (w[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    p = tmp_path / "wm.png"
    Image.fromarray(arr).save(p, format="PNG")
    back = torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    # PNG 無損，故讀回的張量必須與 `embed` 的輸出逐位元相同。
    assert torch.equal(back, w)
    assert torch.equal(extract(back, 32, delta=32.0, coeffs=MID4, repeat=8), bits)


def test_嵌入近似冪等():
    """對已嵌同一組位元的圖再嵌一次，係數已在格點上，輸出不再變動。"""
    x, bits = _natural(seed=5), random_bits(32, seed=5)
    kw = dict(delta=32.0, coeffs=MID4, repeat=8)
    w1 = embed(x, bits, **kw)
    w2 = embed(w1, bits, **kw)
    assert torch.equal(extract(w2, 32, **kw), bits)
    # 不宣稱逐位元不動：殘差要夾回 [0,1] 再量化成 uint8，邊界像素可能移一階。
    assert float((w2 - w1).abs().max()) <= 1.5 / 255.0


# ---------------------------------------------------------------- 攻擊下的退化


def test_位元正確率隨噪聲遞減且不會突然歸零():
    x, bits = _natural(size=128, seed=6), random_bits(32, seed=6)
    w = embed(x, bits, delta=32.0, coeffs=MID4, repeat=8)
    accs = []
    for sigma in (0.0, 0.02, 0.05, 0.10, 0.20):
        y = Purifier("noise", sigma, seed=0).evaluate(w)
        accs.append(bit_accuracy(
            extract(y, 32, delta=32.0, coeffs=MID4, repeat=8), bits))
    assert accs[0] == 1.0
    # 單調不增，容一格位元的抖動。
    for lo, hi in zip(accs[1:], accs[:-1]):
        assert lo <= hi + 1.0 / 32
    assert accs[-1] < accs[0]
    # **這一條不宣稱最強的噪聲還讀得回來。** sigma=0.20 時讀數已在亂猜附近，
    # 32 個位元的亂猜會在 0.5 上下抖，故下界刻意設在 0.5 以下：它擋的是
    # 「整批反相」或「常數輸出」這種讀數（那會是 0.0 或極端值），
    # 不是用來證明存活。
    assert 0.35 <= accs[-1] <= 0.65


def test_多數決確實勝過不重複():
    """同一張圖、同一個 delta、同一組噪聲，只改 repeat。

    **強度必須落在「單槽錯誤率明顯低於 0.5」的區間**，否則這條測試會通過得
    毫無意義：多數決只在單槽比亂猜好的時候才會收斂，錯誤率一到 0.5 附近，
    加再多副本也只是在 0.5 附近抖，而抖出來的差距同樣是「repeat 大的比較高」。
    實測 sigma=0.06、delta=24 就已經在那個區間裡（repeat 1→25 只由 0.375 動到
    0.4375，且 repeat=5 與 repeat=1 相同）。此處取 sigma=0.03、delta=24。

    這是本檔唯一直接支持「冗餘有用」的證據，而它是**單一設定上的比較**，
    不是掃描；換影像或換強度不保證維持同樣的階梯。
    """
    x = _natural(size=128, seed=7)
    n_bits, sigma = 32, 0.03
    bits = random_bits(n_bits, seed=7)
    kw = dict(delta=24.0, coeffs=MID4)

    accs = []
    for repeat in (1, 3, 9, 27):
        w = embed(x, bits, repeat=repeat, **kw)
        y = Purifier("noise", sigma, seed=1).evaluate(w)
        accs.append(bit_accuracy(extract(y, n_bits, repeat=repeat, **kw), bits))
    assert accs[0] < 0.85, f"repeat=1 沒有被噪聲打壞，這個比較沒有意義：{accs}"
    assert all(b >= a for a, b in zip(accs, accs[1:])), accs
    assert accs[-1] == 1.0, accs


def test_裁切把讀數打到隨機猜測附近():
    """**這是限制不是 bug。** 區塊格點被推開之後，讀到的係數與嵌入時的無關。"""
    x, bits = _natural(size=128, seed=8), random_bits(64, seed=8)
    kw = dict(delta=32.0, coeffs=MID4, repeat=8)
    w = embed(x, bits, **kw)
    y = Purifier("crop_resize", 0.1).evaluate(w)
    assert bit_accuracy(extract(y, 64, **kw), bits) < 0.75


# ---------------------------------------------------------------- 必填與拋出


def test_三個旋鈕都沒有預設值():
    """`delta`／`coeffs`／`repeat` 只要有一個變成有預設值，兩端填不一致的
    錯誤就會靜默地量成 0.5，而 0.5 也是「編輯成功」的長相。"""
    import inspect

    for fn, required in ((embed, ("delta", "coeffs", "repeat")),
                         (extract, ("delta", "coeffs", "repeat"))):
        params = inspect.signature(fn).parameters
        for name in required:
            p = params[name]
            assert p.kind is inspect.Parameter.KEYWORD_ONLY, name
            assert p.default is inspect.Parameter.empty, f"{fn.__name__} 的 {name}"

    x, bits = _natural(), random_bits(8, seed=9)
    with pytest.raises(TypeError):
        embed(x, bits, coeffs=MID4, repeat=8)
    with pytest.raises(TypeError):
        embed(x, bits, delta=32.0, repeat=8)
    with pytest.raises(TypeError):
        embed(x, bits, delta=32.0, coeffs=MID4)
    with pytest.raises(TypeError):
        extract(x, 8, coeffs=MID4, repeat=8)


@pytest.mark.parametrize("shape", [(3, 64, 64), (2, 3, 64, 64), (1, 1, 64, 64),
                                   (1, 3, 60, 64), (1, 3, 64, 60)])
def test_形狀不對就拋(shape):
    bits = random_bits(8, seed=10)
    with pytest.raises(ValueError):
        embed(torch.zeros(*shape), bits, delta=32.0, coeffs=MID4, repeat=1)
    with pytest.raises(ValueError):
        extract(torch.zeros(*shape), 8, delta=32.0, coeffs=MID4, repeat=1)


def test_位元數塞不下就拋而且訊息說得出差多少():
    x = _natural(size=64)               # 8×8 個區塊 × 4 個係數 = 256 個槽
    assert slot_count(64, 64, len(MID4)) == 256
    with pytest.raises(ValueError, match="257"):
        embed(x, random_bits(257, seed=11), delta=32.0, coeffs=MID4, repeat=1)
    with pytest.raises(ValueError, match="512"):
        embed(x, random_bits(64, seed=11), delta=32.0, coeffs=MID4, repeat=8)
    with pytest.raises(ValueError):
        extract(x, 64, delta=32.0, coeffs=MID4, repeat=8)


@pytest.mark.parametrize("kwargs", [
    dict(delta=0.0, coeffs=MID4, repeat=1),
    dict(delta=-8.0, coeffs=MID4, repeat=1),
    dict(delta=32.0, coeffs=(), repeat=1),
    dict(delta=32.0, coeffs=((1, 2), (1, 2)), repeat=1),     # 重複的格
    dict(delta=32.0, coeffs=((8, 0),), repeat=1),            # 越界
    dict(delta=32.0, coeffs=((0, -1),), repeat=1),
    dict(delta=32.0, coeffs=MID4, repeat=0),
    dict(delta=32.0, coeffs=MID4, repeat=-3),
])
def test_不可能的參數就拋(kwargs):
    x, bits = _natural(), random_bits(4, seed=12)
    with pytest.raises(ValueError):
        embed(x, bits, **kwargs)
    with pytest.raises(ValueError):
        extract(x, 4, **kwargs)


def test_酬載本身不合法就拋():
    x = _natural()
    with pytest.raises(ValueError):
        embed(x, torch.tensor([], dtype=torch.long),
              delta=32.0, coeffs=MID4, repeat=1)
    with pytest.raises(ValueError, match="0 或 1"):
        embed(x, torch.tensor([0, 1, 2]), delta=32.0, coeffs=MID4, repeat=1)
    with pytest.raises(ValueError):
        extract(x, 0, delta=32.0, coeffs=MID4, repeat=1)


def test_飽和影像承載不了就拋而不是回傳讀不出來的圖():
    """半白半黑的圖在小 delta 下夾不動，`verify` 必須當場拋。"""
    sat = torch.zeros(1, 3, 64, 64)
    sat[:, :, :32] = 1.0
    bits = random_bits(16, seed=13)
    with pytest.raises(RuntimeError, match="對不上"):
        embed(sat, bits, delta=4.0, coeffs=MID4, repeat=4)
    # 關掉 verify 不會讓它變對，只是把錯誤延後——這裡把那件事釘住，
    # 免得日後有人以為 verify=False 是個「比較寬鬆的模式」。
    w = embed(sat, bits, delta=4.0, coeffs=MID4, repeat=4, verify=False)
    assert bit_accuracy(extract(w, 16, delta=4.0, coeffs=MID4, repeat=4), bits) < 1.0


# ---------------------------------------------------------------- 輔助函式


def test_bit_accuracy的定義():
    a = torch.tensor([0, 1, 1, 0])
    assert bit_accuracy(a, a) == 1.0
    assert bit_accuracy(a, 1 - a) == 0.0
    assert bit_accuracy(a, torch.tensor([0, 1, 0, 1])) == 0.5
    with pytest.raises(ValueError):
        bit_accuracy(a, torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        bit_accuracy(torch.tensor([]), torch.tensor([]))


def test_band_coeffs與槽數():
    assert band_coeffs(0, 0) == ((0, 0),)
    assert band_coeffs(1, 1) == ((0, 1), (1, 0))
    assert len(band_coeffs(3, 5)) == 15
    assert (0, 5) in band_coeffs(3, 5)      # 帶邊緣的格步長大，見模組 docstring
    with pytest.raises(ValueError):
        band_coeffs(5, 3)
    with pytest.raises(ValueError):
        band_coeffs(-1, 3)

    assert slot_count(512, 512, 4) == 64 * 64 * 4
    with pytest.raises(ValueError):
        slot_count(60, 64, 4)
    with pytest.raises(ValueError):
        slot_count(64, 64, 0)


# ---------------------------------------------------------------- 驅動程式的守門


def _probe():
    """`scripts/watermark_probe.py`。放在函式裡 import，讓本檔其餘測試在
    這支腳本壞掉時仍然跑得動——兩者的失效原因不同，不該綁在一起。"""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import watermark_probe

    return watermark_probe


def _probe_args(*extra):
    return _probe().build_parser().parse_args(
        ["--images", "i", "--out", "o.csv", *extra])


@pytest.mark.parametrize("extra, pattern", [
    ([], "--delta 必填"),
    (["--delta", "48"], "--coeffs 必填"),
    (["--delta", "48", "--coeffs", "1,2"], "--repeat 必填"),
    (["--delta", "0", "--coeffs", "1,2", "--repeat", "8"], "--delta 必須為正"),
    (["--delta", "48", "--coeffs", "9,2", "--repeat", "8"], r"\[0,8\)"),
    (["--delta", "48", "--coeffs", "1,2", "1,2", "--repeat", "8"], "重複"),
    (["--delta", "48", "--coeffs", "1,2", "--repeat", "0"], "正整數"),
    (["--delta", "48", "--coeffs", "1,2", "--repeat", "8",
      "--payload-bits", "99999"], "個槽"),
    (["--delta", "48", "--coeffs", "1,2", "--repeat", "8",
      # 訊息在加入 ip2p_edit／vae_roundtrip 兩個「非淨化」算子之後改寫過：
      # 可用清單不再只有 `KINDS`，故只釘前半段。
      "--op", "not_an_op"], "未知的算子"),
])
def test_驅動程式在載指標之前就擋下設定錯誤(extra, pattern):
    """全部必須是 `SystemExit`，而且發生在 `MetricSuite` 之前。

    `--coeffs 9,2` 這一條特別容易漏：`check_args` 若只檢查格式不檢查範圍，
    就要等到 `embed` 才由 `src/watermark/qim.py` 拋出——那時 LPIPS 與 DISTS
    的權重已經載完了。
    """
    with pytest.raises(SystemExit, match=pattern):
        _probe().check_args(_probe_args(*extra))


def test_驅動程式的三個旋鈕沒有預設值():
    args = _probe_args()
    assert args.delta is None and args.coeffs is None and args.repeat is None


def test_驅動程式的算子清單是資料驅動的():
    """`--op` 可覆蓋，未給時用 `PROBE_OPS`；擴散編輯那一欄日後由此加入。"""
    m = _probe()
    assert [k for k, _ in m.PROBE_OPS] == [
        "identity", "jpeg", "jpeg", "blur", "blur", "noise", "quantize",
        "crop_resize"]
    assert m.parse_op("jpeg:30") == ("jpeg", 30.0)
    assert m.parse_op("identity") == ("identity", 0.0)
    assert m.coeffs_to_field(m.parse_coeffs(["1,2", "2,1"])) == "1,2;2,1"
    assert m.parse_coeffs(["band:1-1"]) == ((0, 1), (1, 0))


def test_random_bits由種子決定():
    assert torch.equal(random_bits(32, 5), random_bits(32, 5))
    assert not torch.equal(random_bits(32, 5), random_bits(32, 6))
    assert random_bits(32, 5).shape == (32,)
    with pytest.raises(ValueError):
        random_bits(0, 5)


def test_填錯頻帶或重複倍數就讀不回原酬載():
    """頻帶與 repeat 填錯時讀數必須掉到亂猜，否則那兩個欄位不必進 CSV。

    **`delta` 不在這條測試裡，因為它填錯不一定讀不回來**，而這件事本身要
    記下來：低通影像的中頻係數幾乎全部落在 QIM 的第 0 個格子裡（|c| < Δ/2），
    此時位元其實只編碼在**係數的正負號**上，任何 Δ' 只要讓判定仍是看正負號
    就照樣讀得回來。實測嵌入用 Δ=32、萃取用 Δ'=24／31／48 全部是 1.000，
    Δ'=17 是 0.953，而 Δ'=12 是 0.000（整批反相）。所以 `delta` 不一致的
    症狀是**時好時壞**，比「一律掉到 0.5」更難察覺——這正是它必須逐列寫進
    CSV、而不是靠讀數事後回推的理由。
    """
    x, bits = _natural(size=128, seed=14), random_bits(64, seed=14)
    kw = dict(delta=32.0, coeffs=MID4, repeat=8)
    w = embed(x, bits, **kw)
    assert bit_accuracy(extract(w, 64, **kw), bits) == 1.0
    for wrong in (dict(kw, coeffs=((3, 2), (2, 3), (3, 3), (4, 1))),
                  dict(kw, repeat=4)):
        assert bit_accuracy(extract(w, 64, **wrong), bits) < 0.75

    # Δ' = 12 對 Δ = 32：格子的相位整個反過來，讀出來是酬載的補數。
    assert bit_accuracy(extract(w, 64, **dict(kw, delta=12.0)), bits) < 0.25
