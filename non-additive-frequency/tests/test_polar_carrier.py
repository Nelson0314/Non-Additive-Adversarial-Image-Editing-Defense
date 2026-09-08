"""極座標可分離內容場的四個不變量（`scripts/polar_carrier_probe.py`）。

構造很短，而錯了不會拋錯，只會讓不變性靜默消失：

1. **原點是影像中心。** 算子繞的是影像中心（`rotate_random` 的 `affine_grid`、
   `crop_resize` 的 `CROP_MODE = "center"`）。取成支撐質心時不變性完全不成立。
2. **`g(φ)` 在 `φ = ±π` 的接縫上連續。** 不連續會留一條半徑方向的硬邊，
   那條邊是高頻的，正好是要避開的東西。
3. **`f(r)` 對旋轉、`g(φ)` 對縮放，各自不變**——而且是**對整個群**，不是對
   某一個參數值。這一條是與 `runs/log_periodic_probe/` 的分界：那個候選只在
   `s = 1.2488` 有值，相鄰的 1.15 與 1.30 歸零。
4. **場的值域落在 [0,1]。** 內容場會被 `clamp`，超界時被夾掉的那一部分
   不會出現在任何讀數上。

只用 CPU 與合成場，不載任何權重、不讀任何影像。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _q in (ROOT, ROOT / "scripts"):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

from polar_carrier_probe import _rot, cosine, field, polar  # noqa: E402
from src.purify import ops  # noqa: E402

SIDE = 128


def _centred(kind, seed=0):
    c = field(kind, SIDE, seed)
    return c - c.mean()


def test_原點在影像中心():
    r, phi = polar(SIDE)
    h = SIDE // 2
    # 半徑在正中央的四個像素上最小，四角最大。
    assert float(r[h - 1:h + 1, h - 1:h + 1].max()) < float(r.min()) + 0.02
    assert float(r[0, 0]) == pytest.approx(float(r[-1, -1]), abs=1e-5)
    assert float(r[0, -1]) == pytest.approx(float(r[-1, 0]), abs=1e-5)


def test_角度場在接縫上連續():
    """`φ = ±π` 那一條接縫。不連續時左右兩欄的差會與畫面別處差一個量級。"""
    c = field("angular", SIDE, 0)
    h = SIDE // 2
    seam = (c[0, :, h:, 0] - c[0, :, h:, -1]).abs().mean()   # 跨過 ±π
    typical = (c[0, :, :, 1:] - c[0, :, :, :-1]).abs().mean()
    assert float(seam) < 20.0 * float(typical), (float(seam), float(typical))


def test_半徑場對旋轉不變():
    """內接圓內精確；整張圖上因為補零的四角會低一些，故量內接圓。"""
    c = _centred("radial")
    r, _ = polar(SIDE)
    inner = (r < 0.9).expand_as(c)
    for deg in (2.0, 5.0, 15.0, 45.0):
        y = _rot(c + 0.5, deg) - 0.5
        assert cosine(y[inner], c[inner]) > 0.97, deg


def test_角度場對中心縮放不變而且沒有窄峰():
    """與 `runs/log_periodic_probe/` 的分界：那裡只有一個比例有值。"""
    c = _centred("angular")
    vals = [cosine(ops.crop_resize(c + 0.5, f) - 0.5, c)
            for f in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25)]
    assert min(vals) > 0.99, vals
    # 平的，不是尖峰：最大與最小的差要小。
    assert max(vals) - min(vals) < 0.01, vals


def test_兩個構造換掉的是不同的一半():
    """`f(r)` 對縮放、`g(φ)` 對旋轉都只是等變。兩者不可以同時宣稱。"""
    fr, gp = _centred("radial"), _centred("angular")
    assert cosine(ops.crop_resize(fr + 0.5, 0.15) - 0.5, fr) < 0.5
    assert cosine(_rot(gp + 0.5, 25.0) - 0.5, gp) < 0.5


def test_場的值域落在單位區間():
    for kind in ("radial", "angular"):
        c = field(kind, SIDE, 3)
        assert tuple(c.shape) == (1, 3, SIDE, SIDE)
        assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0
        # 不是常數場——常數場的不變性是平凡的。
        assert float(c.std()) > 0.05


# ---------------------------------------------- 參數化本身（`PatchPolarParam`）
#
# 上面六條量的是**探針裡的生成函式**。派工走的是
# `src/defense/patch_param.py::PatchPolarParam`，兩者是不同的程式，
# 不變性要在**實際會被跑的那一份**上再釘一次。


def _param(kind, cls=None, **kw):
    from src.defense.patch_param import PatchPolarParam
    m = torch.zeros(1, 1, SIDE, SIDE)
    m[..., :20, :20] = 1.0
    return (cls or PatchPolarParam)(radius=0.05, mask=m, placement="complement",
                                    polar=kind, **kw)


def _x(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, SIDE, SIDE, generator=g)


def test_參數量是三乘以bins():
    x = _x()
    p = _param("radial", bins=64)
    p.reset(x, 0)
    assert tuple(p.c.shape) == (1, 3, 64)
    assert tuple(p._field(x).shape) == (1, 3, SIDE, SIDE)


def test_梯度通到剖面():
    """線性內插對剖面可微；寫成 `index_select` 之後梯度斷掉不會拋錯。"""
    x = _x()
    p = _param("angular")
    p.reset(x, 0)
    p.render(x).sum().backward()
    assert p.c.grad is not None
    assert float(p.c.grad.abs().sum()) > 0


def test_參數化的半徑場對旋轉不變():
    """用預設 bins。**不變性的成立條件是剖面夠平滑**，見
    `test_不變性隨bins下降`——那一條把條件本身釘住。"""
    from src.defense.patch_param import PatchPolarRandomParam
    x = _x()
    p = _param("radial", cls=PatchPolarRandomParam)
    p.reset(x, 0)
    c = p._field(x)
    c = c - c.mean()
    r, _ = polar(SIDE)
    inner = (r < 0.9).expand_as(c)
    for deg in (2.0, 5.0, 15.0):
        y = _rot(c + 0.5, deg) - 0.5
        assert cosine(y[inner], c[inner]) > 0.97, deg


def test_參數化的角度場對中心縮放不變且沒有窄峰():
    from src.defense.patch_param import PatchPolarRandomParam
    x = _x()
    p = _param("angular", cls=PatchPolarRandomParam)
    p.reset(x, 0)
    c = p._field(x)
    c = c - c.mean()
    vals = [cosine(ops.crop_resize(c + 0.5, f) - 0.5, c)
            for f in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25)]
    assert min(vals) > 0.99, vals
    assert max(vals) - min(vals) < 0.01, vals


def test_不變性隨bins下降_這是構造的成立條件():
    """**這個構造不是無條件成立的。** 格數太多時剖面本身是高頻的、場在取樣上
    混疊，算子的雙線性重取樣會把混疊的部分毀掉。預設的 32 落在 0.997 以上，
    256 掉到 0.79——把這件事寫成測試，改預設值時才不會靜默失去不變性。"""
    from src.defense.patch_param import PatchPolarParam, PatchPolarRandomParam
    x = _x()
    vals = {}
    for bins in (16, 32, 256):
        p = _param("angular", cls=PatchPolarRandomParam, bins=bins)
        p.reset(x, 0)
        c = p._field(x)
        c = c - c.mean()
        vals[bins] = min(cosine(ops.crop_resize(c + 0.5, f) - 0.5, c)
                         for f in (0.02, 0.10, 0.25))
    assert vals[16] > 0.99 and vals[32] > 0.99, vals
    assert vals[256] < 0.90, vals
    assert PatchPolarParam.POLAR_BINS <= 32, "預設 bins 超過不變性還成立的範圍"


def test_角度場沒有接縫():
    """環繞內插寫成夾取時，`φ = ±π` 會有一條半徑方向的硬邊。"""
    from src.defense.patch_param import PatchPolarRandomParam
    x = _x()
    p = _param("angular", cls=PatchPolarRandomParam)
    p.reset(x, 0)
    assert torch.allclose(p.c[:, :, 0], p.c[:, :, 0])
    h = SIDE // 2
    f = p._field(x)
    seam = (f[0, :, h:, 0] - f[0, :, h:, -1]).abs().mean()
    typical = (f[0, :, :, 1:] - f[0, :, :, :-1]).abs().mean()
    assert float(seam) < 20.0 * float(typical), (float(seam), float(typical))


def test_不與平舖或帶限併用():
    """兩個旗標作用在二維可學張量上，這裡的可學張量是一維剖面。
    靜默忽略時 CSV 的欄位照樣寫著設定值。"""
    with pytest.raises(ValueError, match="patch-tile"):
        _param("radial", tile=32)
    with pytest.raises(ValueError, match="patch-res"):
        _param("radial", res=8)


def test_值域與參數檢查():
    with pytest.raises(ValueError, match="polar"):
        _param("spiral")
    with pytest.raises(ValueError, match="bins"):
        _param("radial", bins=4)


def test_恆等起點是支撐內每格的平均色():
    """子空間裡沒有原圖，但有一個最近點。空的格要落回支撐內的整體平均，
    不可以落回 0——0 是黑色，會在剖面上造出一條假的硬邊。"""
    x = _x()
    p = _param("radial", bins=32)
    p.reset(x, 0)
    c = p.c.detach()
    assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0
    sup = p.support.to(x.dtype)
    mean_in = float((x * sup).sum() / (sup.sum() * 3))
    # 沒有任何一格是黑的（那是「空格落回 0」的症狀）。
    assert float(p.c.detach().mean(1).min()) > 0.1 * mean_in


def test_隨機臂不最佳化且形狀正確():
    from src.defense.patch_param import PatchPolarRandomParam
    x = _x()
    p = _param("angular", cls=PatchPolarRandomParam, bins=48)
    p.reset(x, 0)
    assert p.params() == []
    assert tuple(p.c.shape) == (1, 3, 48)
    assert not p.c.requires_grad


def test_兩欄逐列寫進geometry():
    x = _x()
    p = _param("angular", bins=64)
    p.reset(x, 0)
    g = p.geometry()
    assert g["patch_polar"] == "angular"
    assert g["patch_polar_bins"] == 64
    assert g["patch_res"] == 1 and g["patch_tile"] == 0


# ------------------------------------------------ 進得了最佳化迴圈（整合守門）
#
# 上面的梯度測試只確認 `backward` 有值。真正會靜默失效的是**迴圈**那一層：
# `params()` 回傳錯的張量、`project()` 把剖面夾死、支撐外被動到——三者都不會
# 拋錯，只會讓那一批跑完之後每一格看起來都正常而解沒有動。
#
# 用一個合成損失跑真實的 `run_param_pgd`，不載任何擴散模型。


def _pgd_setup():
    from src.defense.param_pgd import run_param_pgd
    from src.defense.patch_param import PatchPolarParam
    x = _x(2)
    m = torch.zeros(1, 1, SIDE, SIDE)
    m[..., :24, :24] = 1.0
    p = PatchPolarParam(radius=0.05, mask=m, placement="complement",
                        polar="angular", bins=32, init="random")
    return run_param_pgd, x, m, p


def test_迴圈真的把損失壓下去():
    """`params()` 漏掉剖面時損失一步都不動，而 `trace` 看起來只是很平。"""
    run_param_pgd, x, _m, p = _pgd_setup()
    tgt = torch.zeros_like(x)

    def loss(y):
        return (y - tgt).pow(2).mean()

    p.reset(x, 0)
    before = float(loss(p.render(x)).detach())
    res = run_param_pgd(x, p, loss, steps=40, step_size=0.02)
    assert float(loss(res.x_def)) < before * 0.9, (before, float(loss(res.x_def)))


def test_迴圈跑完之後主體仍然逐位元不動():
    """支撐外必須逐位元等於原圖——這是本專案對每一個參數化的硬條件。"""
    run_param_pgd, x, m, p = _pgd_setup()
    res = run_param_pgd(x, p, lambda y: (y - torch.zeros_like(y)).pow(2).mean(),
                        steps=20, step_size=0.02)
    outside = (p.support <= 0) if p.support.dtype != torch.bool else ~p.support
    o3 = outside.expand_as(x)
    assert torch.equal(res.x_def[o3], x[o3])


def test_迴圈跑完之後剖面仍在值域內且形狀不變():
    run_param_pgd, x, _m, p = _pgd_setup()
    run_param_pgd(x, p, lambda y: (y - torch.zeros_like(y)).pow(2).mean(),
                  steps=20, step_size=0.02)
    assert tuple(p.c.shape) == (1, 3, 32)
    assert float(p.c.min()) >= 0.0 and float(p.c.max()) <= 1.0


def test_最佳化之後不變性仍然成立():
    """`bins` 是頻寬上限，故**最佳化不可能把不變性學掉**——這是這個構造與
    「訓練出一個剛好抗某個算子的解」的分界。跑完之後再量一次。

    目標刻意取一張隨機影像而不是全零：全零會把剖面壓成常數場，
    那時扣掉直流恰為零、餘弦是 0/0，量到的是 `nan` 而不是不變性。
    """
    run_param_pgd, x, _m, p = _pgd_setup()
    tgt = _x(9)
    run_param_pgd(x, p, lambda y: (y - tgt).pow(2).mean(),
                  steps=40, step_size=0.05)
    c = p._field(x).detach()
    c = c - c.mean()
    assert float(c.std()) > 1e-3, "剖面塌成常數場，這一條就量不到東西了"
    vals = [cosine(ops.crop_resize(c + 0.5, f) - 0.5, c)
            for f in (0.02, 0.10, 0.25)]
    assert min(vals) > 0.99, vals
