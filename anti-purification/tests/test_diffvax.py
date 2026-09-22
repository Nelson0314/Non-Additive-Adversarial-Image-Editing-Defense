"""`src/baselines/diffvax.py` 的驗收 —— 替身模型，不載真權重，全程 CPU。

DiffVax 的移植有四個「寫反了也不會有症狀」的地方，這裡逐一釘住：

1. **遮罩極性。** 本 repo 的 PNG 是白＝重繪（`scripts/make_masks.py`），
   官方的 `mask_batch` 也是 1＝編輯區（`utils.py:38-43` 的
   `masked_image = image * (mask < 0.5)`）。兩邊一致、不需反相——寫反的話
   擾動會落到重繪區裡，而防禦圖看起來完全正常。
2. **擾動只落在遮罩之外。** 對應 `unet_out * (1 - mask)`。
3. **兩項損失的算術與方向**（`alpha * L_noise` 與 `L_edit`），含官方那個
   「分子三通道、分母一通道」的常數 3。
4. **沒有硬性 L-infinity 上界。** 官方的 `eps = 32/255` 從未被引用；本檔的
   `linf_eps` 預設 `None`（關閉），開了才夾。

外加 checkpoint 載入的輸入輸出形狀契約（用自己存的 state_dict，不碰真權重）。
"""

import pytest
import torch
import torch.nn as nn

from src.baselines.diffvax import (
    CHECKPOINT_NUM_PARAMS, CHECKPOINT_NUM_TENSORS, CLAMP_MAX, CLAMP_MIN,
    DIFFVAX_ALPHA, MASK_THRESHOLD, NB_FILTER, PAPER_RESOLUTION, SIZE_DIVISOR,
    NestedUNet, binarise_mask, diffvax_terms, immunise, load_immunizer,
)


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------


class _ConstUNet(nn.Module):
    """輸出與輸入同形、每個元素都是 `value` 的常數張量。

    常數讓每一步都能手算；`value` 可以開得比 32/255 大，用來檢查預設路徑
    真的沒有任何硬性上界。
    """

    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.seen = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen = x.detach().clone()
        return torch.full_like(x, self.value)


def _mask(side: int = 32, lo: int = 8, hi: int = 24) -> torch.Tensor:
    """中央的方塊為白（1）＝ 重繪區，其餘為黑（0）＝ 保留區。"""
    m = torch.zeros(1, 1, side, side)
    m[:, :, lo:hi, lo:hi] = 1.0
    return m


def _image(side: int = 32) -> torch.Tensor:
    g = torch.linspace(0.0, 1.0, side)
    return (g.view(1, 1, side, 1) * g.view(1, 1, 1, side)).repeat(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# 1. 遮罩極性：本 repo 的白 對上 DiffVax 的 mask_batch
# ---------------------------------------------------------------------------


def test_binarise_mask_white_is_edit_region():
    """白（1）二值化後仍是 1，即論文的 `~M` / 程式的 `mask_batch`。"""
    m = torch.tensor([[[[0.0, 0.2, 0.49, 0.5, 0.8, 1.0]]]])
    got = binarise_mask(m, MASK_THRESHOLD)
    # `utils.py:38-39`：`<0.5` 歸 0、`>=0.5` 歸 1。0.5 本身算重繪。
    assert got.tolist() == [[[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]]


def test_binarise_mask_accepts_three_channel_png():
    """`make_masks.py` 經 `save_image` 寫出的是三通道灰階 PNG。"""
    m = _mask().repeat(1, 3, 1, 1)
    got = binarise_mask(m)
    assert got.shape == (1, 1, 32, 32)
    assert torch.equal(got, _mask())


def test_binarise_mask_rejects_inconsistent_channels():
    m = _mask().repeat(1, 3, 1, 1)
    m[:, 1] = 1.0 - m[:, 1]
    with pytest.raises(ValueError, match="通道之間不一致"):
        binarise_mask(m)


def test_masked_image_zeroes_the_white_region():
    """進網路的是 `image * (mask < 0.5)`：白區歸零、黑區保留。

    這是極性對應的實質判準。若本 repo 的白被誤讀成「保留區」，歸零的就會
    是黑區，而輸出的防禦圖照樣是一張看起來正常的圖。
    """
    x01, m = _image(), _mask()
    res = immunise(_ConstUNet(0.0), x01, m)
    x = x01 * 2.0 - 1.0
    assert torch.allclose(res["x_masked"], x * (1.0 - m))
    assert float(res["x_masked"][:, :, 8:24, 8:24].abs().max()) == 0.0
    # 黑區（保留區）不可以被動到。
    assert torch.allclose(res["x_masked"][:, :, :8], x[:, :, :8])


# ---------------------------------------------------------------------------
# 2. 擾動只落在遮罩之外
# ---------------------------------------------------------------------------


def test_delta_is_zero_inside_the_repaint_mask():
    """對應 `unet_out = unet_out * (1 - mask)`。"""
    res = immunise(_ConstUNet(0.7), _image(), _mask())
    d, me = res["delta"], res["mask_edit"]
    assert float((d * me).abs().max()) == 0.0
    # 最小值要**只取免疫區**。`d * (1 - me)` 在重繪區恆為 0，對整張取 min
    # 永遠是 0，那個斷言無論擾動寫成什麼都會通過。
    keep = me.expand_as(d) == 0
    assert float(d[keep].abs().min()) == pytest.approx(0.7)


def test_edit_region_keeps_the_original_pixels():
    """`demo.py:297` 的 `recover_image(..., background=True)`：重繪區放回原圖。"""
    x01, m = _image(), _mask()
    res = immunise(_ConstUNet(0.7), x01, m)
    assert torch.allclose(res["x_def01"] * m, x01 * m, atol=1e-6)
    # 保留區確實被改了，否則這個測試等於什麼都沒驗。
    assert float(((res["x_def01"] - x01) * (1.0 - m)).abs().max()) > 1e-3


def test_restore_edit_region_off_leaves_mid_grey():
    """關掉合成時，重繪區是 `x_adv` 的 0，換算回 `[0,1]` 就是 0.5 的中灰。"""
    res = immunise(_ConstUNet(0.7), _image(), _mask(),
                   restore_edit_region=False)
    inside = res["x_def01"][:, :, 8:24, 8:24]
    assert torch.allclose(inside, torch.full_like(inside, 0.5))


# ---------------------------------------------------------------------------
# 3. 兩項損失的算術與方向
# ---------------------------------------------------------------------------


def test_alpha_is_the_config_value_not_the_signature_default():
    """`configs/train.yml` 的 `alpha: 4`；函式簽名的 `alpha=1` 不是實跑值。"""
    assert DIFFVAX_ALPHA == 4.0


def test_terms_arithmetic_is_hand_computable():
    """常數輸入下兩項都可手算，含官方那個「分子三通道、分母一通道」的 3。"""
    m = _mask()                       # 重繪區 16*16 = 256 個像素
    keep = 1.0 - m
    edited = torch.full((1, 3, 32, 32), -0.25)
    x_masked = torch.zeros(1, 3, 32, 32)
    x_adv = torch.full((1, 3, 32, 32), 0.1) * keep

    t = diffvax_terms(edited, x_adv, x_masked, m)
    # 分子跨 3 通道、分母只有 1 通道，故是 3 * |值|，不是 |值|。
    # 容差用 rel 而不是 pytest.approx 的預設絕對值：L1 範數在 float32 上把
    # 2304 個 0.1 累加起來，捨入誤差約 2e-5 相對量（float64 下同一式回到
    # 0.29999999999998855）。0.25 是二進位精確值，故 l_edit 仍可逐位相等。
    assert float(t["l_edit"]) == pytest.approx(3 * 0.25)
    assert float(t["l_noise"]) == pytest.approx(3 * 0.1, rel=1e-4)
    assert float(t["alpha_l_noise"]) == pytest.approx(
        DIFFVAX_ALPHA * 3 * 0.1, rel=1e-4)
    assert float(t["total"]) == pytest.approx(3 * 0.25 + 4 * 3 * 0.1, rel=1e-4)


def test_l_edit_falls_as_the_edit_approaches_the_zero_target():
    """`target_image = torch.zeros_like(img_out)`：把重繪區壓成中灰才算成功。"""
    m = _mask()
    z = torch.zeros(1, 3, 32, 32)
    far = diffvax_terms(torch.full((1, 3, 32, 32), 0.8), z, z, m)["l_edit"]
    near = diffvax_terms(torch.full((1, 3, 32, 32), 0.1), z, z, m)["l_edit"]
    assert float(near) < float(far)
    assert float(diffvax_terms(z, z, z, m)["l_edit"]) == 0.0


def test_l_noise_rises_with_the_perturbation_magnitude():
    m = _mask()
    keep = 1.0 - m
    z = torch.zeros(1, 3, 32, 32)
    small = diffvax_terms(z, torch.full_like(z, 0.02) * keep, z, m)["l_noise"]
    large = diffvax_terms(z, torch.full_like(z, 0.20) * keep, z, m)["l_noise"]
    assert float(small) < float(large)


def test_l_edit_ignores_the_immunisation_region_and_vice_versa():
    """兩項各自只看自己那一側。互換遮罩會讓兩項同時歸零。"""
    m = _mask()
    keep = 1.0 - m
    edited = torch.full((1, 3, 32, 32), 0.5) * keep      # 只有保留區有值
    x_masked = torch.zeros(1, 3, 32, 32)
    x_adv = torch.full((1, 3, 32, 32), 0.5) * m          # 只有重繪區有值
    t = diffvax_terms(edited, x_adv, x_masked, m)
    assert float(t["l_edit"]) == 0.0
    assert float(t["l_noise"]) == 0.0


def test_terms_match_immunise_output():
    """`x_adv - x_masked` 恰等於 `delta`（官方減的是 masked image，不是原圖）。

    影像壓到 `[0, 0.5]`，使 `x + 0.3` 不觸及 `clamp` 的上界——碰到 clamp 時
    兩者本來就會差開，那是 clamp 的作用，不是本測試要驗的事。
    """
    x01, m = _image() * 0.5, _mask()
    res = immunise(_ConstUNet(0.3), x01, m)
    assert torch.allclose(res["x_adv"] - res["x_masked"], res["delta"],
                          atol=1e-6)
    t = diffvax_terms(torch.zeros_like(x01), res["x_adv"], res["x_masked"],
                      res["mask_edit"])
    assert float(t["l_noise"]) == pytest.approx(3 * 0.3, rel=1e-4)


# ---------------------------------------------------------------------------
# 4. 沒有硬性 L-infinity 上界
# ---------------------------------------------------------------------------


def test_no_linf_budget_by_default():
    """官方的 `eps = 32/255` 與 `immunize_img(epsilon=32)` 從未被引用。

    替身輸出 0.9（遠大於 32/255 ≈ 0.1255），預設路徑必須原樣放行；
    唯一的限制是後面的 `clamp(-1, 1)`。
    """
    res = immunise(_ConstUNet(0.9), _image(), _mask())
    outside = res["delta"] * (1.0 - res["mask_edit"])
    assert float(outside.abs().max()) == pytest.approx(0.9)
    assert float(outside.abs().max()) > 32.0 / 255.0
    assert float(res["x_adv"].max()) <= CLAMP_MAX
    assert float(res["x_adv"].min()) >= CLAMP_MIN


def test_linf_budget_applies_only_when_asked():
    """`linf_eps` 是本專案為了等失真比較加的開關，預設關閉。"""
    eps = 8.0 / 255.0
    res = immunise(_ConstUNet(0.9), _image(), _mask(), linf_eps=eps)
    d, me = res["delta"], res["mask_edit"]
    assert float(d.abs().max()) == pytest.approx(eps)
    assert float((d * me).abs().max()) == 0.0        # 夾過之後仍然只在遮罩外


def test_linf_budget_rejects_non_positive():
    with pytest.raises(ValueError, match="linf_eps"):
        immunise(_ConstUNet(0.9), _image(), _mask(), linf_eps=0.0)


# ---------------------------------------------------------------------------
# 5. checkpoint 的輸入輸出形狀契約
# ---------------------------------------------------------------------------


def test_nested_unet_matches_the_official_checkpoint_shape(tmp_path):
    """官方權重是純 state_dict，212 個 tensor、9,170,721 個參數。

    這裡不載真權重，只用自己建的同一個網路存一份 state_dict，驗證
    `load_immunizer` 的契約（`strict=True`、`eval()`、不需要梯度）以及
    tensor 數與參數量與官方的實測值相符——數字對不上就代表結構被改動了。
    """
    net = NestedUNet(num_classes=3)
    sd = net.state_dict()
    assert len(sd) == CHECKPOINT_NUM_TENSORS
    assert sum(v.numel() for v in sd.values()) == CHECKPOINT_NUM_PARAMS

    path = tmp_path / "stub.pth"
    torch.save(sd, path)
    loaded = load_immunizer(path)

    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())

    side = SIZE_DIVISOR * 2
    out = loaded(torch.zeros(1, 3, side, side))
    assert out.shape == (1, 3, side, side)


def test_load_immunizer_is_strict(tmp_path):
    sd = NestedUNet(num_classes=3).state_dict()
    sd.pop("final.bias")
    path = tmp_path / "broken.pth"
    torch.save(sd, path)
    with pytest.raises(RuntimeError):
        load_immunizer(path)


def test_load_immunizer_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="diffvax_trained.pth"):
        load_immunizer(tmp_path / "nope.pth")


def test_immunise_shape_contract():
    x01, m = _image(), _mask()
    res = immunise(_ConstUNet(0.1), x01, m)
    assert res["x_def01"].shape == x01.shape
    assert res["delta"].shape == x01.shape
    assert res["mask_edit"].shape == (1, 1, 32, 32)
    assert float(res["x_def01"].min()) >= 0.0
    assert float(res["x_def01"].max()) <= 1.0


def test_side_must_be_divisible_by_sixteen():
    """四層 MaxPool2d(2,2)。不整除時拋錯，不靜默縮放。"""
    side = 40
    x01 = torch.rand(1, 3, side, side)
    m = torch.zeros(1, 1, side, side)
    with pytest.raises(ValueError, match="整除"):
        immunise(_ConstUNet(0.1), x01, m)


def test_paper_resolution_is_divisible():
    assert PAPER_RESOLUTION % SIZE_DIVISOR == 0
    assert NB_FILTER == (32, 64, 128, 256, 512)


def test_image_and_mask_sizes_must_agree():
    with pytest.raises(ValueError, match="尺寸不同"):
        immunise(_ConstUNet(0.1), _image(32), _mask(16, 4, 12))
