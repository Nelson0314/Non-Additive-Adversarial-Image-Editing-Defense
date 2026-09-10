# 等距色彩場與同色碰撞 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把顏色載體的**幅度**變成一個可掃、可量、可否證的自變數，並加上一個
奇異值恰為 1 的等距臂與一個不需要擴散梯度的定位破壞目標，然後用四個真讀數
（位移、主體錨定身分、輸出臉數、外觀）量出顏色族到底有沒有天花板。

**Architecture:** 現有的 `src/defense/lowfreq_color.py` 已經提供「不加高頻」的
兩個載體（`ChromaAffineParam` 鎖亮度＋奇異值上界，`RegionPaletteParam` 低通
權重圖）與量測（`highfreq_report`）。缺的是三件事：(1) 顏色位移的幅度沒有
旋鈕，`runs/lowfreq/` 那 1120 列全部是 `max_gain=1.0`、`T_distance=0`、未訓練，
等於恆等；(2) 沒有奇異值恰為 1 的等距臂，`_clamp_singular_values` 只設上界，
起點的 `T0_ab` 可能是收縮映射（去飽和），白付了位移；(3) 已跑的顏色批次只有
代理讀數（`clip_margin`），沒有位移與主體錨定身分。本計畫先補這三件，再加
元件 B1（同色碰撞，模型無關的目標函數）。

**Tech Stack:** PyTorch、skimage（CIEDE2000）、InstructPix2Pix（fp32）、pytest。

**Spec:** 本文自足。背景數據見 `runs/objective_pilot/README.md`、
`runs/carrier_purify_response/README.md`、`runs/lowfreq/*/lowfreq_scan.csv`。

## Global Constraints

- **載體不得往空間高頻加東西**，優先於其餘各項。判準：`highfreq_report` 的
  `hf_ratio_rgb_total` ≤ 1。每個新臂的測試都要釘住這一條。
- **受保護主體逐位元不動。** `support = 0` 的像素必須與原圖逐位元相同
  （`torch.equal`，不是 `allclose`）。
- **不得逐像素 clamp／逐像素縮放**回色域：那是一張有邊的圖，等於換一種方式
  加高頻。只准整體常數（`gamut_scale`）或處處可微的軟裁（`soft_gamut`）。
- **不設成敗判準。** 助手不得自訂門檻判斷方法有沒有效；所有欄位照報。
- **命名不含日期、流水號、順序詞。**
- **GPU 工作一律送遠端**，遠端 repo `/nfs/home/nelson0314/WACV-s3`，
  先 `source ~/env.sh` 再 `cd`。攻擊評測用 fp32。
- commit message 用英文。文件用繁體中文。
- 本機 Python `C:/Users/nelso/miniconda3/envs/wacv/python.exe`；測試
  `python -m pytest -q --ignore=runs`。

---

### Task 1: 幅度旋鈕（全域常數插值）

顏色位移的幅度目前由目標配色與 `source_mean` 的距離隱含決定，沒有旋鈕。加一個
**空間常數**的插值係數 `amplitude`：

    out = src + amplitude * (mapped - src)

空間常數所以不可能引入任何空間頻率；且對 `amplitude` 落在 [0,1]，有效映射
`(1-a)·I + a·M` 的奇異值不超過 `max(1, sigma_max(M))`，故 `max_gain = 1`
時「不放大對比」的保證由凸性自動保留。`amplitude` 不參與梯度，是投影的一部分，
與 `gamut_scale` 同性質。

**Files:**
- Modify: `src/defense/lowfreq_color.py`（`ChromaAffineParam.__init__`、
  `raw_rgb`、`diagnostics`，`RegionPaletteParam` 同步）
- Test: `tests/test_lowfreq_amplitude.py`

**Interfaces:**
- Consumes: 無
- Produces: `ChromaAffineParam(..., amplitude: float = 1.0)`；
  `RegionPaletteParam(..., amplitude: float = 1.0)`；
  兩者的 `set_amplitude(a: float) -> None`；
  `ChromaAffineParam.effective_chroma_matrix() -> torch.Tensor`；
  `diagnostics()` 新增欄位 `amplitude`（float）。

- [ ] **Step 1: 寫失敗的測試**

```python
# tests/test_lowfreq_amplitude.py
import torch
from src.defense.lowfreq_color import ChromaAffineParam


def _image(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 64, 64, generator=g, dtype=torch.float32)


def _param(amplitude, support=None):
    if support is None:
        support = torch.ones(1, 1, 64, 64)
    return ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=support, radius=0.2, max_gain=1.0, amplitude=amplitude)


def test_amplitude_zero_is_bit_exact_identity():
    x = _image()
    p = _param(0.0)
    p.reset(x, seed=0)
    assert torch.equal(p.render(x), x)


def test_amplitude_is_monotone_in_colour_displacement():
    x = _image()
    d = []
    for a in (0.25, 0.5, 1.0):
        p = _param(a)
        p.reset(x, seed=0)
        d.append(float((p.render(x) - x).abs().mean()))
    assert d[0] < d[1] < d[2]


def test_amplitude_preserves_the_gain_bound():
    x = _image()
    for a in (0.25, 0.5, 1.0):
        p = _param(a)
        p.reset(x, seed=0)
        s = torch.linalg.svdvals(p.effective_chroma_matrix())
        assert float(s.max()) <= 1.0 + 1e-9


def test_amplitude_appears_in_diagnostics():
    x = _image()
    p = _param(0.5)
    p.reset(x, seed=0)
    assert p.diagnostics(x)['amplitude'] == 0.5
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_lowfreq_amplitude.py -q`
Expected: FAIL，`ChromaAffineParam.__init__() got an unexpected keyword argument 'amplitude'`

- [ ] **Step 3: 實作**

在 `ChromaAffineParam.__init__` 的簽名末尾加 `amplitude=1.0`，並在
`super().__init__(...)` 之前存下：

```python
        if not (0.0 <= float(amplitude) <= 1.0):
            raise ValueError('amplitude 必須落在 [0,1]：超過 1 的外插會讓有效'
                             '映射的奇異值突破 max_gain，那就不再保證不放大對比')
        self.amplitude = float(amplitude)
```

加兩個方法：

```python
    def set_amplitude(self, a: float) -> None:
        """幅度是投影的一部分，不參與梯度，與 `gamut_scale` 同性質。"""
        if not (0.0 <= float(a) <= 1.0):
            raise ValueError('amplitude 必須落在 [0,1]')
        self.amplitude = float(a)

    def effective_chroma_matrix(self):
        """含幅度插值之後真正作用在色度上的 2x2。

        `(1-a)·I + a·M` 的奇異值不超過 max(1, sigma_max(M))，所以
        `max_gain = 1` 時幅度插值不可能破壞「不放大色度對比」那條保證。
        """
        m = self.chroma_matrix()
        eye = torch.eye(2, dtype=m.dtype, device=m.device)
        return (1.0 - self.amplitude) * eye + self.amplitude * m
```

`raw_rgb` 裡把映射結果換成插值後的：在 `out_ab = out_ab + self.target_mean...`
之後、`if self.gamut == 'scale'` 之前插入

```python
        # 幅度是空間常數，故不可能引入任何空間頻率。
        out_ab = lab[:, 1:] + self.amplitude * (out_ab - lab[:, 1:])
```

`diagnostics` 的回傳字典加 `'amplitude': self.amplitude`。
`RegionPaletteParam.__init__` 同樣加 `amplitude=1.0`、`set_amplitude`，並在它
混合完各區域的色度之後套同一行插值；`diagnostics` 同樣加欄位。

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_lowfreq_amplitude.py -q`
Expected: 4 passed

- [ ] **Step 5: 跑全部測試確認沒有回歸**

Run: `python -m pytest -q --ignore=runs`
Expected: 1335 passed / 2 skipped / 1 xfailed（原 1331 加本檔 4 筆）

- [ ] **Step 6: Commit**

```bash
git add tests/test_lowfreq_amplitude.py src/defense/lowfreq_color.py
git commit -m "Add a spatially constant amplitude knob to the low-frequency colour carriers"
```

---

### Task 2: 求解到指定 CIEDE2000 的幅度

幅度要能對齊到一個指定的 ΔE00 才能跨臂比較（等失真錨點）。ΔE00 在
`amplitude = 0` 時為 0 且隨它單調上升，故二分法可解。**不可達時不得靜默**：
回報達到的值與一個旗標。

**Files:**
- Create: `src/defense/color_amplitude.py`
- Test: `tests/test_color_amplitude.py`

**Interfaces:**
- Consumes: Task 1 的 `set_amplitude`、`render`
- Produces: `solve_amplitude(param, x01, target_delta_e, *, lo=0.0, hi=1.0, tol=1e-3, iters=32) -> dict`，
  回傳 `{'amplitude': float, 'delta_e00': float, 'reached': bool}`；
  `delta_e00(a01, b01) -> float`（薄封裝，走 `src/metrics/suite.py::_delta_e00`，
  不另寫一份色差實作）。

- [ ] **Step 1: 寫失敗的測試**

```python
# tests/test_color_amplitude.py
import torch
from src.defense.color_amplitude import delta_e00, solve_amplitude
from src.defense.lowfreq_color import ChromaAffineParam


def _image(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 64, 64, generator=g, dtype=torch.float32)


def _param():
    return ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=torch.ones(1, 1, 64, 64), radius=0.2, max_gain=1.0)


def test_delta_e00_of_identical_images_is_zero():
    x = _image()
    assert delta_e00(x, x) < 1e-6


def test_solve_hits_a_reachable_target():
    x = _image()
    p = _param()
    p.reset(x, seed=0)
    full = delta_e00(x, p.render(x))
    want = full * 0.5
    out = solve_amplitude(p, x, want)
    assert out['reached'] is True
    assert abs(out['delta_e00'] - want) < 0.2
    assert 0.0 < out['amplitude'] < 1.0


def test_solve_reports_an_unreachable_target_instead_of_pretending():
    x = _image()
    p = _param()
    p.reset(x, seed=0)
    full = delta_e00(x, p.render(x))
    out = solve_amplitude(p, x, full * 10.0)
    assert out['reached'] is False
    assert out['amplitude'] == 1.0
    assert abs(out['delta_e00'] - full) < 1e-6


def test_solve_leaves_the_param_at_the_solved_amplitude():
    x = _image()
    p = _param()
    p.reset(x, seed=0)
    out = solve_amplitude(p, x, delta_e00(x, p.render(x)) * 0.5)
    assert p.amplitude == out['amplitude']
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_color_amplitude.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.defense.color_amplitude'`

- [ ] **Step 3: 實作**

```python
# src/defense/color_amplitude.py
"""把顏色載體的幅度對齊到一個指定的 CIEDE2000。

跨臂比較一律要先對齊失真，否則只是在比誰付得多。幅度（`amplitude`）是空間
常數，ΔE00 在 `amplitude = 0` 時為 0 且隨它單調上升，所以二分法取得到。

**不可達時回報，不靜默。** 目標超過 `amplitude = 1` 能到的值時回傳
`reached = False` 與該臂實際到得了的 ΔE00；呼叫端要把這兩欄照報。
"""
from __future__ import annotations

import torch

from src.metrics.suite import _delta_e00


def delta_e00(a01: torch.Tensor, b01: torch.Tensor) -> float:
    """兩張 [0,1] RGB 影像的平均 CIEDE2000。

    薄封裝：色差只有 `src/metrics/suite.py` 那一份實作（走 skimage），
    這裡不另寫，避免「量測用的色差」與「求解用的色差」悄悄變成兩個東西。
    """
    return _delta_e00(a01.detach(), b01.detach())


@torch.no_grad()
def solve_amplitude(param, x01: torch.Tensor, target_delta_e: float, *,
                    lo: float = 0.0, hi: float = 1.0,
                    tol: float = 1e-3, iters: int = 32) -> dict:
    if not (target_delta_e > 0):
        raise ValueError('target_delta_e 必須為正')
    param.set_amplitude(hi)
    top = delta_e00(x01, param.render(x01))
    if top <= target_delta_e:
        return {'amplitude': hi, 'delta_e00': float(top), 'reached': False}
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        param.set_amplitude(mid)
        got = delta_e00(x01, param.render(x01))
        if got < target_delta_e:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    param.set_amplitude(hi)
    return {'amplitude': float(hi),
            'delta_e00': float(delta_e00(x01, param.render(x01))),
            'reached': True}
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_color_amplitude.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/defense/color_amplitude.py tests/test_color_amplitude.py
git commit -m "Solve the colour amplitude that lands on a requested CIEDE2000"
```

---

### Task 3: 等距臂（奇異值恰為 1）

`max_gain = 1` 只設上界；起點 `T0_ab` 的奇異值可能遠小於 1，那是收縮映射，
色度被壓扁（去飽和）而白付了位移。等距臂把 2×2 投影到 O(2)：

    M = U V^T   （SVD 的極分解），奇異值恰為 1

奇異值恰為 1 表示色度梯度既不放大也不縮小，**高頻新增在構造上為零**，而色相
可以轉到任意角度，顏色位移不受限於「目標配色離來源有多遠」。

**Files:**
- Modify: `src/defense/lowfreq_color.py`
- Test: `tests/test_chroma_isometry.py`

**Interfaces:**
- Consumes: Task 1 的 `amplitude`、`effective_chroma_matrix`
- Produces: `ChromaAffineParam(..., isometric: bool = False)`；
  模組級 `_project_orthogonal(m: torch.Tensor) -> torch.Tensor`；
  `diagnostics()` 新增 `isometric`（int 0/1）。

- [ ] **Step 1: 寫失敗的測試**

```python
# tests/test_chroma_isometry.py
import torch
from src.defense.lowfreq_color import ChromaAffineParam, _project_orthogonal


def _image(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 64, 64, generator=g, dtype=torch.float32)


def _param(isometric, support=None, amplitude=1.0):
    if support is None:
        support = torch.ones(1, 1, 64, 64)
    return ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=support, radius=0.2, max_gain=1.0,
        amplitude=amplitude, isometric=isometric)


def test_projection_returns_singular_values_of_one():
    m = torch.tensor([[2.0, 0.3], [-0.1, 0.4]], dtype=torch.float64)
    s = torch.linalg.svdvals(_project_orthogonal(m))
    assert torch.allclose(s, torch.ones(2, dtype=torch.float64), atol=1e-9)


def test_isometric_arm_has_unit_gain():
    x = _image()
    p = _param(True)
    p.reset(x, seed=0)
    s = torch.linalg.svdvals(p.chroma_matrix())
    assert torch.allclose(s, torch.ones(2, dtype=s.dtype), atol=1e-9)


def test_isometric_arm_adds_no_high_frequency():
    x = _image()
    p = _param(True)
    p.reset(x, seed=0)
    assert p.diagnostics(x)['hf_ratio_rgb_total'] <= 1.0


def test_isometric_arm_moves_more_colour_than_the_bounded_arm():
    x = _image()
    a, b = _param(True), _param(False)
    a.reset(x, seed=0)
    b.reset(x, seed=0)
    ea = (a.render(x) - x).pow(2).mean()
    eb = (b.render(x) - x).pow(2).mean()
    assert float(ea) > float(eb)


def test_support_zero_pixels_are_bit_exact():
    support = torch.zeros(1, 1, 64, 64)
    support[:, :, :32] = 1.0
    x = _image()
    p = _param(True, support=support)
    p.reset(x, seed=0)
    out = p.render(x)
    assert torch.equal(out[:, :, 32:], x[:, :, 32:])
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_chroma_isometry.py -q`
Expected: FAIL，`cannot import name '_project_orthogonal'`

- [ ] **Step 3: 實作**

在 `_clamp_singular_values` 下方加：

```python
def _project_orthogonal(m):
    """極分解取正交因子：奇異值恰為 1。

    `_clamp_singular_values` 只設上界，起點的 `T0_ab` 常常是收縮映射，色度被
    壓扁（去飽和）而白付了位移。奇異值恰為 1 時色度梯度既不放大也不縮小，
    「不加高頻」由構造成立，而色相仍可轉到任意角度。
    """
    u, _, vh = torch.linalg.svd(m)
    return u @ vh
```

`__init__` 加 `isometric=False`，存成 `self.isometric = bool(isometric)`。
`chroma_matrix` 改成：

```python
    def chroma_matrix(self):
        """實際生效的 2x2 映射。等距臂投影到 O(2)，否則只夾奇異值上界。"""
        m = self.T0_ab + self.delta
        if self.isometric:
            return _project_orthogonal(m)
        return _clamp_singular_values(m, self.max_gain)
```

`reset` 裡的 `self.T0_ab` 在等距臂改走同一個投影：

```python
        self.T0_ab = (_project_orthogonal(self.T0[1:, 1:]) if self.isometric
                      else _clamp_singular_values(self.T0[1:, 1:], self.max_gain))
```

`diagnostics` 加 `'isometric': int(self.isometric)`。

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_chroma_isometry.py -q`
Expected: 5 passed

- [ ] **Step 5: 跑全部測試**

Run: `python -m pytest -q --ignore=runs`
Expected: 全綠。既有 `tests/test_lowfreq_color.py` 不得有回歸：非等距路徑
逐位元不變。

- [ ] **Step 6: Commit**

```bash
git add src/defense/lowfreq_color.py tests/test_chroma_isometry.py
git commit -m "Add an isometric chroma arm whose singular values are exactly one"
```

---

### Task 4: 天花板探針的派工腳本

自變數：臂 × ΔE00。**不訓練**——問的是「合法的顏色位移推到自然的極限，編輯
還會不會完成」，訓練是下一批的事。

| 軸 | 值 |
|---|---|
| 臂 | `chroma_bounded`（`max_gain=1`）、`chroma_isometric`、`region_palette`、`undefended` |
| ΔE00 | 6.3（現行工作點）、20、30 |
| 影像 | `data/ncf_manifest.json` 的五張 |
| 指令類 | 衣物改色、頭部配件 `put a hat on the person`、背景 `change the background to a snowy street` |
| 攻擊種子 | 17001／27001／47001 |

`undefended` 臂每個（影像, 指令類, 種子）都要有，作為逐種子的對照：種子層
的攻擊失敗會讓整批讀數變成雜訊。

**Files:**
- Create: `scripts/color_ceiling.py`
- Create: `configs/color_ceiling.json`
- Create: `runs/color_ceiling/README.md`
- Test: `tests/test_color_ceiling.py`

**Interfaces:**
- Consumes: Task 2 的 `solve_amplitude`、Task 3 的 `isometric`
- Produces: `scripts/color_ceiling.py` 的
  `build_param(arm: str, x01, *, support, target_mean, target_cov, blur_sigma: float) -> object`
  與模組級 `COLUMNS: list[str]`（CSV 欄位的合約）。

- [ ] **Step 1: 寫失敗的測試**

CSV 欄位是這批的合約，先把它釘住；GPU 的部分不進單元測試。

```python
# tests/test_color_ceiling.py
import torch
from scripts.color_ceiling import COLUMNS, build_param


def test_columns_carry_the_four_readouts_and_the_appearance_group():
    for name in ('arm', 'image', 'class', 'seed', 'delta_e_target',
                 'amplitude', 'delta_e_reached',
                 'edit_lpips', 'subject_id_def', 'subject_id_orig',
                 'n_faces_edit_def', 'n_faces_edit_orig',
                 'input_psnr', 'input_dists', 'final_deltaE00',
                 'hf_ratio_rgb_total', 'support_outside_max_abs'):
        assert name in COLUMNS


def test_build_param_rejects_an_unknown_arm():
    x = torch.rand(1, 3, 64, 64)
    try:
        build_param('nope', x, support=torch.ones(1, 1, 64, 64),
                    target_mean=[60., 25., -20.],
                    target_cov=[[80., 0., 0.], [0., 40., 0.], [0., 0., 40.]],
                    blur_sigma=24.0)
    except ValueError as e:
        assert 'nope' in str(e)
    else:
        raise AssertionError('未知的臂必須拋錯，不得靜默走預設')


def test_isometric_arm_is_wired_to_the_isometric_flag():
    x = torch.rand(1, 3, 64, 64)
    p = build_param('chroma_isometric', x, support=torch.ones(1, 1, 64, 64),
                    target_mean=[60., 25., -20.],
                    target_cov=[[80., 0., 0.], [0., 40., 0.], [0., 0., 40.]],
                    blur_sigma=24.0)
    assert p.isometric is True
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_color_ceiling.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'scripts.color_ceiling'`

- [ ] **Step 3: 實作**

`scripts/color_ceiling.py`。**照抄 `scripts/objective_pilot.py` 的既有慣例**
（載入 manifest、支撐的取法、IP2P 的建法、主體錨定讀數的呼叫、逐格 CSV 的
寫法），只換掉自變數。身分讀數走 `src/metrics/identity.py` 的
`subject_identity_row()`，**不是** `embed()`：後者取畫面上面積最大的臉，
編輯輸出多長一張臉時量到的是別人。`build_param` 依 `arm` 回傳
`ChromaAffineParam(max_gain=1.0, isometric=False)`、
`ChromaAffineParam(max_gain=1.0, isometric=True)`、
或 `RegionPaletteParam(blur_sigma=...)`，未知的臂拋 `ValueError`。
每格先 `solve_amplitude(param, x01, delta_e_target)`，把
`amplitude`、`delta_e00`、`reached` 記進列裡（欄名 `amplitude`、
`final_deltaE00`、`delta_e_reached`）。

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_color_ceiling.py -q`
Expected: 3 passed

- [ ] **Step 5: 本機 CPU 冒煙測試**

Run: `python scripts/color_ceiling.py --config configs/color_ceiling.json --limit 1 --device cpu --no-attack`
Expected: 寫出一列 CSV，`hf_ratio_rgb_total` ≤ 1、`support_outside_max_abs == 0`

- [ ] **Step 6: Commit**

```bash
git add scripts/color_ceiling.py configs/color_ceiling.json tests/test_color_ceiling.py runs/color_ceiling/README.md
git commit -m "Add the colour ceiling probe: amplitude by arm with the four readouts"
```

- [ ] **Step 7: 派到遠端**

同步 repo，`source ~/env.sh` 再 `cd /nfs/home/nelson0314/WACV-s3`，
`bash scripts/free_cards.sh --assert <卡號>` 之後自己再用
`nvidia-smi --query-compute-apps=gpu_uuid,pid` 複驗一次。單格攻擊 21 秒
（fp32），格數 = 3 臂 × 3 個 ΔE00 × 5 張 × 3 類 × 3 種子，加
`undefended` 的 45 格（不掃 ΔE00）＝ 450 格，約 3 GPU-hours。
**批次跑到一半不可以清目錄。**

---

### Task 5: 元件 B1 同色碰撞

顏色映射對擴散編輯是語意上空的變換，所以改讓顏色去做它獨有的事：把指令要
改的那塊區域，在顏色統計上與周邊**合流**，讓指令找不到可指認的目標。目標
函數不需要擴散梯度，成本比 `latent_norm` 低一個數量級。

以 Lab 的前兩階矩衡量可分性（區域 R 與其膨脹環 E）：

    L_collide = || mu_R - mu_E ||^2 + || sigma_R - sigma_E ||^2

**可否證的預測**：屬性改色類指令的完成率下降，帽子與背景類不變。三類一起
掉的話，機制解釋就不是「定位」。

**Files:**
- Create: `src/defense/collision_loss.py`
- Test: `tests/test_collision_loss.py`

**Interfaces:**
- Consumes: `src/defense/ncf_param.py::rgb_to_lab`、
  `src/purify/ops.py::gaussian_blur`
- Produces: `make_collision_loss(region: torch.Tensor, ring: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]`
  （回傳純量，**要最小化**）；`ring_of(region: torch.Tensor, width: int) -> torch.Tensor`。

- [ ] **Step 1: 寫失敗的測試**

```python
# tests/test_collision_loss.py
import torch
from src.defense.collision_loss import make_collision_loss, ring_of


def _split_image():
    x = torch.zeros(1, 3, 64, 64)
    x[:, 0] = 0.9          # 左半紅
    x[:, :, :, 32:] = 0.2  # 右半暗
    return x


def _region():
    m = torch.zeros(1, 1, 64, 64)
    m[:, :, 16:48, 4:28] = 1.0
    return m


def test_ring_is_outside_the_region_and_non_empty():
    r = _region()
    ring = ring_of(r, width=6)
    assert float(ring.sum()) > 0
    assert float((ring * r).sum()) == 0


def test_loss_is_zero_when_region_and_ring_match():
    x = torch.full((1, 3, 64, 64), 0.5)
    r = _region()
    loss = make_collision_loss(r, ring_of(r, width=6))
    assert float(loss(x)) < 1e-6


def test_loss_is_positive_when_they_differ():
    x = _split_image()
    r = _region()
    loss = make_collision_loss(r, ring_of(r, width=6))
    assert float(loss(x)) > 1e-3


def test_loss_is_differentiable_and_points_downhill():
    x = _split_image().requires_grad_(True)
    r = _region()
    loss = make_collision_loss(r, ring_of(r, width=6))
    value = loss(x)
    value.backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0


def test_empty_region_raises_instead_of_returning_nan():
    x = _split_image()
    empty = torch.zeros(1, 1, 64, 64)
    try:
        make_collision_loss(empty, ring_of(_region(), width=6))(x)
    except ValueError as e:
        assert 'region' in str(e)
    else:
        raise AssertionError('空區域必須拋錯，回傳 nan 是靜默失效')
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_collision_loss.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.defense.collision_loss'`

- [ ] **Step 3: 實作**

```python
# src/defense/collision_loss.py
"""同色碰撞：把指令要改的區域在顏色統計上與周邊合流。

顏色映射對擴散編輯是語意上空的變換——可逆、不移除主體資訊、不與指令衝突，
所以模型照樣完成指令（`runs/objective_pilot/`：位移 0.28、盲評 15 格裡 14 格
判為指令完成）。這一支不再試圖把 latent 推遠，改為破壞**定位**：區域與其
周邊的顏色統計一致時，「把襯衫改成紅色」沒有可指認的襯衫。

前兩階矩而不是完整分布：一階與二階已經決定了 Monge–Kantorovitch 轉移，
與本專案既有的色彩統計用同一組量，且不需要可微的排序。
"""
from __future__ import annotations

from typing import Callable

import torch

from src.purify.ops import gaussian_blur

from .ncf_param import rgb_to_lab


def ring_of(region: torch.Tensor, width: int) -> torch.Tensor:
    """區域外的一圈環。膨脹減自己，故與區域不重疊。"""
    if width < 1:
        raise ValueError('width 必須至少為 1')
    if region.ndim != 4 or region.shape[1] != 1:
        raise ValueError('region 必須是 (1,1,H,W)')
    sigma = float(width) / 2.0
    grown = (gaussian_blur(region, sigma) > 1e-3).to(region.dtype)
    return (grown - region).clamp(0.0, 1.0)


def _moments(lab: torch.Tensor, mask: torch.Tensor, what: str):
    w = mask.to(lab.dtype)
    total = w.sum()
    if float(total) < 2.0:
        raise ValueError(f'{what} 的權重不足兩個像素，統計沒有意義')
    mean = (lab * w).sum(dim=(0, 2, 3)) / total
    var = (((lab - mean[None, :, None, None]) ** 2) * w).sum(dim=(0, 2, 3)) / total
    return mean, var.clamp_min(1e-12).sqrt()


def make_collision_loss(region: torch.Tensor, ring: torch.Tensor
                        ) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `loss(x01) -> 純量`，**要最小化**。

    Lab 單位下 mu 與 sigma 同尺度，故兩項不另設權重；加權會多一個沒有理由
    的超參數。
    """
    def loss(x01: torch.Tensor) -> torch.Tensor:
        lab = rgb_to_lab(x01)
        mu_r, sd_r = _moments(lab, region, 'region')
        mu_e, sd_e = _moments(lab, ring, 'ring')
        return (mu_r - mu_e).pow(2).sum() + (sd_r - sd_e).pow(2).sum()
    return loss
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_collision_loss.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/defense/collision_loss.py tests/test_collision_loss.py
git commit -m "Add the colour collision objective: merge the target region into its surround"
```

---

### Task 6: B1 進派工，與 A 的工作點同表

**Files:**
- Modify: `scripts/color_ceiling.py`（新增 `collision` 臂）
- Modify: `configs/color_ceiling.json`
- Modify: `runs/color_ceiling/README.md`
- Test: `tests/test_color_ceiling.py`（追加一筆）

**Interfaces:**
- Consumes: Task 5 的 `make_collision_loss`、`ring_of`；Task 4 的 `COLUMNS`
- Produces: `collision` 臂，訓練走 `src/defense/param_pgd.py::run_param_pgd`，
  CSV 多兩欄 `collision_first`、`collision_last`。

- [ ] **Step 1: 追加失敗的測試**

```python
def test_collision_arm_records_its_objective_trace():
    from scripts.color_ceiling import COLUMNS
    assert 'collision_first' in COLUMNS
    assert 'collision_last' in COLUMNS
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_color_ceiling.py -q`
Expected: FAIL，`assert 'collision_first' in COLUMNS`

- [ ] **Step 3: 實作**

`COLUMNS` 加兩欄；`build_param('collision', ...)` 回傳
`ChromaAffineParam(max_gain=1.0, isometric=True)`（等距臂當載體，因為它的
可達色相最寬）。訓練走既有的 `run_param_pgd`，損失是
`make_collision_loss(region, ring_of(region, width=16))`。`region` 依指令類取：
衣物走 `carrier_mask(x, "clothes")`，背景走它的補集，頭部配件走
`face_subject_mask` 的膨脹環。注意 `run_param_pgd` 在讀 `params()` 之前會先
`param.reset(x01, seed)`，訓練前做的初始化會被抹掉且不拋錯——幅度要在
`reset` 之後再解。

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_color_ceiling.py -q`
Expected: 4 passed

- [ ] **Step 5: 本機 CPU 冒煙測試**

Run: `python scripts/color_ceiling.py --config configs/color_ceiling.json --arms collision --limit 1 --device cpu --steps 5 --no-attack`
Expected: `collision_last < collision_first`，`hf_ratio_rgb_total` ≤ 1

- [ ] **Step 6: Commit**

```bash
git add scripts/color_ceiling.py configs/color_ceiling.json tests/test_color_ceiling.py runs/color_ceiling/README.md
git commit -m "Wire the collision arm into the colour ceiling probe"
```

---

### Task 7: 淨化那一趟

只讀已存的防禦圖，不重跑攻擊。

**Files:**
- Modify: `runs/color_ceiling/README.md`

- [ ] **Step 1: 跑保留率**

Run: `python scripts/phase_retention.py --runs runs/color_ceiling --ops identity,blur,jpeg75,jpeg30,crop_resize,jpeg_then_resize,noise,quantize,rotate,adverse_cleaner,gridpure,fdpure`
Expected: 逐格總增益與淨增益兩欄並列，表尾印空白地板的絕對值。幾何類算子
走 `LPIPS(編輯(p(原圖)), 編輯(p(防禦圖)))` 的參照，`reference` 欄逐列標明。

- [ ] **Step 2: 把數字與圖寫進 README，不下成敗結論**

`runs/color_ceiling/README.md` 記設定、目錄、量測欄位與中位數。防禦圖、
淨化圖、編輯圖都要留下並呈現。**不寫「成立／不成立」「值得／不值得再跑」。**

- [ ] **Step 3: Commit**

```bash
git add runs/color_ceiling/README.md
git commit -m "Record the purification pass over the colour ceiling probe"
```
