"""把「要求的頻譜交不交得出來」變成損失裡的一項。

為什麼
────────────────────────────────────────────────────────────────────
`synthesize` 是一個投影。block 32、hop 8 表示每個像素由 **16 個重疊視窗**
加總，而 `theta` 在每個（視窗, 頻格）上是自由的——相鄰視窗可以帶著互相矛盾
的相位，那樣一組係數**不對應任何真實影像**，重疊相加只能交出最近的那一張。

實測（`runs/stft_consistency/`，由已存的權重重建）：

| 工作點 | `amp_dev` | `phase_rho` | 投影前的像素 L∞ |
|---|---|---|---|
| `ig_d25`（主線） | 0.175 | 0.882 | 1.09 |
| `po_r45` | 0.304 | 0.854 | 3.21 |
| `po_r60` | 0.480 | 0.834 | 6.03 |

要求的頻譜有 **18–48% 的幅度交不出來**，相位也丟掉 11–17%，而投影前的像素
值域遠超出合法的 0–1。防禦圖上那層浮雕般的刻紋就是這個投影誤差；四個
`po_*` 工作點的 `amp_dev` 排序與人眼看到的醜的排序一位不差。

這一項做什麼
────────────────────────────────────────────────────────────────────
在原本的損失上加一項

    w · ‖ |analyze(synthesize(rot))| − |rot| ‖ / ‖ |rot| ‖

於是在**損失值相同**的解裡，最佳化會偏好可實現的那些。這與「把擾動移到別的
地方」是不同的軸——挪動支撐不會讓一組互相矛盾的係數變得可實現
（`runs/ip2p_pixel_matched/README.md` 已否證那一條）。

**這不是 Griffin–Lim。** `--gl-iters` 是在**前向**做迭代投影，把解硬推到可
實現集合上，代價是它同時把效果投影掉（FND-051：壓 `amp_dev` 會壓低可達
上限）。這一項是**軟的**：它只改變偏好，最佳化仍可用效果換一致性，換多少
由 `w` 決定。兩者不可混在同一列報表上。

成本
────────────────────────────────────────────────────────────────────
每步多一次 `analyze`（`rot` 與 `x_def` 都是既有前向算過的，由呼叫端傳入）。
相對於 VAE 編碼 ＋ 兩次 UNet 前向可以忽略。

**權重 `w` 沒有出處，是本專案指定的。** 它是量綱無關的相對量，故 `w` 與
主損失的量級直接可比。
"""

from __future__ import annotations

from typing import Callable

import torch


def amplitude_deviation(rot: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    """相對幅度偏差。`rot` 是要求的頻譜，`got` 是投影後再分析回來的頻譜。

    分母用 `rot` 自己，所以這是**相對**量，跨影像與跨工作點可比，也不會因為
    影像本身的能量大小而讓權重 `w` 的意義漂掉。
    """
    if rot.shape != got.shape:
        raise ValueError(f"形狀不合：{tuple(rot.shape)} 對 {tuple(got.shape)}")
    num = (got.abs() - rot.abs()).norm()
    return num / rot.abs().norm().clamp_min(1e-12)


def weight_at(step: int, steps: int, w0: float, decay_frac: float) -> float:
    """第 `step` 步的權重。`decay_frac <= 0` 時恆為 `w0`（逐位元等於不退火）。

    為什麼要退火
    ────────────────────────────────────────────────────────────
    固定權重會把**失真封頂**。實測（`runs/arch_cons_matched`）：w=0.30 的
    半徑由 2.5 拉到 6.0（2.4 倍），DISTS 只由 0.0835 走到 0.1145，到不了
    基準線的 0.153；而編輯結果是重畫還是劣化看起來由失真水準決定——
    0.15 附近重畫、0.11 附近劣化。也就是懲罰項把圖變好看的方式，是不去
    它有效的那個地方。

    退火要問的是「懲罰是不是只在早期需要」：先用它把解推進可實現的盆地，
    再把權重線性降到零，讓失真長回 0.15 而形狀留下來。

    `decay_frac` 是**權重歸零的位置佔總步數的比例**：0.5 表示在半程歸零，
    之後與完全不加懲罰相同。**這個值沒有出處，是本專案指定的。**
    """
    if steps <= 0:
        raise ValueError(f"steps 必須為正，收到 {steps}")
    if decay_frac <= 0:
        return w0
    end = decay_frac * steps
    if step >= end:
        return 0.0
    return w0 * (1.0 - step / end)


def make_consistency_term(module_fn: Callable[[], object], x01: torch.Tensor,
                          weight: float, steps: int = 0,
                          decay_frac: float = 0.0
                          ) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `term(x_def)`，加到主損失上。`weight <= 0` 時回傳 `None`。

    **第一個參數是 getter 不是模組本身。** `PhaseParam.module` 在 `reset()`
    被呼叫之前是 `None`，而 `reset()` 發生在 `run_param_pgd` 內部——呼叫端
    在那之前拿不到模組。實際踩過：守門在建構時就讀 `param.module`，三個工作
    點全部被自己的守門擋下（`runs/ip2p_consistency/cw_*.log`）。改成延後解析，
    模組在第一次前向時才取。

    `x_def` 是前向已經算好的防禦圖，所以這裡只需要再取一次 `analyze`；`rot`
    由 `requested_spectrum` 重算——那是決定性運算，與前向逐位元相同。

    **梯度必須通得過**：`rot` 依賴 `theta`，`got` 依賴 `x_def` 而 `x_def`
    也依賴 `theta`，兩條路都要保留，否則這一項只會懲罰其中一半。
    """
    if weight <= 0:
        return None
    if decay_frac > 0 and steps <= 0:
        raise ValueError("decay_frac > 0 時必須給 steps")

    state = {"step": 0}

    def term(x_def: torch.Tensor) -> torch.Tensor:
        module = module_fn()
        if module is None or not hasattr(module, "requested_spectrum"):
            raise SystemExit(
                "一致性懲罰取不到相位模組——該參數化沒有 requested_spectrum")
        w = weight_at(state["step"], steps, weight, decay_frac) if decay_frac > 0             else weight
        if w == 0.0:
            # 權重歸零之後整項不算，省下每步一次 analyze——退火的後半段因此
            # 與完全不加懲罰**同樣快**，而不是算出零再加上去。
            return x_def.sum() * 0.0
        rot = module.requested_spectrum(x01)
        got = module.analyze(x_def)
        return w * amplitude_deviation(rot, got)

    term.advance = lambda i: state.__setitem__("step", i)
    return term
