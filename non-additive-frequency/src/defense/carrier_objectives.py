"""pilot 的四個臂：三個目標函數加一個同可達集合的隨機對照。

四個臂
────────────────────────────────────────────────────────────────────
    latent_norm      ‖E(x')‖₂ → 0，DCT-Shield §4.2 的目標
    cfg_shift        壓完整三分支引導預測對影像條件的依賴
    image_guidance   只壓 s_I 那一項（cfg_shift 的子集，留作消融）
    edit_task        現行的解碼輸出指令完成度（對照現況）

隨機對照不是第五個目標函數，是**同一個載體不最佳化**：`freeze_as_control`
把 `params()` 清空，`run_param_pgd` 於是一步都不走，防禦圖就是初始化那一張。
配 `box_corner_init` 時它抽的是**角點**，與 sign 更新的可達集合相同——
`color_param.RAND_DRAWS` 上方那段記著為什麼不能抽盒內均勻：低自由度的
參數化上均勻抽樣互相抵消，失真只到角點的三分之一，兩條曲線不重疊，
等失真內插整片 `out_of_range`，那不是「贏過隨機」是根本沒比到。

`latent_norm` 直接沿用 `src/baselines/dct_shield.make_latent_norm_loss`，
不另寫一份：那是論文式 `L(δ) = ‖E(x')‖₂`（L2 範數，不是均方），兩份各自
維護時其中一份改了不會有症狀。
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from src.baselines.dct_shield import make_latent_norm_loss
from src.defense.cfg_shift_loss import make_cfg_shift_loss
from src.defense.color_param import RAND_DRAWS, _draw
from src.defense.image_guidance_loss import make_image_guidance_loss

OBJECTIVES = ("latent_norm", "cfg_shift", "image_guidance", "edit_task")


class StepwiseObjective:
    """把無狀態的 `loss(x)` 包成訓練迴圈要的 `set_step` 介面。

    `edit_task_training.run_edit_task_training` 每一步呼叫 `loss.set_step(i)`，
    因為 `EditTaskLoss` 用步數索引決定這一步跑哪個任務、抽哪個噪聲。其餘三個
    目標自己帶生成器、不需要步數，但介面必須一致，否則訓練迴圈得為每個臂分支，
    而分支正是本專案記過的靜默失效來源（欄位清單兩處各自維護那一則）。

    `fixed` 是與訓練目標**同一條路徑**的決定性評估函數，用來判收斂；
    `None` 表示這個目標本身就是決定性的（`latent_norm` 沒有抽樣）。
    """

    def __init__(self, name: str, fn: Callable[[torch.Tensor], torch.Tensor],
                 fixed: Optional[Callable] = None):
        self.name = str(name)
        self._fn = fn
        self.fixed = fixed if fixed is not None else fn
        self.step = 0

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x)


def make_objective(name: str, ip2p, *, x_clean: torch.Tensor,
                   zt_mode: Optional[str] = None,
                   text_embeds: Optional[torch.Tensor] = None,
                   s_t: Optional[float] = None, s_i: Optional[float] = None,
                   t_min: int = 1, t_max: int = 1000, samples: int = 1,
                   seed: int = 0, normalise: bool = False,
                   eval_draws: int = 8, eval_seed: int = 90001):
    """依名稱造一個臂。缺了該臂必要的參數就當場拋錯，不填合理預設。

    `edit_task` 不在這裡造——它需要任務清單與編輯選項，由
    `edit_task_loss.EditTaskLoss` 自己組，且它本來就有 `set_step`。
    這裡接受這個名字只是為了讓 `OBJECTIVES` 是完整清單，呼叫到會明確拒絕。
    """
    if name not in OBJECTIVES:
        raise ValueError(f"未知的目標 {name!r}，必須是 {OBJECTIVES}")
    if name == "edit_task":
        raise ValueError(
            "edit_task 由 EditTaskLoss 自己組（它需要任務清單與編輯選項），"
            "不經過本工廠")

    if name == "latent_norm":
        return StepwiseObjective(name, make_latent_norm_loss(ip2p))

    if zt_mode is None:
        raise ValueError(
            f"{name} 需要 zt_mode：IP2P 由純噪聲起步，中間步的 z_t 分布依賴"
            "條件、無法解析，兩個候選都是近似，故必填")
    common = dict(zt_mode=zt_mode, x_clean=x_clean, t_min=t_min, t_max=t_max,
                  samples=samples, seed=seed, normalise=normalise)
    if name == "image_guidance":
        fn = make_image_guidance_loss(ip2p, text_embeds=text_embeds, **common)
    else:
        if s_t is None or s_i is None:
            raise ValueError(
                "cfg_shift 需要 s_t 與 s_i：本項壓的是實際取樣式，強度必須是"
                "攻擊者真的會用的那一組")
        fn = make_cfg_shift_loss(ip2p, text_embeds=text_embeds,
                                 s_t=s_t, s_i=s_i, **common)
    return StepwiseObjective(name, fn, fn.make_fixed(eval_draws, eval_seed))


def box_corner_init(param, x01: torch.Tensor, seed: int, *,
                  draw: str = "corner") -> None:
    """把可學參數移到 ±radius 盒角再投影，就地修改 `param`。

    做法：在 `reset` 之後對每個參數張量**加上**一組 ±radius 的角點抽樣，
    再交給 `param.project()` 執行該載體自己的約束。之所以是「加上」而不是
    「設成」——兩個載體的 `reset` 都把可學參數放在盒心（`NCFColorParam` 的
    `delta`／`u` 是零，`ReColorAdvParam` 的 `grid` 是恆等座標），所以加一組
    ±radius 就精確落在盒角；換成別的載體時這個前提要重驗。

    **得到的點不是可達集合的邊界。** 顏色載體的可達集合不是盒子：`project()`
    先夾 L∞ 盒、再過 `epsilon_lab` 的效果球與奇異值上界，角點會被拉回集合
    內部；而真正決定顏色走多遠的是 `T0`（配色）與 `amplitude`，兩者都不在
    `delta` 的盒子裡。引用這一支產生的對照時要照這個定義寫，不可寫成
    「可達集合的邊界」。
    """
    if draw not in RAND_DRAWS:
        raise ValueError(f"draw 只能是 {RAND_DRAWS}，收到 {draw!r}")
    radius = getattr(param, "radius", None)
    if radius is None:
        raise AttributeError(
            f"{type(param).__name__} 沒有 radius，邊界的位置沒有定義；"
            "不要猜一個尺度")
    ps = param.params()
    if not ps:
        raise ValueError("這個載體沒有可學參數，談不上起點")
    with torch.no_grad():
        for k, p in enumerate(ps):
            step = _draw(tuple(p.shape), -float(radius), float(radius), draw,
                         int(seed) + 1009 * k)
            p.add_(step.to(device=p.device, dtype=p.dtype))
    param.project()


class BoxCornerStartParam:
    """把邊界初始化搬進載體自己的 `reset`，其餘全部委派給內層。

    **為什麼不能在訓練前先做一次就好**：`run_param_pgd` 在讀 `params()` 之前
    會先呼叫 `param.reset(x01, seed)`（`param_pgd.py:767`），把先做好的角點抽樣
    整個抹掉，於是邊界臂安靜地從恆等起點跑、隨機對照安靜地變成「沒有防禦」。
    兩者都不會拋錯，也不會在讀數上長得像失效。

    這個形狀與 `color_param.ColorCurveRandomParam.reset`（先 `super().reset`
    再抽）相同，只是改成委派而不是繼承，因為 pilot 有兩個不同的載體類別。
    """

    def __init__(self, inner, *, draw: str = "corner"):
        if draw not in RAND_DRAWS:
            raise ValueError(f"draw 只能是 {RAND_DRAWS}，收到 {draw!r}")
        self._inner = inner
        self._draw = draw

    @property
    def name(self) -> str:
        return f"{self._inner.name}_edge"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self._inner.reset(x01, seed)
        box_corner_init(self._inner, x01, seed, draw=self._draw)

    def params(self):
        return self._inner.params()

    def __getattr__(self, item):
        return getattr(self._inner, item)


class FrozenParam:
    """同一個載體、不最佳化：`params()` 為空，其餘全部委派給內層。

    `run_param_pgd` 在 `if not ps` 直接返回，於是防禦圖就是初始化那一張。
    這樣寫而不是為每個載體各寫一個 `*RandomParam` 子類，是因為 pilot 有兩個
    載體（NCF 與 ReColorAdv）而隨機對照的語意只有一句話：不要動它。
    """

    def __init__(self, inner):
        if not inner.params():
            raise ValueError("內層載體本來就沒有可學參數，不需要再凍結一次")
        self._inner = inner

    @property
    def name(self) -> str:
        return f"{self._inner.name}_frozen"

    def params(self):
        return []

    def __getattr__(self, item):
        return getattr(self._inner, item)


def freeze_as_control(param, *, draw: str = "corner") -> FrozenParam:
    return FrozenParam(BoxCornerStartParam(param, draw=draw))


class WithRegulariser:
    """目標函數加上一個載體自己的正則項，介面與 `StepwiseObjective` 相同。

    正則項也進固定評估：收斂監看的必須是**正在被最佳化的那個量**，否則曲線
    走平不代表訓練走平。
    """

    def __init__(self, inner, term: Callable[[], torch.Tensor], weight: float):
        if weight <= 0:
            raise ValueError("weight 必須為正；不要用 0 權重假裝接上了正則項")
        self._inner, self._term, self.weight = inner, term, float(weight)
        self.name = f"{inner.name}+reg{weight:g}"

    def set_step(self, step: int) -> None:
        self._inner.set_step(step)

    @property
    def step(self) -> int:
        return self._inner.step

    def fixed(self, x: torch.Tensor):
        return self._inner.fixed(x) + self.weight * self._term()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner(x) + self.weight * self._term()
