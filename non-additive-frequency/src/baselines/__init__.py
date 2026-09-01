"""baseline 攻擊層：五篇論文，一個 PGD 骨幹。

**這些是別人論文的方法，不是我們的。** 每一個超參數、每一個損失項、每一個
值域轉換都必須有查證過的出處（`docs/reference/SOURCE_AUDIT.md` 與
`docs/_audit_*.md`）。查不到的項目一律在 `prepare` 中拋出
`NotImplementedError` 並寫明缺什麼——填一個看起來合理的值會讓整批實驗的
比較基礎失效，且事後補不回來。

用法：

    from src.baselines import REGISTRY, run_pgd

    spec = REGISTRY["mist"]
    res = run_pgd(sd, x01, spec, target01=mist_png, strength=0.3)
    x_def = res.x_adv01

六個項目（DIA 有兩個變體）。`modified_from_paper` 為真者，報表必須在該列
加註 `modification_note`，否則會被讀成原論文設定：

| 鍵 | 改寫 | 原生 eps（`[0,1]` 尺度） |
|---|---|---|
| `photoguard_c` | 是（移植為 img2img） | L2 半徑 8 |
| `mist` | 否 | L∞ 16/255 |
| `dia_pt` | 否 | L∞ 0.025 |
| `dia_r` | 否 | L∞ 0.025 |
| `advpaint` | 是（全圖 mask、移植為 img2img） | L∞ 0.03 |
| `promptflare` | 是（全圖 mask、移植為 img2img） | L∞ 6/255 |
"""

from src.baselines import advpaint, dia, mist, photoguard, promptflare
from src.baselines.pgd import (
    BaselineSpec,
    PGDResult,
    ValueRange,
    initial_point,
    project,
    run_pgd,
    step_size_at,
    update_direction,
)

_SPECS = (
    photoguard.SPEC,
    mist.SPEC,
    dia.SPEC_PT,
    dia.SPEC_R,
    advpaint.SPEC,
    promptflare.SPEC,
)

REGISTRY = {s.name: s for s in _SPECS}

# 鍵與 spec 自己的 name 必須一致。不一致時報表會以鍵標示、而 spec 的
# discrepancy_note 以 name 標示，兩者對不起來。
assert len(REGISTRY) == len(_SPECS), "REGISTRY 有重複的 name"


def get(name: str) -> BaselineSpec:
    if name not in REGISTRY:
        raise KeyError(f"未知的 baseline {name!r}；可用的是 {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = [
    "BaselineSpec",
    "PGDResult",
    "REGISTRY",
    "ValueRange",
    "get",
    "initial_point",
    "project",
    "run_pgd",
    "step_size_at",
    "update_direction",
]
