"""複合模塊 — 把數個注入位置合成單一 φ。`docs/ARCH_2026-08-05.md` §3.2。

## 為什麼需要它

APA 移植（條件 N3）同時需要兩個位置：

| 階段 | 位置 | 能力 |
|---|---|---|
| 一：視覺一致性 | LoRA（`site_weight.py`） | `patches_model` |
| 二：攻擊有效性 | latent 注入（`site_latent.py`） | `eps_hook` |

`generator.py` 的檢查是「四種能力至少提供其一」，並未禁止同時提供兩種——
但在此之前沒有任何模塊這樣做，該路徑從未被驗證過。

處置是新增本模塊而**不改 `generator.py`**：後者依能力分派、不比對 site 名稱，
複合模塊只要如實回報自己聚合後的能力，分派邏輯就不需要知道成員有幾個。
在此加 `if isinstance(module, CompositeResidual)` 會直接違反該設計。

## 能力的聚合規則

| 能力 | 聚合方式 | 理由 |
|---|---|---|
| `pixel_residual` | 依序串接，前者的輸出餵給後者 | 像素變換可以合成 |
| `eps_hook` | 依序套用於同一步的 ε̂ | 每步的修正量疊加 |
| `emb_residual` | 相加 | 三者都是「要加到嵌入上的殘差」 |
| `patches_model` | 任一為真即為真 | 只要有人改動模型，φ 就進得了計算圖 |

順序即 `members` 的順序，且**不排序**——串接不可交換，
自動排序會讓「換一個順序」變成看不見的變因。

## 停用與逐位元等同

`disable()` 廣播到全部成員。這條性質必須是逐位元的，不是近似的：
φ = 0 的對照 `x_base = G(x; φ=0)` 是全部保真度量測的基準，
若它與「模塊不存在」有任何差異，該差異會被算進防禦的失真預算裡。
"""

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from src.residual.base import ResidualModule


class CompositeResidual(ResidualModule):
    site = "composite"

    def __init__(self, members: List[ResidualModule], names: List[str]):
        super().__init__()
        if len(members) != len(names):
            raise ValueError(
                f"members 有 {len(members)} 個但 names 有 {len(names)} 個；"
                "兩者必須一一對應，否則 param_groups() 的分組會錯位"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"組名重複：{names}。param_groups() 的鍵會互相覆蓋")
        if not members:
            raise ValueError("複合模塊至少要有一個成員")
        # ModuleList 而非 list：後者不會被 nn.Module 認出，
        # `parameters()` 會回傳空的，優化器拿不到任何參數而靜默不更新。
        self.members = nn.ModuleList(members)
        self.names = list(names)

    # ---- 開關（廣播）----

    def enable(self) -> None:
        super().enable()
        for m in self.members:
            m.enable()

    def disable(self) -> None:
        super().disable()
        for m in self.members:
            m.disable()

    def remove(self) -> None:
        for m in self.members:
            m.remove()

    # ---- 能力聚合 ----

    def pixel_residual(self, x01: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return x01 if self._any_pixel() else None
        out = None
        cur = x01
        for m in self.members:
            r = m.pixel_residual(cur)
            if r is not None:
                out, cur = r, r
        return out

    def _any_pixel(self) -> bool:
        """停用時仍需回答「本模塊是不是像素側的」。

        不能靠呼叫 `pixel_residual` 來判斷——停用的成員回傳什麼由各自決定
        （`site_warp` 回傳 x01、基底回傳 None），那會讓能力判定隨開關狀態改變，
        而 `generator.prepare()` 正是用它來決定走哪條路徑的。
        """
        return any(type(m).pixel_residual is not ResidualModule.pixel_residual
                   for m in self.members)

    def eps_hook(self, ts: torch.Tensor, steps: int) -> Optional[Callable]:
        if not self.enabled:
            return None
        hooks = [h for h in (m.eps_hook(ts, steps) for m in self.members)
                 if h is not None]
        if not hooks:
            return None
        if len(hooks) == 1:
            return hooks[0]

        def chained(eps: torch.Tensor, step_idx: int, t: torch.Tensor):
            for h in hooks:
                eps = h(eps, step_idx, t)
            return eps

        return chained

    def emb_residual(self, emb: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        total = None
        for m in self.members:
            d = m.emb_residual(emb)
            if d is not None:
                total = d if total is None else total + d
        return total

    def patches_model(self) -> bool:
        return any(m.patches_model() for m in self.members)

    # ---- 參數分組 ----

    def param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """{組名: 該成員的參數}。APA 用它分階段優化。

        成員自己若也是複合模塊，其分組會以 `外層名/內層名` 展開，
        避免巢狀時鍵互相覆蓋。
        """
        groups: Dict[str, List[nn.Parameter]] = {}
        for name, m in zip(self.names, self.members):
            sub = m.param_groups()
            if list(sub) == ["default"]:
                groups[name] = sub["default"]
            else:
                for k, v in sub.items():
                    groups[f"{name}/{k}"] = v
        return groups

    # ---- 診斷 ----

    def raw_residual(self) -> None:
        """複合模塊沒有單一的「注入前像素殘差」——成員各自的意義不同，
        相加或串接後的量不對應任何一個位置的參數化。逐成員的診斷請直接
        取用 `members`。"""
        return None

    def rank_trace(self, ts=None, steps=None) -> list:
        out = []
        for m in self.members:
            out.extend(m.rank_trace(ts, steps))
        return out

    def __repr__(self) -> str:
        inner = ", ".join(f"{n}={type(m).__name__}(site {m.site})"
                          for n, m in zip(self.names, self.members))
        return f"CompositeResidual({inner})"
