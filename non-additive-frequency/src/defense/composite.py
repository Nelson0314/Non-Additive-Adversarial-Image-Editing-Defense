"""把多個載體串起來：整圖濾鏡與衣物換色同時作用。

為什麼要串
────────────────────────────────────────────────────────────────────
先前的分工是「衣物類指令由衣物載體處理，頭部與背景類由整圖濾鏡處理」，一格
只有一個載體在作用。`runs/color_scaleup_search/` 量到那組分工裡最無效的正是
衣物類：指令本身就是改色，防禦寫進色度就被攻擊整個覆蓋掉，防禦後的編輯圖與
未防禦的編輯圖位移中位數只有 0.02。

串起來之後，兩個支撐的參數各自獨立，指令類別不再決定用哪一個載體。

順序與統計
────────────────────────────────────────────────────────────────────
第 k 段看到的是第 k−1 段的輸出，`reset` 也在那張圖上做，所以每一段的
Monge–Kantorovitch 統計取自它實際要變換的那張圖。**代價是那組統計在
`reset` 之後就固定了**：前一段的參數之後被搜尋改動時，後一段的來源統計是
舊的。這不影響可行性（後一段的映射照樣套用），但它是一個近似，量到的
`support_deltaE00` 仍以最終輸出為準。

受保護像素
────────────────────────────────────────────────────────────────────
支撐外逐位元不動這個性質**只在每一段各自的支撐上成立**，串接之後的保護區是
所有段支撐的交集的補集。整圖濾鏡的支撐是整張照片，所以只要它在串裡，就沒有
逐位元不動的像素；該側要看的是身分讀數與 `gate_*` 欄位。
"""
from __future__ import annotations

from typing import List

import torch

from .lowfreq_color import highfreq_report


class CompositeParam:
    """依序套用多個載體。每一段都要有 `reset`／`render`／`params`／`project`。"""

    name = 'composite'

    def __init__(self, stages, tags=None):
        stages = list(stages)
        if len(stages) < 2:
            raise ValueError('至少要兩段，否則直接用那一段本身')
        self.stages = stages
        self.tags = list(tags) if tags is not None else [
            getattr(s, 'name', f'stage{k}') for k, s in enumerate(stages)]
        if len(self.tags) != len(self.stages):
            raise ValueError('tags 的長度要與 stages 相同')

    @property
    def radius(self):
        rs = [getattr(s, 'radius', None) for s in self.stages]
        rs = [r for r in rs if r is not None]
        return max(rs) if rs else None

    @property
    def amplitude(self):
        return [getattr(s, 'amplitude', 1.0) for s in self.stages]

    def set_amplitude(self, a):
        """純量廣播到每一段；序列則逐段指定。"""
        if isinstance(a, (list, tuple)):
            if len(a) != len(self.stages):
                raise ValueError('逐段指定幅度時長度要與 stages 相同')
            for s, v in zip(self.stages, a):
                s.set_amplitude(v)
            return
        for s in self.stages:
            s.set_amplitude(a)

    def reset(self, x01, seed=0):
        current = x01
        for s in self.stages:
            s.reset(current, seed)
            current = s.render(current).detach()

    def params(self) -> List[torch.Tensor]:
        out = []
        for s in self.stages:
            out.extend(s.params())
        return out

    def project(self):
        for s in self.stages:
            s.project()

    def render(self, x):
        current = x
        for s in self.stages:
            current = s.render(current)
        return current

    def state_dict(self):
        return {'stages': [s.state_dict() for s in self.stages]}

    def load_state_dict(self, state):
        for s, d in zip(self.stages, state['stages']):
            s.load_state_dict(d)

    @torch.no_grad()
    def diagnostics(self, x):
        row = {'name': self.name, 'stages': '|'.join(self.tags)}
        current = x
        for tag, s in zip(self.tags, self.stages):
            for k, v in s.diagnostics(current).items():
                row[f'{tag}_{k}'] = v
            current = s.render(current)
        row.update(highfreq_report(x, self.render(x)))
        return row
