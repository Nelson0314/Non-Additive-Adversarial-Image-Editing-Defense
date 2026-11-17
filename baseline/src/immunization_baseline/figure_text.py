"""圖內文字的替換與字級對應：包裝其他專案的繪圖函式時使用，不修改其原始碼。

`relabel(mapping, sizes)` 在範圍內替換 `matplotlib.axes.Axes.text` 的字串（整串比對）並依 `sizes` 改字級：
`sizes` 為 {原字級: 新字級}，未給 fontsize 的呼叫以 `sizes[None]` 為新字級（缺此鍵則沿用 rcParams）。
範圍結束後還原 `Axes.text`。
"""
from contextlib import contextmanager

import matplotlib.axes


@contextmanager
def relabel(mapping, sizes=None):
    sizes = sizes or {}
    original = matplotlib.axes.Axes.text
    used = set()

    def text(self, x, y, s, *args, **kwargs):
        if s in mapping:
            used.add(s)
            s = mapping[s]
        key = kwargs.get("fontsize", kwargs.get("size"))
        if key in sizes:
            kwargs.pop("size", None)
            kwargs["fontsize"] = sizes[key]
        return original(self, x, y, s, *args, **kwargs)

    matplotlib.axes.Axes.text = text
    try:
        yield
    finally:
        matplotlib.axes.Axes.text = original
    unused = set(mapping) - used
    if unused:
        raise SystemExit(f"圖中沒有這些要替換的字串：{sorted(unused)}")
