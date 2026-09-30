"""把 baseline 版面的預設值補進共用 pipeline 的必填參數。"""
from __future__ import annotations

import sys
from typing import Callable, Mapping, Optional, Sequence


def run_with_defaults(main: Callable[[list], None], defaults: Mapping[str, object],
                      argv: Optional[Sequence[str]] = None) -> None:
    """呼叫端未給的旗標以 `defaults` 補上，其餘參數原樣交給 `main`。"""
    args = list(sys.argv[1:] if argv is None else argv)
    given = {a.split("=", 1)[0] for a in args if a.startswith("--")}
    prefix = []
    for flag, value in defaults.items():
        if flag not in given:
            prefix += [flag, str(value)]
    main(prefix + args)
