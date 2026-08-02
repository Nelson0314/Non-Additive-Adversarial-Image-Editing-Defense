"""產生 targeted 模式的目標影像：512² 的中性灰。

取 RGB 128/255，與 PhotoGuard 的 encoder attack 一致（Salman et al.,
arXiv:2302.06588，其目標是讓模型把受保護影像看成一張灰圖）。

`targeted` 的 `L_def = LPIPS(y_def, y_target)` 是**最小化**，本來就有界，
故不需要 hinge——`margin` 對該模式完全不起作用。這一點在報告中必須寫清楚，
否則 `targeted` 那幾格的 margin 欄位會被誤讀成「設得太小所以沒發展起來」。

執行：python scripts/make_target_gray.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    out = ROOT / "data" / "targets"
    out.mkdir(parents=True, exist_ok=True)
    a = np.full((512, 512, 3), 128, dtype=np.uint8)
    Image.fromarray(a).save(out / "gray.png")
    print(f"[target] 寫出 {out / 'gray.png'}（512x512 RGB 128）")


if __name__ == "__main__":
    main()
