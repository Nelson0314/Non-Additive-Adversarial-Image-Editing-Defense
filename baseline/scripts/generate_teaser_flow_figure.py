"""論文 Fig. 1 teaser（流程版面）以論文正文用詞重出：呼叫 color trial tools/generate_teaser_flow_figure_compact.py 的 main()
（只呼叫、不修改），圖內 "No defense" 改為 "Undefended"、欄標 "Protected\\nimage" 改為 "Immunized\\nimage"；版面、字級與內容不變。
其餘參數原樣轉給該工具（--run-dir、--prompt-index、--purification、--baseline、--name 等），輸出改到 --output-root。
Fig. 1 的設定（color trial HANDOVER「Fig. 1 teaser（殭屍）」）：
    --run-dir artifacts/teaser/teaser_man_x_zombieskin_tg7p5_ig1p8 --prompt-index 0 --purification jpeg30
    --baseline photoguard_encoder dia_pt --name teaser_zombie_compact
用法（遠端 baseline 專案根，CPU）：$PY scripts/generate_teaser_flow_figure.py [--output-root <目錄>] <該工具的參數>
"""
import importlib.util
import sys
from pathlib import Path

from _figure_text import relabel

TOOL = Path("../color/trials/lut3d_defense_and_naturalness/tools/generate_teaser_flow_figure_compact.py")
LABELS = {"No defense": "Undefended", "Protected\nimage": "Immunized\nimage"}


def main():
    args = sys.argv[1:]
    out = Path("artifacts/teaser_paper_format")
    if "--output-root" in args:
        i = args.index("--output-root")
        out = Path(args[i + 1])
        del args[i:i + 2]
    spec = importlib.util.spec_from_file_location("teaser_flow_compact", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    tool.OUT = out
    sys.argv = [str(TOOL), *args]
    with relabel(LABELS):
        tool.main()


if __name__ == "__main__":
    main()
