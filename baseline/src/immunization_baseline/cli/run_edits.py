"""IP2P 與 inpainting 編輯：`immunization_core.pipelines.editing`，預設資料根與輸出為本專案版面。

`--data` 預設 `data/portraits`，`--out` 預設 `artifacts/undefended_edits`；其餘參數
與說明見 `python -m immunization_core.pipelines.editing --help`。
"""
from immunization_baseline import layout
from immunization_baseline.cli._defaults import run_with_defaults
from immunization_core.pipelines import editing


def main(argv=None) -> None:
    run_with_defaults(editing.main, {"--data": layout.PORTRAITS,
                                     "--out": layout.UNDEFENDED_EDITS}, argv)


if __name__ == "__main__":
    main()
