"""IP2P 與 inpainting 編輯：`immunization_core.pipelines.editing`，`--data` 預設 `data/portraits`，`--out` 預設 `artifacts/undefended_edits`。"""
from immunization_core.pipelines import editing
from immunization_style import layout
from immunization_style.cli._defaults import run_with_defaults


def main(argv=None) -> None:
    run_with_defaults(editing.main, {"--data": layout.PORTRAITS, "--out": layout.UNDEFENDED_EDITS}, argv)


if __name__ == "__main__":
    main()
