"""IP2P 與 inpainting 編輯：`immunization_core.pipelines.editing`，`--data` 預設 `data/portraits`，`--out` 預設 `artifacts/undefended_edits`。"""
from immunization_color import layout
from immunization_color.cli._defaults import run_with_defaults
from immunization_core.pipelines import editing


def main(argv=None) -> None:
    run_with_defaults(editing.main, {"--data": layout.PORTRAITS, "--out": layout.UNDEFENDED_EDITS}, argv)


if __name__ == "__main__":
    main()
