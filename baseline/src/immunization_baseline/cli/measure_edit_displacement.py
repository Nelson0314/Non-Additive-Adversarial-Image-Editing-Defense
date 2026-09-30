"""編輯位移：`immunization_core.pipelines.displacement`，`--data-root` 預設 `data/portraits`。"""
from immunization_baseline import layout
from immunization_baseline.cli._defaults import run_with_defaults
from immunization_core.pipelines import displacement


def main(argv=None) -> None:
    run_with_defaults(displacement.main, {"--data-root": layout.PORTRAITS}, argv)


if __name__ == "__main__":
    main()
