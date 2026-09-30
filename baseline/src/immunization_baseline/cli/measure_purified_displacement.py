"""淨化後的位移保留量：`immunization_core.pipelines.retention`，`--data` 預設 `data/portraits`。"""
from immunization_baseline import layout
from immunization_baseline.cli._defaults import run_with_defaults
from immunization_core.pipelines import retention


def main(argv=None) -> None:
    run_with_defaults(retention.main, {"--data": layout.PORTRAITS}, argv)


if __name__ == "__main__":
    main()
