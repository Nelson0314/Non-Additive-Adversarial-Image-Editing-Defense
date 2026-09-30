"""color 防禦條件的讀取：唯一正本為 `configs/conditions.yaml`。

`python -m immunization_color.conditions <條件名>` 逐行印出該條件的額外參數，
未知條件以結束碼 2 拒絕；`--list` 印出全部條件名。
"""
from __future__ import annotations

import argparse
import sys

import yaml

from immunization_color import layout

CONDITIONS_FILE = layout.PROJECT / "configs" / "conditions.yaml"
CONDITIONS = {name: [str(a) for a in args] for name, args in
              yaml.safe_load(CONDITIONS_FILE.read_text(encoding="utf-8"))["conditions"].items()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("condition", nargs="?")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print(" ".join(CONDITIONS))
        return
    if args.condition not in CONDITIONS:
        print(f"未知的條件：{args.condition}；{CONDITIONS_FILE} 中有 {list(CONDITIONS)}", file=sys.stderr)
        raise SystemExit(2)
    for item in CONDITIONS[args.condition]:
        print(item)


if __name__ == "__main__":
    main()
