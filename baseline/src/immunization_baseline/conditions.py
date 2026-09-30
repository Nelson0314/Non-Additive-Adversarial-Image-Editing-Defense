"""baseline 防禦條件的讀取：唯一正本為 `configs/conditions.yaml`。

`CONDITIONS` 依檔中順序列出全部條件的設定；`main_table_conditions()` 為主表的十二列；
`solver_conditions()` 為本專案自行求解的條件（排除 imported）。
`python -m immunization_baseline.conditions [--main-table|--solvable]` 印出空白分隔的條件名。
"""
from __future__ import annotations

import argparse

import yaml

from immunization_baseline import layout

CONDITIONS_FILE = layout.CONFIGS / "conditions.yaml"
SOLVERS = ("pgd", "dct_shield", "feedforward", "imported")


def _load() -> tuple:
    spec = yaml.safe_load(CONDITIONS_FILE.read_text(encoding="utf-8"))
    conditions = spec["conditions"]
    for name, entry in conditions.items():
        if entry.get("solver") not in SOLVERS:
            raise ValueError(f"{CONDITIONS_FILE}：{name} 的 solver 必須是 {SOLVERS}")
        if entry["solver"] == "pgd" and "spec" not in entry:
            raise ValueError(f"{CONDITIONS_FILE}：{name} 缺 spec")
        if entry["solver"] != "imported" and "solver_prompt" not in entry:
            raise ValueError(f"{CONDITIONS_FILE}：{name} 缺 solver_prompt")
        if entry["solver"] == "imported" and entry["main_table"]:
            missing = {"spec_source", "solver_prompt", "solver_prompt_source"} - set(entry)
            if missing:
                raise ValueError(f"{CONDITIONS_FILE}：匯入條件 {name} 缺 {sorted(missing)}")
    return conditions, spec["content_prompt_source"]


CONDITIONS, CONTENT_PROMPT_SOURCE = _load()


def main_table_conditions() -> list:
    return [name for name, entry in CONDITIONS.items() if entry["main_table"]]


def solver_conditions() -> list:
    return [name for name, entry in CONDITIONS.items() if entry["solver"] != "imported"]


def conditions_of(solver: str) -> list:
    return [name for name, entry in CONDITIONS.items() if entry["solver"] == solver]


def pgd_spec(name: str):
    """`模組.屬性` 解析為 immunization_baseline.attacks 下的 BaselineSpec。"""
    import importlib

    module, attribute = CONDITIONS[name]["spec"].split(".")
    return getattr(importlib.import_module(f"immunization_baseline.attacks.{module}"), attribute)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--main-table", action="store_true")
    group.add_argument("--solvable", action="store_true")
    args = parser.parse_args(argv)
    names = (main_table_conditions() if args.main_table else
             solver_conditions() if args.solvable else list(CONDITIONS))
    print(" ".join(names))


if __name__ == "__main__":
    main()
