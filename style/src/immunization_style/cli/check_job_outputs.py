"""驗收 `scripts/run_style_prompt_jobs.sh` 各階段的輸出；不載入模型，只讀 CSV 與檔案。

子命令（結束碼 0 為通過，1 為輸出不完整，並於 stderr 列出缺少的鍵或檔案）

    defense  --dir <防禦輸出目錄> --style <風格> --images <影像>...
        `results.csv` 對每張影像各有一列（`image`、`style` 相符），且各有一張
        `<影像>__<風格>_si*__def.png` 或 `__def_infeasible.png`。
    feasible --dir <防禦輸出目錄> --style <風格> --images <影像>...
        印出有 `__def.png`（可送編輯）的影像，以空白分隔；不做驗收。
    edits    --dir <編輯輸出目錄> --arm <arm> --images <影像>... --data-root <資料集>
        `preflight.csv` 中該 arm 的 (影像, prompt_index) 恰為各影像 × `prompts.yaml` 的
        `edits.ip2p` 條數，且每格的輸出 PNG 存在。
    readout  --csv <讀數 CSV> --style <風格> --expect <strength>=<影像>[,<影像>...]...
        讀數的 (strength, image, prompt_index) 恰為各 strength 的影像 × 指令條數。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml


def read(path: Path) -> list:
    if not path.is_file():
        raise SystemExit(f"缺少 {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def fail(problems: list) -> None:
    for problem in problems:
        print(problem, file=sys.stderr)
    raise SystemExit(1 if problems else 0)


def defense_pngs(directory: Path, style: str, name: str, suffix: str) -> list:
    return sorted(directory.glob(f"{name}__{style}_si*__{suffix}.png"))


def check_defense(args) -> None:
    rows = read(args.dir / "results.csv")
    problems = []
    for name in args.images:
        count = sum(r["image"] == name and r.get("style") == args.style for r in rows)
        if count != 1:
            problems.append(f"results.csv 中 {name}／{args.style} 有 {count} 列，應為 1")
        if not (defense_pngs(args.dir, args.style, name, "def")
                or defense_pngs(args.dir, args.style, name, "def_infeasible")):
            problems.append(f"缺少 {name} 的防禦圖")
    fail(problems)


def feasible(args) -> None:
    print(" ".join(name for name in args.images if defense_pngs(args.dir, args.style, name, "def")))


def prompt_count(data_root: Path) -> int:
    prompts = yaml.safe_load((data_root / "prompts.yaml").read_text(encoding="utf-8"))
    return len(prompts["edits"]["ip2p"])


def check_edits(args) -> None:
    rows = [r for r in read(args.dir / "preflight.csv") if r["arm"] == args.arm]
    n = prompt_count(args.data_root)
    expected = {(name, str(k)) for name in args.images for k in range(n)}
    found = [(r["image"], r["prompt_index"]) for r in rows]
    problems = []
    if sorted(found) != sorted(expected):
        problems.append(f"{args.arm} 的格子 {sorted(set(found) ^ expected)} 與預期不符（或重複）")
    for name, k in sorted(expected & set(found)):
        if not (args.dir / args.arm / f"{name}__p{k}.png").is_file():
            problems.append(f"缺少 {args.arm}/{name}__p{k}.png")
    fail(problems)


def check_readout(args) -> None:
    rows = [r for r in read(args.csv) if r["style"] == args.style]
    n = prompt_count(args.data_root)
    expected = set()
    for item in args.expect:
        strength, _, images = item.partition("=")
        expected |= {(strength, name, str(k)) for name in images.split(",") if name for k in range(n)}
    found = [(r["strength"], r["image"], r["prompt_index"]) for r in rows]
    problems = []
    if sorted(found) != sorted(expected):
        problems.append(f"讀數 {args.style} 的鍵與預期不符（或重複）：{sorted(set(found) ^ expected)[:10]}")
    fail(problems)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("defense", "feasible"):
        p = sub.add_parser(stage)
        p.add_argument("--dir", type=Path, required=True)
        p.add_argument("--style", required=True)
        p.add_argument("--images", nargs="+", required=True)
    p = sub.add_parser("edits")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p = sub.add_parser("readout")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--expect", nargs="+", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    {"defense": check_defense, "feasible": feasible, "edits": check_edits,
     "readout": check_readout}[args.stage](args)


if __name__ == "__main__":
    main()
