"""主表匯入條件的出處欄更正：`spec_source`、`solver_prompt`、`solver_prompt_source` 改為
`baseline/configs/conditions.yaml` 該條目的值。

原值由舊版 `import_defense_artifacts` 依封存腳本（`paper_baseline.py` 的 color 臂、「三個項都不經過
text encoder」）寫入，與現行 color 方法不符。只改這三欄；逐表驗證列數、欄序與其他欄位逐值不變，
並印出每欄的原值與新值。

用法（repo 根）
    python archive/migration/correct_import_provenance.py
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "baseline"
FIELDS = ("spec_source", "solver_prompt", "solver_prompt_source")


def main() -> None:
    entries = yaml.safe_load((BASELINE / "configs/conditions.yaml").read_text(encoding="utf-8"))["conditions"]
    for name, entry in entries.items():
        if entry["solver"] != "imported" or not entry["main_table"]:
            continue
        path = BASELINE / "results" / f"defense_{name}.csv"
        data = path.read_bytes()
        reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
        header, rows = reader.fieldnames, list(reader)
        new_rows = [{**row, **{f: entry[f] for f in FIELDS}} for row in rows]
        assert len(new_rows) == len(rows)
        for old, new in zip(rows, new_rows):
            assert all(old[k] == new[k] for k in header if k not in FIELDS)
        for field in FIELDS:
            print(f"{path.relative_to(REPO)} {field}: {sorted({r[field] for r in rows})} → {entry[field]!r}")
        out = io.StringIO(newline="")
        terminator = "\r\n" if b"\r\n" in data.split(b"\n", 1)[0] + b"\n" else "\n"
        writer = csv.DictWriter(out, fieldnames=header, lineterminator=terminator)
        writer.writeheader()
        writer.writerows(new_rows)
        path.write_bytes(out.getvalue().encode("utf-8"))
        print(f"[DONE] {path.relative_to(REPO)}（{len(new_rows)} 列）")


if __name__ == "__main__":
    main()
