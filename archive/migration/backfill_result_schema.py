"""第 8 項：把既有結果 CSV 補成程式的新 schema，不保留兩種 schema。

子命令
    additional-metrics
        `baseline/results/additional_metrics/` 五份表補 `device`（原程式固定 CPU），
        aesthetic 與 vmaf 另補 `unavailable_metrics`（既有列的指標皆有值，故為空字串）。
        不需要影像，已於本機執行。
    import-hashes --source-settings <檔案>
        `baseline/results/defense_color.csv` 補 `source_sha256`、`original_sha256`、
        `defended_sha256`、`source_settings`、`source_settings_sha256`，並在
        `baseline/artifacts/defenses/color/` 寫出 `import_manifest.json`。需要遠端影像，
        於第 10 項切換後在 repo 根執行；任一檔案缺少即中止，不寫任何檔。

兩者都驗證：列數與鍵集合不變、原有欄位逐值不變、新欄位符合定義。

用法（repo 根）
    python archive/migration/backfill_result_schema.py additional-metrics
    python archive/migration/backfill_result_schema.py import-hashes \\
        --source-settings color/artifacts/defenses/color/results.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "baseline"
METRICS = BASELINE / "results" / "additional_metrics"


def read(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames), list(reader)


def render(fields, rows) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def insert_after(fields, anchor, new):
    index = fields.index(anchor) + 1
    return fields[:index] + new + fields[index:]


def verify(old_fields, old_rows, new_fields, new_rows, added, keys):
    assert len(old_rows) == len(new_rows)
    assert [f for f in new_fields if f not in added] == old_fields
    assert sorted(tuple(r[k] for k in keys) for r in old_rows) == \
        sorted(tuple(r[k] for k in keys) for r in new_rows)
    for old, new in zip(old_rows, new_rows):
        assert all(old[f] == new[f] for f in old_fields)


def additional_metrics() -> None:
    plans = {
        "fidelity.csv": (None, {"device": "cpu"}, ("condition", "image")),
        "displacement.csv": (None, {"device": "cpu"}, ("condition", "scenario", "image", "prompt_index")),
        "retention.csv": (None, {"device": "cpu"},
                          ("condition", "purifier", "scenario", "image", "prompt_index")),
        "aesthetic.csv": (None, {"device": "cpu", "unavailable_metrics": ""}, ("condition", "image")),
        "vmaf.csv": ("vmaf", {"device": "cpu", "unavailable_metrics": ""},
                     ("pairing", "condition", "image", "scenario", "prompt_index", "purifier")),
    }
    outputs = {}
    for name, (anchor, added, keys) in plans.items():
        path = METRICS / name
        fields, rows = read(path)
        if set(added) <= set(fields):
            raise SystemExit(f"{path} 已是新 schema")
        metric_columns = [f for f in fields if f.startswith("aes_") or f == "vmaf"]
        empty = [(i, c) for i, r in enumerate(rows) for c in metric_columns if r[c] == ""]
        if empty:
            raise SystemExit(f"{path} 有空的選配指標，不能補為全部可用：{empty[:5]}")
        new_fields = insert_after(fields, anchor, list(added)) if anchor else fields + list(added)
        new_rows = [{**r, **added} for r in rows]
        verify(fields, rows, new_fields, new_rows, set(added), keys)
        outputs[path] = render(new_fields, new_rows)
    for path, data in outputs.items():
        path.write_bytes(data)
        print(f"[DONE] {path.relative_to(REPO)}")


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"缺少檔案：{path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_hashes(source_settings: Path) -> None:
    path = BASELINE / "results" / "defense_color.csv"
    fields, rows = read(path)
    added = ["source_sha256", "original_sha256", "defended_sha256", "source_settings",
             "source_settings_sha256"]
    if set(added) <= set(fields):
        raise SystemExit(f"{path} 已是新 schema")
    settings = {"path": source_settings.as_posix(), "sha256": sha256(source_settings)}
    images, new_rows = [], []
    for row in rows:
        source = BASELINE / row["source_png"]
        original = BASELINE / row["data_root"] / row["class"] / f"{row['image']}.png"
        defended = BASELINE / "artifacts" / "defenses" / row["condition"] / \
            f"{row['image']}__{row['condition']}__def.png"
        record = {"image": row["image"], "source_png": row["source_png"], "source_sha256": sha256(source),
                  "original_png": original.relative_to(BASELINE).as_posix(), "original_sha256": sha256(original),
                  "defended_png": defended.relative_to(BASELINE).as_posix(), "defended_sha256": sha256(defended)}
        images.append(record)
        new_rows.append({**row, "source_sha256": record["source_sha256"],
                         "original_sha256": record["original_sha256"],
                         "defended_sha256": record["defended_sha256"],
                         "source_settings": settings["path"], "source_settings_sha256": settings["sha256"]})
    new_fields = fields + added
    verify(fields, rows, new_fields, new_rows, set(added), ("image", "condition"))
    conditions = {r["condition"] for r in rows}
    if len(conditions) != 1:
        raise SystemExit(f"{path} 應只有一個條件：{conditions}")
    condition = conditions.pop()
    manifest = {"condition": condition, "variant": condition, "norm": rows[0]["norm"],
                "budget": rows[0]["eps"], "source_dir": Path(rows[0]["source_png"]).parent.as_posix(),
                "source_settings": settings, "images": images}
    path.write_bytes(render(new_fields, new_rows))
    target = BASELINE / "artifacts" / "defenses" / condition / "import_manifest.json"
    target.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"[DONE] {path.relative_to(REPO)}、{target.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("additional-metrics")
    hashes = sub.add_parser("import-hashes")
    hashes.add_argument("--source-settings", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "additional-metrics":
        additional_metrics()
    else:
        import_hashes(args.source_settings)


if __name__ == "__main__":
    main()
