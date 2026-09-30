"""`io` 的 CSV 寫入：失敗時保留原檔，成功時原子取代且不留暫存檔。"""
import csv

import pytest

from immunization_core import io


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("writer", [io.write_csv, io.write_sorted_csv])
def test_failed_write_keeps_existing_rows(tmp_path, writer):
    path = tmp_path / "table.csv"
    writer(path, [{"a": 1, "b": 2}])
    before = path.read_bytes()

    class Unwritable:
        def __str__(self):
            raise RuntimeError("中途失敗")

    with pytest.raises(RuntimeError, match="中途失敗"):
        writer(path, [{"a": 3, "b": 4}, {"a": Unwritable(), "b": 6}])
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["table.csv"]


@pytest.mark.parametrize("writer", [io.write_csv, io.write_sorted_csv])
def test_successful_write_replaces_file(tmp_path, writer):
    path = tmp_path / "table.csv"
    writer(path, [{"a": 1}])
    writer(path, [{"a": 2}, {"a": 3, "b": 4}])
    assert [r["a"] for r in read(path)] == ["2", "3"]
    assert read(path)[1]["b"] == "4"
    assert [p.name for p in tmp_path.iterdir()] == ["table.csv"]


def test_interrupted_row_iteration_keeps_existing_rows(tmp_path):
    path = tmp_path / "table.csv"
    io.write_rows_atomic(path, ["a"], [{"a": 1}])

    def rows():
        yield {"a": 2}
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        io.write_rows_atomic(path, ["a"], rows())
    assert read(path) == [{"a": "1"}]
