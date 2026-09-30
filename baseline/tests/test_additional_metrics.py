"""選配指標的後端判定：只有建立後端的錯誤記為缺席原因。"""
import pytest

from immunization_baseline.cli import measure_additional_metrics as metrics


def test_optional_backend_failures_are_recorded_with_reasons():
    def missing():
        raise ImportError("no module named pyiqa")

    available, absent = metrics.optional_backends({"aes_a": lambda: "model", "aes_b": missing})
    assert available == {"aes_a": "model"}
    assert absent == {"aes_b": "ImportError: no module named pyiqa"}
    assert metrics.unavailable_text(absent) == "aes_b: ImportError: no module named pyiqa"
    assert metrics.unavailable_text({}) == ""


def test_unexpected_backend_errors_are_not_swallowed():
    def broken():
        raise KeyError("bug")

    with pytest.raises(KeyError):
        metrics.optional_backends({"aes_a": broken})


def test_missing_ffmpeg_marks_vmaf_unavailable(monkeypatch):
    monkeypatch.setenv("PATH", "")
    _, absent = metrics.optional_backends({"vmaf": metrics.libvmaf_available})
    assert "vmaf" in absent


def test_existing_tables_carry_the_current_schema():
    import csv
    from immunization_baseline import layout
    for name in ("fidelity", "displacement", "retention", "aesthetic", "vmaf"):
        with (layout.RESULTS / "additional_metrics" / f"{name}.csv").open(encoding="utf-8") as stream:
            header = next(csv.reader(stream))
        assert "device" in header
        if name in ("aesthetic", "vmaf"):
            assert "unavailable_metrics" in header
