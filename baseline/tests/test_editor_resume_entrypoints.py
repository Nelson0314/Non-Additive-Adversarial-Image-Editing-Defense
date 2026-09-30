"""執行兩個 CLI 的 main，僅以 stub 取代模型與資料來源。"""
import sys
import types

import pytest

from immunization_baseline import resume_state
from immunization_baseline.cli import run_flux_edits, run_ultraedit_edits


def driver(kind, tmp_path, monkeypatch):
    module = {"flux_preview": run_flux_edits, "ultraedit_full": run_ultraedit_edits}[kind]
    captured = []
    original = tmp_path / "original.png"
    original.write_bytes(b"original")
    layout = types.SimpleNamespace(
        PROJECT=tmp_path, PORTRAITS=tmp_path / "data", RESULTS=tmp_path / "results",
        CONFIGS=tmp_path / "configs", DEFENSES=tmp_path / "defenses",
        PURIFIED_EDITS=tmp_path / "purified_edits", FLUX_EDITS=tmp_path / "flux_edits",
        ULTRAEDIT_EDITS=tmp_path / "ultraedit_edits")

    def no_model(*args):
        raise AssertionError("model loader called")

    monkeypatch.setattr(module, "layout", layout)
    monkeypatch.setattr(module, "load_items",
                        lambda _: ([{"name": "a", "path": original}], {"ip2p": ["edit"]}))
    monkeypatch.setattr(module, "load_pipeline", no_model)
    monkeypatch.setattr(module, "protocol_digest",
                        lambda config: captured.append(config) or resume_state.protocol_digest(config))
    if kind == "flux_preview":
        argv = ["run_flux_edits", "--arm", "undefended", "--instruction-indices", "0"]
        out_csv = tmp_path / "results/flux/edits_undefended.csv"
        png = tmp_path / "flux_edits/undefended/a__p0.png"
    else:
        monkeypatch.setattr(module, "input_png", lambda *args: original)
        prompts = tmp_path / "prompts.json"
        prompts.write_text('{"verbatim": ["edit"]}')
        argv = ["run_ultraedit_edits", "--arms", "undefended", "--purifiers", "none",
                "--guidance", "2.5", "--image-guidance", "1.5", "--prompt-sets", str(prompts)]
        out_csv = tmp_path / "results/ultraedit/edits/undefended.csv"
        png = tmp_path / "ultraedit_edits/undefended_edits/ultraedit_undefended/a__p0.png"
    return module, captured, argv, out_csv, original, png


@pytest.mark.parametrize("kind", ["flux_preview", "ultraedit_full"])
@pytest.mark.parametrize("change", ["none", "guidance", "missing_png", "legacy"])
def test_cli_resume_validates_before_model_load(tmp_path, monkeypatch, kind, change):
    ns, configs, argv, csv_path, original, png = driver(kind, tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(AssertionError, match="model loader called"):
        ns.main()
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(b"existing edit")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict.fromkeys(ns.FIELDS, "")
    row.update(arm="undefended", image="a", prompt_index="0", purifier="none",
               protocol_id=resume_state.protocol_digest(configs[-1]), png=str(png),
               input_png=str(original), input_sha256=resume_state.file_digest(original))
    if "reference_png" in row:
        row.update(reference_png=str(original), reference_sha256=resume_state.file_digest(original))
    row = {k: v for k, v in row.items() if k in ns.FIELDS}
    resume_state.write_rows_atomic(csv_path, ns.FIELDS, [row])
    if change == "guidance":
        monkeypatch.setattr(sys, "argv", argv + ["--guidance", "9"])
    elif change == "missing_png":
        png.unlink()
    elif change == "legacy":
        csv_path.write_text("image,prompt_index\na,0\n")
    before = csv_path.read_bytes()
    if change == "none":
        ns.main()
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            ns.main()
    assert csv_path.read_bytes() == before
