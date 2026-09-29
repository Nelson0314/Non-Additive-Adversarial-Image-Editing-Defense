"""執行兩個 CLI 的 main，僅以 stub 取代模型與資料來源。"""
import argparse
import ast
import csv
import json
from pathlib import Path
import sys
import time
import types

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
import resume_state


def driver(kind, tmp_path):
    source = Path(__file__).parents[1] / "code" / f"edit_{kind}.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("main", "load_done", "output_png", "write_preflight")]
    captured = []
    namespace = dict(argparse=argparse, csv=csv, json=json, Path=Path, time=time,
                     file_digest=resume_state.file_digest,
                     load_resume_rows=resume_state.load_resume_rows,
                     protocol_digest=lambda config: captured.append(config) or resume_state.protocol_digest(config),
                     write_rows_atomic=resume_state.write_rows_atomic)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    namespace["__file__"] = str(source)
    namespace["paths"] = types.SimpleNamespace(PORTRAITS=tmp_path / "data", SOURCE_HOME=tmp_path,
                                               RESULTS=tmp_path / "results", IMAGES=tmp_path / "images")
    original = tmp_path / "original.png"
    original.write_bytes(b"original")
    namespace["load_items"] = lambda _: ([{"name": "a", "path": original}], {"ip2p": ["edit"]})
    namespace["input_png"] = lambda *args: original
    namespace["OPERATORS"] = []
    namespace["EDITORS"] = {"sd3-ultraedit": {"repo": "model", "call_kw": {"negative_prompt": ""}}}

    def no_model(*args):
        raise AssertionError("model loader called")

    namespace["load_pipeline"] = no_model
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    if kind == "flux_preview":
        argv = [str(source), "--arm", "undefended", "--instruction-indices", "0"]
        out_csv = tmp_path / "results/flux_full_undefended.csv"
        png = tmp_path / "images/flux_full/undefended/a__p0.png"
    else:
        prompts = tmp_path / "prompts.json"
        prompts.write_text('{"verbatim": ["edit"]}')
        argv = [str(source), "--arms", "undefended", "--purifiers", "none", "--guidance", "2.5",
                "--image-guidance", "1.5", "--prompt-sets", str(prompts)]
        out_csv = tmp_path / "results/ultraedit_full/undefended.csv"
        png = tmp_path / "images/ultraedit_full/edit_preflight/ultraedit_undefended/a__p0.png"
    return namespace, captured, argv, out_csv, original, png


@pytest.mark.parametrize("kind", ["flux_preview", "ultraedit_full"])
@pytest.mark.parametrize("change", ["none", "guidance", "missing_png", "legacy"])
def test_cli_resume_validates_before_model_load(tmp_path, monkeypatch, kind, change):
    ns, configs, argv, csv_path, original, png = driver(kind, tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(AssertionError, match="model loader called"):
        ns["main"]()
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(b"existing edit")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict.fromkeys(ns["FIELDS"], "")
    row.update(arm="undefended", image="a", prompt_index="0", purifier="none",
               protocol_id=resume_state.protocol_digest(configs[-1]), png=str(png),
               input_png=str(original), input_sha256=resume_state.file_digest(original))
    if "reference_png" in row:
        row.update(reference_png=str(original), reference_sha256=resume_state.file_digest(original))
    row = {k: v for k, v in row.items() if k in ns["FIELDS"]}
    resume_state.write_rows_atomic(csv_path, ns["FIELDS"], [row])
    if change == "guidance":
        monkeypatch.setattr(sys, "argv", argv + ["--guidance", "9"])
    elif change == "missing_png":
        png.unlink()
    elif change == "legacy":
        csv_path.write_text("image,prompt_index\na,0\n")
    before = csv_path.read_bytes()
    if change == "none":
        ns["main"]()
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            ns["main"]()
    assert csv_path.read_bytes() == before
