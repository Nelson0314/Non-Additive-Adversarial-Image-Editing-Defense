"""空組別與參照缺格須在建立 LPIPS 前拒絕。"""
import ast
import csv
from pathlib import Path

import pytest


def read_groups(*args):
    path = Path(__file__).parents[1] / "code/style_prompt_readout.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("edits", "read_groups")]
    namespace = {"Path": Path, "csv": csv}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["read_groups"](*args)


def table(root, group, names=("man_00",)):
    directory = root / group
    directory.mkdir()
    with (directory / "preflight.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=["scenario", "image", "prompt_index", "arm", "prompt", "input_png"])
        w.writeheader()
        for name in names:
            w.writerow(dict(scenario="ip2p", image=name, prompt_index=0,
                            arm="ip2p_test", prompt="edit", input_png="input.png"))


def test_missing_reference_names_directory(tmp_path):
    with pytest.raises(SystemExit, match="ref_oil"):
        read_groups(tmp_path, tmp_path, ["oil"], ["capped"])


def test_missing_requested_group_is_not_silently_skipped(tmp_path):
    table(tmp_path, "ref_oil")
    with pytest.raises(SystemExit, match="capped_oil"):
        read_groups(tmp_path, tmp_path, ["oil"], ["capped"])


def test_reference_key_mismatch_is_named(tmp_path):
    table(tmp_path, "ref_oil")
    table(tmp_path, "capped_oil", ("woman_00",))
    with pytest.raises(SystemExit, match="woman_00"):
        read_groups(tmp_path, tmp_path, ["oil"], ["capped"])


def test_valid_subset_remains_supported(tmp_path):
    table(tmp_path, "ref_oil", ("man_00", "woman_00"))
    table(tmp_path, "capped_oil")
    assert len(read_groups(tmp_path, tmp_path, ["oil"], ["capped"])) == 2


def test_filter_with_no_matches_reports_requested_images(tmp_path):
    table(tmp_path, "ref_oil")
    with pytest.raises(SystemExit, match="absent"):
        read_groups(tmp_path, tmp_path, ["oil"], ["capped"], {"absent"})
