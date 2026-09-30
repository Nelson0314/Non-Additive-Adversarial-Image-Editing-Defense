"""文字檔讀寫須明確指定編碼。

未指定 `encoding` 時 Python 取平台預設（Windows 繁體中文環境為 cp950），與以 UTF-8 寫入的檔案
或 bash 以 printf 追加的內容混用即無法解碼。掃描 core、baseline、color、style 的 `src/`、`tests/`、
`scripts/` 下的 Python 檔（不含 `vendor/`），`read_text`、`write_text` 與文字模式的 `open` 呼叫
都必須帶 `encoding=`。
"""
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROJECTS = ("core", "baseline", "color", "style")
BINARY_OPENERS = {"Image", "tarfile", "gzip", "zipfile"}


def text_mode(call: ast.Call) -> bool:
    positional = call.args[1:2] if isinstance(call.func, ast.Name) else call.args[:1]
    for node in [*positional, *(k.value for k in call.keywords if k.arg == "mode")]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return "b" not in node.value
    return True


def unencoded_calls(source: str) -> list:
    out = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in ("read_text", "write_text", "open"):
            continue
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in BINARY_OPENERS:
            continue
        if name == "open" and not text_mode(node):
            continue
        if not any(k.arg == "encoding" for k in node.keywords):
            out.append((node.lineno, name))
    return out


def python_files() -> list:
    return sorted(path for project in PROJECTS for part in ("src", "tests", "scripts")
                  for path in (REPO / project / part).rglob("*.py")
                  if "vendor" not in path.parts and "__pycache__" not in path.parts)


def test_checker_flags_calls_without_encoding():
    source = ('p.write_text("x")\np.read_text()\nopen(f)\nopen(f, "rb")\n'
              'Image.open(f)\np.write_text("x", encoding="utf-8")\np.open("a", encoding="utf-8")\n')
    assert unencoded_calls(source) == [(1, "write_text"), (2, "read_text"), (3, "open")]


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_text_io_specifies_encoding(path):
    assert unencoded_calls(path.read_text(encoding="utf-8")) == []
