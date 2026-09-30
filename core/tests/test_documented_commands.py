"""文件中 core 工具指令的選項須為該工具實際接受的選項（不保留舊參數別名，文件寫錯即執行失敗）。

掃描根目錄與各專案的 README、STATUS、CLAUDE.md 與 `docs/`（不含 `docs/reference/` 與 `archive/`）：
在提到 `core/scripts/` 工具名稱的那一行，取名稱之後到反引號或表格分隔 `|` 為止的片段，其中的 `--選項`
須出現在該工具的 `case` 標籤（shell）或 `add_argument`（Python）中。
"""
from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "core/scripts"
DOCUMENTS = [p for p in [REPO / "README.md", REPO / "CLAUDE.md",
                         *[d for project in ("core", "baseline", "color", "style")
                           for d in (REPO / project).glob("*.md")],
                         *[d for project in ("core", "baseline", "color", "style")
                           for d in (REPO / project / "docs").glob("*.md")]]
             if p.is_file()]


def accepted(tool: Path) -> set:
    text = tool.read_text(encoding="utf-8")
    if tool.suffix == ".sh":
        return set(re.findall(r"(--[a-z][a-z-]*)\)", text)) | set(re.findall(r'"(--[a-z][a-z-]*)"', text))
    return set(re.findall(r'add_argument\(\s*"(--[a-z][a-z-]*)"', text))


TOOLS = {p.name: accepted(p) for p in SCRIPTS.iterdir() if p.suffix in (".sh", ".py")}


def documented_flags(line: str) -> list:
    out = []
    for name in TOOLS:
        for match in re.finditer(r"(?<![\w-])" + re.escape(name), line):
            rest = re.split(r"[`|]", line[match.end():], maxsplit=1)[0]
            out += [(name, flag) for flag in re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", rest)]
    return out


def test_checker_finds_flags_after_tool_names():
    line = "`run_with_gpu_lease.sh --workdir <目錄> <名稱>` 與 `measure_free_gpus.sh --assert 0`"
    assert sorted(documented_flags(line)) == [("measure_free_gpus.sh", "--assert"), ("run_with_gpu_lease.sh", "--workdir")]
    assert "--work-dir" in TOOLS["run_with_gpu_lease.sh"] and "--workdir" not in TOOLS["run_with_gpu_lease.sh"]


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_documented_options_exist(document):
    wrong = [(n, flag, line.strip()[:80])
             for n, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1)
             for name, flag in documented_flags(line) if flag not in TOOLS[name]]
    assert not wrong, wrong
