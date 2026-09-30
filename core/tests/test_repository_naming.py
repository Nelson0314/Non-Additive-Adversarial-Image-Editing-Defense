"""命名規範（根目錄 `CLAUDE.md`「命名」）：掃描 core、baseline、color、style 的版控檔名，排除 `vendor/`。

檢查大小寫（小寫 snake_case；固定名稱文件與 `docs/` 主題文件例外）、美式拼法、流程用語、測試檔名，
以及 `cli/` 與 `scripts/` 入口的動詞。repo 有 git 時取 `git ls-files`；單獨複製的專案目錄則走檔案系統，
略過不入版控的 `artifacts/`、`runtime/`、`trials/` 與快取。
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]
PROJECTS = ("core", "baseline", "color", "style")
FIXED_DOCUMENTS = {"README.md", "STATUS.md", "DESIGN.md", "TRIALS.md", "CLAUDE.md"}
VERBS = ("generate_", "run_", "apply_", "measure_", "import_", "sweep_", "evaluate_")
LIBRARIES = {"env.sh", "gpu_lease.sh", "gpu_policy.sh"}
UNTRACKED = {"vendor", "artifacts", "runtime", "trials", "__pycache__", ".pytest_cache"}

TOKEN = r"[a-z0-9]+(?:\.[0-9]+)?"
NAME = re.compile(rf"{TOKEN}(?:_{TOKEN})*")
FILE = re.compile(rf"{TOKEN}(?:_{TOKEN})*(?:\.[a-z0-9]+)+")
TOPIC_DOCUMENT = re.compile(r"[A-Z0-9]+(?:_[A-Z0-9]+)*\.md")
BRITISH = re.compile(r"colour|defence|immunis|optimis|normalis|behaviour|licence|centre|^grey$")
PROCESS = re.compile(r"^(?:test|try|tmp|temp|new|old|pilot|preview|full|final|draft|copy|backup|bak|wip|"
                     r"latest|legacy|misc|v[0-9]+|round[0-9]*|r[0-9]+|(?:19|20)[0-9]{6})$")


def tracked_paths() -> list:
    if (REPO / ".git").exists() and shutil.which("git"):
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "--", *PROJECTS],
                             capture_output=True, text=True, check=True).stdout.split("\n")
        return [p for p in out if p]
    paths = []
    for project in PROJECTS:
        for path in (REPO / project).rglob("*"):
            parts = path.relative_to(REPO).parts
            if path.is_file() and not UNTRACKED & set(parts):
                paths.append("/".join(parts))
    return paths


def problems(path: str) -> list:
    """一條相對 repo 根的路徑違反的規則；空串列為符合。"""
    parts = path.split("/")
    if "vendor" in parts:
        return []
    out = []
    for depth, part in enumerate(parts[1:], start=1):
        is_file = depth == len(parts) - 1
        parent = parts[:depth]
        if part.startswith(".") or part in ("__init__.py", "__pycache__"):
            continue
        if is_file and (part in FIXED_DOCUMENTS or ("docs" in parent and TOPIC_DOCUMENT.fullmatch(part))):
            continue
        checked = part[1:] if is_file and part.endswith(".py") and re.match(r"_[a-z]", part) else part  # 私有模組
        if not (FILE if is_file else NAME).fullmatch(checked):
            out.append(f"{part}：須為小寫 snake_case")
        stem = checked.split(".")[0] if is_file else part
        tokens = stem.split("_")
        if is_file and parent[-1] == "tests" and tokens[0] == "test":
            tokens = tokens[1:]
        for token in tokens:
            if BRITISH.search(token):
                out.append(f"{part}：{token} 不是美式拼法")
            if PROCESS.match(token):
                out.append(f"{part}：{token} 是流程或順序用語")
        if re.search(r"(?:^|_)queue_[a-z](?:_|$)", stem):
            out.append(f"{part}：佇列以單一字母區分")
    name = parts[-1]
    if parts[-2] == "tests" and name.endswith(".py") and name not in ("__init__.py", "conftest.py") \
            and not name.startswith("test_") and not name.endswith("_stub.py"):
        out.append(f"{name}：測試檔須為 test_<受測模組或行為>.py")
    entry = (parts[-2] == "cli" and name.endswith(".py") and not name.startswith("_")) or \
            (parts[-2] == "scripts" and name.endswith((".sh", ".py")) and name not in LIBRARIES)
    if entry and not name.startswith(VERBS):
        out.append(f"{name}：程式入口須以 {'、'.join(VERBS)} 開頭")
    return out


@pytest.mark.parametrize("path", [
    "baseline/data/targets/MIST.png", "core/tests/test_purify_new_ops.py", "color/scripts/queue_job.sh",
    "style/results/defenses/r11/results.csv", "core/src/immunization_core/colour.py",
    "baseline/results/ADDITIVE_TRANSFER.md", "color/data/color_lpips-reference.csv",
    "core/scripts/free_cards.sh", "style/src/immunization_style/cli/check_job_outputs.py",
    "baseline/results/sweeps/flux/guidance_v2.csv", "core/tests/helpers.py", "color/runtime_old/x.csv",
])
def test_checker_rejects_known_violations(path):
    assert problems(path)


@pytest.mark.parametrize("path", [
    "baseline/docs/ADDITIVE_TRANSFER.md", "baseline/results/aligned/README.md", "core/tests/carrier_stub.py",
    "color/results/purified_edits/color/crop_resize0.1/preflight.csv", "baseline/vendor.lock.json",
    "baseline/results/sweeps/flux/guidance/guidance_3p5.csv", "core/scripts/gpu_lease.sh",
    "baseline/src/immunization_baseline/cli/_defaults.py", "style/docs/references/neucom_134591.json",
    "baseline/vendor/scripts/free_cards.sh", "core/tests/test_purify_crop_resize_chain.py",
])
def test_checker_accepts_conforming_names(path):
    assert problems(path) == []


def test_tracked_file_names_follow_the_naming_rules():
    paths = tracked_paths()
    assert paths, "沒有取得任何檔名"
    violations = {path: found for path in paths if (found := problems(path))}
    assert not violations, "\n".join(f"{p}: {'；'.join(v)}" for p, v in sorted(violations.items()))
