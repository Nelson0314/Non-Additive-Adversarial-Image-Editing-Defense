"""第 9 項：各專案單獨複製到 repo 外的目錄後，仍可 import、`--help` 與 pytest；並比對雜湊。

步驟（每個專案各自進行，互不共用目錄）
1. `git archive <rev> <專案>` 解到空目錄：只含版控檔，不含其他專案、`artifacts/`、`runtime/`。
2. 逐檔比對：解出檔案的 SHA-256 等於 `git cat-file` 取得的 blob（換行已正規化為 LF）。
3. 以 `PYTHONPATH=src[:vendor]`、工作目錄為該副本、清空其他 PYTHONPATH 執行：
   - import `src/`（及 `vendor/`）下的全部模組，且 `immunization_core` 必須來自副本；
   - 每個含 `argparse` 與 `__main__` 的模組以 `--help` 執行，結束碼 0，且未初始化 CUDA；
     執行時設定 `HF_HUB_OFFLINE=1`、`CUDA_VISIBLE_DEVICES=` 以排除聯網與取卡；
   - 每個 `.sh` 通過 `bash -n`；
   - `python -m pytest -q`。
4. 另以 `core.autocrlf=true`（Windows 簽出設定）clone 一份，工作目錄每個版控檔的 SHA-256
   等於其 blob，確認 `.gitattributes` 使各平台簽出位元組一致。

任一步失敗即記錄並以結束碼 1 結束。結果寫到 `--report`（JSON）。

用法（repo 根）
    python archive/migration/verify_standalone.py --work-dir <空的暫存目錄> --report <報告.json>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJECTS = {"core": ["src"], "baseline": ["src", "vendor"], "color": ["src", "vendor"],
            "style": ["src", "vendor"]}

HELP = """
import runpy, sys
import torch
name = sys.argv[1]
sys.argv = [name, "--help"]
try:
    runpy.run_module(name, run_name="__main__", alter_sys=True)
    code = 0
except SystemExit as stop:
    code = stop.code or 0
if torch.cuda.is_initialized():
    raise SystemExit(f"{name}: --help 初始化了 CUDA")
raise SystemExit(code)
"""

IMPORT = """
import importlib, pkgutil, sys
from pathlib import Path
root = Path.cwd().resolve()
names = []
for base in sys.argv[1:]:
    for info in pkgutil.walk_packages([base]):
        names.append(info.name)
        importlib.import_module(info.name)
import immunization_core
assert Path(immunization_core.__file__).resolve().is_relative_to(root), immunization_core.__file__
print(len(names))
"""


def git(*args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, **kwargs)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blobs(rev: str, prefix: str) -> dict:
    listing = git("ls-tree", "-r", "-z", rev, "--", prefix).stdout.split(b"\0")
    out = {}
    for entry in filter(None, listing):
        meta, path = entry.split(b"\t", 1)
        if meta.split()[1] == b"blob":
            out[path.decode()] = meta.split()[2].decode()
    ids = "\n".join(out.values()).encode() + b"\n"
    contents = git("cat-file", "--batch", input=ids).stdout
    hashes, stream = {}, io.BytesIO(contents)
    for path, object_id in out.items():
        head = stream.readline().split()
        assert head[0].decode() == object_id, path
        hashes[path] = sha(stream.read(int(head[2])))
        stream.read(1)
    return hashes


def run(command, cwd, env, timeout=600) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)


def cli_modules(copy: Path, roots: list) -> list:
    names = []
    for path in sorted((copy / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "__main__" in text and "argparse" in text or path.parent.name == "cli" and "__main__" in text:
            names.append(".".join(path.relative_to(copy / "src").with_suffix("").parts))
    return names


def verify_project(name: str, roots: list, rev: str, work: Path) -> dict:
    copy = work / name
    data = git("archive", "--format=tar", rev, name).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(work, filter="data")
    report = {"failures": []}
    expected = blobs(rev, name)
    files = {p.relative_to(work).as_posix(): sha(p.read_bytes())
             for p in copy.rglob("*") if p.is_file()}
    report["files"] = len(files)
    if files != expected:
        report["failures"].append({"hash_mismatch": sorted(set(files.items()) ^ set(expected.items()))[:20]})
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env.update(PYTHONPATH=os.pathsep.join(roots), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    result = run([sys.executable, "-c", IMPORT, *roots], copy, env)
    if result.returncode:
        report["failures"].append({"import": result.stderr[-2000:]})
    else:
        report["modules_imported"] = int(result.stdout.split()[-1])
    helps = cli_modules(copy, roots)
    report["help_ok"] = []
    for module in helps:
        result = run([sys.executable, "-c", HELP, module], copy, env, timeout=120)
        if result.returncode or "usage" not in result.stdout.lower():
            report["failures"].append({"help": module, "stderr": result.stderr[-1500:]})
        else:
            report["help_ok"].append(module)
    shells = sorted(copy.rglob("*.sh"))
    for path in shells:
        result = run(["bash", "-n", path.as_posix()], copy, env)
        if result.returncode:
            report["failures"].append({"bash_n": path.relative_to(copy).as_posix(), "stderr": result.stderr})
    report["shell_scripts"] = len(shells)
    result = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], copy, env, timeout=1800)
    report["pytest"] = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode:
        report["failures"].append({"pytest": result.stdout[-3000:] + result.stderr[-1000:]})
    return report


def verify_windows_checkout(rev: str, work: Path) -> dict:
    clone = work / "windows_clone"
    subprocess.run(["git", "clone", "-q", "-c", "core.autocrlf=true", "--no-checkout", str(REPO), str(clone)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "-c", "core.autocrlf=true", "checkout", "-q", rev],
                   check=True, capture_output=True)
    expected = blobs(rev, ".")
    mismatched = [path for path, digest in expected.items()
                  if sha((clone / path).read_bytes()) != digest]
    return {"files": len(expected), "mismatched": mismatched[:50], "mismatched_count": len(mismatched)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--projects", nargs="+", default=list(PROJECTS))
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if work.is_relative_to(REPO):
        raise SystemExit("--work-dir 必須在 repo 之外")
    work.mkdir(parents=True, exist_ok=False)
    rev = git("rev-parse", args.rev, text=True).stdout.strip()
    report = {"rev": rev, "python": sys.version.split()[0], "projects": {}}
    for name in args.projects:
        report["projects"][name] = verify_project(name, PROJECTS[name], rev, work / "standalone")
        status = "FAIL" if report["projects"][name]["failures"] else "ok"
        print(f"[{status}] {name}: {report['projects'][name].get('pytest', '')}", flush=True)
    report["windows_checkout"] = verify_windows_checkout(rev, work)
    print(f"[{'FAIL' if report['windows_checkout']['mismatched_count'] else 'ok'}] windows checkout: "
          f"{report['windows_checkout']['files']} 檔", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                           newline="\n")
    failed = any(p["failures"] for p in report["projects"].values()) or \
        report["windows_checkout"]["mismatched_count"]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
