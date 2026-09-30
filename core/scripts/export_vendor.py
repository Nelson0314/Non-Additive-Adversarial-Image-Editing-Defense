"""將 immunization_core 的已提交版本匯出為專案的 vendor 快照並寫入 lock。

快照來源為 git HEAD 中的 `core/src/immunization_core`；core 有未提交變更時拒絕，
使 lock 記錄的 commit 與檔案內容一致。目的目錄 `<專案>/vendor/immunization_core`
整體取代，lock 寫入 `<專案>/vendor.lock.json`，含 commit、樹雜湊與逐檔 SHA-256。

用法
    python core/scripts/export_vendor.py <專案目錄>
    python core/scripts/export_vendor.py <專案目錄> --check   # 只驗證快照與 lock 一致
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PACKAGE = "core/src/immunization_core"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def digests(root: Path) -> dict:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


def export(repo: Path, project: Path) -> dict:
    if git(repo, "status", "--porcelain", "--", PACKAGE).strip():
        raise SystemExit(f"{PACKAGE} 有未提交的變更；先提交再匯出")
    commit = git(repo, "rev-parse", "HEAD").strip()
    tree = git(repo, "rev-parse", f"HEAD:{PACKAGE}").strip()
    target = project / "vendor" / "immunization_core"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    files = git(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", PACKAGE).split()
    for name in files:
        destination = target / Path(name).relative_to(PACKAGE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(subprocess.run(["git", "show", f"HEAD:{name}"], cwd=repo,
                                               check=True, capture_output=True).stdout)
    lock = {"package": "immunization_core", "source": PACKAGE, "commit": commit,
            "tree": tree, "files": digests(target)}
    (project / "vendor.lock.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return lock


def check(project: Path) -> None:
    lock = json.loads((project / "vendor.lock.json").read_text(encoding="utf-8"))
    actual = digests(project / "vendor" / "immunization_core")
    if actual != lock["files"]:
        changed = sorted(set(actual.items()) ^ set(lock["files"].items()))
        raise SystemExit(f"vendor 快照與 lock 不一致：{[name for name, _ in changed]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check(args.project)
        print(f"[OK] {args.project / 'vendor.lock.json'}")
        return
    repo = Path(git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel").strip())
    lock = export(repo, args.project.resolve())
    print(f"[DONE] {len(lock['files'])} 個檔案，commit {lock['commit'][:12]}")


if __name__ == "__main__":
    main()
