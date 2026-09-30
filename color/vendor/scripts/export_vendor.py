"""將 core 的已提交版本匯出為專案的 vendor 快照並寫入 lock。

快照來源為 git HEAD 中的 `core/src/immunization_core`（套件）與 `core/scripts`
（GPU 租約工具）；兩者有未提交變更時拒絕，使 lock 記錄的 commit 與檔案內容一致。
目的目錄 `<專案>/vendor/immunization_core` 與 `<專案>/vendor/scripts` 整體取代，
lock 寫入 `<專案>/vendor.lock.json`，含 commit、樹雜湊與逐檔 SHA-256。

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

SOURCES = {"immunization_core": "core/src/immunization_core", "scripts": "core/scripts"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def digests(root: Path) -> dict:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


def export(repo: Path, project: Path) -> dict:
    dirty = git(repo, "status", "--porcelain", "--", *SOURCES.values()).strip()
    if dirty:
        raise SystemExit(f"core 有未提交的變更；先提交再匯出：\n{dirty}")
    commit = git(repo, "rev-parse", "HEAD").strip()
    lock = {"commit": commit, "components": {}}
    for component, source in SOURCES.items():
        target = project / "vendor" / component
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        files = git(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", source).split()
        for name in files:
            destination = target / Path(name).relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(subprocess.run(["git", "show", f"HEAD:{name}"], cwd=repo,
                                                   check=True, capture_output=True).stdout)
        lock["components"][component] = {
            "source": source, "tree": git(repo, "rev-parse", f"HEAD:{source}").strip(),
            "files": digests(target)}
    (project / "vendor.lock.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return lock


def check(project: Path) -> None:
    lock = json.loads((project / "vendor.lock.json").read_text(encoding="utf-8"))
    for component, entry in lock["components"].items():
        actual = digests(project / "vendor" / component)
        if actual != entry["files"]:
            changed = sorted({name for name, _ in set(actual.items()) ^ set(entry["files"].items())})
            raise SystemExit(f"vendor/{component} 與 lock 不一致：{changed}")


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
    count = sum(len(entry["files"]) for entry in lock["components"].values())
    print(f"[DONE] {count} 個檔案，commit {lock['commit'][:12]}")


if __name__ == "__main__":
    main()
