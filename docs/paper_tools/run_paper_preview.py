"""Compile docs/paper/ with Tectonic, once or on every change, and copy the PDF to the desktop.

    python docs/paper_tools/run_paper_preview.py            # compile once
    python docs/paper_tools/run_paper_preview.py --watch    # recompile when a source file changes
    python docs/paper_tools/run_paper_preview.py --watch --open

The PDF is written to docs/paper/build/main.pdf and copied to the desktop as
paper.pdf after every successful compilation. A failed compilation prints the
TeX errors and keeps the previous PDF in both places.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PAPER = Path(__file__).resolve().parents[1] / "paper"
DESKTOP_PDF = Path.home() / "OneDrive" / "桌面" / "paper.pdf"
SOURCE_SUFFIXES = {".tex", ".bib", ".sty", ".bst", ".cls", ".png", ".jpg", ".jpeg", ".pdf", ".eps"}
BUILD_DIR = "build"


def source_state(paper: Path) -> dict:
    state = {}
    for path in paper.rglob("*"):
        if BUILD_DIR in path.relative_to(paper).parts or not path.is_file():
            continue
        if path.suffix.lower() in SOURCE_SUFFIXES:
            state[path] = path.stat().st_mtime_ns
    return state


def compile_paper(tectonic: str, paper: Path) -> bool:
    (paper / BUILD_DIR).mkdir(exist_ok=True)
    started = time.time()
    result = subprocess.run(
        [tectonic, "-X", "compile", "main.tex", "--outdir", BUILD_DIR, "--keep-logs"],
        cwd=paper, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    lines = (result.stdout + result.stderr).splitlines()
    errors = [line for line in lines if line.startswith("error")]
    undefined = sorted({line for line in lines if "undefined" in line.lower()})
    stamp = time.strftime("%H:%M:%S")
    if result.returncode != 0:
        print(f"[{stamp}] compile failed ({time.time() - started:.0f} s)")
        for line in errors or lines[-20:]:
            print("  " + line)
        return False
    shutil.copy2(paper / BUILD_DIR / "main.pdf", DESKTOP_PDF)
    print(f"[{stamp}] {DESKTOP_PDF} ({time.time() - started:.0f} s)")
    for line in undefined:
        print("  " + line)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER, help="directory containing main.tex")
    parser.add_argument("--watch", action="store_true", help="recompile whenever a source file changes")
    parser.add_argument("--open", action="store_true", help="open the PDF after the first compilation")
    parser.add_argument("--interval", type=float, default=1.0, help="polling interval in seconds")
    args = parser.parse_args()

    tectonic = shutil.which("tectonic")
    if tectonic is None:
        sys.exit("tectonic not found on PATH; see docs/paper_tools/README.md")
    paper = args.paper.resolve()
    if not (paper / "main.tex").is_file():
        sys.exit(f"main.tex not found in {paper}")

    ok = compile_paper(tectonic, paper)
    if args.open and ok:
        os.startfile(DESKTOP_PDF)
    if not args.watch:
        return 0 if ok else 1

    print(f"watching {paper} (Ctrl+C to stop)")
    state = source_state(paper)
    try:
        while True:
            time.sleep(args.interval)
            current = source_state(paper)
            if current != state:
                # Editors often write a file in several steps; wait until the tree is stable.
                time.sleep(0.5)
                state = source_state(paper)
                compile_paper(tectonic, paper)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
