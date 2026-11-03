"""在同一次 FLUX.1-Kontext 載入下依序執行多次 `run_flux_edits`。

`--jobs-file` 每個非空、非 `#` 開頭的行是一次 `run_flux_edits` 的參數（以 shlex 切分）。各行的協定、輸出、
續跑與失敗規則與單獨執行 `run_flux_edits` 相同；差別只在模型只在第一個有待做格的行載入一次，之後的行重用同一個
pipeline。任一行失敗即停止並以非零結束碼結束，之後的行不執行。

用法（遠端 baseline 專案根，CUDA_VISIBLE_DEVICES 由租約設定）：
    python -m immunization_baseline.cli.run_flux_edit_jobs --jobs-file <檔>
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from immunization_baseline.cli import run_flux_edits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-file", type=Path, required=True)
    args = ap.parse_args()
    jobs = [shlex.split(line) for line in args.jobs_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not jobs:
        raise SystemExit(f"{args.jobs_file} 沒有工作")

    load = run_flux_edits.load_pipeline
    cached = []

    def load_once():
        if not cached:
            t = time.time()
            cached.append(load())
            print(f"[LOAD ONCE] {time.time() - t:.1f}s", flush=True)
        return cached[0]

    run_flux_edits.load_pipeline = load_once
    for k, job in enumerate(jobs, 1):
        print(f"[JOB {k}/{len(jobs)}] run_flux_edits {shlex.join(job)}", flush=True)
        sys.argv = ["run_flux_edits", *job]
        run_flux_edits.main()
    print(f"[ALLDONE] {len(jobs)} 項（{args.jobs_file}）", flush=True)


if __name__ == "__main__":
    main()
