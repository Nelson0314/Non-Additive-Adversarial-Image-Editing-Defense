"""Run an auditable, serial IP2P helmet sweep on one free remote GPU.

--plan-only writes planned.csv without loading models or claiming run evidence.
Each observed row gets four EMPTY visual-review fields; a person must inspect
the PNG beside the original before filling them (1=yes, 0=no).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PORTRAITS = [f"{sex}_{i:02d}" for sex in ("man", "woman") for i in range(4)]
ORIGINAL = "Let the person wear a helmet"
SPECIFIC = "Put a single open-face bicycle helmet on the person's head"
NEGATIVE = "extra person, multiple people, extra face, duplicate face, different person"
REVIEW_FIELDS = ("helmet_appeared", "extra_person", "face_swapped", "usable")


def cells(run_id):
    """Parent links are one-variable comparisons, including within the grid."""
    result = []
    for ti, s_t in enumerate((7.5, 6.0, 5.0)):
        for ii, s_i in enumerate((1.5, 1.8, 2.2)):
            cell = f"t{int(s_t * 10)}_i{int(s_i * 10)}"
            parent = (f"t{int(s_t * 10)}_i{int((1.5, 1.8, 2.2)[ii - 1] * 10)}"
                      if ii else f"t{int((7.5, 6.0, 5.0)[ti - 1] * 10)}_i15"
                      if ti else "")
            result.append(dict(cell=cell, compare_to=parent,
                               changed_variable="s_i" if ii else "s_t" if ti else "baseline",
                               s_t=s_t, s_i=s_i, instruction=ORIGINAL, negative_prompt=""))
    for name, variable, instruction, negative in (
        ("specific", "instruction", SPECIFIC, ""),
        ("negative", "negative_prompt", ORIGINAL, NEGATIVE),
    ):
        result.append(dict(cell=name, compare_to="t75_i15", changed_variable=variable,
                           s_t=7.5, s_i=1.5, instruction=instruction,
                           negative_prompt=negative))
    for cell in result:
        cell.update(suffix=f"_{run_id}_{cell['cell']}", seed=20260812, steps=50)
        cell["arm"] = "ip2p" + cell["suffix"]
    return result


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def key(row):
    return row["arm"], row["image"], str(row["prompt_index"])


def verify_growth(before, after, cell, repo):
    old = {key(row): row for row in before}
    new = {key(row): row for row in after}
    if len(old) != len(before) or len(new) != len(after):
        raise RuntimeError("Duplicate (arm, image, prompt_index) CSV keys")
    expected = {(cell["arm"], name, "2") for name in PORTRAITS}
    if set(new) - set(old) != expected or len(after) != len(before) + 8:
        raise RuntimeError("Sweep did not add exactly eight distinct rows")
    if any(new.get(k) != row for k, row in old.items()):
        raise RuntimeError("An earlier CSV row was changed or removed")
    observed = [row for row in after if key(row) in expected]
    for row in observed:
        for field in ("s_t", "s_i", "seed", "steps"):
            if float(row[field]) != float(cell[field]):
                raise RuntimeError(f"Recorded {field} differs from the planned cell")
        for field in ("instruction", "negative_prompt"):
            if row[field] != cell[field]:
                raise RuntimeError(f"Recorded {field} differs from the planned cell")
        if not (repo / row["output_png"]).is_file():
            raise RuntimeError(f"Missing output: {row['output_png']}")
    return observed


def gpu_state(gpu):
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    selected = [r for r in csv.reader(listing.splitlines()) if int(r[0]) == gpu]
    if len(selected) != 1:
        raise RuntimeError(f"GPU {gpu} not found")
    _, uuid, memory = [part.strip() for part in selected[0]]
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True)
    pids = {int(row[1]) for row in csv.reader(applications.splitlines())
            if row and row[0].strip() == uuid}
    return int(memory), pids


def run_cell(command, gpu, env, log_path, repo):
    memory, pids = gpu_state(gpu)
    if pids or memory >= 1024:
        raise RuntimeError(f"GPU {gpu} is occupied: {memory} MiB, PIDs {pids}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        try:
            # Verify dispatch with ps, then check races immediately and throughout.
            subprocess.run(["ps", "-p", str(process.pid), "-o", "pid,ppid,args"], check=True)
            while process.poll() is None:
                memory, pids = gpu_state(gpu)
                foreign = pids - {process.pid}
                if foreign or (process.pid not in pids and memory >= 1024):
                    raise RuntimeError(f"GPU dispatch race on {gpu}: PIDs {pids}, {memory} MiB")
                time.sleep(1)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            # A failed ps/nvidia-smi check must not leave an unmonitored GPU job.
            if process.poll() is None:
                process.terminate()  # exact child PID; never pkill -f
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(5))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id must contain only letters, numbers, underscores or hyphens")
    if not args.plan_only and (args.gpu is None or sys.platform != "linux"):
        parser.error("Real runs require Linux and one explicit GPU in 0..4")
    repo = Path(__file__).resolve().parent.parent
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("--out must be a new directory; existing evidence is never overwritten")
    args.out.mkdir(parents=True)
    plan = cells(args.run_id)
    write_csv(args.out / "planned.csv", [
        dict(cell, image=name, prompt_index=2, status="planned")
        for cell in plan for name in PORTRAITS
    ])
    if args.plan_only:
        print(f"Planned only: {len(plan)} settings, {len(plan) * 8} images; 0 runs")
        return

    env = dict(os.environ, HF_HOME="/var/cache/huggingface",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    started = time.time()
    evidence = []
    verification = []
    csv_path = args.out / "preflight.csv"
    for cell in plan:
        before = read_csv(csv_path)
        command = [sys.executable, "scripts/edit_preflight.py", "--data", "data/portraits",
                   "--out", str(args.out), "--scenarios", "ip2p",
                   "--images", *PORTRAITS, "--prompt-indices", "2",
                   "--ip2p-instruction", cell["instruction"],
                   "--negative-prompt", cell["negative_prompt"],
                   "--s-t", str(cell["s_t"]), "--s-i", str(cell["s_i"]),
                   "--seed", str(cell["seed"]), "--suffix", cell["suffix"],
                   "--require-new-arm"]
        run_cell(command, args.gpu, env, args.out / f"{cell['cell']}.log", repo)
        after = read_csv(csv_path)
        observed = verify_growth(before, after, cell, repo)
        evidence.extend(dict(row, cell=cell["cell"], compare_to=cell["compare_to"],
                             changed_variable=cell["changed_variable"],
                             **{field: "" for field in REVIEW_FIELDS},
                             review_status="unreviewed", review_notes="") for row in observed)
        write_csv(args.out / "sweep.csv", evidence)
        verification.append(dict(cell=cell["cell"], rows_before=len(before),
                                 rows_after=len(after), new_rows=len(observed),
                                 unique_keys=len({key(row) for row in after})))
        (args.out / "verification.json").write_text(
            json.dumps(verification, indent=2) + "\n", encoding="utf-8")
        print(f"Verified {cell['cell']}: {len(before)} -> {len(after)} rows", flush=True)
    (args.out / "timing.json").write_text(json.dumps(dict(
        gpu=args.gpu, gpus_used=1, started_unix=started, finished_unix=time.time(),
        wall_seconds=time.time() - started, generated_images=len(evidence)), indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
