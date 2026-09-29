"""Record the pre-migration evidence baseline: blob ids, worktree hashes, CSV shapes."""
import csv, hashlib, io, json, subprocess, sys
root = sys.argv[1]; out = sys.argv[2]
ls = subprocess.run(["git", "-C", root, "ls-files", "-s"], capture_output=True, text=True, encoding="utf-8", check=True).stdout
files = {}
for line in ls.splitlines():
    meta, path = line.split("\t", 1)
    mode, blob, _ = meta.split()
    data = open(f"{root}/{path}", "rb").read()
    rec = {"blob": blob, "worktree_sha256": hashlib.sha256(data).hexdigest(),
           "lf_sha256": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}
    if path.endswith(".csv"):
        text = data.decode("utf-8-sig").replace("\r\n", "\n")
        rows = list(csv.reader(io.StringIO(text)))
        rec["header"] = rows[0] if rows else []
        rec["rows"] = max(len(rows) - 1, 0)
    files[path] = rec
json.dump({"commit": subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "files": files}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)
print(len(files), sum(1 for p in files if p.endswith(".csv")))
