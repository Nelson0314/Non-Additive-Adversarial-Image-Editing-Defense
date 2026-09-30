"""Coordinator-side applier: verify HEAD, reviewed worktree files and patch hashes, then commit each patch in order."""
import hashlib, json, subprocess, sys
P = sys.argv[1]
s = json.load(open(f"{P}/series.json", encoding="utf-8"))
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert head.startswith(s["base"][:7]) or head == s["base"], (head, s["base"])
bad = []
for k, v in s.get("worktree_sha256", {}).items():
    if v is None:
        continue
    if hashlib.sha256(open(k, "rb").read()).hexdigest() != v:
        bad.append(k)
assert not bad, bad[:10]
for e in s["entries"]:
    assert hashlib.sha256(open(f"{P}/{e['patch']}", "rb").read()).hexdigest() == e["sha256"], e["patch"]
    subprocess.run(["git", "apply", "--cached", f"{P}/{e['patch']}"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", e["message"]], check=True)
    print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())
