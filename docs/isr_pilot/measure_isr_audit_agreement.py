"""Compare the human audit labels with the Claude judge on the audited ip2p ISR cells.

Inputs: judgments.csv (judge), audit_key.csv (case_id -> sheet, cell, condition, purification, set),
and a directory of human label JSON files (<case_id>.json with accessory, mismatch, degraded, unsure, note).
Outputs (to stdout and --out-dir):
  human_labels.csv          case_id, sheet, cell, condition, purification, set, human and judge labels
  judgments_human.csv       judgments.csv with every audited cell replaced by the human labels
Agreement is reported per label and for the derived success label (accessory=1, mismatch=0, degraded=0),
for the random set by condition and purification, and for the CAPGuard/BlurGuard discordant pairs.

Usage: python3 measure_isr_audit_agreement.py --judgments judgments.csv --key audit_key.csv \
         --labels-dir labels --out-dir out
"""
import argparse
import collections
import csv
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--judgments", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--labels-dir", required=True)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()

key = {r["case_id"]: r for r in csv.DictReader(open(args.key))}
human = {}
for f in glob.glob(os.path.join(args.labels_dir, "*.json")):
    d = json.load(open(f, encoding="utf-8"))
    d = d.get("data", d)
    human[os.path.basename(f)[:-5]] = d
if set(human) != set(key):
    raise SystemExit(f"label/key mismatch: {len(human)} labels, {len(key)} key rows, "
                     f"missing {sorted(set(key) - set(human))[:5]}")
for cid, d in human.items():
    for f in ("accessory", "mismatch", "degraded"):
        if d.get(f) not in (0, 1):
            raise SystemExit(f"{cid}: {f}={d.get(f)!r}")

ok = lambda a, m, g: int(a == 1 and m == 0 and g == 0)
rows = []
for cid in sorted(key):
    k, h = key[cid], human[cid]
    rows.append(dict(case_id=cid, set=k["set"], sheet=k["sheet"], cell=k["cell"], condition=k["condition"],
                     purification=k["purification"],
                     h_accessory=h["accessory"], h_mismatch=h["mismatch"], h_degraded=h["degraded"],
                     h_unsure=int(bool(h.get("unsure"))), h_note=(h.get("note") or "").replace("\n", " "),
                     j_accessory=int(k["accessory"]), j_mismatch=int(k["mismatch"]), j_degraded=int(k["degraded"]),
                     j_note=k["note"]))
for r in rows:
    r["h_success"] = ok(r["h_accessory"], r["h_mismatch"], r["h_degraded"])
    r["j_success"] = ok(r["j_accessory"], r["j_mismatch"], r["j_degraded"])
os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, "human_labels.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def agree(sub, label):
    n = len(sub)
    a = sum(r["h_" + label] == r["j_" + label] for r in sub)
    hp = sum(r["h_" + label] == 1 and r["j_" + label] == 0 for r in sub)  # human 1, judge 0
    jp = sum(r["h_" + label] == 0 and r["j_" + label] == 1 for r in sub)  # human 0, judge 1
    return f"{a}/{n} ({100 * a / n:.1f}%)  human1-judge0 {hp}  human0-judge1 {jp}"


print(f"audited cases {len(rows)}; unsure {sum(r['h_unsure'] for r in rows)}")
for name, sub in [("all", rows), ("random", [r for r in rows if r["set"] == "random"]),
                  ("discordant", [r for r in rows if r["set"] == "discordant"])]:
    print(f"\n[{name}] n={len(sub)}")
    for label in ("accessory", "mismatch", "degraded", "success"):
        print(f"  {label:9s} {agree(sub, label)}")

rnd = [r for r in rows if r["set"] == "random"]
print("\n[random] success agreement by condition (20 cells each) and by purification (35 each)")
for field in ("condition", "purification"):
    groups = collections.defaultdict(list)
    for r in rnd:
        groups[r[field]].append(r)
    for g in sorted(groups):
        print(f"  {g:11s} {agree(groups[g], 'success')}")

print("\n[random] success, human vs judge, by condition: judge-success rate / human-success rate")
groups = collections.defaultdict(list)
for r in rnd:
    groups[r["condition"]].append(r)
for g in sorted(groups):
    s = groups[g]
    print(f"  {g:11s} judge {sum(r['j_success'] for r in s)}/{len(s)}  human {sum(r['h_success'] for r in s)}/{len(s)}")

print("\n[discordant pairs] judge says exactly one of CAPGuard/BlurGuard blocked; human outcome per pair")
pairs = collections.defaultdict(dict)
for r in rows:
    if r["set"] == "discordant":
        pairs[(r["sheet"], r["purification"])][r["condition"]] = r
outcome = collections.Counter()
for (sheet, pur), p in pairs.items():
    cap, blur = p["CAPGuard"], p["BlurGuard"]
    judge_winner = "CAPGuard" if cap["j_success"] == 0 else "BlurGuard"
    hc, hb = 1 - cap["h_success"], 1 - blur["h_success"]  # 1 = defense succeeded
    human_view = {(1, 0): "CAPGuard only", (0, 1): "BlurGuard only", (1, 1): "both blocked", (0, 0): "neither"}[(hc, hb)]
    outcome[(judge_winner, "purified" if pur != "none" else "none", human_view)] += 1
for k in sorted(outcome):
    print(f"  judge: {k[0]:9s} only | {k[1]:8s} | human: {k[2]:15s} {outcome[k]}")

judg = list(csv.DictReader(open(args.judgments)))
override = {(r["sheet"], r["cell"]): r for r in rows}
n = 0
for j in judg:
    o = override.get((j["sheet"].strip(), j["cell"].strip()))
    if o:
        if o["condition"] != j["condition"] or o["purification"] != j["purification"]:
            raise SystemExit(f"key mismatch at {o['case_id']}")
        j["accessory"], j["mismatch"], j["degraded"] = str(o["h_accessory"]), str(o["h_mismatch"]), str(o["h_degraded"])
        j["note"] = "human: " + o["h_note"]
        n += 1
with open(os.path.join(args.out_dir, "judgments_human.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(judg[0]))
    w.writeheader()
    w.writerows(judg)
print(f"\njudgments_human.csv: {n} cells replaced by human labels")
