"""Apply the 206-image human audit (three-way labels) to the judge's ISR judgments.

Each audit label is one of: edit succeeded (success=1), edit failed (success=0), or the
undefended reference edit under the same purification failed (ref_fail=1). A ref_fail label
marks the slot (image x prompt x purification) as ineligible: the Undefended cell of that slot
is set to failure, so the conditional ISR drops the slot from every condition's denominator.
Every other audited cell takes the human success label (accessory=success, mismatch=0, degraded=0).

Outputs (to --out-dir):
  audit_labels.csv      case_id, set, sheet, cell, condition, purification, human and judge labels
  judgments_audited.csv judgments with the audit applied
  agreement.txt         judge/human agreement on the derived success label, excluding ref_fail cells

Usage: python generate_isr_audited_judgments.py --judgments judgments_all306.csv \
         --key audit206/audit_key.csv --labels-dir audit206/labels --out-dir audit206
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

key = {r["case_id"]: r for r in csv.DictReader(open(args.key, encoding="utf-8"))}
human = {}
for f in glob.glob(os.path.join(args.labels_dir, "*.json")):
    d = json.load(open(f, encoding="utf-8"))
    cid = os.path.basename(f)[:-5]
    if "ref_fail" not in d:  # saved by the page version without ref_fail/success fields
        if d.get("accessory") not in (0, 1):
            raise SystemExit(f"{cid}: accessory={d.get('accessory')!r}")
        d = dict(d, ref_fail=0, success=d["accessory"])
    if d["ref_fail"] == 1:
        if d.get("success") is not None:
            raise SystemExit(f"{cid}: ref_fail with success={d['success']!r}")
    elif d["ref_fail"] != 0 or d.get("success") not in (0, 1):
        raise SystemExit(f"{cid}: ref_fail={d['ref_fail']!r} success={d.get('success')!r}")
    human[cid] = d
if set(human) != set(key):
    raise SystemExit(f"label/key mismatch: {len(human)} labels, {len(key)} key rows, "
                     f"missing {sorted(set(key) - set(human))[:5]}")

judg = list(csv.DictReader(open(args.judgments, encoding="utf-8")))
index = {(j["sheet"].strip(), j["cell"].strip()): j for j in judg}
undefended = {(j["sheet"].strip(), j["purification"].strip()): j for j in judg if j["condition"] == "Undefended"}
ok = lambda r: int(r["accessory"].strip() == "1" and r["mismatch"].strip() == "0" and r["degraded"].strip() == "0")

rows = []
for cid in sorted(key):
    k, h = key[cid], human[cid]
    j = index.get((k["sheet"], k["cell"]))
    if j is None or j["condition"] != k["condition"] or j["purification"] != k["purification"]:
        raise SystemExit(f"key mismatch at {cid}")
    rows.append(dict(case_id=cid, set=k["set"], sheet=k["sheet"], cell=k["cell"], condition=k["condition"],
                     purification=k["purification"], h_ref_fail=h["ref_fail"],
                     h_success="" if h["ref_fail"] else h["success"], h_unsure=int(bool(h.get("unsure"))),
                     h_note=(h.get("note") or "").replace("\n", " "), j_success=ok(j),
                     j_reference_success=ok(undefended[(k["sheet"], k["purification"])])))
os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, "audit_labels.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

out = []


def agree(sub):
    n = len(sub)
    a = sum(r["h_success"] == r["j_success"] for r in sub)
    hp = sum(r["h_success"] == 1 and r["j_success"] == 0 for r in sub)
    jp = sum(r["h_success"] == 0 and r["j_success"] == 1 for r in sub)
    return f"{a}/{n} ({100 * a / n:.1f}%)  human1-judge0 {hp}  human0-judge1 {jp}"


rated = [r for r in rows if not r["h_ref_fail"]]
out.append(f"audited cells {len(rows)}; ref_fail {len(rows) - len(rated)}; unsure {sum(r['h_unsure'] for r in rows)}")
out.append(f"ref_fail cells whose reference the judge rated a success: "
           f"{sum(r['j_reference_success'] for r in rows if r['h_ref_fail'])}/{len(rows) - len(rated)}")
for name in ("all", "random", "discordant"):
    sub = [r for r in rated if name == "all" or r["set"] == name]
    out.append(f"[{name}] success agreement {agree(sub)}")
for field in ("condition", "purification"):
    groups = collections.defaultdict(list)
    for r in rated:
        if r["set"] == "random":
            groups[r[field]].append(r)
    for g in sorted(groups):
        out.append(f"  random by {field} {g:11s} {agree(groups[g])}")

slots = set()
n_cells = 0
for r in rows:
    if r["h_ref_fail"]:
        slots.add((r["sheet"], r["purification"]))
    else:
        j = index[(r["sheet"], r["cell"])]
        j["accessory"], j["mismatch"], j["degraded"] = str(r["h_success"]), "0", "0"
        j["note"] = "human: " + r["h_note"]
        n_cells += 1
for s in slots:
    u = undefended[s]
    u["accessory"], u["mismatch"], u["degraded"] = "0", "0", "0"
    u["note"] = "human: reference edit failed; slot excluded"
out.append(f"judgments_audited.csv: {n_cells} cells take the human label; {len(slots)} slots excluded by ref_fail")
with open(os.path.join(args.out_dir, "judgments_audited.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(judg[0]))
    w.writeheader()
    w.writerows(judg)
open(os.path.join(args.out_dir, "agreement.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
