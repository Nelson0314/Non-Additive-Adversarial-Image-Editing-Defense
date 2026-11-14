"""Write audit/judgments_human_corrected.csv: audit/judgments_human.csv with eight labels corrected.

The eight cells are CAPGuard (ImmuLUT) edits whose label (seven human, one judge) marks the edit as
failed although the sheet shows the requested accessory worn by the person. Each was checked against the sheet
`<sheet>.jpg` in baseline artifacts/pexels_portraits/ip2p_isr100_sheets/. The source file is not
modified; the corrected file has one added column, `correction_note`, empty for unchanged cells.

Usage: python3 generate_isr_corrected_judgments.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "audit", "judgments_human.csv")
TARGET = os.path.join(HERE, "audit", "judgments_human_corrected.csv")

# (sheet, cell) -> (purification, labels in the source file, reason); man_37__p2 C5-R7 is a judge label, the others human
CORRECTIONS = {
    ("man_37__p2", "C2-R7"): ("none", ("0", "1", "0"), "sheet shows a metal helmet on the head, same as the undefended edit"),
    ("man_37__p2", "C3-R7"): ("jpeg50", ("0", "1", "0"), "sheet shows a metal helmet on the head, same as the undefended edit"),
    ("man_37__p2", "C4-R7"): ("blur2", ("0", "1", "0"), "sheet shows a dark helmet on the head, same as the undefended edit"),
    ("man_37__p2", "C5-R7"): ("vae_trans", ("0", "0", "0"), "sheet shows a metal helmet on the head, same as the undefended edit"),
    ("woman_50__p1", "C5-R7"): ("vae_trans", ("0", "1", "0"), "sheet shows a police uniform with shoulder patch and badge"),
    ("woman_22__p3", "C5-R7"): ("vae_trans", ("0", "0", "0"), "sheet shows a red bowtie at the collar"),
    ("woman_97__p2", "C2-R7"): ("none", ("0", "0", "0"), "sheet shows a metal helmet on the head"),
    ("woman_97__p2", "C3-R7"): ("jpeg50", ("0", "1", "0"), "sheet shows a metal helmet on the head"),
}
LABELS = ("accessory", "mismatch", "degraded")

with open(SOURCE, newline="") as f:
    reader = csv.DictReader(f)
    fields = list(reader.fieldnames)
    rows = list(reader)

applied = set()
for row in rows:
    key = (row["sheet"].strip(), row["cell"].strip())
    row["correction_note"] = ""
    if key not in CORRECTIONS:
        continue
    purification, old, reason = CORRECTIONS[key]
    found = tuple(row[name].strip() for name in LABELS)
    if row["condition"] != "CAPGuard" or row["purification"] != purification or found != old:
        raise SystemExit(f"unexpected source row for {key}: {row}")
    row["accessory"], row["mismatch"], row["degraded"] = "1", "0", "0"
    source = "human" if row["note"].startswith("human:") else "judge"
    row["correction_note"] = f"{source} label {'/'.join(old)} -> 1/0/0: {reason}"
    applied.add(key)
if applied != set(CORRECTIONS):
    raise SystemExit(f"cells not found: {sorted(set(CORRECTIONS) - applied)}")

with open(TARGET, "w", newline="") as f:
    writer = csv.DictWriter(f, fields + ["correction_note"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {TARGET}: {len(rows)} rows, {len(applied)} corrected")
