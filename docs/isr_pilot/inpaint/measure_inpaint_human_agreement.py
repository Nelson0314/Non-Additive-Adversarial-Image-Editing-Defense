"""Agreement between the human review and the Claude Sonnet judgments for the SD Inpainting ISR.

Inputs: answers.csv (item,undefended_success,content,mismatch,degraded,note) from the interactive
review page, and key.csv of the review sample (item,...,condition,purification,sonnet_success,...).
The sample holds 60 cells in which Sonnet judged the Undefended edit successful (10 per defense
condition, 5 judged defense successes and 5 judged defense failures). A cell enters the comparison
only when the human reviewer also judges the Undefended edit successful.
"""
import csv
import sys
from collections import Counter, defaultdict


def main(answers_path, key_path):
    key = {r["item"]: r for r in csv.DictReader(open(key_path))}
    ans = {r["item"]: r for r in csv.DictReader(open(answers_path))}
    assert set(ans) == set(key), (len(ans), len(key))
    excluded = [i for i in key if ans[i]["undefended_success"] != "1"]
    print(f"Items: {len(key)}; Undefended edit judged unsuccessful by the human reviewer: {len(excluded)}")
    by_cond = Counter(key[i]["condition"] for i in excluded)
    print("  excluded by condition: " + ", ".join(f"{c} {n}" for c, n in sorted(by_cond.items())))
    conf = Counter()
    per_cond = defaultdict(Counter)
    labels = Counter()
    for i, k in key.items():
        a = ans[i]
        if a["undefended_success"] != "1":
            continue
        human_edit_ok = a["content"] == "1" and a["mismatch"] == "0" and a["degraded"] == "0"
        sonnet_edit_ok = k["sonnet_success"] == "1"
        conf[(sonnet_edit_ok, human_edit_ok)] += 1
        per_cond[k["condition"]][sonnet_edit_ok == human_edit_ok] += 1
        if not human_edit_ok:
            labels["content=0"] += a["content"] == "0"
            labels["mismatch=1"] += a["mismatch"] == "1"
            labels["degraded=1"] += a["degraded"] == "1"
    n = sum(conf.values())
    agree = conf[(True, True)] + conf[(False, False)]
    print(f"\nCompared cells: {n}; agreement {agree}/{n} ({100 * agree / n:.1f}%)")
    print("Confusion (rows Sonnet, columns human; 'edit ok' = edit carried out, i.e. defense failed):")
    print(f"  Sonnet edit ok   : human edit ok {conf[(True, True)]:3d} | human blocked {conf[(True, False)]:3d}")
    print(f"  Sonnet blocked   : human edit ok {conf[(False, True)]:3d} | human blocked {conf[(False, False)]:3d}")
    print("Agreement by condition:")
    for c in sorted(per_cond):
        a, d = per_cond[c][True], per_cond[c][False]
        print(f"  {c:12s} {a}/{a + d}")
    print("Human labels on cells judged blocked: " + ", ".join(f"{k} {v}" for k, v in labels.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
