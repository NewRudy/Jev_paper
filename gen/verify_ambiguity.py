"""Independent verifier for the ambiguity typology (A1 / A4 / A5).

Re-reads the rendered state text and checks the *semantic* precondition of
each type, rather than trusting the generator:
  A1  the queried pair really is diagonal (equal |dr| and |dc|)
  A4  no line places both queried systems in a subnet
  A5  the log really contains one assertion and its negation
  training anchors: every state carries both complementary labels
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DIR4 = {"north", "south", "east", "west"}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


def check_a1(rows, expect_convention):
    """Re-parse the coordinates and confirm the queried pair is diagonal and
    the stored label is the north-south component under our convention. When
    expect_convention, the convention sentence must also be present."""
    sys.path.insert(0, str(Path(__file__).parent))
    from verify import parse_state
    bad = 0
    for r in rows:
        st, q = r["state"], r["questions"]["q1"]
        parsed = parse_state(st)
        if not parsed:
            bad += 1
            continue
        _, _, ents = parsed
        ins = q["instructions"]
        present = sorted((n for n in ents if re.search(r"\b%s\b" % re.escape(n), ins)),
                         key=lambda n: ins.index(n))
        if len(present) < 2:
            bad += 1
            continue
        pa, pb = ents[present[0]], ents[present[1]]
        dr, dc = pa[0] - pb[0], pa[1] - pb[1]
        if dr == 0 or dc == 0 or abs(dr) != abs(dc):
            bad += 1
            continue
        want = "north" if dr < 0 else "south"
        if q["label"] != want:
            bad += 1
        if expect_convention and "Tie-break rule" not in st:
            bad += 1
    return bad


def check_a4(rows):
    """No line may place both queried systems in a subnet, and the label must
    be the abstention option."""
    bad = 0
    for r in rows:
        st, q = r["state"], r["questions"]["q1"]
        pair = re.findall(r"- (\w+) and (\w+) do not appear", st)
        if len(pair) != 1:
            bad += 1
            continue
        a, b = pair[0]
        for line in st.split("\n"):
            if a in line and b in line and "do not appear" not in line:
                bad += 1
                break
        else:
            if q["label"] != "not_recorded":
                bad += 1
    return bad


def check_a5(rows):
    bad = 0
    for r in rows:
        st, q = r["state"], r["questions"]["q1"]
        dirs = re.findall(r"to the (north|south|east|west)|is (north|south|east|west) of|"
                          r"to its (north|south|east|west)", st)
        flat = [d for t in dirs for d in t if d]
        if len(flat) < 2 or not any(OPP.get(flat[0]) == f for f in flat[1:]):
            bad += 1                      # no contradiction present
        if q["label"] != "cannot_determine":
            bad += 1
        opts = list(q["criteria"].keys())
        if "cannot_determine" not in opts:
            bad += 1
    return bad


def check_dual(rows):
    by = defaultdict(set)
    for r in rows:
        by[r["state"]].add(str(r["questions"]["q1"]["label"]))
    return sum(1 for v in by.values() if len(v) != 2)


def main():
    D = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    jobs = [("amb_a1_stated.jsonl", lambda r: check_a1(r, True)),
            ("amb_a1_unstated.jsonl", lambda r: check_a1(r, False)),
            ("amb_a4.jsonl", check_a4), ("amb_a5.jsonl", check_a5)]
    for name, fn in jobs:
        p = D / name
        if not p.exists():
            print(f"{name:22s} MISSING")
            continue
        rows = [json.loads(l) for l in open(p)]
        bad = fn(rows)
        print(f"{name:22s} n={len(rows):4d}  FAILS={bad}  {'OK' if bad == 0 else 'PROBLEM'}")
    for name in ("amb_train_a1a5.jsonl", "amb_train_a1only.jsonl"):
        p = D / name
        if p.exists():
            rows = [json.loads(l) for l in open(p)]
            uniq = len({r["state"] for r in rows})
            print(f"{name:22s} n={len(rows):4d}  distinct states={uniq}")


if __name__ == "__main__":
    main()
