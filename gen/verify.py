"""Independent verifier: re-parse state TEXT and recompute every label.

Template-exact parsing (mirrors every instruction template in generator.py),
never trusts generator internals. Reports mismatches + noul true rates.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

LINE_PATTERNS = [
    re.compile(r"- the (.+?) is at row (\d+), column (\d+)\."),
    re.compile(r"- the (.+?) stands at cell \((\d+), (\d+)\)\."),
    re.compile(r"- the (.+?): row (\d+), column (\d+)\."),
]


def parse_state(text: str):
    ents = {}
    for pat in LINE_PATTERNS:
        for name, r, c in pat.findall(text):
            ents[name] = (int(r), int(c))
    m = re.search(r"(\d+) rows? and (\d+) columns", text) or \
        re.search(r"(\d+)x(\d+) grid", text) or \
        re.search(r"(\d+)-row by (\d+)-column grid", text)
    if not m or not ents:
        return None
    return int(m.group(1)), int(m.group(2)), ents


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dir_of(pa, pb, eight=False):
    dr, dc = pa[0] - pb[0], pa[1] - pb[1]
    if dr == 0 and dc == 0:
        return None
    ns = "north" if dr < 0 else "south" if dr > 0 else ""
    ew = "west" if dc < 0 else "east" if dc > 0 else ""
    if eight:
        return (ns + ew) if (ns and ew) else (ns or ew)
    if abs(dr) == abs(dc):          # diagonal: no dominant axis
        return None
    return ns if abs(dr) > abs(dc) else ew


def solve(q, ents):
    t, ins = q["type"], q["instructions"].lower()
    ins = ins.rstrip("?")

    # ---- direction choice (4 or 8 options) --------------------------------
    opts = list(q["criteria"].keys()) if t == "choice" else []
    dir_opts = {"north", "south", "east", "west", "northeast", "northwest",
                "southeast", "southwest"}
    if t == "choice" and set(opts) <= dir_opts:
        eight = "eight" in ins
        # "if you stand at {b}, ... is {a}"
        m = re.search(r"stand at the (\w+)", ins)
        if m:
            anchor = ents[m.group(1)]
            m2 = re.search(r"(?:direction|directions) is the (\w+)$", ins)
            if not m2:
                return None
            return dir_of(ents[m2.group(1)], anchor, eight)
        # "... is {a} from {b}" / "does {a} lie from {b}"
        m = re.search(r"is the (\w+) from the (\w+)$", ins) or \
            re.search(r"does the (\w+) lie from the (\w+)$", ins)
        if not m:
            return None
        return dir_of(ents[m.group(1)], ents[m.group(2)], eight)

    # ---- compare choice ----------------------------------------------------
    if t == "choice":
        # "which is closer to {a}: {b} or {c}?"
        m = re.search(r"closer to the (\w+): the (\w+) or the (\w+)$", ins)
        if m:
            an, bn, cn = m.group(1), m.group(2), m.group(3)
        else:
            # "which of {b} or {c} is nearer to {a}?"
            m = re.search(r"which of the (\w+) or the (\w+) is nearer to the (\w+)$", ins)
            if m:
                bn, cn, an = m.group(1), m.group(2), m.group(3)
            else:
                # "standing at {a}, which one is closer, {b} or {c}?"
                m = re.search(r"standing at the (\w+), which one is closer, the (\w+) or the (\w+)$", ins)
                if not m:
                    return None
                an, bn, cn = m.group(1), m.group(2), m.group(3)
        db, dc = dist(ents[an], ents[bn]), dist(ents[an], ents[cn])
        if db == dc:
            return None
        return bn if db < dc else cn

    if t == "noul":
        # ---- direction noul: "is {a} north of {b}" / "does {a} lie to the north of {b}"
        m = re.search(r"is the (\w+) (north|south|east|west) of the (\w+)$", ins) or \
            re.search(r"does the (\w+) lie to the (north|south|east|west) of the (\w+)$", ins)
        if m:
            pa, pb = ents[m.group(1)], ents[m.group(3)]
            return {"north": pa[0] < pb[0], "south": pa[0] > pb[0],
                    "west": pa[1] < pb[1], "east": pa[1] > pb[1]}[m.group(2)]
        # ---- threshold: three symmetric templates
        m = re.search(r"is the (\w+) within (\d+) grid steps of the (\w+)$", ins)
        if m:
            return dist(ents[m.group(1)], ents[m.group(3)]) <= int(m.group(2))
        m = re.search(r"is the step distance between the (\w+) and the (\w+) at most (\d+)$", ins)
        if m:
            return dist(ents[m.group(1)], ents[m.group(2)]) <= int(m.group(3))
        m = re.search(r"does it take at most (\d+) steps to walk from the (\w+) to the (\w+)$", ins)
        if m:
            return dist(ents[m.group(2)], ents[m.group(3)]) <= int(m.group(1))
        # ---- between: "is {c} strictly between {a} and {b} on a straight line"
        #           or "do {a}, {c} and {b} lie on one straight line with {c} in the middle"
        #           or "is {c} positioned on the straight segment joining {a} and {b}"
        m = re.search(r"is the (\w+) strictly between the (\w+) and the (\w+)", ins)
        if m:
            c, a, b = ents[m.group(1)], ents[m.group(2)], ents[m.group(3)]
        else:
            mm = re.search(r"with the (\w+) in the middle", ins)
            if mm:
                # "{a}, {c} and {b}" — middle one is in mm, others flank
                trio = re.search(r"do the (\w+), the (\w+) and the (\w+) lie", ins)
                if not trio:
                    return None
                c = ents[mm.group(1)]
                rest = [ents[trio.group(i)] for i in (1, 3)]
                a, b = rest
            else:
                m3 = re.search(r"is the (\w+) positioned on the straight segment joining the (\w+) and the (\w+)", ins)
                if not m3:
                    return None
                c, a, b = (ents[m3.group(i)] for i in (1, 2, 3))
        pa, pb, pc = a, b, c
        if pc == pa or pc == pb:
            return False
        if pa[0] == pb[0] == pc[0]:
            return (pa[1] - pc[1]) * (pc[1] - pb[1]) > 0
        if pa[1] == pb[1] == pc[1]:
            return (pa[0] - pc[0]) * (pc[0] - pb[0]) > 0
        if abs(pa[0] - pb[0]) == abs(pa[1] - pb[1]) \
                and abs(pa[0] - pc[0]) == abs(pa[1] - pc[1]):
            return (pa[0] - pc[0]) * (pc[0] - pb[0]) > 0
        return False

    if t == "score":
        tiers = q["criteria"]
        m0 = re.search(r"at most (\d+) grid steps from the (.+)", tiers[0])
        m1 = re.search(r"(\d+) to (\d+) grid steps from the (.+)", tiers[1])
        if not (m0 and m1):
            return None
        target = ents[m0.group(2)]
        mm = re.search(r"(?:the|at) ([a-z]+)$", ins)
        if not mm or mm.group(1) not in ents:
            return None
        d = dist(ents[mm.group(1)], target)
        if d <= int(m0.group(1)):
            return 0
        if d <= int(m1.group(2)):
            return 1
        return 2
    return None


def m_last(b, c, cond):
    return b if cond else c


def main(paths):
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        bad = collections.Counter()
        trues = 0
        noul_total = 0
        n = 0
        for s in rows:
            parsed = parse_state(s["state"])
            if parsed is None:
                bad["state_parse"] += 1
                continue
            _, _, ents = parsed
            for qid, q in s["questions"].items():
                n += 1
                got = solve(q, ents)
                if got is None:
                    bad[q["type"] + "_unsolved"] += 1
                elif got != q["label"]:
                    bad[q["type"] + "_mismatch"] += 1
                else:
                    if q["type"] == "noul":
                        noul_total += 1
                        if got is True:
                            trues += 1
        print("%-22s samples=%4d questions=%5d  FAILS=%s  noul_true=%d/%d"
              % (Path(p).name, len(rows), n, dict(bad) or "NONE",
                 trues, noul_total))


if __name__ == "__main__":
    main(sys.argv[1:])
