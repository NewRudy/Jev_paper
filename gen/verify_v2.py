"""Verify v2 chain & ambiguity data: parse relation facts from state text,
compose the path displacement independently, compare with stored labels.

Usage: python3 gen/verify_v2.py data/chain_*.jsonl data/ambiguous.jsonl
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import generator as G

DIRVEC = {d: G.DIR8_VEC[d] for d in G.DIR8_VEC}

FACT_PATTERNS = [
    (re.compile(r"- the (\w+) is (north|south|east|west|northeast|northwest|southeast|southwest) of the (\w+)\."),
     lambda m: (m.group(1), m.group(2), m.group(3))),          # a is d of b
    (re.compile(r"- the (\w+) lies to the (north|south|east|west|northeast|northwest|southeast|southwest) of the (\w+)\."),
     lambda m: (m.group(1), m.group(2), m.group(3))),          # a lies d of b
    (re.compile(r"- the (\w+) has the (\w+) to its (north|south|east|west|northeast|northwest|southeast|southwest)\."),
     lambda m: (m.group(2), m.group(3), m.group(1))),          # b has a to its d
]


def parse_facts(text):
    """Return list of (a, b, vec) meaning position(a) = position(b) + vec."""
    edges = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        hit = None
        for pat, fn in FACT_PATTERNS:
            m = pat.match(line)
            if m:
                hit = fn(m)
                break
        if hit is None:
            return None
        a, d, b = hit
        edges.append((a, b, DIRVEC[d]))
    return edges or None


def compose(edges, x, y):
    """BFS from x to y accumulating displacement; returns X relative to Y."""
    adj = collections.defaultdict(list)
    for a, b, v in edges:
        adj[a].append((b, v))          # pos(b) = pos(a) - v  (a is v of b)
        adj[b].append((a, (-v[0], -v[1])))
    seen = {x}
    q = collections.deque([(x, (0, 0))])
    while q:
        node, acc = q.popleft()
        if node == y:
            return acc                 # acc = pos(x) - pos(y) = x relative to y
        for nxt, v in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, (acc[0] + v[0], acc[1] + v[1])))
    return None


def to_dir8(v):
    for name, vec in DIRVEC.items():
        if v == vec:
            return name
    return None


def solve_chain(sample):
    q = sample["questions"]["q1"]
    edges = parse_facts(sample["state"])
    if edges is None:
        return "PARSE_FAIL"
    ins = q["instructions"].lower()
    m = re.search(r"is the (\w+) from the (\w+)", ins) or \
        re.search(r"does the (\w+) lie relative to the (\w+)", ins)
    noul = False
    if not m:
        m = re.search(r"is the (\w+) north of the (\w+)", ins) or \
            re.search(r"does the (\w+) lie to the north of the (\w+)", ins)
        noul = True
    if not m:
        return "Q_PARSE_FAIL"
    x, y = m.group(1), m.group(2)
    rel = compose(edges, x, y)          # x relative to y
    if rel is None:
        return "NO_PATH"
    if noul:
        return rel[0] < 0
    d = to_dir8(rel)
    return d if d is not None else "NOT_DIR8"


def solve_amb(sample):
    """Reuse verify.parse_state; check diagonal + tie-break label + ideal probs."""
    sys.path.insert(0, str(Path(__file__).parent))
    from verify import parse_state
    q = sample["questions"]["q1"]
    parsed = parse_state(sample["state"])
    if not parsed:
        return "STATE_FAIL"
    _, _, ents = parsed
    ins = q["instructions"].lower()
    m = re.search(r"directions? is the (\w+) from the (\w+)", ins)
    if m:
        a, b = m.group(1), m.group(2)
    else:
        m = re.search(r"stand at the (\w+), which cardinal direction is the (\w+)", ins)
        if not m:
            return "Q_FAIL"
        b, a = m.group(1), m.group(2)
    pa, pb = ents[a], ents[b]
    dr, dc = pa[0] - pb[0], pa[1] - pb[1]
    if dr == 0 or dc == 0 or abs(dr) != abs(dc):
        return "NOT_DIAGONAL"
    ns = "north" if dr < 0 else "south"
    ew = "west" if dc < 0 else "east"
    ip = q.get("ideal_probs", {})
    ok_label = q["label"] == ns
    ok_ideal = ip.get(ns) == 0.5 and ip.get(ew) == 0.5 and len(ip) == 2
    return "OK" if (ok_label and ok_ideal) else "MISMATCH(ns=%s,ew=%s,label=%s,ideal=%s)" % (ns, ew, q["label"], ip)


def main(paths):
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        bad = collections.Counter()
        for s in rows:
            if "chain" in str(p):
                got = solve_chain(s)
                if got != s["questions"]["q1"]["label"]:
                    bad[str(got)] += 1
            else:
                got = solve_amb(s)
                if got != "OK":
                    bad[got] += 1
        print("%-26s samples=%d  FAILS=%s"
              % (Path(p).name, len(rows), dict(bad) or "NONE"))


if __name__ == "__main__":
    main(sys.argv[1:])
