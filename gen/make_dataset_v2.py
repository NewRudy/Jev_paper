"""Build the v2 Spatial System One dataset: relation-fact chains (k-hop
composition), systematicity split, and inherent-ambiguity calibration set.

Usage: python3 gen/make_dataset_v2.py [--seed 11]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import generator as G

OUT = Path("data")


def dedup(rng, gen, n, seen):
    out = []
    while len(out) < n:
        s = gen()
        if s is None:
            continue
        h = hash(s["state"] + json.dumps(s["questions"], sort_keys=True)[:120])
        if h in seen:
            continue
        seen.add(h)
        out.append(s)
    return out


def strip_meta(samples):
    return [{"state": s["state"], "questions": s["questions"]} for s in samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(exist_ok=True)

    seen = set()

    # ---- chain train: k=1 (300) + k=2 (300), 25% noul ----
    train = []
    for k, n in [(1, 300), (2, 300)]:
        part = dedup(rng, lambda k=k: G.gen_chain(
            rng, k, noul=(rng.random() < 0.25)), n, seen)
        train.extend(part)
    rng.shuffle(train)
    write(OUT / "chain_train_k12.jsonl", strip_meta(train))

    # ---- chain tests: k = 1..4, 100 each (k=3,4 are compositional OOD) ----
    for k in range(1, 5):
        part = dedup(rng, lambda k=k: G.gen_chain(
            rng, k, noul=(rng.random() < 0.25)), 100, seen)
        write(OUT / ("chain_test_k%d.jsonl" % k), strip_meta(part))

    # ---- ambiguity calibration set (pure eval, 150) ----
    amb = dedup(rng, lambda: G.gen_ambiguous(rng), 150, set())
    write(OUT / "ambiguous.jsonl", strip_meta(amb))

    # ---- coordinate contrastive 1600 pairs (scaling ladder prefixes) ----
    seen2 = set()
    big = []
    while len(big) < 3200:
        pair = G.make_pair(rng, (6, 9), (4, 6), False, False)
        if pair and all(hash(s["state"]) not in seen2 for s in pair):
            for s in pair:
                seen2.add(hash(s["state"]))
            big.extend(pair)
    write(OUT / "coord_train_1600.jsonl", big)

    # ---- report ----
    for f in sorted(OUT.glob("*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        ks = collections.Counter()
        noul = 0
        for r in rows:
            for q in r["questions"].values():
                if q["type"] == "noul":
                    noul += 1
        m = rows[0].get("meta", {}) if rows and isinstance(rows[0], dict) else {}
        print("%-28s %4d samples  %4d questions  noul=%d"
              % (f.name, len(rows), sum(len(r["questions"]) for r in rows), noul))


def write(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
