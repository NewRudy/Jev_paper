"""Generate X1 depth-extrapolation data (R1 review request).

Train: chains with k=1..4 (in-distribution for ALL depths).
Test:  k=5, k=6 — never seen in training. If direct answering collapses at
k=5,6 while PMC (train-free composition) holds, the "depth extrapolation
without retraining" claim is established.

Usage: python3 gen/make_dataset_deep.py [--seed 23]
"""
from __future__ import annotations

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(exist_ok=True)

    seen = set()

    # train: 250 per k for k=1..4 (1000 total; 25% noul like k12 set)
    train = []
    for k in (1, 2, 3, 4):
        part = dedup(rng, lambda k=k: G.gen_chain(
            rng, k, noul=(rng.random() < 0.25)), 250, seen)
        train.extend(part)
    rng.shuffle(train)
    with open(OUT / "chain_train_deep_k14.jsonl", "w") as f:
        for s in train:
            f.write(json.dumps({"state": s["state"],
                                "questions": s["questions"]},
                               ensure_ascii=False) + "\n")

    # tests: k=5,6 (100 each)
    for k in (5, 6):
        part = dedup(rng, lambda k=k: G.gen_chain(
            rng, k, noul=(rng.random() < 0.25)), 100, seen)
        with open(OUT / ("chain_test_k%d.jsonl" % k), "w") as f:
            for s in part:
                f.write(json.dumps({"state": s["state"],
                                    "questions": s["questions"]},
                                   ensure_ascii=False) + "\n")

    # report
    for name in ("chain_train_deep_k14", "chain_test_k5", "chain_test_k6"):
        rows = [json.loads(l) for l in open(OUT / (name + ".jsonl"))]
        noul = sum(1 for r in rows if r["questions"]["q1"]["type"] == "noul")
        print(f"{name}: {len(rows)} samples, noul={noul}")


if __name__ == "__main__":
    main()
