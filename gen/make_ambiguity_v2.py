"""Build ambiguity typology v2 datasets + the E3 random-anchor control.

Outputs (data/):
  amb_a1.jsonl        eval, tie-break ambiguity (diagonal pair, 4-way)
  amb_a4.jsonl        eval, evidence absence (no connecting fact)
  amb_a5.jsonl        eval, conflicting evidence (contradictory facts)
  amb_mixed.jsonl     eval, all three pooled (300)
  amb_train_a1a5.jsonl training, A1+A5 dual-label anchors (CE ≡ 50/50 soft)
  ctrl_random60.jsonl E3 control: 60 RANDOM non-ambiguous chain samples

Usage: python3 gen/make_ambiguity_v2.py [--seed 31]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import generator as G
import ambiguity_v2 as A

OUT = Path("data")


def dedup(gen, n, seen, tries=400):
    out = []
    guard = 0
    while len(out) < n and guard < n * tries:
        guard += 1
        s = gen()
        if s is None:
            continue
        h = hash(s["state"] + s["questions"]["q1"]["instructions"])
        if h in seen:
            continue
        seen.add(h)
        out.append(s)
    return out


def write(name, rows):
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"state": r["state"], "questions": r["questions"]},
                               ensure_ascii=False) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(exist_ok=True)

    seen = set()
    pending = []          # (name, rows) written after all generators run
    a1 = dedup(lambda: A.gen_a1(rng), 150, seen)
    a4 = dedup(lambda: A.gen_a4(rng), 150, seen)
    a5 = dedup(lambda: A.gen_a5(rng), 150, seen)
    mixed = a1 + a4 + a5
    rng.shuffle(mixed)

    # training anchors: A1 + A5 dual-label (A4 has no meaningful dual label)
    train_recs = []
    for s in (dedup(lambda: A.gen_a1(rng), 40, seen) +
              dedup(lambda: A.gen_a5(rng), 40, seen)):
        d = A.to_dual_label(s)
        if d:
            train_recs.extend(d)
    rng.shuffle(train_recs)

    # E5: A1-only anchors (train on one ambiguity type, test transfer to the
    # others) -> does the repair teach "express uncertainty" or just A1?
    a1_only = []
    for s in dedup(lambda: A.gen_a1(rng), 40, seen):
        d = A.to_dual_label(s)
        if d:
            a1_only.extend(d)
    rng.shuffle(a1_only)
    pending.append(("amb_train_a1only", a1_only))

    # E6: anchor-ratio sweep files (5/20/40 percent of chain size)
    chain_rows = [json.loads(l) for l in open(OUT / "chain_train_k12.jsonl")]
    for pct in (5, 20, 40):
        keep = min(len(train_recs), int(pct / 100 * len(chain_rows)))
        sub = list(train_recs); rng.shuffle(sub)
        combined = chain_rows + sub[:keep]
        rng.shuffle(combined)
        pending.append((f"amb_r{pct}", combined))

    # E3 control: 60 random NON-ambiguous chain samples (same count, same
    # format, no ambiguity information) -> isolates "diversity" from
    # "ambiguity-specific anchoring"
    ctrl = []
    seen_c = set()
    while len(ctrl) < 60:
        s = G.gen_chain(rng, rng.choice([1, 2]), noul=(rng.random() < 0.25))
        if not s:
            continue
        h = hash(s["state"])
        if h in seen_c:
            continue
        seen_c.add(h)
        ctrl.append({"state": s["state"], "questions": s["questions"]})

    counts = {
        "amb_a1": write("amb_a1", a1),
        "amb_a4": write("amb_a4", a4),
        "amb_a5": write("amb_a5", a5),
        "amb_mixed": write("amb_mixed", mixed),
        "amb_train_a1a5": write("amb_train_a1a5", train_recs),
        "ctrl_random60": write("ctrl_random60", ctrl),
    }
    for name, rows in pending:
        counts[name] = write(name, rows)

    # sanity: every training anchor state must appear twice with complementary labels
    by_state = {}
    for r in train_recs:
        by_state.setdefault(r["state"], set()).add(str(r["questions"]["q1"]["label"]))
    dual = sum(1 for v in by_state.values() if len(v) == 2)
    print(json.dumps(counts, indent=2))
    print(f"training anchor states: {len(by_state)}, dual-label: {dual}, "
          f"single-label (must be 0): {len(by_state) - dual}")


if __name__ == "__main__":
    main()
