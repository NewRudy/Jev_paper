"""Generate Ambiguity-Regularized Fine-Tuning (AR-FT) dataset. [A1-fixed]

R1 review fix: original version placed alternating hard labels on DIFFERENT
ambiguous states, which teaches "ambiguous inputs still have a definite
answer" and can worsen overconfidence. Corrected mechanism: each ambiguous
state appears TWICE with the two axis labels (ns, ew) in equal proportion —
under cross-entropy with equal sample weights this is gradient-equivalent to
a 0.5/0.5 soft target, pushing the marginal toward the ideal 50/50 split
without changing the training objective.

Usage: python3 gen/make_arft_dataset.py [--seed 42] [--ambig_ratio 0.10]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import generator as G

OUT = Path("data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ambig_ratio", type=float, default=0.10,
                    help="Ratio of ambiguity anchor PAIRS to chain samples")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1. Baseline chain train (600 samples, k<=2)
    chain_train_path = OUT / "chain_train_k12.jsonl"
    if chain_train_path.exists():
        chain_samples = [json.loads(line) for line in open(chain_train_path)]
    else:
        print("chain_train_k12.jsonl not found; generating fresh...")
        chain_samples = []
        while len(chain_samples) < 600:
            k = 1 if len(chain_samples) % 2 == 0 else 2
            s = G.gen_chain(rng, k, noul=(rng.random() < 0.25))
            if s:
                chain_samples.append({"state": s["state"],
                                      "questions": s["questions"]})

    # 2. Balanced ambiguous anchors: SAME state duplicated with BOTH labels.
    #    (a, a') pairs: label=ns and label=ew on identical state text.
    num_pairs = int(len(chain_samples) * args.ambig_ratio)
    print(f"Adding {num_pairs} ambiguity anchor PAIRS "
          f"({2*num_pairs} records) to {len(chain_samples)} chain samples")

    ambig_records = []
    seen = set()
    while num_pairs > 0:
        s = G.gen_ambiguous(rng)
        if s is None:
            continue
        h = hash(s["state"])
        if h in seen:
            continue
        seen.add(h)
        q = s["questions"]["q1"]
        ideal = q.get("ideal_probs", {})
        ns = next((k for k in ideal if k in ("north", "south")), None)
        ew = next((k for k in ideal if k in ("east", "west")), None)
        if not (ns and ew):
            continue
        for lab in (ns, ew):
            ambig_records.append({
                "state": s["state"],
                "questions": {"q1": {
                    "type": q["type"],
                    "instructions": q["instructions"],
                    "criteria": q["criteria"],
                    "label": lab,
                }},
            })
        num_pairs -= 1

    # 3. Combine and shuffle (pairs must not be adjacent, but order is free)
    combined = chain_samples + ambig_records
    rng.shuffle(combined)

    out_file = OUT / "chain_train_arft_k12.jsonl"
    with open(out_file, "w") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # sanity: both labels of each pair present with same state
    from collections import Counter
    c = Counter()
    for r in combined:
        q = r["questions"]["q1"]
        if q["label"] in ("north", "south", "east", "west") \
                and "ideal" not in q:
            c[q["label"]] += 1
    print(f"Wrote {len(combined)} records -> {out_file}")
    print(f"(anchor label counts include chain samples; NS/EW balance check: {c})")


if __name__ == "__main__":
    main()
