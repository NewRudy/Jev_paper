"""Generate Ambiguity-Regularized Fine-Tuning (AR-FT) dataset.

Combines standard k=1,2 chain training samples with balanced ambiguity anchor pairs.
Training on empirical ambiguity anchors prevents logit overconfidence and preserves
ECE calibration without degrading in-distribution accuracy.
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
    ap.add_argument("--ambig_ratio", type=float, default=0.10, help="Ratio of ambiguous anchors to chain samples (default 10%)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    
    # 1. Load or generate baseline chain train (600 samples)
    chain_train_path = OUT / "chain_train_k12.jsonl"
    if chain_train_path.exists():
        chain_samples = [json.loads(line) for line in open(chain_train_path)]
    else:
        print("chain_train_k12.jsonl not found; generating fresh...")
        chain_samples = []
        for k, n in [(1, 300), (2, 300)]:
            for _ in range(n):
                s = G.gen_chain(rng, k, noul=(rng.random() < 0.25))
                if s:
                    chain_samples.append({"state": s["state"], "questions": s["questions"]})

    num_ambig = int(len(chain_samples) * args.ambig_ratio)
    print(f"Adding {num_ambig} ambiguous anchor samples to {len(chain_samples)} chain samples (ratio={args.ambig_ratio})...")

    # 2. Generate balanced ambiguous samples
    # For diagonal points (|dr| == |dc|), generate pairs with balanced labels (one favoring NS, one favoring EW)
    ambig_samples = []
    seen = set()
    while len(ambig_samples) < num_ambig:
        s = G.gen_ambiguous(rng)
        if s is None:
            continue
        h = hash(s["state"])
        if h in seen:
            continue
        seen.add(h)
        
        # Balance label between primary (ns) and secondary (ew)
        q = s["questions"]["q1"]
        ideal = list(q.get("ideal_probs", {}).keys())
        if len(ideal) == 2:
            chosen_label = ideal[len(ambig_samples) % 2]
            q_clean = {
                "type": q["type"],
                "instructions": q["instructions"],
                "criteria": q["criteria"],
                "label": chosen_label
            }
            ambig_samples.append({"state": s["state"], "questions": {"q1": q_clean}})

    # 3. Combine and shuffle
    combined = chain_samples + ambig_samples
    rng.shuffle(combined)

    out_file = OUT / "chain_train_arft_k12.jsonl"
    with open(out_file, "w") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(combined)} samples in {out_file} (Chain: {len(chain_samples)}, Ambig: {len(ambig_samples)})")


if __name__ == "__main__":
    main()
