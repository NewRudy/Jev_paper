"""Build the Spatial System One pilot dataset (Kev fine-tune JSONL format).

Usage:
    python3 gen/make_dataset.py [--out-dir data] [--seed 7]

Outputs train / eval_id / eval_ood_size / eval_ood_count / eval_ood_rule /
train_random (RQ2 ablation), each one JSON request per line, plus stats.txt.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import generator as G

SPLITS = {
    # name: (rows_range, ents_range, n_pairs, use_dir8, ood_rule)
    "train":         ((6, 9), (4, 6), 400, False, False),
    "eval_id":       ((6, 9), (4, 6), 75,  False, False),
    "eval_ood_size": ((12, 15), (4, 6), 50, False, False),
    "eval_ood_count": ((6, 9), (7, 9), 50, False, False),
    "eval_ood_rule": ((6, 9), (4, 6), 50, True,  True),
}


def qtype(q: dict) -> str:
    if q["type"] == "noul":
        if "straight" in q["instructions"] or "line" in q["instructions"] \
                or "segment" in q["instructions"]:
            return "between"
        return "threshold" if (
            "within" in q["instructions"] or "steps" in q["instructions"]
            or "distance" in q["instructions"].lower()) else "direction_noul"
    if q["type"] == "choice":
        crit = list(q["criteria"].keys())
        if crit and crit[0] in G.DIR8:
            return "direction8_choice" if len(crit) == 8 else "direction_choice"
        return "compare_choice"
    return "score"


def build_split(rng, rows, ents, n_pairs, use_dir8, ood_rule, seen):
    samples = []
    for _ in range(n_pairs):
        for _try in range(200):
            pair = G.make_pair(rng, rows, ents, use_dir8, ood_rule)
            if pair is None:
                continue
            if any(hash(s["state"]) in seen for s in pair):
                continue
            for s in pair:
                seen.add(hash(s["state"]))
            samples.extend(pair)
            break
    return samples


def stats_for(samples) -> str:
    types = collections.Counter()
    labels = collections.Counter()
    qcount = collections.Counter()
    for s in samples:
        qcount[len(s["questions"])] += 1
        for q in s["questions"].values():
            types[qtype(q)] += 1
            lb = q["label"]
            labels[str(lb) if not isinstance(lb, bool) else lb] += 1
    lines = [
        "  samples: %d" % len(samples),
        "  questions per sample: %s" % dict(sorted(qcount.items())),
        "  question types: %s" % dict(sorted(types.items())),
        "  top labels: %s" % labels.most_common(12),
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    report = []
    for name, (rows, ents, n_pairs, use_dir8, ood_rule) in SPLITS.items():
        seen = set()
        samples = build_split(rng, rows, ents, n_pairs, use_dir8, ood_rule, seen)
        path = out / ("%s.jsonl" % name)
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        report.append("[%s] -> %s\n%s" % (name, path.name, stats_for(samples)))

        # RQ2 ablation: same distribution, no contrastive constraint
        if name == "train":
            seen2 = set()
            rnd = []
            while len(rnd) < len(samples):
                s = G.make_single(rng, rows, ents, use_dir8=False)
                if s and hash(s["state"]) not in seen2:
                    seen2.add(hash(s["state"]))
                    rnd.append(s)
            p2 = out / "train_random.jsonl"
            with open(p2, "w") as f:
                for s in rnd:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            report.append("[train_random] -> %s\n%s"
                          % (p2.name, stats_for(rnd)))

    txt = "\n\n".join(report)
    (out / "stats.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
