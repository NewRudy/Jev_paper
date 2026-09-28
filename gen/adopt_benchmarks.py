"""Adopt standard public benchmarks as evaluation/testbeds.

Why: an in-house synthetic grid is the weakest point of this study (a
reviewer's first attack). StepGame (ACL 2022) and CLUTRR (2019) are the
standard k-hop chain benchmarks with published LLM baselines, and
ProofWriter (2020) supplies the open-world "Unknown" epistemic state that our
A4 (evidence absence) type corresponds to.

This module downloads them from HuggingFace, converts each item into the typed
judgment question format (state + choice/noul question + label), and writes
Kev-compatible JSONL. Labels are taken verbatim from the source datasets — no
re-derivation — and the source split (train/test) is preserved so that
"train on k<=K, test on k>K" stays meaningful.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

OUT = Path("data")
HF = "datasets"


def _try_datasets(repo: str, config: str | None = None, split: str = "test"):
    from datasets import load_dataset
    kw = {"trust_remote_code": True}
    if config:
        kw["name"] = config
    return load_dataset(repo, split=split, **kw)


# ---------------------------------------------------------------- StepGame
# config "qaK" == K reasoning hops (verified: qa1 -> 1 sentence, qa5 -> 5)
STEP_LABELS = {
    "above": "north", "below": "south", "left": "west", "right": "east",
    "upper-right": "northeast", "upper-left": "northwest",
    "lower-right": "southeast", "lower-left": "southwest",
}
DIR8 = ["north", "south", "east", "west", "northeast", "northwest",
        "southeast", "southwest"]
DIR_DESCR = {
    "north": "toward the top of the map (smaller row numbers)",
    "south": "toward the bottom of the map (larger row numbers)",
    "east": "toward the right (larger column numbers)",
    "west": "toward the left (smaller column numbers)",
    "northeast": "both up and to the right",
    "northwest": "both up and to the left",
    "southeast": "both down and to the right",
    "southwest": "both down and to the left",
}
CONV = ("Directions use a map convention: north is toward the top of the map "
        "(smaller row numbers), south toward the bottom, east toward the right "
        "(larger column numbers), west toward the left.")


def convert_stepgame(limit_per_k: int = 100, max_k: int = 6):
    """StepGame (ACL 2022). Labels are taken verbatim from the source; we only
    re-express the question in typed-choice form."""
    from datasets import load_dataset
    ds = load_dataset("tasksource/stepgame", split="test")
    buckets = defaultdict(list)
    for row in ds:
        cfg = row.get("config") or ""
        if not cfg.startswith("qa"):
            continue
        k = int(cfg[2:])
        if k < 1 or k > max_k or len(buckets[k]) >= limit_per_k:
            continue
        gold = STEP_LABELS.get((row.get("label") or "").strip().lower())
        if gold is None:                      # 'overlap' has no clean mapping
            continue
        story = row.get("story") or ""
        q = (row.get("question") or "").strip()
        facts = [s.strip().rstrip(".") + "." for s in story.split(";") if s.strip()]
        state = "Here is a spatial description:\n" + "\n".join("- " + f for f in facts) + "\n" + CONV
        m = re.search(r"relation of the agent (\S+) to the agent (\S+)", q)
        if not m:
            continue
        a, b = m.group(1).rstrip("?"), m.group(2).rstrip("?")
        qq = {
            "type": "choice",
            "instructions": "Based only on the description above, in which of the "
                            "eight directions is %s from %s?" % (a, b),
            "criteria": {d: DIR_DESCR[d] for d in DIR8},
            "label": gold,
        }
        buckets[k].append({"state": state, "questions": {"q1": qq}})
    return buckets


# ---------------------------------------------------------------- CLUTRR
KINSHIP_NOTE = ("Kinship terms follow the usual family sense: a person's parent "
                "is their father or mother; a parent's parent is a grandparent; "
                "siblings share a parent; spouses are husband or wife.")


def convert_clutrr(limit_per_k: int = 100, max_k: int = 6):
    """CLUTRR (2019): k-hop kinship inference. A different composition algebra
    (transitive kinship over a family graph) from the spatial task, so it is
    used to test whether the depth-coverage effect is domain-general, NOT for
    the convolution-based remedy (which is undefined here)."""
    from datasets import load_dataset
    ds = load_dataset("tasksource/clutrr", split="test")   # full k=2..10 range
    names = ds.features["label"].names
    buckets = defaultdict(list)
    for row in ds:
        k = int(row.get("hops") or 0)
        if k < 1 or k > max_k or len(buckets[k]) >= limit_per_k:
            continue
        story = (row.get("story") or "").strip()
        query = (row.get("query") or "").strip()
        head, tail = row.get("head"), row.get("tail")
        if not story or not query or not head or not tail:
            continue
        gold = names[int(row["label"])]
        state = ("Background:\n" + story + "\n\n" + KINSHIP_NOTE)
        qq = {
            "type": "choice",
            "instructions": "Reasoning only over the background above, which kinship "
                            "term relates %s to %s?" % (tail, head),
            "criteria": {r: None for r in names},
            "label": gold,
        }
        buckets[k].append({"state": state, "questions": {"q1": qq}})
    return buckets


def write_buckets(name, buckets):
    n = 0
    for k, rows in sorted(buckets.items()):
        p = OUT / f"{name}_k{k}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += len(rows)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["stepgame", "clutrr", "both"])
    ap.add_argument("--per-k", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.which in ("stepgame", "both"):
        b = convert_stepgame(args.per_k)
        print("stepgame:", {k: len(v) for k, v in sorted(b.items())},
              "total:", write_buckets("sg", b))
    if args.which in ("clutrr", "both"):
        b = convert_clutrr(args.per_k)
        print("clutrr:", {k: len(v) for k, v in sorted(b.items())},
              "total:", write_buckets("cl", b))


if __name__ == "__main__":
    main()
