"""Probabilistic Modular Composition (PMC) Evaluator for Chain Tasks.

Decomposes a k-hop chain question into k atomic 1-hop judgment queries,
performs exact discrete 2D spatial convolution of probability distributions,
and compares System 1 (Direct) vs System 1.5 (PMC Decomposed) across:
- Accuracy
- Composite Confidence & Calibration
- Cost-Accuracy Pareto Frontier (Confidence-based Routing)
"""
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DIR8_VEC = {
    "north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1),
    "northeast": (-1, 1), "northwest": (-1, -1),
    "southeast": (1, 1), "southwest": (1, -1),
}

DIR8 = list(DIR8_VEC.keys())

FACT_PATTERNS = [
    (re.compile(r"- the (\w+) is (north|south|east|west|northeast|northwest|southeast|southwest) of the (\w+)\."),
     lambda m: (m.group(1), m.group(2), m.group(3))),          # a is d of b
    (re.compile(r"- the (\w+) lies to the (north|south|east|west|northeast|northwest|southeast|southwest) of the (\w+)\."),
     lambda m: (m.group(1), m.group(2), m.group(3))),          # a lies d of b
    (re.compile(r"- the (\w+) has the (\w+) to its (north|south|east|west|northeast|northwest|southeast|southwest)\."),
     lambda m: (m.group(2), m.group(3), m.group(1))),          # b has a to its d
]


def parse_facts(text: str) -> Optional[List[Tuple[str, str, Tuple[int, int]]]]:
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
        edges.append((a, b, DIR8_VEC[d]))
    return edges or None


def find_path(edges: List[Tuple[str, str, Tuple[int, int]]], start: str, end: str) -> Optional[List[str]]:
    """BFS to find sequence of nodes from start to end."""
    adj = collections.defaultdict(list)
    for a, b, _ in edges:
        adj[b].append(a)  # a relative to b -> edge from b to a
        adj[a].append(b)  # bidirectional search
    
    seen = {start}
    q = collections.deque([[start]])
    while q:
        path = q.popleft()
        node = path[-1]
        if node == end:
            return path
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(path + [nxt])
    return None


def parse_query_endpoints(instructions: str) -> Optional[Tuple[str, str, bool]]:
    """Returns (target_x, reference_y, is_noul). Target is asked relative to reference."""
    ins = instructions.lower()
    m = re.search(r"is the (\w+) from the (\w+)", ins) or \
        re.search(r"where does the (\w+) lie relative to the (\w+)", ins) or \
        re.search(r"does the (\w+) lie relative to the (\w+)", ins)
    noul = False
    if not m:
        m = re.search(r"is the (\w+) north of the (\w+)", ins) or \
            re.search(r"does the (\w+) lie to the north of the (\w+)", ins)
        noul = True
    if not m:
        return None
    return m.group(1), m.group(2), noul


def decompose_sample(sample: dict) -> Optional[List[dict]]:
    """Extract atomic 1-hop sub-questions along the path from reference y to target x."""
    q1 = sample["questions"]["q1"]
    edges = parse_facts(sample["state"])
    if not edges:
        return None
    endpoints = parse_query_endpoints(q1["instructions"])
    if not endpoints:
        return None
    x, y, _ = endpoints
    path = find_path(edges, start=y, end=x)
    if not path or len(path) < 2:
        return None
    
    sub_questions = []
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        # Query: in which direction is dst from src?
        sub_q = {
            "src": src,
            "dst": dst,
            "instructions": f"Based on the facts above, in which of the eight directions is the {dst} from the {src}?",
            "type": "choice"
        }
        sub_questions.append(sub_q)
    return sub_questions


def convolve_discrete_distributions(hop_probs: List[Dict[str, float]]) -> Dict[Tuple[int, int], float]:
    """Exact discrete 2D spatial convolution of direction probability distributions."""
    curr_dist: Dict[Tuple[int, int], float] = {(0, 0): 1.0}
    for hop in hop_probs:
        next_dist: Dict[Tuple[int, int], float] = collections.defaultdict(float)
        for (r, c), p in curr_dist.items():
            for d_name, p_d in hop.items():
                if d_name in DIR8_VEC:
                    dr, dc = DIR8_VEC[d_name]
                    next_dist[(r + dr, c + dc)] += p * p_d
        curr_dist = next_dist
    return curr_dist


def marginalize_to_choice(disp_dist: Dict[Tuple[int, int], float]) -> Tuple[str, float, float]:
    """Marginalize 2D displacement distribution back into 8 direction choices.
    Returns (pred_label, composite_confidence, entropy).
    """
    dir_probs = collections.defaultdict(float)
    for (r, c), p in disp_dist.items():
        if r == 0 and c == 0:
            continue
        ns = "north" if r < 0 else "south" if r > 0 else ""
        ew = "west" if c < 0 else "east" if c > 0 else ""
        d = ns + ew if ns and ew else (ns or ew)
        if d in DIR8_VEC:
            dir_probs[d] += p
    
    if not dir_probs:
        return "none", 0.0, 0.0
    
    # Normalize
    total = sum(dir_probs.values())
    if total > 0:
        dir_probs = {k: v / total for k, v in dir_probs.items()}
    
    best_dir, conf = max(dir_probs.items(), key=lambda kv: kv[1])
    entropy = -sum(v * math.log(v + 1e-12) for v in dir_probs.values() if v > 0)
    return best_dir, conf, entropy


def marginalize_to_noul(disp_dist: Dict[Tuple[int, int], float]) -> Tuple[bool, float, float]:
    """Marginalize 2D displacement distribution to boolean north query."""
    p_yes = sum(p for (r, c), p in disp_dist.items() if r < 0)
    p_no = 1.0 - p_yes
    pred = p_yes >= 0.5
    conf = max(p_yes, p_no)
    entropy = -(p_yes * math.log(p_yes + 1e-12) + p_no * math.log(p_no + 1e-12))
    return pred, conf, entropy


def simulate_routing_pareto(direct_results: List[dict], decompose_results: List[dict], thresholds: List[float]):
    """Simulate confidence-guided routing across thresholds alpha."""
    pareto_points = []
    n = len(direct_results)
    for alpha in thresholds:
        correct = 0
        total_forwards = 0
        decomposed_count = 0
        for d, m in zip(direct_results, decompose_results):
            if d["confidence"] >= alpha:
                # Direct route
                if d["pred"] == d["label"]:
                    correct += 1
                total_forwards += 1
            else:
                # Decomposed route
                decomposed_count += 1
                if m["pred"] == d["label"]:
                    correct += 1
                total_forwards += m["k"]
        
        acc = correct / n if n > 0 else 0
        avg_forwards = total_forwards / n if n > 0 else 0
        pareto_points.append({
            "alpha": round(alpha, 2),
            "acc": round(acc, 4),
            "avg_forwards": round(avg_forwards, 2),
            "pct_decomposed": round(decomposed_count / n * 100, 1),
        })
    return pareto_points


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate or decompose chain test data")
    ap.add_argument("--test_file", type=str, default="data/chain_test_k3.jsonl")
    args = ap.parse_args()
    
    p = Path(args.test_file)
    if not p.exists():
        print(f"File {p} does not exist.")
        exit(1)
    
    rows = [json.loads(line) for line in open(p)]
    decomposed_count = 0
    for r in rows:
        subs = decompose_sample(r)
        if subs:
            decomposed_count += 1
    
    print(f"Successfully decomposed {decomposed_count}/{len(rows)} samples in {p.name}")
