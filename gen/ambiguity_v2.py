"""Ambiguity typology v2 (A1 / A4 / A5) for calibration studies.

Replaces the weak A2/A3 drafts: near-tie distances and threshold boundaries
still have a DETERMINATE answer, so any "ideal 50/50" had to be stipulated
rather than derived. The three types below are cases where a human would
genuinely disagree, or where no answer is derivable at all.

A1  tie-break ambiguity : entity pair is diagonal; a 4-way cardinal question
    has two defensible answers. Ideal: 0.5/0.5 over the two axis directions.
A4  evidence-absence    : two entities with NO connecting fact. No derivation
    is possible. Ideal: low confidence on every option.
A5  conflicting evidence: two facts that contradict each other. Ideal: low
    confidence; any high-confidence answer is an error.

All labels are computed by the rule engine. A1 carries a dual-label training
form (gradient-equivalent to a 50/50 soft target under CE).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import generator as G

DIR4 = G.DIR4
DIR_DESCR = G.DIR_DESCR


# ------------------------------------------------------------------ A1
def gen_a1(rng: random.Random, rows_range=(6, 9), ents_range=(4, 6)) -> Optional[dict]:
    """Diagonal pair, 4-way cardinal question, two defensible answers."""
    for _ in range(120):
        rows, cols = rng.randint(*rows_range), rng.randint(*rows_range)
        n = rng.randint(*ents_range)
        world = G.sample_world(rng, rows, cols, n)
        names = list(world.entities)
        a, b = rng.sample(names, 2)
        ca, cb = world.entities[a], world.entities[b]
        dr, dc = ca.row - cb.row, ca.col - cb.col
        if dr == 0 or dc == 0 or abs(dr) != abs(dc):
            continue
        ns = "north" if dr < 0 else "south"
        ew = "west" if dc < 0 else "east"
        tpl = rng.choice([
            "In which of the four cardinal directions is {a} from {b}?",
            "If you stand at {b}, which cardinal direction is {a}?",
        ])
        q = {
            "type": "choice",
            "instructions": tpl.format(a=G.art(a), b=G.art(b)),
            "criteria": {d: DIR_DESCR[d] for d in DIR4},
        }
        return {
            "state": G.render(world, G.render_plan(rng, world)),
            "questions": {"q1": {**q, "label": ns, "ideal_probs": {ns: 0.5, ew: 0.5}}},
            "meta": {"atype": "A1", "ideal": "split"},
        }
    return None


# ------------------------------------------------------------------ A4
A4_INTROS = [
    "Here are the access records for several systems:",
    "The log below lists system access records:",
    "Recorded entries for the following systems:",
]
A4_LINE = [
    "- {a} is reachable from the {b} subnet.",
    "- {a} belongs to the {b} subnet.",
    "- the {a} subnet includes the {b}.",
]
A4_Q = [
    "Is {a} in the same subnet as {b}?",
    "Can traffic from {a} reach {b} directly?",
    "Are {a} and {b} on the same subnet?",
]


def gen_a4(rng: random.Random, k_other: int = 2) -> Optional[dict]:
    """Ask about a pair with NO connecting fact; other facts are present so
    the state is not trivially empty. No derivation is possible."""
    pool = list(G.LANDMARKS)
    rng.shuffle(pool)
    subject, target, others = pool[0], pool[1], pool[2:2 + k_other]
    lines = [rng.choice(A4_INTROS)]
    spares = [x for x in pool if x not in (subject, target)]
    for name in others:
        peer = rng.choice([x for x in spares if x != name])
        lines.append(rng.choice(A4_LINE).format(a=G.art(name), b=G.art(peer)))
    # noise relations among fillers only - never the queried pair,
    # so no derivation for (subject, target) exists
    for j in range(len(others) - 1):
        x, y = others[j], others[j + 1]
        lines.append('- the %s is %s of the %s.' % (x, rng.choice(DIR4), y))
        x, y = rng.sample([o for o in others] * 2, 2) if len(others) >= 2 else (subject, subject)
        if x == y:
            continue
        d = rng.choice(DIR4)
        lines.append(f"- the {x} is {d} of the {y}.")
    lines.append("- No other relationship between these systems is recorded.")
    state = "\n".join(lines)
    q = {
        "type": "choice",
        "instructions": rng.choice(A4_Q).format(a=G.art(subject), b=G.art(target)),
        "criteria": {
            "yes_connected": "A recorded fact establishes the relationship",
            "no_not_connected": "No recorded fact establishes it",
            "insufficient_information": "The records do not settle the question",
        },
    }
    return {
        "state": state,
        "questions": {"q1": {**q, "label": "insufficient_information",
                             "ideal_probs": {"insufficient_information": 0.7,
                                             "yes_connected": 0.15,
                                             "no_not_connected": 0.15}}},
        "meta": {"atype": "A4", "ideal": "low_conf"},
    }


# ------------------------------------------------------------------ A5
A5_INTROS = [
    "The audit log contains the following entries:",
    "Records from the change-management system:",
    "The following entries were logged:",
]


def gen_a5(rng: random.Random) -> Optional[dict]:
    """Two contradictory facts about the same pair."""
    pool = list(G.LANDMARKS)
    a, b = rng.sample(pool, 2)
    c = rng.choice([x for x in pool if x not in (a, b)])
    axis = rng.choice(DIR4)
    opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}[axis]
    facts = [
        rng.choice([
            f"- the {a} is {axis} of the {b}.",
            f"- the {b} has the {a} to its {axis}.",
        ]),
        rng.choice([
            f"- the {a} is {opposite} of the {b}.",
            f"- the {b} has the {a} to its {opposite}.",
        ]),
    ]
    rng.shuffle(facts)
    state = "\n".join([rng.choice(A5_INTROS)] + facts)
    q = {
        "type": "noul",
        "instructions": f"Based on the log, is the {a} {axis} of the {b}?",
        "criteria": {"true": "A recorded entry supports this", "false": "A recorded entry contradicts this"},
    }
    return {
        "state": state,
        "questions": {"q1": {**q, "label": False, "ideal_probs": {"True": 0.5, "False": 0.5}}},
        "meta": {"atype": "A5", "ideal": "split"},
    }


# ------------------------------------------------------------------ assembly
def to_dual_label(sample: dict) -> Optional[List[dict]]:
    """A1/A5 training form: same state, both complementary labels, so CE
    gradient equals a 50/50 soft target."""
    q = sample["questions"]["q1"]
    ideal = q.get("ideal_probs") or {}
    if q["type"] == "choice" and len(ideal) == 2:
        labs = list(ideal.keys())
    elif q["type"] == "noul":
        labs = [True, False]
    else:
        return None
    out = []
    for lab in labs:
        qq = {k: v for k, v in q.items() if k != "ideal_probs"}
        qq["label"] = lab
        out.append({"state": sample["state"], "questions": {"q1": qq}})
    return out
