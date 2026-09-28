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
# Purely a membership/reachability vocabulary: no spatial language here, so a
# stray "north of" can never leak in and make the queried pair derivable.
SYSTEMS = ["billing_svc", "auth_gateway", "vault_node", "report_db", "media_cdn",
           "policy_engine", "ledger_svc", "session_cache", "audit_svc", "batch_runner"]
A4_INTROS = [
    "The network inventory below records subnet membership:",
    "From the network inventory:",
    "Inventory extract:",
]
A4_LINE = [
    "- {a} is deployed inside the {b} subnet.",
    "- {a} runs in the {b} subnet.",
    "- {b} subnet hosts {a}.",
]
A4_Q = [
    "Are {a} and {b} in the same subnet?",
    "Is there a recorded shared subnet for {a} and {b}?",
]


def gen_a4(rng: random.Random, k_other: int = 3) -> Optional[dict]:
    """Ask about a pair with NO connecting membership fact. Other membership
    facts are present, so the state is not trivially empty, but no transitive
    path joins the queried pair. No derivation is possible -> the only correct
    behaviour is low confidence."""
    names = rng.sample(SYSTEMS, k_other + 2)
    subject, target, fillers = names[0], names[1], names[2:]
    subnets = ["alpha", "beta", "gamma", "delta"][:k_other]

    lines = [rng.choice(A4_INTROS)]
    # each filler is placed in one subnet; subject/target are in none of them,
    # and no line mentions both
    for name, sn in zip(fillers, subnets):
        lines.append(rng.choice(A4_LINE).format(a=name, b=sn))
    lines.append("- %s and %s do not appear in any subnet entry above."
                 % (subject, target))
    state = "\n".join(lines)
    q = {
        "type": "choice",
        "instructions": rng.choice(A4_Q).format(a=subject, b=target),
        "criteria": {
            "same_subnet": "A recorded entry places both in one subnet",
            "different_subnets": "A recorded entry places them in different subnets",
            "not_recorded": "No recorded entry settles the question",
        },
    }
    return {
        "state": state,
        "questions": {"q1": {**q, "label": "not_recorded",
                             "ideal_probs": {"not_recorded": 0.7,
                                             "same_subnet": 0.15,
                                             "different_subnets": 0.15}}},
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
