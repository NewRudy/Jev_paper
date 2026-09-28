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
# "Specification gap", measured as a controlled pair.
#
# A diagonal pair asked with a four-way cardinal question has two defensible
# answers, so the label is only determined once a convention is fixed. We do
# NOT assert an "ideal distribution" (that would be circular). Instead each
# item is emitted twice:
#   A1s  the tie-break convention is WRITTEN INTO the state
#   A1u  the same item with the convention omitted
# and we score both against the same convention label. The comparison isolates
# one thing: whether stating the convention changes the model's confidence. If
# confidence is unchanged, the model was never reading the spec.
#
# This is the literature's standard move (StepGame/CLUTRR declare a convention
# and make the label deterministic); the paired design is the contribution.
DIR4 = G.DIR4
DIR_DESCR = G.DIR_DESCR
CONVENTION = (
    "Tie-break rule for diagonal placements: when an entity lies diagonally, "
    "report the north-south component (north or south), not the east-west one."
)


def _a1_pair(rng, rows_range=(6, 9), ents_range=(4, 6)):
    for _ in range(200):
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
        plan = G.render_plan(rng, world)
        body = G.render(world, plan)
        q = {
            "type": "choice",
            "instructions": "In which of the four cardinal directions is %s from %s?"
                            % (G.art(a), G.art(b)),
            "criteria": {d: DIR_DESCR[d] for d in DIR4},
        }
        stated = {"state": body + "\n" + CONVENTION,
                  "questions": {"q1": {**q, "label": ns}}}
        unstated = {"state": body,
                    "questions": {"q1": {**q, "label": ns}}}
        return stated, unstated, {"a": a, "b": b, "ns": ns, "ew": ew}
    return None, None, None


def gen_a1_stated(rng, **kw):
    s, _u, meta = _a1_pair(rng, **kw)
    if s is None:
        return None
    return {**s, "meta": {"atype": "A1s", "spec": "stated", **meta}}


def gen_a1_unstated(rng, **kw):
    _s, u, meta = _a1_pair(rng, **kw)
    if u is None:
        return None
    return {**u, "meta": {"atype": "A1u", "spec": "unstated", **meta}}


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
        "type": "choice",
        "instructions": "What does the log establish about the %s's position "
                        "relative to the %s?" % (a, b),
        "criteria": {
            "%s of the %s" % (axis, b): "One entry supports this reading",
            "%s of the %s" % (opposite, b): "One entry supports this reading",
            "cannot_determine": "The entries contradict each other",
        },
    }
    return {
        "state": state,
        "questions": {"q1": {**q, "label": "cannot_determine"}},
        "meta": {"atype": "A5"},
    }


# ------------------------------------------------------------------ assembly
def to_dual_label(sample: dict) -> Optional[List[dict]]:
    """Legacy dual-label form. Under the v3 redesign no type needs it (A5 is a
    single deterministic label, A1s is convention-deterministic); kept so that
    older callers keep working."""
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
