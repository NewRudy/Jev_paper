"""Grid-world spatial-relation dataset generator for Kev / System One fine-tuning.

Pure stdlib. Labels are computed by a rule engine (zero noise). Contrastive
pairs differ by moving ONE entity 1-2 grid steps, flipping the main question's
label; filler questions never reference the moved entity (asserted).

Coordinate convention (stated in every state text):
  row 1 = northernmost row, column 1 = westernmost column,
  distance = Manhattan grid steps.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

# ---------------------------------------------------------------- entity pools
LANDMARKS = [
    "tower", "farmhouse", "windmill", "barn", "warehouse", "cabin",
    "depot", "silo", "chapel", "mill", "gate", "workshop",
]
WATERS = ["lake", "pond", "spring", "marsh", "reservoir", "creek"]

DIR4 = ["north", "south", "east", "west"]
DIR8 = DIR4 + ["northeast", "northwest", "southeast", "southwest"]


@dataclass(frozen=True)
class Cell:
    row: int
    col: int


class GridWorld:
    def __init__(self, rows: int, cols: int, entities: dict[str, Cell]):
        self.rows = rows
        self.cols = cols
        self.entities = dict(entities)

    # -- geometry ------------------------------------------------------------
    def in_bounds(self, c: Cell) -> bool:
        return 1 <= c.row <= self.rows and 1 <= c.col <= self.cols

    def occupied(self, c: Cell) -> bool:
        return c in self.entities.values()

    def step_dist(self, a: str, b: str) -> int:
        ca, cb = self.entities[a], self.entities[b]
        return abs(ca.row - cb.row) + abs(ca.col - cb.col)

    def direction4(self, a: str, b: str) -> str | None:
        """Direction of `a` relative to `b` using the dominant axis, or None."""
        ca, cb = self.entities[a], self.entities[b]
        dr, dc = ca.row - cb.row, ca.col - cb.col
        if dr == 0 and dc == 0 or abs(dr) == abs(dc):
            return None
        if abs(dr) > abs(dc):
            return "north" if dr < 0 else "south"
        return "west" if dc < 0 else "east"

    def direction8(self, a: str, b: str) -> str | None:
        ca, cb = self.entities[a], self.entities[b]
        dr, dc = ca.row - cb.row, ca.col - cb.col
        if dr == 0 and dc == 0:
            return None
        ns = "north" if dr < 0 else "south" if dr > 0 else ""
        ew = "west" if dc < 0 else "east" if dc > 0 else ""
        return ns + ew if ns and ew else (ns or ew)

    def is_axis(self, a: str, b: str, axis: str) -> bool:
        """Strict test, e.g. axis='north' means a.row < b.row."""
        ca, cb = self.entities[a], self.entities[b]
        return {
            "north": ca.row < cb.row, "south": ca.row > cb.row,
            "west": ca.col < cb.col, "east": ca.col > cb.col,
        }[axis]

    def strictly_between(self, a: str, b: str, c: str) -> bool:
        """c strictly inside the segment a-b on a shared row/col/diagonal."""
        pa, pb, pc = (self.entities[x] for x in (a, b, c))
        if pc == pa or pc == pb:
            return False
        same_row = pa.row == pb.row == pc.row
        same_col = pa.col == pb.col == pc.col
        diag = (
            abs(pa.row - pb.row) == abs(pa.col - pb.col)
            and abs(pa.row - pc.row) == abs(pa.col - pc.col)
            and (pa.row - pc.row) * (pc.row - pb.row) > 0
        )
        if same_row:
            return (pa.col - pc.col) * (pc.col - pb.col) > 0
        if same_col:
            return (pa.row - pc.row) * (pc.row - pb.row) > 0
        return diag

    # -- manipulation ---------------------------------------------------------
    def move(self, name: str, to: Cell) -> "GridWorld | None":
        if not self.in_bounds(to) or self.occupied(to):
            return None
        ents = dict(self.entities)
        ents[name] = to
        return GridWorld(self.rows, self.cols, ents)

    def neighbors(self, name: str, max_step: int = 2):
        """Candidate destination cells within Chebyshev distance <= max_step."""
        c = self.entities[name]
        out = []
        for dr in range(-max_step, max_step + 1):
            for dc in range(-max_step, max_step + 1):
                if dr == 0 and dc == 0:
                    continue
                out.append(Cell(c.row + dr, c.col + dc))
        return out

    def signature(self) -> tuple:
        return (self.rows, self.cols, tuple(sorted(self.entities.items())))


# ---------------------------------------------------------------- score rules
def make_signal_rule(target: str) -> dict:
    """Distance-to-target signal tiers (closer = stronger)."""
    tiers = [
        "strong: at most 2 grid steps from the %s" % target,
        "fair: 3 to 5 grid steps from the %s" % target,
        "weak: more than 5 grid steps from the %s" % target,
    ]
    return {
        "target": target,
        "tiers": tiers,
        "compute": lambda w, p: min(2, 1 if w.step_dist(p, target) <= 5 else 2)
        if w.step_dist(p, target) > 2 else 0,
        "instruction": "How strong is the signal at the %s?" % target,
        "question": "How strong is the signal at {p}?",
    }


def make_water_rule(target: str) -> dict:
    """Distance-to-water access tiers (closer = better access)."""
    tiers = [
        "good: at most 3 grid steps from the %s" % target,
        "moderate: 4 to 6 grid steps from the %s" % target,
        "poor: more than 6 grid steps from the %s" % target,
    ]
    return {
        "target": target,
        "tiers": tiers,
        "compute": lambda w, p: min(2, 1 if w.step_dist(p, target) <= 6 else 2)
        if w.step_dist(p, target) > 3 else 0,
        "question": "How good is the water access at {p}?",
    }


# ---------------------------------------------------------------- question specs
@dataclass
class QSpec:
    qid: str
    q: dict            # Kev question dict without label
    label: object      # choice: option name; noul: bool; score: int
    entities: tuple    # entity names the question references
    compute: object    # callable(world) -> label, for perturbation re-checks


# ---- instruction template pools ---------------------------------------------
TPL_DIR_CHOICE = [
    "Looking at the map, in which direction is {a} from {b}?",
    "In which of the four cardinal directions does {a} lie from {b}?",
    "If you stand at {b}, which cardinal direction is {a}?",
]
TPL_DIR8_CHOICE = [
    "In which of the eight directions is {a} from {b}?",
    "If you stand at {b}, which of the eight compass directions is {a}?",
]
TPL_DIR_NOUL = {
    "north": ["Is {a} north of {b}?", "Does {a} lie to the north of {b}?"],
    "south": ["Is {a} south of {b}?", "Does {a} lie to the south of {b}?"],
    "east": ["Is {a} east of {b}?", "Does {a} lie to the east of {b}?"],
    "west": ["Is {a} west of {b}?", "Does {a} lie to the west of {b}?"],
}
TPL_CMP_CHOICE = [
    "Which is closer to {a}: {b} or {c}?",
    "Measured in grid steps, which of {b} or {c} is nearer to {a}?",
    "Standing at {a}, which one is closer, {b} or {c}?",
]
TPL_THR_NOUL = [
    "Is {a} within {k} grid steps of {b}?",
    "Is the step distance between {a} and {b} at most {k}?",
    "Does it take at most {k} steps to walk from {a} to {b}?",
]
TPL_BET_NOUL = [
    "Is {c} strictly between {a} and {b} on a straight line?",
    "Do {a}, {c} and {b} lie on one straight line with {c} in the middle?",
    "Is {c} positioned on the straight segment joining {a} and {b}?",
]
DIR_DESCR = {
    "north": "toward the top edge of the map (smaller row numbers)",
    "south": "toward the bottom edge of the map (larger row numbers)",
    "west": "toward the left edge of the map (smaller column numbers)",
    "east": "toward the right edge of the map (larger column numbers)",
    "northeast": "both up and to the right",
    "northwest": "both up and to the left",
    "southeast": "both down and to the right",
    "southwest": "both down and to the left",
}


def art(name: str) -> str:
    return "the %s" % name


# ---------------------------------------------------------------- world sampling
def sample_world(rng, rows: int, cols: int, n_entities: int) -> GridWorld:
    pool = rng.sample(LANDMARKS + WATERS, n_entities)
    cells = rng.sample(
        [Cell(r, c) for r in range(1, rows + 1) for c in range(1, cols + 1)],
        n_entities,
    )
    return GridWorld(rows, cols, dict(zip(pool, cells)))


# ---------------------------------------------------------------- question sampling
def sample_main_question(rng, world: GridWorld, qid: str, use_dir8: bool = False,
                         rule=None) -> QSpec | None:
    names = list(world.entities)
    kind = rng.choice(
        ["dir8_choice"] if use_dir8 else
        ["dir_choice", "dir_choice", "dir_noul", "cmp_choice", "cmp_choice",
         "thr_noul", "thr_noul", "bet_noul", "score"]
    )

    if kind in ("dir_choice", "dir8_choice"):
        dirs = DIR4 if kind == "dir_choice" else DIR8
        a, b = rng.sample(names, 2)
        d = (world.direction4(a, b) if kind == "dir_choice"
             else world.direction8(a, b))
        if d is None:
            return None
        tpl = rng.choice(TPL_DIR_CHOICE if kind == "dir_choice" else TPL_DIR8_CHOICE)
        q = {
            "type": "choice",
            "instructions": tpl.format(a=art(a), b=art(b)),
            "criteria": {x: DIR_DESCR[x] for x in dirs},
        }
        return QSpec(qid, q, d, (a, b),
                     lambda w, a=a, b=b, dirs=dirs:
                     (w.direction4(a, b) if len(dirs) == 4 else w.direction8(a, b)))

    if kind == "dir_noul":
        axis = rng.choice(DIR4)
        a, b = rng.sample(names, 2)
        tpl = rng.choice(TPL_DIR_NOUL[axis])
        q = {
            "type": "noul",
            "instructions": tpl.format(a=art(a), b=art(b)),
            "criteria": {
                "true": "%s satisfies the strict direction test" % art(a),
                "false": "it does not (equal position or opposite direction)",
            },
        }
        return QSpec(qid, q, world.is_axis(a, b, axis), (a, b),
                     lambda w, a=a, b=b, axis=axis: w.is_axis(a, b, axis))

    if kind == "cmp_choice":
        if len(names) < 3:
            return None
        a, b, c = rng.sample(names, 3)
        if world.step_dist(a, b) == world.step_dist(a, c):
            return None
        winner = b if world.step_dist(a, b) < world.step_dist(a, c) else c
        tpl = rng.choice(TPL_CMP_CHOICE)
        q = {
            "type": "choice",
            "instructions": tpl.format(a=art(a), b=art(b), c=art(c)),
            "criteria": {b: "distance in grid steps to %s is smaller" % art(b),
                         c: "distance in grid steps to %s is smaller" % art(c)},
        }
        return QSpec(qid, q, winner, (a, b, c),
                     lambda w, a=a, b=b, c=c: (
                         b if w.step_dist(a, b) < w.step_dist(a, c)
                         else None if w.step_dist(a, b) == w.step_dist(a, c)
                         else c))

    if kind == "thr_noul":
        a, b = rng.sample(names, 2)
        d = world.step_dist(a, b)
        if d <= 2:          # need room to flip by crossing the threshold
            return None
        # k=d -> label true; k<d-? -> false; keep true/false roughly balanced
        k = rng.choice([d, d, d - 1, d - 1, d - 2])
        tpl = rng.choice(TPL_THR_NOUL)
        q = {
            "type": "noul",
            "instructions": tpl.format(a=art(a), b=art(b), k=k),
            "criteria": {
                "true": "the step distance is at most %d" % k,
                "false": "the step distance is greater than %d" % k,
            },
        }
        return QSpec(qid, q, d <= k, (a, b),
                     lambda w, a=a, b=b, k=k: w.step_dist(a, b) <= k)

    if kind == "bet_noul":
        if len(names) < 3:
            return None
        a, b = rng.sample(names, 2)
        c = rng.choice([x for x in names if x not in (a, b)])
        tpl = rng.choice(TPL_BET_NOUL)
        q = {
            "type": "noul",
            "instructions": tpl.format(a=art(a), b=art(b), c=art(c)),
            "criteria": {
                "true": "all three share one straight line and %s is strictly "
                        "inside the segment" % art(c),
                "false": "they are not collinear, or %s is not strictly inside"
                         % art(c),
            },
        }
        return QSpec(qid, q, world.strictly_between(a, b, c), (a, b, c),
                     lambda w, a=a, b=b, c=c: w.strictly_between(a, b, c))

    if kind == "score":
        if rule is None:
            targets = [n for n in names if n in WATERS]
            if not targets:
                return None
            rule = make_water_rule(rng.choice(targets))
        others = [n for n in names if n != rule["target"]]
        if not others:
            return None
        p = rng.choice(others)
        q = {
            "type": "score",
            "instructions": rule["question"].format(p=art(p)),
            "criteria": rule["tiers"],
        }
        return QSpec(qid, q, rule["compute"](world, p), (p, rule["target"]),
                     lambda w, rule=rule, p=p: rule["compute"](w, p))

    return None


def sample_filler(rng, world: GridWorld, avoid: set[str], qid: str,
                  use_dir8=False, rule=None) -> QSpec | None:
    for _ in range(12):
        q = sample_main_question(rng, world, qid, use_dir8=use_dir8, rule=rule)
        if q and not (set(q.entities) & avoid):
            return q
    return None


# ---------------------------------------------------------------- perturbation
def perturb_flip(rng, world: GridWorld, main: QSpec, max_step: int = 2):
    """Move ONE referenced entity <=2 steps so that main's label flips."""
    movers = list(main.entities)
    rng.shuffle(movers)
    for mover in movers:
        cands = world.neighbors(mover, max_step)
        rng.shuffle(cands)
        for to in cands:
            w2 = world.move(mover, to)
            if w2 is None:
                continue
            try:
                new_label = main.compute(w2)
            except KeyError:
                continue
            if new_label != main.label and new_label is not None:
                return w2, mover
    return None, None


# ---------------------------------------------------------------- rendering
def tweak_collinear(rng, world: GridWorld, prob: float = 0.5) -> GridWorld:
    """With probability `prob`, move one entity onto the straight segment
    between two others so that `strictly_between` true cases exist."""
    if rng.random() > prob:
        return world
    names = list(world.entities)
    for _ in range(10):
        a, b, c = rng.sample(names, 3)
        pa, pb = world.entities[a], world.entities[b]
        inner = []
        for r in range(1, world.rows + 1):
            for col in range(1, world.cols + 1):
                cell = Cell(r, col)
                if cell in (pa, pb) or world.occupied(cell):
                    continue
                tmp = GridWorld(world.rows, world.cols,
                                {**world.entities, c: cell})
                if tmp.strictly_between(a, b, c):
                    inner.append(cell)
        if inner:
            ents = dict(world.entities)
            ents[c] = rng.choice(inner)
            return GridWorld(world.rows, world.cols, ents)
    return world


INTROS = [
    "Here is a grid map with {r} rows and {c} columns. Row 1 is the "
    "northernmost row and column 1 is the westernmost column. Distances are "
    "measured in grid steps (up, down, left or right moves between cells).",
    "The map below is a {r}x{c} grid. Row 1 lies at the northern edge and "
    "column 1 at the western edge. The distance between two cells is the "
    "number of up/down/left/right steps.",
    "Consider a {r}-row by {c}-column grid. Rows are numbered from north "
    "(row 1) to south, and columns from west (column 1) to east. Step "
    "distance counts up/down/left/right moves.",
]
LINE_TPL = [
    "- {name} is at row {r}, column {c}.",
    "- {name} stands at cell ({r}, {c}).",
    "- {name}: row {r}, column {c}.",
]


def render(world: GridWorld, plan: dict) -> str:
    lines = [INTROS[plan["intro"]].format(r=world.rows, c=world.cols)]
    for name, cell in sorted(world.entities.items()):
        tpl = LINE_TPL[plan["lines"][name]]
        lines.append(tpl.format(name=art(name), r=cell.row, c=cell.col))
    return "\n".join(lines)


def render_plan(rng, world: GridWorld) -> dict:
    return {
        "intro": rng.randrange(len(INTROS)),
        "lines": {n: rng.randrange(len(LINE_TPL)) for n in world.entities},
    }


# ---------------------------------------------------------------- pair assembly
def make_pair(rng, rows_range, ents_range, use_dir8=False, ood_rule=False):
    """Return (sample_a, sample_b) or None; the two states differ in ONE
    entity's position, flipping the main question's label."""
    for _ in range(60):
        rows = rng.randint(*rows_range)
        cols = rng.randint(*rows_range)
        n = rng.randint(*ents_range)
        world = sample_world(rng, rows, cols, n)
        world = tweak_collinear(rng, world)
        rule = None
        if ood_rule:
            targets = [x for x in world.entities if x in ("warehouse", "depot", "silo", "gate")]
            if targets:
                rule = make_signal_rule(rng.choice(targets))
        main = sample_main_question(rng, world, "q1", use_dir8=use_dir8,
                                    rule=rule)
        if main is None:
            continue
        w2, mover = perturb_flip(rng, world, main)
        if w2 is None:
            continue
        fillers = []
        qid_n = 2
        for _ in range(4):
            if len(fillers) >= 2:
                break
            f = sample_filler(rng, world, {mover}, "q%d" % qid_n,
                              use_dir8=False, rule=None)
            if f and f.qid not in [x.qid for x in fillers]:
                # label must be identical in both worlds (no reference to mover)
                assert f.compute(world) == f.compute(w2), "filler leaked mover"
                fillers.append(f)
                qid_n += 1
        plan = render_plan(rng, world)
        out = []
        for w in (world, w2):
            m = replace(main, label=main.compute(w))
            fl = [replace(f, label=f.compute(w)) for f in fillers]
            qs = {}
            for spec in [m] + fl:
                qq = dict(spec.q)
                qq["label"] = spec.label
                qs[spec.qid] = qq
            out.append({"state": render(w, plan), "questions": qs})
        return out
    return None


def make_single(rng, rows_range, ents_range, use_dir8=False):
    """Random (non-contrastive) sample for the RQ2 ablation."""
    for _ in range(60):
        rows = rng.randint(*rows_range)
        cols = rng.randint(*rows_range)
        n = rng.randint(*ents_range)
        world = sample_world(rng, rows, cols, n)
        plan = render_plan(rng, world)
        qs = {}
        qid_n = 1
        for _ in range(6):
            if len(qs) >= rng.randint(3, 4):
                break
            q = sample_main_question(rng, world, "q%d" % qid_n,
                                     use_dir8=use_dir8)
            if q and q.qid not in qs:
                qq = dict(q.q)
                qq["label"] = q.label
                qs[q.qid] = qq
                qid_n += 1
        if len(qs) >= 2:
            return {"state": render(world, plan), "questions": qs}
    return None


# ================================================================ v2: chains
DIR8_VEC = {
    "north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1),
    "northeast": (-1, 1), "northwest": (-1, -1),
    "southeast": (1, 1), "southwest": (1, -1),
}
CHAIN_INTROS = [
    "Consider the following spatial facts about several places:",
    "You are given these facts about where places lie relative to each other:",
    "Some spatial facts, stated one per line:",
]
FACT_TPL = [
    "- {a} is {d} of {b}.",
    "- {a} lies to the {d} of {b}.",
    "- {b} has {a} to its {d}.",
]
CHAIN_Q_TPL = [
    "Based on the facts above, in which of the eight directions is {x} from {y}?",
    "Combining the facts, where does {x} lie relative to {y}?",
]
CHAIN_NOUL_TPL = [
    "Based on the facts above, is {x} north of {y}?",
    "Combining the facts, does {x} lie to the north of {y}?",
]


def vec_dir8(v):
    """Vector back to an 8-direction name, or None if not one."""
    for name, vec in DIR8_VEC.items():
        if v == vec:
            return name
    return None


def gen_chain(rng, k: int, qid: str = "q1", noul: bool = False):
    """Relation-fact chain of length k: E0 -e1- E1 -e2- ... -ek- Ek.
    Question asks Ek relative to E0 (requires composing k relations)."""
    if k + 1 > len(LANDMARKS) + len(WATERS):
        return None
    for _ in range(80):
        names = rng.sample(LANDMARKS + WATERS, k + 1)
        vecs = []
        for _i in range(k):
            a, b = names[_i], names[_i + 1]
            # direction of names[i+1] relative to names[i]
            d = rng.choice(DIR8)
            vecs.append(DIR8_VEC[d])
        total = (sum(v[0] for v in vecs), sum(v[1] for v in vecs))
        if noul:
            label = total[0] < 0          # strictly north
            q = dict(type="noul",
                     instructions=rng.choice(CHAIN_NOUL_TPL).format(
                         x=art(names[-1]), y=art(names[0])))
        else:
            label = vec_dir8(total)
            if label is None:             # sum not an 8-direction; retry
                continue
            q = dict(type="choice",
                     instructions=rng.choice(CHAIN_Q_TPL).format(
                         x=art(names[-1]), y=art(names[0])),
                     criteria={d: DIR_DESCR[d] for d in DIR8})
        # render: shuffled fact sentences; edge d = direction of names[i+1]
        # relative to names[i], so the SUBJECT is names[i+1]
        facts = []
        for i in range(k):
            a, b = names[i + 1], names[i]
            d = DIR8_VEC_REV(vecs[i])
            facts.append((rng.choice(FACT_TPL), a, b, d))
        rng.shuffle(facts)
        intro = rng.choice(CHAIN_INTROS)
        lines = [intro] + [t.format(a=art(a), b=art(b), d=d) for (t, a, b, d) in facts]
        state = "\n".join(lines)
        return {"state": state,
                "questions": {qid: {**q, "label": label}},
                "meta": {"k": k, "noul": noul}}
    return None


def DIR8_VEC_REV(v):
    for name, vec in DIR8_VEC.items():
        if vec == v:
            return name
    return None


# ============================================================ v2: ambiguity
AMBIG_Q_TPL = [
    "In which of the four cardinal directions is {a} from {b}?",
    "If you stand at {b}, which cardinal direction is {a}?",
]


def gen_ambiguous(rng, rows_range=(6, 9)):
    """Diagonal pairs (|dr|==|dc|): the 4-way question is inherently ambiguous;
    the calibrated ideal is ~0.5/0.5 over the two axis directions."""
    for _ in range(80):
        rows = rng.randint(*rows_range)
        cols = rng.randint(*rows_range)
        n = rng.randint(4, 6)
        world = sample_world(rng, rows, cols, n)
        names = list(world.entities)
        a, b = rng.sample(names, 2)
        ca, cb = world.entities[a], world.entities[b]
        dr, dc = ca.row - cb.row, ca.col - cb.col
        if dr == 0 or dc == 0 or abs(dr) != abs(dc):
            continue
        ns = "north" if dr < 0 else "south"
        ew = "west" if dc < 0 else "east"
        q = dict(type="choice",
                 instructions=rng.choice(AMBIG_Q_TPL).format(a=art(a), b=art(b)),
                 criteria={d: DIR_DESCR[d] for d in DIR4})
        # label: by our tie-break convention (north/south wins); ideal dist 0.5/0.5
        return {"state": render(world, render_plan(rng, world)),
                "questions": {"q1": {**q, "label": ns,
                                     "ideal_probs": {ns: 0.5, ew: 0.5}}},
                "meta": {"k": 0, "noul": False, "ambiguous": True}}
    return None
