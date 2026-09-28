"""Build paper tables straight from kev's rows.json — no hand-transcribed numbers.

Why: every result so far was scraped out of Kaggle log text with a regex, which
is a standing source of transcription errors and gives no uncertainty. This
reads the per-question rows kev already writes (probability vector, label
index, optionally raw logits) and recomputes everything, with Wilson intervals
for rates and a paired bootstrap for arm-vs-arm differences.

    python3 gen/make_results.py --runs runs --out results
    python3 gen/make_results.py --runs runs --compare arft:std --out results

Outputs results/tables.md plus results/raw_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ metrics
def metrics_from_rows(rows):
    """Recompute the metric set from per-question rows."""
    n = 0
    acc = 0.0
    brier = 0.0
    nll = 0.0
    confs = []
    corrects = []
    for r in rows:
        p = np.asarray(r["p"], dtype=float)
        if p.sum() > 0:
            p = p / p.sum()
        y = int(r["label"])
        n += 1
        ok = int(p.argmax() == y)
        acc += ok
        brier += float(((p - np.eye(len(p))[y]) ** 2).sum())
        nll += float(-math.log(max(p[y], 1e-12)))
        conf = float(p.max())
        confs.append(conf)
        corrects.append(ok)
    if n == 0:
        return None
    confs = np.asarray(confs)
    corrects = np.asarray(corrects)
    # ECE with proper per-bin confidence means
    bins = 10
    e = 0.0
    for b in range(bins):
        m = (confs >= b / bins) & (confs < (b + 1) / bins)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(confs[m].mean() - corrects[m].mean())
    # selective prediction: risk-coverage, sorted by confidence descending
    order = np.argsort(-confs)
    cov = np.arange(1, n + 1) / n
    risk = 1.0 - corrects[order].cumsum() / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))
    conf_err = float(np.mean((confs >= 0.9) & (corrects == 0)))
    return {
        "n": n,
        "acc": acc / n,
        "ece": float(e),
        "brier": brier / n,
        "nll": nll / n,
        "mean_conf": float(confs.mean()),
        "confident_error_rate": conf_err,
        "coverage_at_0_9": float(np.mean(confs >= 0.9)),
        "aurc": aurc,
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def paired_bootstrap(rows_a, rows_b, key="ok", n_boot=2000, seed=0):
    """Bootstrap the paired difference in accuracy (rows are aligned by id)."""
    a = {r.get("record_id", r.get("_id", i)): metrics_from_rows([r]) for i, r in enumerate(rows_a)}
    ia = [(r.get("record_id", r.get("_id", i)),
           int(np.asarray(r["p"]).argmax() == int(r["label"])))
          for i, r in enumerate(rows_a)]
    ib = {(r.get("record_id", r.get("_id", i))): int(np.asarray(r["p"]).argmax() == int(r["label"]))
          for i, r in enumerate(rows_b)}
    keys = [k for k, _ in ia if k in ib]
    if not keys:
        return None
    d = np.asarray([v - ib[k] for k, v in ia if k in ib], dtype=float)
    obs = float(d.mean())
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(d)) for _ in range(len(d))]
        boot.append(float(d[idx].mean()))
    boot.sort()
    lo, hi = boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]
    p = 2 * min(sum(1 for x in boot if x <= 0), sum(1 for x in boot if x >= 0)) / n_boot
    return {"diff": obs, "ci95": [lo, hi], "p_two_sided": min(1.0, p), "n": len(d)}


# ------------------------------------------------------------------ loading
def load_arm(run_dir: Path):
    """Return {dataset_tag: rows} for one arm directory."""
    out = {}
    for rj in run_dir.rglob("rows.json"):
        tag = rj.parent.name
        out[tag] = json.loads(rj.read_text())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="pairs like arft:std to test an arm against another")
    ap.add_argument("--only", default=None, help="substring filter on dataset tags")
    args = ap.parse_args()

    runs = Path(args.runs)
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    table = {}          # arm -> tag -> metrics
    raw = {}
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        arm = d.name
        arms = load_arm(d)
        if not arms:
            continue
        table[arm], raw[arm] = {}, {}
        for tag, rows in arms.items():
            if args.only and args.only not in tag:
                continue
            m = metrics_from_rows(rows)
            if m:
                lo, hi = wilson(round(m["acc"] * m["n"]), m["n"])
                m["acc_ci95"] = [lo, hi]
                table[arm][tag] = m
                raw[arm][tag] = len(rows)

    # ---- markdown ----
    md = ["# Results (auto-generated from rows.json)", ""]
    tags = sorted({t for a in table.values() for t in a})
    if not tags:
        print("no rows.json found under", runs)
        return
    md.append("## Per-arm metrics\n")
    md.append("| arm | dataset | n | acc [95% CI] | ECE | Brier | mean_conf | conf_err@0.9 | AURC |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for arm in sorted(table):
        for tag in sorted(table[arm]):
            m = table[arm][tag]
            ci = "[%.3f, %.3f]" % (m["acc_ci95"][0], m["acc_ci95"][1])
            md.append(f"| {arm} | {tag} | {m['n']} | {m['acc']:.3f} {ci} | {m['ece']:.3f} | "
                      f"{m['brier']:.3f} | {m['mean_conf']:.3f} | {m['confident_error_rate']:.3f} | {m['aurc']:.3f} |")
    md.append("")

    # ---- paired comparisons ----
    if args.compare:
        md.append("## Paired comparisons (accuracy difference, record-clustered bootstrap)\n")
        md.append("| A | B | dataset | diff | 95% CI | p | n |")
        md.append("|---|---|---|---|---|---|---|")
        for pair in args.compare:
            A, B = pair.split(":")
            if A not in raw or B not in raw:
                continue
            for tag in sorted(set(raw[A]) & set(raw[B])):
                ra = json.loads((runs / A / tag / "rows.json").read_text())
                rb = json.loads((runs / B / tag / "rows.json").read_text())
                r = paired_bootstrap(ra, rb)
                if r:
                    md.append(f"| {A} | {B} | {tag} | {r['diff']:+.3f} | "
                              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | "
                              f"{r['p_two_sided']:.4f} | {r['n']} |")
        md.append("")

    (outdir / "tables.md").write_text("\n".join(md))
    (outdir / "raw_metrics.json").write_text(json.dumps(table, indent=2))
    print("\n".join(md[:60]))
    print(f"\n[written] {outdir/'tables.md'} and {outdir/'raw_metrics.json'}")


if __name__ == "__main__":
    main()
