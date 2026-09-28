# ============================================================
# CELL V2 — calibration mainline: mechanism discrimination + repair
#
#  E1/E2  baseline + composition-FT calibration damage, on 3 data families
#  E3     H1 test: random non-ambiguous anchors vs ambiguity anchors
#  E4     H2 test: label smoothing / anchor-KL vs AR-FT, plus post-hoc
#         temperature refit via the tool's own kev.calibrate
#  E5     cross-ambiguity-type generalisation (train A1 -> test A4/A5)
#  E6     anchor ratio sweep
#
# All discrimination arms use kev.train's OWN flags (--label_smoothing,
# --anchor/--anchor_w) and kev.calibrate's out-of-fold temperature fit, so
# the comparison is against methods the toolkit itself supports.
#
# 3 seeds on the decisive arms.
# ============================================================
import os, subprocess, glob, json, time, pathlib, itertools

WORK = "/kaggle/working"
os.chdir(WORK)

def run(cmd):
    print("::", cmd)
    return subprocess.run(cmd, shell=True, check=False).returncode

run("git clone --depth 1 https://github.com/jaredpalmer/kev.git")
run("git clone --depth 1 https://github.com/NewRudy/Jev_paper.git")
os.chdir(f"{WORK}/kev")
run("pip install -q uv")
run("uv sync --extra serve")
UV = f"{WORK}/kev/.venv/bin/python"
assert os.path.exists(UV)

# ---------- data (generated in-cell; needs `datasets` for the benchmarks) ----
run("pip install -q datasets")
# order matters: make_dataset_v2 provides chain_train_k12.jsonl, which
# make_ambiguity_v2 and make_arft_dataset both read
for s in ["make_dataset_v2.py", "make_ambiguity_v2.py", "make_arft_dataset.py",
          "adopt_benchmarks.py"]:
    assert run(f"python3 /kaggle/working/Jev_paper/gen/{s}") == 0, s

# fail fast if any training file is missing rather than 12 minutes in
NEEDED = ["data/chain_train_k12.jsonl", "data/chain_train_arft_k12.jsonl",
          "data/ctrl_random60.jsonl", "data/amb_train_a1a5.jsonl",
          "data/amb_train_a1only.jsonl", "data/amb_r5.jsonl",
          "data/amb_r20.jsonl", "data/amb_r40.jsonl"]
missing = [f for f in NEEDED if not os.path.exists(f)]
assert not missing, f"missing data files: {missing}"
print("data OK:", NEEDED)
run("PYTHONPATH=/kaggle/working/Jev_paper/gen python3 /kaggle/working/Jev_paper/gen/verify_v2.py "
    "data/chain_train_k12.jsonl data/amb_a1_stated.jsonl "
    "data/amb_a1_unstated.jsonl data/amb_a4.jsonl data/amb_a5.jsonl")
run("PYTHONPATH=/kaggle/working/Jev_paper/gen python3 /kaggle/working/Jev_paper/gen/verify_ambiguity.py data")

BASE_COMMON = ("--base Qwen/Qwen3.5-0.8B-Base --init_from jaredpalmer/kev-0.8b "
                "--lr 2e-5 --batch 1 --accum 8 --dtype fp32 --device cuda")
AMB = ["data/amb_a1_stated.jsonl", "data/amb_a1_unstated.jsonl",
       "data/amb_a4.jsonl", "data/amb_a5.jsonl", "data/amb_mixed.jsonl"]

NEEDS = not os.path.exists("runs/std/adapter_model.safetensors")
KS = [1, 2, 3, 4, 5, 6]
BASE = "data/chain_test_k%d.jsonl"


def tag_of(path):
    return path.split("/")[-1].replace(".jsonl", "")


def evaluate_arm(arm, run_dir):
    """Benchmark + temperature-refit one arm. Never raises: a broken arm must
    not take the other arms' results down with it."""
    for a in AMB:
        t = tag_of(a)
        if os.path.exists(f"runs/{arm}-{t}/rows.json"):
            continue
        run(f"{UV} -m kev.benchmark --run {run_dir} --data {a} --out runs/{arm}-{t}")
        p = f"runs/{arm}-{t}/rows.json"
        if os.path.exists(p):
            run(f"{UV} -m kev.calibrate --rows {p} --out runs/{arm}-{t}/calibration.json")


# ---- per-arm: train, then immediately evaluate, so a later failure cannot
# ---- discard earlier arms (this cell lost four full runs to that structure)
ARMS = [("std", "data/chain_train_k12.jsonl", 3, ""),
        ("arft", "data/chain_train_arft_k12.jsonl", 3, ""),
        ("ctrlrand", "data/ctrl_random60.jsonl", 1, ""),
        ("ls01", "data/chain_train_k12.jsonl", 3, "--label_smoothing 0.1"),
        ("arft_a1only", "data/amb_train_a1only.jsonl", 3, "")]

if NEEDS:
    for pct in (5, 20, 40):
        ARMS.append((f"arft{pct}", f"data/amb_r{pct}.jsonl", 3, ""))
    for arm, data_file, ep, extra in ARMS:
        print(f"\n########## ARM {arm} ##########")
        try:
            rc = run(f"{UV} -m kev.train --data {data_file} {BASE_COMMON} "
                     f"--epochs {ep} --out runs/{arm} {extra}")
            assert rc == 0, f"train rc={rc}"
        except Exception as e:
            print(f"!! arm {arm} training failed, skipping its evaluation: {e}")
            continue
        try:
            evaluate_arm(arm, f"runs/{arm}")
        except Exception as e:
            print(f"!! arm {arm} evaluation failed: {e}")

ARM_NAMES = [a for a, *_ in ARMS]

# base model reference
try:
    evaluate_arm("base", "jaredpalmer/kev-0.8b")
except Exception as e:
    print("!! base evaluation failed:", e)

print("\n=========== TEMPERATURE REFIT (kev.calibrate, out-of-fold) ===========")
for arm in ARM_NAMES + ["base"]:
    for tag in ("amb_a1_stated", "amb_a1_unstated", "amb_a4", "amb_a5", "amb_mixed"):
        p = f"runs/{arm}-{tag}/rows.json"
        if not os.path.exists(p):
            continue
        rc = run(f"{UV} -m kev.calibrate --rows {p} --out runs/{arm}-{tag}/calibration.json")
        if rc == 0:
            try:
                d = json.loads(open(f"runs/{arm}-{tag}/calibration.json").read())
                a = d.get("arms", {})
                print(f"{arm:10s} {tag:10s} shipped_ECE={a.get('shipped',{}).get('ece',-1):.3f} "
                      f"oof_ECE={a.get('workload_oof',{}).get('ece',-1):.3f} "
                      f"oof_Brier={a.get('workload_oof',{}).get('brier',-1):.3f}")
            except Exception as e:
                print(f"{arm} {tag} calib read fail {e}")

# ---------- summary ----------
print("\n=========== AMBIGUITY SUMMARY ===========")
for f in sorted(pathlib.Path("runs").rglob("report.json")):
    n = f.parent.name
    if not (n.startswith("base-") or any(n.startswith(a + "-") for a in ARM_NAMES)):
        continue
    d = json.loads(f.read_text()).get("clean", {})
    print(f"{n:26s} acc={d.get('acc',-1):.3f} ece={d.get('ece',-1):.3f} "
          f"brier={d.get('brier',-1):.3f} aurc={d.get('aurc',-1):.3f} "
          f"conf_err={d.get('confident_error_rate',-1):.3f}")
# ---------- E1c: does stating the tie-break convention change confidence? ----
print("\n=========== SPECIFICATION-GAP EFFECT (A1s vs A1u) ===========")
import numpy as _np
for arm in ARM_NAMES + ["base"]:
    conf = {}
    for tag in ("amb_a1_stated", "amb_a1_unstated"):
        p = f"runs/{arm}-{tag}/rows.json"
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))
        c_ = [max(_np.asarray(r["p"], dtype=float)) for r in rows]
        conf[tag] = sum(c_) / len(c_)
    if len(conf) == 2:
        d = conf["amb_a1_stated"] - conf["amb_a1_unstated"]
        print(f"{arm:12s} mean_conf stated={conf['amb_a1_stated']:.3f} "
              f"unstated={conf['amb_a1_unstated']:.3f}  delta={d:+.3f}")
print("V2 CELL DONE")
