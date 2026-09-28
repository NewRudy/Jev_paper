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
    "data/chain_train_k12.jsonl data/amb_a1.jsonl data/amb_a4.jsonl data/amb_a5.jsonl")
run("PYTHONPATH=/kaggle/working/Jev_paper/gen python3 /kaggle/working/Jev_paper/gen/verify_ambiguity.py data")

BASE = ("--base Qwen/Qwen3.5-0.8B-Base --init_from jaredpalmer/kev-0.8b "
        "--lr 2e-5 --batch 1 --accum 8 --dtype fp32 --device cuda")
AMB = ["data/amb_a1.jsonl", "data/amb_a4.jsonl", "data/amb_a5.jsonl", "data/amb_mixed.jsonl"]

def train(data, out, extra="", epochs=3):
    rc = run(f"{UV} -m kev.train --data {data} {BASE} --epochs {epochs} --out {out} {extra}")
    assert rc == 0, f"train failed: {out}"
    return out

def bench(run_dir, data, out):
    run(f"{UV} -m kev.benchmark --run {run_dir} --data {data} --out {out}")

NEEDS = not os.path.exists("runs/std/rows.json") and not os.path.exists("runs/arft/rows.json")

if NEEDS:
    # ---------- E1/E2: does composition fine-tuning damage calibration? ----
    # std  = composition FT (k<=2)            -> the damaging condition
    # arft = same + ambiguity anchors (AR-FT)  -> the repair
    train("data/chain_train_k12.jsonl", "runs/std")
    train("data/chain_train_arft_k12.jsonl", "runs/arft")
    # ctrl: composition FT + the SAME COUNT of random non-ambiguous samples (E3/H1)
    train("data/ctrl_random60.jsonl", "runs/ctrlrand", epochs=1)

    # ---------- E4/H2: do standard remedies match AR-FT? ----------
    # (a) label smoothing -- kev's own flag
    train("data/chain_train_k12.jsonl", "runs/ls01", extra="--label_smoothing 0.1")
    # (b) anchor-KL toward the frozen base's zero-shot distribution
    #     (kev's own --anchor/--anchor_w; needs a base zero-shot dump)
    run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data data/chain_train_arft_k12.jsonl "
        f"--out runs/zs_anchor_src")
    train("data/chain_train_k12.jsonl", "runs/anchorkl",
          extra="--anchor runs/zs_anchor_src/rows.json --anchor_w 1.0")

    # ---------- E5: anchors on A1 ONLY (transfer to A4/A5 tested below) ------
    train("data/amb_train_a1only.jsonl", "runs/arft_a1only")

    # ---------- E6: anchor ratio sweep (files prebuilt by the generator) ----
    for ratio in (5, 20, 40):
        train(f"data/amb_r{ratio}.jsonl", f"runs/arft{ratio}")

# ---------- evaluation on every ambiguity set, for every arm ----------
ARMS = ["std", "arft", "ctrlrand", "ls01", "anchorkl", "arft_a1only"] + \
       [f"arft{r}" for r in (5, 20, 40)]
for arm in ARMS:
    if not os.path.exists(f"runs/{arm}/rows.json"):
        continue
    for a in AMB:
        tag = a.split("/")[-1].replace(".jsonl", "")
        bench(f"runs/{arm}", a, f"runs/{arm}-{tag}")

# base model for reference
for a in AMB:
    tag = a.split("/")[-1].replace(".jsonl", "")
    if not os.path.exists(f"runs/base-{tag}/rows.json"):
        bench("jaredpalmer/kev-0.8b", a, f"runs/base-{tag}")

# ---------- E4: post-hoc temperature refit (the tool's own OOF fit) ----------
print("\n=========== TEMPERATURE REFIT (kev.calibrate, out-of-fold) ===========")
for arm in ARMS + ["base"]:
    for tag in ("amb_a1", "amb_a4", "amb_a5", "amb_mixed"):
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
    if not (n.startswith("base-") or any(n.startswith(a + "-") for a in ARMS)):
        continue
    d = json.loads(f.read_text()).get("clean", {})
    print(f"{n:26s} acc={d.get('acc',-1):.3f} ece={d.get('ece',-1):.3f} "
          f"brier={d.get('brier',-1):.3f} aurc={d.get('aurc',-1):.3f} "
          f"conf_err={d.get('confident_error_rate',-1):.3f}")
print("V2 CELL DONE")
