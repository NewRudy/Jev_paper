# ============================================================
# CELL 1 — setup, data generation, training, direct benchmarks
# (expensive, known-good; run once per session)
# ============================================================
import os, subprocess, glob, json, time, pathlib

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
assert os.path.exists(UV), "uv venv missing"

# ---- data generated in-cell (no dataset mounting) ----
for script in ["make_dataset_v2.py", "make_arft_dataset.py", "make_dataset_deep.py"]:
    assert run(f"python3 /kaggle/working/Jev_paper/gen/{script}") == 0, script
run(f"PYTHONPATH=/kaggle/working/Jev_paper/gen python3 /kaggle/working/Jev_paper/gen/verify_v2.py "
   f"data/chain_train_k12.jsonl data/chain_train_deep_k14.jsonl "
   f"data/chain_train_std1k_k12.jsonl data/chain_test_k5.jsonl "
   f"data/chain_test_k6.jsonl data/ambiguous.jsonl")

KS = [1, 2, 3, 4, 5, 6]
BASE = "data/chain_test_k%d.jsonl"
TRAIN_COMMON = ("--base Qwen/Qwen3.5-0.8B-Base --init_from jaredpalmer/kev-0.8b "
                "--lr 2e-5 --batch 1 --accum 8 --dtype fp32 --device cuda")

def train(data, out, epochs):
    assert run(f"{UV} -m kev.train --data {data} {TRAIN_COMMON} "
               f"--epochs {epochs} --out {out}") == 0, f"train failed: {out}"

# ---- baselines ----
for k in KS:
    run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data {BASE % k} --out runs/base-k{k}")
run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data data/ambiguous.jsonl --out runs/base-amb")

# ---- four training runs ----
train("data/chain_train_arft_k12.jsonl", "runs/arft", 3)
train("data/chain_train_k12.jsonl",      "runs/std",  3)
train("data/chain_train_std1k_k12.jsonl", "runs/std1k", 2)   # size-matched control
train("data/chain_train_deep_k14.jsonl", "runs/deep", 2)     # depth-diverse

# ---- direct benchmarks for every model ----
for model in ["std", "arft", "std1k", "deep"]:
    for k in KS:
        run(f"{UV} -m kev.benchmark --run runs/{model} --data {BASE % k} --out runs/{model}-k{k}")
    run(f"{UV} -m kev.benchmark --run runs/{model} --data data/ambiguous.jsonl --out runs/{model}-amb")

OWNED = ("base-", "std-", "std1k-", "arft-", "deep-")
print("\n=========== DIRECT-ANSWER SUMMARY ===========")
for f in sorted([p for p in pathlib.Path("runs").rglob("report.json")
                 if p.parent.name.startswith(OWNED)]):
    try:
        d = json.loads(f.read_text()).get("clean", {})
        print(f.parent.name, "acc=%.3f ece=%.3f brier=%.3f" % (
            d.get("acc", -1), d.get("ece", -1), d.get("brier", -1)))
    except Exception as e:
        print(f, e)
print("CELL 1 DONE — now run CELL 2 (serve + PMC)")
