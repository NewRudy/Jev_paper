# ============================================================
# Spatial System One / System 1.5 — W2 experiments (Kaggle T4, 0.8B)
# E1: Standard FT (k<=2) vs Baselines (k=1..4)
# E3/C3: AR-FT (Ambiguity-Regularized Fine-Tuning) vs Standard FT Calibration
# ============================================================
import os, subprocess, glob, json, pathlib

WORK = "/kaggle/working"
os.chdir(WORK)

def run(cmd):
    print("::", cmd)
    return subprocess.run(cmd, shell=True, check=False).returncode

run("git clone --depth 1 https://github.com/jaredpalmer/kev.git")
os.chdir(f"{WORK}/kev")
run("pip install -q uv")
run("uv sync --extra serve")
UV = f"{WORK}/kev/.venv/bin/python"
assert os.path.exists(UV), "uv venv missing"

# Data setup
DATA_IN = sorted(glob.glob("/kaggle/input/**/chain_train_k12.jsonl", recursive=True))
os.makedirs("data", exist_ok=True)
assert DATA_IN, "spatial dataset not mounted"
src = os.path.dirname(DATA_IN[0])
run(f"cp {src}/*.jsonl data/")

# 1. Base model evaluation (Zero-shot)
print("\n>>> 1. Benchmarking Base Kev-0.8B...")
for k in range(1, 5):
    run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data data/chain_test_k{k}.jsonl --out runs/base-chain-k{k}")
run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data data/ambiguous.jsonl --out runs/base-amb")

# 2. Standard Fine-Tuning (k<=2)
print("\n>>> 2. Training Standard Kev-0.8B (chain_train_k12)...")
rc = run(f"{UV} -m kev.train --data data/chain_train_k12.jsonl --base Qwen/Qwen3.5-0.8B-Base "
         f"--init_from jaredpalmer/kev-0.8b --epochs 3 --lr 2e-5 --batch 1 --accum 8 "
         f"--dtype fp16 --device cuda --out runs/chain")
if rc != 0:
    run(f"{UV} -m kev.train --data data/chain_train_k12.jsonl --base Qwen/Qwen3.5-0.8B-Base "
        f"--init_from jaredpalmer/kev-0.8b --epochs 3 --lr 2e-5 --batch 1 --accum 8 "
        f"--dtype fp32 --device cuda --out runs/chain")

for k in range(1, 5):
    run(f"{UV} -m kev.benchmark --run runs/chain --data data/chain_test_k{k}.jsonl --out runs/chain-eval-k{k}")
run(f"{UV} -m kev.benchmark --run runs/chain --data data/ambiguous.jsonl --out runs/chain-amb")

# 3. Ambiguity-Regularized Fine-Tuning (AR-FT)
HAVE_ARFT = os.path.exists("data/chain_train_arft_k12.jsonl")
if HAVE_ARFT:
    print("\n>>> 3. Training AR-FT Kev-0.8B (chain_train_arft_k12)...")
    rc = run(f"{UV} -m kev.train --data data/chain_train_arft_k12.jsonl --base Qwen/Qwen3.5-0.8B-Base "
             f"--init_from jaredpalmer/kev-0.8b --epochs 3 --lr 2e-5 --batch 1 --accum 8 "
             f"--dtype fp16 --device cuda --out runs/chain_arft")
    if rc != 0:
        run(f"{UV} -m kev.train --data data/chain_train_arft_k12.jsonl --base Qwen/Qwen3.5-0.8B-Base "
            f"--init_from jaredpalmer/kev-0.8b --epochs 3 --lr 2e-5 --batch 1 --accum 8 "
            f"--dtype fp32 --device cuda --out runs/chain_arft")

    for k in range(1, 5):
        run(f"{UV} -m kev.benchmark --run runs/chain_arft --data data/chain_test_k{k}.jsonl --out runs/arft-eval-k{k}")
    run(f"{UV} -m kev.benchmark --run runs/chain_arft --data data/ambiguous.jsonl --out runs/arft-amb")

# 4. Summary Table
print("\n" + "="*50)
print("=== EXPERIMENT SUMMARY RESULTS ===")
print("="*50)
summary = collections.defaultdict(dict)
for f in sorted(pathlib.Path("runs").rglob("*.json")):
    tag = f.parent.name
    try:
        d = json.loads(f.read_text())
        clean = d.get("clean", d)
        acc = clean.get("acc", -1)
        ece = clean.get("ece", -1)
        brier = clean.get("brier", -1)
        print(f"{tag:25s} | acc={acc:.3f} | ece={ece:.3f} | brier={brier:.3f}")
    except Exception as e:
        pass
