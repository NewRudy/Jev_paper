# ============================================================
# Enterprise Policy Routing Benchmark — Kaggle T4 Run (0.8B)
# Evaluates multi-hop policy decision chains (k=1..4)
# Demonstrates cross-domain compositional collapse and System 1.5 viability
# ============================================================
import os, subprocess, glob, json, pathlib, collections

WORK = "/kaggle/working"
os.chdir(WORK)

def run(cmd):
    print("::", cmd)
    return subprocess.run(cmd, shell=True, check=False).returncode

# 1. Setup Kev environment
if not os.path.exists(f"{WORK}/kev"):
    run("git clone --depth 1 https://github.com/jaredpalmer/kev.git")
os.chdir(f"{WORK}/kev")
run("pip install -q uv")
run("uv sync --extra serve")
UV = f"{WORK}/kev/.venv/bin/python"
assert os.path.exists(UV), "uv venv missing"

# 2. Setup Data
DATA_IN = sorted(glob.glob("/kaggle/input/**/policy_train_k12.jsonl", recursive=True))
os.makedirs("data", exist_ok=True)
assert DATA_IN, "policy dataset not mounted"
src = os.path.dirname(DATA_IN[0])
run(f"cp {src}/policy_*.jsonl data/")

# 3. Base model evaluation (Zero-shot)
print("\n>>> 1. Benchmarking Base Kev-0.8B on Policy Chains...")
for k in range(1, 5):
    run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data data/policy_test_k{k}.jsonl --out runs/base-policy-k{k}")

# 4. Fine-tuning on k<=2 Policy Chains
print("\n>>> 2. Training Kev-0.8B on policy_train_k12...")
rc = run(f"{UV} -m kev.train --data data/policy_train_k12.jsonl --base Qwen/Qwen3.5-0.8B-Base "
         f"--init_from jaredpalmer/kev-0.8b --epochs 3 --lr 2e-5 --batch 1 --accum 8 "
         f"--dtype fp16 --device cuda --out runs/policy_ft")
if rc != 0:
    run(f"{UV} -m kev.train --data data/policy_train_k12.jsonl --base Qwen/Qwen3.5-0.8B-Base "
        f"--init_from jaredpalmer/kev-0.8b --epochs 3 --lr 2e-5 --batch 1 --accum 8 "
        f"--dtype fp32 --device cuda --out runs/policy_ft")

# 5. Evaluate Fine-Tuned Model on k=1..4
for k in range(1, 5):
    run(f"{UV} -m kev.benchmark --run runs/policy_ft --data data/policy_test_k{k}.jsonl --out runs/policy-eval-k{k}")

# 6. Results summary
print("\n" + "="*50)
print("=== POLICY ROUTING BENCHMARK RESULTS ===")
print("="*50)
for f in sorted(pathlib.Path("runs").rglob("*.json")):
    tag = f.parent.name
    if "policy" in tag:
        try:
            d = json.loads(f.read_text())
            clean = d.get("clean", d)
            acc = clean.get("acc", -1)
            ece = clean.get("ece", -1)
            brier = clean.get("brier", -1)
            print(f"{tag:25s} | acc={acc:.3f} | ece={ece:.3f} | brier={brier:.3f}")
        except Exception:
            pass
