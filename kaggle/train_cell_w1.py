# ============================================================
# Spatial System One — W1 experiments (Kaggle T4, 0.8B, fp32)
# E1 chain: train k<=2, test k=1..4 (compositional generalization)
# E2 ladder: coord contrastive prefixes 400/800/1600/3200 samples
# E3 ambiguity: calibration probe on diagonal pairs
# ============================================================
import os, subprocess, glob, json

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

# 数据：新独立数据集 spatial-chain（挂载最新版）；坐标阶梯数据缺失则跳过 E2
DATA_IN = sorted(glob.glob("/kaggle/input/**/chain_train_k12.jsonl", recursive=True))
os.makedirs("data", exist_ok=True)
assert DATA_IN, "spatial-chain dataset not mounted"
src = os.path.dirname(DATA_IN[0])
run(f"cp {src}/*.jsonl data/")
HAVE_COORD = bool(glob.glob("/kaggle/input/**/coord_train_1600.jsonl", recursive=True))
print("coord ladder data available:", HAVE_COORD)

# ---------- E1: chain (baseline first) ----------
for k in range(1, 5):
    run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data data/chain_test_k{k}.jsonl --out runs/base-chain-k{k}")

rc = run(f"{UV} -m kev.train --data data/chain_train_k12.jsonl --base Qwen/Qwen3.5-0.8B-Base "
         f"--init_from jaredpalmer/kev-0.8b --epochs 3 --lr 2e-5 --batch 1 --accum 8 "
         f"--dtype fp16 --device cuda --out runs/chain")
if rc != 0:
    run(f"{UV} -m kev.train --data data/chain_train_k12.jsonl --base Qwen/Qwen3.5-0.8B-Base "
        f"--init_from jaredpalmer/kev-0.8b --epochs 3 --lr 2e-5 --batch 1 --accum 8 "
        f"--dtype fp32 --device cuda --out runs/chain")
for k in range(1, 5):
    run(f"{UV} -m kev.benchmark --run runs/chain --data data/chain_test_k{k}.jsonl --out runs/chain-eval-k{k}")

# ---------- E2: coordinate scaling ladder (needs v2 coord data; skipped otherwise) ----------
if HAVE_COORD:
    run(f"head -n 400  data/coord_train_1600.jsonl > data/coord_s400.jsonl")
    run(f"head -n 800  data/coord_train_1600.jsonl > data/coord_s800.jsonl")
    run(f"head -n 1600 data/coord_train_1600.jsonl > data/coord_s1600.jsonl")
    run(f"cp data/coord_train_1600.jsonl data/coord_s3200.jsonl")
    for n in [400, 800, 1600, 3200]:
        rc = run(f"{UV} -m kev.train --data data/coord_s{n}.jsonl --base Qwen/Qwen3.5-0.8B-Base "
                 f"--init_from jaredpalmer/kev-0.8b --epochs 2 --lr 2e-5 --batch 1 --accum 8 "
                 f"--dtype fp16 --device cuda --out runs/coord{n}")
        if rc != 0:
            run(f"{UV} -m kev.train --data data/coord_s{n}.jsonl --base Qwen/Qwen3.5-0.8B-Base "
                f"--init_from jaredpalmer/kev-0.8b --epochs 2 --lr 2e-5 --batch 1 --accum 8 "
                f"--dtype fp32 --device cuda --out runs/coord{n}")
        run(f"{UV} -m kev.benchmark --run runs/coord{n} --data data/eval_id.jsonl --out runs/coord{n}-eval")
        run(f"{UV} -m kev.benchmark --run runs/coord{n} --data data/eval_ood_rule.jsonl --out runs/coord{n}-oodrule")
else:
    print("SKIP E2 ladder: coord_train_1600.jsonl not mounted")

# ---------- E3: ambiguity calibration probe ----------
run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data data/ambiguous.jsonl --out runs/base-amb")
run(f"{UV} -m kev.benchmark --run runs/chain --data data/ambiguous.jsonl --out runs/chain-amb")
if HAVE_COORD:
    run(f"{UV} -m kev.benchmark --run runs/coord3200 --data data/ambiguous.jsonl --out runs/coord3200-amb")

print("\n=========== ALL RESULTS ===========")
import pathlib
for f in sorted([p for p in pathlib.Path("runs").rglob("*.json") if any(p.parts[1].startswith(x) for x in ("base","chain","coord","eval","random"))]):
    try:
        d = json.loads(f.read_text())
        clean = d.get("clean", d)
        print(f, "acc=%.3f ece=%.3f brier=%.3f" % (
            clean.get("acc", -1), clean.get("ece", -1), clean.get("brier", -1)))
    except Exception as e:
        print(f, "unreadable", e)
