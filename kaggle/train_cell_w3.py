# ============================================================
# W3 experiments (R1 review response) — Kaggle T4, 0.8B, fp32
#  X1: deep training (k=1..4) -> test k=5,6  [direct vs PMC]
#  A1: AR-FT fixed (dual-label anchors) vs standard FT vs base
#  X2a: JS divergence (convolved vs direct) on k=2
#  X2b: permute-vote baseline (judgment-model self-consistency analogue)
#  All data generated IN-CELL from the public repo (no dataset mounting).
# ============================================================
import os, subprocess, glob, json, time

WORK = "/kaggle/working"
os.chdir(WORK)

def run(cmd):
    print("::", cmd)
    return subprocess.run(cmd, shell=True, check=False).returncode

# ---------- 0) repos + env ----------
run("git clone --depth 1 https://github.com/jaredpalmer/kev.git")
run("git clone --depth 1 https://github.com/NewRudy/Jev_paper.git")
os.chdir(f"{WORK}/kev")
run("pip install -q uv")
run("uv sync --extra serve")
UV = f"{WORK}/kev/.venv/bin/python"
assert os.path.exists(UV), "uv venv missing"

# ---------- 1) generate + verify ALL data in-cell (pure stdlib, seconds) ----------
for script in ["make_dataset_v2.py", "make_arft_dataset.py", "make_dataset_deep.py"]:
    rc = run(f"python3 /kaggle/working/Jev_paper/gen/{script}")
    assert rc == 0, script
# copy generator for verify imports
run("cp /kaggle/working/Jev_paper/gen/*.py /kaggle/working/kev/ 2>/dev/null || true")
os.makedirs("data", exist_ok=True)
run(f"ls -la data/ | head -25")
# generators (run above) wrote into ./data (cwd = kev); verify THAT data
# (arft set mixes chain+coord samples; its chain part == chain_train_k12,
#  its anchors are construction-verified separately — see check_arft.py)
run(f"PYTHONPATH=/kaggle/working/Jev_paper/gen python3 /kaggle/working/Jev_paper/gen/verify_v2.py "
   f"data/chain_train_k12.jsonl "
   f"data/chain_train_deep_k14.jsonl "
   f"data/chain_test_k5.jsonl "
   f"data/chain_test_k6.jsonl "
   f"data/ambiguous.jsonl")

BASE = "data/chain_test_k%d.jsonl"
KS = [1, 2, 3, 4, 5, 6]

# ---------- 2) baselines: base kev-0.8b on k=1..6 + ambiguity ----------
for k in KS:
    run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data {BASE % k} --out runs/base-k{k}")
run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data data/ambiguous.jsonl --out runs/base-amb")

# ---------- 3) three training runs (A1 fixed AR-FT, standard, deep) ----------
TRAIN_COMMON = "--base Qwen/Qwen3.5-0.8B-Base --init_from jaredpalmer/kev-0.8b " \
               "--lr 2e-5 --batch 1 --accum 8 --dtype fp32 --device cuda "
def train(data, out, epochs):
    rc = run(f"{UV} -m kev.train --data {data} {TRAIN_COMMON} "
             f"--epochs {epochs} --out {out}")
    assert rc == 0, f"train failed: {out}"

train("data/chain_train_arft_k12.jsonl", "runs/arft", 3)     # A1-fixed
train("data/chain_train_k12.jsonl",      "runs/std",  3)     # standard FT (3ep, matches arft)
train("data/chain_train_deep_k14.jsonl", "runs/deep", 2)     # X1 deep

# ---------- 4) direct evals for all models on k=1..6 + ambiguity ----------
for model in ["std", "arft", "deep"]:
    for k in KS:
        run(f"{UV} -m kev.benchmark --run runs/{model} --data {BASE % k} --out runs/{model}-k{k}")
    run(f"{UV} -m kev.benchmark --run runs/{model} --data data/ambiguous.jsonl --out runs/{model}-amb")

print("\n=========== DIRECT-ANSWER SUMMARY (acc) ===========")
import pathlib
for f in sorted(pathlib.Path("runs").glob("*k*/report.json")) + \
         sorted(pathlib.Path("runs").glob("*amb*/report.json")):
    try:
        d = json.loads(f.read_text()).get("clean", {})
        print(f.parent.name, "acc=%.3f ece=%.3f brier=%.3f" % (
            d.get("acc", -1), d.get("ece", -1), d.get("brier", -1)))
    except Exception as e:
        print(f, e)

# ---------- 5) serve deep model; PMC + permute-vote via serve ----------
import threading, socket
def serve(run_dir, port):
    subprocess.Popen(
        f"KEV_DTYPE=fp32 {UV} -m kev.serve --run {run_dir} --port {port}",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # poll until the server answers
    import urllib.request as _u
    for _ in range(60):
        time.sleep(5)
        try:
            with _u.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    print(f"serve ready on {port} after poll")
                    return
        except Exception:
            pass
    print(f"WARN: serve on {port} not ready after 300s; continuing anyway")

serve("runs/deep", 8009)
sys_path = "/kaggle/working/Jev_paper/gen"
import sys
sys.path.insert(0, sys_path)
import eval_decompose as ED

print("\n=========== PMC + PERMUTE (deep model) ===========")
pmc_results = {}
for k in KS:
    recs = ED.eval_file_via_serve(f"data/chain_test_k{k}.jsonl", 8009, permute_n=k)
    summ = ED.summarize(recs)
    pmc_results[k] = {"summary": summ, "records_sample": recs[:3]}
    print(f"k={k}: {json.dumps(summ)}")
    if k == 2:
        print("JS divergence (independence check):",
              json.dumps(ED.js_divergence_check(recs)))
    with open(f"runs/pmc-k{k}.json", "w") as f:
        json.dump({"summary": summ, "records": recs}, f)

print("\nW3 done. All artifacts under /kaggle/working/kev/runs/")
