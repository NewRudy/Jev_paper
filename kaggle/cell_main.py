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

# Everything below is skipped when the checkpoints already exist in this
# session, so the expensive part runs once and later edits only re-run PMC.
NEEDS_TRAIN = not os.path.exists("runs/deep/adapter_model.safetensors")

# ---- baselines ----
if NEEDS_TRAIN:
  for k in KS:
    run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data {BASE % k} --out runs/base-k{k}")
  run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b --data data/ambiguous.jsonl --out runs/base-amb")

# ---- four training runs ----
if NEEDS_TRAIN:
  print(">>> training (first run in this session)")
  train("data/chain_train_arft_k12.jsonl", "runs/arft", 3)
  train("data/chain_train_k12.jsonl",      "runs/std",  3)
  train("data/chain_train_std1k_k12.jsonl", "runs/std1k", 2)   # size-matched control
  train("data/chain_train_deep_k14.jsonl", "runs/deep", 2)     # depth-diverse

# ---- direct benchmarks for every model ----
if NEEDS_TRAIN:
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# CELL 2 — serve + PMC + permute + JS (re-runnable; cheap)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

sys.path.insert(0, "/kaggle/working/Jev_paper/gen")
import eval_decompose as ED
print("eval_decompose loaded from", ED.__file__)

def serve(run_dir, port):
    subprocess.Popen(
        f"KEV_DTYPE=fp32 {UV} -m kev.serve --run {run_dir} --port {port}",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import urllib.request as _u
    for _ in range(90):
        time.sleep(5)
        try:
            with _u.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    print(f"serve ready on {port}")
                    return True
        except Exception:
            pass
    print(f"WARN: serve on {port} not ready after 450s")
    return False

MODEL = os.environ.get("PMC_MODEL", "deep")
PORT = 8009
serve(f"runs/{MODEL}", PORT)

results = {}
for k in KS:
    try:
        recs = ED.eval_file_via_serve(f"data/chain_test_k{k}.jsonl", PORT, permute_n=k)
        summ = ED.summarize(recs)
        results[k] = summ
        print(f"k={k}: {json.dumps(summ)}")
        if k == 2:
            print("JS divergence (independence check):",
                  json.dumps(ED.js_divergence_check(recs)))
        with open(f"runs/pmc-{MODEL}-k{k}.json", "w") as f:
            json.dump({"summary": summ, "records": recs}, f)
    except Exception as e:
        import traceback
        print(f"k={k} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        results[k] = {"error": str(e)}

print("\n=========== PMC vs DIRECT vs PERMUTE (model=%s) ===========" % MODEL)
for k in sorted(results):
    r = results[k]
    if "error" in r:
        print(f"k={k}: ERROR {r['error'][:60]}")
    else:
        d = r.get("acc_direct"); p = r.get("acc_pmc"); v = r.get("acc_permute")
        f = lambda x: "  n/a" if x is None else "%.3f" % x
        print(f"k={k}: direct={f(d)}  pmc={f(p)}  permute={f(v)}  n={r.get('n')}")

print("saved:", sorted(str(p) for p in pathlib.Path("runs").glob(f"pmc-{MODEL}-k*.json")))
