# ============================================================
# CELL 2 — serve + PMC decomposition + permute-vote + JS divergence
# (cheap; re-runnable within the same session after edits)
# ============================================================
import subprocess, json, time, sys, os, pathlib

WORK = "/kaggle/working"
os.chdir(f"{WORK}/kev")
UV = f"{WORK}/kev/.venv/bin/python"
KS = [1, 2, 3, 4, 5, 6]

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
