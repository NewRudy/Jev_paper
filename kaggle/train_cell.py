# ============================================================
# Spatial System One — Kev fine-tune pilot (Kaggle T4, fp16)
# 粘贴到 Kaggle Notebook 的一个 cell 里运行。
# 前置：Settings 里开 Internet、Accelerator 选 GPU T4 x2；
#       Add Input 挂载 spatial-systemone 数据集（含 data/*.jsonl）。
# ============================================================
import os, subprocess, glob, json

WORK = "/kaggle/working"
os.chdir(WORK)

def run(cmd, **kw):
    print("::", cmd)
    return subprocess.run(cmd, shell=True, check=False, **kw).returncode

# 1) clone kev + uv 环境
run("git clone --depth 1 https://github.com/jaredpalmer/kev.git")
os.chdir(f"{WORK}/kev")
run("pip install -q uv")
run("uv sync --extra serve")
UV = f"{WORK}/kev/.venv/bin/python"
assert os.path.exists(UV), "uv venv missing"

# 2) 数据：从挂载的 Dataset 拷进来（zip 内含顶层目录，用递归 glob）
DATA_IN = sorted(glob.glob("/kaggle/input/**/data/train.jsonl", recursive=True))
os.makedirs("data", exist_ok=True)
if DATA_IN:
    src = os.path.dirname(DATA_IN[0])
    run(f"cp {src}/*.jsonl data/")
else:
    print("!! 未找到挂载数据集，请 Add Input: spatial-systemone")
    raise SystemExit(1)

# 3) 微调前 baseline（0.8B 原版在我们数据上的起点）
run(f"{UV} -m kev.benchmark --run jaredpalmer/kev-0.8b "
    f"--data data/eval_id.jsonl --out runs/base-eval")

# 4) 对比数据微调（fp16；T4 不支持 bf16，若 fp16 不被接受则退 fp32）
COMMON = ("--data data/train.jsonl --base Qwen/Qwen3.5-0.8B-Base "
          "--init_from jaredpalmer/kev-0.8b --epochs 2 --lr 2e-5 "
          "--batch 1 --accum 8 --device cuda --out runs/spatial")
rc = run(f"{UV} -m kev.train {COMMON} --dtype fp16")
if rc != 0:
    print("fp16 失败，退 fp32")
    run(f"{UV} -m kev.train {COMMON} --dtype fp32")

# 5) 全套评测
for split in ["eval_id", "eval_ood_size", "eval_ood_count", "eval_ood_rule"]:
    run(f"{UV} -m kev.benchmark --run runs/spatial --data data/{split}.jsonl "
        f"--out runs/eval-{split}")

# 6) RQ2 消融：随机（非对比）数据同规模微调
RCOMMON = COMMON.replace("data/train.jsonl", "data/train_random.jsonl").replace(
    "runs/spatial", "runs/random")
rc = run(f"{UV} -m kev.train {RCOMMON} --dtype fp16")
if rc != 0:
    run(f"{UV} -m kev.train {RCOMMON} --dtype fp32")
run(f"{UV} -m kev.benchmark --run runs/random --data data/eval_id.jsonl "
    f"--out runs/eval-random")

# 7) 汇总
print("\n=========== RESULTS ===========")
for f in sorted(glob.glob("runs/**/results*.json", recursive=True) +
                glob.glob("runs/**/*.json", recursive=True)):
    print("--", f)
print("完整结果在 /kaggle/working/kev/runs/ 下，Session 保存后可下载")
