# Jev Paper — 判断模型多跳空间组合推理研究

> System One 判断模型（Jev/Kev 类）的多跳组合推理：能力边界、置信引导分解、校准脆弱性。
> 目标期刊：Applied Intelligence（中科院 3 区）。

## 🔍 给评测/协作 AI 的入口（先读这个，按顺序）

1. **[RESEARCH_LOG.md](RESEARCH_LOG.md) — 研究轨迹与失败记录**（先读这个）
   全部 idea 转向、被推翻的假设、四次工程失败教训、v10 完整实测数据、三个待答问题。
   **当前 v3 方案的���心假设已被自己的对照实验推翻，文档诚实记录了这一点。**
2. **[RESEARCH_PLAN_V2.md](RESEARCH_PLAN_V2.md) — 现行方案**（v3 设计，实验矩阵 E1–E8）
3. **[RESEARCH_PLAN.md](RESEARCH_PLAN.md) — v1 方案**（组合推理线，§11 有 R1 审稿回应记录）
4. **[results/v10_raw.json](results/v10_raw.json)** — 机器可读实测结果

建议评判角度：
1. RESEARCH_LOG §4 的"温度缩放修指标不修决策"命题够不够三区
2. RESEARCH_LOG §3.3 的受控对照缺陷（两组非配对）修复后是否还有救
3. §3.4 A4/A5 明显过易（多数臂满分），有无更好的"欠定输入"构造
4. v1 组合推理线（§1.2）作为备选是否值得退回

## 仓库结构

```
RESEARCH_PLAN.md     研究方案（核心文档，评审对象）
gen/
  generator.py       数据生成器：网格世界坐标域 + 关系事实链（k跳）+ 歧义构造
  make_dataset.py    v1 数据入口（坐标域对比/随机/OOD）
  make_dataset_v2.py v2 数据入口（链式 k=1..4、歧义集、1600对坐标）
  verify.py          坐标域独立验证器（文本→解析→规则重算→比对标签）
  verify_v2.py       链式/歧义独立验证器（关系解析→路径向量组合→比对）
data/
  chain_train_k12.jsonl    链式训练集（600，k∈{1,2}，25% Noul）
  chain_test_k1..k4.jsonl  链式测试（各100；k=3,4 为组合外推）
  ambiguous.jsonl          内在歧义校准集（150，对角线，理想分布 0.5/0.5）
  coord_train_1600.jsonl   坐标域对比对全集（3200样本，规模阶梯用）
  train/train_random/eval_*  坐标域 pilot 与 OOD 评测集
kaggle/
  train_cell.py       pilot 实验 cell（v2 run：坐标域）
  train_cell_w1.py    W1 实验 cell（v4 run：链式 E1 + 歧义 E5）
```

全部标签由规则引擎程序化生成（零噪声），且通过**独立验证器闭环**：
从渲染文本重新解析世界状态、独立重算答案、与标签比对——当前全部数据 **0 错误**。

## 已有真实结果（Kaggle 可复现，notebook: kev-spatial-pilot）

E1 直答能力边界（kev-0.8b，600 样本微调，只训 k≤2）：

| k | 基线 | 微调后 |
|---|------|--------|
| 1 | 0.99 | 1.00 |
| 2 | 0.15 | **0.99** |
| 3 | 0.41 | **0.23**（外推崩塌） |
| 4 | 0.30 | 0.45 |

E5 歧义校准探针：微调使歧义集 ECE 从 0.15 恶化到 0.42（训练分布无歧义 → 丢弃合理不确定性）。

## 复现

```bash
# 数据生成与验证（纯标准库，本地零成本）
python3 gen/make_dataset.py --seed 7      # 坐标域
python3 gen/make_dataset_v2.py --seed 11  # 链式 + 歧义
python3 gen/verify.py data/*.jsonl        # 坐标域验证
python3 gen/verify_v2.py data/chain_*.jsonl data/ambiguous.jsonl

# 训练（Kaggle T4 免费额度可跑）：kaggle/ 目录的 cell 粘贴到
# Kaggle Notebook，挂载数据集后 Save & Run All
```

实验载体：[jaredpalmer/kev](https://github.com/jaredpalmer/kev)（Apache 2.0，Qwen3.5 底座 + LoRA + pointer head）。

## 当前状态（2026-09-24）

- ✅ 数据平台 + E1/E5 初步真实结果（k=3 崩塌至 23%，歧义 ECE 恶化至 0.42）
- ✅ **E2 PMC 概率分解评测器已就绪**（`gen/eval_decompose.py`，支持离散 2D 卷积与置信自适应 Pareto 分析）
- ✅ **AR-FT 歧义微调数据集已就绪**（`gen/make_arft_dataset.py`，生成 `data/chain_train_arft_k12.jsonl`）
- ✅ **W2 Kaggle 实验 Cell 已就绪**（`kaggle/train_cell_w2.py`，支持对比基线、标准微调与 AR-FT）
- ✅ **双 AI 协同与互审工作手册已建立**（`COLLABORATION_GUIDE.md`）
- 🏃 待办：Kaggle 运行 W2 cell 获取 AR-FT 与基线数据 → Pareto 曲线分析 → 撰写初稿投递 Applied Intelligence
