# Jev Paper — 判断模型多跳空间组合推理研究

> System One 判断模型（Jev/Kev 类）的多跳组合推理：能力边界、置信引导分解、校准脆弱性。
> 目标期刊：Applied Intelligence（中科院 3 区）。

## 🔍 给评测/协作 AI 的入口（先读这个）

**[RESEARCH_PLAN.md](RESEARCH_PLAN.md) — 完整研究方案（自包含）**：创新点、实验矩阵 E1–E6、已有真实结果、预注册判定标准、逻辑链、风险登记。

建议评判角度：
1. 创新点（C1/C2/C3）是否站得住，与相关工作表的差异是否成立
2. 逻辑链有无断点或循环论证
3. 实验设计有无致命混杂变量
4. 判定标准是否可证伪
5. "空间链任务"作为测试平台的内部效度与外部效度

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

- ✅ 数据平台 + E1/E5 初步结果
- ⏸️ **研究方案待外部 AI 评审**（通过后解冻写作）
- ⬜ E2 分解实验（评测器待写）→ E3/E4 分析 → 扩容/3 seeds/温度重拟合 → Jev/GLM 基线 → 4B
