# ARAO 核心实现

Adaptive Rhetorical Adversarial Optimization（ARAO）的可复现核心实现：在轴对齐的修辞策略空间上做无梯度搜索，并用语义门控保证改写不与原文矛盾，从而支持「发现 → 诊断 → 定向修复」的闭环。

本仓库包含方法与流程的核心代码。

## 安装

建议使用独立的 Python 3.10+ 环境。`requirements.txt` 中的 PyTorch 与 Transformers 版本对应论文的复现环境；。

```bash
python -m pip install -r requirements.txt
```

## 数据

仓库附带了一个Politifact数据集。需要的格式：

- CSV：`text,label` 两列；若没有 `text` 列，会回退读取 `title`。
- JSONL：每行 `{"id": "...", "text": "...", "label": 0}`。
- 标签必须是从 0 开始的整数类别 ID。可用全局参数 `--text-column`、`--label-column` 指定其他列名。

准备独立的 `data/train.csv`、`data/dev.csv`、`data/test.csv`。训练入口只接收 train/dev，checkpoint 按 dev ROC-AUC 选择；缺少数据、无效标签或 train/dev 文本重叠都会直接报错。

`scripts/prepare_politifact.py` 是一个可参考的预处理示例，输入是 FakeNewsNet 的 `politifact_real.csv` / `politifact_fake.csv`：

```bash
python scripts/prepare_politifact.py --real politifact_real.csv --fake politifact_fake.csv --output data/politifact
```


## 快速开始

```bash
export OPENAI_API_KEY=...
# export OPENAI_BASE_URL=https://.../v1
```

以下命令的输出目录必须是**新目录**。`--config`、`--seed` 等全局参数放在子命令之前。

```bash
# 1. Vanilla 检测器：只训练原始数据
python -m arao --seed 42 train --mode vanilla --train data/train.csv --dev data/dev.csv --output runs/vanilla_s42

# 2. 完整在线 ARAO：原始数据更新与 Algorithm 1 交替执行
python -m arao --seed 42 train --mode online --train data/train.csv --dev data/dev.csv --output runs/online_s42

# 3. 对固定 checkpoint 执行 matched ARAO 攻击
python -m arao --seed 42 attack --checkpoint runs/vanilla_s42/best --data data/test.csv --output runs/attack_s42

# 4. 从攻击轨迹计算绝对轴对齐及轨迹匹配 null
python -m arao --seed 42 diagnose --results runs/attack_s42/results.jsonl --output runs/alignment_s42

# 5. 单轴扫描：默认强度 0、0.1、…、0.6
python -m arao --seed 42 probe --checkpoint runs/vanilla_s42/best --data data/test.csv --axes authority scarcity commitment --output runs/probe_s42

# 6. 定向修复：固定增广集、默认增广比例 0.25、额外微调 2 epochs
python -m arao --seed 42 repair --checkpoint runs/vanilla_s42/best --train data/train.csv --dev data/dev.csv --axes authority scarcity --output runs/repair_s42
```

需要先检查连接与流程时，可在 `attack` / `probe` 后加 `--limit 2`。只要显式的小规模探索结果；正式评估请省略该参数。

## 配置

| 文件 | 用途 |
| --- | --- |
| `configs/politifact.json` | **默认**：`N=4, T=4`（B=16）的快速配置，便于先跑通管线 |

候选预算 `B = population_size × iterations`，每个候选需要一次 Planner 和一次 Rewriter 调用，因此单样本的 LLM 调用数为 `2B`。B=16 时约 32 次调用。考虑到费用建议16次调用即可。

## 模块与职责

| 文件 | 单一职责 |
| --- | --- |
| `space.py` | 轴定义、坐标校验、单轴及多轴干预 |
| `generation.py` | 数值策略到 guideline，再到文本（Planner / Rewriter） |
| `gate.py` | 双向非矛盾打分与阈值判断 |
| `objective.py` | 正确类别概率的负对数 |
| `controller.py` | 种群采样、中心化排名、均值与完整协方差更新、漂移记录 |
| `search.py` | 只协调上述组件，支持可选的逐轮训练回调 |
| `diagnosis.py` | 轴对齐、轨迹匹配 null、单轴强度扫描 |
| `training.py` | 原始数据训练、在线对抗训练、固定增广集生成 |
| `adapters/huggingface.py` | 检测器与 NLI 的分词、推理、梯度和权重保存 |
| `adapters/openai_chat.py` | API 通信与 token 统计 |
| `interfaces.py` / `types.py` | 可替换接口与共享数据结构 |
| `config.py` / `factory.py` | 参数管理与依赖装配 |
| `__main__.py` / `data.py` / `metrics.py` | 命令行、数据 I/O、评估 |
