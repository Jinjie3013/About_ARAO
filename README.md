# ARAO 核心实现

Adaptive Rhetorical Adversarial Optimization（ARAO）的可复现核心实现：在轴对齐的修辞策略空间上做无梯度搜索，并用语义门控保证改写不与原文矛盾，从而支持「发现 → 诊断 → 定向修复」的闭环。

本仓库包含方法与流程的核心代码，以及论文实验所用的 PolitiFact 标注子集（`data/politifact.jsonl`）。

## 安装

建议使用独立的 Python 3.10+ 环境。`requirements.txt` 中的 PyTorch 与 Transformers 版本对应论文的复现环境；CUDA wheel 请按本机环境安装。

```bash
python -m pip install -r requirements.txt
```

## 数据

仓库附带一份 PolitiFact 标注子集 `data/politifact.jsonl`：1,056 条标题。

其他数据集请自备，支持两种格式：

- CSV：`text,label` 两列；若没有 `text` 列，会回退读取 `title`。
- JSONL：每行 `{"id": "...", "text": "...", "label": 0}`。
- 标签必须是从 0 开始的整数类别 ID。可用全局参数 `--text-column`、`--label-column` 指定其他列名。

CLI 需要按 split 分开的文件，把附带的数据拆开：

```bash
python - <<'PY'
import json, pathlib
rows = [json.loads(l) for l in open("data/politifact.jsonl", encoding="utf-8")]
out = pathlib.Path("data/politifact"); out.mkdir(exist_ok=True)
for split in ("train", "dev", "test"):
    with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            if r["split"] == split:
                f.write(json.dumps({k: r[k] for k in ("id", "text", "label")}, ensure_ascii=False) + "\n")
PY
```

训练入口只接收 train/dev，checkpoint 按 dev ROC-AUC 选择；缺少数据或无效标签会直接报错。**train/dev 文本重叠默认也会报错**——官方 PolitiFact 划分里有 25 条标题跨 split 重复（多为 `Transcripts` 这类采集噪声），要直接复现该划分需显式放开：

```bash
python -m arao train --allow-overlap --mode vanilla --train data/politifact/train.jsonl --dev data/politifact/dev.jsonl --output runs/vanilla_s42
```

若要严格无重叠，可先按规范化标题去重（剩 983 条）再重新划分。`scripts/prepare_politifact.py` 提供了这样的示例（去重 + 分层），输入是 FakeNewsNet 的 `politifact_real.csv` / `politifact_fake.csv`：

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

## 配置

| 文件 | 用途 |
| --- | --- |
| `configs/politifact.json` | **默认**：`N=4, T=4`（B=16）的快速配置，便于先跑通管线 |
| `configs/politifact_paper.json` | 论文口径：`N=16, T=8`（B=128），其余超参对应 Table 10 |
| `configs/*_local.json` | 本机替代模型（不同 LLM / NLI checkpoint）的覆盖配置，已在 `.gitignore` 中排除 |

候选预算 `B = population_size × iterations`，每个候选需要一次 Planner 和一次 Rewriter 调用，因此单样本的 LLM 调用数为 `2B`。**考虑到费用，建议 B=16 **，需要更强攻击时再上调 `iterations` 或 `population_size`。

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
