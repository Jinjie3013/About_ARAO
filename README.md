# ARAO 核心实现

Adaptive Rhetorical Adversarial Optimization（ARAO）的可复现核心实现：在轴对齐的修辞策略空间上做无梯度搜索，并用语义门控保证改写不与原文矛盾，从而支持「发现 → 诊断 → 定向修复」的闭环。

本仓库包含方法与流程的核心代码，以及论文实验所用的 PolitiFact 标注子集（`data/politifact.jsonl`）。训练好的模型不入库，请按 `runs/` 流程自行生成。

## 安装

建议使用独立的 Python 3.10+ 环境。`requirements.txt` 中的 PyTorch 与 Transformers 版本对应论文的复现环境；CUDA wheel 请按本机环境安装。

```bash
python -m pip install -r requirements.txt
```

## 数据

仓库附带一份 PolitiFact 标注子集 `data/politifact.jsonl`：1,056 条标题，采用论文实验的 7:1.5:1.5 分层划分（train 739 / dev 158 / test 159）。字段为 `id`、`text`、`label`、`split`，**标签编码 `fake_news = 0`、`real_news = 1`**。来源、统计与许可见 [data/README.md](data/README.md)。

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

请遵守各数据集的原始许可。

## 快速开始

进行 LLM 生成前，设置接口凭据；使用兼容服务时可额外指定 `OPENAI_BASE_URL`：

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
| `configs/politifact_paper.json` | 论文口径：`N=16, T=8`（B=128），其余超参对应 Table 10 |
| `configs/*_local.json` | 本机替代模型（不同 LLM / NLI checkpoint）的覆盖配置，已在 `.gitignore` 中排除 |

候选预算 `B = population_size × iterations`，每个候选需要一次 Planner 和一次 Rewriter 调用，因此单样本的 LLM 调用数为 `2B`。B=16 时约 32 次调用/样本；论文的 B=128 是 256 次。**考虑到费用，日常验证建议从 B=16 这一档起**，需要更强攻击时再上调 `iterations` 或 `population_size`。

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

核心组件通过 Protocol 交互。替换 LLM、NLI 或检测器时，实现对应接口即可；搜索循环不需要知道 API、分词器、张量或优化器。

## 核心行为

1. 为每个输入初始化一个高斯搜索分布，采样并裁剪到 `[-1,1]^k`。
2. 通过两个独立 LLM 阶段生成 guideline 和改写。
3. 计算 `G = 1 - max(p_con(x,x'), p_con(x',x))`，默认阈值 `0.85`。
4. **只有过门控的候选**计算 `-log p(y|x')`，并参与 NES 更新。
5. 按 Algorithm 1 同时更新均值和完整协方差，两个更新均使用旧均值。
6. 在线训练时，每轮将**全部过门控改写**传入一次检测器训练回调。
7. 固定检测器评估时，选取预算内已评估、目标值最大的有效候选；全部被拒绝则保留原文。

## 控制器归一化（重要）

Algorithm 1 写了「centered rank-based utilities（RANKNORM）」，但**没有给出归一化常数**。而这个常数正是 `mean_lr` 的单位：同一个 `η=0.03` 配上不同归一化，实际步长会差一个数量级。

本实现默认 `search.rank_norm = "unit"`（`u = rank/(N-1) - 0.5`，常见 ES/NES 约定），使论文给的 `η=0.03` 真正能移动分布中心。用 `scripts/controller_probe.py` 可以在**不调用任何 LLM** 的情况下量化这件事：

```bash
python scripts/controller_probe.py
```

```
population=16 iterations=8 budget=128 seeds=40 (sampling sigma=0.5, box half-width=1.0)
norm    mean_lr  ||mu_T||     gain  saturated  sigma_T
------------------------------------------------------
unit       0.03     0.509    0.439      0.00%    0.494
l1         0.03     0.119    0.136      0.00%    0.500
raw        0.03     3.431    1.867     82.50%    0.454
```

- `||mu_T||` 是搜索中心走过的距离（0 = 控制器空转，1.0 = 半个搜索盒）。
- `gain` 是最后一轮与第一轮的平均目标值之差，正值表示分布确实往高目标区域移动了。
- `l1`（`sum(|u|)=1`）会把有效步长压缩到 `unit` 的 1/4.27（N=16），控制器几乎不动。
- `raw` 会在几轮内撞到盒角并让协方差膨胀，只保留作消融，不是默认值。

漂移随预算增长：`unit` 在 B=128 时约 1σ，在 B=32 时只有约 0.15σ。若要更强的自适应性，应提高 `population_size` / `iterations`，而不是盲目放大 `mean_lr`。

每轮漂移都会写进结果的 `control_trace`，因此「控制器是否真的在工作」是可观测、可复算的，而不是只能靠描述。

## 评估与诊断细节

- 二分类 AUC 使用 `P(y=1)`；三分类使用 macro one-vs-rest ROC-AUC。若样本缺少必要类别，AUC 为 `null`，不会用 0.5 替代。
- ASR 的分母是原始文本被正确分类的样本数；成功要求这些样本在最终攻击文本上被误分类。
- 轴对齐使用严格成功样本的 `abs(s[d]) / ||s||`，相对中性原点计算。null 为每条成功搜索轨迹均匀抽取一个过门控候选，重复 50 次，分别报告 MAA、null MAA、差值。原始对齐是描述性量，不直接解释为因果归因。
- 单轴扫描的 `t=0` 仍调用 Planner-Rewriter。扫描只对过门控改写计算改写 AUC；同时报告「完整原集 AUC − 有效改写 AUC」和同一保留子集上的配对差值，以显示门控覆盖变化。
- 测试集只用于最终冻结检测器的评估与研究性诊断。实际用于部署的轴选择应来自 train/dev；修复改写只从训练集生成。
- 极端言论任务可将 `run.domain` 改为 `extremist`，并按任务设置 `models.num_labels`；主流程不变。当前提示词面向英文。

## 输出

`run.json` 保存有效配置、种子、依赖版本和输入文件 SHA256。

`summary.json` 保存指标及已报告的 LLM token 用量。

攻击还会逐样本写 `results.jsonl`：原始概率、每个候选的策略、guideline、文本、门控分数、有效候选目标值，以及 `control_trace`。每一行在完成时立即落盘，中断不会丢失已完成样本。

`control_trace` 每个元素对应 Algorithm 1 的一轮：`iteration`、`accepted`（该轮过门控候选数）、`max_abs_utility`、`mean_step`、`mean_norm`（`||mu_t||`）、`cov_trace`（`trace(Sigma_t)`）、`cov_min_eig`。

## 论文未完全指定的实现选择

以下值是**本实现选择**，均可在 JSON 中修改：

| 项目 | 本实现选择 |
| --- | --- |
| NLI checkpoint | 默认 `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`；论文未给出具体检查点 |
| NLI contradiction 标签 | 从 `id2label` 读取；通用 `LABEL_0/1/2` 必须显式配置下标 |
| 初始搜索分布 | `mu=0, Sigma=0.25*I`，即 `initial_std=0.5` |
| 排名效用归一化 | `rank_norm="unit"`；`l1` 与 `raw` 见「控制器归一化」一节 |
| 协方差稳定 | 对称化，并将特征值下限设为 `1e-6` |
| 在线训练调度 | 每个原始 batch 先做一次 clean 更新，再对选中样本执行在线搜索和逐轮有效改写更新 |
| 训练攻击覆盖 | `attack_fraction`；小比例时按每个 batch 向上取整，0 表示只做 clean 更新 |
| 定向修复强度 | `0.6`；增广数量定义为追加数量/原始训练数量，整数向上取整 |
| 其他训练值 | AdamW weight decay `0.01`，梯度裁剪 `1.0`，无额外 scheduler |
| 截断与设备 | 检测器单文本、NLI 文本对都截断至 512 tokens；默认 NLI 放 CPU，检测器自动选择 GPU/CPU |
| reasoning 类 LLM | 若 `llm.model` 是 reasoning 模型，`max_tokens` 会同时覆盖思考与回答，且思考模式下 `temperature`/`top_p` 不生效。可用 `llm.disable_thinking` 关闭思考以降低延迟与成本，但这会改变生成的改写文本，需要在结果中标注 |

门控实现遵循非矛盾边界，**不保证事实或标签完全保持**。温度 0 与固定本地种子也不能保证远程 LLM 在不同模型版本之间逐字一致。多随机种子实验可分别运行 `--seed 42/43/44`，再汇总独立运行结果。

## 验证

```bash
python -m unittest discover -s tests -v
```

核心测试使用本地测试替身；模型适配器测试临时构造极小的随机 BERT，不下载模型、不调用 LLM。覆盖双向门控、拒绝候选隔离、预算、最佳候选选择、全部有效候选训练、完整协方差更新、排名并列、归一化尺度、控制器漂移、重叠接受开关、定向修复配额、数据校验、指标分母、序列化、NLI 标签映射和微批次平均损失。

## 参考

- Hugging Face [Auto Classes](https://huggingface.co/docs/transformers/model_doc/auto)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
