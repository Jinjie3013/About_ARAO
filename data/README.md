# PolitiFact 数据

`politifact.jsonl`：1,056 条标好类的 PolitiFact 标题，采用论文实验所用的划分。

## 字段

| 字段 | 说明 |
| --- | --- |
| `id` | 由 split 和行号生成（`politifact-<split>-<index>`），原始数据无 ID 列 |
| `text` | 新闻标题（原始数据只提供 `title`，没有正文） |
| `label` | **`fake_news = 0`，`real_news = 1`** |
| `split` | `train` / `dev` / `test` |

标签方向沿用原实验管线（`{'real': 1, 'fake': 0}`）。ROC-AUC 与 ASR 对这个编码是对称的——改分类别编号只会交换打分列，不会改变指标值——但混用两套编码会让「P(y=1)」的含义相反，务必统一。

## 划分与统计

7 : 1.5 : 1.5，按标签分层，`random_state = 42`。

| split | 合计 | real (1) | fake (0) |
| --- | --- | --- | --- |
| train | 739 | 437 | 302 |
| dev | 158 | 93 | 65 |
| test | 159 | 94 | 65 |
| **总计** | **1,056** | **624** | **432** |

标题长度：最短 10、中位 59、最长 340 字符。

## 用法

仓库的 CLI 需要按 split 分开的文件。把 JSONL 拆成三个 JSONL：

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

然后：

```bash
python -m arao train --mode vanilla --train data/politifact/train.jsonl --dev data/politifact/dev.jsonl --output runs/vanilla_s42
python -m arao attack --checkpoint runs/vanilla_s42/best --data data/politifact/test.jsonl --output runs/attack_s42
```

## 注意：跨 split 的重复标题

原始标题里有 25 条会同时出现在多个 split 中，绝大多数是采集噪声而非真实重复新闻，例如 `Transcripts`、`American news is under construction`、`LexisNexis(R) Publisher`。

本仓库默认会拒绝存在 train/dev 文本重叠的数据，以免静默泄漏。要直接使用论文的原始划分，需要显式放开：

```bash
python -m arao train --allow-overlap --mode vanilla --train ... --dev ... --output runs/vanilla_s42
```

如果希望严格无重叠，可以先按规范化标题去重（会剩下 983 条），再重新划分。

## 来源与许可

数据来自 FakeNewsNet 的 PolitiFact 部分（Shu et al., *FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media*, Big Data 2020）。这里只包含标题文本，不含正文、URL 与社交上下文。

该数据受原作者的使用条款约束，请遵循 FakeNewsNet 的许可与引用要求；本仓库仅出于复现目的转载这一份标注子集。若原作者要求，将移除本文件。
