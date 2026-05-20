# 模型与数据格式

voxlink 的工件都是纯文本 JSON，便于版本控制与跨平台复用。

## 数据 manifest（JSONL）

每行一条样本：

```json
{"audio": "data/a.wav", "text": "one two three"}
{"audio": "data/b.wav", "text": "yes no"}
```

- `audio`：WAV 文件路径（相对或绝对）。
- `text`：参考文本。
- 其余字段会被读入 `Utterance.meta`（例如问答任务的 `prompt`）。

## 保存的模型目录

`SpeechLMPipeline.save(dir)` 会写出四个文件：

| 文件 | 内容 |
|------|------|
| `frontend.json` | 前端配置（`FrontendConfig`） |
| `codebook.json` | k-means 码本与 `UnitConfig` |
| `vocab.json` | 联合词表的 token 列表 |
| `lm.json` | n-gram 语言模型（配置 + 计数表） |

用 `SpeechLMPipeline.from_pretrained(dir)` 原样恢复。

## 码本文件

```json
{
  "config": {"n_units": 24, "max_iters": 50, "tol": 1e-4, "seed": 0, "dedup": true},
  "centers": [[...], [...], ...]
}
```

`centers` 是 `n_units × 特征维度` 的浮点矩阵。

## 语言模型文件

```json
{
  "vocab_size": 41,
  "config": {"order": 4, "smoothing": 0.001, "seed": 0},
  "counts": [ {"": {...}}, {"12": {...}}, ... ]
}
```

`counts[n]` 是第 n 阶的计数表，key 为空格连接的上下文，value 为 `{下一个 token 的
id: 计数}`。空上下文的 key 为 `""`。
