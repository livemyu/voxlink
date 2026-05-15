# 评估方法：WER 与 CER

voxlink 用两种标准指标衡量语音到文本的质量。

## 词错误率（WER）

$$\text{WER} = \frac{S + D + I}{N}$$

其中 $S/D/I$ 分别是替换、删除、插入的词数，$N$ 是参考文本的词数。voxlink 通过
**词级编辑距离**（Levenshtein）计算 $S+D+I$。

- 对齐前先做文本归一化：NFKC、转小写、去标点、折叠空白（见 `voxlink.text`）。
- 参考为空时：假设也为空则 WER=0，否则 WER=1。

## 字错误率（CER）

与 WER 相同，但在**字符级**计算。对中文等无显式分词的语言，CER 通常比 WER 更
合适。归一化时忽略空白字符。

## 语料级聚合

`corpus_metrics` 不是对每条样本的 WER 取平均，而是按**总编辑数 / 总长度**聚合：

$$\text{WER}_\text{corpus} = \frac{\sum_i (S_i + D_i + I_i)}{\sum_i N_i}$$

这样长句不会被短句稀释，更贴近工业界的报告方式。返回的 `CorpusMetrics` 同时给
出 `wer`、`cer`、`num_utts`、`ref_words`、`ref_chars`。

## 注意

- 这些指标只衡量文本层面的差异，不理解语义；
- 归一化策略会影响结果，跨项目比较时应对齐归一化方式。
