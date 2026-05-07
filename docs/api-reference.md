# API 参考

本页列出 voxlink 的主要公开接口。完整签名请查阅源码 docstring。

## 顶层

```python
import voxlink
voxlink.__version__
```

`from voxlink import ...` 直接可得：`SpeechLMPipeline`、`FrontendConfig`、
`UnitConfig`、`ModelConfig`、`Waveform`、`FeatureMatrix`、`Utterance`，以及异常类
`VoxlinkError` / `AudioError` / `FeatureError` / `ConfigError` / `ModelError`。

## 配置 `voxlink.config`

- `FrontendConfig(sample_rate=16000, frame_length_ms=25, frame_shift_ms=10, n_fft=512, n_mels=80, n_mfcc=13, fmin=20, fmax=8000, preemphasis=0.97)`
- `UnitConfig(n_units=100, max_iters=50, tol=1e-4, seed=0, dedup=True)`
- `ModelConfig(order=3, smoothing=0.1, seed=0)`

非法参数会抛 `ConfigError`。

## 特征 `voxlink.features`

- `Frontend(config=None, *, feature="mfcc", normalize=True)` — 可调用，`__call__(wave) -> FeatureMatrix`
- `power_spectrogram(wave, config) -> Matrix`
- `log_mel_spectrogram(wave, config) -> Matrix`
- `mfcc(wave, config) -> Matrix`
- `cmvn(feats, *, variance=False) -> Matrix`

## 离散单元 `voxlink.units`

- `Quantizer(config=None)`：`.fit(frames)`、`.encode(feats) -> list[int]`、
  `.save(path)`、`.load(path)`
- `dedup_units(units) -> list[int]`

## 语言模型 `voxlink.ngram`

- `NgramLM(vocab_size, config=None)`：`.fit(sequences)`、`.logprobs(prefix)`、
  `.score(seq)`、`.sample_next(prefix, rng)`、`.save/.load`

## 解码 `voxlink.decode`

- `greedy_decode(model, prompt, max_new_tokens, eos) -> list[int]`
- `sample_decode(model, prompt, max_new_tokens, eos, *, rng, temperature=1.0, top_k=0)`
- `beam_search(model, prompt, max_new_tokens, eos, *, beam_size=4, length_penalty=0.0)`

## 评估 `voxlink.eval`

- `wer(ref, hyp) -> float`、`cer(ref, hyp) -> float`
- `levenshtein(a, b) -> int`
- `corpus_metrics(refs, hyps) -> CorpusMetrics`

## 管线 `voxlink.pipeline`

- `SpeechLMPipeline(frontend_config=None, unit_config=None, model_config=None)`
  - `.fit(utterances, *, task=ASR)`
  - `.transcribe(wave, *, max_new_tokens=32, beam_size=1)`
  - `.answer(wave, question, ...)`
  - `.follow(wave, instruction, ...)`
  - `.encode_units(wave) -> list[int]`
  - `.save(dir)` / `SpeechLMPipeline.from_pretrained(dir)`
