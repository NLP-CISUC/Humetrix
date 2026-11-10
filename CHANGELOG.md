# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2025-11-04

### Added

- Initial release of Humetrix.

## 0.3.0 - 2026-11-10

### Changed
- spacy models are no longer installed as package dependencies; they are downloaded automatically on first use.
- Quantum Entropy and Surprise Ratio now use lightweight tokenizer pipelines (`zh_core_web_sm` for Chinese and `spacy.blank` for other languages) instead of full spacy models. Tokenization and previous scores are unchanged.
- Replaced deprecated `polars-u64-idx` with `polars[rt64]>=1.34`.
- **Breaking:** `embeddings_path` now expects a directory with `embeddings.safetensors` and `vocab.txt`; gensim files are no longer supported.
- **Breaking:** `skipgram_path` now expects safetensors weights, and must be given together with `skipgram_vocab_path`.
- Quantum Entropy metrics return `None` for texts that do not split into exactly two segments (setup and punchline), instead of throwing an error.

### Added
- `numpy` and `scipy` as explicit dependencies.
- GloVe embeddings, skip-gram models and n-gram tables are downloaded automatically from the Hugging Face Hub when no path is given.
- `huggingface-hub` and `safetensors` as explicit dependencies.
- `device` parameter in `HumorAnalyzer` to choose where neural models run.
- Added `KaoDistinctiveness` to package's `__init__.py`.

### Removed
- Analysis and plotting packages (`marimo`, `seaborn`, `matplotlib`, `altair`, `pyspark`, `datasets`, etc.) from the package requirements.
- Unused `spacy[cuda12x]` extra.

### Fixed
- Local-Global Surprise failed with recent `transformers` versions.
- Spacy's default sentencizer punctuation list is no longer modified globally.
- Local-Global Surprise no longer disables pytorch gradients globally.
- Skipgram emdedding dimension is read from the weights instead of being fixed at 300.
