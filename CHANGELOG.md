# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2025-11-04

### Added

- Initial release of Humetrix.

## [Unreleased]

### Changed
- spacy models are no longer installed as package dependencies; they are downloaded automatically on first use.
- Quantum Entropy and Surprise Ratio now use lightweight tokenizer pipelines (`zh_core_web_sm` for Chinese and `spacy.blank` for other languages) instead of full spacy models. Tokenization and previous scores are unchanged.
- Replaced deprecated `polars-u64-idx` with `polars[rt64]>=1.34`.

### Added
- `numpy` and `scipy` as explicit dependencies.

### Removed
- Analysis and plotting packages (`marimo`, `seaborn`, `matplotlib`, `altair`, `pyspark`, `datasets`, etc.) from the package requirements.
- Unused `spacy[cuda12x]` extra.
