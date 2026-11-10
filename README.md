# Humetrix

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/license/mit)

Automatic metrics for humorous texts. Currently supporting English, Spanish, French, Portuguese and Mandarin Chinese.

| Metric | Method | Reference |
|---|---|---|
| Quantum Uncertainty / Incongruity | `quantum_uncertainty`, `quantum_incongruity` | [Liu and Hou (2023)](https://aclanthology.org/2023.findings-eacl.152) |
| Surprise Ratio | `local_global_surprise` | [He et al. (2019)](http://aclweb.org/anthology/N19-1172) |
| Ambiguity / Distinctiveness | `kao_ambiguity`, `kao_distinctiveness` | [Kao et al. (2016)](https://onlinelibrary.wiley.com/doi/10.1111/cogs.12269) |

## Description

Humetrix is a Python library that provides a collection of automatic metrics for evaluating humor in text. Quantum Entropy metrics apply to any text with a setup and a punchline, while Surprise Ratio and Ambiguity/Distinctiveness are designed for puns and need the pun and alternative words.

All resources are downloaded automatically on first use, but the repository also contains the scripts used to train our models (see [docs/skipgram.md](docs/skipgram.md))

## Installation

Humetrix requires Python 3.12 or above:

```bash
pip install humetrix
```

PyTorch is installed from PyPI by default. For a CPU-only install, install PyTorch first following the [official instructions](https://pytorch.org/get-started/locally/).

## Usage

Here is a simple example of how to use Humetrix:

```python
from humetrix import HumorAnalyzer

# Needed resources are downloaded automatically on first use
analyzer = HumorAnalyzer(language='en')
sent = 'Not all math puns are bad. Just sum of them.'
sign = 'sum'
alt_sign = 'some'

qi_score = analyzer.quantum_incongruity(sent)
qi_xlm_score = analyzer.quantum_incongruity(sent, backend='transformer')
qu_score = analyzer.quantum_uncertainty(sent)
qu_xlm_score = analyzer.quantum_uncertainty(sent, backend='transformer')
lgs_score = analyzer.local_global_surprise(sent, sign, alt_sign)
kao_amb_score = analyzer.kao_ambiguity(sent, sign, alt_sign)
kao_dist_score = analyzer.kao_distinctiveness(sent, sign, alt_sign)
```

Metrics return `None` when they cannot be computed.

Use `HumorAnalyzer(<language>, device="cpu")` too choose where neural models run, and `transformer_model_name` to choose the model for the `transformer` backend.

## Resources

Embeddings, skip-gram models, n-gram tables, spaCy and transformer models are downloaded on first use and cached locally, so the first call of each metric may take a while. They are hosted at [NLP-CISUC/humetrix-resources](https://huggingface.co/NLP-CISUC/humetrix-resources).

## Development

To reproduce the experiments of the paper, clone the repository and use [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The resources have their own licenses, listed in the [resources repository](https://huggingface.co/NLP-CISUC/humetrix-resources).

## How to cite

```bibtex
To be announced
```
