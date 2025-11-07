# Humetrix

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/license/mit)

Automatic metrics for evaluation of puns.

## Description

Humetrix is a Python library that provides a collection of automatic metrics for evaluating humor in text, focusing on puns. It includes implementations of several state-of-the-art humor metrics, as well as tools for training your own models.

## Installation

To install Humetrix, first make sure you are using python 3.12 or above.

We recommend using [uv](https://docs.astral.sh/uv/) as your environment manager:

```bash
uv sync
```

If you prefer using `pip` directly, run:

```bash
pip install -r requirements.txt
pip install .
```

## Usage

Here is a simple example of how to use Humetrix:

```python
from humetrix import HumorAnalyzer

# You need to provide further resources if required by the wanted metric
analyzer = HumorAnalyzer(language='en',
                         embeddings_path='data/embeddings/en/glove_s300.gensim', # Gensim-formated word embeddings
                         skipgram_path='results/skipgram/en.pt', # Distant Skip-gram model (check docs/skipgram.md)
                         ngram_dir_path='data/ngrams/en', # Google n-gram frequencies (1-gram and 3-grams)
                         transformer_model_name='FacebookAI/xlm-roberta-base') # HuggingFace transformer model
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## How to cite

```bibtex
To be announced
```
