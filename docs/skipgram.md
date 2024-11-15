# How to run skip-gram models

## Bookcorpus

1. Download the dataset from [here](https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2).
2. Concatenate the two parts with `cat bookcorpus_large_p*.txt > bookcorpus.txt`

## CroissantLLM Dataset

1. Download using HuggingFace's datasets library.

```python
from datasets import load_dataset

ds = load_dataset('croissantllm/croissant_dataset', data_files='french_303b_1/*/*.arrow')
df = ds['train'].to_pandas()

with open('data/croissant.txt', 'w', encoding='utf-8') as f:
    for text in df['text']:
    	f.write(f'{text}\n')
```
