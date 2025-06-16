# How to train skip-gram models

The skip-gram models were trained using scripts based on [pungen's](https://github.com/hhexiy/pungen) implementation. These script require that the input data follows a specific preprocessing. This document explains how to download and prepare the data for using training the skip-gram models.

## Where to download the data from

### Chinese Wikipedia (维基百科)

1. Download the dataset using [HuggingFace](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.zh).

```python
from datasets import load_dataset

ds = load_dataset('wikimedia/wikipedia', '20231101.zh')
df = ds['train'].to_pandas()
df.to_csv('wiki-zh.txt', sep='\0', header=False, index=False, columns=['text'])
```

2. Convert all the text to simplified Chinese using [OpenCC](https://github.com/BYVoid/OpenCC).

```bash
opencc -i wiki-zh.txt -o wiki-zh-simplified.txt -c t2s.json
```
3. Tokenize the text using [Jieba](https://github.com/fxsjy/jieba) through candlewill's `tokenization.py` script. Remember to edit the script to use the correct input and output files.

```python
...
input = 'wiki-zh-simplified.txt'
output = 'wiki-zh-simplified-seg.txt'
...
```

### Bookcorpus

1. Download the dataset from [here](https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2).
2. Concatenate the two parts with `cat books_large_p*.txt > bookcorpus.txt`

### CroissantLLM Dataset

1. Download using HuggingFace's datasets library.

```python
from datasets import load_dataset

ds = load_dataset('croissantllm/croissant_dataset', data_files='french_303b_1/*/*.arrow')
df = ds['train'].to_pandas()

with open('croissant.txt', 'w', encoding='utf-8') as f:
    for text in df['text']:
        f.write(f'{text}\n')
```

### BRWaC

1. Request access through the corpus' official [website](https://www.inf.ufrgs.br/pln/wiki/index.php?title=BrWaC).
2. Download the corpus and extract the files.
3. We use the CoNLL-U format, so we can take advantage of the text being already tokenized.
4. Use our `scripts/preprocessing/brwac_conll.sh` script to convert the CoNLL-U format to a simple text file. Make sure to edit the script to use the correct input and output files.

```bash
corpus_file="brwac.conll"
out_file="brwac.txt"
```

### SBWC

1. Download the clean corpus from [here](https://github.com/crscardellino/sbwce). Make sure that you have the clean version, with one sentence per line.

## How to prepare the data

With the previous steps, you should have one file for each dataset, with one sentence per line. All files should be in the `data` directory. The next steps will convert the text to the format required by pungen.

### Tokenize the texts

For each dataset, we need to tokenize the texts using spacy with the `scripts/preprocessing/large_corpora.py` script. This script will save the tokenized texts in the `data/skipgram` directory using the Parquet format for efficiency.

```bash
python scripts/preprocessing/large_corpora.py --input {corpus name}.txt \
                                               --model {spacy model name}
```

Make sure to use the corresponding spacy model for each language:

- Chinese: `zh_core_web_sm`
- English: `en_core_web_sm`
- French: `fr_core_news_sm`
- Portuguese: `pt_core_news_sm`
- Spanish: `es_core_news_sm`

If the dataset is already tokenized (such as BRWAC or Bookcorpus), you can skip this step with the `--no-spacy` flag.

## How to train the models

First, you need to install [pungen](https://github.com/hhexiy/pungen). We recommend doing this in a separate python environment, as it requires installing an older version of python and its libraries. The environment we used contains the following packages:

- Python 3.6.13
- Fairseq 0.6.0
- Numpy 1.19.5
- PyTorch 1.10.1 with CUDA 11.3
- Spacy 2.2.0
- Tqdm 4.64.1

To train the models, you need to first preprocess the data using the following command.

```bash
python -m pungen.wordvec.preprocess --data-dir data/{corpus name}/skipgram \
       --corpus {output file from the previous step} \
       --min-dist 5 --max-dist 10 --threshold 80 \
       --vocab data/{corpus name}/skipgram/dict.txt
```

After preprocessing the data, you can train the model using the following command.

```bash
python -m pungen.wordvec.train --weights --cuda --data data/{corpus name}/skipgram/train.bin \
    --save_dir models/{corpus name}/skipgram \
    --mb 3500 --epoch 15 \
    --vocab data/{corpus name}/skipgram/dict.txt
```

Remember to replace `{corpus name}` with the name of the dataset you are using and `{output file from the previous step}` with the file generated in the previous step.
