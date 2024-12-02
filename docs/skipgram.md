# How to train skip-gram models

The skip-gram models were trained using [pungen's](https://github.com/hhexiy/pungen) scripts. These script require that the input data follows a specific format. This document explains how to download and prepare the data for using pungen.

## Where to download the data from

### Chinese Wikipedia (维基百科)

1. Download the dataset from [here](https://dumps.wikimedia.org/zhwiki/20241001/zhwiki-20241001-pages-articles-multistream.xml.bz2).
2. Use [candlewill's](https://github.com/candlewill/Chinsese_word_vectors) `process_wiki.py` script to extract the text from the wiki dump. Remember to edit the script to use the correct input and output files.

```python
...
inp, outp = 'zhwiki-20241001-pages-articles-multistream.xml.bz2', 'wiki-zh.txt'
...
```

3. Convert all the text to simplified Chinese using [OpenCC](https://github.com/BYVoid/OpenCC).

```bash
opencc -i wiki-zh.txt -o wiki-zh-simplified.txt -c t2s.json
```
4. Tokenize the text using [Jieba](https://github.com/fxsjy/jieba) through candlewill's `tokenization.py` script. Remember to edit the script to use the correct input and output files.

```python
...
input = 'wiki-zh-simplified.txt'
output = 'wiki-zh-simplified-seg.txt'
...
```

### Bookcorpus

1. Download the dataset from [here](https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2).
2. Concatenate the two parts with `cat bookcorpus_large_p*.txt > bookcorpus.txt`

### CroissantLLM Dataset

1. Download using HuggingFace's datasets library.

```python
from datasets import load_dataset

ds = load_dataset('croissantllm/croissant_dataset', data_files='french_303b_1/*/*.arrow')
df = ds['train'].to_pandas()

with open('data/croissant.txt', 'w', encoding='utf-8') as f:
    for text in df['text']:
        f.write(f'{text}\n')
```

### BRWaC

1. Request access through the corpus' official [website](https://www.inf.ufrgs.br/pln/wiki/index.php?title=BrWaC).
2. Download the corpus and extract the files.
3. We use the CoNLL-U format, so we can take advantage of the text being already tokenized.

### SBWC

1. Download the clean corpus from [here](https://github.com/crscardellino/sbwce). Make sure that you have the clean version, with one sentence per line.

## How to prepare the data

With the previous steps, you should have one file for each dataset, with one sentence per line. All files should be in the `data` directory. The next steps will convert the text to the format required by pungen.

In sum, all datasets, except for BRWaC, require using the `scripts/convert_skipgram/convert_txt.py` script. The BRWaC dataset requires using the `scripts/convert_skipgram/brwac.sh` script.

### Chinese Wikipedia (维基百科)

Run the following command to prepare the data for training.

```bash
python scripts/convert_skipgram/convert_txt.py --input data/wiki-zh-simplified-seg.txt \
                                               --model zh_core_web_sm
```

### Bookcorpus

Run the following command to prepare the data for training.

```bash
python scripts/convert_skipgram/convert_txt.py --input data/bookcorpus.txt \
                                               --model en_core_web_sm
```

### CroissantLLM Dataset

Run the following command to prepare the data for training.

```bash
python scripts/convert_skipgram/convert_txt.py --input data/croissant.txt \
                                               --model fr_core_news_sm
```

### BRWaC

Run the following command to prepare the data for training.

```bash
./scripts/convert_skipgram/brwac.sh
```

### SBWC

Run the following command to prepare the data for training.

```bash
python scripts/convert_skipgram/convert_txt.py --input data/sbwc.txt \
                                               --model es_core_news_sm
```

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
