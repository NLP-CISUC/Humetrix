# Organization of the project's data files

The data is organized within the `data/` directory, which is not publicly distributed. The expected organization of the folder is as follows:

```
data/
├── embeddings
│   ├── en
│   │   ├── glove_s300.gensim
│   │   └── glove_s300.gensim.vectors.npy
│   ├── es
│   │   └── ...
│   ├── fr
│   │   └── ...
│   ├── pt
│   │   └── ...
│   └── zh
│       └── ...
├── humor_interpretation
│   ├── humicroedit.json
│   └── puntuguese.json
├── humor_recognition
│   ├── chumor.json
│   ├── clemencio.json
│   ├── HAHA@IberLEF2019.json
│   └── ...
├── large_corpora
│   ├── bookcorpus.txt
│   ├── brwac.txt
│   ├── croissant.txt
│   └── ...
└── ngrams
    ├── en
    │   ├── 1gram.csv
    │   └── 3gram.csv
    ├── es
    │   └── ...
    ├── fr
    │   └── ...
    ├── pt
    │   └── ...
    └── zh
        └── ...
```

The `data/` directory is organized according to the different tasks that the datasets are used for. In the following sections, we detail the datasets used, where to download them from, and how to prepare them.

All commands below should be run from the root directory of the project.

## Humor datasets

We work with several datasets for humor recognition and interpretation, organized in the `data/humor_recognition/` and `data/humor_interpretation/` directories, respectively.

### Humicroedit

Humicroedit can be downloaded from [here](https://www.cs.rochester.edu/u/nhossain/humicroedit.html). In this work, we use the [Full Dataset Release](https://cs.rochester.edu/u/nhossain/semeval-2020-task-7-dataset.zip).

After the dataset is downloaded and decompressed, we use the scripts `utils/conversion/humor_interpretation/humicroedit_to_json.py` and `utils/conversion/humor_recognition/humicroedit_to_json.py` to convert the dataset to our JSON format. To do this, run the following commands:

```bash
# Humor interpretation
python utils/conversion/humor_interpretation/humicroedit_to_json.py \
       -t [PATH_TO_DATASET]/subtask-1/train.csv \
       -d [PATH_TO_DATASET]/subtask-1/dev.csv \
       -s [PATH_TO_DATASET]/subtask-1/test.csv

# Humor recognition
python utils/conversion/humor_recognition/humicroedit_to_json.py \
       -t [PATH_TO_DATASET]/subtask-1/train.csv \
       -d [PATH_TO_DATASET]/subtask-1/dev.csv \
       -s [PATH_TO_DATASET]/subtask-1/test.csv
```

### Puntuguese

This corpus can be downloaded from the following Github page: [https://github.com/Superar/Puntuguese](https://github.com/Superar/Puntuguese) and converted using the `utils/conversion/humor_interpretation/puntuguese_to_json.py` and `utils/conversion/humor_recognition/puntuguese_to_json.py` scripts through the following commands:

```bash
# Humor interpretation
python utils/conversion/humor_interpretation/puntuguese_to_json.py \
       -c [PATH_TO_DATASET]/data/puns.json

# Humor recognition
python utils/conversion/humor_recognition/puntuguese_to_json.py \
       -d [PATH_TO_DATASET]/data/classification_corpus.json
```

Note that the input file that is passed to the scripts is different for the two tasks.

### Clemêncio et al. (2020)

This Portuguese dataset can be downloaded from [the project's Github page](https://github.com/NLP-CISUC/Recognizing-Humor-in-Portuguese). Since this is a pun recognition-only dataset, we only use the `utils/conversion/humor_recognition/clemencio_to_json.py` script to convert the dataset.

```bash
# Humor recognition
python utils/conversion/humor_recognition/clemencio_to_json.py \
       -d [PATH_TO_DATASET]/Datasets/Balanceados/all.txt
```

### HAHA@IberLEF 2019

This dataset can be downloaded from [here](https://www.fing.edu.uy/inco/grupos/pln/haha/2019/index.html#data), with two separete files: [train](https://www.fing.edu.uy/inco/grupos/pln/haha/2019/data/haha_2019_train.csv) and [test](https://www.fing.edu.uy/inco/grupos/pln/haha/2019/data/haha_2019_test_gold.csv) data. We highlight that there are two links for the test data; we use the one with gold annotations. Since this corpus does not contain the pun location task, we only convert it using the `utils/conversion/humor_recognition/haha_to_json.py` script:

```bash
# Humor recognition
python utils/conversion/humor_recognition/haha_to_json.py \
       -t [PATH_TO_DATASET]/haha_2019_train.csv \
       -s [PATH_TO_DATASET]/haha_2019_test_gold.csv
```

### HAHA@IberLEF 2021

The 2021 version of the HAHA corpus can be obtained from their [CodaLab page](https://competitions.codalab.org/competitions/30090) and follows roughly the same format as the 2019 one. Therefore, we use the same script to convert it, but including the newly-introduced validation dataset. The command to convert the dataset is as follows:

```bash
# Humor recognition
python utils/conversion/humor_recognition/haha_to_json.py \
       -t [PATH_TO_DATASET]/haha_2021_train.csv \
       -d [PATH_TO_DATASET]/haha_2021_dev_gold.csv \
       -s [PATH_TO_DATASET]/haha_2021_test_gold.csv
```

### HUHU@IberLEF 2023

The HUHU dataset is unfortunately close-sourced, so we cannot provide an easy way of downloading it. However, if you have access to the dataset, you can convert it using the `utils/conversion/humor_recognition/huhu_to_json.py` script with the following command:

```bash
# Humor recognition
python utils/conversion/humor_recognition/huhu_to_json.py \
       -d [PATH_TO_DATASET]/train.csv
```

### Joker CLEF

The JOKER shared task provided datasets in multiple languages, but unfortunately they are not publicly available as the registration for the task is closed. Nonetheless, if you have access to them, you can convert them using the single `utils/conversion/humor_recognition/joker_to_json.py` script, which already handles all the languages.

```bash
# Humor recognition
python utils/conversion/humor_recognition/joker_to_json.py \
       -d [PATH_TO_DATASET]/Task\ 1\ -\ detection/
```

### SemEval 2017 Task 7

The data for this task can be downloaded from their [official page](https://alt.qcri.org/semeval2017/task7/index.php?id=data-and-resources). Specifically, we use the [trial and test data](https://alt.qcri.org/semeval2017/task7/data/uploads/semeval2017_task7.tar.xz). Then, we convert the data using the `utils/conversion/humor_recognition/semeval_to_json.py` script:

```bash
# Humor recognition
python utils/conversion/humor_recognition/semeval_to_json.py \
       -td [PATH_TO_DATASET]/data/test/subtask1-heterographic-test.xml \
       -tl [PATH_TO_DATASET]/data/test/subtask1-heterographic-test.gold \
       -md [PATH_TO_DATASET]/data/test/subtask1-homographic-test.xml \
       -ml [PATH_TO_DATASET]/data/test/subtask1-homographic-test.gold
```

### Chumor

The Chinese humor dataset is available through their [HuggingFace page](https://huggingface.co/datasets/MichiganNLP/Chumor). We can download and convert it using the `utils/conversion/humor_recognition/chumor_to_json.py` script:

```bash
python utils/conversion/humor_recognition/chumor_to_json.py
```

## Large corpora

To train the skipgram model and get word frequency counts, we needed to download a large corpus of texts in each of the languages we work with. The corpora we used are as follows:

- **English**: [BookCorpus](https://huggingface.co/datasets/bookcorpus)
- **Portuguese**: [BRWaC](https://www.inf.ufrgs.br/pln/wiki/index.php?title=BrWaC#Current_version)
- **Spanish**: [SBWC](https://github.com/crscardellino/sbwce)
- **French**: [Croissant](https://huggingface.co/datasets/croissantllm/croissant_dataset)
- **Chinese**: [Wikipedia](https://dumps.wikimedia.org/)

For more details on how to download and preprocess the data, please refer to [`docs/skipgram.md`](skipgram.md).

## Word Embeddings (GloVe)

For the Quantum Entropy-based metrics, we use GloVe embeddings with 300 dimensions, following the original work. The embeddings are to be included to the `data/embeddings/[LANGUAGE]/` directory, all under the name `glove_s300.gensim`. The embeddings are available from different sources:

- **English**: [GloVe](https://nlp.stanford.edu/projects/glove/)
- **Portuguese**: [NILC Embeddings](https://www.nilc.icmc.usp.br/embeddings)
- **Spanish**: [SBWCE Project](https://github.com/crscardellino/sbwce)
- **French**: [French Word Embeddings](https://github.com/Ismailhachimi/French-Word-Embeddings)
- **Chinese**: We trained our own embeddings using the Wikipedia corpus and the original GloVe code.

These word embeddings are usually distributed with the Word2Vec format, but we convert them to the Gensim format for easier loading. To do this, we recommend creating a new virtual environment, with Python 3.10, since Gensim is not compatible with Python 3.12. Then, we can convert the embeddings using the following command:

```bash
import gensim

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('/path/to/glove/embeddings.txt', binary=False)
word_vectors.save('glove_s300.gensim')
```

We highlight that the gensim format results in two files: `glove_s300.gensim` and `glove_s300.gensim.vectors.npy`. Both are needed for the code to work, so make sure to include them in the same directory.

## N-gram counts

To compute the local-global surprise metrics, we needed to have 1-gram and 3-gram counts for each language across large corpora. The n-grams counts for English, Spanish, French, and Chinese were obtained from the [Google Books n-gram frequency lists](https://github.com/orgtre/google-books-ngram-frequency). For Portuguese, we use [N-gram frequency counts from BRWaC](https://github.com/Superar/Portuguese-Ngram-Frequencies).
