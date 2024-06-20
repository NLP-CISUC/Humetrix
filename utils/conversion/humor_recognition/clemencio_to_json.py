# python utils/conversion/humor_recognition/clemencio_to_json.py -d ../../Resources/Corpora/Recognizing-Humor-in-Portuguese/Datasets/Balanceados/all.txt
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import spacy

parser = ArgumentParser()
parser.add_argument('--datapath', '-d',
                    help='Path to André Clemêncio\'s TSV dataset.',
                    required=True, type=Path)
args = parser.parse_args()


df = pd.read_csv(args.datapath, sep='\t',
                 names=['text', 'label'])
df.index = 'clemencio.' + df.index.astype(str)
df['label'] = df['label'].map({'N': 0, 'H': 1})

# Filter out texts
punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
punct_chars.extend([',', ';'])
nlp = spacy.load('pt_core_news_lg')
nlp.add_pipe('sentencizer', config={'punct_chars': punct_chars})
with nlp.select_pipes(enable='sentencizer'):
    docs = pd.Series(nlp.pipe(df['text']), index=df.index)
num_sents = docs.apply(lambda x: len(list(x.sents)))
df = df.loc[num_sents == 2, :]

df.to_json('data/humor_recognition/clemencio.json',
           force_ascii=False,
           indent=4,
           orient='index')
