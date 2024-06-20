# python utils/conversion/humor_recognition/puntuguese_to_json.py -d ../../Resources/Corpora/BRHuM/data/classification_corpus.json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import spacy

parser = ArgumentParser()
parser.add_argument('--data', '-d',
                    help='Path to classification data from Puntuguese in JSON.',
                    required=True, type=Path)
args = parser.parse_args()


df = pd.read_json(args.data, orient='index')
df.index = 'puntuguese.' + df.index

# Filter out texts
punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
punct_chars.extend([',', ';'])
nlp = spacy.load('pt_core_news_lg')
nlp.add_pipe('sentencizer', config={'punct_chars': punct_chars})
with nlp.select_pipes(enable='sentencizer'):
    docs = pd.Series(nlp.pipe(df['text']), index=df.index)
num_sents = docs.apply(lambda x: len(list(x.sents)))
df = df.loc[num_sents == 2, :]

df.to_json('data/humor_recognition/puntuguese.json',
               force_ascii=False,
               indent=4, orient='index')
