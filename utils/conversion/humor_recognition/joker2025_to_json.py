# uv run utils/conversion/humor_recognition/joker2025_to_json.py -c ../Resources/Corpora/Task\ 1\ -\ Humor-aware\ Information\ Retrieval/PT/joker_2025_task1_corpus_pt.json -t ../Resources/Corpora/Task\ 1\ -\ Humor-aware\ Information\ Retrieval/PT/joker_2025_task1_qrels_train_pt.json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import spacy


parser = ArgumentParser()
parser.add_argument(
    '--corpus',
    '-c',
    help='Path to JOKER 2025 Task 1 corpus JSON file',
    required=True,
    type=Path,
)
parser.add_argument(
    '--train',
    '-t',
    help='Path to the JSON file with training labels',
    required=True,
    type=Path,
)
args = parser.parse_args()

corpus_df = pd.read_json(args.corpus)
train_df = pd.read_json(args.train)

# Group by docid and set label to 1 if any qrel is 1, otherwise 0
doc_labels = (
    train_df.groupby('docid')['qrel']
    .apply(lambda x: 1 if (x == 1).any() else 0)
    .reset_index(name='label')
)

df = corpus_df.merge(doc_labels, on='docid', how='left')
df['label'] = df['label'].fillna(0).astype(int)
df['docid'] = (
    'joker_pt.'
    + df['docid'].astype(str)
    + '.'
    + df['label'].map({0: 'N', 1: 'H'})
)
df = df.set_index('docid')

# Filter out texts
punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
punct_chars.extend([',', ';'])
nlp = spacy.load('pt_core_news_lg')
nlp.add_pipe('sentencizer', config={'punct_chars': punct_chars})
with nlp.select_pipes(enable='sentencizer'):
    docs = pd.Series(nlp.pipe(df['text']), index=df.index)
num_sents = docs.apply(lambda x: len(list(x.sents)))
df = df.loc[num_sents == 2, :]

df.to_json(
    f'data/humor_recognition/joker_clef_pt.json',
    force_ascii=False,
    indent=4,
    orient='index',
)
