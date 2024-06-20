# python utils/conversion/humor_recognition/joker_to_json.py -d ../../Resources/Corpora/JOKER-CLEF2023/Task\ 1\ -\ detection/
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import spacy

SPACY_MODELS = {'pt': 'pt_core_news_lg',
                'en': 'en_core_web_trf',
                'fr': 'fr_dep_news_trf',
                'es': 'es_dep_news_trf'}

parser = ArgumentParser()
parser.add_argument('--datapath', '-d',
                    help='Path to the task 1 folder of JOKER CLEF 2023.',
                    required=True, type=Path)
args = parser.parse_args()


for lang in args.datapath.iterdir():
    datafile = lang / f'joker_detection_{lang.stem.upper()}_train_input.json'
    labelfile = lang / f'joker_detection_{lang.stem.upper()}_train_qrels.json'

    data_df = pd.read_json(datafile)
    label_df = pd.read_json(labelfile)

    df = data_df.copy()
    df['label'] = label_df['wordplay'].map({'no': 0, 'yes': 1})
    id_labels = df['label'].map({0: 'N', 1: 'H'})
    df['id'] = df['id'].str.replace('_', '.')
    df['id'] = 'joker_' + df['id'].astype(str) + '.' + id_labels
    df = df.drop_duplicates().dropna().set_index('id')

    # Filter out texts
    punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
    punct_chars.extend([',', ';'])
    nlp = spacy.load(SPACY_MODELS[lang.stem])
    nlp.add_pipe('sentencizer', config={'punct_chars': punct_chars})
    with nlp.select_pipes(enable='sentencizer'):
        docs = pd.Series(nlp.pipe(df['text']), index=df.index)
    num_sents = docs.apply(lambda x: len(list(x.sents)))
    df = df.loc[num_sents == 2, :]

    df.to_json(f'data/humor_recognition/joker_clef_{lang.stem}.json',
               force_ascii=False,
               indent=4, orient='index')
