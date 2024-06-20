# python utils/conversion/humor_recognition/huhu_to_json.py -d ../../Resources/Corpora/HUHU@IberLEF2023/train.csv
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import spacy

parser = ArgumentParser()
parser.add_argument('--datapath', '-d',
                    help='Path to the HUHU CSV train file.',
                    required=True, type=Path)
args = parser.parse_args()


train_df = pd.read_csv(args.datapath,
                       usecols=['index', 'tweet', 'humor'])
new_df = train_df.rename(columns={'index': 'id',
                                  'tweet': 'text',
                                  'humor': 'label'})
new_df['id'] = 'huhu2023.' + new_df['id'].astype(str)
new_df = new_df.set_index('id')

# Filter out texts
punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
punct_chars.extend([',', ';'])
nlp = spacy.load('es_dep_news_trf')
nlp.add_pipe('sentencizer', config={'punct_chars': punct_chars})
with nlp.select_pipes(enable='sentencizer'):
    docs = pd.Series(nlp.pipe(new_df['text']), index=new_df.index)
num_sents = docs.apply(lambda x: len(list(x.sents)))
new_df = new_df.loc[num_sents == 2, :]

new_df.to_json('data/humor_recognition/HUHU@IberLEF2023.json',
               force_ascii=False,
               indent=4,
               orient='index')

