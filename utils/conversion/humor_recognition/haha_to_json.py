# python utils/conversion/humor_recognition/haha_to_json.py -t ../../Resources/Corpora/HAHA@IberLEF2021/haha_2021_train.csv -d ../../Resources/Corpora/HAHA@IberLEF2021/haha_2021_dev_gold.csv -s ../../Resources/Corpora/HAHA@IberLEF2021/haha_2021_test_gold.csv
# python utils/conversion/humor_recognition/haha_to_json.py -t ../../Resources/Corpora/HAHA@IberLEF2019/haha_2019_train.csv -s ../../Resources/Corpora/HAHA@IberLEF2019/haha_2019_test_gold.csv
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import spacy

parser = ArgumentParser()
parser.add_argument(
    '--train',
    '-t',
    help='Train CSV file from HAHA@IberLEF 2021.',
    required=True,
    type=Path,
)
parser.add_argument(
    '--test',
    '-s',
    help='Test CSV file from HAHA@IberLEF 2021.',
    required=True,
    type=Path,
)
parser.add_argument(
    '--dev',
    '-d',
    help='Dev CSV file from HAHA@IberLEF 2021.',
    required=False,
    type=Path,
)
args = parser.parse_args()


corpus_name = args.train.parent.stem
all_dfs = list()
train_df = pd.read_csv(
    args.train, usecols=['id', 'text', 'is_humor', 'humor_rating']
)
test_df = pd.read_csv(
    args.test, usecols=['id', 'text', 'is_humor', 'humor_rating']
)
if args.dev:
    dev_df = pd.read_csv(
        args.dev, usecols=['id', 'text', 'is_humor', 'humor_rating']
    )
    all_dfs.append(dev_df)
all_dfs.append(train_df)
all_dfs.append(test_df)


new_df = pd.concat(all_dfs).rename(
    columns={'is_humor': 'label', 'humor_rating': 'funniness'}
)
new_df['id'] = f'{corpus_name}.' + new_df['id'].astype(str)
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

new_df.to_json(
    f'data/humor_recognition/{corpus_name}.json',
    force_ascii=False,
    indent=4,
    orient='index',
)
