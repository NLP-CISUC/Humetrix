# python utils/conversion/humor_recognition/humicroedit_to_json.py -t ../../Resources/Corpora/Humicroedit/subtask-1/train.csv -d ../../Resources/Corpora/Humicroedit/subtask-1/dev.csv -s ../../Resources/Corpora/Humicroedit/subtask-1/test.csv
import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import spacy

parser = ArgumentParser()
parser.add_argument('--train', '-t',
                    help='Train CSV file from Humicroedit.',
                    required=True, type=Path)
parser.add_argument('--dev', '-d',
                    help='Dev CSV file from Humicroedit.',
                    required=True, type=Path)
parser.add_argument('--test', '-s',
                    help='Test CSV file from Humicroedit.',
                    required=True, type=Path)
args = parser.parse_args()


def do_edit(row):
    return re.sub(r'<.*/>', row['edit'], row['original'])


train_df = pd.read_csv(args.train)
dev_df = pd.read_csv(args.dev)
test_df = pd.read_csv(args.test)

df = pd.concat([train_df, dev_df, test_df])
df['id'] = 'humicroedit.' + df['id'].astype(str) + '.N'
df['humorous'] = df.apply(do_edit, axis='columns')

original_df = df.loc[:, ['id', 'original']]
original_df['text'] = original_df['original'].str.replace(r'<(.*)/>',
                                                          r'\1',
                                                          regex=True)
original_df['label'] = 0
original_df = original_df.drop(columns='original')


humorous_df = df.loc[:, ['id', 'humorous']]
humorous_df['id'] = humorous_df['id'].str.replace('.N$',
                                                  '.H',
                                                  regex=True)
humorous_df['label'] = 1
humorous_df.columns = ['id', 'text', 'label']

new_df = pd.concat([original_df, humorous_df]).set_index('id')

# Filter out texts
punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
punct_chars.extend([',', ';'])
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('sentencizer', config={'punct_chars': punct_chars})
with nlp.select_pipes(enable='sentencizer'):
    docs = pd.Series(nlp.pipe(new_df['text']), index=new_df.index)
num_sents = docs.apply(lambda x: len(list(x.sents)))
new_df = new_df.loc[num_sents == 2, :]

new_df.to_json('data/humor_recognition/humicroedit.json',
               force_ascii=False,
               indent=4, orient='index')

