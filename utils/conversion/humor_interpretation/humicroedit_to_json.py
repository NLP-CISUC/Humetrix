# python utils/conversion/humor_interpretation/humicroedit_to_json.py -t ../../Resources/Corpora/Humicroedit/subtask-1/train.csv -d ../../Resources/Corpora/Humicroedit/subtask-1/dev.csv -s ../../Resources/Corpora/Humicroedit/subtask-1/test.csv
import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

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
df['id'] = 'humicroedit.' + df['id'].astype(str) + '.H'
df['interpretation'] = df['original'].str.extract(r'<(.*)/>')
df['text'] = df.apply(do_edit, axis='columns')
df = df.query('grades > 0').copy()
df = df.rename(columns={'edit': 'location',
                        'meanGrade': 'funniness'})
df = df.drop(columns=['original', 'grades'])
df = df.set_index('id')

df.to_json('data/humor_interpretation/humicroedit.json',
           force_ascii=False,
           indent=4, orient='index')
