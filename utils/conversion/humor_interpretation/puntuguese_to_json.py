# python utils/conversion/humor_interpretation/puntuguese_to_json.py -c ../../Resources/Corpora/BRHuM/data/puns.json

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser()
parser.add_argument('--corpus', '-c',
                    help='Puntuguese puns JSON file.',
                    required=True, type=Path)
args = parser.parse_args()


df = pd.read_json(args.corpus, dtype={'id': str})
df['location'] = df['signs'].str[0].str['pun sign']
df['interpretation'] = df['signs'].str[0].str['alternative sign'].str[0]
df['id'] = 'puntuguese.' + df['id'] + '.H'
df = df.drop(columns='signs').set_index('id')

df.to_json('data/humor_interpretation/puntuguese.json',
           force_ascii=False,
           indent=4, orient='index')
