# uv run utils/conversion/humor_interpretation/cup_to_json.py -d ../Resources/Corpora/context-situated-pun-generation/data/context_situated_pun.csv
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


parser = ArgumentParser()
parser.add_argument(
    '--datapath', '-d', help="Path to CUP's CSV file", required=True, type=Path
)
args = parser.parse_args()

df = pd.read_csv(args.datapath).dropna()
df = df.loc[df['user_pun'] != '{}']
df = df.loc[:, ['pun_word', 'alter_word', 'user_pun']]
df = df.rename(
    columns={
        'pun_word': 'location',
        'alter_word': 'interpretation',
        'user_pun': 'text',
    }
)
df.index = 'cup.' + df.index.astype(str)

df.to_json(
    'data/humor_interpretation/cup.json',
    force_ascii=False,
    indent=4,
    orient='index',
)
