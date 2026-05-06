# uv run utils/conversion/humor_interpretation/cup_to_json.py -d ../Resources/Corpora/context-situated-pun-generation/data/context_situated_pun.csv
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import spacy


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

# Filter out texts
punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
punct_chars.extend([',', ';'])
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('sentencizer', config={'punct_chars': punct_chars})
with nlp.select_pipes(enable='sentencizer'):
    docs = pd.Series(nlp.pipe(df['text']), index=df.index)
num_sents = docs.apply(lambda x: len(list(x.sents)))
df = df.loc[num_sents == 2, :]

df.to_json(
    'data/humor_interpretation/cup.json',
    force_ascii=False,
    indent=4,
    orient='index',
)
