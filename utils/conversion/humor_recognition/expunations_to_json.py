# uv run utils/conversion/humor_recognition/expunations_to_json.py -d ../Resources/Corpora/expunations/data/expunations_annotated_full.json -s data/humor_recognition/semeval.json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import statistics


parser = ArgumentParser()
parser.add_argument(
    '--datapath',
    '-d',
    help="Path to ExPUNations's full JSON file",
    required=True,
    type=Path,
)
parser.add_argument(
    '--semeval',
    '-s',
    help="Path to SemEval's dataset already converted to json",
    required=True,
    type=Path,
)
args = parser.parse_args()

expun_df = pd.read_json(args.datapath)
semeval_df = pd.read_json(args.semeval, orient='index')


# Keep only jokes that have been pre-filtered by semeval's conversion
# Get texts from semeval's corpus
semeval_df['ID'] = semeval_df.index.str.lstrip('semeval.')
expun_df = expun_df.loc[:, ['ID', 'Funniness (1-5)', 'Is a Joke?']]
expun_df = expun_df.merge(semeval_df, on='ID', how='inner')

expun_df['label'] = expun_df['Is a Joke?'].apply(statistics.mode)
expun_df['funniness'] = expun_df['Funniness (1-5)'].apply(statistics.mean)
expun_df = expun_df.drop(columns=['Funniness (1-5)', 'Is a Joke?'])

expun_df['ID'] = 'expunations.' + expun_df['ID']
expun_df = expun_df.set_index('ID')

expun_df.to_json(
    'data/humor_recognition/expunations.json',
    force_ascii=False,
    indent=4,
    orient='index',
)
