from pathlib import Path
import pandas as pd

corpora_path = Path('data/humor_recognition')

cup_path = Path('data/humor_interpretation/cup.json')
cup_df = pd.read_json(cup_path, orient='index')
cup_df['label'] = 1
cup_df['corpus'] = 'cup'
cup_df = cup_df[['corpus', 'label', 'text']]

dfs = [cup_df]
for corpus_path in corpora_path.iterdir():
    df = pd.read_json(corpus_path, orient='index')
    df['corpus'] = corpus_path.stem
    dfs.append(df)

print(pd.concat(dfs).groupby(['corpus', 'label']).count())
