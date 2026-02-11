from pathlib import Path
import pandas as pd

corpora_path = Path('data/humor_recognition')

dfs = list()
for corpus_path in corpora_path.iterdir():
    df = pd.read_json(corpus_path, orient='index')
    df['corpus'] = corpus_path.stem
    dfs.append(df)

print(pd.concat(dfs).groupby(['corpus', 'label']).count())
