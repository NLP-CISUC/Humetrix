from pathlib import Path

import pandas as pd
from tqdm import tqdm
from humetrix import HumorAnalyzer

# Only use corpora for humor interpretation (we need pun and alternative words)
corpora = {'en': ['cup'],
           'pt': ['puntuguese']}
paths = {'cup': 'data/humor_interpretation/cup.json',
         'puntuguese': 'data/humor_interpretation/puntuguese.json'}
results_path = Path('results/local_global_surprise')
results_path.mkdir(exist_ok=True, parents=True)


def get_surprise_score(row, analyzer):
    return analyzer.local_global_surprise(row['text'],
                                          row['location'],
                                          row['interpretation'])


for language, datasets in corpora.items():
    analyzer = HumorAnalyzer(language)
    for corpus in datasets:
        print('- '*10 + corpus + ' -'*10)
        df = pd.read_json(paths[corpus], orient='index').reset_index()

        tqdm.pandas(desc='Local Global Surprise')
        df['local global surprise'] = df.progress_apply(get_surprise_score,
                                                        axis='columns',
                                                        analyzer=analyzer)

        savepath = (results_path / corpus).with_suffix('.jsonl')
        df.to_json(savepath, orient='records', lines=True, force_ascii=False)
