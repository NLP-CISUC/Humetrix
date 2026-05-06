from pathlib import Path

import func_timeout
import pandas as pd
from tqdm import tqdm

from humetrix import HumorAnalyzer

# Only use corpora for humor interpretation (we need pun and alternative words)
corpora = {'en': ['cup'],
           'pt': ['puntuguese']}
paths = {'cup': 'data/humor_interpretation/cup.json',
         'puntuguese': 'data/humor_interpretation/puntuguese.json'}
results_path = Path('results/kaoetal')
results_path.mkdir(exist_ok=True, parents=True)


def get_kaoetal_scores(row, analyzer):
    ambiguity = analyzer.kao_ambiguity(row['text'],
                                       row['location'],
                                       row['interpretation'])
    distinct = analyzer.kao_distinctiveness(row['text'],
                                            row['location'],
                                            row['interpretation'])
    return [ambiguity, distinct]


for language, datasets in corpora.items():
    analyzer = HumorAnalyzer(language,
                             skipgram_path=f'results/skipgram/{language}.pt',
                             ngram_dir_path=f'data/ngrams/{language}')
    for corpus in datasets:
        print('- '*10 + corpus + ' -'*10)
        df = pd.read_json(paths[corpus], orient='index').reset_index()
        df['char_count'] = df['text'].str.len()
        df = df.sort_values('char_count').reset_index(drop=True)

        df['ambiguity'] = None
        df['distinctiveness'] = None
        timeout_count = 0
        savepath = (results_path / corpus).with_suffix('.jsonl')

        groups = df.groupby('char_count')
        pbar = tqdm(total=len(df), desc=f"Processing {corpus} (Timeouts: 0)")

        for char_len, group in groups:
            for idx in group.index:
                row = df.loc[idx]

                try:
                    res = func_timeout.func_timeout(1800, analyzer.kao_ambiguity,
                                                    args=(row['text'],
                                                          row['location'],
                                                          row['interpretation']))
                    distinct = func_timeout.func_timeout(1800, analyzer.kao_distinctiveness,
                                                         args=(row['text'],
                                                               row['location'],
                                                               row['interpretation']))
                    df.at[idx, 'ambiguity'] = res
                    df.at[idx, 'distinctiveness'] = distinct
                except func_timeout.FunctionTimedOut:
                    timeout_count += 1
                    pbar.set_description(f"Processing {corpus} (Timeouts: {timeout_count})")
                    continue
                finally:
                    pbar.update(1)
            df.to_json(savepath, orient='records', lines=True, force_ascii=False)
        pbar.close()
        print(f"Processing complete. Total timeouts: {timeout_count}")

