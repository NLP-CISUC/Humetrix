import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from humetrix.metrics import LocalGlobalSurprise
from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_surprise_score(row, tokenizer, lm):
    surprise = LocalGlobalSurprise(row['text'],
                                   row['location'],
                                   row['interpretation'])
    return surprise.score(tokenizer, lm)


results = list()

# Semeval 2020 task 7 (Humicroedit)

print('------------ Humicroedit ------------')
start_time = time.time()
corpus_path = Path('data/humor_interpretation/humicroedit.json')
df = pd.read_json(corpus_path, orient='index')
checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
lm = AutoModelForMaskedLM.from_pretrained(checkpoint)
df['local global surprise'] = df.apply(get_surprise_score,
                                       axis='columns',
                                       tokenizer=tokenizer,
                                       lm=lm)
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# Puntuguese

print('------------ Puntuguese ------------')
start_time = time.time()
corpus_path = Path('data/humor_interpretation/puntuguese.json')
df = pd.read_json(corpus_path, orient='index')
checkpoint = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
lm = AutoModelForMaskedLM.from_pretrained(checkpoint)
df['local global surprise'] = df.apply(get_surprise_score,
                                       axis='columns',
                                       tokenizer=tokenizer,
                                       lm=lm)
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

df = pd.concat(results).set_index('id')
df = df[['text', 'location', 'interpretation',
         'local global surprise', 'funniness']]
df.to_csv('results/metrics_behavior/local_global_surprise_multicorpus.csv')
