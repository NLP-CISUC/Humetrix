import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from feedbac.metrics import QuantumUncertainty
from transformers import pipeline


def get_qu_score(sentence, embeddings, lang):
    qu = QuantumUncertainty(sentence, embeddings, lang)
    return qu.score()


results = list()
total_start_time = time.time()

# Semeval 2017 task 7
print('------------ Semeval 2017 task 7 ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/semeval.json')
df = pd.read_json(corpus_path, orient='index')
embeddings = pipeline('feature-extraction', 'FacebookAI/xlm-roberta-base')
df['quantum uncertainty'] = df['text'].apply(get_qu_score,
                                             embeddings=embeddings,
                                             lang='en')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# Semeval 2020 task 7 (Humicroedit)
print('------------ Humicroedit ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/humicroedit.json')
df = pd.read_json(corpus_path, orient='index')
df['quantum uncertainty'] = df['text'].apply(get_qu_score,
                                             embeddings=embeddings,
                                             lang='en')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# JOKER CLEF 2023 (English)
print('------------ JOKER CLEF 2023 (English) ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/joker_clef_en.json')
df = pd.read_json(corpus_path, orient='index')
df['quantum uncertainty'] = df['text'].apply(get_qu_score,
                                             embeddings=embeddings,
                                             lang='en')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# JOKER CLEF 2023 (French)
print('------------ JOKER CLEF 2023 (French) ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/joker_clef_fr.json')
df = pd.read_json(corpus_path, orient='index')
df['quantum uncertainty'] = df['text'].apply(get_qu_score,
                                             embeddings=embeddings,
                                             lang='fr')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# JOKER CLEF 2023 (Spanish)
print('------------ JOKER CLEF 2023 (Spanish) ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/joker_clef_es.json')
df = pd.read_json(corpus_path, orient='index')
df['quantum uncertainty'] = df['text'].apply(get_qu_score,
                                             embeddings=embeddings,
                                             lang='es')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# HAHA@IberLEF 2019
print('------------ HAHA@IberLEF 2019 ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/HAHA@IberLEF2019.json')
df = pd.read_json(corpus_path, orient='index')
df['quantum uncertainty'] = df['text'].apply(get_qu_score,
                                             embeddings=embeddings,
                                             lang='es')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# HAHA@IberLEF 2021
print('------------ HAHA@IberLEF 2021 ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/HAHA@IberLEF2021.json')
df = pd.read_json(corpus_path, orient='index')
df['quantum uncertainty'] = df['text'].apply(get_qu_score,
                                             embeddings=embeddings,
                                             lang='es')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# HUHU@IberLEF 2023
print('------------ HUHU@IberLEF 2023 ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/HUHU@IberLEF2023.json')
df = pd.read_json(corpus_path, orient='index')
df['quantum uncertainty'] = df['text'].apply(get_qu_score,
                                             embeddings=embeddings,
                                             lang='es')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# Corpus by André Clemêncio
print('------------ André Clemêncio ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/clemencio.json')
df = pd.read_json(corpus_path, orient='index')
df['quantum uncertainty'] = df['text'].apply(get_qu_score,
                                             embeddings=embeddings,
                                             lang='pt')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# Puntuguese
print('------------ Puntuguese ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/puntuguese.json')
df = pd.read_json(corpus_path, orient='index')
df['quantum uncertainty'] = df['text'].apply(get_qu_score,
                                             embeddings=embeddings,
                                             lang='pt')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

df = pd.concat(results)
df.to_csv('results/metrics_behavior/quantum_uncertainty_multicorpus_xlmr.csv')

total_time = time.time() - total_start_time
print('Total time: ', timedelta(seconds=total_start_time))
