import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from humetrix.metrics import QuantumIncongruity
from gensim.models import KeyedVectors


def get_qi_score(sentence, embeddings, lang):
    qi = QuantumIncongruity(sentence, embeddings, lang)
    return qi.score()


results = list()

# Semeval 2017 task 7

print('------------ Semeval 2017 task 7 ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/semeval.json')
df = pd.read_json(corpus_path, orient='index')
embeddings_path = '../../Resources/Embeddings/English/glove_s300.gensim'
embeddings = KeyedVectors.load(embeddings_path)
df['quantum incongruity'] = df['text'].apply(get_qi_score,
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
df['quantum incongruity'] = df['text'].apply(get_qi_score,
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
df['quantum incongruity'] = df['text'].apply(get_qi_score,
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
embeddings_path = '../../Resources/Embeddings/French/WE_models/glove_s300.gensim'
embeddings = KeyedVectors.load(embeddings_path)
df['quantum incongruity'] = df['text'].apply(get_qi_score,
                                             embeddings=embeddings,
                                             lang='fr')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

# Corpus by André Clemêncio
print('------------ André Clemêncio ------------')
start_time = time.time()
corpus_path = Path('data/humor_recognition/clemencio.json')
df = pd.read_json(corpus_path, orient='index')
embeddings_path = '../../Resources/Embeddings/Portuguese/glove_s300.gensim'
embeddings = KeyedVectors.load(embeddings_path)
df['quantum incongruity'] = df['text'].apply(get_qi_score,
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
df['quantum incongruity'] = df['text'].apply(get_qi_score,
                                             embeddings=embeddings,
                                             lang='pt')
results.append(df.copy())
end_time = time.time()
elapsed_time = end_time - start_time
print(timedelta(seconds=elapsed_time))

df = pd.concat(results)
df.to_csv('results/metrics_behavior/quantum_incongruity_multicorpus.csv')
