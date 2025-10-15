import os

import pandas as pd
import spacy
from dotenv import load_dotenv
from huggingface_hub import login
from humetrix.configs import SPACY_MODELS

load_dotenv()
login(os.getenv('HF_TOKEN'))

df = pd.read_csv("hf://datasets/MichiganNLP/Chumor/test.tsv", sep="\t")
df.index = ['chumor.' + str(i) for i in df.index]

punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
punct_chars.extend([',', ';'])
nlp = spacy.load(SPACY_MODELS['zh'])
nlp.add_pipe('sentencizer', config={'punct_chars': punct_chars})
with nlp.select_pipes(enable='sentencizer'):
    docs = pd.Series(nlp.pipe(df['Joke']), index=df.index)
num_sents = docs.apply(lambda x: len(list(x.sents)))
df = df.loc[num_sents == 2, :][['Joke', 'Label']]
df.columns = ['text', 'label']
df['label'] = df['label'].map({'bad': 0, 'good': 1})

df.to_json('data/humor_recognition/chumor.json',
           force_ascii=False,
           indent=4, orient='index')
