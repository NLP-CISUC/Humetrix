import gc
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import spacy
import torch
from gensim.models import KeyedVectors
from humetrix import SPACY_MODELS, QuantumIncongruity, QuantumUncertainty
from tqdm import tqdm
from transformers import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_qu_score(sentence, embeddings, lang):
    qu = QuantumUncertainty(sentence, embeddings, lang)
    score = qu.score()
    del qu
    gc.collect()
    return score


def get_qi_score(sentence, embeddings, lang):
    qi = QuantumIncongruity(sentence, embeddings, lang)
    score = qi.score()
    del qi
    gc.collect()
    return score


parser = ArgumentParser()
parser.add_argument('--corpus', '-c',
                    help='Corpus to calculate QE metric.',
                    required=True, type=Path)
parser.add_argument('--glove', '-g',
                    help='GLoVe embeddings path.',
                    required=False, type=str)
parser.add_argument('--huggingface', '-hf',
                    help='HuggingFace model name.',
                    required=False, type=str)
parser.add_argument('--language', '-l',
                    help='Corpus language.',
                    required=True, type=str,
                    choices=['en', 'es', 'fr', 'pt'])
parser.add_argument('--incongruity', '-i', action='store_true',
                    help='Calculate QE-Incongruity.')
parser.add_argument('--uncertainty', '-u', action='store_true',
                    help='Calculate QE-Uncertainty.')
args=parser.parse_args()

if not args.incongruity and not args.uncertainty:
    print('Select at least one metric to compute.')
    exit(1)
if not args.glove and not args.huggingface:
    print('Give at least one embedding model (GloVe or HuggingFace).')
    exit(1)

# Load spacy model
spacy_model = spacy.load(SPACY_MODELS[args.language])

# Load embeddings or HuggingFace model
glove_embeddings = KeyedVectors.load(args.glove) if args.glove else None
hf_embeddings = (pipeline('feature-extraction', args.huggingface, device=device)
                 if args.huggingface else None)

# Load corpus
df = pd.read_json(args.corpus, orient='index')

# Compute metrics
if args.incongruity and glove_embeddings:
    tqdm.pandas(desc='Incongruity + GloVe')
    df['QE-I + GloVe'] = df['text'].progress_apply(get_qi_score,
                                                   embeddings=glove_embeddings,
                                                   lang=spacy_model)
if args.incongruity and hf_embeddings:
    tqdm.pandas(desc='Incongruity + HuggingFace')
    df['QE-I + HF'] = df['text'].progress_apply(get_qi_score,
                                                embeddings=hf_embeddings,
                                                lang=spacy_model)
if args.uncertainty and glove_embeddings:
    tqdm.pandas(desc='Uncertainty + GloVe')
    df['QE-U + GloVe'] = df['text'].progress_apply(get_qu_score,
                                                   embeddings=glove_embeddings,
                                                   lang=spacy_model)
if args.uncertainty and hf_embeddings:
    tqdm.pandas(desc='Uncertainty + HuggingFace')
    df['QE-U + HF'] = df['text'].progress_apply(get_qu_score,
                                                embeddings=hf_embeddings,
                                                lang=spacy_model)

results_path = Path('results/quantum_entropy') / args.corpus.name
results_path.parent.mkdir(exist_ok=True, parents=True)
df.to_json(results_path, orient='index', force_ascii=False, indent=4)

