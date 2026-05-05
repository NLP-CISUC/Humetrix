from itertools import product

import polars as pl
from spacy import load

from humetrix.configs import CONTENT_WORD_TAGS, SPACY_MODELS
from humetrix.kaoetal import KaoConfig, KaoMetricBase

sentence = 'You don\'t want to bee around a hive for too long.'
tokens = ['want', 'bee', 'hive', 'long']
pun_sign = 'bee'
alt_sign = 'be'
ngram1 = pl.read_csv('data/ngrams/en/1gram.csv')
ngram3 = pl.read_csv('data/ngrams/en/3gram.csv')

nlp = load(SPACY_MODELS['en'])
kao_config = KaoConfig(sentence, 'en', nlp, None)
kao_config._tokens = tokens
kao_config._f_configs = [
    list(p) for p in product([0, 1], repeat=len(kao_config._tokens))
]
kao_config._f_config_prior = 1 / (2 ** len(kao_config._tokens))
kao_config._log_f_config_prior = -len(kao_config._tokens)

kao_base = KaoMetricBase(kao_config, pun_sign, alt_sign, ngram1, ngram3)

pun_sign_3gram_prob = 2.75e-06
alt_sign_3gram_prob = 0.0676358
probs_3gram = [(0.0983657, 0.0983657), (2.75e-06, 0.0676358),
               (2.87e-05, 2.87e-05), (0.862273, 0.862273)]
relatedness = pl.read_csv('data/kaoetal/relatedness_clean.csv')



# for idx in range(len(kao_base.config._tokens)):
#     prob_trigram = -1
#
#     trigram = ' '.join(kao_base.config._tokens[max(0, idx - 2) : min(len(kao_base.config._tokens), idx + 1)])
#
#     if trigram in kao_base.ngram3['ngram']:
#         prob_trigram = kao_base.ngram3.filter(pl.col('ngram') == trigram)['prob'].item()
#     else:
#         bigram = ' '.join(kao_base.config._tokens[max(0, idx - 2) : min(len(kao_base.config._tokens), idx)])
#         freq_bigram = 0
#         if bigram in kao_base.ngram2['ngram']:
#             freq_bigram = kao_base.ngram2.filter(pl.col('ngram') == bigram)['freq'].item()
#         prob_trigram = 1 / (freq_bigram + len(kao_base.ngram3))
#     print((idx, prob_trigram))
