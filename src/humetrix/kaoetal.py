from itertools import product
from typing import List

import numpy as np
import polars as pl
import spacy
import torch
from spacy.language import Language

from humetrix import SPACY_MODELS
from humetrix.skipgram import build_vocabulary, SGNS


class KaoConfig():
    def __init__(self, sentence: str, spacy_model: Language, skipgram: SGNS) -> None:
        self.spacy_model = spacy_model
        self.skipgram = skipgram
        self.text = sentence
        self.tokens = self.tokenize_sentence()
        self.f_configs = [list(p) for p in product([0, 1], repeat=len(self.tokens))]
        self.f_config_prior = 1/(2 ** len(self.tokens)) # p(\vec{f})
        self.log_f_config_prior = -len(self.tokens)

    def tokenize_sentence(self) -> List[str]:
        doc = self.spacy_model(self.text)
        tokens = [token.lower_ for token in doc]
        return tokens

class KaoAmbiguity():
    def __init__(self,
                 config: KaoConfig,
                 pun_sign: str,
                 alt_sign: str,
                 ngram1: pl.DataFrame,
                 ngram3: pl.DataFrame) -> None:
        self.config = config
        self.ngram1 = ngram1
        self.ngram3 = ngram3
        self.pun_sign = pun_sign
        self.alt_sign = alt_sign


    def sign_prior(self, sign: str):
        '''P(m)'''
        vocab_size = self.ngram1.height
        sign_lower = sign.lower()
        row = self.ngram1.filter(pl.col('ngram') == sign_lower)
        freq_sign = 0
        if row.height > 0:
            freq_sign = row.select(pl.col('freq')).item()
        freq_cum = self.ngram1.select(pl.col('freq')).sum().item()
        prob_sign = (freq_sign + 1) / (freq_cum + vocab_size)
        return prob_sign

    def word_posterior(self, f_config: List[int], idx: int, sign: str) -> float:
        '''p(w_i|m, f_i)'''
        f_i = f_config[idx]

        if f_i == 1:
            return 1e-5 # TODO: Implement logic

        trigram = ' '.join(self.config.tokens[max(0, idx-2):min(len(self.config.tokens), idx+1)])
        bigram = ' '.join(self.config.tokens[max(0, idx-2):min(len(self.config.tokens), idx)])
        row = self.ngram3.filter(pl.col('ngram') == trigram)
        freq_trigram = 0
        if row.height > 0:
            freq_trigram = row.select(pl.col('freq')).item()
        freq_bigram = self.ngram3.filter(pl.col('ngram').str.starts_with(bigram)).select(pl.col('freq')).sum().item()
        prob_trigram = (freq_trigram + 1) / (freq_bigram + self.ngram3.height)
        return prob_trigram

    def sign_posterior(self, sign: str) -> float:
        '''p(m|\\vec{w}) = \\sum_f p(m) p(f) \\prod_i p(w_i|m, f_i)'''
        # Work on log scale to avoid underflow
        log_sign_prior = np.log2(self.sign_prior(sign))
        total_prob = 0.0
        for f_config in self.config.f_configs:
            sum_log_word_posteriors = 0.0
            for i in range(len(self.config.tokens)):
                sum_log_word_posteriors += np.log2(self.word_posterior(f_config, i, sign))
            total_prob += np.exp2(log_sign_prior + self.config.log_f_config_prior + sum_log_word_posteriors)
        return total_prob

    def score(self) -> float:
        prob_pun_sign = self.sign_posterior(self.pun_sign)
        prob_alt_sign = self.sign_posterior(self.alt_sign)

        # Normalize to calculate entropy
        prob_sum = prob_pun_sign + prob_alt_sign
        prob_pun_sign /= prob_sum
        prob_alt_sign /= prob_sum

        if prob_pun_sign <= 0 or prob_alt_sign <= 0:
            return -1
        return -(prob_pun_sign * np.log2(prob_pun_sign) + prob_alt_sign * np.log2(prob_alt_sign))

if __name__ == '__main__':
    sentences = ['O que diz um coelho quando abre uma porta? Primeiro as cenouras.',
                 'O que diz um coelho quando abre uma porta? Primeiro as senhoras.',
                 'Qual é o youtuber que mais economiza na luz? O Jovem Led.',
                 'Qual é o youtuber que mais economiza na luz? O Whindersson Nunes.',
                 'Porque é que o computador não pára de espirrar? Porque apanhou um vírus.',
                 'Porque é que o computador não pára de avariar? Porque apanhou um vírus.',
                 'Qual o livro que conta a história dos imigrantes do sertão??? Vim das secas.',
                 'Qual o livro que conta a história dos imigrantes do sertão??? O quinze',
                 'O que é que acontece quando o Frodo morre? Passam-lhe uma certidão de Hobbit.',
                 'O que é que acontece quando o Frodo morre? Passam-lhe uma certidão de óbito.']
    signs = ['cenouras', 'cenouras',
             'led', 'led',
             'vírus', 'vírus',
             'vim das secas', 'vim das secas',
             'hobbit', 'hobbit']
    alt_signs = ['senhoras', 'senhoras',
                 'nerd', 'nerd',
                 'vírus', 'vírus',
                 'vidas secas', 'vidas secas',
                 'óbito', 'óbito']

    ngram1 = pl.read_csv('data/ngrams/pt/1gram.csv')
    ngram3 = pl.read_csv('data/ngrams/pt/3gram.csv')
    spacy_model = spacy.load(SPACY_MODELS['pt'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = build_vocabulary('data/ngrams/pt/1gram.csv')
    skipgram = SGNS(vocab, 300)
    state_dict = torch.load('results/skipgram/pt.pt', weights_only=True,
                            map_location=device)
    skipgram.load_state_dict(state_dict)
    skipgram.eval()

    sentence_idx = 2
    config = KaoConfig(sentences[sentence_idx], spacy_model, skipgram)
    kao_ambiguity = KaoAmbiguity(config, signs[sentence_idx], alt_signs[sentence_idx], ngram1, ngram3)
    score = kao_ambiguity.score()
    print(f'Score: {score}')
