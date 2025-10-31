from itertools import product
from typing import List, Dict

import numpy as np
import polars as pl
from spacy.language import Language
from scipy.stats import entropy

from .configs import CONTENT_WORD_TAGS
from .skipgram import SGNS


class KaoConfig():
    def __init__(self, sentence: str, language: str, spacy_model: Language, skipgram: SGNS) -> None:
        self.spacy_model = spacy_model
        self.skipgram = skipgram
        self.text = sentence
        self.language = language
        self.tokens = self.tokenize_sentence()
        self.f_configs = [list(p) for p in product([0, 1], repeat=len(self.tokens))]
        self.f_config_prior = 1/(2 ** len(self.tokens)) # p(\vec{f})
        self.log_f_config_prior = -len(self.tokens)

    def tokenize_sentence(self) -> List[str]:
        doc = self.spacy_model(self.text)
        tokens = [token.lower_ for token in doc
                  if token.pos_ in CONTENT_WORD_TAGS[self.language]]
        return tokens

class KaoMetricBase():
    def __init__(self,
                 config: KaoConfig,
                 pun_sign: str,
                 alt_sign: str,
                 ngram1: pl.DataFrame,
                 ngram3: pl.DataFrame) -> None:
        self.config = config
        self.ngram1 = ngram1
        self.ngram3 = ngram3
        self._prepare_ngram_freqs()
        self.pun_sign = pun_sign
        self.alt_sign = alt_sign

    def _prepare_ngram_freqs(self):
        prior_smooth = (pl.col('freq') + 1) / (pl.col('freq').sum() + pl.col('freq').len())
        self.ngram1 = self.ngram1.with_columns(prior_smooth.alias('prob'))

        bigram = pl.col('ngram').str.extract(r'^(.*) ')
        bigram_sum = pl.col('freq').sum().over(bigram)
        posterior_smooth = ((pl.col('freq') + 1) / (bigram_sum + pl.col('freq').len()))
        self.ngram3 = self.ngram3.with_columns(posterior_smooth.alias('prob'))
        self.ngram2 = self.ngram3.group_by(bigram).sum()

    def _word_posterior(self, f_config: List[int], idx: int, sign: str) -> float:
        '''p(w_i|m, f_i)'''
        f_i = f_config[idx]

        if f_i == 1:
            min_dist = 5
            max_dist = 10

            tokens = self.config.tokens
            unk_token = '<unk>'
            padded_tokens = ([unk_token] * max_dist) + tokens + ([unk_token] * max_dist)
            padded_idx = idx + max_dist

            left_context = []
            right_context = []
            for d in range(min_dist, max_dist + 1):
                left_context.append(padded_tokens[padded_idx - d])
                right_context.append(padded_tokens[padded_idx + d])
            context_tokens = left_context + right_context

            skipgram_model = self.config.skipgram

            target_word = tokens[idx]
            target_word_idx = skipgram_model.get_word_idx(target_word)
            context_word_indices = [skipgram_model.get_word_idx(w) for w in context_tokens]
            return skipgram_model.predict_prob(target_word_idx, context_word_indices)

        trigram = ' '.join(self.config.tokens[max(0, idx-2):min(len(self.config.tokens), idx+1)])
        if trigram in self.ngram3['ngram']:
            return self.ngram3.filter(pl.col('ngram') == trigram)['prob'].item()

        bigram = ' '.join(self.config.tokens[max(0, idx-2):min(len(self.config.tokens), idx)])
        freq_bigram = 0
        if bigram in self.ngram2['ngram']:
            freq_bigram = self.ngram2.filter(pl.col('ngram') == bigram)['freq'].item()
        prob_trigram = 1 / (freq_bigram + len(self.ngram3))
        return prob_trigram

    def _f_config_posteriors(self, sign: str) -> Dict:
        posteriors = {}
        for f_config in self.config.f_configs:
            for i in range(len(self.config.tokens)):
                if (i, f_config[i]) not in posteriors:
                    posteriors[(i, f_config[i])] = self._word_posterior(f_config, i, sign)
        return posteriors

class KaoAmbiguity(KaoMetricBase):
    def _sign_prior(self, sign: str):
        '''P(m)'''
        sign_lower = sign.lower()
        if sign_lower in self.ngram1['ngram']:
            return self.ngram1.filter(pl.col('ngram') == sign_lower)['prob'].item()
        prob_sign = 1 / (self.ngram1['freq'].sum() + len(self.ngram1))
        return prob_sign

    def _sign_posterior(self, sign: str) -> float:
        '''p(m|\\vec{w}) = \\sum_f p(m) p(f) \\prod_i p(w_i|m, f_i)'''
        # Work on log scale to avoid underflow
        log_sign_prior = np.log2(self._sign_prior(sign))
        total_prob = 0.0
        posteriors = self._f_config_posteriors(sign)
        for f_config in self.config.f_configs:
            sum_log_word_posteriors = 0.0
            for i in range(len(self.config.tokens)):
                sum_log_word_posteriors += np.log2(posteriors[(i, f_config[i])])
            total_prob += np.exp2(log_sign_prior + self.config.log_f_config_prior + sum_log_word_posteriors)
        return total_prob

    def score(self) -> float:
        prob_pun_sign = self._sign_posterior(self.pun_sign)
        prob_alt_sign = self._sign_posterior(self.alt_sign)

        # Normalize to calculate entropy
        prob_sum = prob_pun_sign + prob_alt_sign
        prob_pun_sign /= prob_sum
        prob_alt_sign /= prob_sum

        if prob_pun_sign <= 0 or prob_alt_sign <= 0:
            return 0 # 0xlog0 = 0
        return -(prob_pun_sign * np.log2(prob_pun_sign) + prob_alt_sign * np.log2(prob_alt_sign))

class KaoDistinctiveness(KaoMetricBase):
    def score(self) -> float:
        sampled_pun_sign = []
        sampled_alt_sign = []
        for f_config in self.config.f_configs:
            pun_log_prob = self.config.log_f_config_prior
            posteriors = self._f_config_posteriors(self.pun_sign)
            for i in range(len(self.config.tokens)):
                pun_log_prob += np.log2(posteriors[(i, f_config[i])])

            alt_log_prob = self.config.log_f_config_prior
            posteriors = self._f_config_posteriors(self.alt_sign)
            for i in range(len(self.config.tokens)):
                alt_log_prob += np.log2(posteriors[(i, f_config[i])])

            sampled_pun_sign.append(np.exp2(pun_log_prob))
            sampled_alt_sign.append(np.exp2(alt_log_prob))

        kl1 = entropy(sampled_pun_sign, sampled_alt_sign)
        kl2 = entropy(sampled_alt_sign, sampled_pun_sign)
        return kl1 + kl2
