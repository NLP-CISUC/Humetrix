from itertools import product

import numpy as np
import polars as pl
from spacy import load

from humetrix.configs import CONTENT_WORD_TAGS, SPACY_MODELS
from humetrix.kaoetal import KaoConfig, KaoMetricBase


class TestBase(KaoMetricBase):
    def __init__(self, config, pun_sign, alt_sign, row_index, ngram1_df, ngram3_df, relatedness_df):
        self.config = config
        self.pun_sign = pun_sign
        self.alt_sign = alt_sign
        self.row_index = row_index
        self.ngram1_df = ngram1_df
        self.ngram3_df = ngram3_df
        self.relatedness_df = relatedness_df

    def _word_posterior(self, f_config, idx, sign):
        f_i = f_config[idx]
        word = self.config._tokens[idx]
        col = 'm1_trigram' if sign == self.pun_sign else 'm2_trigram'
        df_slice = self.ngram3_df.filter(pl.col('index') == self.row_index)
        trigram_prob = df_slice[col].to_list()[idx] if len(df_slice) > idx else 1e-10

        if f_i == 0: # Get trigram probability
            return trigram_prob

        # f_i == 1, get value from the relatedness annotation
        if word == sign:
            return float(np.exp(13.0) * trigram_prob)
        if word in (self.pun_sign, self.alt_sign):
            return float(np.exp(0.0) * trigram_prob)

        r = self.relatedness_df.filter(
            ((pl.col('word1') == sign) & (pl.col('word2') == word)) |
            ((pl.col('word1') == word) & (pl.col('word2') == sign))
        )
        if len(r) == 0:
            return 1e-10

        r_val = r['relatedness'].to_list()[0]
        return float(np.exp(r_val) * trigram_prob)

class TestAmbiguity(TestBase):
    # Changed to use Kao et al.'s unigram probabilities
    def _sign_prior(self, sign: str):
        prob = self.ngram1_df.filter(pl.col('word') == sign)['unigram']
        if len(prob) > 0:
            return prob.item()
        return 1e-10

    # Exact same implementation, just changing the base class
    def _sign_posterior(self, sign: str) -> float:
        log_sign_prior = np.log(self._sign_prior(sign))
        total_prob = 0.0
        posteriors = self._f_config_posteriors(sign)
        for f_config in self.config._f_configs:
            sum_log_word_posteriors = 0.0
            for i in range(len(self.config._tokens)):
                sum_log_word_posteriors += np.log(posteriors[(i, f_config[i])])
            total_prob += np.exp(
                log_sign_prior
                + self.config._log_f_config_prior
                + sum_log_word_posteriors
            )
        return total_prob

    def score(self) -> float:
        prob_pun_sign = self._sign_posterior(self.pun_sign)
        prob_alt_sign = self._sign_posterior(self.alt_sign)

        prob_sum = prob_pun_sign + prob_alt_sign
        if prob_sum > 0:
            prob_pun_sign /= prob_sum
            prob_alt_sign /= prob_sum
        else:
            return 0.0

        if prob_pun_sign <= 0 or prob_alt_sign <= 0:
            return 0  # 0xlog0 = 0
        return -(
            prob_pun_sign * np.log(prob_pun_sign)
            + prob_alt_sign * np.log(prob_alt_sign)
        )


nlp = load(SPACY_MODELS['en'])
kaoetal_data = (pl.read_csv('data/kaoetal/data-agg.csv')
                  .filter(pl.col('sentenceType') == 'pun')
                  .with_columns(pl.col('sentence').str.replace('#', '')))
kaoetal_results = pl.read_csv('data/kaoetal/data.csv')
kaoetal_relatedness = pl.read_csv('data/kaoetal/relatedness_clean.csv')
kaoetal_ngram1 = pl.read_csv('data/kaoetal/unigrams_clean.csv')
kaoetal_ngram3 = pl.read_csv('data/kaoetal/trigrams_clean.csv')

for row in kaoetal_data.iter_rows(named=True):
    row_idx = row[''] - 1
    tokens = kaoetal_ngram3.filter(pl.col('index') == row_idx)['word'].to_list()
    kao_config = KaoConfig(row['sentence'], 'en', nlp, None)

    # Check tokens and POS cleaning
    try:
        assert len(set(tokens) - set(kao_config._tokens)) == 0
    except AssertionError:
        doc = nlp(row['sentence'])
        missing_toks = set(tokens) - set(kao_config._tokens)
        missing_toks_pos = [(tok.lower_, tok.pos_) for tok in doc
                            if tok.lower_ in missing_toks]
        if missing_toks_pos:
            print(f'Missing in id {row_idx}: {missing_toks_pos}')

    # Ensure tokens are the same and recalculate configs
    kao_config._tokens = tokens
    kao_config._f_configs = [list(p) for p in product([0, 1], repeat=len(tokens))]
    kao_config._f_config_prior = 1 / (2 ** len(tokens))
    kao_config._log_f_config_prior = len(tokens) * np.log(0.5)

    ambiguity_metric = TestAmbiguity(
        config=kao_config,
        pun_sign=row['m1'],
        alt_sign=row['m2'],
        row_index=row_idx,
        ngram1_df=kaoetal_ngram1,
        ngram3_df=kaoetal_ngram3,
        relatedness_df=kaoetal_relatedness
    )

    score = ambiguity_metric.score()
    target_score = kaoetal_results.filter(pl.col('idx') == row_idx)['ambiguity'].item()
    score_diff = abs(score - target_score)

    try:
        assert score_diff <= 1e-5
    except AssertionError:
        print(f"id {row_idx}: calculated {score:.6f}, target {target_score:.6f} -> Difference: {score_diff}")
