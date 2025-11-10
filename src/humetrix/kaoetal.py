"""
This module implements the humor metrics proposed by Kao et al. (2016)_[1].

References
----------
[1] Justine T. Kao, Roger Levy, and Noah D. Goodman. 2016. A Computational Model of Linguistic Humor in Puns. Cogn Sci 40, 5 (July 2016), 1270–1285. https://doi.org/10.1111/cogs.12269
"""

from itertools import product
from typing import Dict, List

import numpy as np
import polars as pl
from scipy.stats import entropy
from spacy.language import Language

from .configs import CONTENT_WORD_TAGS
from .skipgram import SGNS


class KaoConfig:
    """
    A configuration class for the Kao et al. (2016) metrics.

    Parameters
    ----------
    sentence : str
        The sentence to be analyzed.
    language : str
        The language of analysis.
    spacy_model : Language
        The spaCy language model to use for tokenization.
    skipgram : SGNS
        Distant skip-gram model_[1] used to approximate the empirical association
        measures from the original paper.

    References
    ----------
    [1] He He, Nanyun Peng, and Percy Liang. 2019. Pun Generation with Surprise. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019. Association for Computational Linguistics, Minneapolis, 1734–1744. https://doi.org/10.18653/v1/N19-1172
    """

    def __init__(
        self,
        sentence: str,
        language: str,
        spacy_model: Language,
        skipgram: SGNS,
    ) -> None:
        self.spacy_model = spacy_model
        self.skipgram = skipgram
        self.text = sentence
        self.language = language
        self._tokens = self._tokenize_sentence()
        self._f_configs = [
            list(p) for p in product([0, 1], repeat=len(self._tokens))
        ]
        self._f_config_prior = 1 / (2 ** len(self._tokens))  # p(\vec{f})
        self._log_f_config_prior = -len(self._tokens)

    def _tokenize_sentence(self) -> List[str]:
        """
        Tokenize the sentence and filter to content words only.

        Returns
        -------
        list of str
            A list of content word tokens.
        """
        doc = self.spacy_model(self.text)
        tokens = [
            token.lower_
            for token in doc
            if token.pos_ in CONTENT_WORD_TAGS[self.language]
        ]
        return tokens


class KaoMetricBase:
    """
    A base class for the Kao et al. (2016) metrics.

    This class provides common functions for calculating both metrics proposed
    by Kao et al. (2016)_[1]: ambiguity and distinctiveness. Namely, it handles
    the preparation of n-gram probabilities and the calculation of word
    posteriors given a hidden variable configuration (:math:`P(w_i|m,f_i)`).

    Parameters
    ----------
    config : KaoConfig
        The configuration object.
    pun_sign : str
        The pun sign. The word or phrase that is ambiguous or humorous.
    alt_sign : str
        The alternative sign. The word or phrase that is evoked by the pun sign.
    ngram1 : polars.DataFrame
        A DataFrame with 1-gram frequencies following the format of
        orgtre/google-books-ngram-frequency. We expect at least two columns:
        'ngram' (str) and 'freq' (int).
    ngram3 : polars.DataFrame
        A DataFrame with 3-gram frequencies following the format of
        orgtre/google-books-ngram-frequency. We expect at least two columns:
        'ngram' (str) and 'freq' (int).

    References
    ----------
    [1] Justine T. Kao, Roger Levy, and Noah D. Goodman. 2016. A Computational Model of Linguistic Humor in Puns. Cogn Sci 40, 5 (July 2016), 1270–1285. https://doi.org/10.1111/cogs.12269
    """

    def __init__(
        self,
        config: KaoConfig,
        pun_sign: str,
        alt_sign: str,
        ngram1: pl.DataFrame,
        ngram3: pl.DataFrame,
    ) -> None:
        self.config = config
        self.ngram1 = ngram1
        self.ngram3 = ngram3
        self._prepare_ngram_freqs()
        self.pun_sign = pun_sign
        self.alt_sign = alt_sign

    def _prepare_ngram_freqs(self):
        """
        Calculate smoothed probabilities from 1-grams and 3-grams counts.

        This method used Laplace (add-one) smoothing to handle any tokens that
        have a zero count.
        """
        prior_smooth = (pl.col('freq') + 1) / (
            pl.col('freq').sum() + pl.col('freq').len()
        )
        self.ngram1 = self.ngram1.with_columns(prior_smooth.alias('prob'))

        bigram = pl.col('ngram').str.extract(r'^(.*) ')
        bigram_sum = pl.col('freq').sum().over(bigram)
        posterior_smooth = (pl.col('freq') + 1) / (
            bigram_sum + pl.col('freq').len()
        )
        self.ngram3 = self.ngram3.with_columns(posterior_smooth.alias('prob'))
        self.ngram2 = self.ngram3.group_by(bigram).sum()

    def _word_posterior(
        self, f_config: List[int], idx: int, sign: str
    ) -> float:
        r"""
        Calculate a word's posterior probability: :math:`P(w_i|m,f_i)`.

        Parameters
        ----------
        f_config : list of int
            The sentence's hidden variable configuration.
        idx : int
            The index of the target word in the sentence.
        sign : str
            The sign (pun or alternative) being evaluated, representing the
            sentence meaning.

        Returns
        -------
        float
            The posterior probability of the target word.

        Notes
        -----
        If the hidden variable :math:`f_i = 1`, we estimate the association
        measures using a distant skip-gram model_[1] to calculate the probability
        :math:`P(w_i|m)`. Otherwise, we use smoothed n-gram probabilities
        :math:`P(w_i|\mathrm{bigram}_i)`.

        References
        ----------
        [1] He He, Nanyun Peng, and Percy Liang. 2019. Pun Generation with Surprise. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019. Association for Computational Linguistics, Minneapolis, 1734–1744. https://doi.org/10.18653/v1/N19-1172
        """
        f_i = f_config[idx]

        if f_i == 1:
            skipgram_model = self.config.skipgram
            tokens = self.config._tokens

            current_word = tokens[idx]
            current_word_idx = skipgram_model.get_word_idx(current_word)
            sign_word_idx = skipgram_model.get_word_idx(sign)

            # Using skip-gram to calculate P(word | sign), where the sign is
            # treated as the target and the current word as the context.
            return skipgram_model.predict_prob(
                input_word_idx=sign_word_idx,
                output_word_idx=current_word_idx,
            )

        trigram = ' '.join(
            self.config._tokens[
                max(0, idx - 2) : min(len(self.config._tokens), idx + 1)
            ]
        )
        if trigram in self.ngram3['ngram']:
            return self.ngram3.filter(pl.col('ngram') == trigram)['prob'].item()

        bigram = ' '.join(
            self.config._tokens[
                max(0, idx - 2) : min(len(self.config._tokens), idx)
            ]
        )
        freq_bigram = 0
        if bigram in self.ngram2['ngram']:
            freq_bigram = self.ngram2.filter(pl.col('ngram') == bigram)[
                'freq'
            ].item()
        prob_trigram = 1 / (freq_bigram + len(self.ngram3))
        return prob_trigram

    def _f_config_posteriors(self, sign: str) -> Dict:
        """
        Calculate word posteriors for all f_configs.

        Parameters
        ----------
        sign : str
            The sign (pun or alternative) being evaluated, representing the
            sentence meaning.

        Returns
        -------
        dict
            A dictionary mapping (word index, f_i) to the posterior probability
            :math:`P(w_i|m,f_i)`.

        Notes
        -----
        This method caches the results to avoid redundant calculations across
        different f_configs, i.e. if two f_configs have the same value for a
        particular word index, the posterior is only calculated once.
        """
        posteriors = {}
        for f_config in self.config._f_configs:
            for i in range(len(self.config._tokens)):
                if (i, f_config[i]) not in posteriors:
                    posteriors[(i, f_config[i])] = self._word_posterior(
                        f_config, i, sign
                    )
        return posteriors


class KaoAmbiguity(KaoMetricBase):
    """Calculates the ambiguity metric from Kao et al. (2016)."""

    def _sign_prior(self, sign: str):
        """
        Calculate the prior probability of a sentence meaning: :math:`P(m)`.

        Parameters
        ----------
        sign : str
            The sign (pun or alternative) being evaluated, representing the
            sentence meaning.

        Returns
        -------
        float
            The prior probability of the sentence meaning.
        """
        sign_lower = sign.lower()
        if sign_lower in self.ngram1['ngram']:
            return self.ngram1.filter(pl.col('ngram') == sign_lower)[
                'prob'
            ].item()
        prob_sign = 1 / (self.ngram1['freq'].sum() + len(self.ngram1))
        return prob_sign

    def _sign_posterior(self, sign: str) -> float:
        r"""
        Calculate the posterior probability of a sentence meaning: :math:`P(m|\vec{w})`.

        Parameters
        ----------
        sign : str
            The sign (pun or alternative) being evaluated, representing the
            sentence meaning.

        Returns
        -------
        float
            The posterior probability of the sentence meaning.

        Notes
        -----
        This method marginalizes over all hidden variable configurations:
        :math:`P(m|\vec{w}) = \sum_{\vec{f}} P(m) P(\vec{f}) \prod_i P(w_i|m, f_i)`.
        We work on the log scale to avoid underflow.
        """
        # Work on log scale to avoid underflow
        log_sign_prior = np.log2(self._sign_prior(sign))
        total_prob = 0.0
        posteriors = self._f_config_posteriors(sign)
        for f_config in self.config._f_configs:
            sum_log_word_posteriors = 0.0
            for i in range(len(self.config._tokens)):
                sum_log_word_posteriors += np.log2(posteriors[(i, f_config[i])])
            total_prob += np.exp2(
                log_sign_prior
                + self.config._log_f_config_prior
                + sum_log_word_posteriors
            )
        return total_prob

    def score(self) -> float:
        r"""
        Calculate the ambiguity score.

        Returns
        -------
        float
            Ambiguity score.

        Notes
        -----
        Since :math:`P(m_a|\vec{w}) + P(m_b|\vec{w})` is not necessarily 1, we
        normalize the posteriors before calculating the entropy.
        """
        prob_pun_sign = self._sign_posterior(self.pun_sign)
        prob_alt_sign = self._sign_posterior(self.alt_sign)

        # Normalize to calculate entropy
        prob_sum = prob_pun_sign + prob_alt_sign
        prob_pun_sign /= prob_sum
        prob_alt_sign /= prob_sum

        if prob_pun_sign <= 0 or prob_alt_sign <= 0:
            return 0  # 0xlog0 = 0
        return -(
            prob_pun_sign * np.log2(prob_pun_sign)
            + prob_alt_sign * np.log2(prob_alt_sign)
        )


class KaoDistinctiveness(KaoMetricBase):
    """Calculates the distinctiveness metric from Kao et al. (2016)."""

    def score(self) -> float:
        """
        Calculate the distinctiveness score.

        Returns
        -------
        float
            Distinctiveness score.
        """
        sampled_pun_sign = []
        sampled_alt_sign = []
        for f_config in self.config._f_configs:
            pun_log_prob = self.config._log_f_config_prior
            posteriors = self._f_config_posteriors(self.pun_sign)
            for i in range(len(self.config._tokens)):
                pun_log_prob += np.log2(posteriors[(i, f_config[i])])

            alt_log_prob = self.config._log_f_config_prior
            posteriors = self._f_config_posteriors(self.alt_sign)
            for i in range(len(self.config._tokens)):
                alt_log_prob += np.log2(posteriors[(i, f_config[i])])

            sampled_pun_sign.append(np.exp2(pun_log_prob))
            sampled_alt_sign.append(np.exp2(alt_log_prob))

        kl1 = entropy(sampled_pun_sign, sampled_alt_sign)
        kl2 = entropy(sampled_alt_sign, sampled_pun_sign)
        return kl1 + kl2
