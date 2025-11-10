"""
This module provides a centralized API for analyzing humor in text using various metrics.

The HumorAnalyzer class requires at least the analysis language. Depending on the metrics
to be used, additional resources such as word embeddings, skip-gram models, and n-gram models
may need to be provided.

Resources are loaded as needed to optimize performance.
"""

from pathlib import Path
from typing import Dict, Union

from gensim.models import KeyedVectors
import polars as pl
import spacy
from spacy.language import Language
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, Pipeline, pipeline

from .configs import SPACY_MODELS, TRANSFORMER_MODELS
from .kaoetal import KaoAmbiguity, KaoConfig, KaoDistinctiveness
from .quantum_entropy import QuantumIncongruity, QuantumUncertainty
from .skipgram import SGNS, build_vocabulary
from .surprise import LocalGlobalSurprise


class HumorAnalyzer:
    """
    A class to analyze humor in text using different metrics.

    Some of the metrics implemented here require external resources such as word embeddings,
    skip-gram models, and n-gram models.

    Supported metrics include:
    - Quantum Incongruity
    - Quantum Uncertainty
    - Local-Global Surprise
    - Kao et al.'s Ambiguity
    - Kao et al.'s Distinctiveness

    Parameters
    ----------
    language : str
        The language of analysis.
    embeddings_path : str, optional
        Path to gensim-formatted word embeddings file. Defaults to None.
    skipgram_path : str, optional
        Path to the skip-gram model file. Defaults to None.
    ngram_dir_path : str, optional
        Path to the directory containing n-gram models. Defaults to None.
    transformer_model_name : str, optional
        Name of the transformer model to use. Defaults to None.

    Raises
    ------
    ValueError
        If the specified language is not supported.
    """

    def __init__(
        self,
        language: str,
        embeddings_path: Union[str, None] = None,
        skipgram_path: Union[str, None] = None,
        ngram_dir_path: Union[str, None] = None,
        transformer_model_name: Union[str, None] = None,
    ) -> None:
        self.language = language
        if language not in SPACY_MODELS:
            raise ValueError(f'Unsupported language: {language}')
        self._transformer_model_name = transformer_model_name
        self._paths = {
            'embeddings': embeddings_path,
            'skipgram': skipgram_path,
            'ngram_dir': ngram_dir_path,
        }
        self._models = {}

    def _load_model(
        self, name: str
    ) -> Union[Language, Dict, KeyedVectors, pl.DataFrame, SGNS, Pipeline]:
        """
        Load and cache the specified model.

        Parameters
        ----------
        name : str
            The name of the model to load.

        Returns
        -------
        spacy model or dict or gensim KeyedVectors or polars DataFrame or SGNS model or transformers pipeline
            The loaded model.

        Raises
        ------
        ValueError
            If the model type is unknown or required paths are not provided.
        """
        if name not in self._models:
            if name == 'spacy':
                model_name = SPACY_MODELS[self.language]
                self._models[name] = spacy.load(model_name)
            elif name == 'bert':
                model_name = TRANSFORMER_MODELS[self.language]
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                lm = AutoModelForMaskedLM.from_pretrained(model_name)
                self._models[name] = {'tokenizer': tokenizer, 'lm': lm}
            elif name == 'embeddings':
                if not self._paths['embeddings']:
                    raise ValueError(
                        'Please provide `embeddings_path` to the HumorAnalyzer to use word embeddings.'
                    )
                self._models[name] = KeyedVectors.load(
                    self._paths['embeddings']
                )
            elif name == 'ngram1' or name == 'ngram3':
                if not self._paths['ngram_dir']:
                    raise ValueError(
                        'Please provide `ngram_dir_path` to the HumorAnalyzer to use n-gram models.'
                    )
                ngram_dir_path = Path(self._paths['ngram_dir'])
                ngram_path = ngram_dir_path / f'{name[-1]}gram.csv'
                self._models[name] = pl.read_csv(ngram_path)
            elif name == 'kao_models':
                if not self._paths['skipgram'] or not self._paths['ngram_dir']:
                    raise ValueError(
                        'To use Kao et al. (2016) models, provide `skipgram_path` and `ngram_dir_path`.'
                    )
                ngram_dir_path = Path(self._paths['ngram_dir'])
                vocab = build_vocabulary(ngram_dir_path / '1gram.csv')
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )
                sgns = SGNS(vocab, 300)
                sgns.load_state_dict(
                    torch.load(self._paths['skipgram'], map_location=device)
                )
                sgns.eval()
                self._models[name] = {
                    'skipgram': sgns,
                    'ngram1': self._load_model('ngram1'),
                    'ngram3': self._load_model('ngram3'),
                }
            elif name == 'transformer_pipeline':
                if self._transformer_model_name:
                    model_name = self._transformer_model_name
                else:
                    model_name = TRANSFORMER_MODELS[self.language]
                self._models[name] = pipeline('feature-extraction', model_name)
            else:
                raise ValueError(f'Unknown model type: {name}')
        return self._models[name]

    def quantum_incongruity(
        self, sentence: str, backend: str = 'word_embeddings'
    ) -> Union[float, None]:
        """
        Calculate the quantum incongruity of a sentence.

        Parameters
        ----------
        sentence : str
            The sentence to analyze.
        backend : {'word_embeddings', 'transformer'}, optional
            The backend to use for embeddings. Defaults to 'word_embeddings'.

        Returns
        -------
        float or None
            The quantum incongruity score, or None if it cannot be calculated.

        Raises
        ------
        ValueError
            If an unknown backend is specified.

        See Also
        --------
        quantum_entropy.QuantumIncongruity.score : Method that performs the actual scoring.
        """
        if backend == 'word_embeddings':
            embeddings = self._load_model('embeddings')
        elif backend == 'transformer':
            embeddings = self._load_model('transformer_pipeline')
        else:
            raise ValueError(
                f'Unknown backend for quantum incongruity: {backend}'
            )
        scorer = QuantumIncongruity(
            sentence=sentence,
            embeddings=embeddings,
            spacy_model=self._load_model('spacy'),
        )
        return scorer.score()

    def quantum_uncertainty(
        self, sentence: str, backend: str = 'word_embeddings'
    ) -> Union[float, None]:
        """
        Calculate the quantum uncertainty of a sentence.

        Parameters
        ----------
        sentence : str
            The sentence to analyze.
        backend : {'word_embeddings', 'transformer'}, optional
            The backend to use for embeddings. Defaults to 'word_embeddings'.

        Returns
        -------
        float or None
            The quantum uncertainty score, or None if it cannot be calculated.

        Raises
        ------
        ValueError
            If an unknown backend is specified.

        See Also
        --------
        quantum_entropy.QuantumUncertainty.score : Method that performs the actual scoring.
        """
        if backend == 'word_embeddings':
            embeddings = self._load_model('embeddings')
        elif backend == 'transformer':
            embeddings = self._load_model('transformer_pipeline')
        else:
            raise ValueError(
                f'Unknown backend for quantum uncertainty: {backend}'
            )
        scorer = QuantumUncertainty(
            sentence=sentence,
            embeddings=embeddings,
            spacy_model=self._load_model('spacy'),
        )
        return scorer.score()

    def local_global_surprise(
        self, sentence: str, pun_sign: str, alt_sign: str
    ) -> float:
        """
        Calculate the local-global surprise of a sentence.

        Parameters
        ----------
        sentence : str
            The sentence to analyze.
        pun_sign : str
            The pun sign. The word or phrase that is ambiguous or humorous.
        alt_sign : str
            The alternative sign. The word or phrase that is evoked by the pun sign.

        Returns
        -------
        float
            The local-global surprise score.

        See Also
        --------
        surprise.LocalGlobalSurprise.score : Method that performs the actual scoring.
        """
        bert_models = self._load_model('bert')
        scorer = LocalGlobalSurprise(
            sentence=sentence, pun_sign=pun_sign, alt_sign=alt_sign
        )
        return scorer.score(
            tokenizer=bert_models['tokenizer'], lm=bert_models['lm']
        )

    def kao_ambiguity(
        self, sentence: str, pun_sign: str, alt_sign: str
    ) -> float:
        """
        Calculate Kao et al.'s ambiguity of a sentence.

        Parameters
        ----------
        sentence : str
            The sentence to analyze.
        pun_sign : str
            The pun sign. The word or phrase that is ambiguous or humorous.
        alt_sign : str
            The alternative sign. The word or phrase that is evoked by the pun sign.

        Returns
        -------
        float
            Ambiguity score.

        See Also
        --------
        kaoetal.KaoAmbiguity.score : Method that performs the actual scoring.
        """
        kao_models = self._load_model('kao_models')
        config = KaoConfig(
            sentence=sentence,
            language=self.language,
            spacy_model=self._load_model('spacy'),
            skipgram=kao_models['skipgram'],
        )
        scorer = KaoAmbiguity(
            config=config,
            pun_sign=pun_sign,
            alt_sign=alt_sign,
            ngram1=kao_models['ngram1'],
            ngram3=kao_models['ngram3'],
        )
        return scorer.score()

    def kao_distinctiveness(
        self, sentence: str, pun_sign: str, alt_sign: str
    ) -> float:
        """
        Calculate Kao et al.'s distinctiveness of a sentence.

        Parameters
        ----------
        sentence : str
            The sentence to analyze.
        pun_sign : str
            The pun sign. The word or phrase that is ambiguous or humorous.
        alt_sign : str
            The alternative sign. The word or phrase that is evoked by the pun sign.

        Returns
        -------
        float
            Distinctiveness score.

        See Also
        --------
        kaoetal.KaoDistinctiveness.score : Method that performs the actual scoring.
        """
        kao_models = self._load_model('kao_models')
        config = KaoConfig(
            sentence=sentence,
            language=self.language,
            spacy_model=self._load_model('spacy'),
            skipgram=kao_models['skipgram'],
        )
        scorer = KaoDistinctiveness(
            config=config,
            pun_sign=pun_sign,
            alt_sign=alt_sign,
            ngram1=kao_models['ngram1'],
            ngram3=kao_models['ngram3'],
        )
        return scorer.score()
