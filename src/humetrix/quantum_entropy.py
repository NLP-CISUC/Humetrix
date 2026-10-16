"""
This module implements the quantum entropy-based humor metrics from Liu and Hou (2023).

References
----------
[1] Yang Liu and Yuexian Hou. 2023. Mining Effective Features Using Quantum Entropy for Humor Recognition. In Findings of the Association for Computational Linguistics: EACL 2023, May 2023. Association for Computational Linguistics, Dubrovnik, Croatia, 2048–2053. Retrieved August 23, 2023 from https://aclanthology.org/2023.findings-eacl.152
"""

from abc import abstractmethod
from typing import List, Tuple, Union

import numpy as np
import spacy
import torch
from gensim.models import KeyedVectors
from spacy.language import Language
from transformers import Pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QuantumEntropy:
    """
    A base class for quantum entropy-based humor metrics.

    This class provides general methods for tokenizing sentences and calculating
    density matrices. Specific humor metrics can be implemented by subclassing
    this class and implementing the `score` method.

    Parameters
    ----------
    sentence : str
        The sentence to be analyzed.
    embeddings : Gensim embeddings or Transformers feature extraction pipeline
        The embedding model to use for token representations.
    spacy_model : spaCy model
        The spaCy language model for tokenization.
    mean_vector : Torch tensor or None
        Average vector to remove anisotropy.
    """

    def __init__(
        self,
        sentence: str,
        embeddings: Union[KeyedVectors, Pipeline],
        spacy_model: Language,
        mean_vector: Union[None, torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.text = sentence
        self.embeddings = embeddings
        self.spacy_model = spacy_model
        self.mean_vector = mean_vector

    def tokenize_sentence(self) -> Tuple[List[str], List[str]]:
        """
        Tokenize the sentence into setup and punchline.

        Returns
        -------
        tuple containing two lists of str:
            The tokens in the joke setup and punchline, respectively.

        Notes
        -----
        The spaCy model is modified to include a sentencizer component.
        """
        if 'sentencizer' not in self.spacy_model.pipe_names:
            punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
            punct_chars.extend([',', ';'])
            self.spacy_model.add_pipe(
                'sentencizer', config={'punct_chars': punct_chars}
            )

        with self.spacy_model.select_pipes(enable='sentencizer'):
            doc = self.spacy_model(self.text)

        setup, punchline = list(doc.sents)
        setup_toks = [token.lower_ for token in setup]
        punchline_toks = [token.lower_ for token in punchline]
        return setup_toks, punchline_toks

    def token_embeddings(self, tokens: List[str]) -> torch.Tensor:
        r"""
        Calculate the embeddings for each given token.

        Parameters
        ----------
        tokens : list of str
            The tokens to calculate word embeddings for.

        Returns
        -------
        torch.Tensor
            Word embeddings.

        Notes
        -----
        If the embedding model is a Transformer, we average the subtokens into one single word embedding.
        """
        if isinstance(self.embeddings, KeyedVectors):
            tok_embeddings = [
                self.embeddings[token] for token in tokens if token in self.embeddings
            ]
        else:
            tokenized = self.embeddings.tokenizer(
                tokens, is_split_into_words=True, return_tensors="pt"
            )
            tokenized = tokenized.to(self.embeddings.model.device)
            word_ids = tokenized.word_ids()
            with torch.no_grad():
                outputs = self.embeddings.model(**tokenized)
            last_hidden_state = outputs.last_hidden_state[0]

            tok_embeddings = []
            for word_id in range(len(tokens)):
                token_indices = [i for i, w in enumerate(word_ids) if w == word_id]
                if not token_indices:
                    continue
                token_embeddings = last_hidden_state[token_indices]
                tokens_avg = torch.mean(token_embeddings, dim=0).cpu()
                tok_embeddings.append(tokens_avg)
        tok_embeddings = np.array(tok_embeddings)
        tok_embeddings = torch.tensor(tok_embeddings).to(device)
        return tok_embeddings

    def density_matrix(self, tokens: List[str]) -> Union[torch.Tensor, None]:
        r"""
        Calculate the density matrix for a list of tokens.

        Parameters
        ----------
        tokens : list of str
            The tokens to calculate the density matrix for.

        Returns
        -------
        torch.Tensor or None
            The density matrix, or None if no valid embeddings were found.

        Notes
        -----
        The density matrix is calculated as the mean of the outer products of
        the normalized token embeddings:
        :math:`\rho = \frac{1}{|l|} \sum_{i=1}^l |w_i\rangle\langle w_i|`,
        where :math:`|w_i\rangle = \frac{\vec{w_i}}{\|\vec{w_i}\|}`.

        If the underlying QuantumEntropy object has a mean vector, all embeddings
        are first centralized before normalization.
        """
        tok_embeddings = self.token_embeddings(tokens)

        if len(tok_embeddings) == 0:
            return None

        if self.mean_vector is not None:
            tok_embeddings -= self.mean_vector.to(tok_embeddings)

        dim = tok_embeddings[0].shape[0]
        d_matrix = torch.zeros(dim, dim)

        norm_embeddings = torch.nn.functional.normalize(tok_embeddings)
        d_matrix = torch.matmul(
            norm_embeddings.unsqueeze(-1), norm_embeddings.unsqueeze(-2)
        )
        d_matrix = torch.mean(d_matrix, dim=0)
        return d_matrix

    def von_neumann_entropy(
        self, d_matrix: torch.Tensor, hermitian: bool = True
    ) -> float:
        r"""
        Calculate the von Neumann entropy of a density matrix.

        Parameters
        ----------
        d_matrix : torch.Tensor
            The density matrix, with trace equal to 1.
        hermitian : bool, default=True
            Whether `d_matrix` is Hermitian. If True, eigenvalues are computed
            with a symmetric solver; otherwise, a general solver is used and
            only the real part of the eigenvalues if kept.

        Returns
        _______
        float
            The von Neumann entropy.

        Notes
        -----
        The entropy is calculated from the eigenvalues:
        :math:`-\sum_i \lambda_i \log \lambda_i`, ignoring eigenvalues
        smaller than :math:`10^{-12}`. We use ln.
        """
        eigenvalues = (
            torch.linalg.eigvalsh(d_matrix)
            if hermitian
            else torch.linalg.eigvals(d_matrix).real
        )
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
        return entropy.item()

    @abstractmethod
    def score(self) -> Union[float, None]:
        """
        Calculate the humor score.
        """
        ...


class QuantumUncertainty(QuantumEntropy):
    """Liu and Hou (2023) quantum entropy Uncertainty scoring."""

    def score(self) -> Union[float, None]:
        """
        Calculate the quantum uncertainty score.

        Returns
        -------
        float or None
            The quantum uncertainty score, or None if the density matrix
            could not be calculated.
        """
        setup_toks, _ = self.tokenize_sentence()
        d_matrix = self.density_matrix(setup_toks)

        if d_matrix is None:
            return None

        entropy = self.von_neumann_entropy(d_matrix)
        return entropy


class QuantumIncongruity(QuantumEntropy):
    """Liu and Hou (2023) quantum entropy Incongruity scoring."""

    def score(self) -> Union[float, None]:
        """
        Calculate the quantum incongruity score.

        Returns
        -------
        float or None
            The quantum incongruity score, or None if the density matrices
            could not be calculated.
        """
        setup_toks, punchline_toks = self.tokenize_sentence()
        setup_d_matrix = self.density_matrix(setup_toks)
        punch_d_matrix = self.density_matrix(punchline_toks)

        if setup_d_matrix is None:
            return None
        if punch_d_matrix is None:
            return None

        sp_d_matrix = torch.matmul(punch_d_matrix, setup_d_matrix)
        # Normalize the combined matrix so its trace (sum of diagonal probabilities) equals 1
        sp_trace = torch.trace(sp_d_matrix)
        if sp_trace > 0:
            sp_d_matrix = sp_d_matrix / sp_trace

        # Setup and punchline uncertainty
        sp_entropy = self.von_neumann_entropy(sp_d_matrix, hermitian=False)

        # Setup uncertainty
        setup_entropy = self.von_neumann_entropy(setup_d_matrix)

        return sp_entropy - setup_entropy
