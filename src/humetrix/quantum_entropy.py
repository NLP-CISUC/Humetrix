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
    """

    def __init__(
        self,
        sentence: str,
        embeddings: Union[KeyedVectors, Pipeline],
        spacy_model: Language,
    ) -> None:
        super().__init__()
        self.text = sentence
        self.embeddings = embeddings
        self.spacy_model = spacy_model

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
        """
        if isinstance(self.embeddings, KeyedVectors):
            tok_embeddings = [
                self.embeddings[token]
                for token in tokens
                if token in self.embeddings
            ]
        else:
            text = ' '.join(tokens)
            tok_embeddings = self.embeddings(text)[0]
        tok_embeddings = np.array(tok_embeddings)
        tok_embeddings = torch.tensor(tok_embeddings).to(device)

        if len(tok_embeddings) == 0:
            return None
        dim = tok_embeddings[0].shape[0]
        d_matrix = torch.zeros(dim, dim)

        norm_embeddings = torch.nn.functional.normalize(tok_embeddings)
        d_matrix = torch.matmul(
            norm_embeddings.unsqueeze(-1), norm_embeddings.unsqueeze(-2)
        )
        d_matrix = torch.mean(d_matrix, dim=0)
        return d_matrix

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

        eigenvalues = torch.linalg.eigvalsh(d_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
        return entropy.item()


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
        sp_eigenvalues = torch.linalg.eigvals(sp_d_matrix).real
        sp_eigenvalues = sp_eigenvalues[sp_eigenvalues > 1e-12]
        sp_entropy = -torch.sum(sp_eigenvalues * torch.log(sp_eigenvalues))

        # Setup uncertainty
        setup_eigenvalues = torch.linalg.eigvalsh(setup_d_matrix)
        setup_eigenvalues = setup_eigenvalues[setup_eigenvalues > 1e-12]
        setup_entropy = -torch.sum(setup_eigenvalues * torch.log(setup_eigenvalues))

        return (sp_entropy - setup_entropy).item()
