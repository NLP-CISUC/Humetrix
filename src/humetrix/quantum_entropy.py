from abc import abstractmethod
from typing import List, Tuple, Union

import numpy as np
import spacy
import torch
from gensim.models import KeyedVectors
from spacy.language import Language
from transformers import Pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QuantumEntropy():
    def __init__(self, sentence: str,
                 embeddings: Union[KeyedVectors, Pipeline],
                 spacy_model: Language) -> None:
        super().__init__()
        self.text = sentence
        self.embeddings = embeddings
        self.spacy_model = spacy_model

    def tokenize_sentence(self) -> Tuple[List[str], List[str]]:
        if 'sentencizer' not in self.spacy_model.pipe_names:
            punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
            punct_chars.extend([',', ';'])
            self.spacy_model.add_pipe('sentencizer', config={'punct_chars': punct_chars})

        with self.spacy_model.select_pipes(enable='sentencizer'):
            doc = self.spacy_model(self.text)

        setup, punchline = list(doc.sents)
        setup_toks = [token.lower_ for token in setup]
        punchline_toks = [token.lower_ for token in punchline]
        return setup_toks, punchline_toks

    def density_matrix(self, tokens: List[str]) -> Union[torch.Tensor, None]:
        if isinstance(self.embeddings, KeyedVectors):
            tok_embeddings = [self.embeddings[token]
                              for token in tokens
                              if token in self.embeddings]
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
        d_matrix = torch.matmul(norm_embeddings.unsqueeze(-1),
                                norm_embeddings.unsqueeze(-2))
        d_matrix = torch.mean(d_matrix, dim=0)
        return d_matrix

    @abstractmethod
    def score(self) -> float:
        ...


class QuantumUncertainty(QuantumEntropy):
    """Liu and Hou (2023) QE-Uncertainty scoring"""

    def score(self) -> float:
        setup_toks, _ = self.tokenize_sentence()
        d_matrix = self.density_matrix(setup_toks)
        entropy = d_matrix * torch.log(d_matrix)
        entropy[entropy.isnan()] = 0 # 0xlog0 = 0
        entropy = -torch.trace(entropy)
        return entropy.item()


class QuantumIncongruity(QuantumEntropy):
    """Liu and Hou (2023) QE-Incongruity scoring"""

    def score(self) -> float:
        setup_toks, punchline_toks = self.tokenize_sentence()
        setup_d_matrix = self.density_matrix(setup_toks)
        punch_d_matrix = self.density_matrix(punchline_toks)

        if setup_d_matrix is None:
            return None
        if punch_d_matrix is None:
            return None

        sp_d_matrix = punch_d_matrix * setup_d_matrix

        # Setup and punchline uncertainty
        sp_entropy = sp_d_matrix * torch.log(sp_d_matrix)
        sp_entropy[sp_entropy.isnan()] = 0 # 0xlog0 = 0
        sp_entropy = -torch.trace(sp_entropy)

        # Setup uncertainty
        setup_entropy = setup_d_matrix * torch.log(setup_d_matrix)
        setup_entropy[setup_entropy.isnan()] = 0 # 0xlog0 = 0
        setup_entropy = -torch.trace(setup_entropy)

        return (sp_entropy - setup_entropy).item()
