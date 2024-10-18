import argparse
from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import spacy
import torch
from gensim.models import KeyedVectors
from spacy.language import Language
from transformers import Pipeline, pipeline

from humetrix import SPACY_MODELS

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

    def score(self):
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


def parse_args() -> argparse.Namespace:
    """Parse arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', '-e',
                        help='Embeddings file in word2vec format.',
                        required=True, type=Path)
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Run script directly to test"""
    embeddings = KeyedVectors.load(str(args.embeddings))
    pipe = pipeline('feature-extraction', 'FacebookAI/xlm-roberta-base')
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

    for sentence in sentences:
        qe_uncertainty = QuantumUncertainty(sentence, embeddings, 'pt')
        qe_incongruity = QuantumIncongruity(sentence, embeddings, 'pt')
        print(f'{sentence}')
        print(f'QE-Uncertainty GloVe: {qe_uncertainty.score():.2f}')
        print(f'QE-Incongruity GloVe: {qe_incongruity.score():.2f}')
        qe_uncertainty = QuantumUncertainty(sentence, pipe, 'pt')
        qe_incongruity = QuantumIncongruity(sentence, pipe, 'pt')
        print(f'QE-Uncertainty XLM-R: {qe_uncertainty.score():.2f}')
        print(f'QE-Incongruity XLM-R: {qe_incongruity.score():.2f}')
        print('*****************')


if __name__ == '__main__':
    args = parse_args()
    main(args)
