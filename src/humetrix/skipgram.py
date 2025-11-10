"""
This module provides functions and a class for working with a distant skip-gram model.

It includes functions for building a vocabulary and creating context windows,
and a PyTorch implementation of a skip-gram with negative sampling (SGNS) model.
The SGNS implementation is based on the one from He et al. (2019)_[1].

References
----------
[1] He He, Nanyun Peng, and Percy Liang. 2019. Pun Generation with Surprise. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019. Association for Computational Linguistics, Minneapolis, 1734–1744. https://doi.org/10.18653/v1/N19-1172
"""

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def build_vocabulary(vocab_file: str) -> pl.DataFrame:
    """
    Build a vocabulary from a file with the Google n-grams format.

    Parameters
    ----------
    vocab_file : str
        Path to the vocabulary file following the format of
        orgtre/google-books-ngram-frequency. We expect at least one columns:
        'ngram' (str).

    Returns
    -------
    polars.DataFrame
        A DataFrame containing the vocabulary with an added 'index' column.

    Notes
    -----
    The vocabulary is padded to ensure its size is a multiple of 8. Special
    symbols <pad>, <unk>, <s>, and </s> are also added to mimic the behavior
    of `fairseq`.
    """
    vocab = pl.read_csv(vocab_file, has_header=True).select(pl.col('ngram'))

    # Add special symbols
    special_symbols = pl.DataFrame(
        {'ngram': ['<pad>', '<unk>', '<s>', '</s>']}, schema={'ngram': pl.Utf8}
    )

    # Pad to multiple of 8
    pad_len = 8 - ((len(vocab) + 4) % 8)
    pad_df = pl.DataFrame({'ngram': []}, schema={'ngram': pl.Utf8})
    if pad_len != 8:
        pad_df = pl.DataFrame(
            {'ngram': [f'madeupword{i:04d}' for i in range(pad_len)]},
            schema={'ngram': pl.Utf8},
        )

    vocab = pl.concat([vocab, special_symbols, pad_df]).with_row_index()
    return vocab


def create_context_windows(
    sentences_df: pl.LazyFrame, max_dist: int, min_dist: int, unk_id: int
) -> pl.LazyFrame:
    """
    Create context windows for distant skip-gram model.

    Parameters
    ----------
    sentences_df : polars.LazyFrame
        A LazyFrame with a 'token ids' column of type list[u64].
    max_dist : int
        Maximum distance for context window.
    min_dist : int
        Minimum distance for context window.
    unk_id : int
        ID for unknown tokens (e.g., <unk>).

    Returns
    -------
    polars.LazyFrame
        A LazyFrame with two columns: 'token ids' (target word) and 'context'
        (list of context words), all converted into vocabulary indices.
    """
    min_length = max_dist - min_dist + 1
    unk_id_lit = pl.lit(unk_id, dtype=pl.UInt64)

    context_windows = (
        sentences_df
        # Remove texts with length <= (max_dist - min_dist + 1)
        .filter(pl.col('token ids').list.len() > min_length)
        # Replicate one row for each token (to create individual context windows)
        .with_columns(
            pl.int_ranges(pl.col('token ids').list.len()).alias('token index')
        )
        .explode('token index')
        # Pad sides to ensure every token has same window size
        .with_columns(
            unk_id_lit.repeat_by(max_dist)
            .list.concat(pl.col('token ids'))
            .alias('token ids')
        )
        .with_columns(
            pl.col('token ids')
            .list.concat(unk_id_lit.repeat_by(max_dist))
            .alias('token ids')
        )
        # Fix token ids because of padding
        .with_columns(pl.col('token index') + max_dist)
        # Create context windows
        .with_columns(
            (pl.col('token index') - max_dist)
            .clip(lower_bound=0)
            .alias('left start'),
            (pl.col('token index') - min_dist)
            .clip(lower_bound=0)
            .alias('left end'),
            (pl.col('token index') + min_dist + 1).alias('right start'),
            (pl.col('token index') + max_dist + 1).alias('right end'),
        )
        .with_columns(
            (pl.col('left end') - pl.col('left start')).alias('left length'),
            (pl.col('right end') - pl.col('right start')).alias('right length'),
        )
        .with_columns(
            pl.col('token ids')
            .list.slice(pl.col('left start'), pl.col('left length'))
            .alias('left'),
            pl.col('token ids')
            .list.slice(pl.col('right start'), pl.col('right length'))
            .alias('right'),
        )
        .select(
            pl.col('token ids').list.get(pl.col('token index')),
            pl.col('left').list.concat(pl.col('right')).alias('context'),
        )
    )
    return context_windows


# Implementation of SGNS.
# Based on the implementation by hhexiy/pungen.
# TODO: Fix predict_prob to be skipgram instead of CBOW


class SGNS(nn.Module):
    """
    An implementation of the skip-gram model with negative sampling.

    This is based on the implementation by hhexiy/pungen.

    Parameters
    ----------
    vocab : polars.DataFrame
        The vocabulary, which should contain 'ngram' and 'index' columns.
    embedding_dim : int
        The dimension of the embeddings.
    vocab_counts : polars.DataFrame, optional
        A DataFrame with word counts for negative sampling. Defaults to None.
    negative_samples : int, optional
        The number of negative samples to use. Defaults to 20.
    """

    def __init__(
        self,
        vocab: pl.DataFrame,
        embedding_dim: int,
        vocab_counts: Optional[pl.DataFrame] = None,
        negative_samples: int = 20,
    ):
        super().__init__()
        self.vocab_df = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples

        self.word_to_idx = {
            row['ngram']: row['index'] for row in vocab.to_dicts()
        }
        self.unk_id = self.word_to_idx.get('<unk>')

        self.embeddings = nn.Embedding(
            self.vocab_size, embedding_dim, sparse=True
        )
        self.output_embeddings = nn.Embedding(
            self.vocab_size, embedding_dim, sparse=True
        )

        # Initialize weights
        initrange = 0.5 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.output_embeddings.weight.data.uniform_(-initrange, initrange)

        self.neg_sampling_weights = None
        if vocab_counts is not None:
            freq_df = (
                self.vocab_df.join(vocab_counts, on='ngram', how='left')
                .fill_null(1)
                .sort('index')
                .with_columns(pl.col('freq').pow(0.75).alias('pow freq'))
                .with_columns(
                    pl.col('pow freq')
                    .truediv(pl.col('pow freq').sum())
                    .alias('weight')
                )
            )
            self.neg_sampling_weights = (
                freq_df.select('weight').to_torch().squeeze(1)
            )

    def get_word_idx(self, word: str) -> int:
        """
        Return the index of a word in the vocabulary.

        Parameters
        ----------
        word : str
            The word to look up.

        Returns
        -------
        int
            The index of the word, or the index of the unknown token if not found.
        """
        return self.word_to_idx.get(word, self.unk_id)

    def predict_prob(
        self, target_word_idx: int, context_word_indices: List[int]
    ) -> float:
        """
        Predict the probability of a target word given a context.

        .. todo::
            Fix `predict_prob` to be skipgram instead of CBOW.


        Parameters
        ----------
        target_word_idx : int
            The index of the target word.
        context_word_indices : list of int
            A list of indices for the context words.

        Returns
        -------
        float
            The probability of the target word.
        """
        device = self.embeddings.weight.device
        context_indices_tensor = torch.LongTensor(context_word_indices).to(
            device
        )
        context_emb = self.embeddings(context_indices_tensor).mean(dim=0)

        all_output_embs = self.output_embeddings.weight
        all_scores = torch.matmul(all_output_embs, context_emb)

        probs = F.softmax(all_scores, dim=0)
        return probs[target_word_idx].item()

    def forward(
        self, target_words: torch.Tensor, context_words: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a forward pass of the model.

        This calculates the loss using negative sampling.

        Parameters
        ----------
        target_words : torch.Tensor
            A tensor of target word indices.
        context_words : torch.Tensor
            A tensor of context word indices.

        Returns
        -------
        torch.Tensor
            The total loss for the batch.
        """
        # Negative sampling
        device = target_words.device
        batch_size = target_words.size(0)
        context_size = context_words.size(1)
        num_neg_words = context_size * self.negative_samples
        if self.neg_sampling_weights is not None:
            negative_words = torch.multinomial(
                self.neg_sampling_weights,
                batch_size * num_neg_words,
                replacement=True,
            ).view(batch_size, -1)
        else:
            negative_words = torch.randint(
                0, self.vocab_size, (batch_size, num_neg_words)
            )
        negative_words = negative_words.to(device)

        target_emb = self.embeddings(target_words)
        context_emb = self.output_embeddings(context_words)
        neg_emb = self.output_embeddings(negative_words)

        pos_score = torch.bmm(context_emb, target_emb.unsqueeze(2)).squeeze(2)
        neg_scores = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze(2)

        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_scores).sum(dim=1).mean()
        total_loss = pos_loss + neg_loss

        return total_loss
