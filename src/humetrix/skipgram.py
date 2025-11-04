import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def build_vocabulary(vocab_file):
    vocab = (pl.read_csv(vocab_file, has_header=True)
             .select(pl.col('ngram')))

    # Add special symbols
    special_symbols = pl.DataFrame({'ngram': ['<pad>', '<unk>', '<s>', '</s>']},
                                   schema={'ngram': pl.Utf8})

    # Pad to multiple of 8
    pad_len = 8 - ((len(vocab) + 4) % 8)
    pad_df = pl.DataFrame({'ngram': []}, schema={'ngram': pl.Utf8})
    if pad_len != 8:
        pad_df = pl.DataFrame({'ngram': [f'madeupword{i:04d}' for i in range(pad_len)]},
                              schema={'ngram': pl.Utf8})

    vocab = pl.concat([vocab, special_symbols, pad_df]).with_row_index()
    return vocab

def create_context_windows(sentences_df, max_dist, min_dist, unk_id):
    """
    Create context windows for distant skip-gram model.

    sentences_df: LazyDataFrame with 'token ids' column of type list[u64].
    max_dist: Maximum distance for context window.
    min_dist: Minimum distance for context window.
    unk_id: ID for unknown tokens (e.g., <unk>).
    """
    min_length = max_dist - min_dist + 1
    unk_id_lit = pl.lit(unk_id, dtype=pl.UInt64)

    context_windows = (sentences_df
                       # Remove texts with length <= (max_dist - min_dist + 1)
                       .filter(pl.col('token ids').list.len() > min_length)
                       # Replicate one row for each token (to create individual context windows)
                       .with_columns(pl.int_ranges(pl.col('token ids').list.len()).alias('token index'))
                       .explode('token index')
                       # Pad sides to ensure every token has same window size
                       .with_columns(unk_id_lit.repeat_by(max_dist).list.concat(pl.col('token ids')).alias('token ids'))
                       .with_columns(pl.col('token ids').list.concat(unk_id_lit.repeat_by(max_dist)).alias('token ids'))
                       # Fix token ids because of padding
                       .with_columns(pl.col('token index') + max_dist)
                       # Create context windows
                       .with_columns(
                           (pl.col('token index') - max_dist).clip(lower_bound=0).alias('left start'),
                           (pl.col('token index') - min_dist).clip(lower_bound=0).alias('left end'),
                           (pl.col('token index') + min_dist + 1).alias('right start'),
                           (pl.col('token index') + max_dist + 1).alias('right end'))
                       .with_columns((pl.col('left end') - pl.col('left start')).alias('left length'),
                                     (pl.col('right end') - pl.col('right start')).alias('right length'))
                       .with_columns(pl.col('token ids').list.slice(pl.col('left start'), pl.col('left length')).alias('left'),
                                     pl.col('token ids').list.slice(pl.col('right start'), pl.col('right length')).alias('right'))
                       .select(pl.col('token ids').list.get(pl.col('token index')),
                               pl.col('left').list.concat(pl.col('right')).alias('context')))
    return context_windows

# Implementation of SGNS.
# Based on the implementation by hhexiy/pungen.

# TODO: Fix model to be skipgram instead of CBOW
class SGNS(nn.Module):
    def __init__(self, vocab, embedding_dim, vocab_counts=None,
                 negative_samples=20):
        super().__init__()
        self.vocab_df = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples

        self.word_to_idx = {row['ngram']: row['index'] for row in vocab.to_dicts()}
        self.unk_id = self.word_to_idx.get('<unk>')

        self.embeddings = nn.Embedding(self.vocab_size, embedding_dim, sparse=True)
        self.output_embeddings = nn.Embedding(self.vocab_size, embedding_dim, sparse=True)

        # Initialize weights
        initrange = 0.5 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.output_embeddings.weight.data.uniform_(-initrange, initrange)

        self.neg_sampling_weights = None
        if vocab_counts is not None:
            freq_df = (self.vocab_df.join(vocab_counts, on='ngram', how='left')
                            .fill_null(1)
                            .sort('index')
                            .with_columns(pl.col('freq').pow(0.75).alias('pow freq'))
                            .with_columns(pl.col('pow freq').truediv(pl.col('pow freq').sum()).alias('weight')))
            self.neg_sampling_weights = freq_df.select('weight').to_torch().squeeze(1)

    def get_word_idx(self, word: str) -> int:
        return self.word_to_idx.get(word, self.unk_id)

    def predict_prob(self, target_word_idx: int, context_word_indices: List[int]) -> float:
        device = self.embeddings.weight.device
        context_indices_tensor = torch.LongTensor(context_word_indices).to(device)
        context_emb = self.embeddings(context_indices_tensor).mean(dim=0)

        all_output_embs = self.output_embeddings.weight
        all_scores = torch.matmul(all_output_embs, context_emb)

        probs = F.softmax(all_scores, dim=0)
        return probs[target_word_idx].item()

    def forward(self, target_words, context_words):
        # Negative sampling
        device = target_words.device
        batch_size = target_words.size(0)
        context_size = context_words.size(1)
        num_neg_words = context_size * self.negative_samples
        if self.neg_sampling_weights is not None:
            negative_words = (torch.multinomial(self.neg_sampling_weights,
                                               batch_size * num_neg_words,
                                               replacement=True)
                                   .view(batch_size, -1))
        else:
            negative_words = torch.randint(0, self.vocab_size, (batch_size, num_neg_words))
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
