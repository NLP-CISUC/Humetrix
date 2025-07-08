import polars as pl


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
    window_size = max_dist - min_dist
    unk_id_lit = pl.lit(unk_id, dtype=pl.UInt64)

    context_windows = (sentences_df
                       # Remove texts with length <= (max_dist - min_dit + 1)
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

