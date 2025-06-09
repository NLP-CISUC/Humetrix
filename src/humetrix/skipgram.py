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

