# This code is based on the implementation by hhexiy and VioletPeng
# Check the original version in:
# https://github.com/hhexiy/pungen

from argparse import ArgumentParser
from pathlib import Path
import polars as pl

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--corpus', type=Path,
                        help='Corpus file path (.txt)',
                        required=True)
    parser.add_argument('--output', type=Path,
                        help='Output directory to save preprocessed corpus and vocabulary',
                        required=True)
    parser.add_argument('--min-dist', type=int,
                        help='Minimum distance to the word',
                        default=5)
    parser.add_argument('--max-dist', type=int,
                        help='Maximum distance to the word',
                        default=10)
    parser.add_argument('--max-vocab', type=int,
                        help='Maximum vocabulary size',
                        default=-1)
    parser.add_argument('--threshold', type=int,
                        help='Minimum word count threshold',
                        default=-1)
    args = parser.parse_args()
    return args

def build_vocabulary(corpus, max_vocab, threshold):
    vocab = (corpus.explode('lemma')
             .group_by('lemma').len()
             .sort('len', descending=True)
             .collect())
    if args.max_vocab > 0:
        vocab = vocab.head(max_vocab)
    if args.threshold > 0:
        vocab = vocab.filter(pl.col('len') >= threshold)

    # Add special symbols
    special_symbols = pl.DataFrame({'lemma': ['<pad>', '<unk>', '<s>', '</s>'],
                                    'len': [1, 1, 1, 1]},
                                   schema={'lemma': pl.Utf8, 'len': pl.UInt32})
    vocab = vocab.vstack(special_symbols)

    # Pad to multiple of 8
    pad_len = 8 - (len(vocab) % 8)
    if pad_len != 8:
        pad_df = pl.DataFrame({'lemma': [f'madeupword{i:04d}' for i in range(pad_len)],
                               'len': [0] * pad_len},
                              schema={'lemma': pl.Utf8, 'len': pl.UInt32})
        vocab = vocab.vstack(pad_df)
    return vocab

def build_skipgram(corpus, vocab, max_dist, min_dist, unk_id):
    min_length = max_dist - min_dist + 1
    window_size = max_dist - min_dist

    skipgram = (corpus.lazy()
                # Convert lemmas to vocab ids
                .explode('lemma')
                .join(vocab.lazy().with_row_index(), on='lemma')
                .group_by('text', maintain_order=True)
                .agg(pl.col('index').alias('lemma id'))
                # Remove texts with length <= (max_dist - min_dit + 1)
                .filter(pl.col('lemma id').list.len() > min_length)
                # Replicate one row for each token (to create individual context windows)
                .with_columns(pl.int_ranges(pl.col('lemma id').list.len()).alias('token id'))
                .explode('token id')
                # Create context windows
                .with_columns((pl.col('token id') - max_dist).clip(lower_bound=0).alias('left start'),
                              (pl.col('token id') - min_dist).clip(lower_bound=0).alias('left end'),
                              (pl.col('token id') + min_dist + 1).alias('right start'),
                              (pl.col('token id') + max_dist + 1).alias('right end'))
                .with_columns((pl.col('left end') - pl.col('left start')).alias('left length'),
                              (pl.col('right end') - pl.col('right start')).alias('right length'))
                .with_columns(pl.col('lemma id').list.slice(pl.col('left start'), pl.col('left length')).alias('left'),
                              pl.col('lemma id').list.slice(pl.col('right start'), pl.col('right length')).alias('right'))
                # Pad context windows to ensure they are the same size
                .select(pl.col('lemma id').list.get(pl.col('token id')),
                        pl.lit(unk_id).repeat_by(window_size - pl.col('left').list.len()).list.concat(pl.col('left')).alias('left'),
                        pl.col('right').list.concat(pl.lit(unk_id).repeat_by(window_size - pl.col('right').list.len())))
                .select(pl.col('lemma id'), pl.col('left').list.concat(pl.col('right'))))
    return skipgram.collect()

def main(args):
    # Read the corpus
    corpus = pl.scan_csv(args.corpus, separator='\t', has_header=False,
                         quote_char=None, new_columns=['text'])

    # Get only lemmas
    corpus = (corpus.with_columns(pl.col('text').str.split(' ').alias('token'))
              .explode('token')
              .select(pl.col('text'),
                      pl.col('token').str.split('|').list.get(1).alias('lemma'))
              .group_by('text', maintain_order=True).agg(pl.col('lemma')))

    vocab = build_vocabulary(corpus, args.max_vocab, args.threshold)
    unk_id = vocab.with_row_index().filter(pl.col('lemma') == '<unk>')['index'].item()
    skipgram = build_skipgram(corpus, vocab, args.max_dist, args.min_dist, unk_id)

    # Save files
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_filepath = output_dir / 'vocab.parquet'
    skipgram_filepath = output_dir / 'data.parquet'

    vocab.write_parquet(vocab_filepath, compression='zstd',
                        compression_level=22, statistics=False)
    skipgram.write_parquet(skipgram_filepath, compression='zstd',
                            compression_level=22, statistics=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
