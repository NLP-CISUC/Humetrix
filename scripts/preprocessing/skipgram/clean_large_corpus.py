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
    parser.add_argument('--no-data', action="store_true",
                        help='Avoid building skipgram data')
    args = parser.parse_args()
    return args

def build_vocabulary(corpus, max_vocab, threshold):
    print('Building vocabulary')
    vocab = (corpus.explode('lemma')
             .group_by('lemma').len()
             .sort('len', descending=True)
             .with_columns(pl.col('len').cast(pl.UInt32)))
    if args.max_vocab > 0:
        vocab = vocab.head(max_vocab)
    if args.threshold > 0:
        vocab = vocab.filter(pl.col('len') >= threshold)

    # Add special symbols
    print('Finalizing vocabulary')
    special_symbols = pl.LazyFrame({'lemma': ['<pad>', '<unk>', '<s>', '</s>'],
                                    'len': [1, 1, 1, 1]},
                                   schema={'lemma': pl.Utf8, 'len': pl.UInt32})
    vocab = pl.concat([vocab, special_symbols])
    return vocab

def build_skipgram(corpus, vocab, max_dist, min_dist):
    print('Building skipgram data')
    min_length = max_dist - min_dist + 1
    window_size = max_dist - min_dist

    skipgram = (corpus.lazy()
                # Remove texts with length <= (max_dist - min_dit + 1)
                .filter(pl.col('lemma').list.len() > min_length)
                # Replicate one row for each token (to create individual context windows)
                .with_columns(pl.int_ranges(pl.col('lemma').list.len()).alias('token id'))
                .explode('token id')
                # Pad sides to ensure every token has same window size
                .with_columns(pl.lit('<unk>').repeat_by(max_dist).list.concat(pl.col('lemma')).alias('lemma'))
                .with_columns(pl.col('lemma').list.concat(pl.lit('<unk>').repeat_by(max_dist)).alias('lemma'))
                # Fix token ids because of padding
                .with_columns(pl.col('token id') + max_dist)
                # Convert lemmas to vocab ids
                .explode('lemma')
                .join(vocab.with_row_index(), on='lemma')
                .group_by(['text id', 'token id'], maintain_order=True)
                .agg(pl.col('index').alias('lemma id'))
                # Create context windows
                .with_columns((pl.col('token id') - max_dist).clip(lower_bound=0).alias('left start'),
                              (pl.col('token id') - min_dist).clip(lower_bound=0).alias('left end'),
                              (pl.col('token id') + min_dist + 1).alias('right start'),
                              (pl.col('token id') + max_dist + 1).alias('right end'))
                .with_columns((pl.col('left end') - pl.col('left start')).alias('left length'),
                              (pl.col('right end') - pl.col('right start')).alias('right length'))
                .with_columns(pl.col('lemma id').list.slice(pl.col('left start'), pl.col('left length')).alias('left'),
                              pl.col('lemma id').list.slice(pl.col('right start'), pl.col('right length')).alias('right'))
                .select(pl.col('lemma id').list.get(pl.col('token id')),
                        pl.col('left').list.concat(pl.col('right'))))
    return skipgram

def main(args):
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the corpus
    corpus = pl.scan_csv(args.corpus, separator='\t', has_header=False,
                         quote_char=None, new_columns=['text'])

    # Get only lemmas
    corpus = (corpus.with_row_index()
              .with_columns(pl.col('text').str.split(' ').alias('token'))
              .explode('token')
              .select(pl.col('index').alias('text id'),
                      pl.col('token').str.split('|').list.get(1).alias('lemma'))
              .group_by('text id', maintain_order=True).agg(pl.col('lemma')))

    vocab = build_vocabulary(corpus, args.max_vocab, args.threshold)
    print(vocab.collect())

    # print('Saving vocabulary')
    # vocab_filepath = output_dir / 'vocab.parquet'
    # vocab.sink_parquet(vocab_filepath, compression='zstd',
    #                    compression_level=22, statistics=False)

    if not args.no_data:
        skipgram = build_skipgram(corpus, vocab, args.max_dist, args.min_dist)
        print('Saving skipgram data')
        skipgram_filepath = output_dir / 'data.parquet'
        skipgram.sink_parquet(skipgram_filepath, compression='zstd',
                              compression_level=22, statistics=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
