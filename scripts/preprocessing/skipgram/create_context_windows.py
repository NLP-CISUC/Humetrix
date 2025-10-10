# This code is based on the implementation by hhexiy and VioletPeng
# Check the original version in:
# https://github.com/hhexiy/pungen

from argparse import ArgumentParser
from pathlib import Path

import polars as pl
from humetrix.skipgram import build_vocabulary, create_context_windows


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--corpus', type=Path,
                        help='Corpus file path (.parquet) with tokens converted to vocab ids',
                        required=True)
    parser.add_argument('--vocab', type=Path,
                        help='Vocabulary file path (.csv)',
                        required=True)
    parser.add_argument('--output', type=Path,
                        help='Output directory to save preprocessed corpus',
                        required=True)
    parser.add_argument('--min-dist', type=int,
                        help='Minimum distance to the word',
                        default=5)
    parser.add_argument('--max-dist', type=int,
                        help='Maximum distance to the word',
                        default=10)
    args = parser.parse_args()
    return args

def main(args):
    # Build vocabulary
    vocab = build_vocabulary(args.vocab)
    unk_id = vocab.filter(pl.col('ngram') == '<unk>')['index'].item()

    corpus = pl.scan_parquet(args.corpus)
    context_df = create_context_windows(corpus, args.max_dist, args.min_dist, unk_id)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    context_filepath = output_dir / 'context_windows.parquet'
    context_df.sink_parquet(context_filepath, compression='zstd',
                            compression_level=22, statistics=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

