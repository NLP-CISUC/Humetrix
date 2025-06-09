# This code is based on the implementation by hhexiy and VioletPeng
# Check the original version in:
# https://github.com/hhexiy/pungen

from argparse import ArgumentParser
from pathlib import Path

import polars as pl
from humetrix.skipgram import build_vocabulary


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--corpus', type=Path,
                        help='Corpus file path (.parquet)',
                        required=True)
    parser.add_argument('--vocab', type=Path,
                        help='Vocabulary file path (.csv)',
                        required=True)
    parser.add_argument('--output', type=Path,
                        help='Output directory to save preprocessed corpus',
                        required=True)
    args = parser.parse_args()
    return args

def main(args):
    # Build vocabulary
    vocab = build_vocabulary(args.vocab)
    unk_id = vocab.filter(pl.col('ngram') == '<unk>')['index'].item()

    # Read the corpus and convert to indices
    corpus = (pl.scan_parquet(args.corpus)
              .with_row_index()
              .select(pl.col('index').alias('text id'),
                      pl.col('text').str.split(' ').alias('token'))
              .explode('token')
              .join(vocab.lazy(), left_on='token', right_on='ngram', how='left')
              .select(pl.col('text id'),
                      pl.col('index').fill_null(unk_id))
              .group_by('text id', maintain_order=True)
              .agg(pl.col('index').alias('token ids'))
              .select(pl.col('token ids')))

    # Save corpus converted to indices
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_id_filepath = output_dir / 'corpus_ids.parquet'
    corpus.sink_parquet(corpus_id_filepath, compression='zstd',
                        compression_level=22, statistics=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
