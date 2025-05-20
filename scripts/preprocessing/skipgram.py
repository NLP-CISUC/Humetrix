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

def main(args):
    # Read the corpus
    corpus = pl.read_csv(args.corpus, separator='\t', has_header=False,
                         quote_char=None, new_columns=['text'])

    # Build vocabulary
    vocab = (corpus.select(pl.col('text')
                           .str.split(' ')
                           .explode()
                           .str.split('|')
                           .list.get(1)
                           .alias('lemma'))
             .group_by('lemma').len()
             .sort('len', descending=True))
    if args.max_vocab > 0:
        vocab = vocab.head(args.max_vocab)
    if args.threshold > 0:
        vocab = vocab.filter(pl.col('len') >= args.threshold)

    # Add special symbols
    special_symbols = pl.DataFrame({'lemma': ['<pad>', '<unk>', '<s>', '</s>'],
                                    'len': [1, 1, 1, 1]},
                                   schema={'lemma': pl.Utf8, 'len': pl.UInt32})
    vocab = vocab.vstack(special_symbols)

    # # Pad to multiple of 8
    # pad_len = 8 - (vocab.shape[0] % 8)
    # if pad_len != 8:
    #     pad_df = pl.DataFrame({'lemma': ['<pad>'] * pad_len, 'len': [1] * pad_len},
    #                           schema={'lemma': pl.Utf8, 'len': pl.UInt32})
    #     vocab = vocab.vstack(pad_df)

    print(vocab.write_csv(separator=' ', include_header=False))

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
