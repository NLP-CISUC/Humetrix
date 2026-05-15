# This code is based on the implementation by hhexiy and VioletPeng
# Check the original version in:
# https://github.com/hhexiy/pungen

import logging
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
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
                        help='Output file path (.parquet)',
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
    logger = logging.getLogger('context_windows')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    # Build vocabulary
    logger.info('Loading vocabulary...')
    vocab = build_vocabulary(args.vocab)
    unk_id = vocab.filter(pl.col('ngram') == '<unk>')['index'].item()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f'Reading corpus from {args.corpus} in batches...')
    pf = pq.ParquetFile(args.corpus)
    
    writer = None
    
    for i, batch in enumerate(pf.iter_batches(batch_size=100000)):
        df = pl.from_arrow(batch)
        context_lazy = create_context_windows(df.lazy(), args.max_dist, args.min_dist, unk_id)
        processed = context_lazy.collect()
        processed = processed.with_columns([
            pl.col('token ids').cast(pl.UInt64),
            pl.col('context').cast(pl.List(pl.UInt64))
        ])
        out_batch = processed.to_arrow()
        
        if writer is None:
            writer = pq.ParquetWriter(args.output, out_batch.schema, compression='zstd', compression_level=22)
            
        writer.write_table(out_batch)
        
        if (i + 1) % 10 == 0:
            logger.info(f'Processed batch {i + 1}')
            
    if writer:
        writer.close()
        
    logger.info(f'Finished processing. Saved to {args.output}')

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

