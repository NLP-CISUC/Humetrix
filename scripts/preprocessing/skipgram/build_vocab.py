import logging
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter

import polars as pl
import pyarrow.parquet as pq

def parse_args():
    parser = ArgumentParser(description='Build vocabulary from preprocessed corpus')
    parser.add_argument('--corpus', '-c',
                        help='Input corpus parquet file (from large_corpora.py)',
                        required=True, type=Path)
    parser.add_argument('--output', '-o',
                        help='Output vocabulary csv file',
                        required=True, type=Path)
    parser.add_argument('--max-vocab', '-m',
                        help='Maximum vocabulary size',
                        type=int, default=50000)
    parser.add_argument('--min-freq', '-f',
                        help='Minimum frequency threshold',
                        type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger = logging.getLogger('build_vocab')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logger.info(f'Reading corpus from {args.corpus} in batches to save memory...')
    
    vocab_counts = Counter()
    try:
        pf = pq.ParquetFile(args.corpus)
    except Exception as e:
        logger.error(f"Failed to open parquet file: {e}")
        return

    logger.info(f'Found {pf.num_row_groups} row groups in the parquet file.')
    
    # Process each batch manually to sidestep Polars' streaming engine limitations
    for i, batch in enumerate(pf.iter_batches(batch_size=500000, columns=['text'])):
        df = pl.from_arrow(batch)
        counts = (df
                  .select(pl.col('text').str.split(' ').alias('ngram'))
                  .explode('ngram')
                  .filter(pl.col('ngram').str.len_chars() > 0)
                  .group_by('ngram')
                  .len()
                  .rename({'len': 'freq'}))
                  
        # Update global counter efficiently without creating intermediate lists
        vocab_counts.update(dict(counts.iter_rows()))
        
        if (i + 1) % 10 == 0:
            logger.info(f'Processed batch {i + 1}')

    logger.info('Finished reading batches. Compiling final vocabulary...')
    
    # Convert the aggregated Python counter back to a Polars DataFrame
    vocab_df = pl.DataFrame({
        'ngram': list(vocab_counts.keys()),
        'freq': list(vocab_counts.values())
    })
    
    logger.info('Sorting vocabulary...')
    vocab_df = vocab_df.sort('freq', descending=True)

    if args.min_freq > 0:
        logger.info(f'Filtering for words with frequency >= {args.min_freq}')
        vocab_df = vocab_df.filter(pl.col('freq') >= args.min_freq)
    
    if args.max_vocab > 0 and len(vocab_df) > args.max_vocab:
        logger.info(f'Truncating vocabulary to top {args.max_vocab} words')
        vocab_df = vocab_df.head(args.max_vocab)
        
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f'Saving vocabulary of size {len(vocab_df)} to {args.output}')
    vocab_df.write_csv(args.output)
    logger.info('Done!')

if __name__ == '__main__':
    main()
