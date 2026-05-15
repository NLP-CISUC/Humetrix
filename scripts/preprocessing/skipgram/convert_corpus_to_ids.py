# This code is based on the implementation by hhexiy and VioletPeng
# Check the original version in:
# https://github.com/hhexiy/pungen

import logging
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
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
                        help='Output file path (.parquet)',
                        required=True)
    args = parser.parse_args()
    return args

def main(args):
    logger = logging.getLogger('convert_corpus')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    # Build vocabulary
    logger.info('Loading vocabulary...')
    vocab = build_vocabulary(args.vocab)
    unk_id = vocab.filter(pl.col('ngram') == '<unk>')['index'].item()
    vocab_lazy = vocab.lazy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f'Reading corpus from {args.corpus} in batches...')
    pf = pq.ParquetFile(args.corpus)
    
    writer = None
    
    for i, batch in enumerate(pf.iter_batches(batch_size=500000, columns=['text'])):
        df = pl.from_arrow(batch)
        
        processed = (df.lazy().with_row_index()
                  .select(pl.col('index').alias('text id'),
                          pl.col('text').str.split(' ').alias('token'))
                  .explode('token')
                  .join(vocab_lazy, left_on='token', right_on='ngram', how='left')
                  .select(pl.col('text id'),
                          pl.col('index').fill_null(unk_id))
                  .group_by('text id', maintain_order=True)
                  .agg(pl.col('index').cast(pl.UInt64).alias('token ids'))
                  .select(pl.col('token ids'))).collect()
                  
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
