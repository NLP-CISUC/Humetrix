from argparse import ArgumentParser
from itertools import islice
import logging
import os
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import spacy
from tqdm import tqdm


def stream_lines(filepath, pbar):
    with filepath.open('r', encoding='utf-8') as f:
        for line in f:
            pbar.update(len(line.encode('utf-8')))
            stripped_line = line.strip()
            if stripped_line:
                yield stripped_line

def write_parquet(texts, writer, savepath):
    if not texts:
        return writer

    text_arrays = pa.array(processed_texts, type=pa.string())
    out_table = pa.Table.from_arrays([text_arrays], names=['text'])

    if writer is None:
        writer = pq.ParquetWriter(
            savepath,
            out_table.schema,
            compression='zstd',
            compression_level=3,
        )

    writer.write_table(out_table)
    return writer


parser = ArgumentParser()
parser.add_argument(
    '--input',
    '-i',
    help='Input corpus txt (one sentence per line)',
    required=True,
    type=Path,
)
parser.add_argument(
    '--model',
    '-m',
    help='Spacy model to use',
    required=False,
    type=str,
    default=None,
)
parser.add_argument(
    '--no-spacy',
    help='Skip Spacy processing if input is already tokenized (with whitespace)',
    action='store_true',
)
args = parser.parse_args()

logger = logging.getLogger('convert')
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Check arguments
if not args.no_spacy and not args.model:
    logger.error('No Spacy model specified. Use --model to specify a model.')
    exit(1)
if args.no_spacy and args.model:
    logger.warning('Ignoring Spacy model because --no-spacy is set.')

logger.info(f'Processing {args.input}')

savepath = Path('data/skipgram/') / args.input.with_suffix('.parquet').name
savepath.parent.mkdir(parents=True, exist_ok=True)

if not args.no_spacy:
    logger.info('Loading Spacy model')

    nlp = spacy.load(args.model, disable=['parser', 'ner'])
    nlp.add_pipe('sentencizer')
    nlp.max_length = 3000000

    logger.info('Processing text with Spacy in batches')
    writer = None
    write_lines = 50000

    total_bytes = os.path.getsize(args.input)

    processed_texts = []
    with tqdm(
        total=total_bytes,
        unit='B',
        unit_scale=True,
        desc='Processing texts',
    ) as pbar:
        for doc in nlp.pipe(
            stream_lines(args.input, pbar), batch_size=10000, n_process=-1
        ):
            sents = [
                ' '.join(
                    [
                        token.lemma_.lower()
                        if token.lemma_
                        else token.text.lower()
                        for token in sent
                        if not token.is_space
                    ]
                )
                for sent in doc.sents
                if len(sent) > 0
            ]
            processed_texts.extend(sents)

            if len(processed_texts) >= write_lines:
                writer = write_parquet(processed_texts, writer, savepath)
                processed_texts = []
        writer = write_parquet(processed_texts, writer, savepath)

    if writer:
        writer.close()
    logger.info(f'Saved processed data to {savepath}.')

else:
    logger.info(f'Start processing (No Spacy)')
    df = pl.scan_csv(
        args.input,
        separator='\0',
        has_header=False,
        new_columns=['text'],
        quote_char=None,
    ).with_columns(
        pl.col('text').str.replace_all(r'\s+', ' ').str.strip_chars()
    )
    df.sink_parquet(
        savepath, compression='zstd', compression_level=3, statistics=False
    )
    logger.info(f'Saved processed data to {savepath}.')
