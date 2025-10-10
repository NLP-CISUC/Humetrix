import logging
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import spacy

parser = ArgumentParser()
parser.add_argument('--input', '-i',
                    help='Input corpus txt (one sentence per line)',
                    required=True, type=Path)
parser.add_argument('--model', '-m',
                    help='Spacy model to use',
                    required=False, type=str,
                    default=None)
parser.add_argument('--no-spacy',
                    help='Skip Spacy processing if input is already tokenized (with whitespace)',
                    action='store_true')
args = parser.parse_args()

logger = logging.getLogger('convert')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Check arguments
if not args.no_spacy and not args.model:
    logger.error('No Spacy model specified. Use --model to specify a model.')
    exit(1)
if args.no_spacy and args.model:
    logger.warning('Ignoring Spacy model because --no-spacy is set.')

logger.info(f'Processing {args.input}')
df = (pl.scan_csv(args.input, separator='\0', has_header=False, new_columns=['text'])
      .with_columns(pl.col('text').str.replace_all(r'\s+', ' ').str.strip_chars()))

if not args.no_spacy:
    logger.info('Loading Spacy model')

    # Only sentencize and tokenize
    nlp = spacy.load(args.model, disable=['tagger', 'parser', 'ner'])
    nlp.enable_pipe('senter')
    nlp.max_length = 3000000

    def process_text(text):
        doc = nlp(text)
        sents = [' '.join([f'{token.text}' for token in sent])
                 for sent in doc.sents
                 if len(sent) > 0]
        return sents

    df = (df.with_columns(pl.col('text').map_elements(process_text, return_dtype=pl.List(pl.String)))
          .explode('text'))

# Save processed data in parquet to save space
savepath = Path('data/skipgram/') / args.input.with_suffix('.parquet').name
savepath.parent.mkdir(parents=True, exist_ok=True)

logger.info(f'Start processing')
df.sink_parquet(savepath, compression='zstd', compression_level=22, statistics=False)
logger.info(f'Saved processed data to {savepath}.')

