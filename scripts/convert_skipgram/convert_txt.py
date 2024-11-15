import logging
from argparse import ArgumentParser
from pathlib import Path

import spacy
from pyspark.sql import SparkSession

parser = ArgumentParser()
parser.add_argument('--input', '-i',
                    help='Input corpus txt',
                    required=True, type=Path)
parser.add_argument('--model', '-m',
                    help='Spacy model to use',
                    required=True, type=str)
args = parser.parse_args()

logger = logging.getLogger('convert')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

spark = SparkSession.builder.appName("wiki-zh").getOrCreate()

nlp = spacy.load(args.model, exclude=['parser', 'ner'])
nlp.enable_pipe('senter')
nlp.max_length = 2000000
has_lemmatizer = 'lemmatizer' in nlp.pipe_names

def process_text(partition):
    for text in partition:
        doc = nlp(text)
        for sent in doc.sents:
            tokens = [
                    f'{token.text}|{token.lemma_ if has_lemmatizer else token.text}|{token.pos_}'
                    for token in sent
                    ]
            yield ' '.join(tokens)

logger.info(f'Processing {args.input}...')
text_rdd = spark.sparkContext.textFile(str(args.input)).repartition(30000)
processed_rdd = text_rdd.mapPartitions(process_text)

# Save processed data one sentence per line
savepath = Path('data/skipgram/') / args.input.name
savepath.parent.mkdir(parents=True, exist_ok=True)
with savepath.open('w', encoding='utf-8') as f:
    for line in processed_rdd.toLocalIterator():
        f.write(f'{line}\n')
logger.info(f'Saved processed data to {savepath}.')
