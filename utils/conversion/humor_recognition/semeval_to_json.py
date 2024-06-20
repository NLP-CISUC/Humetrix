# python utils/conversion/humor_recognition/semeval_to_json.py -td ../../Resources/Corpora/semeval2017_task7/data/test/subtask1-heterographic-test.xml -tl ../../Resources/Corpora/semeval2017_task7/data/test/subtask1-heterographic-test.gold -md ../../Resources/Corpora/semeval2017_task7/data/test/subtask1-homographic-test.xml -ml ../../Resources/Corpora/semeval2017_task7/data/test/subtask1-homographic-test.gold
import re
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
from string import punctuation

import pandas as pd
import spacy

parser = ArgumentParser()
parser.add_argument('--heterographic_data', '-td',
                    help='Path do heterographic puns XML data.',
                    required=True, type=Path)
parser.add_argument('--heterographic_labels', '-tl',
                    help='Path do heterographic puns gold annotation.',
                    required=True, type=Path)
parser.add_argument('--homographic_data', '-md',
                    help='Path do homographic puns XML data.',
                    required=True, type=Path)
parser.add_argument('--homographic_labels', '-ml',
                    help='Path do homographic puns gold annotation.',
                    required=True, type=Path)
args = parser.parse_args()
new_data = list()


def detokenize(text):
    text = ' '.join(text)
    punct = re.escape(punctuation)
    text = re.sub(rf'([{punct}])\W(?=[{punct}])', r'\1', text)
    text = re.sub(rf'(?<=\w)\W([{punct}])', r'\1', text)
    text = re.sub(r'(?<=\w)\'\W(?=\w)', r"'", text)
    return text


heterographic_labels = pd.read_csv(args.heterographic_labels,
                                   sep='\t',
                                   names=['id', 'label'])
heterographic_tree = ET.parse(args.heterographic_data)
heterographic_root = heterographic_tree.getroot()
for text in heterographic_root.findall('text')[:10]:
    id_ = text.attrib['id']
    new_id = f'semeval.{id_}'
    tokens = [str(tok.text) for tok in text.findall('word')]
    detokenized_text = detokenize(tokens)
    label = heterographic_labels.query(f'id == "{id_}"')['label'].values[0]
    new_data.append({'id': new_id,
                     'text': detokenized_text,
                     'label': label})

homographic_labels = pd.read_csv(args.homographic_labels,
                                 sep='\t',
                                 names=['id', 'label'])
homographic_tree = ET.parse(args.homographic_data)
homographic_root = homographic_tree.getroot()
for text in homographic_root.findall('text'):
    id_ = text.attrib['id']
    new_id = f'semeval.{id_}'
    tokens = [str(tok.text) for tok in text.findall('word')]
    detokenized_text = detokenize(tokens)
    label = homographic_labels.query(f'id == "{id_}"')['label'].values[0]
    new_data.append({'id': new_id,
                     'text': detokenized_text,
                     'label': label})

new_df = pd.DataFrame(new_data)
new_df = new_df.set_index('id', drop=True)

# Filter out texts
punct_chars = spacy.pipeline.Sentencizer.default_punct_chars
punct_chars.extend([',', ';'])
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('sentencizer', config={'punct_chars': punct_chars})
with nlp.select_pipes(enable='sentencizer'):
    docs = pd.Series(nlp.pipe(new_df['text']), index=new_df.index)
num_sents = docs.apply(lambda x: len(list(x.sents)))
new_df = new_df.loc[num_sents == 2, :]

new_df.to_json('data/humor_recognition/semeval.json', force_ascii=False,
               indent=4, orient='index')
