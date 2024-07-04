# python utils/conversion/humor_recognition/hindi_to_json.py -d ../../Resources/Corpora/Hindi-English-puns/AnnotatedCorpus.txt
import itertools
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser()
parser.add_argument('--data', '-d',
                    help='Path to AnnotatedCorpus.txt of the Hindi-English pun corpus.',
                    required=True, type=Path)
args = parser.parse_args()


# Fix input XML structure
corpus_xml = list()
with args.data.open(encoding='utf-8') as f:
    lines = f.readlines()
    num_instances = len(lines) // 7

    for i in range(num_instances):
        instance = ''.join(lines[i*7:(i+1)*7]).strip()
        instance = '<instance>' + instance + '</instance>'
        corpus_xml.append(instance)

corpus_xml = ''.join(corpus_xml)
corpus_xml = '<corpus>' + corpus_xml + '</corpus>'
corpus_xml = corpus_xml.replace('&', '&amp;').replace('<<', '<')

# Get instances from input XML
ids = list()
texts = list()
labels = list()

root = ET.fromstring(corpus_xml)
for instance in root.iter('instance'):
    class_ = instance.find('class')
    if class_ is None:
        continue

    id_ = instance.find('tweet_id')
    if id_ is None:
        continue

    ids.append(id_.text)
    tokens = [str(w.text) for w in instance.iter('word')]
    texts.append(' '.join(tokens))
    labels.append(1 if class_.text == 'H' else 0)

df = pd.DataFrame({'id': ids, 'text': texts, 'label': labels})
df['id'] = 'hindi.' + df['id']
df = df.set_index('id')
df.to_json('data/humor_recognition/hindi.json',
           orient='index', force_ascii=False,
           indent=4)
