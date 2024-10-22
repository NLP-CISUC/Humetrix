import spacy
from humetrix.configs import SPACY_MODELS
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("wiki-zh").getOrCreate()
nlp = spacy.load(SPACY_MODELS["zh"], disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')

def process_text(text):
    doc = nlp(text)
    sentences = list()
    for sent in doc.sents:
        tokens = [f'{tok.text}|{tok.text}|{tok.pos_}' for tok in sent]
        sentences.append(' '.join(tokens))
    return '\n'.join(sentences)

filepath = '../../Resources/Corpora/wiki-zh-simplified-seg.txt'
text_rdd = spark.sparkContext.textFile(str(filepath)).repartition(2000)
processed_rdd = text_rdd.mapPartitions(lambda part: [process_text(' '.join(part))])
sentences_rdd = processed_rdd.flatMap(lambda part: part.split('\n'))
print(sentences_rdd.take(5))
