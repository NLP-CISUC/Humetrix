from pathlib import Path

import pandas as pd
from tqdm import tqdm
from humetrix import HumorAnalyzer

corpora = {'en': ['semeval', 'humicroedit', 'joker_clef_en'],
           'fr': ['joker_clef_fr'],
           'es': ['joker_clef_es', 'HAHA@IberLEF2019', 'HAHA@IberLEF2021', 'HUHU@IberLEF2023'],
           'pt': ['clemencio', 'puntuguese'],
           'zh': ['chumor']}
paths = {'semeval': 'data/humor_recognition/semeval.json',
         'humicroedit': 'data/humor_recognition/humicroedit.json',
         'joker_clef_en': 'data/humor_recognition/joker_clef_en.json',
         'joker_clef_fr': 'data/humor_recognition/joker_clef_fr.json',
         'joker_clef_es': 'data/humor_recognition/joker_clef_es.json',
         'HAHA@IberLEF2019': 'data/humor_recognition/HAHA@IberLEF2019.json',
         'HAHA@IberLEF2021': 'data/humor_recognition/HAHA@IberLEF2021.json',
         'HUHU@IberLEF2023': 'data/humor_recognition/HUHU@IberLEF2023.json',
         'clemencio': 'data/humor_recognition/clemencio.json',
         'puntuguese': 'data/humor_recognition/puntuguese.json',
         'chumor': 'data/humor_recognition/chumor.json'}
glove = {'en': 'data/embeddings/en/glove_s300.gensim',
         'es': 'data/embeddings/es/glove_s300.gensim',
         'fr': 'data/embeddings/fr/glove_s300.gensim',
         'pt': 'data/embeddings/pt/glove_s300.gensim',
         'zh': 'data/embeddings/zh/glove_s300.gensim'}
xlmr = 'FacebookAI/xlm-roberta-base'
results_path = Path('results/quantum_entropy')
results_path.mkdir(exist_ok=True, parents=True)

for language, datasets in corpora.items():
    analyzer = HumorAnalyzer(language,
                             embeddings_path=glove[language],
                             transformer_model_name=xlmr)
    for corpus in datasets:
        print('- '*10 + corpus + ' -'*10)
        df = pd.read_json(paths[corpus], orient='index').reset_index()

        tqdm.pandas(desc='Incongruity + GloVe')
        df['QE-I + GloVe'] = df['text'].progress_apply(analyzer.quantum_incongruity)
        
        tqdm.pandas(desc='Incongruity + Huggingface')
        df['QE-I + HF'] = df['text'].progress_apply(analyzer.quantum_incongruity,
                                                    backend='transformer')

        tqdm.pandas(desc='Uncertainty + GloVe')
        df['QE-U + GloVe'] = df['text'].progress_apply(analyzer.quantum_uncertainty)
        
        tqdm.pandas(desc='Uncertainty + Huggingface')
        df['QE-U + HF'] = df['text'].progress_apply(analyzer.quantum_uncertainty,
                                                    backend='transformer')

        savepath = (results_path / corpus).with_suffix('.jsonl')
        df.to_json(savepath, orient='records', lines=True, force_ascii=False)
