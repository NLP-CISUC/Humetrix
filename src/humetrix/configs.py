"""
Definition of constants: language codes, spacy models, transformer models, and content word POS tags.
"""

SPACY_MODELS = {
    'pt': 'pt_core_news_lg',
    'en': 'en_core_web_trf',
    'fr': 'fr_dep_news_trf',
    'es': 'es_dep_news_trf',
    'zh': 'zh_core_web_trf',
}
TRANSFORMER_MODELS = {
    'pt': 'neuralmind/bert-base-portuguese-cased',
    'en': 'bert-base-cased',
    'fr': 'camembert-base',
    'es': 'dccuchile/bert-base-spanish-wwm-cased',
    'zh': 'bert-base-chinese',
}
CONTENT_WORD_TAGS = {
    'pt': {'ADV', 'AUX', 'NOUN', 'PROPN', 'VERB'},
    'en': {
        'FW',
        'JJ',
        'JJR',
        'JJS',
        'NN',
        'NNS',
        'NNP',
        'NNPS',
        'RB',
        'RBR',
        'RBS',
        'VB',
        'VBD',
        'VBG',
        'VBN',
        'VBP',
        'VBZ',
    },
    'fr': {'ADV', 'AUX', 'NOUN', 'PROPN', 'VERB'},
    'es': {'ADV', 'AUX', 'NOUN', 'PROPN', 'VERB'},
}
