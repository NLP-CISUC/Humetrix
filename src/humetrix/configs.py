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
# Pipelines used only for tokenization in QE and Suprise metrics.
# Languages not listed here use spacy.blank(language).
TOKENIZER_MODELS = {
    'zh': 'zh_core_web_sm'
}
TRANSFORMER_MODELS = {
    'pt': 'neuralmind/bert-base-portuguese-cased',
    'en': 'bert-base-cased',
    'fr': 'camembert-base',
    'es': 'dccuchile/bert-base-spanish-wwm-cased',
    'zh': 'bert-base-chinese',
}
CONTENT_WORD_TAGS = {
    'pt': {'ADJ', 'ADV', 'AUX', 'NOUN', 'NUM', 'PART', 'PROPN', 'VERB'},
    'en': {'ADJ', 'ADV', 'AUX', 'NOUN', 'NUM', 'PART', 'PROPN', 'VERB'},
    'fr': {'ADJ', 'ADV', 'AUX', 'NOUN', 'NUM', 'PART', 'PROPN', 'VERB'},
    'es': {'ADJ', 'ADV', 'AUX', 'NOUN', 'NUM', 'PART', 'PROPN', 'VERB'},
}

# Hugging Face Hub repository with GloVe, skip-gram and n-gram resources
RESOURCES_REPO = "NLP-CISUC/humetrix-resources"
RESOURCES_REVISION = "cef7d0ac55fae7c154b4bb588de0be2f6d43432b"

# GloVe embeddings hosted elsewere: language -> (repo_id, revision)
EXTERNAL_EMBEDDINGS = {
    "pt": ("nilc-nlp/glove-300d", "44dac98d46b885135a7fe1565f5c67f0f897a4ea")
}
