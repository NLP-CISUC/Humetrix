# ------------ Semeval 2017 task 7 ------------
python scripts/experiments/run_qe.py \
    --corpus data/humor_recognition/semeval.json \
    --language en \
    --glove ../../Resources/Embeddings/English/glove_s300.gensim \
    --huggingface FacebookAI/xlm-roberta-base \
    --incongruity --uncertainty

# ------------ Humicroedit ------------
python scripts/experiments/run_qe.py \
    --corpus data/humor_recognition/humicroedit.json \
    --language en \
    --glove ../../Resources/Embeddings/English/glove_s300.gensim \
    --huggingface FacebookAI/xlm-roberta-base \
    --incongruity --uncertainty

# ------------ JOKER CLEF 2023 (English) ------------
python scripts/experiments/run_qe.py \
    --corpus data/humor_recognition/joker_clef_en.json \
    --language en \
    --glove ../../Resources/Embeddings/English/glove_s300.gensim \
    --huggingface FacebookAI/xlm-roberta-base \
    --incongruity --uncertainty

# ------------ JOKER CLEF 2023 (French) ------------
python scripts/experiments/run_qe.py \
    --corpus ../../data/humor_recognition/joker_clef_fr.json \
    --language fr \
    --glove ../../../../Resources/Embeddings/French/WE_models/glove_s300.gensim \
    --huggingface FacebookAI/xlm-roberta-base \
    --incongruity --uncertainty

# ------------ JOKER CLEF 2023 (Spanish) ------------
python ../../scripts/experiments/run_qe.py \
    --corpus ../../data/humor_recognition/joker_clef_es.json \
    --language es \
    --huggingface FacebookAI/xlm-roberta-base \
    --incongruity --uncertainty

# ------------ HAHA@IberLEF 2019 ------------
python ../../scripts/experiments/run_qe.py \
    --corpus ../../data/humor_recognition/HAHA@IberLEF2019.json \
    --language es \
    --huggingface FacebookAI/xlm-roberta-base \
    --incongruity --uncertainty

# ------------ HAHA@IberLEF 2021 ------------
python ../../scripts/experiments/run_qe.py \
    --corpus ../../data/humor_recognition/HAHA@IberLEF2021.json \
    --language es \
    --huggingface FacebookAI/xlm-roberta-base \
    --incongruity --uncertainty

# ------------ HUHU@IberLEF 2023 ------------
python ../../scripts/experiments/run_qe.py \
    --corpus ../../data/humor_recognition/HUHU@IberLEF2023.json \
    --language es \
    --huggingface FacebookAI/xlm-roberta-base \
    --incongruity --uncertainty

# ------------ André Clemêncio ------------
python ../../scripts/experiments/run_qe.py \
    --corpus ../../data/humor_recognition/clemencio.json \
    --language pt \
    --glove ../../../../Resources/Embeddings/Portuguese/glove_s300.gensim \
    --huggingface FacebookAI/xlm-roberta-base \
    --incongruity --uncertainty

# ------------ Puntuguese ------------
python ../../scripts/experiments/run_qe.py \
    --corpus ../../data/humor_recognition/puntuguese.json \
    --language pt \
    --glove ../../../../Resources/Embeddings/Portuguese/glove_s300.gensim \
    --huggingface FacebookAI/xlm-roberta-base \
    --incongruity --uncertainty
