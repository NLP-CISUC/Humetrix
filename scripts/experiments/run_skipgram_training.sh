# English

uv run python scripts/training/train_skipgram.py \
    --context-windows data/preprocessed_skipgram/bookcorpus/context_windows.parquet \
    --vocab data/ngrams/en/1gram.csv \
    --number-examples 502802107 \
    --chunk-size 1000000 \
    --batch-size 2048 \
    --output results/skipgram/en.pt 

# Portuguese

uv run python scripts/training/train_skipgram.py \
    --context-windows data/preprocessed_skipgram/brwac/context_windows.parquet \
    --vocab data/ngrams/pt/1gram.csv \
    --number-examples 502802107 \
    --chunk-size 1000000 \
    --batch-size 2048 \
    --output results/skipgram/pt.pt 

# French

uv run python scripts/training/train_skipgram.py \
    --context-windows data/preprocessed_skipgram/croissant/context_windows.parquet \
    --vocab data/ngrams/fr/1gram.csv \
    --number-examples 502802107 \
    --chunk-size 1000000 \
    --batch-size 2048 \
    --output results/skipgram/fr.pt 

# Spanish

uv run python scripts/training/train_skipgram.py \
    --context-windows data/preprocessed_skipgram/sbwc/context_windows.parquet \
    --vocab data/ngrams/es/1gram.csv \
    --number-examples 502802107 \
    --chunk-size 1000000 \
    --batch-size 2048 \
    --output results/skipgram/es.pt 

# Chinese

uv run python scripts/training/train_skipgram.py \
    --context-windows data/preprocessed_skipgram/wiki-zh/context_windows.parquet \
    --vocab data/ngrams/zh/1gram.csv \
    --number-examples 502802107 \
    --chunk-size 1000000 \
    --batch-size 2048 \
    --output results/skipgram/zh.pt 

