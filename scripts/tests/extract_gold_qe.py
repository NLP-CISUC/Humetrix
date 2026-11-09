from pathlib import Path

import polars as pl

RESULTS_DIR = Path("results/quantum_entropy")
OUTPUT = Path("tests/gold/quantum_entropy.jsonl")
CORPORA = {
    "en": "semeval",
    "es": "HAHA@IberLEF2021",
    "fr": "joker_clef_fr",
    "pt": "puntuguese",
    "zh": "chumor",
}
N_EXAMPLES = 3

dfs = []
for language, corpus in CORPORA.items():
    df = (
        pl.read_ndjson(RESULTS_DIR / f"{corpus}.jsonl", infer_schema_length=None)
        .select(["text", "QE-U + GloVe", "QE-U + HF", "QE-I + GloVe", "QE-I + HF"])
        .sample(N_EXAMPLES, seed=42)
        .with_columns(
            pl.lit(language).alias("language"), pl.lit(corpus).alias("corpus")
        )
    )
    dfs.append(df)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pl.concat(dfs).write_ndjson(OUTPUT)
