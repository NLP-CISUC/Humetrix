from pathlib import Path

import polars as pl

RESULTS_DIR = Path("results")
OUTPUT = Path("tests/gold")
CORPORA = {"en": "cup", "pt": "puntuguese"}
TEXT_COLUMNS = ["text", "location", "interpretation"]
METRICS_COLUMNS = {
    "local_global_surprise": ["local global surprise"],
    "kaoetal": ["ambiguity", "distinctiveness"],
}
N_EXAMPLES = 3

for metric, columns in METRICS_COLUMNS.items():
    dfs = []
    for language, corpus in CORPORA.items():
        df = (
            pl.read_ndjson(RESULTS_DIR / metric / f"{corpus}.jsonl")
            .select(TEXT_COLUMNS + columns)
            .drop_nulls(columns)
        )
        if metric == "kaoetal":
            df = df.sort(pl.col("text").str.len_chars()).head(N_EXAMPLES)
        else:
            df = df.sample(N_EXAMPLES, seed=42)
        df = df.with_columns(
            pl.lit(language).alias("language"), pl.lit(corpus).alias("corpus")
        )
        dfs.append(df)
    pl.concat(dfs).write_ndjson(OUTPUT / f"{metric}.jsonl")
