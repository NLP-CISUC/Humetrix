from pathlib import Path

import pandas as pd
import polars as pl

GOLD_PATHS = {
    "cup": Path("data/humor_interpretation/cup.json"),
    "puntuguese": Path("data/humor_interpretation/puntuguese.json"),
}
LLM_PATH = Path("data/humor_interpretation/llm")


def normalize(col):
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def matches(col):
    pred, gold = normalize(col), normalize(f"{col}_gold")
    exact = pred == gold
    contained = (
        (pred.str.len_chars() >= 3)
        & (gold.str.len_chars() >= 3)
        & (
            pred.str.contains(gold, literal=True)
            | gold.str.contains(pred, literal=True)
        )
    )
    return [
        exact.fill_null(False).alias(f"{col} exact"),
        (exact | contained).fill_null(False).alias(f"{col} lenient"),
    ]


dfs = []
for path in LLM_PATH.glob("*/*.jsonl"):
    model, corpus = path.parent.name, path.stem
    gold = pl.from_pandas(
        pd.read_json(GOLD_PATHS[corpus], orient="index")
        .rename_axis("index")
        .reset_index()
    )
    pred = pl.read_ndjson(path)

    df = pred.join(
        gold.select(["index", "location", "interpretation"]), on="index", suffix="_gold"
    ).with_columns(
        pl.col("text")
        .str.to_lowercase()
        .str.contains(pl.col("location").str.to_lowercase(), literal=True)
        .fill_null(False)
        .alias("location in text"),
        *matches("location"),
        *matches("interpretation"),
    )

    dfs.append(
        df.select(
            pl.lit(model).alias("model"),
            pl.lit(corpus).alias("corpus"),
            pl.len().alias("n"),
            pl.col("location").is_null().sum().alias("failures"),
            pl.col(
                "location in text",
                "location exact",
                "location lenient",
                "interpretation exact",
                "interpretation lenient",
            ).mean(),
        )
    )

print(pl.concat(dfs))
