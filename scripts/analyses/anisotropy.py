import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import seaborn as sns
    import torch
    import torch.nn.functional as F

    from itertools import combinations
    from pathlib import Path

    return F, Path, combinations, mo, pl, sns, torch


@app.cell
def _(Path, pl):
    corpora_langs = {
        "semeval": "en",
        "humicroedit": "en",
        "expunations": "en",
        "cup": "en",
        "joker_clef_en": "en",
        "joker_clef_es": "es",
        "HAHA@IberLEF2021": "es",
        "HUHU@IberLEF2023": "es",
        "joker_clef_fr": "fr",
        "joker_clef_pt": "pt",
        "clemencio": "pt",
        "puntuguese": "pt",
        "chumor": "cn",
    }
    corpora_enum = pl.Enum([
        "semeval",
        "humicroedit",
        "expunations",
        "cup",
        "joker_clef_en",
        "joker_clef_es",
        "HAHA@IberLEF2021",
        "HUHU@IberLEF2023",
        "joker_clef_fr",
        "joker_clef_pt",
        "clemencio",
        "puntuguese",
        "chumor",
    ])
    embedding_models = ["GloVe", "XLM-R"]

    mean_vectors_path = Path("results/mean_vectors")
    return corpora_enum, corpora_langs, embedding_models, mean_vectors_path


@app.cell
def _(corpora_enum, pl):
    df = (
        pl.read_ndjson("results/anisotropy.jsonl")
        .with_columns(pl.col("corpus").cast(corpora_enum))
        .unpivot(index=["corpus", "embedding model"],
                variable_name="similarity")
    )
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Anisotropy and intrasentence similarity analysis
    """)
    return


@app.cell
def _(df, mo, pl):
    _sim_enum = pl.Enum([
        "intersentence similarity",
        "intrasentence similarity",
        "intra-inter difference"
    ])
    mo.ui.table(
        df.group_by("embedding model", "similarity")
        .agg(minimum=pl.col("value").min(),
             maximum=pl.col("value").max())
        .sort("embedding model", pl.col("similarity").cast(_sim_enum)),
        selection=None,
        format_mapping={
            "minimum": "{:.3f}",
            "maximum": "{:.3f}"

        }
    )
    return


@app.cell
def _(df, mo, sns):
    _g = sns.catplot(
        df,
        x="value",
        y="corpus",
        hue="embedding model",
        jitter=False,
        col="similarity",
    )


    mo.mpl.interactive(_g.fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyze mean vectors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To address the anisotropy, we want to remove the mean vector of the space from the word embeddings.

    In here, we test if using the own corpora's mean vector is enough for this. If the mean vector of all corpora are close to each other, we can assume that obtaining a vector from a large generic corpus would result in a similar vector.
    """)
    return


@app.cell
def _(
    F,
    combinations,
    corpora_langs,
    embedding_models,
    mean_vectors_path,
    pl,
    torch,
):
    _rows = []

    for _model in embedding_models:
        _mean_vectors = {
            c: F.normalize(torch.load(mean_vectors_path / f"{c}_{_model}.pt").cpu(), dim=0)
            for c in corpora_langs
        }
        for _corpus1, _corpus2 in combinations(corpora_langs, 2):
            if _model == "GloVe" and corpora_langs[_corpus1] != corpora_langs[_corpus2]:
                continue
            sim = (_mean_vectors[_corpus1] @ _mean_vectors[_corpus2]).item()
            _rows.append({
                "model": _model,
                "language 1": corpora_langs[_corpus1],
                "language 2": corpora_langs[_corpus2],
                "corpus 1": _corpus1,
                "corpus 2": _corpus2,
                "similarity": sim,
            })
    mean_vector_similarities = pl.from_records(_rows)
    mean_vector_similarities
    return (mean_vector_similarities,)


@app.cell
def _(mean_vector_similarities, pl):
    (mean_vector_similarities
        .group_by("model", same_language=pl.col("language 1") == pl.col("language 2"))
        .agg(minimum=pl.col("similarity").min(),
             maximum=pl.col("similarity").max())
    )
    return


if __name__ == "__main__":
    app.run()
