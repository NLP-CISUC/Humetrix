import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    import torch
    import torch.nn.functional as F

    from itertools import combinations
    from pathlib import Path

    from polars import selectors as cs
    from scipy.stats import mannwhitneyu, pearsonr, wilcoxon

    return (
        F,
        Path,
        combinations,
        cs,
        mannwhitneyu,
        mo,
        pl,
        sns,
        torch,
        wilcoxon,
    )


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
    anisotropy_df = (
        pl.read_ndjson("results/anisotropy.jsonl")
        .with_columns(pl.col("corpus").cast(corpora_enum))
        .unpivot(index=["corpus", "embedding model"],
                variable_name="similarity")
    )
    anisotropy_df
    return (anisotropy_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Anisotropy and intrasentence similarity analysis
    """)
    return


@app.cell
def _(anisotropy_df, mo, pl):
    _sim_enum = pl.Enum([
        "intersentence similarity",
        "intrasentence similarity",
        "intra-inter difference"
    ])
    mo.ui.table(
        anisotropy_df.group_by("embedding model", "similarity")
        .agg(minimum=pl.col("value").min(),
             maximum=pl.col("value").max())
        .sort("embedding model", pl.col("similarity").cast(_sim_enum)),
        selection=None,
        pagination=False,
        format_mapping={
            "minimum": "{:.3f}",
            "maximum": "{:.3f}"

        }
    )
    return


@app.cell
def _(anisotropy_df, mo, sns):
    _g = sns.catplot(
        anisotropy_df,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does centering embeddings solve anisotropy?
    """)
    return


@app.cell
def _(Path, corpora_enum, corpora_langs, cs, pl):
    _uncentered_results_path = Path("results/quantum_entropy/")

    _uncentered_dfs = []
    for _corpus in corpora_langs:
        _corpus_path = _uncentered_results_path / f"{_corpus}.jsonl"
        _df = (
            pl.read_ndjson(_corpus_path, infer_schema_length=None)
                .with_columns(corpus=pl.lit(_corpus_path.stem, corpora_enum))
        )
        _uncentered_dfs.append(_df)

    uncentered_df = (
        pl.concat(_uncentered_dfs, how="diagonal_relaxed")
        .filter(pl.col("label").is_not_null())
        .select("index", "corpus", "label", "QE-U + GloVe", "QE-U + HF", "QE-I + GloVe", "QE-I + HF")
        .drop_nulls()
        .drop(cs.starts_with("QE-I"))
    )

    uncentered_df
    return (uncentered_df,)


@app.cell
def _(Path, pl):
    _centered_results_path = Path("results/quantum_entropy_centered/")

    _centered_dfs = []
    for _corpus_path in _centered_results_path.iterdir():
        _df = pl.read_ndjson(_corpus_path, infer_schema_length=None)
        _centered_dfs.append(_df)

    centered_df = (
        pl.concat(_centered_dfs, how="diagonal_relaxed")
        .filter(pl.col("label").is_not_null())
        .select("index", "QE-U + GloVe (centered)", "QE-U + HF (centered)")
    )

    centered_df
    return (centered_df,)


@app.cell
def _(centered_df, pl, uncentered_df):
    df = (
        uncentered_df
        .join(centered_df, on="index")
        .with_columns(
            label=pl.when(pl.col("label") == 1)
                .then(pl.lit("Humor"))
                .otherwise(pl.lit("Non-humor"))
        )
        .sort(pl.col("corpus"))
    )
    df
    return (df,)


@app.cell
def _(df, pl):
    df_unpivot = (
        df
        .unpivot(
            on=["QE-U + GloVe", "QE-U + HF", "QE-U + GloVe (centered)", "QE-U + HF (centered)"],
            index=["index", "label", "corpus"],
            variable_name="metric",
        )
        .drop_nulls("value")
        .sort(pl.col("corpus"))
    )
    df_unpivot
    return (df_unpivot,)


@app.cell
def _(df_unpivot, pl):
    df_microedit = (
        df_unpivot.filter(pl.col("corpus").is_in(["humicroedit", "puntuguese"]))
        .with_columns(pl.col("index").str.strip_chars_end(r".[NH]").alias("pair"))
        .drop("index")
        .pivot(on="label", index=["metric", "corpus", "pair"])
    )
    return (df_microedit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statistical Tests
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Statistical difference
    """)
    return


@app.cell
def _(corpora_enum, df_microedit, df_unpivot, mannwhitneyu, mo, pl, wilcoxon):
    _results = {
        "metric": list(),
        "corpus": list(),
        "test": list(),
        "statistic": list(),
        "p-value": list(),
        "n humor": list(),
        "n non-humor": list(),
        "auc": list(),
    }

    for _metric in df_unpivot["metric"].unique():
        for _corpus in df_unpivot["corpus"].unique():
            _humor = df_unpivot.filter(
                (pl.col("corpus") == _corpus)
                & (pl.col("label") == "Humor")
                & (pl.col("metric") == _metric)
            )["value"]
            _non_humor = df_unpivot.filter(
                (pl.col("corpus") == _corpus)
                & (pl.col("label") == "Non-humor")
                & (pl.col("metric") == _metric)
            )["value"]
            _test = mannwhitneyu

            if _corpus in ["humicroedit", "puntuguese"]:
                _humor = df_microedit.filter(
                    (pl.col("metric") == _metric) & (pl.col("corpus") == _corpus)
                )["Humor"]
                _non_humor = df_microedit.filter(
                    (pl.col("metric") == _metric) & (pl.col("corpus") == _corpus)
                )["Non-humor"]
                _test = wilcoxon

            _statistic, _pvalue = _test(
                _humor, _non_humor, alternative="greater", nan_policy="omit"
            )

            _auc = _statistic / (_humor.len() * _non_humor.len())
            if _corpus in ["humicroedit", "puntuguese"]:
                _diff = _humor - _non_humor
                _auc = (_diff > 0).mean() + 0.5 * (_diff == 0).mean()

            _results["metric"].append(_metric)
            _results["corpus"].append(_corpus)
            _results["test"].append(_test.__name__)
            _results["statistic"].append(_statistic)
            _results["p-value"].append(_pvalue)
            _results["n humor"].append(_humor.len())
            _results["n non-humor"].append(_non_humor.len())
            _results["auc"].append(_auc)

    centered_results_df = pl.DataFrame(_results).with_columns(
        (pl.col("p-value") < 0.05).alias("different distributions")
    )

    mo.ui.table(
        centered_results_df
        .select("corpus", "metric", "auc")
        .pivot(index="corpus", on="metric")
        .select(pl.col("corpus").cast(corpora_enum),
                "QE-U + GloVe",
                "QE-U + GloVe (centered)",
                "QE-U + HF",
                "QE-U + HF (centered)")
        .sort("corpus"),
        format_mapping={
            "QE-U + GloVe": "{:.2%}",
            "QE-U + HF": "{:.2%}",
            "QE-U + GloVe (centered)": "{:.2%}",
            "QE-U + HF (centered)": "{:.2%}",
        },
        selection=None,
        pagination=False,
    )
    return (centered_results_df,)


@app.cell
def _(centered_results_df, corpora_enum, mo, pl):
    mo.ui.table(
        centered_results_df
        .select("metric", "corpus", "p-value")
        .pivot(on="metric", index="corpus")
        .select(
            pl.col("corpus").cast(corpora_enum),
            "QE-U + GloVe",
            "QE-U + GloVe (centered)",
            "QE-U + HF",
            "QE-U + HF (centered)"
        )
        .sort("corpus"),
        format_mapping={
            "QE-U + GloVe": "{:.2g}",
            "QE-U + HF": "{:.2g}",
            "QE-U + GloVe (centered)": "{:.2g}",
            "QE-U + HF (centered)": "{:.2g}",
        },
        selection=None,
        pagination=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Correlation with text length
    """)
    return


@app.cell
def _(df, mo, pl):
    _qe_columns = [
        "QE-U + GloVe",
        "QE-U + GloVe (centered)",
        "QE-U + HF",
        "QE-U + HF (centered)",
    ]

    _length_df = pl.read_ndjson(
        "results/qe_shortcuts.jsonl", infer_schema_length=None
    ).select("index", "n_setup")

    _correlations = (
        df.join(_length_df, on="index")
        .group_by("corpus")
        .agg(
            pl.corr(_col, "n_setup", method="spearman").alias(_col)
            for _col in _qe_columns
        )
        .sort("corpus")
    )

    mo.ui.table(
        _correlations,
        format_mapping={_col: "{:.3f}" for _col in _qe_columns},
        selection=None,
        pagination=False,
    )
    return


if __name__ == "__main__":
    app.run()
