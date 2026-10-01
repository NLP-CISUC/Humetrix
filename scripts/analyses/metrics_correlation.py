import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns

    return Path, mo, np, pd, pl, plt, sns


@app.cell
def _(Path, pl):
    root = Path("/home/mlinacio/HDD/Documentos/Doutorado/Projeto/Criatividade/Projetos/Recognition/Humetrix")
    data_path = root / "data"
    results_path = root / "results"

    corpora = ["cup", "puntuguese"]
    corpora_enum = pl.Enum(corpora)
    metric_dirs = ["quantum_entropy", "kaoetal", "local_global_surprise"]
    text_attributes = ["text", "label", "location", "interpretation", "char_count"]
    metrics = {
        "QE-U + GloVe": "QE-U (GloVe)",
        "QE-I + GloVe": "QE-I (GloVe)",
        "QE-U + HF": "QE-U (HF)",
        "QE-I + HF": "QE-I (HF)",
        "ambiguity": "Ambiguity",
        "distinctiveness": "Distinctiveness",
        "local global surprise": "Surprise Ratio"}
    return (
        corpora,
        corpora_enum,
        data_path,
        metric_dirs,
        metrics,
        results_path,
        text_attributes,
    )


@app.cell
def _(pd, pl):
    def read_json_index(path):
        return pl.from_dataframe(
            pd.read_json(path, orient="index").reset_index()
        )

    return (read_json_index,)


@app.cell
def _(corpora, corpora_enum, data_path, pl, read_json_index):
    _dfs = []
    for _corpus in corpora:
        _annotations = (
            read_json_index(data_path / "humor_interpretation" / f"{_corpus}.json")
                .with_columns(label=pl.lit(1))
        )

        _labels_path = data_path / "humor_recognition" / f"{_corpus}.json"
        if _labels_path.exists():
            _texts = _annotations.join(
                read_json_index(_labels_path),
                on=["index", "text", "label"],
                how="full",
                coalesce=True
            )
        else:
            _texts = _annotations
        _dfs.append(
            _texts
            .with_columns(corpus=pl.lit(_corpus, corpora_enum))
        )

    texts_df = (
        pl.concat(_dfs, how="diagonal_relaxed")
        .with_columns(
            n_chars=pl.col("text").str.len_chars(),
            is_homographic=(
                pl.col("location").str.strip_chars().str.to_lowercase()
                ==
                pl.col("interpretation").str.strip_chars().str.to_lowercase()
            )
        )
    )
    texts_df
    return (texts_df,)


@app.cell
def _(corpora, corpora_enum, metric_dirs, pl, results_path, text_attributes):
    _dfs = []
    for _metric_dir in metric_dirs:
        for _corpus in corpora:
            _dfs.append(
                pl.read_ndjson(results_path / _metric_dir / f"{_corpus}.jsonl", infer_schema_length=None)
                .drop(text_attributes, strict=False)
                .with_columns(corpus=pl.lit(_corpus, corpora_enum))
            )

    metrics_df = (
        pl.concat(_dfs, how="diagonal_relaxed")
        .unpivot(index=["index", "corpus"], variable_name="metric")
        .drop_nulls("value")
        .with_columns(
            value=pl.when(
                (pl.col("metric") == "local global surprise")
                &
                (pl.col("value") == -1)
            )
                .then(None)
                .otherwise(pl.col("value"))
        )
    )
    metrics_df
    return (metrics_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Descriptive analysis
    """)
    return


@app.cell
def _(pl, texts_df):
    texts_df.group_by("corpus", "label", "is_homographic").len().sort(pl.all())
    return


@app.cell
def _(metrics_df, pl):
    metrics_df.group_by("corpus", "metric").len().sort(pl.all())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check if every metric row has a corresponding text
    """)
    return


@app.cell
def _(metrics_df, texts_df):
    metrics_df.join(texts_df, on=["index", "corpus"], how="anti").height
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation analysis
    """)
    return


@app.cell
def _(metrics, metrics_df, pl, texts_df):
    df = (
        metrics_df
            .pivot(on="metric", values="value")
            .join(texts_df, on=["index", "corpus"])
            .filter(pl.col("label") == 1)
            .drop_nulls(subset=metrics)
    )
    df.group_by("corpus").len()
    return (df,)


@app.cell
def _(df, metrics, pl):
    corr = {
        _corpus: (
            _corpus_df
            .select(metrics.keys())
            .with_columns(pl.all().rank(method="average"))
            .corr(label="metric")
        )
        for (_corpus,), _corpus_df in df.partition_by("corpus", as_dict=True).items()
    }
    corr_n = dict(df.group_by("corpus").len().iter_rows())
    return corr, corr_n


@app.cell
def _(corr, corr_n, mo):
    def _style(_corr_df):
        return (
            _corr_df.to_pandas()
            .set_index("metric")
            .style
            .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
            .format("{:.2f}")
        )

    for _corpus, _corr_df in corr.items():
        mo.output.append(mo.md(f"### {_corpus} (n={corr_n[_corpus]})"))
        mo.output.append(_style(_corr_df))
    return


@app.cell
def _(corr, corr_n, metrics, mo, np, plt, sns):
    _cup = corr["cup"].drop("metric").to_numpy()
    _puntuguese = corr["puntuguese"].drop("metric").to_numpy()

    _lower = np.tril(np.ones_like(_cup, dtype=bool), k=-1)
    _combined = np.where(_lower, _cup, _puntuguese)
    np.fill_diagonal(_combined, np.nan)

    _labels = [metrics[_metric] for _metric in corr["cup"]["metric"]]

    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        _combined,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(250, 15, s=75, l=45, center="light", as_cmap=True),
        annot=True, fmt=".2f", annot_kws={"size": 9},
        square=True, linewidths=2, linecolor="white",
        xticklabels=_labels, yticklabels=_labels,
        cbar_kws={"label": "Spearman's rho", "shrink": 0.8},
        ax=_ax,
    )
    _ax.tick_params(length=0)
    plt.setp(_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    _ax.set_title(
        f"Lower triangle: CUP (n = {corr_n['cup']})\n"
        f"Upper triangle: Puntuguese (n = {corr_n['puntuguese']})",
        fontsize=9, loc="left"
    )

    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
