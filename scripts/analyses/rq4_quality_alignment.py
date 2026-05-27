import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    from scipy.stats import kruskal
    from scikit_posthocs import posthoc_dunn

    return Path, kruskal, mo, pl, plt, posthoc_dunn, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RQ4: Alignment with Quality
    """)
    return


@app.cell
def _(Path, pl):
    results_paths = [
        Path(
            "/home/mlinacio/HDD/Documentos/Doutorado/Projeto/Criatividade/Projetos/Recognition/Humetrix/results/quantum_entropy"
        ),
        Path(
            "/home/mlinacio/HDD/Documentos/Doutorado/Projeto/Criatividade/Projetos/Recognition/Humetrix/results/kaoetal"
        ),
        Path(
            "/home/mlinacio/HDD/Documentos/Doutorado/Projeto/Criatividade/Projetos/Recognition/Humetrix/results/local_global_surprise"
        ),
    ]

    corpora_enum = pl.Enum(["humicroedit", "expunations", "HAHA@IberLEF2021"])
    return corpora_enum, results_paths


@app.cell
def _(corpora_enum, pl, results_paths):
    dfs = []
    for results_path in results_paths:
        for filepath in results_path.iterdir():
            if filepath.stem not in corpora_enum.categories:
                continue
            df = (
                pl.read_ndjson(filepath, infer_schema_length=None)
                .with_columns(pl.lit(filepath.stem).alias("corpus"))
                .drop(
                    ["label", "location", "interpretation", "char_count"], strict=False
                )
            )

            if "funniness" not in df.columns:
                continue

            dfs.append(df)

    combined_df = pl.concat(dfs, how="diagonal_relaxed")
    df = (
        combined_df.drop("text")
        .drop_nulls(subset="funniness")
        .unpivot(index=["index", "corpus", "funniness"], variable_name="metric")
        .unique(subset=["index", "corpus", "metric"], keep="first")
        .with_columns(pl.col("corpus").cast(corpora_enum))
        .sort("corpus")
    )
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preliminary analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Quantile analysis
    """)
    return


@app.cell
def _(df, mo, pl):
    quantiles = (
        df.group_by(["corpus", "metric"])
        .agg(
            pl.col("value").quantile(0.25).alias("Q1"),
            pl.col("value").quantile(0.5).alias("Q2"),
            pl.col("value").quantile(0.75).alias("Q3"),
        )
        .with_columns((pl.col("Q3") - pl.col("Q1")).alias("IQR"))
        .with_columns(
            (pl.col("Q1") - 1.5 * pl.col("IQR")).alias("F1"),
            (pl.col("Q3") + 1.5 * pl.col("IQR")).alias("F2"),
        )
    )

    mo.ui.table(
        quantiles.sort("corpus", "metric"),
        selection=None,
        pagination=False,
        format_mapping={
            "Q1": "{:.4g}",
            "Q2": "{:.4g}",
            "Q3": "{:.4g}",
            "Q4": "{:.4g}",
            "IQR": "{:.4g}",
            "F1": "{:.4g}",
            "F2": "{:.4g}",
        },
    )
    return (quantiles,)


@app.cell
def _(df, pl, quantiles):
    df_with_quantiles = df.join(quantiles, on=["corpus", "metric"])
    is_low_outlier = pl.col("value") < pl.col("F1")
    is_high_outlier = pl.col("value") > pl.col("F2")
    df_no_outliers = df_with_quantiles.filter(
        ~(is_low_outlier | is_high_outlier)
    ).select(df.columns)

    df_no_outliers.group_by(["corpus", "metric"], maintain_order=True).len()
    return (df_no_outliers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation analysis
    """)
    return


@app.cell
def _(df_no_outliers, mo, pl):
    corr_df = (
        df_no_outliers.group_by(["corpus", "metric"])
        .agg(
            pl.corr(
                pl.col("funniness"),
                pl.col("value"),
                method='spearman'
            ).alias("correlation"),
        )
        .sort("corpus", "metric")
    )

    mo.ui.table(
        corr_df,
        selection=None,
        pagination=False,
        format_mapping={"correlation": "{:.4g}"},
    )
    return (corr_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Funniness distributions
    """)
    return


@app.cell
def _(corpora_enum, df, sns):
    sns.set_theme(style="whitegrid", font_scale=1.2)

    corpus_palette = {
        "humicroedit": "#005A9C",
        "expunations": "#1890FF",
        "HAHA@IberLEF2021": "#E63946",
    }

    sns.displot(
        df,
        x="funniness",
        hue="corpus",
        kind="kde",
        bw_adjust=1.5,
        common_norm=False,
        palette=corpus_palette,
        fill=True,
        alpha=0.1,
        linewidth=1.5,
        hue_order=corpora_enum.categories,
    )
    return (corpus_palette,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Funniness vs. metrics
    """)
    return


@app.cell
def _(corpora_enum, corpus_palette, df_no_outliers, sns):
    _g = sns.lmplot(
        data=df_no_outliers,
        x="funniness",
        y="value",
        hue="corpus",
        col="metric",
        col_wrap=2,
        palette=corpus_palette,
        facet_kws={"sharex": False, "sharey": False, "legend_out": True},
        scatter_kws={"alpha": 0.15, "s": 10},
        line_kws={"linewidth": 1.5},
        hue_order=corpora_enum.categories,
    )

    for handle in _g.legend.legend_handles:
        handle.set_alpha(1.0)
    _g
    return


@app.cell
def _(corpora_enum, corpus_palette, corr_df, mo, plt, sns):
    plt.figure(figsize=(15, 6))
    _g = sns.barplot(
        data=corr_df,
        x="correlation",
        y="metric",
        hue="corpus",
        palette=corpus_palette,
        hue_order=corpora_enum.categories,
        order=['QE-U + GloVe', 'QE-U + HF', 'QE-I + GloVe', 'QE-I + HF'],
    )

    for _container in _g.containers:
        _g.bar_label(_container, fontsize=10, fmt="%.3f", padding=5)

    _g.axvline(0, color="black", linewidth=1, linestyle="--")
    _g.set_title("Spearman Correlation between Metrics and Funniness")
    _g.set_xlabel("Correlation")
    _g.set_ylabel("Metric")
    _g.legend(title="Corpus", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    mo.mpl.interactive(_g)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Binned analysis
    """)
    return


@app.cell
def _(df_no_outliers, pl):
    binned_df = df_no_outliers.with_columns(
        pl.col("funniness")
        .qcut(3, labels=["low", "mid", "high"])
        .over(["corpus", "metric"])
        .cast(pl.Enum(['low', 'mid', 'high']))
        .alias("funniness bin")
    )
    binned_df
    return (binned_df,)


@app.cell
def _(binned_df, corpora_enum, corpus_palette, sns):
    _g = sns.catplot(
        data=binned_df,
        x="funniness bin",
        y="value",
        hue="corpus",
        col="metric",
        col_wrap=2,
        order=["low", "mid", "high"],
        sharey=False,
        palette=corpus_palette,
        kind="point",
        dodge=0.2,
        capsize=0.1,
        markers=["o", "s", "D"],
        linewidth=2.5,
        hue_order=corpora_enum.categories,
        col_order=['QE-U + GloVe', 'QE-U + HF', 'QE-I + GloVe', 'QE-I + HF']
    )

    _g.set_titles(col_template="{col_name}")
    _g.set_ylabels("Value")
    _g.set_xlabels("Funniness bin")
    _g.legend.set_title("Corpus")

    _g
    return


@app.cell
def _(binned_df, corpora_enum, corpus_palette, sns):
    sns.catplot(
        data=binned_df,
        x="funniness bin",
        y="value",
        hue="corpus",
        col="metric",
        col_wrap=2,
        kind="box",
        order=["low", "mid", "high"],
        palette=corpus_palette,
        sharey=False,
        showfliers=False,
        width=0.6,
        linewidth=1.2,
        hue_order=corpora_enum.categories,
        col_order=['QE-U + GloVe', 'QE-U + HF', 'QE-I + GloVe', 'QE-I + HF']
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Statsitical tests (No outliers)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Binned Analysis (Kruskall-Wallis H-test)
    """)
    return


@app.cell
def _(binned_df, corpora_enum, kruskal, mo, pl, posthoc_dunn):
    _results = []
    _metrics = binned_df["metric"].unique()
    for _metric in _metrics:
        for _corpus in corpora_enum.categories:
            _binned_metric = (
                binned_df.filter(
                    (pl.col("metric") == _metric) & (pl.col("corpus") == _corpus)
                )
                .select(["value", "funniness bin"])
                .sort('funniness bin')
                .group_by("funniness bin", maintain_order=True)
                .all()["value"]
                .to_numpy()
            )
            if _binned_metric.shape == (3,):
                _statistic, _pvalue = kruskal(*_binned_metric)

                _dunn_result = posthoc_dunn(_binned_metric, p_adjust="bonferroni")
                _dunn_result.columns = ["low", "mid", "high"]
                _dunn_result.index = ["low", "mid", "high"]

                _results.append(
                    {
                        "metric": _metric,
                        "corpus": _corpus,
                        "statistic": _statistic,
                        "p value": _pvalue,
                        "significantly different": _pvalue < 0.05,
                        "low vs mid dunn": _dunn_result.loc["low", "mid"],
                        "low vs high dunn": _dunn_result.loc["low", "high"],
                        "mid vs high dunn": _dunn_result.loc["mid", "high"],
                    }
                )

    _results_df = (
        pl.DataFrame(_results)
            .sort(
                pl.col('metric').cast(
                    pl.Enum([
                        'QE-U + GloVe',
                        'QE-I + GloVe', 
                        'QE-U + HF',
                        'QE-I + HF'
                    ])
                ),
                pl.col('corpus').cast(corpora_enum)
            )
    )

    mo.ui.table(
        _results_df,
        selection=None,
        pagination=False,
        format_mapping={
            "statistic": "{:.4g}",
            "p value": "{:.4g}",
            "low vs mid dunn": "{:.4g}",
            "low vs high dunn": "{:.4g}",
            "mid vs high dunn": "{:.4g}",
        },
    )
    return


if __name__ == "__main__":
    app.run()
