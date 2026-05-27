import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    from scipy.stats import mannwhitneyu, pearsonr, wilcoxon

    return Path, cs, mannwhitneyu, mo, np, pearsonr, pl, plt, sns, wilcoxon


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RQ1: Discriminative Power
    """)
    return


@app.cell
def _(Path, pl):
    results_path = Path(
        "/home/mlinacio/HDD/Documentos/Doutorado/Projeto/Criatividade/Projetos/Recognition/Humetrix/results/quantum_entropy"
    )

    corpora_enum = pl.Enum(
        [
            "semeval",
            "humicroedit",
            "expunations",
            "joker_clef_en",
            "joker_clef_es",
            "HAHA@IberLEF2021",
            "HUHU@IberLEF2023",
            "joker_clef_fr",
            "joker_clef_pt",
            "clemencio",
            "puntuguese",
            "chumor",
        ]
    )
    return corpora_enum, results_path


@app.cell
def _(corpora_enum, pl, results_path):
    dfs = []
    for filepath in results_path.iterdir():
        if filepath.stem not in corpora_enum.categories:
            continue
        df = (
            pl.read_ndjson(filepath, infer_schema_length=None)
            .with_columns(pl.lit(filepath.stem).alias("corpus"))
            .drop("funniness", strict=False)
        )
        dfs.append(df)

    df = (
        pl.concat(dfs)
        .with_columns(
            pl.when(pl.col("label") == 1)
            .then(pl.lit("Humor"))
            .otherwise(pl.lit("Non-humor"))
            .alias("label"),
            pl.col("corpus").cast(corpora_enum),
        )
        .sort(pl.col("corpus"))
    )
    df
    return (df,)


@app.cell
def _(df, pl):
    df_unpivot = (
        df.drop_nulls()
        .unpivot(
            on=["QE-I + GloVe", "QE-U + GloVe"],
            index=["index", "label", "corpus"],
            variable_name="metric",
        )
        .sort(pl.col("corpus"))
    )
    df_unpivot
    return (df_unpivot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Premliminary analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check null values
    """)
    return


@app.cell
def _(df, pl):
    df.group_by(pl.col("corpus")).agg(
        pl.all().exclude(["index", "text", "label"]).is_null().sum()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since $I(\vec{w}) = U(\vec{s}\vec{p}) - U(\vec{w})$, the values of $I(\vec{w})$ tend to approximate $-U(\vec{w})$ if $U(\vec{s}\vec{p}) \approx 0$. Therefore, we want to check the correlation between Incongruity and Uncertainty to see if this is the case.
    """)
    return


@app.cell
def _(df_unpivot, mo, pl):
    mo.ui.table(
        df_unpivot.pivot("metric", index=["index", "label", "corpus"])
        .group_by(["corpus", "label"])
        .agg(pl.corr("QE-I + GloVe", "QE-U + GloVe").alias("correlation"))
        .sort("corpus", "label")
        .pivot("label", values="correlation"),
        selection=None,
        pagination=False,
        format_mapping={"Humor": "{:.4g}", "Non-humor": "{:.4g}"},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How many sentences have $I(\vec{w}) = -U(\vec{w})$?
    """)
    return


@app.cell
def _(df_unpivot, pl):
    (
        df_unpivot.pivot("metric", values="value")
        .filter(pl.col("QE-I + GloVe") == -pl.col("QE-U + GloVe"))
        .height
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And how are the values of $U(\vec{s}\vec{p}) = I(\vec{w}) + U(\vec{w})$ distributed?
    """)
    return


@app.cell
def _(df_unpivot, mo, pl):
    mo.ui.table(
        df_unpivot.pivot("metric", values="value")
        .with_columns(
            (pl.col("QE-I + GloVe") + pl.col("QE-U + GloVe")).alias("sp entropy")
        )
        .group_by(["corpus", "label"])
        .agg(
            pl.col("sp entropy").mean().alias("avg sp entropy"),
            pl.col("sp entropy").quantile(0.25).alias("sp entropy Q1"),
            pl.col("sp entropy").quantile(0.5).alias("sp entropy Q2"),
            pl.col("sp entropy").quantile(0.75).alias("sp entropy Q3"),
        )
        .sort("corpus", "label"),
        selection=None,
        pagination=False,
        format_mapping={
            "avg sp entropy": "{:.4g}",
            "sp entropy Q1": "{:.4g}",
            "sp entropy Q2": "{:.4g}",
            "sp entropy Q3": "{:.4g}",
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(df_unpivot, mo, plt, sns):
    _g = sns.catplot(
        df_unpivot,
        y="corpus",
        x="value",
        hue="label",
        col="metric",
        kind="violin",
        split=True,
        inner="quart",
        bw_adjust=0.5,
        density_norm="width",
        sharex=False,
        linewidth=1,
        col_order=["QE-U + GloVe", "QE-I + GloVe"],
    )

    _g.fig.suptitle("Quantum Entropy metrics using GloVe embeddings")
    _g.set_titles("")
    for _metric_name, _ax in _g.axes_dict.items():
        _ax.set_ylabel("Corpus")
        _label = "Incongruity" if _metric_name.startswith("QE-I") else "Uncertainty"
        _ax.set_xlabel(_label)
    _g.legend.set_title("Label")
    _separator_positions = [3.5, 6.5, 7.5, 10.5]
    for _ax in _g.axes.flat:
        for _y_pos in _separator_positions:
            _ax.axhline(
                _y_pos, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=0
            )
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(df_unpivot, mo, pl, plt, sns):
    df_microedit = (
        df_unpivot.filter(pl.col("corpus").is_in(["humicroedit", "puntuguese"]))
        .with_columns(pl.col("index").str.strip_chars_end(r".[NH]").alias("pair"))
        .drop("index")
        .pivot(on="label", index=["metric", "corpus", "pair"])
    )

    _g = sns.lmplot(
        df_microedit,
        x="Humor",
        y="Non-humor",
        col="metric",
        hue="corpus",
        hue_order=["humicroedit", "puntuguese"],
        facet_kws={"sharex": False, "sharey": False},
        palette={"humicroedit": "#005A9C", "puntuguese": "#E1306C"},
        scatter_kws={"alpha": 0.5, "s": 30},
        x_bins=20,
        x_ci="sd",
        truncate=False,
        line_kws={"linewidth": 1},
        markers=["o", "s"],
    )

    _g.set_titles("{col_name}")
    mo.mpl.interactive(plt.gcf())
    return (df_microedit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How are the distributions of $I(\vec{w}) + U(\vec{w}) = U(\vec{s}\vec{p})$?
    """)
    return


@app.cell
def _(df_unpivot, mo, pl, sns):
    mo.mpl.interactive(
        sns.boxplot(
            (
                df_unpivot.pivot("metric", values="value").with_columns(
                    (pl.col("QE-I + GloVe") + pl.col("QE-U + GloVe")).alias(
                        "sp entropy"
                    )
                )
            ),
            x="sp entropy",
            y="corpus",
            showfliers=False,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Statistical tests
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statistical difference
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
    }

    for _metric in df_unpivot["metric"].unique():
        for _corpus in corpora_enum.categories:
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
            _results["metric"].append(_metric)
            _results["corpus"].append(_corpus)
            _results["test"].append(_test.__name__)
            _results["statistic"].append(_statistic)
            _results["p-value"].append(_pvalue)

    _results_df = pl.DataFrame(_results).with_columns(
        (pl.col("p-value") < 0.05).alias("different distributions")
    )

    mo.ui.table(
        _results_df,
        format_mapping={"p-value": "{:.3g}".format},
        selection=None,
        pagination=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation
    """)
    return


@app.cell
def _(cs, df_microedit, mo, np, pearsonr, pl):
    def _compute_pearson(row):
        row_humor = np.array(row["Humor"], dtype=np.float64)
        row_non_humor = np.array(row["Non-humor"], dtype=np.float64)
        correlation, _ = pearsonr(row_humor, row_non_humor)
        return correlation

    _results_df = (
        df_microedit.drop_nulls()
        .group_by(["metric", "corpus"])
        .agg(cs.numeric().implode())
        .with_columns(
            pl.struct(pl.all())
            .map_elements(_compute_pearson, return_dtype=pl.Float64)
            .alias("correlation")
        )
        .sort("corpus", "metric")
        .select(["corpus", "metric", "correlation"])
    )

    mo.ui.table(
        _results_df, format_mapping={"correlation": "{:.4g}".format}, selection=None
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cohen's d
    """)
    return


@app.cell
def _(df_unpivot, mo, pl):
    _pooled_sd = (
        (
            (pl.col("n_Humor") - 1) * pl.col("var_Humor")
            + (pl.col("n_Non-humor") - 1) * pl.col("var_Non-humor")
        )
        / (pl.col("n_Humor") + pl.col("n_Non-humor") - 2)
    ).sqrt()
    _cohen_d_expr = (pl.col("mean_Humor") - pl.col("mean_Non-humor")) / _pooled_sd

    _results_df = (
        df_unpivot.group_by(["corpus", "metric", "label"])
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").var().alias("var"),
            pl.len().alias("n"),
        )
        .pivot("label", index=["corpus", "metric"])
        .with_columns(_cohen_d_expr.alias("cohen d"))
        .select(["corpus", "metric", "cohen d"])
        .sort("metric", "corpus")
    )

    mo.ui.table(
        _results_df,
        format_mapping={"cohen d": "{:.3g}"},
        selection=None,
        pagination=False,
    )
    return


if __name__ == "__main__":
    app.run()
