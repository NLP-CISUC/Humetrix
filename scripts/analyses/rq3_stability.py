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

    from scipy.stats import mannwhitneyu
    from statsmodels.stats.weightstats import ttost_ind

    return Path, mannwhitneyu, mo, np, pl, sns, ttost_ind


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RQ3: Cross-linguistic Stability
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

    corpora_enum = pl.Enum(
        [
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
        ]
    )
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
                    ["funniness", "location", "interpretation", "char_count"],
                    strict=False,
                )
            )
            if "label" in df.columns:
                df = df.filter(pl.col("label") == 1)
            dfs.append(df)

    combined_df = pl.concat(dfs, how="diagonal_relaxed")
    df = (
        combined_df.drop("label", "text")
        # Remove error values (-1 for surprise ratio and 0 for Kao et al.)
        .with_columns(
            pl.col("local global surprise").replace(-1, None),
            pl.col("ambiguity").replace(0, None),
            pl.col("distinctiveness").replace(0, None),
        )
        .unpivot(index=["index", "corpus"], variable_name="metric")
        .filter(pl.col("value").is_not_null())
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Quantiles before removing outliers
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Quantiles after removing outliers
    """)
    return


@app.cell
def _(df, mo, pl, quantiles):
    df_with_quantiles = df.join(quantiles, on=["corpus", "metric"])
    is_low_outlier = pl.col("value") < pl.col("F1")
    is_high_outlier = pl.col("value") > pl.col("F2")
    df_no_outliers = df_with_quantiles.filter(
        ~(is_low_outlier | is_high_outlier)
    ).select(df.columns)

    _df_no_outliers_quantiles = (
        df_no_outliers.group_by(["corpus", "metric"])
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
        _df_no_outliers_quantiles.sort("metric", "corpus"),
        selection=None,
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
    return (df_no_outliers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Mean and standard deviation without outliers
    """)
    return


@app.cell
def _(df_no_outliers, mo, pl):
    mo.ui.table(
        df_no_outliers.group_by(["metric", "corpus"])
        .agg(pl.col("value").mean().alias("mean"), pl.col("value").std().alias("std"))
        .sort("metric", "corpus"),
        selection=None,
        format_mapping={"mean": "{:.3g}", "std": "{:.3g}"},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(corpora_enum, df_no_outliers, sns):
    sns.set_theme(style="whitegrid", font_scale=1.2)

    corpus_palette = {
        # English (Blue to Cyan spectrum)
        "semeval": "#08306B",  # Very Dark Blue
        "humicroedit": "#005A9C",  # Medium Blue
        "expunations": "#1890FF",  # Vibrant Blue
        "cup": "#00C2D1",  # Teal/Cyan
        "joker_clef_en": "#80D8FF",  # Light Cyan
        # Spanish (Red spectrum)
        "joker_clef_es": "#990000",  # Dark Crimson
        "HAHA@IberLEF2021": "#E63946",  # Bright Red
        "HUHU@IberLEF2023": "#FF6B6B",  # Soft/Light Red
        # French (Green)
        "joker_clef_fr": "#2A9D8F",  # Persian Green
        # Portuguese (Purple to Pink spectrum)
        "joker_clef_pt": "#4A00E0",  # Deep Indigo
        "clemencio": "#9D4EDD",  # Bright Purple
        "puntuguese": "#E1306C",  # Magenta/Deep Pink
        # Mandarin Chinese (Golden/Amber)
        "chumor": "#D4AC0D",  # Dark Gold / Amber
    }

    _g = sns.displot(
        df_no_outliers,
        x="value",
        hue="corpus",
        col="metric",
        col_wrap=3,
        kind="kde",
        common_norm=False,
        fill=True,
        height=4,
        aspect=1.5,
        facet_kws={"sharex": False, "sharey": False},
        palette=corpus_palette,
        alpha=0.1,
        linewidth=1.5,
        hue_order=corpora_enum.categories,
    )
    _g.set_titles(col_template="{col_name}")
    _g.set_axis_labels("Metric Value", "Density")

    _g
    return (corpus_palette,)


@app.cell
def _(corpus_palette, df_no_outliers, sns):
    _g = sns.catplot(
        data=df_no_outliers,
        x="value",
        y="corpus",
        hue="corpus",
        col="metric",
        col_wrap=2,
        sharex=False,
        palette=corpus_palette,
        kind="violin",
        height=4,
        aspect=1.5,
        linewidth=1,
        inner="quart",
        dodge=False,
    )

    _separator_positions = [4.5, 7.5, 8.5, 11.5]
    for _ax in _g.axes.flat:
        for _y_pos in _separator_positions:
            _ax.axhline(
                _y_pos, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=0
            )
    _g.set_titles(col_template="{col_name}")
    _g.set_axis_labels("Metric Value", "Density")

    _g
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Statistical Tests (No outliers)
    """)
    return


@app.cell
def _(corpora_enum, df_no_outliers, mannwhitneyu, np, pl, ttost_ind):
    import itertools

    _d_margin = 0.2
    _metrics = df_no_outliers["metric"].unique()

    _results = []
    for _metric in _metrics:
        _df_metric = df_no_outliers.filter(pl.col("metric") == _metric)
        _combinations = itertools.combinations(corpora_enum.categories, 2)
        for _corpus_a, _corpus_b in _combinations:
            _data_a = _df_metric.filter(pl.col("corpus") == _corpus_a)[
                "value"
            ].drop_nulls().to_numpy()
            _data_b = _df_metric.filter(pl.col("corpus") == _corpus_b)[
                "value"
            ].drop_nulls().to_numpy()

            if len(_data_a) > 1 and len(_data_b) > 1:
                _statistic, _pvalue = mannwhitneyu(
                    _data_a, _data_b, alternative="two-sided"
                )

                _n_a, _n_b = len(_data_a), len(_data_b)
                _pooled_sd = np.sqrt(
                    (
                        (_n_a - 1) * np.var(_data_a, ddof=1)
                        + (_n_b - 1) * np.var(_data_b, ddof=1)
                    )
                    / (_n_a + _n_b - 2)
                )
                _delta = _d_margin * _pooled_sd
                _tost_p, _, _ = ttost_ind(
                    _data_a, _data_b, -_delta, _delta, usevar="unequal"
                )
                _d = (np.mean(_data_a) - np.mean(_data_b)) / _pooled_sd
            
                _results.append(
                    {
                        "metric": _metric,
                        "corpus_a": _corpus_a,
                        "corpus_b": _corpus_b,
                        "u_statistic": _statistic,
                        "mw_p_value": _pvalue,
                        "cohen_d": _d,
                        "tost_p_value": _tost_p,
                        "significantly different": _pvalue < 0.05,
                        "equivalent": _tost_p < 0.05,
                    }
                )

    results_df = pl.DataFrame(_results).sort(["metric", "tost_p_value"])

    (
        results_df.group_by(["metric"])
        .agg(
            [
                pl.col("equivalent").sum().alias("equivalent pairs"),
                pl.col("equivalent").len().alias("total pairs"),
            ]
        )
        .with_columns(
            (pl.col("equivalent pairs") / pl.col("total pairs") * 100)
            .round(2)
            .alias("stability percentage")
        )
        .sort("stability percentage", descending=True)
    )
    return (results_df,)


@app.cell
def _(mo, pl, results_df):
    non_qe_results = results_df.filter(
        pl.col("metric").is_in(
            ["ambiguity", "distinctiveness", "local global surprise"]
        )
    ).select(pl.all().exclude("u_statistic"))

    mo.ui.table(
        non_qe_results,
        format_mapping={
            "mw_p_value": "{:.4g}".format,
            "cohen_d": "{:.4g}".format,
            "tost_p_value": "{:.4g}".format,
        },
        selection=None
    )
    return


if __name__ == "__main__":
    app.run()
