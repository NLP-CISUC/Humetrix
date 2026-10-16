import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import seaborn as sns

    return mo, pl, sns


@app.cell
def _(pl):
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
    return (corpora_enum,)


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


if __name__ == "__main__":
    app.run()
