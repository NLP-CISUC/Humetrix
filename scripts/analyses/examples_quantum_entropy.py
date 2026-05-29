import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Examples of QE-U's focus on domain
    """)
    return


@app.cell
def _(Path, pl):
    results_paths = [
        Path('results/quantum_entropy/semeval.jsonl'),
        Path('results/quantum_entropy/humicroedit.jsonl'),
        Path('results/quantum_entropy/expunations.jsonl')
    ]

    dfs = []
    for result_path in results_paths:
        dfs.append(
            pl.read_ndjson(result_path, infer_schema_length=None)
            .with_columns(pl.lit(result_path.stem).alias('corpus'))
            .with_columns(
                pl.col('corpus').cast(
                    pl.Enum([
                        'semeval',
                        'humicroedit',
                        'expunations'
                    ])
                )
            )
        )
    df = (
        pl.concat(dfs, how='diagonal')
        .select('corpus', 'text', 'label', 'QE-U + GloVe')
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What are the texts with highest QE-U for each label?
    """)
    return


@app.cell
def _(df, pl):
    (
        df.filter(
            pl.col('QE-U + GloVe') == pl.col('QE-U + GloVe').max().over('corpus', 'label')
        )
            .unique()
            .sort('corpus', 'QE-U + GloVe')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What are the texts that have different annotations between SemEval and ExPUNations?
    """)
    return


@app.cell
def _(df, pl):
    _df_semeval = df.filter(pl.col('corpus') == 'semeval')
    _df_expunations = df.filter(pl.col('corpus') == 'expunations')

    (
        _df_semeval.join(_df_expunations, on='text', suffix=' expunations')
        .filter(pl.col('label') != pl.col('label expunations'))
        .rename({'label': 'label semeval'})
        .select('text', 'label semeval', 'label expunations', 'QE-U + GloVe')
        .sort('QE-U + GloVe', descending=True)
    )
    return


if __name__ == "__main__":
    app.run()
