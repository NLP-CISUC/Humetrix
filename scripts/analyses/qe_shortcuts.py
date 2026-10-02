import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import seaborn as sns
    import spacy
    import statsmodels.formula.api as smf
    import torch

    from gensim.models import KeyedVectors
    from humetrix import QuantumUncertainty

    return (
        KeyedVectors,
        Path,
        QuantumUncertainty,
        mo,
        pl,
        smf,
        sns,
        spacy,
        torch,
    )


@app.cell
def _(Path):
    root = Path("/home/mlinacio/HDD/Documentos/Doutorado/Projeto/Criatividade/Projetos/Recognition/Humetrix")
    results_path = root / "results/quantum_entropy"

    corpora = {"en": ["semeval", "humicroedit", "joker_clef_en", "expunations", "cup"],
               "fr": ["joker_clef_fr"],
               "es": ["joker_clef_es", "HAHA@IberLEF2021", "HUHU@IberLEF2023"],
               "pt": ["clemencio", "puntuguese", "joker_clef_pt"],
               "zh": ["chumor"]}

    glove = {"en": root / "data/embeddings/en/glove_s300.gensim",
             "es": root / "data/embeddings/es/glove_s300.gensim",
             "fr": root / "data/embeddings/fr/glove_s300.gensim",
             "pt": root / "data/embeddings/pt/glove_s300.gensim",
             "zh": root / "data/embeddings/zh/glove_s300.gensim"}

    spacy_models = {
        "pt": "pt_core_news_lg",
        "en": "en_core_web_trf",
        "fr": "fr_dep_news_trf",
        "es": "es_dep_news_trf",
        "zh": "zh_core_web_trf",

    }
    return corpora, glove, results_path, spacy_models


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - What is the length of the setup?
    - What is the rate of OOV tokens?
    - What is the rank of the density matrix?
    """)
    return


@app.cell
def _(
    KeyedVectors,
    QuantumUncertainty,
    corpora,
    glove,
    pl,
    results_path,
    spacy,
    spacy_models,
    torch,
):
    def _setup_features(text, embeddings, nlp):
        scorer = QuantumUncertainty(text, embeddings, nlp)
        setup_toks, _ = scorer.tokenize_sentence()
        n_in_vocab = sum(t in embeddings for t in setup_toks)
        rho = scorer.density_matrix(setup_toks)

        if rho is None:
            rank = 0
        else:
            rank = torch.linalg.matrix_rank(rho).item()

        return {
            "n_setup": len(setup_toks),
            "n_in_vocab": n_in_vocab,
            "oov_rate": 1 - n_in_vocab / len(setup_toks) if setup_toks else None,
            "rank": rank,
        }

    _dfs = []
    for _language, _corpora in corpora.items():
        _nlp = spacy.load(spacy_models[_language])
        _embeddings = KeyedVectors.load(str(glove[_language]))

        for _corpus  in _corpora:
            _df = pl.read_ndjson(results_path / f"{_corpus}.jsonl", infer_schema_length=None)
            _features = pl.DataFrame(
                [_setup_features(_text, _embeddings, _nlp) for _text in _df["text"]]
            )
            _dfs.append(
                pl.concat([_df, _features], how="horizontal")
                .with_columns(corpus=pl.lit(_corpus))
            )

    _df = pl.concat(_dfs, how="diagonal_relaxed")
    _df.write_ndjson(results_path.parent / "qe_shortcuts.jsonl")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation of QE-U with length and OOV rate
    """)
    return


@app.cell
def _(pl, results_path):
    shortcuts_df = pl.read_ndjson(results_path.parent / "qe_shortcuts.jsonl", infer_schema_length=None)
    shortcuts_df
    return (shortcuts_df,)


@app.cell
def _(pl, shortcuts_df):
    _qe = pl.col("QE-U + GloVe")
    _valid = (
        shortcuts_df
        .filter(_qe.is_not_null())
        .with_columns(
            label=pl.when(pl.col("label") == 1)
                .then(pl.lit("Humor"))
                .otherwise(pl.lit("Non-humor"))
        )
    )
    _by_label = (
        _valid
        .group_by("corpus", "label")
        .agg(
            pl.col("n_setup").median().alias("median n_setup"),
            pl.col("rank").median().alias("median rank")
        )
        .pivot("label", index="corpus")
    )
    _by_corpus = (
        _valid
        .group_by("corpus")
        .agg(
            (_qe > pl.col("rank").log() + 1e-4).sum().alias("bound violations"),
            pl.corr(_qe, "n_setup", method="spearman").alias("correlation QE-U, n_setup"),
            pl.corr(_qe, "rank", method="spearman").alias("correlation QE-U, rank"),
            pl.corr(_qe, "oov_rate", method="spearman").alias("correlation QE-U OOV"),
        )
    )

    _by_corpus.join(_by_label, on="corpus").sort("corpus")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear regression analysis
    """)
    return


@app.cell
def _(shortcuts_df):
    shortcuts_df
    return


@app.cell
def _(mo, shortcuts_df, sns):
    _g = sns.relplot(shortcuts_df, x="n_setup", y="QE-U + GloVe", hue="corpus", kind="line")

    mo.mpl.interactive(_g.fig)
    return


@app.cell
def _(mo, pl, shortcuts_df, sns):
    _g = sns.relplot(
        shortcuts_df.with_columns(log_n=pl.col("n_setup").log()),
        x="log_n",
        y="QE-U + GloVe",
        hue="corpus",
        kind="line"
    )

    mo.mpl.interactive(_g.fig)
    return


@app.cell
def _(mo, pl, shortcuts_df, sns):
    _g = sns.relplot(
        shortcuts_df.with_columns(log_n=pl.col("n_setup").log()),
        x="oov_rate",
        y="QE-U + GloVe",
        hue="corpus",
        kind="scatter"
    )

    mo.mpl.interactive(_g.fig)
    return


@app.cell
def _(pl, shortcuts_df, smf):
    _rows = []
    _corpora = [c for c in shortcuts_df["corpus"].unique(maintain_order=True) if c != "cup"]
    for _corpus in _corpora:
        _data = (
            shortcuts_df.filter(
                (pl.col("corpus") == _corpus)
                &
                pl.col("QE-U + GloVe").is_not_null()
            )
            .select(
                ((pl.col("QE-U + GloVe") - pl.col("QE-U + GloVe").mean()) / pl.col("QE-U + GloVe").std()).alias("qeu"),
                pl.col("label").alias("humor"),
                pl.col("n_setup").log().alias("log_n"),
                pl.col("oov_rate").alias("oov")
            )
            .to_pandas()
        )

        _m0 = smf.ols("qeu ~ humor", data=_data).fit(cov_type="HC3")
        _m1 = smf.ols("qeu ~ humor + log_n + oov", data=_data).fit(cov_type="HC3")

        _rows.append({
            "corpus": _corpus,
            "n": len(_data),
            "humor coef (M0)": _m0.params["humor"],
            "hhumor p (M0)": _m0.pvalues["humor"],
            "humor coef (M1)": _m1.params["humor"],
            "humor p (M1)": _m1.pvalues["humor"],
            "% of effect left": 100 * _m1.params["humor"] / _m0.params["humor"],
            "log_n coef (M1)": _m1.params["log_n"],
            "R2 (M0)": _m0.rsquared,
            "R2 (M1)": _m1.rsquared,
        })

    regression_df = pl.DataFrame(_rows)
    regression_df
    return


if __name__ == "__main__":
    app.run()
