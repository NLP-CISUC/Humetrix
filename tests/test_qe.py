from functools import lru_cache
from pathlib import Path

import polars as pl
import pytest
import spacy

from humetrix import HumorAnalyzer

GOLD_PATH = Path(__file__).parent / "gold" / "quantum_entropy.jsonl"
EXAMPLES = pl.read_ndjson(GOLD_PATH).to_dicts()
METRICS = {
    "QE-I + GloVe": ("quantum_incongruity", {}),
    "QE-I + HF": ("quantum_incongruity", {"backend": "transformer"}),
    "QE-U + GloVe": ("quantum_uncertainty", {}),
    "QE-U + HF": ("quantum_uncertainty", {"backend": "transformer"}),
}


@pytest.mark.parametrize("column", METRICS)
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["corpus"])
def test_quantum_entropy(example, column, get_analyzer):
    method, kwargs = METRICS[column]
    analyzer = get_analyzer(example["language"])
    score = getattr(analyzer, method)(example["text"], **kwargs)

    expected = example[column]
    if expected is None:
        assert score is None
    else:
        assert score == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("column", METRICS)
def test_requires_two_segments(column, get_analyzer):
    method, kwargs = METRICS[column]
    analyzer = get_analyzer("pt")

    one_segment = "Nós chamamos de meio ambiente porque já destruímos metade?"
    score = getattr(analyzer, method)(one_segment, **kwargs)
    assert score is None

    multiple_segments = "Era uma vez um homem tão pequeno, mas tão pequeno, que não andava de metro. Andava de centímetro."
    score = getattr(analyzer, method)(multiple_segments, **kwargs)
    assert score is None


def test_default_punc_chars_not_changed(get_analyzer):
    analyzer = get_analyzer("pt")
    analyzer.quantum_uncertainty(
        "O que é um javaqui? Um javali que está muito próximo."
    )
    assert "," not in spacy.pipeline.Sentencizer.default_punct_chars
    assert ";" not in spacy.pipeline.Sentencizer.default_punct_chars


def test_correct_torch_device(get_analyzer):
    analyzer = get_analyzer("pt", device="cpu")
    pipeline = analyzer._load_model("transformer_pipeline")
    assert pipeline.model.device.type == "cpu"

    bert = analyzer._load_model("bert")
    assert bert["lm"].device.type == "cpu"
