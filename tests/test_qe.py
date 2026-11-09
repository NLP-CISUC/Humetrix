from functools import lru_cache
from pathlib import Path

import polars as pl
import pytest

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
