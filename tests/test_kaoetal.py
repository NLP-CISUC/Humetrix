from pathlib import Path

import polars as pl
import pytest

GOLD_PATH = Path(__file__).parent / "gold" / "kaoetal.jsonl"
EXAMPLES = pl.read_ndjson(GOLD_PATH).to_dicts()
METRICS = {"ambiguity": "kao_ambiguity", "distinctiveness": "kao_distinctiveness"}


@pytest.mark.parametrize("column", METRICS)
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["corpus"])
def test_kaoetal(example, column, get_analyzer):
    method = METRICS[column]
    analyzer = get_analyzer(example["language"])
    score = getattr(analyzer, method)(
        example["text"], example["location"], example["interpretation"]
    )

    expected = example[column]
    assert score == pytest.approx(expected, abs=1e-4)
