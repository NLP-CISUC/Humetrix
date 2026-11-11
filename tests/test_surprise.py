from pathlib import Path

import polars as pl
import pytest
import torch

GOLD_PATH = Path(__file__).parent / "gold" / "surprise_ratio.jsonl"
EXAMPLES = pl.read_ndjson(GOLD_PATH).to_dicts()


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["corpus"])
def test_surprise_ratio(example, get_analyzer):
    analyzer = get_analyzer(example["language"])
    score = analyzer.surprise_ratio(
        example["text"], example["location"], example["interpretation"]
    )
    expected = example["surprise ratio"]

    assert score == pytest.approx(expected, abs=1e-4)


def test_keep_torch_grad(get_analyzer):
    analyzer = get_analyzer("pt")
    analyzer.surprise_ratio("Qual o ator favorito do Thor? Odin Carrey.", "Odin Carrey", "O Jim Carrey")

    assert torch.is_grad_enabled()
