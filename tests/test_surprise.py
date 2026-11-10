from pathlib import Path

import polars as pl
import pytest
import torch

GOLD_PATH = Path(__file__).parent / "gold" / "local_global_surprise.jsonl"
EXAMPLES = pl.read_ndjson(GOLD_PATH).to_dicts()


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["corpus"])
def test_local_global_surprise(example, get_analyzer):
    analyzer = get_analyzer(example["language"])
    score = analyzer.local_global_surprise(
        example["text"], example["location"], example["interpretation"]
    )
    expected = example["local global surprise"]

    assert score == pytest.approx(expected, abs=1e-4)


def test_keep_torch_grad(get_analyzer):
    analyzer = get_analyzer("pt")
    analyzer.local_global_surprise("Qual o ator favorito do Thor? Odin Carrey.", "Odin Carrey", "O Jim Carrey")

    assert torch.is_grad_enabled()
