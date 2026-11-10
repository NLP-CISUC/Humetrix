from functools import lru_cache
from typing import Union

import pytest

from humetrix import HumorAnalyzer

TRANSFORMER = "FacebookAI/xlm-roberta-base"


@pytest.fixture(scope="session")
def get_analyzer():
    @lru_cache(maxsize=1)
    def _get_analyzer(language: str, device: Union[str, None] = None) -> HumorAnalyzer:
        return HumorAnalyzer(
            language, transformer_model_name=TRANSFORMER, device=device
        )

    return _get_analyzer
