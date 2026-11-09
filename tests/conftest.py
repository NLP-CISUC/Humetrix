from functools import lru_cache

import pytest

from humetrix import HumorAnalyzer

TRANSFORMER = "FacebookAI/xlm-roberta-base"


@pytest.fixture(scope="session")
def get_analyzer():
    @lru_cache(maxsize=1)
    def _get_analyzer(language: str) -> HumorAnalyzer:
        return HumorAnalyzer(language, transformer_model_name=TRANSFORMER)

    return _get_analyzer
