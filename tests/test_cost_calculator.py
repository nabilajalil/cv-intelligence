
import pytest

from cv_intelligence.advanced_rag.cost_calculator import (
    estimate_api_cost,
)


def test_estimate_api_cost():
    cost = estimate_api_cost(
        input_tokens=1_000_000,
        output_tokens=500_000,
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )

    assert cost == pytest.approx(2.0)


def test_estimate_api_cost_rejects_negative_values():
    with pytest.raises(ValueError):
        estimate_api_cost(
            input_tokens=-1,
            output_tokens=100,
            input_price_per_million=1.0,
            output_price_per_million=2.0,
        )
