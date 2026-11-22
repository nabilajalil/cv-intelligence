
def estimate_api_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Estimate USD API cost from token usage and supplied rates."""
    if min(
        input_tokens,
        output_tokens,
        input_price_per_million,
        output_price_per_million,
    ) < 0:
        raise ValueError("Tokens and prices cannot be negative.")

    input_cost = (
        input_tokens / 1_000_000
    ) * input_price_per_million

    output_cost = (
        output_tokens / 1_000_000
    ) * output_price_per_million

    return input_cost + output_cost
