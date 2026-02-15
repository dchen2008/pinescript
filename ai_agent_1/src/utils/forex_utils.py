"""Forex utility functions: pip conversion, price formatting."""

# EUR/USD pip value: 1 pip = 0.0001 (4th decimal place)
# In 5-decimal pricing: 1 pip = 10 points (10 * 0.00001)
PIP_VALUE = 0.0001


def pips_to_price(pips: float) -> float:
    """Convert pips to price distance."""
    return pips * PIP_VALUE


def price_to_pips(price_diff: float) -> float:
    """Convert price distance to pips."""
    return price_diff / PIP_VALUE


def format_price(price: float) -> str:
    """Format price to 5 decimal places (standard forex)."""
    return f"{price:.5f}"
