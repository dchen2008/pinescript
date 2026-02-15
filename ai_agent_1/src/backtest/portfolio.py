"""Portfolio: balance tracking, compound risk sizing, spread adjustment."""

import math

from src.utils.forex_utils import pips_to_price


class Portfolio:
    """Tracks account balance and computes position sizes."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_percent: float = 1.0,
        max_position_units: int = 100000,
        spread_pips: float = 1.5,
    ):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.risk_percent = risk_percent
        self.max_position_units = max_position_units
        self.spread_pips = spread_pips
        self.spread_price = pips_to_price(spread_pips)
        self.peak_balance = initial_capital
        self.equity_curve = []

    def compute_position_size(self, sl_distance: float) -> float:
        """Compute position size using compound risk sizing.

        Matches PineScript: math.floor(risk / sl_dist), min cap, max 1 unit.

        Args:
            sl_distance: Distance from entry to SL in price units

        Returns:
            Number of units to trade
        """
        if sl_distance <= 0:
            return 0.0

        risk_amount = self.balance * self.risk_percent / 100.0
        units = math.floor(risk_amount / sl_distance)
        return max(min(units, self.max_position_units), 1)

    def adjust_entry_for_spread(self, entry_price: float, is_long: bool) -> float:
        """Adjust entry price for spread cost.

        LONG: buy at ASK = mid + spread
        SHORT: sell at BID = mid - spread
        """
        if is_long:
            return entry_price + self.spread_price
        else:
            return entry_price - self.spread_price

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade's PnL."""
        self.balance += pnl
        self.peak_balance = max(self.peak_balance, self.balance)

    def record_equity(self, bar_idx: int, unrealized_pnl: float = 0.0) -> None:
        """Record equity at a point in time."""
        equity = self.balance + unrealized_pnl
        self.equity_curve.append({
            "bar": bar_idx,
            "balance": self.balance,
            "equity": equity,
        })

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance

    @property
    def net_profit(self) -> float:
        """Net profit in dollars."""
        return self.balance - self.initial_capital

    @property
    def net_profit_pct(self) -> float:
        """Net profit as percentage."""
        return (self.net_profit / self.initial_capital) * 100
