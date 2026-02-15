"""Test portfolio: compound sizing, spread adjustment."""

import pytest

from src.backtest.portfolio import Portfolio
from src.utils.forex_utils import pips_to_price


class TestCompoundSizing:
    def test_basic_sizing(self):
        """1% of $10,000 = $100 risk. With 10 pip SL = 100,000 units."""
        p = Portfolio(initial_capital=10000.0, risk_percent=1.0)
        sl_distance = pips_to_price(10)  # 10 pips = 0.0010
        units = p.compute_position_size(sl_distance)
        # risk = 100, units = 100 / 0.0010 = 100,000
        assert units == pytest.approx(100000.0)

    def test_capped_at_max(self):
        """Position should be capped at max_position_units."""
        p = Portfolio(initial_capital=10000.0, risk_percent=1.0, max_position_units=50000)
        sl_distance = pips_to_price(1)  # Very tight SL -> large position
        units = p.compute_position_size(sl_distance)
        assert units == 50000

    def test_compound_after_profit(self):
        """After $100 profit on $10K, next risk = $101."""
        p = Portfolio(initial_capital=10000.0, risk_percent=1.0)
        p.record_trade(100.0)  # $100 profit
        assert p.balance == pytest.approx(10100.0)

        sl_distance = pips_to_price(10)
        units = p.compute_position_size(sl_distance)
        # risk = 10100 * 0.01 = 101, units = 101 / 0.0010 = 101,000 -> capped at 100K
        assert units == pytest.approx(100000.0)

    def test_compound_smaller_sl(self):
        """After $100 profit, smaller SL shows compound effect."""
        p = Portfolio(initial_capital=10000.0, risk_percent=1.0)
        p.record_trade(100.0)

        sl_distance = pips_to_price(20)  # 20 pips
        units = p.compute_position_size(sl_distance)
        # risk = 101, units = 101 / 0.0020 = 50,500
        assert units == pytest.approx(50500.0)

    def test_zero_sl_returns_zero(self):
        p = Portfolio()
        assert p.compute_position_size(0) == 0.0
        assert p.compute_position_size(-0.001) == 0.0


class TestSpreadAdjustment:
    def test_long_spread(self):
        """Long entry: mid + spread."""
        p = Portfolio(spread_pips=1.5)
        # 1.5 pips = 0.00015
        entry = p.adjust_entry_for_spread(1.08000, is_long=True)
        assert entry == pytest.approx(1.08015)

    def test_short_spread(self):
        """Short entry: mid - spread."""
        p = Portfolio(spread_pips=1.5)
        entry = p.adjust_entry_for_spread(1.08000, is_long=False)
        assert entry == pytest.approx(1.07985)


class TestDrawdown:
    def test_drawdown_calculation(self):
        p = Portfolio(initial_capital=10000.0)
        p.record_trade(500.0)   # Balance: 10500, peak: 10500
        assert p.drawdown == pytest.approx(0.0)

        p.record_trade(-200.0)  # Balance: 10300, peak: 10500
        expected_dd = (10500 - 10300) / 10500
        assert p.drawdown == pytest.approx(expected_dd)


class TestNetProfit:
    def test_net_profit(self):
        p = Portfolio(initial_capital=10000.0)
        p.record_trade(500.0)
        p.record_trade(-200.0)
        assert p.net_profit == pytest.approx(300.0)
        assert p.net_profit_pct == pytest.approx(3.0)
