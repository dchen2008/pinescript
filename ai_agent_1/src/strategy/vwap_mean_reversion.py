"""VWAP Mean Reversion strategy.

Entry:
  LONG:  close < vwap_lower AND rsi < rsi_oversold AND can_trade
  SHORT: close > vwap_upper AND rsi > rsi_overbought AND can_trade

Exit:
  SL: Fixed pips or ATR-based below/above entry
  TP: Return to VWAP (mean reversion target), opposite band, or fixed RR
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base_strategy import BaseStrategy
from src.strategy.position import Position
from src.utils.forex_utils import pips_to_price


class VWAPMeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using session VWAP bands + RSI confirmation."""

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        sl_pips: float = 15.0,
        tp_type: str = "vwap",
        tp_rr: float = 1.5,
    ):
        """
        Args:
            rsi_oversold: RSI threshold for long entries (buy below this).
            rsi_overbought: RSI threshold for short entries (sell above this).
            sl_pips: Fixed SL distance in pips.
            tp_type: "vwap" (return to VWAP), "opposite_band", or "rr" (fixed RR).
            tp_rr: Risk-reward ratio (only used when tp_type="rr").
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.sl_pips = sl_pips
        self.tp_type = tp_type
        self.tp_rr = tp_rr

    def _compute_tp(
        self, entry_price: float, sl_distance: float, is_long: bool, row: pd.Series
    ) -> Optional[float]:
        """Compute TP price based on tp_type."""
        if self.tp_type == "vwap":
            vwap_val = row["vwap"]
            if np.isnan(vwap_val):
                return None
            return vwap_val
        elif self.tp_type == "opposite_band":
            if is_long:
                val = row["vwap_upper"]
            else:
                val = row["vwap_lower"]
            return val if not np.isnan(val) else None
        elif self.tp_type == "rr":
            if is_long:
                return entry_price + sl_distance * self.tp_rr
            else:
                return entry_price - sl_distance * self.tp_rr
        return None

    def on_bar(
        self,
        idx: int,
        row: pd.Series,
        position: Optional[Position],
        can_trade: bool,
        entering_quiet: bool,
    ) -> dict:
        result = {"action": "none"}

        # No trailing or break-even for this strategy — rely on SL/TP
        if position is not None:
            return result

        if not can_trade:
            return result

        close = row["close"]
        rsi_val = row.get("rsi", np.nan)
        vwap_val = row.get("vwap", np.nan)
        vwap_lower = row.get("vwap_lower", np.nan)
        vwap_upper = row.get("vwap_upper", np.nan)

        # Need valid indicator values
        if np.isnan(rsi_val) or np.isnan(vwap_val) or np.isnan(vwap_lower) or np.isnan(vwap_upper):
            return result

        sl_distance = pips_to_price(self.sl_pips)

        # LONG: close below lower band + RSI oversold
        if close < vwap_lower and rsi_val < self.rsi_oversold:
            sl = close - sl_distance
            tp = self._compute_tp(close, sl_distance, True, row)
            # Ensure TP is above entry for longs
            if tp is not None and tp <= close:
                tp = close + sl_distance * self.tp_rr
            return {"action": "open_long", "sl": sl, "tp": tp}

        # SHORT: close above upper band + RSI overbought
        if close > vwap_upper and rsi_val > self.rsi_overbought:
            sl = close + sl_distance
            tp = self._compute_tp(close, sl_distance, False, row)
            # Ensure TP is below entry for shorts
            if tp is not None and tp >= close:
                tp = close - sl_distance * self.tp_rr
            return {"action": "open_short", "sl": sl, "tp": tp}

        return result
