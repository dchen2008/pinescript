"""Huge Movement Trade Strategy (HMTS).

3-signal state machine that detects large moves (60+ pips) followed by a
small retracement, then holds the original position for an extended target.

State machine:
  IDLE -> WATCHING_REVERT -> HMTS_ACTIVE

When HMTS conditions aren't met, behaves like the normal signal strategy
(enter on signal, reverse on opposite signal, trail SL with SuperTrend).
"""

import math
from typing import Optional

import pandas as pd

from src.strategy.base_strategy import BaseStrategy
from src.strategy.position import Position
from src.utils.forex_utils import pips_to_price, price_to_pips


class HMTSStrategy(BaseStrategy):
    """Huge Movement Trade Strategy with 3-signal pattern detection."""

    IDLE = "idle"
    WATCHING_REVERT = "watching_revert"
    HMTS_ACTIVE = "hmts_active"

    def __init__(
        self,
        movement_candles: int = 10,
        movement_pips: float = 60.0,
        revert_candles: int = 6,
        revert_min_pips: float = 10.0,
        revert_max_pips: float = 30.0,
        hmts_tp_rr: float = 7.0,
        allow_second_entry: bool = True,
        base_rr_ratio: float = 1.5,
        trail_with_supertrend: bool = True,
        s1_sl_spread_buffer: bool = True,
        s2_revert_cross_limit: float = 8.0,
        spread_pips: float = 1.5,
    ):
        self.movement_candles = movement_candles
        self.movement_pips = movement_pips
        self.revert_candles = revert_candles
        self.revert_min_pips = revert_min_pips
        self.revert_max_pips = revert_max_pips
        self.hmts_tp_rr = hmts_tp_rr
        self.allow_second_entry = allow_second_entry
        self.base_rr_ratio = base_rr_ratio
        self.trail_with_supertrend = trail_with_supertrend
        self.s1_sl_spread_buffer = s1_sl_spread_buffer
        self.s2_revert_cross_limit = s2_revert_cross_limit
        self.spread_pips = spread_pips

        # State
        self.state = self.IDLE
        self.s1_bar = 0
        self.s1_close = 0.0
        self.s1_direction = 0  # 1=buy, -1=sell
        self.s1_st = 0.0  # S1's SuperTrend level (for cross protection)
        self.s2_bar = 0
        self.s2_close = 0.0
        self.original_sl_distance = 0.0
        self.hmts_direction = 0  # direction of held position
        self.last_known_tp = None
        self.last_known_sl = None
        self.last_known_direction = 0
        self.was_in_position = False

    def _reset_state(self):
        """Reset to IDLE state."""
        self.state = self.IDLE
        self.s1_bar = 0
        self.s1_close = 0.0
        self.s1_direction = 0
        self.s1_st = 0.0
        self.s2_bar = 0
        self.s2_close = 0.0
        self.original_sl_distance = 0.0
        self.hmts_direction = 0
        self.last_known_tp = None
        self.last_known_sl = None
        self.last_known_direction = 0
        self.was_in_position = False

    def on_bar(
        self,
        idx: int,
        row: pd.Series,
        position: Optional[Position],
        can_trade: bool,
        entering_quiet: bool,
    ) -> dict:
        buy_signal = bool(row.get("buy_signal", False))
        sell_signal = bool(row.get("sell_signal", False))
        tup = row["tup"]
        tdown = row["tdown"]
        close = row["close"]
        high = row["high"]
        low = row["low"]

        # Close on quiet window entry
        if entering_quiet and position is not None:
            self._reset_state()
            return {"action": "close", "close_reason": "Quiet Close"}

        # Detect position closed by engine (SL/TP hit)
        if self.was_in_position and position is None:
            result = self._handle_position_closed(idx, row)
            if result["action"] != "none":
                self.was_in_position = False
                return result

        # Track position state for next bar
        if position is not None:
            self.was_in_position = True
            self.last_known_tp = position.tp_price
            self.last_known_sl = position.sl_price
            self.last_known_direction = position.direction
        else:
            self.was_in_position = False

        # --- State machine ---
        if self.state == self.IDLE:
            return self._handle_idle(idx, row, position, can_trade, buy_signal, sell_signal, tup, tdown, close)
        elif self.state == self.WATCHING_REVERT:
            return self._handle_watching_revert(idx, row, position, can_trade, buy_signal, sell_signal, tup, tdown, close, high, low)
        elif self.state == self.HMTS_ACTIVE:
            return self._handle_hmts_active(idx, row, position, tup, tdown)

        return {"action": "none"}

    def _handle_idle(self, idx, row, position, can_trade, buy_signal, sell_signal, tup, tdown, close):
        """IDLE state: normal signal behavior with S2 detection."""
        result = {"action": "none"}

        # Trail SL with SuperTrend
        if self.trail_with_supertrend and position is not None:
            if position.is_long:
                position.trail_sl_with_supertrend(tup)
            else:
                position.trail_sl_with_supertrend(tdown)

        if buy_signal and can_trade:
            sl = tup
            if math.isnan(sl):
                return result

            if self.s1_sl_spread_buffer:
                sl = sl - pips_to_price(self.spread_pips)

            sl_distance = close - sl
            if sl_distance <= 0:
                return result

            tp = close + sl_distance * self.base_rr_ratio

            # Check if this is S2 (we're in short from S1=sell, and buy signal fires)
            if position is not None and position.is_short and self.s1_direction == -1:
                if self._check_s2_transition(idx, position, close):
                    return result  # Hold position, don't reverse

            # Track as new S1
            self.s1_bar = idx
            self.s1_close = close
            self.s1_direction = 1
            self.s1_st = tup

            if position is not None and position.is_short:
                return {"action": "reverse_to_long", "sl": sl, "tp": tp, "close_reason": "Reversal"}
            elif position is None:
                return {"action": "open_long", "sl": sl, "tp": tp}

        if sell_signal and can_trade:
            sl = tdown
            if math.isnan(sl):
                return result

            if self.s1_sl_spread_buffer:
                sl = sl + pips_to_price(self.spread_pips)

            sl_distance = sl - close
            if sl_distance <= 0:
                return result

            tp = close - sl_distance * self.base_rr_ratio

            # Check if this is S2 (we're in long from S1=buy, and sell signal fires)
            if position is not None and position.is_long and self.s1_direction == 1:
                if self._check_s2_transition(idx, position, close):
                    return result  # Hold position, don't reverse

            # Track as new S1
            self.s1_bar = idx
            self.s1_close = close
            self.s1_direction = -1
            self.s1_st = tdown

            if position is not None and position.is_long:
                return {"action": "reverse_to_short", "sl": sl, "tp": tp, "close_reason": "Reversal"}
            elif position is None:
                return {"action": "open_short", "sl": sl, "tp": tp}

        return result

    def _check_s2_transition(self, idx, position, close):
        """Check if S2 qualifies as a huge movement, transition to WATCHING_REVERT."""
        movement_pips = abs(self.s1_close - close) / 0.0001
        candles = idx - self.s1_bar

        if movement_pips >= self.movement_pips and candles <= self.movement_candles:
            self.state = self.WATCHING_REVERT
            self.s2_bar = idx
            self.s2_close = close
            self.hmts_direction = self.s1_direction
            self.original_sl_distance = abs(position.entry_price - position.sl_price)
            return True
        return False

    def _handle_watching_revert(self, idx, row, position, can_trade, buy_signal, sell_signal, tup, tdown, close, high, low):
        """WATCHING_REVERT: waiting for S3 to confirm HMTS pattern."""
        result = {"action": "none"}

        if position is None:
            self._reset_state()
            return result

        # Trail SL with SuperTrend
        if self.trail_with_supertrend:
            if position.is_long:
                position.trail_sl_with_supertrend(tup)
            else:
                position.trail_sl_with_supertrend(tdown)

        # S1-ST cross protection
        cross_limit = pips_to_price(self.s2_revert_cross_limit)
        if self.hmts_direction == 1:  # long position
            if low < self.s1_st - cross_limit:
                self._reset_state()
                return {"action": "close", "close_reason": "S1-ST Cross"}
        else:  # short position
            if high > self.s1_st + cross_limit:
                self._reset_state()
                return {"action": "close", "close_reason": "S1-ST Cross"}

        # Check for S3 (same direction as S1)
        s3_signal = (buy_signal and self.s1_direction == 1) or (sell_signal and self.s1_direction == -1)

        bars_since_s2 = idx - self.s2_bar

        if s3_signal:
            revert_pips = abs(close - self.s2_close) / 0.0001
            if revert_pips >= self.revert_min_pips and revert_pips < self.revert_max_pips:
                # HMTS confirmed!
                self.state = self.HMTS_ACTIVE

                # Update SL to S3's SuperTrend
                if self.hmts_direction == 1:
                    new_sl = tup
                    if not math.isnan(new_sl):
                        position.sl_price = new_sl
                else:
                    new_sl = tdown
                    if not math.isnan(new_sl):
                        position.sl_price = new_sl

                # Set TP based on original SL distance
                if self.hmts_tp_rr > 0:
                    if self.hmts_direction == 1:
                        position.tp_price = position.entry_price + self.original_sl_distance * self.hmts_tp_rr
                    else:
                        position.tp_price = position.entry_price - self.original_sl_distance * self.hmts_tp_rr

                return result
            else:
                # Revert out of range -> pattern failed
                self._reset_state()
                return {"action": "close", "close_reason": "HMTS Revert Failed"}

        # Check timeout
        if bars_since_s2 > self.revert_candles:
            self._reset_state()
            return {"action": "close", "close_reason": "HMTS Revert Timeout"}

        # Suppress counter-trend signals
        # (any signal opposite to held direction is ignored)
        return result

    def _handle_hmts_active(self, idx, row, position, tup, tdown):
        """HMTS_ACTIVE: holding position, trailing with SuperTrend."""
        if position is None:
            return {"action": "none"}

        # Trail SL with SuperTrend (favorable only)
        if position.is_long:
            position.trail_sl_with_supertrend(tup)
        else:
            position.trail_sl_with_supertrend(tdown)

        # Ignore ALL signals
        return {"action": "none"}

    def _handle_position_closed(self, idx, row):
        """Handle position closed by engine (SL/TP hit)."""
        result = {"action": "none"}

        if self.state == self.HMTS_ACTIVE:
            # Infer close reason from current bar
            tp_hit = False
            if self.last_known_tp is not None:
                if self.last_known_direction == 1:
                    tp_hit = row["high"] >= self.last_known_tp
                else:
                    tp_hit = row["low"] <= self.last_known_tp

            if tp_hit and self.allow_second_entry:
                # Re-enter same direction
                tup = row["tup"]
                tdown = row["tdown"]
                close = row["close"]
                direction = self.last_known_direction

                if direction == 1:
                    sl = tup
                    if math.isnan(sl):
                        self._reset_state()
                        return result
                    if self.s1_sl_spread_buffer:
                        sl = sl - pips_to_price(self.spread_pips)
                    sl_distance = close - sl
                    if sl_distance <= 0:
                        self._reset_state()
                        return result
                    tp = close + sl_distance * self.hmts_tp_rr if self.hmts_tp_rr > 0 else None
                    return {"action": "open_long", "sl": sl, "tp": tp}
                else:
                    sl = tdown
                    if math.isnan(sl):
                        self._reset_state()
                        return result
                    if self.s1_sl_spread_buffer:
                        sl = sl + pips_to_price(self.spread_pips)
                    sl_distance = sl - close
                    if sl_distance <= 0:
                        self._reset_state()
                        return result
                    tp = close - sl_distance * self.hmts_tp_rr if self.hmts_tp_rr > 0 else None
                    return {"action": "open_short", "sl": sl, "tp": tp}
            else:
                # SL hit -> back to IDLE
                self._reset_state()
                return result

        elif self.state == self.WATCHING_REVERT:
            self._reset_state()

        return result
