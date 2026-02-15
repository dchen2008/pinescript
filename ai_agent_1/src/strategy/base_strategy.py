"""Abstract strategy interface."""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from src.strategy.position import Position


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    @abstractmethod
    def on_bar(
        self,
        idx: int,
        row: pd.Series,
        position: Optional[Position],
        can_trade: bool,
        entering_quiet: bool,
    ) -> dict:
        """Process a single bar.

        Args:
            idx: Bar index in the dataframe
            row: Current bar data (includes all indicator columns)
            position: Current open position or None
            can_trade: Whether trading is allowed
            entering_quiet: Whether entering quiet window this bar

        Returns:
            Dict with possible actions:
            {
                "action": "open_long" | "open_short" | "close" | "none",
                "sl": float,        # For opens
                "tp": float | None, # For opens
                "close_reason": str, # For closes
            }
        """
        pass
