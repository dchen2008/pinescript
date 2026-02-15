"""OANDA order management: create, modify, close orders."""

import logging
from typing import Optional

from src.data.oanda_client import OandaClient
from src.utils.forex_utils import format_price

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages orders on OANDA demo account."""

    def __init__(self, client: OandaClient, instrument: str = "EUR_USD"):
        self.client = client
        self.instrument = instrument

    def open_position(
        self,
        direction: int,
        units: float,
        sl_price: float,
        tp_price: Optional[float] = None,
    ) -> Optional[dict]:
        """Open a new position via market order.

        Args:
            direction: 1 for long, -1 for short
            units: Number of units (positive)
            sl_price: Stop loss price
            tp_price: Take profit price (optional)

        Returns:
            Dict with 'trade_id' and full response, or None on failure
        """
        signed_units = int(units) if direction == 1 else -int(units)

        try:
            result = self.client.create_market_order(
                instrument=self.instrument,
                units=signed_units,
                stop_loss=sl_price,
                take_profit=tp_price,
            )

            # Extract trade_id from fill response
            trade_id = None
            fill = result.get("orderFillTransaction")
            if fill and fill.get("tradeOpened"):
                trade_id = fill["tradeOpened"]["tradeID"]
            elif fill and fill.get("tradesClosed"):
                # Shouldn't happen for new position, but handle gracefully
                trade_id = fill["tradesClosed"][0].get("tradeID")

            result["trade_id"] = trade_id
            logger.info(
                f"Opened {'LONG' if direction == 1 else 'SHORT'} "
                f"{abs(signed_units)} units @ market, "
                f"SL={format_price(sl_price)}, "
                f"TP={format_price(tp_price) if tp_price else 'None'}, "
                f"trade_id={trade_id}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return None

    def close_all_positions(self) -> list:
        """Close all open trades."""
        results = []
        try:
            trades = self.client.get_open_trades()
            for trade in trades:
                trade_id = trade["id"]
                result = self.client.close_trade(trade_id)
                logger.info(f"Closed trade {trade_id}")
                results.append(result)
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")
        return results

    def modify_sl(self, trade_id: str, new_sl: float) -> Optional[dict]:
        """Update stop loss on an existing trade."""
        try:
            result = self.client.modify_trade(trade_id, stop_loss=new_sl)
            logger.info(f"Modified trade {trade_id} SL to {format_price(new_sl)}")
            return result
        except Exception as e:
            logger.error(f"Failed to modify SL: {e}")
            return None

    def get_current_position(self) -> Optional[dict]:
        """Get current open trade info.

        Returns:
            Dict with trade info or None if flat
        """
        try:
            trades = self.client.get_open_trades()
            if not trades:
                return None

            # Return first trade for instrument
            for trade in trades:
                if trade.get("instrument") == self.instrument:
                    units = int(trade["currentUnits"])
                    sl_price = None
                    tp_price = None
                    if trade.get("stopLossOrder"):
                        sl_price = float(trade["stopLossOrder"]["price"])
                    if trade.get("takeProfitOrder"):
                        tp_price = float(trade["takeProfitOrder"]["price"])
                    return {
                        "trade_id": trade["id"],
                        "direction": 1 if units > 0 else -1,
                        "units": abs(units),
                        "entry_price": float(trade["price"]),
                        "unrealized_pnl": float(trade.get("unrealizedPL", 0)),
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                    }
            return None
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None
