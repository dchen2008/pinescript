#!/usr/bin/env python3
"""Download EUR/USD candle data from OANDA.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --granularity M5 --days 365
"""

import argparse
from datetime import datetime, timedelta, timezone

from src.data.oanda_client import OandaClient
from src.data.data_manager import save_candles, validate_candles


def main():
    parser = argparse.ArgumentParser(description="Download OANDA candle data")
    parser.add_argument("--instrument", default="EUR_USD", help="Instrument (default: EUR_USD)")
    parser.add_argument("--granularity", default="M1", help="Timeframe: M1, M5, H1 (default: M1)")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    parser.add_argument("--price", default="M", choices=["M", "B", "A"],
                        help="Price type: M=mid, B=bid, A=ask (default: M)")
    args = parser.parse_args()

    client = OandaClient()
    to_time = datetime.now(timezone.utc)
    from_time = to_time - timedelta(days=args.days)

    price_label = {"M": "mid", "B": "bid", "A": "ask"}[args.price]
    print(f"Downloading {args.instrument} {args.granularity} ({price_label}) from {from_time.date()} to {to_time.date()}")

    df = client.get_candles(args.instrument, args.granularity, from_time, to_time, price=args.price)

    if df.empty:
        print("No data received. Check your OANDA credentials in .env")
        return

    # Validate
    validation = validate_candles(df)
    print(f"Downloaded {validation['rows']} candles")
    print(f"Date range: {validation['date_range'][0]} to {validation['date_range'][1]}")
    print(f"Valid: {validation['valid']}")

    if validation["duplicates"] > 0:
        print(f"Warning: {validation['duplicates']} duplicate timestamps removed")

    # Save
    filepath = save_candles(df, args.instrument, args.granularity)
    print(f"Saved to: {filepath}")


if __name__ == "__main__":
    main()
