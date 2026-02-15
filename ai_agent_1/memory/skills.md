# Reusable Patterns

## OANDA Data Download
- Max 5000 candles per request
- ~375K candles for 1yr M1 data = ~75 requests
- Always filter for `complete=True` candles
- Rate limit: 0.1s between requests
- Early termination: if batch has <10 complete candles, near current time, stop
- Retry with exponential backoff for ConnectError/ReadTimeout

## Backtest Workflow
1. `python3 -m scripts.download_data --granularity M1`
2. `python3 -m scripts.run_backtest --strategy circle --label my_test`
3. `python3 -m scripts.run_optimizer --strategy circle`

## Parameter Optimization
- Grid: ATR factor x RR ratio x TP type x risk percent
- Rank by avg weekly return, filter by max drawdown
- Use multiprocessing for parallel execution
- Key insight: M1 circle >> M1 signal >> M5 anything

## Running Tests
```bash
python3 -m pytest tests/ -v
```

## Common Debugging
- Check `results/*_weekly.csv` for per-week PnL breakdown
- Check `results/*_trades.csv` for individual trade analysis
- Equity curve plots in `results/*_equity.png`
- Always verify drawdown by scanning weekly data for peak-to-trough
