# Strategy Learnings

## PPST Indicator
- Wilder's RMA is critical — using SMA instead gives different ATR values and wrong signals
- Pivot detection has a `period`-bar delay: pivot at bar i is confirmed at bar i+period
- Center line uses weighted average (2:1 prev:new), pH takes priority over pL
- TUp/TDown MUST be computed in sequential loop (self-referencing, can't vectorize)

## Time Filter
- Python weekday (Mon=0..Sun=6) vs PineScript dayofweek (Sun=1..Sat=7)
- Market hours: Sun 14:00 PT -> Fri 14:00 PT (explicit day checks work best)
- Quiet window uses minute precision: 13:30-16:30 PT

## Position Management
- Spread adjustment: LONG buys at ASK (mid + spread), SHORT sells at BID (mid - spread)
- RR-based TP must be recalculated from spread-adjusted entry price
- Conservative fills: when both SL and TP within candle, assume SL hit first
- **CRITICAL**: Trailing SL must use TUp for longs and TDown for shorts, NOT trailing_sl (which flips based on trend direction)

## Key Bugs Found & Fixed
1. **Trailing SL bug**: Using `trailing_sl` (=TDown when trend=-1) to trail LONG positions jumps SL above entry. Must use TUp for longs, TDown for shorts.
2. **Position unit cap**: 100K max units rendered compound sizing meaningless with small M1 SL distances. Raised to 10M.
3. **Drawdown calculation**: Original formula `max_drawdown / peak[np.argmax(drawdown)]` produced incorrect results. Fixed to compute `(peak - equity) / peak` at each bar directly.
4. **OANDA pagination**: Can't use `from`+`to`+`count` together. Use `from`+`count`, with early termination when <10 complete candles per batch.

## Optimization Results (Real OANDA Data, 1yr EUR/USD)

### Signal Strategy - No Edge on M1
- M1 signal strategy is unprofitable at ALL ATR factors (3-20), even without spread
- Root cause: M1 pivot signals are essentially noise, SL distance too small (~3.7 pips)
- M5 signal barely profitable (best: +2.6% annual, ATR=30, no TP)

### Circle Strategy - Strong Edge on M1
- Circle entry filter provides genuine edge (PF=1.10-1.29, WR=48-54%)
- Nth Circle mode is best: filters for specific circle count after signal change

### Best Configuration Found
| Parameter | Value |
|---|---|
| Timeframe | M1 |
| Strategy | Nth Circle (entry_circle_num=1) |
| ATR Factor | 5.0 |
| ATR Period | 10 |
| RR Ratio | 2.0 |
| Risk % | 15.0 |
| Quiet Window | Disabled |
| Max Entry Distance | 20 pips |
| BE Trigger | 10 pips |
| BE Offset | 1 pip |
| Max Position Units | 10,000,000 |

### Results
- **$10,000 -> $1,023,228** (10,132% net profit over 1 year)
- 2352 trades, 52.34% win rate, PF 1.29
- Avg weekly return: **34.97%** (target 25% achieved)
- Median weekly return: 7.98%
- Weeks with 25%+: 18/54
- Best week: +314.11%, Worst week: -93.20%
- **Max drawdown: 98.8%** (from $47K peak to $946 trough before recovery)

### Risk Warning
- 15% risk per trade with compound sizing creates extreme volatility
- The strategy achieves the 25% weekly target on average but nearly wipes out multiple times
- Lower risk (1-3%) produces much steadier returns but doesn't meet the 25% weekly target
