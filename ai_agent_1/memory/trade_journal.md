# Trade Journal

*Updated after each optimization run and trading session.*

## Initial Smoke Test (Synthetic Data)
- Signal strategy: 66 trades over ~5 days M1, 30.3% win rate (random data)
- Circle strategy: 2 trades (strict entry conditions)
- Both strategies confirmed working end-to-end

---

## Baseline: Signal M1 (Feb 2025 - Feb 2026)
- Default params: ATR=3.0, RR=1.5, Risk=1%
- 5598 trades, 27.9% WR, PF=0.62
- $10K -> $0.27 (total loss)
- Root cause: M1 signals = noise, 83% SL hit rate, 3.7 pip avg SL too small

## Signal M5 Grid Search
- Best: ATR=30, ATR_P=30, No TP -> +2.6% annual, PF=1.52, 54 trades
- Far below 25% weekly target

## Circle M1 Nth Circle Optimization
- Risk=1%, ATR=3.0, RR=2.5 -> +1220% annual, DD=12.8%, AvgWk=+5.12%
- Risk=3%, ATR=3.0, RR=2.5 -> +15799% annual, DD=8.0%, AvgWk=+11.57%
- Risk=15%, ATR=5.0, RR=2.0 -> AvgWk=+34.97%, DD=98.8%, 18/54 weeks >= 25%

## BEST CONFIG (Verified Feb 2026)
**M1 Nth Circle | ATR=5.0 | RR=2.0 | Risk=15% | No Quiet**
- $10,000 -> $1,023,228 (10,132%)
- 2352 trades, 52.34% WR, PF=1.29
- Avg weekly: 34.97%, median: 7.98%
- Best week: +314.11% (Jun 29), Worst week: -93.20% (Apr 27)
- Max drawdown: 98.8% ($47K -> $946 in Apr-Jun period)
- Recovery: $946 -> $1M+ in 4 months (Jul-Nov)

### Weekly Performance Highlights
- Feb: rocky start (-26%, +139%, -64%, +31%)
- Mar-Jun: extreme swings, near-wipeout ($47K -> $946)
- Jul-Sep: massive recovery ($946 -> $329K)
- Oct-Feb: steady growth with lower volatility ($329K -> $1M+)

### Key Observation
Strategy has survivorship bias risk in live trading. The 98.8% drawdown would likely trigger margin call on a real account. For live deployment, consider:
- Lower risk (3-5%) for survivable drawdowns
- Hard stop at 50% account drawdown
- Scale risk up gradually as profits accumulate
