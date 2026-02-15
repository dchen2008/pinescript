#!/usr/bin/env python3
"""WCSE grid search on M5: proximity × tp_rr × sl_buffer."""
import sys
sys.argv = ['vol']
from src.data.data_manager import load_candles
from src.vol import run_vol_backtest

df = load_candles('EUR_USD', 'M5')
print(f'Loaded {len(df)} candles\n')

ppst_params = {'pivot_period': 2, 'atr_factor': 5.0, 'atr_period': 10}
vol_params = {'use_vol_filter': True, 'vol_sma_period': 30, 'vol_threshold': 1.0, 'vol_recovery_bars': 0, 'filtered_exit_bars': 0}
quiet_params = {'enabled': True, 'start_hour': 12, 'start_min': 0, 'end_hour': 14, 'end_min': 30, 'tz': 'America/Los_Angeles'}
strategy_params = {'swing_close': False, 'sl_buffer_pips': 3.0, 'tp_type': 'rr', 'tp_ratio': 2.0, 'tp_fixed_pips': 30.0, 'start_date': None}
sizing_params = {'initial_capital': 10000.0, 'risk_percent': 1.0, 'max_position_units': 10000000, 'spread_pips': 1.5}

# Proximity configs: (c1_wick, c1_close, c2_wick, c2_close, label)
proximities = [
    (2.0, 3.0, 2.0, 3.0, "tight(2/3)"),
    (5.0, 5.0, 5.0, 5.0, "med(5/5)"),
    (10.0, 10.0, 10.0, 10.0, "wide(10/10)"),
    (15.0, 15.0, 10.0, 10.0, "xwide(15/10)"),
]
sl_buffers = [0.0, 1.0, 2.0]
rr_ratios = [2.0, 3.0, 4.0, 5.0]

results = []

print(f"{'Prox':<14} | {'SL':>3} | {'RR':>4} | {'Net Profit':>12} | {'WCSE#':>5} | {'WR':>6} | {'PF':>5} | {'MaxDD':>7}")
print('-' * 78)

for prox in proximities:
    c1w, c1c, c2w, c2c, plabel = prox
    for sl_buf in sl_buffers:
        for rr in rr_ratios:
            wcse_params = {
                'enabled': True, 'entry_times': 1,
                'c1_body_pips': 1.0, 'c1_wick_pips': c1w, 'c1_close_pips': c1c,
                'c2_body_pips': 0.5, 'c2_wick_pips': c2w, 'c2_close_pips': c2c,
                'tp_rr': rr, 'sl_buffer_pips': sl_buf,
                'vf_skip': True, 'vf_threshold': 3.0,
            }
            result = run_vol_backtest(df, ppst_params, vol_params, quiet_params, strategy_params, sizing_params, wcse_params=wcse_params)

            tl = result['trade_log']
            total = len(tl)
            wins = sum(1 for t in tl if t['pnl'] > 0)
            wr = wins / total * 100 if total > 0 else 0
            net = sum(t['pnl'] for t in tl)
            wcse_n = result['stats']['wcse_entries']

            gross_profit = sum(t['pnl'] for t in tl if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in tl if t['pnl'] < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else 0

            ec = result['equity_curve']
            peak = 0.0
            max_dd = 0.0
            for point in ec:
                eq = point['balance'] + point.get('unrealized', 0.0)
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            results.append((plabel, sl_buf, rr, net, wcse_n, wr, pf, max_dd))
            print(f"{plabel:<14} | {sl_buf:>3.0f} | {rr:>4.1f} | ${net:>10,.0f} | {wcse_n:>5} | {wr:>5.1f}% | {pf:>5.2f} | {max_dd:>6.1%}")
        print()

# Top 10
print("\n>>> TOP 10 by Net Profit:")
top10 = sorted(results, key=lambda x: x[3], reverse=True)[:10]
for i, r in enumerate(top10, 1):
    print(f"  {i:>2}. {r[0]:<14} SL={r[1]:.0f} RR={r[2]:.1f} -> ${r[3]:>8,.0f} | WCSE:{r[4]:>3} | WR:{r[5]:.1f}% PF:{r[6]:.2f} DD:{r[7]:.1%}")
