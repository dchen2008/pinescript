# PPST-V1 Pine Script Documentation

## Overview

Enhanced Pivot Point SuperTrend indicator with entry circle markers and webhook alerts for automated trading.

**File:** `ppst_v1.pine`

## Features

1. **PP SuperTrend Indicator** - Original indicator with trend detection
2. **BUY/SELL Signal Labels** - Trend change markers
3. **Entry Circle Markers** - Visual entry opportunities after signal swings
4. **Webhook Alerts** - JSON-formatted alerts for automated trading

---

## Input Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Pivot Point Period | 2 | Bars for pivot detection (lookback/forward) |
| ATR Factor | 3 | Multiplier for ATR bands |
| ATR Period | 10 | Period for ATR calculation |
| Show Pivot Points | false | Display H/L markers |
| Show Buy/Sell Labels | true | Display signal labels |
| Show PP Center Line | false | Display center line |
| Show Support/Resistance | false | Display S/R levels |
| Show Entry Circles | true | Display entry circle markers |

### Backtest Parameters

| Setting | Default | Description |
|---------|---------|-------------|
| Use Time Filter | true | Enable/disable date range filtering |
| Backtest Start Time | 2026-01-01 | Start date for statistics (datetime picker) |
| Backtest End Time | 2026-12-31 | End date for statistics (datetime picker) |
| SL Spread Buffer (pips) | 1.5 | Stop loss spread buffer for calculations |
| TP Ratio (R:R) | 1.5 | Take profit risk-reward multiplier |
| Entry Window (minutes) | 30 | Time window to count entries after signal |
| Max Entries/Period (0=all) | 0 | Max entries to count per period (0=unlimited) |
| Show Statistics Table | true | Display statistics overlay table |

---

## Entry Circle Logic

### LONG Entry (Green Circle) - After BUY Signal

Must meet ALL 4 conditions:
1. `open < resistance` - Open price below PH
2. `close > resistance` - Close price above PH (candle crosses up)
3. `body_mid > resistance` - 50%+ of candle body above PH
4. `close > center` - Close above Center Line

### SHORT Entry (Red Circle) - After SELL Signal

Must meet ALL 4 conditions:
1. `open > support` - Open price above PL
2. `close < support` - Close price below PL (candle crosses down)
3. `body_mid < support` - 50%+ of candle body below PL
4. `close < center` - Close below Center Line

### Entry Timing

| Candle | S/R Reference |
|--------|---------------|
| Signal candle (BUY/SELL) | Original `resistance`/`support` from before signal |
| After signal candle | New `entry_resistance`/`entry_support` from PH/PL formed after signal |

**Note:** ALL qualifying candles during a signal period get circled (not just the first one).

---

## Webhook Setup in TradingView

### Step 1: Create Alert

1. Right-click on chart → "Add Alert"
2. **Condition:** Select `PPST-V1` indicator
3. **Trigger:** Select `Any alert() function call`
4. Leave **Message** field empty (script provides the JSON)

### Step 2: Configure Webhook

1. Enable "Webhook URL" checkbox
2. Enter your webhook endpoint:
   - Local: `http://localhost:5555/webhook`
   - Via ngrok: `https://xxxx.ngrok.io/webhook`

### Step 3: Alert Settings

- **Alert name:** PPST-V1 EUR/USD 1m (or your preference)
- **Expiration:** Open-ended (recommended)
- **Alert actions:** Webhook URL only

---

## Webhook Message Formats

### Message Categories

| Type | Description | Order |
|------|-------------|-------|
| `SIGNAL_SWING` | PP SuperTrend trend change (BUY/SELL) | Fires first |
| `ENTRY` | Filtered entry opportunity (LONG_ENTRY/SHORT_ENTRY) | Fires second |

When both fire on the same candle (signal candle meets all 4 entry conditions), SIGNAL_SWING fires first.

### Signal Types

| Signal | Type | Description |
|--------|------|-------------|
| `BUY` | SIGNAL_SWING | Trend changed to BULLISH |
| `SELL` | SIGNAL_SWING | Trend changed to BEARISH |
| `LONG_ENTRY` | ENTRY | Entry circle triggered after BUY signal |
| `SHORT_ENTRY` | ENTRY | Entry circle triggered after SELL signal |

---

### Category 1: Signal Swing (`type: "SIGNAL_SWING"`)

Fires when PP SuperTrend trend changes direction.

**BUY Signal:**
```json
{
  "type": "SIGNAL_SWING",
  "instrument": "EURUSD",
  "tf": "1",
  "signal": "BUY",
  "price": 1.18070,
  "supertrend": 1.18012,
  "trend": 1,
  "time": "2026-02-04T12:30:00Z"
}
```

**SELL Signal:**
```json
{
  "type": "SIGNAL_SWING",
  "instrument": "EURUSD",
  "tf": "1",
  "signal": "SELL",
  "price": 1.18150,
  "supertrend": 1.18205,
  "trend": -1,
  "time": "2026-02-04T14:45:00Z"
}
```

**SIGNAL_SWING Fields:**

| Field | Format | Description |
|-------|--------|-------------|
| `type` | String | Category identifier: `"SIGNAL_SWING"` |
| `instrument` | String | TradingView ticker (e.g., "EURUSD") |
| `tf` | String | Timeframe ("1" for 1m, "5" for 5m, etc.) |
| `signal` | String | Direction: `"BUY"` or `"SELL"` |
| `price` | Float | Close price (5 decimal places) |
| `supertrend` | Float | SuperTrend value (5 decimal places) |
| `trend` | Integer | Numeric trend: `1` (bullish) or `-1` (bearish) |
| `time` | ISO 8601 | Alert timestamp |

---

### Category 2: Entry (`type: "ENTRY"`)

Fires when candle meets all 4 entry conditions after signal.

**LONG_ENTRY:**
```json
{
  "type": "ENTRY",
  "instrument": "EURUSD",
  "tf": "1",
  "signal": "LONG_ENTRY",
  "price": 1.18120,
  "supertrend": 1.18050,
  "resistance": 1.18100,
  "center": 1.18080,
  "time": "2026-02-04T12:35:00Z"
}
```

**SHORT_ENTRY:**
```json
{
  "type": "ENTRY",
  "instrument": "EURUSD",
  "tf": "1",
  "signal": "SHORT_ENTRY",
  "price": 1.18080,
  "supertrend": 1.18180,
  "support": 1.18100,
  "center": 1.18120,
  "time": "2026-02-04T14:50:00Z"
}
```

**ENTRY Fields:**

| Field | Format | Description |
|-------|--------|-------------|
| `type` | String | Category identifier: `"ENTRY"` |
| `instrument` | String | TradingView ticker (e.g., "EURUSD") |
| `tf` | String | Timeframe ("1" for 1m, "5" for 5m, etc.) |
| `signal` | String | Direction: `"LONG_ENTRY"` or `"SHORT_ENTRY"` |
| `price` | Float | Close price (5 decimal places) |
| `supertrend` | Float | SuperTrend value (5 decimal places) |
| `resistance` | Float | Current resistance level (PH) - LONG_ENTRY only |
| `support` | Float | Current support level (PL) - SHORT_ENTRY only |
| `center` | Float | Center line value |
| `time` | ISO 8601 | Alert timestamp |

**Note:** TradingView returns `EURUSD` format, not `EUR_USD`. Convert on receiving end if needed.

---

## Testing Webhook

### Using curl

```bash
# Test SIGNAL_SWING (BUY)
curl -X POST http://localhost:5555/webhook \
  -H "Content-Type: application/json" \
  -d '{"type":"SIGNAL_SWING","instrument":"EURUSD","tf":"1","signal":"BUY","price":1.18070,"supertrend":1.18012,"trend":1,"time":"2026-02-04T12:30:00Z"}'

# Test SIGNAL_SWING (SELL)
curl -X POST http://localhost:5555/webhook \
  -H "Content-Type: application/json" \
  -d '{"type":"SIGNAL_SWING","instrument":"EURUSD","tf":"1","signal":"SELL","price":1.18150,"supertrend":1.18205,"trend":-1,"time":"2026-02-04T14:45:00Z"}'

# Test ENTRY (LONG_ENTRY)
curl -X POST http://localhost:5555/webhook \
  -H "Content-Type: application/json" \
  -d '{"type":"ENTRY","instrument":"EURUSD","tf":"1","signal":"LONG_ENTRY","price":1.18120,"supertrend":1.18050,"resistance":1.18100,"center":1.18080,"time":"2026-02-04T12:35:00Z"}'

# Test ENTRY (SHORT_ENTRY)
curl -X POST http://localhost:5555/webhook \
  -H "Content-Type: application/json" \
  -d '{"type":"ENTRY","instrument":"EURUSD","tf":"1","signal":"SHORT_ENTRY","price":1.18080,"supertrend":1.18180,"support":1.18100,"center":1.18120,"time":"2026-02-04T14:50:00Z"}'
```

### Using test script

```bash
./scripts/test_webhook.sh signal=BUY tf=1m
./scripts/test_webhook.sh signal=SELL tf=1m
./scripts/test_webhook.sh signal=LONG_ENTRY tf=1m
./scripts/test_webhook.sh signal=SHORT_ENTRY tf=1m
```

---

## Integration with Trading Bot

The webhook signals can be received by the OANDA stream daemon and forwarded to trading bots.

### Stream Daemon Config (`src/oanda_stream_config.yaml`)

```yaml
webhook:
  enabled: true
  http_port: 5555
  http_host: "0.0.0.0"
  auth_token: null
  signal_ttl_seconds: 30
```

### Bot Config (`{account}/config.yaml`)

```yaml
webhook_signals:
  enabled: true
  signal_ttl_seconds: 30
  rest_validation:
    enabled: false
    on_mismatch: "warn"
```

See [STREAM_DAEMON.md](STREAM_DAEMON.md) for full webhook integration details.

---

## Troubleshooting

### No alerts firing

1. Check "Any alert() function call" is selected as trigger
2. Verify indicator is added to chart
3. Check alert is not expired

### Wrong instrument format

TradingView sends `EURUSD`, bot expects `EUR_USD`. The stream daemon webhook handler should convert this automatically.

### Missing entry circles

Entry circles only appear when ALL 4 conditions are met:
- Enable "Show Support/Resistance" and "Show PP Center Line" to debug visually
- Check that a new PH/PL has formed after the signal (for post-signal entries)

### Price precision issues

All prices are formatted to 5 decimal places (`#.#####`) in the script.

---

## Statistics Feature

### Overview

The statistics table provides real-time tracking of signals, entries, and timing analysis. It appears as an overlay in the top-right corner of the chart.

### Time Filter

The time filter allows you to analyze statistics for a specific date range:

| Setting | Description |
|---------|-------------|
| Use Time Filter | Enable to filter by date range, disable to count all bars on chart |
| Backtest Start Time | Start date/time (uses datetime picker) |
| Backtest End Time | End date/time (uses datetime picker) |

**Note:** The filter uses `syminfo.timezone` to ensure consistent timezone handling between bar timestamps and input dates.

### Statistics Tracked

| Statistic | Format | Description |
|-----------|--------|-------------|
| Signal Swings | `103 (B:52 S:51)` | Total signals with BUY/SELL breakdown |
| Entry Points | `164 (L:80 S:84)` | Total entries with LONG/SHORT breakdown |
| Same Bar | `17` | Entries on same candle as signal |
| Entry ≤Xmin | `59` | Entries within configured time window after signal |
| Periods w/ ≥2 Entries | `38` | Signal periods with 2+ entry opportunities |
| Entry/Signal Ratio | `159.2%` | Average entries per signal swing |

### Abbreviations

| Code | Meaning |
|------|---------|
| **B** | BUY signals (trend: bearish → bullish) |
| **S** | SELL signals (trend: bullish → bearish) |
| **L** | LONG entries (entry circles after BUY) |
| **S** | SHORT entries (entry circles after SELL) |

### Max Entries/Period Filter

Controls how many entries per signal period are counted:

| Value | Behavior |
|-------|----------|
| 0 | Count all entries (no filter) |
| 1 | Only first entry per signal period |
| 2 | First and second entry only |
| 3+ | First N entries only |

When filtering is active (value > 0), the table shows both filtered and raw entry counts.

### Statistics Table Example

**With time filter enabled (Jan 26-31, 2026):**
```
┌──────────────────────────────────┐
│ 2026-01-26        │ 2026-01-31   │
│ Filter: ON        │ 7200 bars    │
├──────────────────┬───────────────┤
│ PPST-V1          │ Statistics    │
│ SL Buffer        │ 1.5 pips      │
│ TP Ratio         │ 1.5 R:R       │
│ Entry Window     │ 30 min        │
│ Max Entries      │ All           │
├──────────────────┼───────────────┤
│ Signal Swings    │ 103 (B:52 S:51)│
│ Entry Points     │ 164 (L:80 S:84)│
│ Same Bar         │ 17            │
│ Entry ≤30min     │ 59            │
│ Periods ≥2 Ent   │ 38            │
├──────────────────┼───────────────┤
│ Entry/Signal     │ 159.2%        │
└──────────────────┴───────────────┘
```

**With time filter disabled (counts all bars on chart):**
```
┌──────────────────────────────────┐
│ 2025-10-26        │ 2026-02-05   │
│ Filter: OFF       │ 20645 bars   │
├──────────────────┬───────────────┤
│ PPST-V1          │ Statistics    │
│ SL Buffer        │ 1.5 pips      │
│ TP Ratio         │ 1.5 R:R       │
│ Entry Window     │ 30 min        │
│ Max Entries      │ All           │
├──────────────────┼───────────────┤
│ Signal Swings    │ 370 (B:185 S:185)│
│ Entry Points     │ 499 (L:256 S:243)│
│ Same Bar         │ 75            │
│ Entry ≤30min     │ 24            │
│ Periods ≥2 Ent   │ 125           │
├──────────────────┼───────────────┤
│ Entry/Signal     │ 134.9%        │
└──────────────────┴───────────────┘
```

### Time Window Calculation

The "Entry ≤Xmin" statistic calculates time based on:
- `bars_since_signal * timeframe.multiplier`
- Only works for minute-based timeframes (1m, 5m, 15m, etc.)
- Excludes same-bar entries (counted separately)

### Period Definition

A "period" is the span between two consecutive signal swings (BUY→SELL or SELL→BUY). Entry statistics are tracked per period and reset on each new signal.

### Notes

1. **SL/TP Parameters**: Currently stored as inputs for display only. Not used in calculations (statistics mode only). Can be extended for full backtesting.

2. **Table Visibility**: Toggle "Show Statistics Table" to hide/show the overlay.

3. **Performance**: Statistics are calculated on every bar but the table only renders on the last bar (`barstate.islast`).

4. **Timezone Handling**: The time filter uses `syminfo.timezone` to ensure consistent timezone between bar timestamps and input dates. This prevents timezone mismatch issues that could cause incorrect filtering.

5. **Expected Signal Frequency (1m EUR/USD)**:
   - ~20-24 signal swings per day
   - ~40-70 minutes between signal swings
   - ~130 signals per week
