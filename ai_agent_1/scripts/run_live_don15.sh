#!/usr/bin/env bash
# Live paper trading on OANDA demo account don15 (101-001-35749385-015)
# Runs in background with nohup, logs to logs/don15/
#
# Usage:
#   ./scripts/run_live_don15.sh          # start
#   ./scripts/run_live_don15.sh stop     # stop
#   ./scripts/run_live_don15.sh status   # check if running
#   tail -f logs/don15/live_trading.log  # watch logs

set -euo pipefail
cd "$(dirname "$0")/.."

PIDFILE="logs/don15/live.pid"
LOGDIR="logs/don15"
OUTFILE="$LOGDIR/nohup.out"

mkdir -p "$LOGDIR"

stop_trader() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping live trader (PID $PID)..."
            kill "$PID"
            # Wait up to 10s for graceful shutdown
            for i in $(seq 1 10); do
                if ! kill -0 "$PID" 2>/dev/null; then
                    echo "Stopped."
                    rm -f "$PIDFILE"
                    return 0
                fi
                sleep 1
            done
            echo "Force killing..."
            kill -9 "$PID" 2>/dev/null || true
            rm -f "$PIDFILE"
        else
            echo "PID $PID not running. Cleaning up stale pidfile."
            rm -f "$PIDFILE"
        fi
    else
        echo "No pidfile found — trader not running."
    fi
}

status_trader() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Live trader running (PID $PID)"
            echo "Log: $LOGDIR/live_trading.log"
            echo ""
            echo "=== Last 5 log lines ==="
            tail -5 "$LOGDIR/live_trading.log" 2>/dev/null || echo "(no log yet)"
            return 0
        else
            echo "PID $PID not running (stale pidfile)."
            rm -f "$PIDFILE"
            return 1
        fi
    else
        echo "Live trader not running."
        return 1
    fi
}

case "${1:-start}" in
    stop)
        stop_trader
        exit 0
        ;;
    status)
        status_trader
        exit $?
        ;;
    start)
        # Check if already running
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "Already running (PID $PID). Use '$0 stop' first."
                exit 1
            fi
            rm -f "$PIDFILE"
        fi

        echo "Starting live trader for don15 (101-001-35749385-015)..."

        # Override OANDA_ACCOUNT_ID for this account
        export OANDA_ACCOUNT_ID="101-001-35749385-015"

        nohup python3 -m scripts.run_live \
            --config config/default.yaml \
            --logs-dir "$LOGDIR" \
            > "$OUTFILE" 2>&1 &

        echo $! > "$PIDFILE"
        echo "Started (PID $(cat "$PIDFILE"))"
        echo "Logs:   tail -f $LOGDIR/live_trading.log"
        echo "Output: tail -f $OUTFILE"
        echo "Stop:   $0 stop"
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
