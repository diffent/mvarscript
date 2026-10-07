#!/bin/sh
# Forward forecast run using the hyperparameters found by the parameter study.
#
# This is a thin wrapper around run-defaults.sh that:
#   1. sets the optimized hyperparameters (windowsize / neighbors / knnvarcutoff /
#      elasticalpha / annealmaxiter),
#   2. forces a FRESH data pull (reuseMergedRaw=0 -- no cached mergedraw.csv), and
#   3. enables run-defaults.sh's phase-2 forecast (ntrials=-1).
#
# How the forecast actually works -- the deal with the m*ZTol tolerances:
#   A forecast (ntrials=-1) does NOT recompute the m1/m2/m3 ZTol tolerances; it
#   only *consumes* them.  several.py derives those tolerances solely in backtest
#   mode (ntrials>-1).  So a forecast cannot stand alone: run-defaults.sh still
#   runs its phase-1 backtest first (to produce the ideal tolerances), then
#   phase-2 forecasts the next day using them.  This script therefore runs BOTH
#   phases; DO_FORECAST=1 just turns the forecast phase on.  The phase-1 backtest
#   (BACKTEST_NTRIALS days) is the bulk of the runtime and is not skippable in the
#   current design.
#
# TODO / under advisement (deferred for now): to avoid re-running the backtest on
# every forecast, we could instead (a) let forecast.sh accept pre-computed m*ZTol
# values (passed straight to a lone ntrials=-1 run), and/or (b) auto-load the best
# hyperparameters from a param study's current_best.txt.  Not implemented yet.
#
# Everything is passed via the same env vars run-defaults.sh already honors, so
# no logic is duplicated here.  Override any value from the environment, e.g.:
#   SYMBOLS="NVDA AAPL" WINDOWSIZE=150 ./forecast.sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# --- optimized hyperparameters (edit to the best run from the param study) -----
# Each falls back to the value below only if not already set in the environment.
# These mirror every model hyperparameter env var that run-defaults.sh honors.
export WINDOWSIZE="${WINDOWSIZE:-100}"
export NEIGHBORS="${NEIGHBORS:-2}"
export KNNVARCUTOFF="${KNNVARCUTOFF:-780}"
export ELASTICALPHA="${ELASTICALPHA:-0.01}"      # ElasticNet penalty strength (only used when sublinearType=ElasticNet)
export ANNEALMAXITER="${ANNEALMAXITER:-801}"     # dual_annealing max global iterations for the model-1 solve

# symbols to forecast (first symbol is the forecast target); run-defaults.sh
# defaults these if unset, but set them here for a self-contained forecast.
export SYMBOLS="${SYMBOLS:-AMD NVDA}"

# --- forecast wiring -----------------------------------------------------------
export REUSEMERGEDRAW=0   # always pull fresh data for a real forecast
export DO_FORECAST=1      # enable run-defaults.sh's phase-2 forecast (ntrials=-1)

echo "=== forecast: SYMBOLS='$SYMBOLS' windowsize=$WINDOWSIZE neighbors=$NEIGHBORS knnvarcutoff=$KNNVARCUTOFF elasticalpha=$ELASTICALPHA annealmaxiter=$ANNEALMAXITER ==="
echo "=== fresh data pull (reuseMergedRaw=0), phase-2 forecast enabled (ntrials=-1) ==="

# where run-defaults.sh writes its output; mirror its own OUTDIR default so we
# can find the status file it produces.
OUTDIR="${OUTDIR:-$SCRIPT_DIR}"

# Run the two-phase solver.  This used to be `exec`'d; it is now a normal call so
# we can summarize the forecast afterwards (exec would have replaced this shell).
"$SCRIPT_DIR/run-defaults.sh"
rc=$?

# --- report the forecast_* key/value pairs from the final status file ----------
# The phase-2 forecast is the last thing run-defaults.sh writes to 'status', so
# that file holds this run's forecast.  The status JSON is pretty-printed (one
# key per line), so grep + a light cleanup yields clean "key = value" lines.
STATUS_FILE="$OUTDIR/status"
echo "=== forecast results (forecast_* from $STATUS_FILE) ==="
if [ -f "$STATUS_FILE" ]; then
  # sort -V so forecast_ahead1,2,3(,10...) come out in numeric order
  forecast_pairs=$(grep 'forecast_' "$STATUS_FILE" \
                   | sed 's/^[[:space:]]*//; s/,[[:space:]]*$//; s/"//g; s/: */ = /' \
                   | sort -V)
  if [ -n "$forecast_pairs" ]; then
    printf '%s\n' "$forecast_pairs"
  else
    echo "(no forecast_ keys found in status)"
  fi
else
  echo "warning: no status file at $STATUS_FILE to report forecast_ keys from" >&2
fi

exit $rc
