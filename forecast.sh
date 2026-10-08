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
export DAYSWITHHELD="${DAYSWITHHELD:-0}"          # drop this many most-recent days from the data (several.py daysWithheld)

# symbols to forecast (first symbol is the forecast target); run-defaults.sh
# defaults these if unset, but set them here for a self-contained forecast.
export SYMBOLS="${SYMBOLS:-AMD NVDA}"

# --- forecast wiring -----------------------------------------------------------
export REUSEMERGEDRAW=0   # always pull fresh data for a real forecast
export DO_FORECAST=1      # enable run-defaults.sh's phase-2 forecast (ntrials=-1)

echo "=== forecast: SYMBOLS='$SYMBOLS' windowsize=$WINDOWSIZE neighbors=$NEIGHBORS knnvarcutoff=$KNNVARCUTOFF elasticalpha=$ELASTICALPHA annealmaxiter=$ANNEALMAXITER daysWithheld=$DAYSWITHHELD ==="
echo "=== fresh data pull (reuseMergedRaw=0), phase-2 forecast enabled (ntrials=-1) ==="

# where run-defaults.sh writes its output; mirror its own OUTDIR default so we
# can find the status file it produces.
OUTDIR="${OUTDIR:-$SCRIPT_DIR}"

# Run the two-phase solver.  This used to be `exec`'d; it is now a normal call so
# we can summarize the forecast afterwards (exec would have replaced this shell).
"$SCRIPT_DIR/run-defaults.sh"
rc=$?

# --- report the forecast_* pairs + the zero tolerances from the final status ---
# The phase-2 forecast is the last thing run-defaults.sh writes to 'status', so
# that file holds this run's forecast.  For each forecast_ahead<n> we also show
# the matching m<n>ZTol zero tolerance (computed in the backtest) and the
# resulting UP/DOWN/INDETERMINATE call: a forecast within +-tolerance of zero is
# indeterminate; outside it, the sign gives the direction.  (Done in python for
# the per-model join + float compare; same heredoc pattern run-defaults.sh uses.)
STATUS_FILE="$OUTDIR/status"
echo "=== forecast results (forecast_* from $STATUS_FILE) ==="
if [ -f "$STATUS_FILE" ]; then
  python3 - "$STATUS_FILE" <<'PYEOF'
import json, re, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("(could not read status JSON: %s)" % e)
    sys.exit(0)

fkeys = [k for k in d if k.startswith("forecast_")]
if not fkeys:
    print("(no forecast_ keys found in status)")
    sys.exit(0)

def natkey(k):                       # natural sort: ahead1 < ahead2 < ahead3 < day
    m = re.search(r'(\d+)$', k)
    return (re.sub(r'\d+$', '', k), int(m.group(1)) if m else -1)

def fmt(x):                          # numbers to 4 decimals; leave strings as-is
    return "%.4f" % x if isinstance(x, float) else str(x)

# first pass: build the rows as (key, value, tolkey, tol, call); tolkey/tol/call
# are None for non-ahead keys (e.g. forecast_day)
rows = []
for k in sorted(fkeys, key=natkey):
    v = d[k]
    m = re.match(r'forecast_ahead(\d+)$', k)
    if m:
        tolkey = "m%sZTol" % m.group(1)
        tol = d.get(tolkey)
        try:
            fv, tv = float(v), float(tol)
            if abs(fv) <= tv:
                call = "INDETERMINATE, NO TRADE RECOMMENDED"
            else:
                call = "UP" if fv > 0 else "DOWN"
            rows.append((k, fmt(v), tolkey, fmt(tol), call))
        except (TypeError, ValueError):
            rows.append((k, fmt(v), tolkey, fmt(tol), "(no call)"))
    else:
        rows.append((k, fmt(v), None, None, None))

# second pass: print with aligned columns
keyw = max(len(r[0]) for r in rows)
valw = max(len(r[1]) for r in rows)
tolkeyw = max([len(r[2]) for r in rows if r[2] is not None] or [0])
tolw = max([len(r[3]) for r in rows if r[3] is not None] or [0])
for key, val, tolkey, tol, call in rows:
    if tolkey is None:
        line = "%-*s = %-*s" % (keyw, key, valw, val)
    else:
        line = "%-*s = %-*s   %-*s = %-*s   ->   %s" % (
            keyw, key, valw, val, tolkeyw, tolkey, tolw, tol, call)
    print(line.rstrip())
PYEOF
else
  echo "warning: no status file at $STATUS_FILE to report forecast_ keys from" >&2
fi

exit $rc
