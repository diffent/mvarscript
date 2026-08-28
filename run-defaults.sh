#!/bin/sh
# Two-phase VAR solver run:
#   phase 1) backtest (ntrials >= 0) -- computes the ideal m1/m2/m3 ZTol
#            tolerances and writes them into the 'status' JSON file.
#   phase 2) forecast (ntrials = -1) -- reuses all the same parameters but with
#            the m*ZTol values pulled from phase 1's status file, so we don't
#            have to copy them out of the JSON by hand.
#
# Based on the default settings from solver-defaults.txt (untuned config, KNC
# model, shorting off).  Notes on translation to this .py's options:
#   - epsilon100=200 in the server payload means epsilon=2.0; this .py uses the
#     epsilon1000 option (x1000), so we send epsilon1000=2000 (also epsilon=2.0).
#   - shareCount/costPerTrade come from UI text fields on the server side; here
#     we use the .py's own defaults (1.0 / 10.0).

# run from this script's own directory so the 'running' / 'status' / *.csv files
# land next to the .py
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Directory where all run output (status, running, *.csv, *.pdf) is written.
# Defaults to the script's own directory (original behavior).  Set OUTDIR to run
# in an isolated per-parameter subdirectory (used by the parameter study).
OUTDIR="${OUTDIR:-$SCRIPT_DIR}"
mkdir -p "$OUTDIR" || exit 1
cd "$OUTDIR" || exit 1

# several.py is invoked by absolute path so it can run from any OUTDIR while
# still locating its helpers (found via __file__) in the script directory.
PY="$SCRIPT_DIR/several.py"

# get free keys from polygon.io and cryptocompare.com
# get from your environment variables or hard code here

POLYIOKEY=$POLYIOKEY
CRYPTOCOMPAREKEY=$CRYPTOCOMPAREKEY

# symbols to study (first symbol is the forecast target).
# Overridable via the SYMBOLS env var (used by the symbol study driver); falls
# back to the built-in pair so standalone behavior is unchanged.
SYMBOLS="${SYMBOLS:-AAPL MSFT}"

# number of backtest days for phase 1 
BACKTEST_NTRIALS=100

# one timestamp per script run, shared by the backtest and forecast copies so a
# matching pair is easy to correlate
RUN_TS=$(date +%Y%m%d-%H%M%S)

# run_solver <ntrials> <m1ZTol> <m2ZTol> <m3ZTol>
# every other solver parameter is held constant across both phases.
run_solver() {
  ntrials_arg="$1"
  m1_arg="$2"
  m2_arg="$3"
  m3_arg="$4"

  # control/kill-switch file the .py's monitor thread watches; delete it to stop
  touch running

  # shellcheck disable=SC2086  # SYMBOLS is intentionally word-split into args
  # see options of several.py which can be found by running several.py as a python script w/o args
  # for more details of what these args do
  python3 "$PY" \
    useopen=0 \
    polyiokey="$POLYIOKEY" \
    cryptocomparekey="$CRYPTOCOMPAREKEY" \
    ntrials="$ntrials_arg" \
    coolrate=0 \
    windowsize="${WINDOWSIZE:-200}" \
    neighbors="${NEIGHBORS:-20}" \
    knnvarcutoff="${KNNVARCUTOFF:-400}" \
    volen=21 \
    epsilon1000=2000 \
    exponent=2.0 \
    residExp=1.0 \
    m1ZTol="$m1_arg" \
    m2ZTol="$m2_arg" \
    m3ZTol="$m3_arg" \
    model1minabs=0 \
    shareCount=30.0 \
    costPerTrade=5.0 \
    daysWithheld=0 \
    allowShorting=1 \
    riskFreeRate=4.0 \
    normalize=1 \
    pullDelay=15 \
    uselogit=1 `# uselogit=1 && uselars=0 implies k nearest neighbors` \
    uselars=0  `# uselogit=0 && uselars=1 implies LARS regression` \
    lassolarsbic=1 `#0 implies AIC` \
    larsalpha=100 \
    noboot=1 \
    reuseMergedRaw="${REUSEMERGEDRAW:-0}" `# 1 => skip data pull/align, read mergedraw.csv from OUTDIR (set by the param study on 2nd+ runs)` \
    diffvol=1 \
    dyncutoff=0 \
    scramblesens=1 \
    $SYMBOLS
}

# --- phase 1: backtest. the 0.0 ztols we pass are ignored because backtest mode
#     recomputes the ideal tolerances (several.py gates that on ntrials > -1). ---
echo "=== phase 1: backtest (ntrials=$BACKTEST_NTRIALS) ==="
run_solver "$BACKTEST_NTRIALS" 0.0 0.0 0.0

# keep a timestamped copy of the backtest's status output
cp status "status.backtest.$RUN_TS"

# --- pull the ideal m*ZTol out of the status JSON the backtest just wrote,
#     before phase 2 overwrites it. ---
ZTOLS=$(python3 - <<'PYEOF'
import json
d = json.load(open("status"))
def fmt(x):
    x = float(x)
    s = repr(x)
    if "e" in s or "E" in s:   # avoid sci-notation: several.py needs a '.' to parse as float
        s = "{:.15f}".format(x)
    if "." not in s:
        s += ".0"
    return s
print(fmt(d["m1ZTol"]), fmt(d["m2ZTol"]), fmt(d["m3ZTol"]))
PYEOF
)

if [ -z "$ZTOLS" ]; then
  echo "error: could not read m1ZTol/m2ZTol/m3ZTol from status file; aborting forecast" >&2
  exit 1
fi

# split the three space-separated values
# shellcheck disable=SC2086
set -- $ZTOLS
M1="$1"
M2="$2"
M3="$3"
echo "=== found tolerances: m1ZTol=$M1 m2ZTol=$M2 m3ZTol=$M3 ==="

# --- phase 2: forecast (ntrials=-1) reusing everything, with the found tolerances. ---

# if'd out for now 

if [ 0 -eq 1 ] ; then
  echo "=== phase 2: forecast (ntrials=-1) with found tolerances ==="
  run_solver -1 "$M1" "$M2" "$M3"

  # keep a timestamped copy of the forecast's status output
  cp status "status.forecast.$RUN_TS"
fi
