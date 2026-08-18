#!/bin/sh
# Parameter study over the 'windowsize' and 'neighbors' solver parameters.
#
# It sweeps a 2-D grid: every windowsize value is paired with every neighbors
# value.  Each (windowsize, neighbors) combination runs in its own subdirectory
# named for both values:
#     windowsize=<ws>,neighbors=<nb>/
# and run-defaults.sh is pointed at that subdir (via OUTDIR) so ALL of the run's
# output -- status, running, the downloaded *.csv data, mergedraw.csv, the *.pdf
# plots, etc. -- is kept together, isolated per run.  A convenience copy of each
# run's final status JSON is also left at the top level as
#     status.windowsize=<ws>,neighbors=<nb>
# for quick side-by-side comparison.
#
# The parameters are passed to run-defaults.sh via the WINDOWSIZE and NEIGHBORS
# environment variables (run-defaults.sh falls back to its built-in defaults of
# 200 / 20 when they are unset, so its normal standalone behavior is unchanged).

# run from this script's own directory so subdirs / the .py land in the right place
cd "$(dirname "$0")" || exit 1
SCRIPT_DIR=$(pwd)

# build a space-separated list of NRUNS integer values centered on CENTER,
# spaced STEP apart:   make_grid CENTER STEP NRUNS
make_grid() {
  gc="$1"; gs="$2"; gn="$3"
  gh=$(( (gn - 1) / 2 ))
  out=""
  gi=$(( -gh ))
  while [ "$gi" -le "$gh" ]; do
    out="$out $(( gc + gi * gs ))"
    gi=$(( gi + 1 ))
  done
  echo "$out"
}

# --- study grids --------------------------------------------------------------
# windowsize: centered on run-defaults.sh's default (200), step 20, 5 runs:
#   160 180 200 220 240
WINDOWSIZES=$(make_grid 150 20 5)

# neighbors: centered on run-defaults.sh's default (20), step 5, 3 runs:
#   15 20 25
# NOTE: this is a full cross product -- total runs = (#windowsizes)*(#neighbors).
# Bump the neighbors grid up carefully; each run re-pulls data from the APIs.
NEIGHBORSLIST=$(make_grid 30 5 3)

echo "=== parameter study ==="
echo "windowsize values:$WINDOWSIZES"
echo "neighbors values:$NEIGHBORSLIST"

# accumulates one "<windowsize> <neighbors> <sharpe1> <sharpe2> <sharpe3>" line
# per run for the summary table
RESULTS=""

for WS in $WINDOWSIZES; do
  for NB in $NEIGHBORSLIST; do
    TAG="windowsize=$WS,neighbors=$NB"
    echo
    echo "################################################################"
    echo "### $TAG  ->  subdir $TAG/"
    echo "################################################################"

    # isolated output subdirectory for this run, keyed to both parameter values.
    # clear any leftovers from a prior study so each run starts clean.
    RUNDIR="$SCRIPT_DIR/$TAG"
    rm -rf "$RUNDIR"
    mkdir -p "$RUNDIR"

    # the .py's monitor thread watches for this kill-switch file in its run dir;
    # create it in the subdir up front (run-defaults.sh also touches it there)
    touch "$RUNDIR/running"

    # OUTDIR sends every output file into RUNDIR; WINDOWSIZE/NEIGHBORS set params
    OUTDIR="$RUNDIR" WINDOWSIZE="$WS" NEIGHBORS="$NB" ./run-defaults.sh
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "warning: run-defaults.sh exited $rc for $TAG" >&2
    fi

    # leave a top-level convenience copy of this run's status, tagged with both
    # parameter values (the full output set stays in the subdir)
    if [ -f "$RUNDIR/status" ]; then
      cp "$RUNDIR/status" "$SCRIPT_DIR/status.$TAG"
      echo "=== saved $TAG/ (status copied to status.$TAG) ==="
    else
      echo "warning: no 'status' file produced in $RUNDIR for $TAG" >&2
    fi

    # extract this run's sharpe1/sharpe2/sharpe3 using a real JSON parser
    # (python3's json module), not text scraping, so we correctly handle the
    # JSON structure/number format.  Printed space-separated on one line.
    SHARPES=$(python3 - "$RUNDIR/status" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    print(d["sharpe1"], d["sharpe2"], d["sharpe3"])
except Exception as e:
    print("ERROR ERROR ERROR")
PYEOF
)
    # split into the three values
    set -- $SHARPES
    SHARPE1="$1"; SHARPE2="$2"; SHARPE3="$3"
    echo "=== $TAG  sharpe1=$SHARPE1 sharpe2=$SHARPE2 sharpe3=$SHARPE3 ==="
    RESULTS="$RESULTS$WS $NB $SHARPE1 $SHARPE2 $SHARPE3
"
  done
done

echo
echo "=== parameter study complete ==="
echo "per-run output subdirectories:"
ls -d windowsize=*,neighbors=* 2>/dev/null
echo "top-level status copies:"
ls -1 status.windowsize=*,neighbors=* 2>/dev/null

# summary table: windowsize, neighbors vs sharpe1/sharpe2/sharpe3
echo
echo "=== results: windowsize, neighbors vs sharpe1/sharpe2/sharpe3 ==="
printf '%-12s  %-10s  %-22s  %-22s  %s\n' "windowsize" "neighbors" "sharpe1" "sharpe2" "sharpe3"
printf '%-12s  %-10s  %-22s  %-22s  %s\n' "----------" "---------" "-------" "-------" "-------"
printf '%s' "$RESULTS" | while read -r ws nb s1 s2 s3; do
  [ -n "$ws" ] && printf '%-12s  %-10s  %-22s  %-22s  %s\n' "$ws" "$nb" "$s1" "$s2" "$s3"
done
