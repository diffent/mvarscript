#!/usr/bin/env python3
"""Live monitor for an in-progress parameter/symbol study.

Polls the status JSON files a running study writes (named like
    status.symbols=AMD-NVDA,target=sortino3,windowsize=100,neighbors=11,knnvarcutoff=140
so the run parameters live right in the filename) and flags any run whose
results look interesting: a HIGH sortino ratio paired with a LOW p-value for the
*same model*.

For model n in {1,2,3} the pairing is:
    sortino ratio  -> "sortino<n>"   (downside-deviation sortino)
    p-value        -> "bestM<n>pval"
(The "sortino<n>p" keys are the upside-volatility variant, NOT p-values, so they
are ignored here.)

A hit is reported when   sortino<n> >= SORTINO_MIN   and   bestM<n>pval <= PVAL_MAX.

Runs forever, re-scanning every POLL_SECONDS.  Each scan prints a one-line
heartbeat (so you can see it is alive) followed by a full dump of every
interesting hit found so far, sorted best-first -- so you never have to scroll
back to see the current leaders.  Hits accumulate across scans; if a study
overwrites a status file, that hit's values are refreshed in place.

This script only READS status files -- it never writes or edits anything, so it
is safe to run alongside a live study.

Usage:
    python3 monitor-study.py                     # defaults, scan this script's dir
    python3 monitor-study.py --dir /path/to/runs
    python3 monitor-study.py --sortino-min 0.4 --pval-max 0.05 --interval 2
    python3 monitor-study.py --glob 'status.symbols=*'
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

# defaults (all overridable on the command line)
POLL_SECONDS = 5.0            # seconds between scans
SORTINO_MIN = 0.30            # a sortino ratio at or above this is "high"
PVAL_MAX = 0.10               # a p-value at or below this is "low"
STATUS_GLOB = "status.symbols=*"   # only the param-tagged status copies
# each param-study run makes its own subdir up front; a new one appearing marks
# the start of the next run, so the gap between appearances is a run's duration.
RUN_SUBDIR_GLOB = "*windowsize=*,neighbors=*,knnvarcutoff=*"
MODELS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live monitor for a running study.")
    p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="directory holding the status files (default: this script's dir)")
    p.add_argument("--glob", default=STATUS_GLOB,
                   help=f"glob for status files (default: {STATUS_GLOB!r})")
    p.add_argument("--sortino-min", type=float, default=SORTINO_MIN,
                   help=f"minimum sortino to flag (default: {SORTINO_MIN})")
    p.add_argument("--pval-max", type=float, default=PVAL_MAX,
                   help=f"maximum p-value to flag (default: {PVAL_MAX})")
    p.add_argument("--interval", type=float, default=POLL_SECONDS,
                   help=f"seconds between scans (default: {POLL_SECONDS})")
    return p.parse_args()


def params_from_name(path: str) -> str:
    """The run parameters are the filename with the leading 'status.' stripped."""
    base = os.path.basename(path)
    return base[len("status."):] if base.startswith("status.") else base


def report_subdir_timing(args: argparse.Namespace, timing: dict) -> None:
    """Report the wall-clock time each run subdir took.

    A new subdir means the next run just started, so the gap since the previous
    new subdir is the duration of the run that just finished.  We print that time
    for the subdir that just completed.  `timing` persists across scans: 'known'
    (subdirs seen) and 'last' (name, time of the most recent new subdir).
    """
    subdirs = {p for p in glob.glob(os.path.join(args.dir, RUN_SUBDIR_GLOB))
               if os.path.isdir(p)}
    new = sorted(subdirs - timing["known"])
    now = time.time()

    if not timing["known"]:
        # first scan: adopt what already exists without inventing a time
        timing["known"] = subdirs
        if new:
            timing["last"] = (new[-1], now)
        return

    for path in new:
        if timing["last"] is not None:
            prev_name, prev_time = timing["last"]
            print(f"    subdir done in {fmt_duration(now - prev_time)}: "
                  f"{os.path.basename(prev_name)}", flush=True)
        timing["known"].add(path)
        timing["last"] = (path, now)


def fmt_duration(seconds: float) -> str:
    """Human-friendly h/m/s from a seconds count."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def scan_once(args: argparse.Namespace, found: dict, timing: dict) -> None:
    """One pass: read every matching status file, refresh the accumulated set of
    interesting hits, then dump the whole set (best-first) so the current leaders
    are always visible without scrolling back.

    `found` is the cumulative store, keyed by (path, model) -> (sortino, pval,
    params); it persists across scans and is updated in place when a file's
    values change.
    """
    pattern = os.path.join(args.dir, args.glob)
    files = sorted(glob.glob(pattern))

    interesting_now = 0
    for path in files:
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            # file mid-write / truncated / not yet valid JSON -- skip this pass
            continue

        for n in MODELS:
            sortino = d.get(f"sortino{n}")
            pval = d.get(f"bestM{n}pval")
            if sortino is None or pval is None:
                continue
            try:
                sortino = float(sortino)
                pval = float(pval)
            except (TypeError, ValueError):
                continue
            if sortino >= args.sortino_min and pval <= args.pval_max:
                interesting_now += 1
                # rawReturn<n> = total backtest return for this model (may be
                # absent in older status files written before it was added)
                raw = d.get(f"rawReturn{n}")
                try:
                    raw = float(raw)
                except (TypeError, ValueError):
                    raw = None
                found[(path, n)] = (sortino, pval, raw, params_from_name(path))

    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] scanned {len(files)} files | "
          f"{interesting_now} interesting now (sortino>={args.sortino_min}, "
          f"pval<={args.pval_max}) | {len(found)} found so far", flush=True)

    report_subdir_timing(args, timing)

    # dump everything found so far, most interesting first (highest sortino,
    # then lowest p-value)
    ranked = sorted(found.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
    for rank, ((_path, n), (sortino, pval, raw, params)) in enumerate(ranked, 1):
        raw_str = "     n/a" if raw is None else f"{raw:+.4f}"
        print(f"    {rank:>3}. model {n}: sortino{n}={sortino:.4f}  "
              f"bestM{n}pval={pval:.4f}  rawReturn{n}={raw_str}  |  {params}",
              flush=True)


def main() -> None:
    args = parse_args()
    print(f"=== study monitor: dir={args.dir!r} glob={args.glob!r} "
          f"sortino>={args.sortino_min} pval<={args.pval_max} "
          f"every {args.interval}s (Ctrl-C to stop) ===", flush=True)
    seen: dict = {}
    timing: dict = {"known": set(), "last": None}
    try:
        while True:
            scan_once(args, seen, timing)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n=== monitor stopped ===", flush=True)


if __name__ == "__main__":
    main()
