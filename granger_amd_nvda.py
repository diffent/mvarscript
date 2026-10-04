#!/usr/bin/env python3
"""Granger causality test: does NVDA Granger-cause AMD?  (AMD = f(NVDA))

Side-quest utility, standalone -- does not touch several.py or any run output.

It reads the latest saved per-symbol CSVs (AMD.csv / NVDA.csv, as written by the
solver pipeline), aligns them on Date, optionally restricts to a trailing window
of N points counted back from the most recent day, makes the series stationary
(daily first-difference of Close by default), and runs statsmodels' Granger
causality test for whether NVDA's past helps predict AMD beyond AMD's own past.

By default both assets use delta-close (close-to-close difference).  With
--target-closeopen the caused (target) asset is instead modelled as same-day
Close-Open (the intraday move), while the causing asset still uses delta-close.

The CSVs are newest-first with header: Date,Open,High,Low,Close,Volume

Examples:
  python3 granger_amd_nvda.py                  # auto-find latest, last 250 pts, lags 1..5
  python3 granger_amd_nvda.py --target-closeopen  # AMD modelled as Close-Open
  python3 granger_amd_nvda.py --window 120      # trailing 120 trading days
  python3 granger_amd_nvda.py --window 0        # use all available aligned points
  python3 granger_amd_nvda.py --maxlag 10 --both
  python3 granger_amd_nvda.py --amd path/to/AMD.csv --nvda path/to/NVDA.csv
"""

import argparse
import contextlib
import io
import os
import sys
import warnings

import numpy
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def find_latest(filename, search_root="."):
    """Return the most-recently-modified `filename` under search_root, skipping
    the archive directories (any path component beginning with 'old.')."""
    best = None
    best_mtime = -1.0
    for dirpath, dirnames, filenames in os.walk(search_root):
        # prune archive dirs in-place so we don't descend into them
        dirnames[:] = [d for d in dirnames if not d.startswith("old.")]
        if filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                m = os.path.getmtime(path)
            except OSError:
                continue
            if m > best_mtime:
                best_mtime, best = m, path
    return best


def load_ohlc(path):
    """Load one symbol CSV into a Date-indexed OHLCV DataFrame, sorted
    chronologically (oldest -> newest)."""
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        sys.exit("error: %s missing required Date column" % path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")


def make_stationary(series, mode):
    """Transform a price series for stationarity."""
    if mode == "logret":
        return numpy.log(series).diff()
    if mode == "pct":
        return series.pct_change()
    if mode == "diff":
        return series.diff()
    if mode == "level":
        return series
    sys.exit("error: unknown transform mode %r" % mode)


def adf_report(name, x):
    """Augmented Dickey-Fuller stationarity check (low p-value => stationary)."""
    x = numpy.asarray(x, dtype=float)
    x = x[~numpy.isnan(x)]
    stat, pval = adfuller(x, autolag="AIC")[:2]
    verdict = "stationary" if pval < 0.05 else "NON-stationary"
    print("  ADF %-5s: stat=%8.4f  p=%.4g  -> %s" % (name, stat, pval, verdict))


def run_granger(caused, causing, caused_name, causing_name, maxlag):
    """Test H0: `causing` does NOT Granger-cause `caused`.

    statsmodels tests whether the *second* column Granger-causes the *first*,
    so we pass [caused, causing]."""
    data = numpy.column_stack([caused, causing])
    print("\n=== Granger causality: does %s cause %s ? "
          "(H0: it does NOT) ===" % (causing_name, caused_name))
    print("  lag   F-stat     p-value   signif(5%)")
    # statsmodels still prints its own verbose per-lag tables and warns about
    # the deprecated `verbose` kwarg; mute the warning and swallow its stdout so
    # only our clean summary shows.
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", FutureWarning)
        res = grangercausalitytests(data, maxlag=maxlag)
    any_sig = False
    for lag in range(1, maxlag + 1):
        fstat, pval = res[lag][0]["ssr_ftest"][:2]
        sig = pval < 0.05
        any_sig = any_sig or sig
        print("  %3d  %9.4f  %9.4g   %s" % (lag, fstat, pval, "*" if sig else ""))
    print("  => %s Granger-causes %s at >=1 lag (5%%): %s"
          % (causing_name, caused_name, "YES" if any_sig else "no"))
    return any_sig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amd", help="path to AMD CSV (default: auto-find latest)")
    ap.add_argument("--nvda", help="path to NVDA CSV (default: auto-find latest)")
    ap.add_argument("--col", default="Close",
                    help="price column to use (default: Close)")
    ap.add_argument("--window", type=int, default=250,
                    help="number of trailing points counted back from the most "
                         "recent day; 0 = use all aligned points (default: 250)")
    ap.add_argument("--maxlag", type=int, default=5,
                    help="max lag (trading days) to test, 1..maxlag (default: 5)")
    ap.add_argument("--transform", default="diff",
                    choices=["logret", "pct", "diff", "level"],
                    help="stationarity transform of the price column "
                         "(default: diff = daily first difference of Close)")
    ap.add_argument("--both", action="store_true",
                    help="also test the reverse direction (AMD -> NVDA)")
    ap.add_argument("--target-closeopen", action="store_true",
                    help="model the CAUSED (target) asset as same-day Close-Open "
                         "instead of its delta-close; the CAUSING asset still uses "
                         "delta-close (--transform).  Applies to whichever asset is "
                         "the caused one in each tested direction.")
    args = ap.parse_args()

    amd_path = args.amd or find_latest("AMD.csv")
    nvda_path = args.nvda or find_latest("NVDA.csv")
    if not amd_path or not os.path.exists(amd_path):
        sys.exit("error: could not locate an AMD.csv (use --amd)")
    if not nvda_path or not os.path.exists(nvda_path):
        sys.exit("error: could not locate an NVDA.csv (use --nvda)")

    print("AMD  file: %s" % amd_path)
    print("NVDA file: %s" % nvda_path)

    raw = {"AMD": load_ohlc(amd_path), "NVDA": load_ohlc(nvda_path)}
    for sym, d in raw.items():
        if args.col not in d.columns:
            sys.exit("error: %s CSV missing %r column" % (sym, args.col))
        if args.target_closeopen and ("Open" not in d.columns or "Close" not in d.columns):
            sys.exit("error: %s CSV needs Open and Close for --target-closeopen" % sym)

    # Build the two representations each asset can play:
    #   causing role -> delta-close (the chosen --transform, default diff of Close)
    #   caused  role -> same-day Close-Open if --target-closeopen, else delta-close
    def causing_series(sym):
        return make_stationary(raw[sym][args.col], args.transform)

    def caused_series(sym):
        if args.target_closeopen:
            return raw[sym]["Close"] - raw[sym]["Open"]   # intraday move, already a delta
        return make_stationary(raw[sym][args.col], args.transform)

    caused_label = "Close-Open" if args.target_closeopen else ("delta-close(%s)" % args.transform)
    causing_label = "delta-close(%s)" % args.transform

    # Align all needed series on common dates; dropna removes the leading NaN the
    # delta-close transform introduces, so every column shares the same sample.
    df = pd.DataFrame({
        "AMD_causing": causing_series("AMD"),
        "AMD_caused": caused_series("AMD"),
        "NVDA_causing": causing_series("NVDA"),
        "NVDA_caused": caused_series("NVDA"),
    }).dropna()
    if df.empty:
        sys.exit("error: no overlapping dates between the two files")

    # trailing window counted back from the most recent day
    if args.window and args.window > 0:
        df = df.iloc[-args.window:]

    n = len(df)
    print("\ncaused=%s  causing=%s  points used=%d  (%s -> %s)  maxlag=%d"
          % (caused_label, causing_label, n,
             df.index[0].date(), df.index[-1].date(), args.maxlag))

    need = (args.maxlag + 1) * 3
    if n < need:
        sys.exit("error: only %d points after transform/window; need >= ~%d for "
                 "maxlag=%d (increase --window or lower --maxlag)"
                 % (n, need, args.maxlag))

    print("\nstationarity (ADF on the series actually used):")
    adf_report("AMD caused[%s]" % caused_label, df["AMD_caused"])
    adf_report("NVDA causing[%s]" % causing_label, df["NVDA_causing"])
    if args.both:
        adf_report("NVDA caused[%s]" % caused_label, df["NVDA_caused"])
        adf_report("AMD causing[%s]" % causing_label, df["AMD_causing"])

    # primary question: AMD = f(NVDA)  <=>  does NVDA(delta-close) Granger-cause AMD(caused)?
    run_granger(df["AMD_caused"].values, df["NVDA_causing"].values, "AMD", "NVDA", args.maxlag)

    if args.both:
        run_granger(df["NVDA_caused"].values, df["AMD_causing"].values, "NVDA", "AMD", args.maxlag)

    print("\nnote: Granger causality is predictive precedence, not true causation.")


if __name__ == "__main__":
    main()
