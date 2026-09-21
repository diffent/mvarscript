#!/usr/bin/env python3
"""Drive param-study.py across combinations of stock symbols.

Starts from SYMBOL_POOL and selects SELECT_COUNT symbols at a time.  The first
symbol is the forecast target in run-defaults.sh, so its position matters (e.g.
"NVDA AAPL" and "AAPL NVDA" are two distinct runs), but the remaining symbols
are predictors whose order is irrelevant, so selections that differ only in that
CDR ordering (e.g. "SPY GLD USO" vs "SPY USO GLD") are collapsed to one run.
For each selection it runs the full param-study.py optimizer with the SYMBOLS
environment variable set to that selection.

run-defaults.sh reads SYMBOLS from the environment (falling back to its built-in
default), and param-study.py tags each run's output subdirectory, its status
copy and its results table with the symbols -- and drops a symbols.txt into each
run dir -- so outputs from different selections stay separable.

Change SELECT_COUNT to take three (or more) symbols at a time; everything else
generalizes automatically.

Two knobs control which selections are run:
  RANDOM_SELECTIONS -- if True, draw MAX_SELECTIONS unique selections at random
    from the pool; if False, enumerate them in order (deduped by CDR-as-set).
  MAX_SELECTIONS    -- cap on how many selections to run (0 = no limit; ordered
    mode then enumerates everything).  The full ordered space can be huge for a
    large pool, so random mode + a cap keeps runs bounded.
"""

import itertools
import os
import random
import subprocess
import sys
from pathlib import Path

# run relative to this script so it finds param-study.py / run-defaults.sh
SCRIPT_DIR = Path(__file__).resolve().parent
PARAM_STUDY = SCRIPT_DIR / "param-study.py"

# starting universe of symbols to study
SYMBOL_POOL = [
    "NVDA", "AAPL", "GOOG", "MSFT", "AMZN",
    "TSM", "SPCX", "AVGO", "META", "TSLA",
]

# small test
SYMBOL_POOL = [
    "NVDA", "AAPL", "GOOG", "MSFT", "AMZN"
]

# small test
SYMBOL_POOL = [
    "MSFT", "NVDA", "AAPL"
]

# small test
#SYMBOL_POOL = [
#    "GLD", "SPY", "SLV", "USO"
#]

# Override with (approximately) the top 50 US stocks by market cap.  This is a
# best-effort snapshot as of early 2026 -- membership and ordering drift over
# time, so refresh as needed.  Being last, this assignment wins over the small
# test pools above (which are left in place for easy fallback).
SYMBOL_POOL = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN",
    "META", "AVGO", "TSLA", "JPM", "LLY",
    "WMT", "V", "ORCL", "MA", "XOM",
    "COST", "JNJ", "HD", "PG", "NFLX",
    "BAC", "ABBV", "CRM", "CVX", "KO",
    "AMD", "TMUS", "WFC", "PM", "CSCO",
    "IBM", "GE", "UNH", "LIN", "MCD",
    "ABT", "AXP", "MRK", "PEP", "INTU",
    "NOW", "DIS", "GS", "QCOM", "TXN",
    "CAT", "ISRG", "BKNG", "T", "ADBE",
]

# how many symbols to use per param-study run.  Ordered permutations are
# generated, so this is 2 for pairs, 3 for triples, etc.
SELECT_COUNT = 2

# Selection mode.  If True, draw MAX_SELECTIONS unique selections at random from
# the pool; if False, enumerate them in order (deduped by CDR-as-set).
RANDOM_SELECTIONS = True

# Cap on how many selections to run.  In random mode this many unique selections
# are drawn; in ordered mode the first this-many are taken.  Set to 0 for no
# limit (ordered mode then enumerates the entire deduped space).
MAX_SELECTIONS = 10

# Seed for the random selection RNG so a given pool + settings reproduces the
# same draws across runs.  Set to None for a fresh (non-reproducible) draw.
RANDOM_SEED = 42

# Explicit selections mode.  If this list is non-empty it takes precedence over
# the random/ordered generation above: each inner list is run verbatim (first
# symbol = forecast target), in the order given.  Entries may have any length
# and need not match SELECT_COUNT.  Leave empty ([]) to use RANDOM_SELECTIONS.
# Ordered best-first by suspected lead-lag strength (target first, leader second).
# Re-running only QCOM-AAPL (its earlier study was interrupted at ~19/30 trials).
# Full 10-pair list preserved below -- restore it to run the whole study again.
EXPLICIT_SELECTIONS = [
    [],
]
# EXPLICIT_SELECTIONS = [
#     ["COST", "WMT"],
#     ["AMD", "NVDA"],
#     ["AVGO", "NVDA"],
#     ["BAC", "JPM"],
#     ["T", "TMUS"],
#     ["QCOM", "AAPL"],
#     ["CRM", "MSFT"],
#     ["DIS", "NFLX"],
#     ["CVX", "XOM"],
#     ["MRK", "LLY"]
# ]


def selections(pool: list[str], k: int) -> list[tuple[str, ...]]:
    """Selections of k distinct symbols from pool.

    The first symbol is the forecast target so its position matters, but the
    remaining symbols (the CDR) are predictors whose order is irrelevant to
    param-study.py.  We generate ordered permutations and then weed out the
    ones that only differ in CDR ordering, keying each on (head, CDR-as-set)
    so e.g. (SPY, GLD, USO) and (SPY, USO, GLD) collapse to a single run.
    """
    out = []
    seen = set()
    for combo in itertools.permutations(pool, k):
        key = (combo[0], frozenset(combo[1:]))
        if key in seen:
            continue
        seen.add(key)
        out.append(combo)
    return out


def random_selections(pool: list[str], k: int, count: int) -> list[tuple[str, ...]]:
    """Draw up to `count` unique selections of k symbols at random from pool.

    Uses the same (head, CDR-as-set) dedup key as selections(), so two draws
    that differ only in predictor ordering count as one.  If `count` exceeds the
    number of unique selections available, we stop once we can no longer find a
    new one (bounded by an attempt cap so we never spin forever).
    """
    out = []
    seen = set()
    max_attempts = count * 100 + 1000
    attempts = 0
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        combo = tuple(random.sample(pool, k))
        key = (combo[0], frozenset(combo[1:]))
        if key in seen:
            continue
        seen.add(key)
        out.append(combo)
    return out


def main() -> None:
    if EXPLICIT_SELECTIONS:
        # run the given lists verbatim, in order (first symbol = forecast target)
        combos = [tuple(sel) for sel in EXPLICIT_SELECTIONS]
        mode = "explicit"
    elif RANDOM_SELECTIONS:
        # seed before drawing so the selection set is reproducible (RANDOM_SEED
        # = None leaves the RNG unseeded for a fresh draw each run)
        random.seed(RANDOM_SEED)
        combos = random_selections(SYMBOL_POOL, SELECT_COUNT, MAX_SELECTIONS)
        mode = "random"
    else:
        combos = selections(SYMBOL_POOL, SELECT_COUNT)
        if MAX_SELECTIONS > 0:
            combos = combos[:MAX_SELECTIONS]
        mode = "ordered"
    print(f"=== symbol study ({mode}): {len(combos)} selections ===")

    failures = 0
    for i, combo in enumerate(combos, 1):
        symbols = " ".join(combo)
        print("\n" + "=" * 72)
        print(f"### [{i}/{len(combos)}] SYMBOLS = {symbols}")
        print("=" * 72)

        # SYMBOLS flows through param-study.py into run-defaults.sh, and also
        # drives the output-name tagging in param-study.py.
        env = os.environ | {"SYMBOLS": symbols}
        proc = subprocess.run([sys.executable, str(PARAM_STUDY)],
                              cwd=SCRIPT_DIR, env=env)
        if proc.returncode != 0:
            failures += 1
            print(f"warning: param-study.py exited {proc.returncode} for "
                  f"SYMBOLS={symbols}", file=sys.stderr)

    print(f"\n=== symbol study complete: {len(combos)} selections, "
          f"{failures} failed ===")


if __name__ == "__main__":
    main()
