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
"""

import itertools
import os
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
#SYMBOL_POOL = [
#    "GLD", "SPY", "SLV", "USO"
#]

# how many symbols to use per param-study run.  Ordered permutations are
# generated, so this is 2 for pairs, 3 for triples, etc.
SELECT_COUNT = 3


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


def main() -> None:
    combos = selections(SYMBOL_POOL, SELECT_COUNT)
    print(f"=== symbol study: {len(combos)} selections of "
          f"{SELECT_COUNT} from {len(SYMBOL_POOL)} symbols ===")

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
