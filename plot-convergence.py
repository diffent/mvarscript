#!/usr/bin/env python3
"""Plot the objective time series written by param-study.py to check convergence.

param-study.py writes one `objective.<timestamp>.csv` per study (one row per
hyperparameter combination, in evaluation order, with a running `best_so_far`).
This renders that CSV to a PDF with two curves against the run index:

  * objective    -- the per-run objective value (shows how noisy the landscape
                    is / how much each run scatters)
  * best_so_far  -- the running maximum of the objective; THIS is the
                    convergence curve.  When it flattens -- no improvement over
                    the last many runs -- the search has effectively converged.

A caption reports how many runs have passed since best_so_far last improved,
which is the headline "has it converged?" number.

Usage:
    ./plot-convergence.py                 # newest *objective.*.csv in this dir
    ./plot-convergence.py path/to.csv     # a specific CSV
    ./plot-convergence.py a.csv out.pdf   # explicit output path

Default output is the input path with its extension swapped to .pdf.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to file, no display (matches several.py)
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def newest_csv() -> Path:
    """The most recently modified *objective.*.csv beside this script."""
    cands = sorted(SCRIPT_DIR.glob("*objective.*.csv"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("no *objective.*.csv found; run param-study.py first "
                         "or pass a CSV path")
    return cands[-1]


def load(csv_path: Path):
    """Return (runs, objective, best_so_far) as parallel lists.

    objective entries are floats or None (blank == failed/ERROR run);
    best_so_far entries are floats or None (None only before the first success).
    """
    runs, obj, best = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            runs.append(int(row["run"]))
            obj.append(float(row["objective"]) if row["objective"] else None)
            best.append(float(row["best_so_far"]) if row["best_so_far"] else None)
    if not runs:
        raise SystemExit(f"{csv_path} has no data rows")
    return runs, obj, best


def runs_since_improvement(best: list) -> int:
    """How many runs have passed since best_so_far last increased.

    A large value relative to the total run count is the signal the search has
    stopped finding anything better -- i.e. it has converged.
    """
    last = None
    last_improve_idx = 0
    for i, b in enumerate(best):
        if b is not None and (last is None or b > last):
            last, last_improve_idx = b, i
    return (len(best) - 1) - last_improve_idx


def main() -> None:
    args = sys.argv[1:]
    csv_path = Path(args[0]) if args else newest_csv()
    out_path = Path(args[1]) if len(args) > 1 else csv_path.with_suffix(".pdf")

    runs, obj, best = load(csv_path)

    # objective: scatter only the real (non-failed) runs so gaps are visible
    ox = [r for r, o in zip(runs, obj) if o is not None]
    oy = [o for o in obj if o is not None]
    # best_so_far: step line over the runs where it's defined
    bx = [r for r, b in zip(runs, best) if b is not None]
    by = [b for b in best if b is not None]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ox, oy, s=16, alpha=0.5, color="tab:gray", label="objective (per run)")
    ax.step(bx, by, where="post", color="tab:red", linewidth=2,
            label="best so far")
    if by:
        ax.axhline(by[-1], color="tab:red", linestyle=":", linewidth=1, alpha=0.6)

    stall = runs_since_improvement(best)
    final = by[-1] if by else float("nan")
    ax.set_xlabel("run index (evaluation order)")
    ax.set_ylabel("objective")
    ax.set_title(f"optimization convergence  ({csv_path.name})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.text(0.01, 0.01,
             f"{len(runs)} runs | best = {final:.6g} | "
             f"{stall} runs since last improvement"
             + ("  -> looks converged" if stall >= max(10, len(runs) // 4)
                else "  -> still improving"),
             fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path)
    print(f"wrote {out_path}  ({len(runs)} runs, best={final:.6g}, "
          f"{stall} runs since last improvement)")


if __name__ == "__main__":
    main()
