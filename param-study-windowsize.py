#!/usr/bin/env python3
"""Parameter study over the 'windowsize' and 'neighbors' solver parameters.

Sweeps a 2-D grid: every windowsize value is paired with every neighbors value.
Each (windowsize, neighbors) combination runs in its own subdirectory named for
both values:

    windowsize=<ws>,neighbors=<nb>/

run-defaults.sh is pointed at that subdir (via the OUTDIR env var) so ALL of the
run's output -- status, running, the downloaded *.csv data, mergedraw.csv, the
*.pdf plots, etc. -- is kept together, isolated per run.  A convenience copy of
each run's final status JSON is also left at the top level as

    status.windowsize=<ws>,neighbors=<nb>

for quick side-by-side comparison.

The parameters are passed to run-defaults.sh via the WINDOWSIZE and NEIGHBORS
environment variables (run-defaults.sh falls back to its built-in defaults of
200 / 20 when they are unset, so its normal standalone behavior is unchanged).

This is a Python port of param-study-windowsize.sh; both are kept.
"""

import itertools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# run relative to this script so subdirs / run-defaults.sh land in the right place
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = SCRIPT_DIR / "run-defaults.sh"


def centered_grid(center: int, step: int, n: int) -> list[int]:
    """n integer values centered on `center`, spaced `step` apart.

    e.g. centered_grid(200, 20, 5) -> [160, 180, 200, 220, 240]
    """
    half = (n - 1) // 2
    return [center + i * step for i in range(-half, half + 1)]


# --- study grids --------------------------------------------------------------
# windowsize: centered on run-defaults.sh's default (200), step 20, 5 runs.
WINDOWSIZES = centered_grid(100, 20, 3)

# neighbors: centered on run-defaults.sh's default (20), step 5, 3 runs.
# NOTE: this is a full cross product -- total runs = len(WINDOWSIZES)*len(NEIGHBORS).
# Grow the neighbors grid carefully; each run re-pulls data from the APIs.
NEIGHBORS = centered_grid(10, 5, 3)


@dataclass
class RunResult:
    """One grid point's parameters and the three Sharpe ratios it produced.

    A sharpe field is a float on success, or the string "ERROR" if the run's
    status file was missing/unreadable.
    """

    windowsize: int
    neighbors: int
    sharpe1: float | str
    sharpe2: float | str
    sharpe3: float | str


def read_sharpes(status_path: Path) -> tuple[float | str, float | str, float | str]:
    """Extract sharpe1/2/3 from a run's status JSON with a real JSON parser.

    Returns ("ERROR", "ERROR", "ERROR") if the file is missing or malformed.
    """
    try:
        data = json.loads(status_path.read_text())
        return data["sharpe1"], data["sharpe2"], data["sharpe3"]
    except (OSError, ValueError, KeyError):
        return "ERROR", "ERROR", "ERROR"


def run_one(windowsize: int, neighbors: int) -> RunResult:
    """Run a single (windowsize, neighbors) point in its own clean subdir."""
    tag = f"windowsize={windowsize},neighbors={neighbors}"
    print("\n" + "#" * 64)
    print(f"### {tag}  ->  subdir {tag}/")
    print("#" * 64)

    # isolated output subdirectory, cleared so each run starts from scratch
    rundir = SCRIPT_DIR / tag
    shutil.rmtree(rundir, ignore_errors=True)
    rundir.mkdir(parents=True)

    # the .py's monitor thread watches for this kill-switch file in its run dir;
    # create it up front (run-defaults.sh also touches it there)
    (rundir / "running").touch()

    # OUTDIR routes every output file into rundir; WINDOWSIZE/NEIGHBORS set params
    env = os.environ | {
        "OUTDIR": str(rundir),
        "WINDOWSIZE": str(windowsize),
        "NEIGHBORS": str(neighbors),
    }
    proc = subprocess.run([str(RUN_SCRIPT)], cwd=SCRIPT_DIR, env=env)
    if proc.returncode != 0:
        print(f"warning: run-defaults.sh exited {proc.returncode} for {tag}",
              file=sys.stderr)

    # leave a top-level convenience copy of this run's status, tagged with both
    # parameter values (the full output set stays in the subdir)
    status = rundir / "status"
    if status.is_file():
        shutil.copy(status, SCRIPT_DIR / f"status.{tag}")
        print(f"=== saved {tag}/ (status copied to status.{tag}) ===")
    else:
        print(f"warning: no 'status' file produced in {rundir} for {tag}",
              file=sys.stderr)

    s1, s2, s3 = read_sharpes(status)
    print(f"=== {tag}  sharpe1={s1} sharpe2={s2} sharpe3={s3} ===")
    return RunResult(windowsize, neighbors, s1, s2, s3)


def format_table(results: list[RunResult], timestamp: str) -> str:
    """Render the results as a fixed-width text table (trailing newline)."""
    row = "{:<12}  {:<10}  {:<22}  {:<22}  {}".format
    lines = [
        f"# windowsize/neighbors parameter study results  ({timestamp})",
        row("windowsize", "neighbors", "sharpe1", "sharpe2", "sharpe3"),
        row("----------", "---------", "-------", "-------", "-------"),
    ]
    lines += [
        row(r.windowsize, r.neighbors, str(r.sharpe1), str(r.sharpe2), str(r.sharpe3))
        for r in results
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    print("=== parameter study ===")
    print("windowsize values:", WINDOWSIZES)
    print("neighbors values:", NEIGHBORS)

    results = [run_one(ws, nb) for ws, nb in itertools.product(WINDOWSIZES, NEIGHBORS)]

    print("\n=== parameter study complete ===")
    print("per-run output subdirectories:")
    for d in sorted(SCRIPT_DIR.glob("windowsize=*,neighbors=*")):
        if d.is_dir():
            print(d.name)
    print("top-level status copies:")
    for f in sorted(SCRIPT_DIR.glob("status.windowsize=*,neighbors=*")):
        print(f.name)

    # write the summary table to a date/time-stamped file, then dump it to the
    # console so each study run leaves its own results table on disk.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    table_file = SCRIPT_DIR / f"results.{timestamp}.txt"
    table = format_table(results, timestamp)
    table_file.write_text(table)

    print("\n=== results: windowsize, neighbors vs sharpe1/sharpe2/sharpe3 ===")
    print(table, end="")
    print(f"=== results table written to {table_file} ===")


if __name__ == "__main__":
    main()
