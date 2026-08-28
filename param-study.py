#!/usr/bin/env python3
"""Parameter study over the windowsize, neighbors and knnvarcutoff solver params.

Each (windowsize, neighbors, knnvarcutoff) combination runs in its own
subdirectory named for all three values:

    windowsize=<ws>,neighbors=<nb>,knnvarcutoff=<kv>/

run-defaults.sh is pointed at that subdir (via the OUTDIR env var) so ALL of the
run's output -- status, running, the downloaded *.csv data, mergedraw.csv, the
*.pdf plots, etc. -- is kept together, isolated per run.  A convenience copy of
each run's final status JSON is also left at the top level as

    status.windowsize=<ws>,neighbors=<nb>,knnvarcutoff=<kv>

for quick side-by-side comparison.

The parameters are passed to run-defaults.sh via the WINDOWSIZE, NEIGHBORS and
KNNVARCUTOFF environment variables (run-defaults.sh falls back to its built-in
defaults when they are unset, so its normal standalone behavior is unchanged).

Two run methods are selectable via the RUN_METHOD variable:
  * "grid"     -- exhaustive grid over WINDOWSIZES x NEIGHBORS x KNNVARCUTOFFS.
  * "optimize" -- Optuna (TPE Bayesian) search that maximizes sharpe3, which is
                  far more sample-efficient for this expensive, noisy black box.
Both methods reuse the same per-run plumbing (isolated subdir, status copy,
Sharpe extraction) and emit the same date/time-stamped results table.

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

# Symbols for this study, passed straight through to run-defaults.sh via the
# SYMBOLS env var (set by symbol-study.py).  When set, every per-run output
# subdirectory, status copy and results table is tagged with the symbols so
# outputs from different symbol selections stay separable; a symbols.txt is also
# dropped into each run dir.  Empty (unset) keeps the original untagged names.
SYMBOLS = os.environ.get("SYMBOLS", "").strip()
SYMBOL_TAG = "symbols=" + "-".join(SYMBOLS.split()) if SYMBOLS else ""


def _tagged(name: str) -> str:
    """Prefix a run tag / filename fragment with SYMBOL_TAG when symbols are set."""
    return f"{SYMBOL_TAG},{name}" if SYMBOL_TAG else name


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
# NOTE: this is a full cross product -- total runs is the product of all three
# grid sizes below.  Grow them carefully; each run re-pulls data from the APIs.
NEIGHBORS = centered_grid(10, 5, 3)

# knnvarcutoff: integer >= 0 (run-defaults.sh's default is 400).
KNNVARCUTOFFS = centered_grid(400, 100, 3)


# --- run method ---------------------------------------------------------------
# "grid":    exhaustive grid search over WINDOWSIZES x NEIGHBORS (the original).
# "optimize": Optuna (Bayesian/TPE) search that maximizes sharpe3 over
#            (windowsize, neighbors).  Far more sample-efficient than a grid for
#            an expensive, noisy black box on a small evaluation budget.
RUN_METHOD = "optimize"

# --- optimizer settings (only used when RUN_METHOD == "optimize") -------------
# Each run-defaults.sh evaluation is expensive (re-pulls data), so the search is
# capped at OPT_MAX_RUNS trials.  Optuna's TPE sampler proposes each next point
# from a model of the runs seen so far, so it spends the budget far better than a
# grid or a finite-difference gradient method would.
OPT_TARGET = "sharpe3"                # top-level status-JSON key to MAXIMIZE;
                                      # any numeric key works (e.g. sortino2)
OPT_MAX_RUNS = 10                      # number of Optuna trials (== run-defaults runs)
OPT_WINDOWSIZE_RANGE = (100, 200)     # (min, max) inclusive search range
OPT_NEIGHBORS_RANGE = (5, 20)         # (min, max) inclusive search range
OPT_KNNVARCUTOFF_RANGE = (200, 400)   # (min, max) inclusive search range; integer >= 0
OPT_WINDOWSIZE_STEP = 10              # search windowsize on this integer grid step (must be >= 1)
OPT_NEIGHBORS_STEP = 3                # search neighbors on this integer grid step (must be >= 1)
OPT_KNNVARCUTOFF_STEP = 10            # search knnvarcutoff on this integer grid step (must be >= 1)
OPT_SEED = 42                         # RNG seed for reproducible trial suggestions
OPT_FAIL_PENALTY = -1e6               # sharpe3 assigned to a failed/ERROR run
OPT_BEST_FILE = "current_best.txt"     # live "best so far" file, refreshed each trial


# --- results table columns ----------------------------------------------------
# Top-level status-JSON keys to show (in order) as metric columns in the results
# table.  Any numeric key several.py writes works; unknown keys show "ERROR".
# The OPT_TARGET objective is always included (appended if not already listed).
TABLE_KEYS = [
    "sharpe1", "sharpe2", "sharpe3",
    "sortino1", "sortino2", "sortino3",
    "sortino1p", "sortino2p", "sortino3p",
    "bestM1pval", "bestM2pval", "bestM3pval",
]


@dataclass
class RunResult:
    """One evaluated point: its parameters, reported metrics, and the objective.

    `metrics` maps each requested status key (TABLE_KEYS + OPT_TARGET) to its
    value -- a float on success, or "ERROR" if that key was missing/unreadable.
    """

    windowsize: int
    neighbors: int
    knnvarcutoff: int
    metrics: dict[str, float | str]   # status key -> value
    target: float | str              # value of OPT_TARGET (the optimize objective)


def read_status_values(status_path: Path, keys: list[str]) -> dict[str, float | str]:
    """Read the given top-level keys from a run's status JSON, as floats.

    Returns {key: float} per key, or {key: "ERROR"} for any key that is absent,
    non-numeric, or when the file is missing/malformed.
    """
    try:
        data = json.loads(status_path.read_text())
    except (OSError, ValueError):
        data = {}
    out: dict[str, float | str] = {}
    for k in keys:
        try:
            out[k] = float(data[k])
        except (KeyError, ValueError, TypeError):
            out[k] = "ERROR"
    return out


def run_one(windowsize: int, neighbors: int, knnvarcutoff: int) -> RunResult:
    """Run a single (windowsize, neighbors, knnvarcutoff) point in a clean subdir."""
    tag = _tagged(f"windowsize={windowsize},neighbors={neighbors},knnvarcutoff={knnvarcutoff}")
    print("\n" + "#" * 64)
    print(f"### {tag}  ->  subdir {tag}/")
    print("#" * 64)

    # isolated output subdirectory, cleared so each run starts from scratch
    rundir = SCRIPT_DIR / tag
    shutil.rmtree(rundir, ignore_errors=True)
    rundir.mkdir(parents=True)

    # record which symbols this run used, for clarity when comparing runs
    if SYMBOLS:
        (rundir / "symbols.txt").write_text(SYMBOLS + "\n")

    # the .py's monitor thread watches for this kill-switch file in its run dir;
    # create it up front (run-defaults.sh also touches it there)
    (rundir / "running").touch()

    # OUTDIR routes every output file into rundir; the *NAME* env vars set params
    env = os.environ | {
        "OUTDIR": str(rundir),
        "WINDOWSIZE": str(windowsize),
        "NEIGHBORS": str(neighbors),
        "KNNVARCUTOFF": str(knnvarcutoff),
    }
    # tee the run's stdout+stderr to a per-run log in its output folder
    log_path = rundir / "run.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen([str(RUN_SCRIPT)], cwd=SCRIPT_DIR, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
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

    # read the display metrics plus the objective (dedup, preserving order)
    keys = list(dict.fromkeys(TABLE_KEYS + [OPT_TARGET]))
    metrics = read_status_values(status, keys)
    target = metrics.get(OPT_TARGET, "ERROR")
    print(f"=== {tag}  target({OPT_TARGET})={target} ===")
    return RunResult(windowsize, neighbors, knnvarcutoff, metrics, target)


def _fmt_metric(value: float | str) -> str:
    """Compact display for a metric: 6 significant digits, or the raw string."""
    return f"{value:.6g}" if isinstance(value, (int, float)) else str(value)


def format_table(results: list[RunResult], timestamp: str) -> str:
    """Render the results as an auto-width text table (trailing newline).

    Columns are the three parameters followed by one column per TABLE_KEYS entry
    (plus OPT_TARGET if not already listed).  The objective column is starred.
    """
    # metric columns: the configured keys, with OPT_TARGET appended if missing
    metric_keys = list(dict.fromkeys(TABLE_KEYS + [OPT_TARGET]))
    headers = ["windowsize", "neighbors", "knnvarcutoff"]
    headers += [k + ("*" if k == OPT_TARGET else "") for k in metric_keys]

    rows = [
        [str(r.windowsize), str(r.neighbors), str(r.knnvarcutoff)]
        + [_fmt_metric(r.metrics.get(k, "ERROR")) for k in metric_keys]
        for r in results
    ]

    # size each column to the widest cell (header or any row)
    widths = [len(h) for h in headers]
    for cells in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, cells)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    out = [
        f"# windowsize/neighbors/knnvarcutoff parameter study results  ({timestamp})",
        f"# objective (*) = {OPT_TARGET}",
        line(headers),
        line(["-" * w for w in widths]),
    ]
    out += [line(cells) for cells in rows]
    return "\n".join(out) + "\n"


def numeric_target(result: RunResult, penalty: float) -> float:
    """The run's objective value as a float; failed/ERROR runs map to `penalty`."""
    return float(result.target) if isinstance(result.target, (int, float)) else penalty


def run_grid() -> list[RunResult]:
    """Exhaustive grid search over WINDOWSIZES x NEIGHBORS x KNNVARCUTOFFS."""
    print("windowsize values:  ", WINDOWSIZES)
    print("neighbors values:   ", NEIGHBORS)
    print("knnvarcutoff values:", KNNVARCUTOFFS)
    return [run_one(ws, nb, kv)
            for ws, nb, kv in itertools.product(WINDOWSIZES, NEIGHBORS, KNNVARCUTOFFS)]


def run_optimize() -> list[RunResult]:
    """Optuna (TPE) search maximizing OPT_TARGET; returns the unique runs performed.

    Each trial proposes a (windowsize, neighbors, knnvarcutoff) point, evaluated
    by a real run-defaults.sh run.  Results are cached so a repeated suggestion
    does not spend the run budget twice.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # quiet per-trial spam
    print(f"optuna TPE search: up to {OPT_MAX_RUNS} trials, maximizing {OPT_TARGET}")
    print(f"  windowsize   in {OPT_WINDOWSIZE_RANGE} step {OPT_WINDOWSIZE_STEP}")
    print(f"  neighbors    in {OPT_NEIGHBORS_RANGE} step {OPT_NEIGHBORS_STEP}")
    print(f"  knnvarcutoff in {OPT_KNNVARCUTOFF_RANGE} step {OPT_KNNVARCUTOFF_STEP}")
    best_file = SCRIPT_DIR / OPT_BEST_FILE
    print(f"  best-so-far written live to {best_file}")

    cache: dict[tuple[int, int, int], RunResult] = {}
    results: list[RunResult] = []

    def objective(trial: "optuna.Trial") -> float:
        ws = trial.suggest_int("windowsize", *OPT_WINDOWSIZE_RANGE,
                               step=OPT_WINDOWSIZE_STEP)
        nb = trial.suggest_int("neighbors", *OPT_NEIGHBORS_RANGE,
                               step=OPT_NEIGHBORS_STEP)
        kv = trial.suggest_int("knnvarcutoff", *OPT_KNNVARCUTOFF_RANGE,
                               step=OPT_KNNVARCUTOFF_STEP)
        key = (ws, nb, kv)
        result = cache.get(key)
        if result is None:
            result = run_one(ws, nb, kv)
            cache[key] = result
            results.append(result)
        else:
            print(f"=== reusing cached run for "
                  f"windowsize={ws},neighbors={nb},knnvarcutoff={kv} ===")
        return numeric_target(result, OPT_FAIL_PENALTY)

    def write_best(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        """Refresh the live best-so-far file after every completed trial.

        Written atomically (temp file + os.replace) so a viewer that reads it
        mid-update never sees a half-written file.
        """
        done = sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE)
        best = study.best_trial
        tag = (f"windowsize={best.params['windowsize']},"
               f"neighbors={best.params['neighbors']},"
               f"knnvarcutoff={best.params['knnvarcutoff']}")
        text = (
            "# current best so far (refreshed after each optuna trial)\n"
            f"updated:        {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"trials done:    {done}/{OPT_MAX_RUNS}\n"
            f"best trial #:   {best.number}\n"
            f"windowsize:     {best.params['windowsize']}\n"
            f"neighbors:      {best.params['neighbors']}\n"
            f"knnvarcutoff:   {best.params['knnvarcutoff']}\n"
            f"objective:      {OPT_TARGET} = {best.value}\n"
            f"run subdir:     {tag}/\n"
        )
        tmp = best_file.with_suffix(".tmp")
        tmp.write_text(text)
        os.replace(tmp, best_file)  # atomic on the same filesystem

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=OPT_SEED),
    )
    study.optimize(objective, n_trials=OPT_MAX_RUNS, callbacks=[write_best])

    best = study.best_params
    print(f"\n=== optuna best: windowsize={best['windowsize']} "
          f"neighbors={best['neighbors']} knnvarcutoff={best['knnvarcutoff']} "
          f"{OPT_TARGET}={study.best_value} "
          f"({len(study.trials)} trials, {len(results)} unique runs) ===")
    print(f"=== best-so-far file: {best_file} ===")
    return results


def main() -> None:
    print("=== parameter study ===")
    print(f"run method: {RUN_METHOD}")

    if RUN_METHOD == "grid":
        results = run_grid()
    elif RUN_METHOD == "optimize":
        results = run_optimize()
    else:
        raise SystemExit(f"unknown RUN_METHOD {RUN_METHOD!r}; use 'grid' or 'optimize'")

    print("\n=== parameter study complete ===")
    # the leading '*' matches the optional 'symbols=...,' prefix when set
    print("per-run output subdirectories:")
    for d in sorted(SCRIPT_DIR.glob("*windowsize=*,neighbors=*,knnvarcutoff=*")):
        if d.is_dir():
            print(d.name)
    print("top-level status copies:")
    for f in sorted(SCRIPT_DIR.glob("status.*windowsize=*,neighbors=*,knnvarcutoff=*")):
        print(f.name)

    # write the summary table to a date/time-stamped file, then dump it to the
    # console so each study run leaves its own results table on disk.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    table_file = SCRIPT_DIR / _tagged(f"results.{timestamp}.txt")
    table = format_table(results, timestamp)
    table_file.write_text(table)

    print(f"\n=== results: params vs sharpe1/sharpe2/sharpe3 "
          f"(objective: target={OPT_TARGET}) ===")
    print(table, end="")
    print(f"=== results table written to {table_file} ===")


if __name__ == "__main__":
    main()
