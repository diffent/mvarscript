#!/usr/bin/env python3
"""Reconstruct an objective time series from an OLD param-study run folder.

param-study.py only started writing objective.<ts>.csv recently, so older study
folders (e.g. old.*/ archives) have no such file.  But every per-run subdir they
DO contain still holds a `status` JSON with the objective, and its parameters are
encoded in the subdir name:

    symbols=AMD-NVDA,target=adjustedSortino1,windowsize=30,neighbors=2,...

and the `status` file's modification time is when that run finished.  Because
param-study.py evaluates trials sequentially, ordering the subdirs by that mtime
reproduces the optimization step order.  This script walks a run folder, reads
each subdir's objective (the target named in the folder/subdir, default
adjustedSortino1), orders them by mtime, and writes the SAME objective.*.csv that
param-study.py now emits -- so plot-convergence.py can plot it unchanged.

Usage:
    ./extract-objective.py <run-folder> [out.csv]
    ./extract-objective.py "old.Sat_Oct_10_10:10:59_EDT_2026.200daybacktest"

Default output is <run-folder>/objective.extracted.csv.  Then:
    ./plot-convergence.py "<run-folder>/objective.extracted.csv"
"""

import importlib.util
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_param_study():
    """Import param-study.py as a module despite its hyphenated filename.

    Reuses its RunResult, write_objective_csv, read_status_values and TABLE_KEYS
    so the extracted CSV is identical in schema to a live study's output.
    """
    spec = importlib.util.spec_from_file_location("param_study",
                                                  SCRIPT_DIR / "param-study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# subdir / status-copy name -> its key=value parameters
_INT_PARAMS = ("windowsize", "neighbors", "knnvarcutoff", "annealmaxiter")


def parse_params(name: str) -> dict:
    """Pull key=value tokens out of a run subdir name (ints where expected)."""
    kv = dict(re.findall(r"([A-Za-z]+)=([^,]+)", name))
    out = {}
    for k in _INT_PARAMS:
        try:
            out[k] = int(kv.get(k, 0))
        except ValueError:
            out[k] = 0
    try:
        out["elasticalpha"] = float(kv.get("elasticalpha", 0.0))
    except ValueError:
        out["elasticalpha"] = 0.0
    out["target"] = kv.get("target", "adjustedSortino1")
    out["symbols"] = kv.get("symbols", "")  # e.g. "AMD-NVDA"
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1]).resolve()
    if not folder.is_dir():
        raise SystemExit(f"not a directory: {folder}")

    ps = _load_param_study()

    # every per-run subdir that actually produced a status file
    subdirs = [d for d in folder.iterdir()
               if d.is_dir() and re.search(r"windowsize=.*,neighbors=.*,"
                                           r"knnvarcutoff=", d.name)
               and (d / "status").is_file()]
    if not subdirs:
        raise SystemExit(f"no per-run subdirs with a status file under {folder}")

    # order by status mtime == trial completion order (runs are sequential)
    subdirs.sort(key=lambda d: (d / "status").stat().st_mtime)

    # target objective: taken from the subdirs (consistent within one study).
    # Point the reused helpers at it so the CSV's objective column / symbols tag
    # match what a live study for this target would have written.
    first = parse_params(subdirs[0].name)
    ps.OPT_TARGET = first["target"]
    ps.SYMBOLS = first["symbols"].replace("-", " ")  # "AMD-NVDA" -> "AMD NVDA"

    metric_keys = list(dict.fromkeys(ps.TABLE_KEYS + [ps.OPT_TARGET]))
    results = []
    for d in subdirs:
        p = parse_params(d.name)
        metrics = ps.read_status_values(d / "status", metric_keys)
        results.append(ps.RunResult(
            windowsize=p["windowsize"], neighbors=p["neighbors"],
            knnvarcutoff=p["knnvarcutoff"], annealmaxiter=p["annealmaxiter"],
            elasticalpha=p["elasticalpha"], metrics=metrics,
            target=metrics.get(ps.OPT_TARGET, "ERROR"), rundir=d.name))

    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "objective.extracted.csv"
    ps.write_objective_csv(results, out_path)
    print(f"wrote {out_path}  ({len(results)} runs, target={ps.OPT_TARGET})")
    print(f"plot it with:  ./plot-convergence.py \"{out_path}\"")


if __name__ == "__main__":
    main()
