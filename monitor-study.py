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
    python3 monitor-study.py                     # defaults, scan the current directory
    python3 monitor-study.py --dir /path/to/runs
    python3 monitor-study.py --sortino-min 0.4 --pval-max 0.05 --interval 2
    python3 monitor-study.py --glob 'status.symbols=*'
    python3 monitor-study.py --open-pdfs                 # pop each run's equity-curve
                                                          # PDF (pval<0.01) in Preview
    python3 monitor-study.py --open-pdfs --open-max 5 --open-pval-max 0.005
"""

import argparse
import glob
import json
import os
import subprocess
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
OUTFILE = "interesting.txt"   # ranked hits are (over)written here each cycle
MODELS = (1, 2, 3)
# per-run equity-curve plot that lives inside each run's folder (the folder name
# is the run's param tag, i.e. the rightmost column of this monitor's output)
PDF_NAME = "gainsOverTimeCumulative.pdf"
OPEN_PVAL_MAX = 0.01          # only auto-open PDFs for runs with a hit below this


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live monitor for a running study.")
    p.add_argument("--dir", default=os.getcwd(),
                   help="directory holding the status files (default: the current "
                        "working directory, so running it from inside a run/archive "
                        "dir scans that dir)")
    p.add_argument("--glob", default=STATUS_GLOB,
                   help=f"glob for status files (default: {STATUS_GLOB!r})")
    p.add_argument("--sortino-min", type=float, default=SORTINO_MIN,
                   help=f"minimum sortino to flag (default: {SORTINO_MIN})")
    p.add_argument("--pval-max", type=float, default=PVAL_MAX,
                   help=f"maximum p-value to flag (default: {PVAL_MAX})")
    p.add_argument("--interval", type=float, default=POLL_SECONDS,
                   help=f"seconds between scans (default: {POLL_SECONDS})")
    p.add_argument("--outfile", default=OUTFILE,
                   help=f"file (in --dir) overwritten each cycle with the current "
                        f"ranked hits (default: {OUTFILE!r})")
    p.add_argument("--open-pdfs", action="store_true",
                   help=f"macOS only: open each qualifying run's {PDF_NAME} in "
                        f"Preview.  A PDF already open in Preview is not re-opened.")
    p.add_argument("--open-pval-max", type=float, default=OPEN_PVAL_MAX,
                   help=f"only open PDFs for runs with a hit p-value below this "
                        f"(default: {OPEN_PVAL_MAX})")
    p.add_argument("--open-max", type=int, default=0, metavar="N",
                   help="cap the number of distinct PDFs opened, best-first "
                        "(0 = no cap, default)")
    p.add_argument("--tabbed", action=argparse.BooleanOptionalAction, default=True,
                   help="after opening, collapse Preview's windows into tabs of a "
                        "single window (Window > Merge All Windows; needs a one-time "
                        "Accessibility grant).  Use --no-tabbed to keep separate "
                        "windows. (default: tabbed)")
    return p.parse_args()


def params_from_name(path: str) -> str:
    """The run parameters are the filename with the leading 'status.' stripped."""
    base = os.path.basename(path)
    return base[len("status."):] if base.startswith("status.") else base


def report_subdir_timing(args: argparse.Namespace, timing: dict) -> None:
    """Report the wall-clock time each run subdir took.

    A new subdir means the next run just started, so the gap since the previous
    new subdir is the duration of the run that just finished.  We print that time
    for the subdir that just completed, plus the running average of all such gaps
    seen so far.  `timing` persists across scans: 'known' (subdirs seen), 'last'
    (name, time of the most recent new subdir) and 'deltas' (measured gaps).
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
            delta = now - prev_time
            timing["deltas"].append(delta)
            avg = sum(timing["deltas"]) / len(timing["deltas"])
            print(f"    subdir done in {fmt_duration(delta)} "
                  f"(avg {fmt_duration(avg)} over {len(timing['deltas'])}): "
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


def preview_open_paths() -> set:
    """Set of real file paths currently open in macOS Preview.

    Returns an empty set when not on macOS, when Preview is not already running
    (we deliberately do NOT launch it just to ask), or when the query fails --
    in those cases we fall back to the per-session dedup only.
    """
    if sys.platform != "darwin":
        return set()
    try:
        # only query if Preview is already up, so we never spawn it ourselves
        if subprocess.run(["pgrep", "-x", "Preview"],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0:
            return set()
        # Emit one POSIX path per line.  We must NOT let osascript return the
        # list comma-separated: our run folders contain commas, so a comma split
        # would shred the paths.  A linefeed delimiter cannot occur in a path.
        script = ('set out to ""\n'
                  'tell application "Preview"\n'
                  '  repeat with d in documents\n'
                  '    set out to out & (path of d) & linefeed\n'
                  '  end repeat\n'
                  'end tell\n'
                  'return out')
        out = subprocess.run(["osascript", "-e", script],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if out.returncode != 0:
        return set()
    paths = (line.strip() for line in out.stdout.splitlines() if line.strip())
    return {os.path.realpath(p) for p in paths}


def merge_preview_windows_into_tabs() -> None:
    """Best-effort: collapse all Preview windows into tabs of one window via the
    Window > Merge All Windows menu item (GUI scripting).

    Requires Accessibility permission for whatever runs osascript (grant once in
    System Settings > Privacy & Security > Accessibility).  If that is not
    granted, or there is only one window, this silently no-ops and the PDFs stay
    as separate windows (or tabs, per the system 'Prefer tabs' setting).
    """
    if sys.platform != "darwin":
        return
    script = ('tell application "Preview" to activate\n'
              'tell application "System Events" to tell process "Preview"\n'
              '  try\n'
              '    click menu item "Merge All Windows" of menu "Window" of menu bar 1\n'
              '  end try\n'
              'end tell')
    try:
        subprocess.run(["osascript", "-e", script],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def open_best_pdfs(args: argparse.Namespace, ranked: list, opened: set) -> None:
    """Open the equity-curve PDF for the best qualifying runs in Preview.

    `ranked` is the best-first list of hits; each hit's run folder is its params
    string (the rightmost column of this monitor's output / the param-study
    subdir name).  A run is opened only if it has a hit with p-value below
    --open-pval-max.  Each PDF is passed as a two-component relative path
    (``<run-folder>/gainsOverTimeCumulative.pdf``) run from --dir, so Preview's
    title carries the folder, letting you tell which run each graph came from.
    A PDF already open in Preview, or already opened this session, is skipped.
    All new PDFs are opened in a single `open` call and (unless --no-tabbed) then
    merged into tabs of one window.
    """
    if not args.open_pdfs:
        return

    # distinct qualifying run folders, best-first
    folders: list = []
    for (_path, _n), (_sortino, pval, _raw, params) in ranked:
        if pval < args.open_pval_max and params not in folders:
            folders.append(params)
    if args.open_max > 0:
        folders = folders[:args.open_max]

    already = preview_open_paths()
    to_open: list = []   # (folder, two-component relative pdf path)
    for folder in folders:
        if folder in opened:
            continue
        rel_pdf = os.path.join(folder, PDF_NAME)   # two components, relative to --dir
        abs_pdf = os.path.realpath(os.path.join(args.dir, rel_pdf))
        if not os.path.isfile(abs_pdf):
            print(f"    (no {PDF_NAME} in {folder})", flush=True)
            continue
        if abs_pdf in already:
            opened.add(folder)                     # already up in Preview; leave it
            continue
        to_open.append((folder, rel_pdf))

    if not to_open:
        return

    # open them all at once so they arrive together (and tab cleanly)
    try:
        subprocess.Popen(["open"] + [rel for _folder, rel in to_open], cwd=args.dir)
    except OSError as e:
        print(f"    warning: could not open PDFs: {e}", flush=True)
        return
    for folder, rel in to_open:
        opened.add(folder)
        print(f"    opened {rel}", flush=True)

    if args.tabbed:
        # let Preview create the windows before asking it to merge them to tabs
        time.sleep(1.5)
        merge_preview_windows_into_tabs()


def scan_once(args: argparse.Namespace, found: dict, timing: dict, opened: set) -> None:
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

    # optionally pop open the equity-curve PDFs for the best qualifying runs
    open_best_pdfs(args, ranked, opened)

    lines = []
    for rank, ((_path, n), (sortino, pval, raw, params)) in enumerate(ranked, 1):
        raw_str = "     n/a" if raw is None else f"{raw:+.4f}"
        lines.append(f"    {rank:>3}. model {n}: sortino{n}={sortino:.4f}  "
                     f"bestM{n}pval={pval:.4f}  rawReturn{n}={raw_str}  |  {params}")
    for line in lines:
        print(line, flush=True)

    # overwrite the outfile each cycle with the current ranked list (a snapshot,
    # not a running log), so there is always an up-to-date file to inspect
    header = (f"# study monitor -- interesting hits as of {stamp} "
              f"(sortino>={args.sortino_min}, pval<={args.pval_max})\n"
              f"# {len(found)} found so far\n")
    try:
        with open(os.path.join(args.dir, args.outfile), "w") as fh:
            fh.write(header)
            fh.write("\n".join(lines) + ("\n" if lines else ""))
    except OSError as e:
        print(f"    warning: could not write {args.outfile}: {e}", flush=True)


def main() -> None:
    args = parse_args()
    print(f"=== study monitor: dir={args.dir!r} glob={args.glob!r} "
          f"sortino>={args.sortino_min} pval<={args.pval_max} "
          f"every {args.interval}s (Ctrl-C to stop) ===", flush=True)
    seen: dict = {}
    timing: dict = {"known": set(), "last": None, "deltas": []}
    opened: set = set()   # run folders whose PDF we have already opened this session
    try:
        while True:
            scan_once(args, seen, timing, opened)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n=== monitor stopped ===", flush=True)


if __name__ == "__main__":
    main()
