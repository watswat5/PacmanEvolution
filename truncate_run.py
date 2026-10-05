#!/usr/bin/env python3
"""Copy the first N generations of a finished run into a new run folder.

    python truncate_run.py SOURCE_RUN DESTINATION_RUN --through 200

Lets an existing long run serve as the baseline of an experiment that stops
earlier, without training it again. The copy holds history.csv cut to N rows,
the generation champions 0 .. N-1, and best_overall.pkl set to the champion
with the highest tournament score in those generations (the first one on a
tie, as the trainer does). It has no checkpoint, so it cannot be resumed.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--through", type=int, required=True, help="keep generations 0 .. N-1")
    args = parser.parse_args()

    with (args.source / "history.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["generation"]) < args.through]
    if len(rows) < args.through:
        raise SystemExit(f"{args.source} has only {len(rows)} generations")
    if (args.destination / "history.csv").exists():
        raise SystemExit(f"{args.destination} already holds a run")
    args.destination.mkdir(parents=True, exist_ok=True)
    with (args.destination / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        name = f"generation_{int(row['generation']):04d}_best.pkl"
        shutil.copyfile(args.source / name, args.destination / name)
    best = max(rows, key=lambda row: float(row["tournament_fitness"]))
    shutil.copyfile(
        args.source / f"generation_{int(best['generation']):04d}_best.pkl",
        args.destination / "best_overall.pkl",
    )
    (args.destination / "summary.txt").write_text(
        f"First {args.through} generations of {args.source}. "
        f"Best fitness: {float(best['tournament_fitness'])} (generation {int(best['generation'])})\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.destination}: {len(rows)} generations, best from generation {int(best['generation'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
