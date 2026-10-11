#!/usr/bin/env python3
"""Regenerate FastCore's copies of the project's run scripts.

    python FastCore/sync_scripts.py              every script below except HAND_EDITED
    python FastCore/sync_scripts.py NAME.sh ...  only these

Each copy is the original with three changes: it starts in the project root,
it calls FastCore.train_board_graph / FastCore.train_original_board instead of
the GeneticAlgo trainers, and it relaunches itself by absolute path. Run this
after editing an original script so the C++ copy follows. The originals are
never modified.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = (
    "run_board_graph_steady.sh", "run_board_graph_overnight.sh", "run_board_graph_conditions.sh",
    "run_board_graph_ablation.sh", "run_board_graph_endgame_experiment.sh",
    "run_board_graph_repair_comparison.sh", "run_board_graph_islands.sh", "run_original_board.sh",
    "run_board_graph_short.sh", "run_board_graph_long.sh", "run_board_graph_large_population.sh",
    "run_board_graph_evasion_experiment.sh", "run_board_graph_islands_evasion.sh",
)
# FastCore copies changed by hand, skipped unless named on the command line.
HAND_EDITED: tuple[str, ...] = ()
HEADER = ("# FastCore copy of {name}, made by FastCore/sync_scripts.py: the same run\n"
          "# with its training games played in C++ (FastCore/README.md). Do not edit;\n"
          "# edit the original and rerun sync_scripts.py. Build first with\n"
          "# FastCore/build.sh. Threads: BOARD_GRAPH_THREADS (default: all CPUs).\n")
CD = 'cd "$(dirname "$0")"\n'


def convert(name: str, text: str) -> str:
    first, rest = text.split("\n", 1)
    text = first + "\n" + HEADER.format(name=name) + rest
    text = text.replace("-m GeneticAlgo.train_board_graph", "-m FastCore.train_board_graph")
    text = text.replace("-m GeneticAlgo.train_original_board", "-m FastCore.train_original_board")
    if CD in text:
        before, after = text.split(CD, 1)
        text = before + 'script="$(realpath "$0")"\ncd "$(dirname "$0")/.."\n' + after.replace('"$0"', '"$script"')
    return text


def main() -> int:
    import sys
    names = sys.argv[1:] or [name for name in SCRIPTS if name not in HAND_EDITED]
    for name in names:
        source = ROOT / name
        if not source.exists():
            print(f"skipped {name}: not found")
            continue
        target = ROOT / "FastCore" / name
        target.write_text(convert(name, source.read_text()))
        os.chmod(target, 0o755)
        print(f"wrote FastCore/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
