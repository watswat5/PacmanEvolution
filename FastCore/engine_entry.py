"""Shared start-up for the FastCore entry points."""
from __future__ import annotations

import os
import sys

from . import backend


def take_threads(argv: list[str]) -> tuple[list[str], int]:
    """Remove --workers/--threads from argv; return the thread count to use.

    The original trainers spread games over worker processes (--workers).
    The C++ engine uses threads in one process instead, so the trainer gets
    --workers 1 and the requested number becomes the thread count.
    """
    threads = int(os.environ.get("BOARD_GRAPH_THREADS", "0") or 0)
    kept, skip = [], False
    for index, item in enumerate(argv):
        if skip:
            skip = False
            continue
        for flag in ("--workers", "--threads"):
            if item == flag:
                threads = int(argv[index + 1]) or threads
                skip = True
                break
            if item.startswith(flag + "="):
                threads = int(item.split("=", 1)[1]) or threads
                break
        else:
            kept.append(item)
    if not threads:
        threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return kept + ["--workers", "1"], threads


def start() -> int:
    sys.argv[1:], threads = take_threads(sys.argv[1:])
    backend.install(threads)
    print(f"Game engine: C++ (FastCore), {threads} threads; worker processes are not used")
    return threads
