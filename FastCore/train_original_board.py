#!/usr/bin/env python3
"""GeneticAlgo.train_original_board with its games played in C++.

    python -m FastCore.train_original_board <the same arguments> [--threads N]
"""
import GeneticAlgo.train_original_board as minimiser

from .engine_entry import start

if __name__ == "__main__":
    start()
    raise SystemExit(minimiser.main())
