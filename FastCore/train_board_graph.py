#!/usr/bin/env python3
"""GeneticAlgo.train_board_graph with its training games played in C++.

    python -m FastCore.train_board_graph <the same arguments> [--threads N]

Identical arguments, files and checkpoints; --workers (or --threads) sets the
C++ thread count. Checkpoints made by either version can be resumed by the
other.
"""
import GeneticAlgo.train_board_graph as trainer

from .engine_entry import start

# Checkpoints written by the original script refer to __main__.GenerationStats.
GenerationStats = trainer.GenerationStats

if __name__ == "__main__":
    start()
    raise SystemExit(trainer.main())
