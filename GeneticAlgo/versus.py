#!/usr/bin/env python3
"""Play Pac-Man side by side against a saved tree, one step per key press.

    python -m GeneticAlgo.versus path/to/individual.pkl

Both sides start on the same random board (a new one every game) with the
same ghosts. Each maze gets its own wall colour from a 16-colour retro palette. Each arrow key moves you one tile and lets the tree choose its
own move on its copy of the board, so the two games stay in step. A side
stops when it dies or clears the board; keep pressing to let the other finish.

Scoring is the arcade's: 10 per dot, 50 per energizer, and 200, 400, 800, 1600
for successive ghosts eaten on one energizer. A clear always beats a non-clear;
otherwise the higher score wins.

Keys: arrows or WASD move, Space keeps going the way you are facing,
N starts a new random board, Esc quits.
"""
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

from PacmanGame import PacmanGame, precompute_board_graph, render_board
from PacmanGame.generate_tetris_board import generate_board
from PacmanGame.pacman import GHOST_SCORE, PELLET_SCORE
from PacmanGame.render_board import POWER_PELLET

from .train_board_graph import build_primitive_set, choose_move

# Wall colours, one picked per maze. No blue (the usual wall colour), and
# toned down so Pac-Man, the ghosts and the pellets stay the brightest things.
RETRO_PALETTE = (
    (200, 50, 74),    # crimson
    (192, 80, 80),    # brick
    (216, 100, 44),   # burnt orange
    (224, 160, 112),  # peach
    (168, 100, 52),   # brown
    (200, 155, 30),   # mustard
    (156, 200, 72),   # lime
    (140, 168, 42),   # olive
    (47, 168, 79),    # green
    (92, 184, 160),   # seafoam
    (31, 168, 144),   # teal
    (190, 160, 110),  # tan
    (176, 72, 192),   # purple
    (208, 76, 156),   # magenta
    (224, 120, 144),  # rose
    (140, 140, 156),  # slate
)


def maze_color(seed: int) -> tuple[int, int, int]:
    """The wall colour of the maze with this seed (always the same for it)."""
    return RETRO_PALETTE[random.Random(seed).randrange(len(RETRO_PALETTE))]


# Arcade points. The game's own score (1,000 per pellet, 100 per ghost) is a
# training signal, so this mode keeps the real Pac-Man score beside it.
DOT_POINTS = 10
ENERGIZER_POINTS = 50
GHOST_POINTS = (200, 400, 800, 1600)  # successive ghosts on one energizer


def energizers_left(game: PacmanGame) -> int:
    return sum(bool(value & POWER_PELLET) for _, _, value in game.board.iter_tiles())


KEY_MOVES = {
    "Up": "up", "w": "up", "W": "up",
    "Down": "down", "s": "down", "S": "down",
    "Left": "left", "a": "left", "A": "left",
    "Right": "right", "d": "right", "D": "right",
}


class VersusMatch:
    """Two games on the same board: one moved by the caller, one by a tree."""

    def __init__(self, tree, primitive_set=None, seed: int | None = None):
        self.tree = tree
        self.primitive_set = primitive_set or build_primitive_set()
        self.new_board(seed)

    def new_board(self, seed: int | None = None) -> None:
        self.seed = random.SystemRandom().randrange(2**31) if seed is None else seed
        board = generate_board(self.seed)
        graph = precompute_board_graph(board)
        self.human = PacmanGame(board.copy(), board_graph=graph)
        self.tree_game = PacmanGame(board.copy(), board_graph=graph)
        self.total_pellets = board.pellets_remaining
        self.steps = {"human": 0, "tree": 0}
        self.score = {"human": 0, "tree": 0}
        # Ghosts eaten on the current energizer, which sets the next one's value.
        self._ghost_chain = {"human": 0, "tree": 0}
        self.wall_color = maze_color(self.seed)
        self._rng = random.Random(self.seed)

    @staticmethod
    def finished(game: PacmanGame) -> bool:
        return game.game_over or game.won

    @property
    def over(self) -> bool:
        return self.finished(self.human) and self.finished(self.tree_game)

    def step(self, move: str | None = None) -> None:
        """Advance both sides one step. ``move`` None keeps your direction."""
        if not self.finished(self.human):
            self._scored_step("human", self.human, move or self.human.player_direction)
        if not self.finished(self.tree_game):
            choice, _ = choose_move(
                self.tree,
                self.tree_game.get_move_observations(),
                self.primitive_set,
                self.tree_game.player_direction,
                self._rng,
            )
            self._scored_step("tree", self.tree_game, choice or "stay")

    def _scored_step(self, side: str, game: PacmanGame, move: str) -> None:
        """Step one game and add the arcade points for what it ate."""
        score_before = game.board.score
        energizers_before = energizers_left(game)
        game.step(move)
        self.steps[side] += 1
        gained = game.board.score - score_before
        ate_pellet = gained >= PELLET_SCORE
        ghosts = (gained % PELLET_SCORE) // GHOST_SCORE
        if energizers_left(game) < energizers_before:
            self.score[side] += ENERGIZER_POINTS
            self._ghost_chain[side] = 0
        elif ate_pellet:
            self.score[side] += DOT_POINTS
        for _ in range(ghosts):
            chain = min(self._ghost_chain[side], len(GHOST_POINTS) - 1)
            self.score[side] += GHOST_POINTS[chain]
            self._ghost_chain[side] += 1

    def eaten(self, game: PacmanGame) -> int:
        return self.total_pellets - game.board.pellets_remaining

    def status(self, side: str) -> str:
        game = self.human if side == "human" else self.tree_game
        eaten = self.eaten(game)
        state = "CLEARED" if game.won else "DEAD" if game.game_over else (
            "powered" if game.board.player_powerup else "playing"
        )
        return (
            f"Score {self.score[side]:,} • "
            f"{eaten}/{self.total_pellets} pellets "
            f"({eaten / max(1, self.total_pellets):.0%}) • "
            f"{self.steps[side]} steps • {state}"
        )

    def result(self) -> str:
        """A clear always beats a non-clear; otherwise the higher score wins."""
        human = (self.human.won, self.score["human"])
        tree = (self.tree_game.won, self.score["tree"])
        if human == tree:
            return "Draw"
        return "PLAYER wins" if human > tree else "AI wins"


def run_window(match: VersusMatch, name: str, scale: int | None) -> int:
    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError as exc:
        raise SystemExit(f"This needs Tkinter and Pillow: {exc}") from exc

    root = tk.Tk()
    root.title("Pac-Man: player against AI")
    root.configure(background="#202428")
    sample = render_board(match.human.board)
    if scale is None:
        # Largest whole-number size at which both boards fit on the screen.
        fit_width = (root.winfo_screenwidth() - 80) // (2 * sample.width)
        fit_height = (root.winfo_screenheight() - 220) // sample.height
        scale = max(1, min(4, fit_width, fit_height))

    text = {"background": "#202428", "foreground": "white"}
    panels = {}
    titles = (("human", "PLAYER (you)"), ("tree", f"AI ({name})"))
    for column, (side, title) in enumerate(titles):
        frame = tk.Frame(root, background="#202428", padx=8, pady=8)
        frame.grid(row=0, column=column)
        tk.Label(frame, text=title, font=("TkDefaultFont", 16, "bold"), **text).pack()
        image = tk.Label(frame, background="#202428")
        image.pack()
        status = tk.Label(frame, pady=4, font=("TkDefaultFont", 11), **text)
        status.pack()
        panels[side] = (image, status)
    footer = tk.Label(root, pady=6, **text)
    footer.grid(row=1, column=0, columnspan=2, sticky="ew")
    photos = {}

    def redraw() -> None:
        for side, game in (("human", match.human), ("tree", match.tree_game)):
            picture = render_board(game.board, match.wall_color)
            if scale != 1:
                picture = picture.resize(
                    (picture.width * scale, picture.height * scale), resample=0
                )
            photos[side] = ImageTk.PhotoImage(picture)
            image, status = panels[side]
            image.configure(image=photos[side])
            status.configure(text=match.status(side))
        keys = "Arrows/WASD move • Space keeps going • N new board • Esc quits"
        if match.over:
            footer.configure(text=f"{match.result()} • board {match.seed} • N for a new board • Esc quits")
        elif match.finished(match.human):
            footer.configure(text=f"PLAYER is done; any move key lets the AI continue • board {match.seed}")
        else:
            footer.configure(text=f"{keys} • board {match.seed}")

    def on_key(event) -> None:
        key = getattr(event, "keysym", "")
        if key == "Escape":
            root.destroy()
            return
        if key in ("n", "N"):
            match.new_board()
        elif key == "space":
            match.step(None)
        elif key in KEY_MOVES:
            match.step(KEY_MOVES[key])
        else:
            return
        redraw()

    redraw()
    root.bind("<Key>", on_key)
    root.focus_force()
    root.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("individual", type=Path, help="saved tree (.pkl) to play against")
    parser.add_argument("--seed", type=int, help="board seed for the first game (default: random)")
    parser.add_argument("--scale", type=int, choices=range(1, 5), help="image size (default: fit the screen)")
    args = parser.parse_args()
    with args.individual.open("rb") as handle:
        tree = pickle.load(handle)
    match = VersusMatch(tree, seed=args.seed)
    return run_window(match, args.individual.stem, args.scale)


if __name__ == "__main__":
    raise SystemExit(main())
