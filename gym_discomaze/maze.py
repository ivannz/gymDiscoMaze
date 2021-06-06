import numpy as np
from . import _maze

# constants
P_BACKTRACK, P_BORDER, P_WALL = 8, 4, 0  # nibble bit shifts

W, S, E, N = 8, 4, 2, 1  # bits within each nibble
WSEN = W | S | E | N     # mask to isolate a nibble

# convenience structures: direction atals and all directions
ATLAS = {
    0: (0, 0, +0, +0),  # bogus, 'stay-in-place', durection
    W: (W, E, +0, -1),
    S: (S, N, +1, +0),
    E: (E, W, +0, +1),
    N: (N, S, -1, +0),
}

DIRECTIONS = [0, W, S, E, N]
DIR_LABELS = ['stay', 'west', 'south', 'east', 'north']


def generate(n_row, n_col, *, generator=None):
    """Perfect Maze generator using random DFS.

    Details
    -------
    Implements iterative version of the depth-first maze builder.

    Links
    -----
    https://en.wikipedia.org/wiki/Maze_generation_algorithm#Iterative_implementation
    https://web.archive.org/web/20150816164625/http://mazeworks.com/mazegen/mazetut/index.htm
    """
    # re-package the random bit generator from the legacy random state
    if isinstance(generator, np.random.RandomState):
        generator = generator._bit_generator
    generator = np.random.default_rng(generator)

    return _maze.generate_perfect_maze(n_row, n_col, generator=generator)
