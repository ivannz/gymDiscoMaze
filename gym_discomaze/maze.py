import numpy as np
from random import Random

# constants
P_BACKTRACK, P_BORDER, P_WALL = 8, 4, 0  # nibble bit shifts

W, S, E, N = 8, 4, 2, 1  # bits within each nibble
WSEN = W | S | E | N     # mask to isolate a nibble

# convenience structures: direction atals and all directions
ATLAS = {
    W: (W, E, +0, -1),
    S: (S, N, +1, +0),
    E: (E, W, +0, +1),
    N: (N, S, -1, +0),
}

DIRECTIONS = [W, S, E, N]


def generate(n_row, n_col, *, random=None):
    """Perfect Maze generator using random DFS.

    Details
    -------
    Implements iterative version of the depth-first maze builder.

    Links
    -----
    https://en.wikipedia.org/wiki/Maze_generation_algorithm#Iterative_implementation
    https://web.archive.org/web/20150816164625/http://mazeworks.com/mazegen/mazetut/index.htm
    """
    random_ = random or Random()
    assert isinstance(random_, Random)

    # initialize the maze grid: erect walls and add borders
    data = np.zeros((n_row, n_col), dtype=np.uint16)
    data |= WSEN << P_WALL
    data[+0, :] |= N << P_BORDER
    data[-1, :] |= S << P_BORDER
    data[:, +0] |= W << P_BORDER
    data[:, -1] |= E << P_BORDER

    def get_walls(i, j, d):
        '''Check if the cell has all walls intact.'''
        # ignore indestructible borders
        if (data[i, j] >> P_BORDER) & d:
            return 0

        _, _, u, v = ATLAS[d]
        cell = data[i + u, j + v]

        # check the cell's walls and borders
        return ((cell >> P_WALL) | (cell >> P_BORDER)) & WSEN

    # since al positions are accessible in a perfect maze it doesn't matter
    #  where we start from, provided it is within the bordered region.
    i, j = 0, 0

    # DFS build a maze `O(4 * n_row * n_col)`.
    while True:
        # get all walls, removing indestructible borders from candidates
        neighbours = [d for d in ATLAS if get_walls(i, j, d) == WSEN]
        if not neighbours:
            # get the direction to backtrack to
            backtrack = (data[i, j] >> P_BACKTRACK) & WSEN
            data[i, j] &= ~(WSEN << P_BACKTRACK)

            if not backtrack:
                break

            _, _, u, v = ATLAS[backtrack]
            i, j = i + u, j + v

        else:
            # pick a direction
            fwd, rev, u, v = ATLAS[random_.choice(neighbours)]

            # tear down the first wall
            data[i, j] &= ~(fwd << P_WALL)

            # move to the specified cell
            i, j = i + u, j + v

            # tear down the second wall
            data[i, j] &= ~(rev << P_WALL)

            # keep the backtracking direction
            data[i, j] |= rev << P_BACKTRACK

    # build the maze from the data
    maze = np.zeros((2 * n_row + 1, 2 * n_col + 1), dtype=bool)

    walls = ((data >> P_WALL) | (data >> P_BORDER)) & WSEN
    for i in range(n_row):
        ver = slice(2 * i + 0, 2 * i + 3)
        for j in range(n_col):
            cell, hor = walls[i, j], slice(2 * j + 0, 2 * j + 3)
            if cell & W:
                maze[ver, 2 * j + 0] = True

            if cell & E:
                maze[ver, 2 * j + 2] = True

            if cell & N:
                maze[2 * i + 0, hor] = True

            if cell & S:
                maze[2 * i + 2, hor] = True
    return maze
