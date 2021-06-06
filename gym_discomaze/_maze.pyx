# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: emit_code_comments=True

import numpy as np
cimport numpy as np
cimport cython
np.import_array()

from libc.stdint cimport uint32_t, uint16_t, uint8_t

# lookup table of the number of set bits in a `uint8`
cdef uint8_t *nbits = [
    # cols - lo 4bit, rows - hi 4bit
    0, 1, 1, 2,   1, 2, 2, 3,   1, 2, 2, 3,   2, 3, 3, 4,
    1, 2, 2, 3,   2, 3, 3, 4,   2, 3, 3, 4,   3, 4, 4, 5,
    1, 2, 2, 3,   2, 3, 3, 4,   2, 3, 3, 4,   3, 4, 4, 5,
    2, 3, 3, 4,   3, 4, 4, 5,   3, 4, 4, 5,   4, 5, 5, 6,

    1, 2, 2, 3,   2, 3, 3, 4,   2, 3, 3, 4,   3, 4, 4, 5,
    2, 3, 3, 4,   3, 4, 4, 5,   3, 4, 4, 5,   4, 5, 5, 6,
    2, 3, 3, 4,   3, 4, 4, 5,   3, 4, 4, 5,   4, 5, 5, 6,
    3, 4, 4, 5,   4, 5, 5, 6,   4, 5, 5, 6,   5, 6, 6, 7,

    1, 2, 2, 3,   2, 3, 3, 4,   2, 3, 3, 4,   3, 4, 4, 5,
    2, 3, 3, 4,   3, 4, 4, 5,   3, 4, 4, 5,   4, 5, 5, 6,
    2, 3, 3, 4,   3, 4, 4, 5,   3, 4, 4, 5,   4, 5, 5, 6,
    3, 4, 4, 5,   4, 5, 5, 6,   4, 5, 5, 6,   5, 6, 6, 7,

    2, 3, 3, 4,   3, 4, 4, 5,   3, 4, 4, 5,   4, 5, 5, 6,
    3, 4, 4, 5,   4, 5, 5, 6,   4, 5, 5, 6,   5, 6, 6, 7,
    3, 4, 4, 5,   4, 5, 5, 6,   4, 5, 5, 6,   5, 6, 6, 7,
    4, 5, 5, 6,   5, 6, 6, 7,   5, 6, 6, 7,   6, 7, 7, 8,
    # N_2 = [[0, 1], [1, 2]], N_4 = N_2 (+) N_2, and N_8 = N_4 (+) N_4
]


# modern numpy's prng
from numpy.random cimport bitgen_t

cdef uint32_t random_tomax(bitgen_t *rng, uint32_t max) nogil:
    """Draw a random integer N with 0 <= N <= max.

    Details
    -------
    Uses rejection sampling to avoid bias, hence the runtime is stochastic.
    """
    # roll out own `random_interval` so as not to link against
    #  an extra library for a single quite simple function.
    #  https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c#L1022
    #  https://numpy.org/devdocs/reference/random/examples/cython/extending.pyx.html
    cdef uint32_t mask, value

    mask = max
    mask |= mask >>  1
    mask |= mask >>  2
    mask |= mask >>  4
    mask |= mask >>  8
    mask |= mask >> 16

    # at most 50% rejection if `max` = 2^p, then `mask` = 2^{p+1} - 1.
    value = rng.next_uint32(rng.state) & mask
    while value > max:
        value = rng.next_uint32(rng.state) & mask

    return value


cdef uint32_t random_onehot(bitgen_t *rng, uint32_t mask) nogil:
    """Draw a random one-hot among the set bits of the given 32bit mask.

    If the mask represents a set P, then return the mask of a random singleton:
        for a given P \\subset \\{0..31\\}  # represented as a binary mask
        return X = \\{j\\} where j \\sim P  # ditto
    """
    cdef int n_setbits = nbits[(mask >>  0) & 255] + nbits[(mask >>  8) & 255] \
                       + nbits[(mask >> 16) & 255] + nbits[(mask >> 24) & 255]

    # no need for randomness if the choice is obvious (single or no bits)
    if n_setbits < 2:
        return mask

    # pick an index of a set bit at random
    cdef int i_setbit = random_tomax(rng, n_setbits - 1)

    # A bit twiddling hack for ``Counting bits set, Brian Kernighan's way''
    #    \url{https://graphics.stanford.edu/~seander/bithacks.html}
    # We repurpose this hack for getting a random one-hot
    cdef uint32_t temp
    while mask and i_setbit >= 0:  # min(i_setbit, n_setbits)
        # it doesn't matter which end we count the set bits from
        i_setbit -= 1

        # clear the least significant bit set in the `mask`
        temp, mask = mask, mask & (mask - 1)

    # `temp` and `mask` are equal except for the most recently cleared lsb
    return temp ^ mask

    # cdef int onehot2 = 1  # one-hot corresponding to the zero-th index
    # while not (onehot & onehot2): onehot2 <<= 1


# cell bit-fields' values and offsets
cdef enum:
    W = 8, S = 4, E = 2, N = 1  # values
    BACKTRACK = 8, BORDER = 4, WALL = 0  # offsets
    WSEN = W | S | E | N


cdef uint16_t[:, ::1] random_perfect_maze(bitgen_t *rng, uint16_t[:, ::1] cells) nogil:
    """Perfect Maze generator using random DFS.

    Details
    -------
    Implements iterative version of the depth-first maze builder.

    Links
    -----
    Based on
    https://web.archive.org/web/20150816164625/http://mazeworks.com/mazegen/mazetut/index.htm
    https://en.wikipedia.org/wiki/Maze_generation_algorithm#Iterative_implementation
    """
    cdef int neighbours, direction, backtrack

    # run a random direction dfs through a lattice of cells starting at (0, 0)
    cdef int r=0, c=0  # must be inside the bordered region
    while True:
        # determine accessible neighbours: a border indicates out-of-bounds
        neighbours = ((cells[r, c] >> BORDER) & WSEN) ^ WSEN

        # ignore accessible neighbours which do not have all walls erect
        if neighbours & W:
            if ((cells[r, c-1] >> WALL) & WSEN) != WSEN:
                neighbours ^= W

        if neighbours & E:
            if ((cells[r, c+1] >> WALL) & WSEN) != WSEN:
                neighbours ^= E

        if neighbours & N:
            if ((cells[r-1, c] >> WALL) & WSEN) != WSEN:
                neighbours ^= N

        if neighbours & S:
            if ((cells[r+1, c] >> WALL) & WSEN) != WSEN:
                neighbours ^= S

        # backtrack if the current cell is a dead end
        if not neighbours:
            backtrack = (cells[r, c] >> BACKTRACK) & WSEN
            cells[r, c] &= ~(WSEN << BACKTRACK)

            # undefined direction means that we've returned to the start
            if not backtrack:
                break

            # a valid backtrack always returns us to a valid cell
            elif backtrack & W:
                c -= 1

            elif backtrack & E:
                c += 1

            elif backtrack & N:
                r -= 1

            elif backtrack & S:
                r += 1

            continue
        # end if

        # fetch the random wall to knock down
        direction = random_onehot(rng, neighbours)

        # tear down the first wall
        cells[r, c] &= ~(direction << WALL)

        # move to the specified cell
        if direction == W:
            c -= 1
            backtrack = E

        elif direction == E:
            c += 1
            backtrack = W

        elif direction == S:
            r += 1
            backtrack = N

        elif direction == N:
            r -= 1
            backtrack = S

        # tear down the second wall
        cells[r, c] &= ~(backtrack << WALL)

        # keep the backtracking direction
        cells[r, c] |= backtrack << BACKTRACK

    # end while

    return cells


cdef uint16_t[:, ::1] reset_rectangle_maze(uint16_t[:, ::1] cells) nogil:
    """Reset the rectangular array of cells."""

    cdef int r, c, n=cells.shape[0], m=cells.shape[1]

    # wall up each cell in the lattice
    for r in range(n):
        for c in range(m):
            cells[r, c] = WSEN << WALL

    # west-east borders
    for r in range(n):
        cells[r, 0] |= W << BORDER
        cells[r, m-1] |= E << BORDER

    # north-south borders
    for c in range(m):
        cells[0, c] |= N << BORDER
        cells[n-1, c] |= S << BORDER

    return cells


# https://numpy.org/doc/stable/reference/random/extending.html#cython
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer

@cython.embedsignature(True)
def generate_perfect_maze(int n, int m, *, generator):
    """Perfect Maze generator using random DFS."""
    assert isinstance(generator, np.random.Generator)

    cdef const char *capsule_name = "BitGenerator"
    cdef bitgen_t *rng

    capsule = generator.bit_generator.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")

    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    # generate cell representation of a rectangular perfect maze
    cdef uint16_t[:, ::1] cells = np.empty((n, m), dtype=np.uint16)
    reset_rectangle_maze(cells)
    with generator.bit_generator.lock:
        random_perfect_maze(rng, cells)

    # output maze is a boolean array
    cdef uint8_t[:, ::1] maze = np.empty((2*n + 1, 2*m + 1), dtype=np.bool)
    cdef int r, c, cell, walls
    with nogil:
        # build the binary maze from the cell data
        for r in range(2 * n + 1):
            for c in range(2 * m + 1):
                # only 'rooms' with odd coordinates correspond to cells
                if not ((r & 1) and (c & 1)):
                    maze[r, c] = True
                    continue

                # make the room passable
                maze[r, c] = False

                # knock down rooms' walls indicated by the cell's data
                cell = cells[r >> 1, c >> 1]
                walls = ((cell >> BORDER) | (cell >> WALL)) & WSEN
                if not (walls & W):
                    maze[r, c-1] = False

                if not (walls & E):
                    maze[r, c+1] = False

                if not (walls & N):
                    maze[r-1, c] = False

                if not (walls & S):
                    maze[r+1, c] = False

    return np.asarray(maze)
