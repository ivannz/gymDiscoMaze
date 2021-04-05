import numpy as np

from gym import Env
from gym.spaces import Discrete, Box

from random import Random
from sys import maxsize

from . import maze


class BaseMap:
    EMPTY = 0  # hardcoded zero id

    def __init__(self, n_row, n_col):
        self.map = np.full((n_row, n_col), self.EMPTY, dtype=int)

    @property
    def shape(self):
        return self.map.shape

    @property
    def size(self):
        return self.map.size

    def coordinates_of(self, kind=EMPTY):
        return np.stack((self.map == kind).nonzero(), 0).T

    def __getitem__(self, index):
        return self.map[index]

    def __setitem__(self, index, value):
        assert value != self.EMPTY, 'use `del obj[i, j]` to free space'
        self.map[index] = value

    def __delitem__(self, index):
        self.map[index] = self.EMPTY

    def __repr__(self):
        text = 'x'.join(map(str, self.shape))
        return type(self).__name__ + '(' + text + ')'

    def __eq__(self, other):
        return self.map == other

    def is_empty(self, i, j):
        return self[i, j] == self.EMPTY

    def relocate(self, p0, p1):
        if self.is_empty(*p1):
            # can relocate to non-obstructed tiles only
            self[p1] = self[p0]
            del self[p0]
            return self.EMPTY

        # return the displaced content
        return self[p1]


class MazeMap(BaseMap):
    WALL = -1

    def __init__(self, n_row, n_col, *, random=None):
        super().__init__(1 + 2 * n_row, 1 + 2 * n_col)

        self.random_ = random or Random()
        assert isinstance(self.random_, Random)

        walls = maze.generate(n_row, n_col, random=self.random_)
        self.map[walls] = self.WALL


class RandomDiscoMaze(Env):
    """Random Disco Maze

    Details
    -------
    Custom implementation of the Random Disco Maze environment from section 4.1
    of [Badia et al. (2020)](https://arxiv.org/abs/2002.06038).
    """
    DIRECTION = maze.DIRECTIONS[:]
    PLAYER = 1  # hardcoded id of the player

    metadata = {
        'render.modes': {'human', 'rgb_array', 'state_pixels'},
    }

    def __init__(self, n_row=10, n_col=10, *, n_colors=5, n_targets=1, random=None):
        # super().__init__()
        self.random_ = random or Random()
        assert isinstance(self.random_, Random)

        from matplotlib.cm import hot
        *extra, = map(tuple, hot(np.linspace(0.2, 0.8, num=n_colors))[:, :-1])
        self.COLORS = ((0., 0., 0.), (1., 1., 1.), (0.3, 0.3, 1.0), *extra)

        self.n_row, self.n_col, self.n_targets = n_row, n_col, n_targets

        # cache the pixels so that consecutive calls to `.render` with
        #  `mode` other than `state_pixels` yield the same result.
        self._state = None
        self.reset()

        # actions are the cardinal directions
        self.action_space = Discrete(4)

        # the observation space is state `pixels'
        self.observation_space = Box(
            low=0., high=1., dtype=float, shape=(*self.maze.shape, 3))

    @property
    def state(self):
        return self._state.copy()

    def spawn(self, n=1):
        # generate positions
        *empty, = map(tuple, self.maze.coordinates_of(MazeMap.EMPTY))
        positions = self.random_.sample(empty, k=n)
        for i, j in positions:
            self.maze[i, j] = len(self.objects)
            self.objects.append((i, j))
            self.targets.add(self.maze[i, j])

        return positions

    def reset(self):
        self.maze = MazeMap(self.n_row, self.n_col, random=self.random_)

        # create the player : `None` represents the empty space
        i, j = self.random_.choice(self.maze.coordinates_of(MazeMap.EMPTY))
        self.maze[i, j] = self.PLAYER
        self.objects = [None, (i, j)]
        self.is_alive = True

        # ... and the targets
        self.targets = set()
        self.spawn(self.n_targets)

        self._state = self.update()
        return self._state

    def update(self, *, maze=None):
        maze = maze or self.maze
        assert isinstance(maze, BaseMap)

        # pick random colors and paint walls with them
        colors = self.random_.choices(self.COLORS[3:], k=maze.size)
        pix = np.array(colors).reshape(*maze.shape, 3)

        # assign proper colors to empty space, the player and the targets
        n_row, n_col = maze.shape
        for i in range(n_row):
            for j in range(n_col):
                if maze[i, j] == MazeMap.EMPTY:
                    pix[i, j] = self.COLORS[0]

                elif maze[i, j] == self.PLAYER:
                    pix[i, j] = self.COLORS[1]

                elif maze[i, j] in self.targets:
                    pix[i, j] = self.COLORS[2]

        return pix

    def _move(self, oid, dir):
        i, j = self.objects[oid]

        _, _, u, v = maze.ATLAS[dir]
        u, v = u + i, v + j

        dest_id = self.maze[u, v]
        if dest_id in self.targets:
            del self.maze[u, v]  # consume the target

        if dest_id == MazeMap.WALL:
            del self.maze[u, v]  # displace the wall

        if self.maze.relocate((i, j), (u, v)) == MazeMap.EMPTY:
            self.objects[oid] = u, v  # update the position

        return dest_id

    def step(self, action):
        dest_id = MazeMap.EMPTY
        if action is not None and self.is_alive:
            dest_id = self._move(self.PLAYER, self.DIRECTION[action])

        # check winning conditions
        reward = 0.
        if dest_id in self.targets:
            self.targets.remove(dest_id)
            self.objects[dest_id] = None
            reward = 1.

        # check termination conditions: maze hazards, or no targets left
        any_targets = bool(self.targets) or (self.n_targets == 0)
        is_terminal = (dest_id == MazeMap.WALL) or not any_targets

        self.is_alive = (dest_id != MazeMap.WALL) and self.is_alive

        self._state = self.update()
        return self._state, reward, is_terminal, {}

    def render(self, mode='state_pixels'):
        assert mode in self.metadata['render.modes']
        if mode == 'state_pixels':
            return self.state

        # other rendering
        if not hasattr(self, '_viewer'):
            from .render import Renderer
            self._viewer = Renderer(*self.maze.shape, pixel=(10, 10))

        return self._viewer.render(self._state, mode)

    def seed(self, seed=None):
        if seed is None:
            seed = Random().randrange(maxsize)

        self.random_.seed(seed)
        return [seed]

    @property
    def viewer(self):
        # bypass for compatibility with certain scripts in gym
        if not hasattr(self, '_viewer'):
            return None

        return self._viewer.viewer

    def close(self):
        if hasattr(self, '_viewer'):
            self._viewer.close()
            del self._viewer
