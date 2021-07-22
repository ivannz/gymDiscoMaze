import numpy

from queue import deque
from ..env import RandomDiscoMaze


def bfs(map, x, y):
    # simple bfs to precompute shortest path costs to the goal
    cost = numpy.full_like(map, numpy.nan, dtype=float)

    # seed the bfs with the source node
    frontier = deque([(x, y)])
    cost[x, y], max = 0, 0

    # frontier holds the explored cells, with the shortest path's
    # length in the cost array
    while frontier:
        x, y = frontier.popleft()
        for i, j in (x+1, y), (x, y+1), (x-1, y), (x, y-1):
            if numpy.isnan(cost[i, j]) and map[i, j] != -1:
                v = cost[i, j] = 1 + cost[x, y]
                frontier.append((i, j))
                max = v if max < v else max

    return cost / max


class ExploreRandomDiscoMaze(RandomDiscoMaze):
    """DiscoMaze with goal-oriented reward shaping."""
    def __init__(self, n_row=10, n_col=10, *, n_colors=5,
                 field=None, generator=None, alpha=10.):
        self.alpha = alpha
        super().__init__(n_row, n_col, field=field, generator=generator,
                         n_colors=n_colors, n_targets=0)

    @property
    def player(self):
        return self.objects[self.PLAYER]

    def reset(self):
        obs = super().reset()

        # generate the unobserved goal coordinates
        empty = self.maze.coordinates_of(self.maze.EMPTY)
        self.goal = tuple(self.generator_.choice(empty))

        # precimpute the reward based on shortes path to the goal
        rewards = 1 - bfs(self.maze.map, *self.goal)
        self.proximity_reward = numpy.power(rewards, self.alpha)
        return obs

    def step(self, action):
        obs, rew, fin, info = super().step(action)
        pos = self.player

        rew = self.proximity_reward[pos] if not fin else -1.
        fin = fin or pos == self.goal

        return obs, rew, fin, dict(goal=self.goal, player=pos)
