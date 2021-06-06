import numpy as np

from gym.spaces import Box, Dict

from ..env import RandomDiscoMaze


class RandomDiscoMazeWithPosition(RandomDiscoMaze):
    """DiscoMaze with current coordinates added to the observation space."""
    PLAYER = RandomDiscoMaze.PLAYER

    def __init__(self, n_row=10, n_col=10, *, n_colors=5, n_targets=1,
                 field=None, generator=None):
        super().__init__(n_row, n_col, field=field, generator=generator,
                         n_colors=n_colors, n_targets=n_targets)

        # position has integer coordinates in a 2d-box
        self.observation_space = Dict(
            position=Box(*self.maze.shape, dtype=int, shape=(2,)),
            state=self.observation_space,
        )

    def observation(self, *, by=PLAYER):
        return dict(
            position=np.array(self.objects[by]),
            state=super().observation(by=by),
        )
