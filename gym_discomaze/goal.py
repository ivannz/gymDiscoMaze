import numpy as np
from gym import GoalEnv, spaces

from .env import RandomDiscoMaze, BaseMap


class RandomDiscoGoal(GoalEnv):
    def __init__(self, n_row=10, n_col=10, *, n_colors=5, random=None):
        super().__init__()

        self.env = RandomDiscoMaze(n_row, n_col, n_targets=0,
                                   n_colors=n_colors, random=random)
        self.observation_space = spaces.Dict(dict.fromkeys([
            'observation',
            'achieved_goal',
            'desired_goal',
        ], self.env.observation_space))
        self.reset()

    def seed(self, seed):
        return self.env.seed(seed)

    def _obs(self):
        return {
            'observation': self.env.state.copy(),
            'achieved_goal': self.env.state.copy(),
            'desired_goal': self.goal_state.copy(),
        }

    @property
    def player(self):
        return self.env.objects[self.env.PLAYER]

    @property
    def viewer(self):
        return self.env.viewer

    def reset(self):
        self.env.reset()

        # generate the goal coordinates and the state
        empty = self.env.maze.coordinates_of(self.env.maze.EMPTY)
        self.goal = i, j = tuple(self.env.random_.choice(empty))

        # create a new map and generate a state for it
        self.goal_maze = BaseMap(*self.env.maze.shape)
        self.goal_maze.map[:] = self.env.maze.map

        # deleted the current player and place another one at the goal
        self.goal_maze[self.goal] = self.env.PLAYER
        del self.goal_maze[self.player]

        self.goal_state = self.env.update(maze=self.goal_maze)
        return self._obs()

    def step(self, action):
        _, reward, is_terminal, info = self.env.step(action)

        has_reached = self.player == self.goal
        reward = float(has_reached)
        return self._obs(), reward, (is_terminal or has_reached), {
            **info, "is_success": has_reached
        }

    def render(self, mode='state_pixels'):
        return self.env.render(mode)
