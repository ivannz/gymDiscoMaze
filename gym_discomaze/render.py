import numpy as np

from pyglet import gl

from gym.envs.classic_control.rendering import Viewer
from gym.envs.classic_control.rendering import FilledPolygon


class Renderer:
    def __init__(self, n_row, n_col, *, pixel=(1, 1)):
        h, w = pixel
        self.viewer = Viewer(n_col * h, n_row * w)
        self.viewer.set_bounds(0, n_col, 0, n_row)

        pixels = []
        for i in range(n_row):
            for j in range(n_col):
                pixel = FilledPolygon([
                    (j+0, n_row-(i+0)), (j+0, n_row-(i+1)),
                    (j+1, n_row-(i+1)), (j+1, n_row-(i+0)),
                ])
                pixels.append(pixel)
                self.viewer.add_geom(pixel)
        self.pixels = np.array(pixels).reshape(n_row, n_col)

    @property
    def shape(self):
        return self.pixels.shape

    def __getitem__(self, index):
        return self.pixels[index]

    def close(self):
        self.viewer.close()
        del self.viewer
        del self.pixels

    def render(self, pixels, mode='human'):
        n_row, n_col = self.shape
        for i in range(n_row):
            for j in range(n_col):
                self[i, j].set_color(*pixels[i, j])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
