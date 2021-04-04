import numpy as np

from pyglet.gl import GL_QUADS
from pyglet.graphics import vertex_list
from gym.envs.classic_control.rendering import Viewer as _Viewer


class Viewer(_Viewer):
    """Allow default closing sequence"""
    def window_closed_by_user(self):
        type(self.window).on_close(self.window)
        super().window_closed_by_user()
        self.window.close()


class FilledQuadArray:
    """Pixel array using pyglet's streamlined array drawing."""

    def __init__(self, quads, colors):
        self.quads, self.colors = quads, colors

    def render(self):
        # `quads` is a flattened array of shape `(n_quads, 4, 2)`
        # `colors` is an array of shape `(n_quads, 3)`
        vl = vertex_list(
            len(self.quads) // 2, ('v2f', self.quads),
            ('c3f', list(np.tile(self.colors, (1, 4)).flat))
        )
        vl.draw(GL_QUADS)
        vl.delete()


class Renderer:
    def __init__(self, n_row, n_col, *, pixel=(1, 1)):
        h, w = pixel
        self.viewer = Viewer(n_col * w, n_row * h)
        self.viewer.set_bounds(0, n_col, 0, n_row)

        # pyglet uses lower left as the origin, while we use upper left
        self._quads = []  # create quads in numpy's order
        for i in range(n_row):
            for j in range(n_col):
                self._quads.extend((
                    j+0, n_row-(i+0),  # ur
                    j+0, n_row-(i+1),  # lr
                    j+1, n_row-(i+1),  # ll
                    j+1, n_row-(i+0),  # ul
                ))

    def render(self, pixels, mode='human'):
        self.viewer.add_onetime(
            FilledQuadArray(self._quads, pixels.reshape(-1, 3)))
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        self.viewer.close()
        del self.viewer
