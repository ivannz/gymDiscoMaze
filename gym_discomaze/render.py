import numpy as np

from pyglet.gl import GL_QUADS
from pyglet.graphics import vertex_list
from gym.envs.classic_control.rendering import Viewer as _Viewer


class Viewer(_Viewer):
    """Allow default closing sequence"""
    def window_closed_by_user(self):
        # call gym's viewer's overriding handler
        super().window_closed_by_user()

        # if pyglet.app.event_loop is being used, the default `on_close` is
        #  also call `.close()` on the window, closing the it immediately.
        type(self.window).on_close(self.window)

        # ... however, when there is no event loop, the window lingers, so
        #  we close it once more (has no effect is it has already been closed).
        self.window.close()


class FilledQuadArray:
    """Pixel array using pyglet's streamlined array drawing."""

    def __init__(self, quads, colors):
        # assume float dtype of colors and qauds
        self.quads, self.colors = quads, colors
        self.kind = 'c4f' if colors.shape[-1] == 4 else 'c3f'

    def render(self):
        # `quads` is a flattened array of shape `(n_quads, 4, 2)`
        # `colors` is an array of shape `(n_quads, [3 or 4])`
        vl = vertex_list(
            len(self.quads) // 2, ('v2f', self.quads),
            (self.kind, list(np.tile(self.colors, (1, 4)).flat))
        )
        vl.draw(GL_QUADS)
        vl.delete()


class Renderer:
    def __init__(self, n_row, n_col, *, pixel=(1, 1)):
        self.n_row, self.n_col, self.pixel = n_row, n_col, pixel

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

        self.open()

    def render(self, pixels, mode='human'):
        self.viewer.add_onetime(
            FilledQuadArray(self._quads, pixels.reshape(-1, pixels.shape[-1])))
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def open(self):
        assert not self.is_open

        h, w = self.pixel
        self.viewer = Viewer(self.n_col * w, self.n_row * h)
        self.viewer.set_bounds(0, self.n_col, 0, self.n_row)

    def close(self):
        self.viewer.close()
        del self.viewer

    @property
    def is_open(self):
        return hasattr(self, 'viewer')
