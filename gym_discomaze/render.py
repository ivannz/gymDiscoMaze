import sys
import numpy as np

from pyglet import gl
from pyglet.image import ImageData
from pyglet.window import Window, key
from pyglet.canvas import Display

from random import randint


class SimpleImageViewer(Window):
    def __init__(self, caption=None, *, scale=None, display=None, vsync=False):
        scale = scale or (1, 1)
        if isinstance(scale, (int, float)):
            scale = scale, scale
        assert all(isinstance(p, (int, float)) and p > 0 for p in scale)
        self._init_scale = self.scale = scale

        super().__init__(caption=caption, resizable=False,
                         vsync=vsync, display=Display(display))

        # randomly displace the window
        pos_y = randint(0, max(self.screen.height - self.height, 0))
        pos_x = randint(0, max(self.screen.width - self.width, 0))
        self.set_location(self.screen.x + pos_x, self.screen.y + pos_y)

    @property
    def window(self):
        return self

    @property
    def isopen(self):
        return not self._was_closed

    def close(self):
        if not sys.is_finalizing():
            super().close()

    def on_close(self):
        super().on_close()
        self.close()

    def on_key_press(self, symbol, modifiers):
        if not hasattr(self, 'texture') or symbol not in {
            key.PLUS, key.MINUS, key.EQUAL, key.UNDERSCORE, key._0
        }:
            return super().on_key_press(symbol, modifiers)

        # zoom in on plus
        if symbol in (key.PLUS, key.EQUAL):
            self.scale = self.scale[0] + 1, self.scale[1] + 1

        # zoom in out on minus
        elif symbol in (key.MINUS, key.UNDERSCORE):
            self.scale = max(1, self.scale[0] - 1), max(1, self.scale[1] - 1)

        # reset zoom on numeric zero
        elif symbol == key._0:
            self.scale = self._init_scale

        sw, sh = self.scale
        self.set_size(self.texture.width * sw, self.texture.height * sh)

    def on_draw(self):
        self.clear()

        if hasattr(self, 'texture'):
            # pixelart-like nearest neighbour magnification
            gl.glTexParameteri(
                gl.GL_TEXTURE_2D,
                gl.GL_TEXTURE_MAG_FILTER,
                gl.GL_NEAREST
            )

            # transparent against black bg (by zeroing the factor)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendEquation(gl.GL_FUNC_ADD)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ZERO)

            # blit onto the buffer resizing to the current window's dims
            self.texture.blit(0, 0, width=self.width, height=self.height)

    def set_size(self, width, height):
        # call on_resize to correctly adjust the gl viewport
        super().set_size(width, height)
        if not self.resizeable:  # tom-eyy-teu, tom-ahh-teu
            super().on_resize(width, height)  # call parent's method

    def imshow(self, data):
        if not self.isopen:
            return False

        assert data.dtype == np.uint8

        height, width, *channels = data.shape
        assert len(channels) == 1 and channels[0] in (3, 4)
        format = 'RGBA' if channels[0] == 4 else 'RGB'

        # switch to our GL context and create the texture
        self.switch_to()
        self.texture = ImageData(
            width, height, format, data.tobytes(),
            pitch=-width * len(format)).get_texture()

        # resize the window to the array's dims
        sw, sh = self.scale
        self.set_size(self.texture.width * sw, self.texture.height * sh)

        # handle os events and redraw
        self.dispatch_events()
        self.on_draw()
        self.flip()

        return self.isopen
