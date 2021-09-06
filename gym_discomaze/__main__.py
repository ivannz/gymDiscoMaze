import time
import argparse

from pyglet.window import key, Window

from . import RandomDiscoMaze


# global state controlled by kbd handlers, consumed by `rollout`
class SimpleUIControl:
    """A bare-bones keyboard event handelr for pyglet UI."""
    action, pause, restart, waiting = None, False, False, False

    def __init__(self, keymap):
        self.KEYMAP = keymap

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ENTER:
            self.restart = True
            return

        if symbol == key.SPACE:
            self.pause = not self.pause
            return

        if symbol in self.KEYMAP and not self.waiting:
            self.action, self.waiting = symbol, True
            return

    def on_key_release(self, symbol, modifiers):
        if symbol in self.KEYMAP and self.waiting:
            self.action, self.waiting = None, False
            return

    def register(self, window):
        assert isinstance(window, Window)

        window.push_handlers(
            self.on_key_press,
            self.on_key_release,
        )

        return self


parser = argparse.ArgumentParser(
    description='Play the Random Disco Maze.',
    add_help=True)

parser.add_argument(
    '--n_row', type=int, required=False, default=15,
    help='the number of rows in the maze.')

parser.add_argument(
    '--n_col', type=int, required=False, default=15,
    help='the number of columns in the maze.')

parser.add_argument(
    '--n_colors', type=int, required=False, default=5,
    help='the number of colours in the palette to randomly paint the walls.')

parser.add_argument(
    '--n_targets', type=int, required=False, default=1,
    help='the number of colours in the palette to randomly paint the walls.')

parser.add_argument(
    '--partial', required=False, dest='partial', action='store_true',
    help='Limit the observable field to 5x5.')

parser.add_argument(
    '--seed', required=False, default=None,
    help='PRNG seed to use.')

parser.set_defaults(n_row=15, n_col=15, n_colors=5, n_targets=5,
                    partial=False, seed=None)

args, _ = parser.parse_known_args()
print(vars(args))

env = RandomDiscoMaze(args.n_row, args.n_col,
                      n_targets=args.n_targets,
                      n_colors=args.n_colors,
                      field=(2, 2) if args.partial else None)

env.seed(args.seed)

# print key bindings
KEYMAP = dict(zip([None, key.A, key.S, key.D, key.W],
                  ['stay', 'west', 'south', 'east', 'north']))
print({chr(k): n for k, n in KEYMAP.items() if k is not None})

ctrl = SimpleUIControl(KEYMAP)

env.render(mode='human')  # sets up the viewer gui, so that the next line works
ctrl.register(env.unwrapped.viewer.window)


def rollout(env, ctrl):
    total_reward = 0
    ctrl.restart, is_terminal = False, False

    obs = env.reset()
    while not (ctrl.restart or is_terminal):
        act = env.named_actions[KEYMAP[ctrl.action]]
        ctrl.action = None  # avoid sticky actions

        obs, rew, is_terminal, info = env.step(act)
        if rew != 0:
            print("reward %0.3f" % rew)

        if is_terminal:  # pause on termination
            ctrl.pause = True

        total_reward += rew
        # rendering and ui event loop
        while env.render(mode='human'):
            time.sleep(0.04)
            if not ctrl.pause:
                break

        else:
            return False

    print("reward %0.2f" % (total_reward))
    return True


while rollout(env, ctrl):
    pass
