import time
import argparse

from . import RandomDiscoMaze

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

KEYMAP = dict(zip(map(ord, 'asdw'), range(4)))

human_agent_action = None
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key == 0xFF1B:
        human_wants_restart = True

    if key == 32:
        human_sets_pause = not human_sets_pause

    if key in KEYMAP:
        human_agent_action = KEYMAP[key]


def key_release(key, mod):
    global human_agent_action

    if key in KEYMAP:
        if human_agent_action == KEYMAP[key]:
            human_agent_action = None


env = RandomDiscoMaze(args.n_row, args.n_col,
                      n_targets=args.n_targets,
                      n_colors=args.n_colors,
                      field=(2, 2) if args.partial else None)
env.seed(args.seed)

env.render(mode='human')
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause

    total_reward = 0
    human_wants_restart, is_terminal = False, False

    obs = env.reset()
    while not (human_wants_restart or is_terminal):
        obs, rew, is_terminal, info = env.step(human_agent_action)
        if rew != 0:
            print("reward %0.3f" % rew)

        if is_terminal:  # pause on termination
            human_sets_pause = True

        total_reward += rew
        while True:
            if not env.render(mode='human'):
                return False

            time.sleep(0.05)
            if not human_sets_pause:
                break

    print("reward %0.2f" % (total_reward))
    return True


while rollout(env):
    pass
