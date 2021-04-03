import time

from . import RandomDiscoMaze

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


env = RandomDiscoMaze(15, 15, n_targets=1, n_colors=5)


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
