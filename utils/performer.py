from environment import SIMULATOR

MAX_EPISODE_LENGTH = 40


def performer(solver, args, render=False):
    my_simulator = SIMULATOR()
    state = my_simulator.reset()
    episode_reward = 0

    if render:
        my_simulator.render()

    for i in range(MAX_EPISODE_LENGTH):
        action, _ = solver.search(state, args)

        state, reward, terminal = my_simulator.step(action)

        if render:
            print(SIMULATOR.ACTIONS[action], reward)
            my_simulator.render()

        episode_reward += reward
        if terminal:
            break

    return episode_reward
