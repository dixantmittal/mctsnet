from environment import ENVIRONMENT

MAX_EPISODE_LENGTH = 10


def check_performance(solver, n_simulations):
    my_simulator = ENVIRONMENT()
    state = my_simulator.reset()
    episode_reward = 0

    for i in range(MAX_EPISODE_LENGTH):
        action = solver.search(state, n_simulations)
        state, reward, terminal = my_simulator.step(action)

        episode_reward += reward

        if terminal:
            break

    return episode_reward
