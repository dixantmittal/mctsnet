from collections import namedtuple
from copy import deepcopy

import numpy as np
import torch as t
from recordclass import recordclass

from simulators.base import BaseSimulator
from utils.basic import manhattan

Vector = recordclass('Vector', 'x y')
State = namedtuple('State', 'rover rocks quality')


class RockSample(BaseSimulator):
    MAP_SIZE = 7
    ROCKS = [Vector(1, 3), Vector(2, 2), Vector(3, 4), Vector(5, 5)]
    NUM_OF_ROCKS = len(ROCKS)

    n_actions = 5

    # Actions
    ACTIONS_UP = 0
    ACTIONS_DOWN = 1
    ACTIONS_LEFT = 2
    ACTIONS_RIGHT = 3
    ACTIONS_SAMPLE = 4

    ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SAMPLE']

    ILLEGAL_ACTION_REWARD = -100
    BAD_ROCK_REWARD = -10
    GOOD_ROCK_REWARD = 10
    EXIT_REWARD = 10
    MOVE_REWARD = 0

    def __init__(self):
        super().__init__()

    def reset(self):
        # fixed position
        rocks = deepcopy(RockSample.ROCKS)
        rover = Vector(0, RockSample.MAP_SIZE // 2)

        # # random start position
        # rover.x = np.random.RandomState().randint(0, RockSample.MAP_SIZE)
        # rover.y = np.random.RandomState().randint(0, RockSample.MAP_SIZE)

        quality = np.random.RandomState().binomial(1, 0.5, RockSample.NUM_OF_ROCKS).tolist()

        self.state = State(rover, rocks, quality)

        return deepcopy(self.state)

    def render(self):
        string = np.asarray([' '] * (RockSample.MAP_SIZE + 1) * RockSample.MAP_SIZE).reshape(RockSample.MAP_SIZE, RockSample.MAP_SIZE + 1)

        for rock, quality in zip(self.state.rocks, self.state.quality):
            string[rock[1], rock[0]] = quality

        for i in range(RockSample.MAP_SIZE):
            string[i, RockSample.MAP_SIZE] = '~'

        string[self.state.rover[1], self.state.rover[0]] = 'R'

        print(string)

    @staticmethod
    def simulate(state, action):
        rover, rocks, quality = deepcopy(state)

        reward = RockSample.MOVE_REWARD
        terminal = False

        # navigation actions
        if action == RockSample.ACTIONS_UP:
            if rover.y == 0:
                reward = RockSample.ILLEGAL_ACTION_REWARD
            rover.y = max(rover.y - 1, 0)

        elif action == RockSample.ACTIONS_DOWN:
            if rover.y == RockSample.MAP_SIZE - 1:
                reward = RockSample.ILLEGAL_ACTION_REWARD
            rover.y = min(rover.y + 1, RockSample.MAP_SIZE - 1)

        elif action == RockSample.ACTIONS_LEFT:
            if rover.x == 0:
                reward = RockSample.ILLEGAL_ACTION_REWARD
            rover.x = max(rover.x - 1, 0)

        elif action == RockSample.ACTIONS_RIGHT:
            if rover.x >= RockSample.MAP_SIZE - 1:
                reward = RockSample.EXIT_REWARD
                terminal = True
            rover.x = min(rover.x + 1, RockSample.MAP_SIZE)

        # Sample
        elif action == RockSample.ACTIONS_SAMPLE:
            # if there is rock at sampling place, return quality
            if rover in rocks:
                idx = rocks.index(rover)
                observation = quality[idx]

                reward = RockSample.GOOD_ROCK_REWARD * observation + RockSample.BAD_ROCK_REWARD * (1 - observation)

                quality[idx] = 0

            else:
                reward = RockSample.ILLEGAL_ACTION_REWARD

        return State(rover, rocks, quality), reward, terminal

    @staticmethod
    def rollout(state, action, use_heuristics=True):
        state, _, _ = RockSample.simulate(state, action)
        reward = 0.99 ** (RockSample.MAP_SIZE - state.rover.x) * RockSample.EXIT_REWARD

        if use_heuristics:
            for idx, rock in enumerate(state.rocks):
                reward = reward + state.quality[idx] * RockSample.GOOD_ROCK_REWARD / max(manhattan(rock, state.rover), 1)
        return reward

    @staticmethod
    def tensor_shape():
        return 3, RockSample.MAP_SIZE, RockSample.MAP_SIZE + 1

    @staticmethod
    def state_to_tensor(state):
        rover, rocks, qualities = state

        # 0 for rover position, 1 for rock position, 2 for quality
        tensor = t.zeros((3, RockSample.MAP_SIZE, RockSample.MAP_SIZE + 1))

        tensor[0, rover.y, rover.x] = 1
        for rock, quality in zip(rocks, qualities):
            tensor[1, rock.y, rock.x] = 1
            tensor[2, rock.y, rock.x] = (1 if quality == 1 else -1)

        return t.constant_pad_nd(tensor, (1, 1, 1, 1), value=-1)
