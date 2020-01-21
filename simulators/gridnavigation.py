import random
from collections import namedtuple
from copy import deepcopy

import numpy as np
import torch

from simulators.base import Base
from utils.basic import to_one_hot, Vector, manhattan

State = namedtuple('State', 'current goal')

tensor_cache = {}


class GridNavigation(Base):
    n_actions = 4

    MAP_SIZE = 5

    obstacles = [Vector(1, 1), Vector(3, 3), Vector(2, 4)]

    # Actions
    ACTIONS_UP = 0
    ACTIONS_DOWN = 1
    ACTIONS_LEFT = 2
    ACTIONS_RIGHT = 3
    ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    # Rewards
    ILLEGAL_ACTION_REWARD = -100
    EXIT_REWARD = 10
    MOVE_REWARD = -1

    def __init__(self):
        super().__init__()

    def reset(self):
        current = Vector(random.randint(0, GridNavigation.MAP_SIZE - 1), random.randint(0, GridNavigation.MAP_SIZE - 1))
        while current in GridNavigation.obstacles:
            current = Vector(random.randint(0, GridNavigation.MAP_SIZE - 1), random.randint(0, GridNavigation.MAP_SIZE - 1))

        goal = Vector(random.randint(0, GridNavigation.MAP_SIZE - 1), random.randint(0, GridNavigation.MAP_SIZE - 1))
        while goal in GridNavigation.obstacles:
            goal = Vector(random.randint(0, GridNavigation.MAP_SIZE - 1), random.randint(0, GridNavigation.MAP_SIZE - 1))

        self.state = State(current, goal)

        return State(current, goal)

    def render(self):
        string = np.array([['~'] * GridNavigation.MAP_SIZE] * GridNavigation.MAP_SIZE)

        for obstacle in GridNavigation.obstacles:
            x, y = obstacle
            string[y][x] = 'O'

        current, goal = self.state

        string[current.y][current.x] = 'C'
        string[goal.y][goal.x] = 'G'

        print()
        for i in range(GridNavigation.MAP_SIZE):
            for j in range(GridNavigation.MAP_SIZE):
                print(string[i][j], end='')
            print()

    @staticmethod
    def simulate(state, action):
        current, goal = deepcopy(state)

        reward = GridNavigation.MOVE_REWARD

        if action == GridNavigation.ACTIONS_UP:
            if current.y == 0:
                reward = GridNavigation.ILLEGAL_ACTION_REWARD
            current.y = max(current.y - 1, 0)

        elif action == GridNavigation.ACTIONS_DOWN:
            if current.y == GridNavigation.MAP_SIZE - 1:
                reward = GridNavigation.ILLEGAL_ACTION_REWARD
            current.y = min(current.y + 1, GridNavigation.MAP_SIZE - 1)

        elif action == GridNavigation.ACTIONS_LEFT:
            if current.x == 0:
                reward = GridNavigation.ILLEGAL_ACTION_REWARD
            current.x = max(current.x - 1, 0)

        elif action == GridNavigation.ACTIONS_RIGHT:
            if current.x == GridNavigation.MAP_SIZE - 1:
                reward = GridNavigation.ILLEGAL_ACTION_REWARD
            current.x = min(current.x + 1, GridNavigation.MAP_SIZE - 1)

        if current == goal:
            reward = GridNavigation.EXIT_REWARD
            terminal = True
        else:
            terminal = False

        if current in GridNavigation.obstacles:
            reward = GridNavigation.ILLEGAL_ACTION_REWARD

        return State(current, goal), reward, terminal

    @staticmethod
    def rollout(state, action, use_heuristics=True):
        (current, goal), reward, terminal = GridNavigation.simulate(state, action)
        return reward + (not terminal) * GridNavigation.MOVE_REWARD * manhattan(current, goal)

    @staticmethod
    def tensor_shape():
        return GridNavigation.MAP_SIZE * 4

    @staticmethod
    def state_to_tensor(state):
        return tensor_cache[str(state)]


for i in range(GridNavigation.MAP_SIZE):
    for j in range(GridNavigation.MAP_SIZE):
        current = Vector(i, j)
        for a in range(GridNavigation.MAP_SIZE):
            for b in range(GridNavigation.MAP_SIZE):
                goal = Vector(a, b)

                tensor = [to_one_hot(i, GridNavigation.MAP_SIZE),
                          to_one_hot(j, GridNavigation.MAP_SIZE),
                          to_one_hot(a, GridNavigation.MAP_SIZE),
                          to_one_hot(b, GridNavigation.MAP_SIZE)]

                tensor = torch.cat(tensor, dim=0)

                tensor_cache[str(State(current, goal))] = tensor
