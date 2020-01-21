import random

import numpy as np


class Gridworld(object):
    def __init__(self, size):
        self.n_states = size * size
        self.n_actions = 5

        self.border = size

        self.reward = np.zeros((size, size))

        # set obstacles
        self.obstacles = [(np.random.randint(size), np.random.randint(size)) for i in range(size)]
        for o in self.obstacles:
            self.reward[o[0], o[1]] = -10

        # set goal
        self.goal = (np.random.randint(size), np.random.randint(size))
        self.reward[self.goal[0], self.goal[1]] = 10

    def reset(self):
        return np.random.randint(self.n_states)

    def step(self, action, state):
        x = state // self.border
        y = state % self.border

        if action == 0:
            x = max(x - 1, 0)
        elif action == 1:
            y = min(y + 1, self.border - 1)
        elif action == 2:
            x = min(x + 1, self.border - 1)
        elif action == 3:
            y = max(y - 1, 0)
        elif action == 4:
            pass
        else:
            return None

        return x * self.border + y, self.reward[x, y], None, None

    def solve(self):
        actions = np.zeros_like(self.reward, dtype=np.int) + 10

        x, y = self.goal

        for i in range(self.border):
            actions[i, y] = (0 if i > x else 2)

        for i in range(self.border):
            actions[x, i] = (3 if i > y else 1)

        for i in range(self.border):
            if i == x:
                continue
            for j in range(self.border):
                if j == y:
                    continue

                if i < x and j < y:
                    if self.reward[i, min(j + 1, self.border - 1)] == -10:
                        actions[i, j] = 2
                    elif self.reward[min(i + 1, self.border - 1), j] == -10:
                        actions[i, j] = 1
                    else:
                        actions[i, j] = random.choice([1, 2])
                if i < x and j > y:
                    if self.reward[min(i + 1, self.border - 1), j] == -10:
                        actions[i, j] = 3
                    elif self.reward[i, max(j - 1, 0)] == -10:
                        actions[i, j] = 2
                    else:
                        actions[i, j] = random.choice([2, 3])
                if i > x and j < y:
                    if self.reward[max(i - 1, 0), j] == -10:
                        actions[i, j] = 1
                    elif self.reward[i, min(j + 1, self.border - 1)] == -10:
                        actions[i, j] = 0
                    else:
                        actions[i, j] = random.choice([0, 1])
                if i > x and j > y:
                    if self.reward[max(i - 1, 0), j] == -10:
                        actions[i, j] = 3
                    elif self.reward[i, max(j - 1, 0)] == -10:
                        actions[i, j] = 0
                    else:
                        actions[i, j] = random.choice([0, 3])

        actions[x, y] = 4

        return actions


class Navigation1D:
    def __init__(self, size):
        self.n_states = size
        self.n_actions = 2
        self.d_states = 1

        self.reward = np.zeros(size)
        self.reward[np.random.randint(size)] = 1

    def reset(self):
        return np.random.randint(self.n_states)

    def step(self, action, state):
        x = int(state)
        if action == 0:
            next_state = max(x - 1, 0)
        elif action == 1:
            next_state = min(x + 1, self.n_states - 1)
        else:
            return None

        return next_state, self.reward[next_state], self.reward[next_state] == 1, None

    def solve(self):
        actions = np.ones(self.n_states, dtype=np.int)

        i = 0
        while i < self.n_states and self.reward[i] != 1:
            i += 1

        while i < self.n_states:
            actions[i] = 0
            i += 1

        return actions
