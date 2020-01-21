import numpy as np

from environment import ENVIRONMENT


class Node:
    def __init__(self, state, terminal=False):
        self.state = state

        self.children = {}
        self.N = ENVIRONMENT.n_actions
        self.N_a = np.ones(ENVIRONMENT.n_actions)
        self.Q = np.zeros(ENVIRONMENT.n_actions)

        if not terminal:
            for i in range(ENVIRONMENT.n_actions):
                self.Q[i] = ENVIRONMENT.rollout(state, i)

        self.terminal = terminal


class MCTS:
    c = 100

    @staticmethod
    def search(state, n_simulations, root):
        if root is None:
            root = Node(state)

        for i in range(n_simulations):
            node = root
            path = []
            # Start simulation and add a new child
            terminal = False
            while not terminal:
                # Choose which branch to explore/exploit based on embedding memory
                Q = node.Q + MCTS.c * np.sqrt(np.log(node.N) / node.N_a)
                action = int(np.argmax(Q))

                # simulate with action to get next state
                state, reward, terminal = ENVIRONMENT.simulate(node.state, action)

                path.append((node, action, reward))

                # keep on traversing if the child exists
                if node.children.get(action) is None:
                    # add new child
                    node.children[action] = Node(state, terminal)
                    break
                else:
                    node = node.children[action]

            # backup values through the path to root
            for node, action, reward in reversed(path):
                node.N += 1
                node.N_a[action] += 1
                node.Q[action] = node.Q[action] + (reward + np.max(node.children[action].Q) - node.Q[action]) / node.N_a[action]

        return int(np.argmax(root.Q)), root
