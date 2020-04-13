import torch as t
import torch.nn as nn

from environment import SIMULATOR
from modules.commons import hidden_size, d_memory, ResidualLinear


# Updates the embedding based on child's new embedding and current node's old embedding
class Backup(nn.Module):
    def __init__(self):
        super().__init__()

        self.memory = nn.Linear(d_memory, hidden_size)
        self.child_memory = nn.Linear(d_memory, hidden_size)
        self.action = nn.Linear(SIMULATOR.n_actions, hidden_size)
        self.reward = nn.Linear(1, hidden_size)

        d_input = 4 * hidden_size

        self.forget = nn.Sequential(nn.Linear(d_input, hidden_size), nn.ReLU(),
                                    ResidualLinear(hidden_size),
                                    ResidualLinear(hidden_size),
                                    nn.Linear(hidden_size, d_memory))

        self.update = nn.Sequential(nn.Linear(d_input, hidden_size), nn.ReLU(),
                                    ResidualLinear(hidden_size),
                                    ResidualLinear(hidden_size),
                                    nn.Linear(hidden_size, d_memory))

        self.info = nn.Sequential(nn.Linear(d_input, hidden_size), nn.ReLU(),
                                  ResidualLinear(hidden_size),
                                  ResidualLinear(hidden_size),
                                  nn.Linear(hidden_size, d_memory))

    def forward(self, memory, child_memory, action, reward):
        e_memory = t.relu(self.memory(memory))
        e_child_memory = t.relu(self.child_memory(child_memory))
        e_action = t.relu(self.action(action))
        e_reward = t.relu(self.reward(reward))

        phi = t.cat((e_memory, e_child_memory, e_action, e_reward), dim=0)

        forget = t.sigmoid(self.forget(phi))
        update = t.tanh(self.update(phi))
        info = t.relu(self.info(phi))

        return memory * forget + update * info
