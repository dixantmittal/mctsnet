import torch as t
import torch.nn as nn

from environment import SIMULATOR
from modules.commons import ResidualLinear, hidden_size, d_memory


# Guides the search in the POMCP tree
class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.memory = nn.Linear(d_memory, hidden_size)
        self.children_memory = nn.Linear(d_memory * SIMULATOR.n_actions, hidden_size)

        self.policy = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(),
                                    ResidualLinear(hidden_size),
                                    ResidualLinear(hidden_size),
                                    nn.Linear(hidden_size, SIMULATOR.n_actions))

    def forward(self, node):
        memory = t.relu(self.memory(node.tensors.memory))
        children = t.relu(self.children_memory(t.cat(node.tensors.children)))

        x = t.cat([memory, children], dim=0)

        return self.policy(x)
