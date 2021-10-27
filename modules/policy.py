import torch as t
import torch.nn as nn

from environment import SIMULATOR
from modules.commons import hidden_size, d_memory


# Guides the search in the search tree
class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.f_policy = t.nn.Sequential(t.nn.Linear(d_memory, hidden_size), t.nn.ReLU(),
                                        t.nn.Linear(hidden_size, hidden_size), t.nn.ReLU(),
                                        t.nn.Linear(hidden_size, SIMULATOR.n_actions))

    def forward(self, memory):
        return self.f_policy(memory)
