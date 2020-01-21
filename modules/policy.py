from torch import nn as nn

from environment import ENVIRONMENT
from modules.common import DenseBlock, hidden_size


# Guides the search in the MCTSnet tree
class Policy(nn.Module):
    def __init__(self, d_memory):
        super().__init__()
        d_input = d_memory + d_memory * ENVIRONMENT.n_actions
        self.layers = nn.Sequential(nn.Linear(d_input, hidden_size), nn.ReLU(),
                                    DenseBlock(1, hidden_size),
                                    nn.Linear(hidden_size, ENVIRONMENT.n_actions))

    def forward(self, x):
        return self.layers(x)
