from torch import nn as nn

from environment import ENVIRONMENT
from modules.common import DenseBlock, hidden_size


# Output the policy at the end of the search
class Readout(nn.Module):
    def __init__(self, d_memory):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(d_memory, hidden_size), nn.ReLU(),
                                    DenseBlock(1, hidden_size),
                                    nn.Linear(hidden_size, ENVIRONMENT.n_actions))

    def forward(self, x):
        return self.layers(x)
