from torch import nn as nn

from environment import ENVIRONMENT
from modules.common import DenseBlock, hidden_size


# Initialises a leaf node with memory
class Memory(nn.Module):
    def __init__(self, d_memory):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(ENVIRONMENT.tensor_shape(), hidden_size), nn.ReLU(),
                                    DenseBlock(1, hidden_size),
                                    nn.Linear(hidden_size, d_memory))

    def forward(self, x):
        return self.layers(x)
