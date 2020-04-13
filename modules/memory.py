import torch.nn as nn

from environment import SIMULATOR
from modules.commons import d_memory, ResidualConv


# Initialises a leaf node with memory
class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        channels, _, _ = SIMULATOR.tensor_shape()
        self.memory = nn.Sequential(nn.Conv2d(channels, 64, 1, 1),
                                    ResidualConv(64),
                                    ResidualConv(64),
                                    nn.Conv2d(64, 128, 1, 1),
                                    nn.AdaptiveMaxPool2d((1, 1)),
                                    nn.Flatten(),
                                    nn.Linear(128, d_memory))

    def forward(self, x):
        return self.memory(x.unsqueeze(0)).squeeze()
