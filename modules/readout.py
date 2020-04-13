import torch.nn as nn

from environment import SIMULATOR
from modules.commons import hidden_size, d_memory, ResidualLinear


# Output the policy at the end of the search
class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(d_memory, hidden_size),
                                     nn.ReLU(),
                                     ResidualLinear(hidden_size),
                                     ResidualLinear(hidden_size),
                                     nn.Linear(hidden_size, SIMULATOR.n_actions))

    def forward(self, x):
        return self.readout(x)
