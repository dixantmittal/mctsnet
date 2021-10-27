import torch as t

from environment import SIMULATOR
from modules.commons import hidden_size, d_memory


# Output the policy at the end of the search
class Readout(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = t.nn.Sequential(t.nn.Linear(d_memory, hidden_size), t.nn.ReLU(),
                                       t.nn.Linear(hidden_size, hidden_size), t.nn.ReLU(),
                                       t.nn.Linear(hidden_size, SIMULATOR.n_actions))

    def forward(self, x):
        return self.readout(x)
