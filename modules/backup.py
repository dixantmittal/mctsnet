import torch as t
import torch.nn as nn

from environment import SIMULATOR
from modules.commons import d_memory


# Updates the embedding based on child's new embedding and current node's old embedding
class Backup(nn.Module):
    def __init__(self):
        super().__init__()

        self.update = t.nn.GRUCell(input_size=d_memory + SIMULATOR.n_actions + 1,
                                   hidden_size=d_memory)

    def forward(self, memory, child_memory, action, reward):
        phi = t.cat((child_memory, action, reward), dim=0)
        return self.update(phi.unsqueeze(0), memory.unsqueeze(0)).squeeze()
