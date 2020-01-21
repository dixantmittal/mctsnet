import torch
from torch import nn as nn

from environment import ENVIRONMENT
from modules.common import DenseBlock, hidden_size
from utils.basic import to_one_hot


# Updates the memory based on child's updated memory and node's current memory
class Backup(nn.Module):
    def __init__(self, d_memory):
        super().__init__()
        d_input = d_memory + d_memory + ENVIRONMENT.n_actions + 1  # 1 for reward
        self.transform = nn.Sequential(nn.Linear(d_input, hidden_size), nn.ReLU(),
                                       DenseBlock(1, hidden_size),
                                       nn.Linear(hidden_size, d_memory))
        self.forget = nn.Linear(d_input, d_memory)
        self.add = nn.Linear(d_input, d_memory)

    def forward(self, current_memory, child_memory, action, reward):
        action = to_one_hot(action, ENVIRONMENT.n_actions)
        reward = torch.tensor([reward]).float()
        phi = torch.cat((current_memory, child_memory, action, reward), dim=0)

        transform = self.transform(phi)
        forget = torch.sigmoid(self.forget(phi))
        add = torch.tanh(self.add(phi))

        return current_memory * forget + transform * add
