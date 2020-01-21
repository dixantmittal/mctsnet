import torch
import torch.nn as nn

hidden_size = 128


class DenseBlock(nn.Module):
    def __init__(self, n_blocks, layer_size):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(layer_size, layer_size) for i in range(n_blocks)])

    def forward(self, x):
        outputs = [x]
        for linear in self.linear:
            outputs.append(torch.relu(linear(x)))
            x = torch.sum(torch.stack(outputs), dim=0)

        return x
