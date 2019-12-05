import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def one_hot(tensor, n_classes):
    y_onehot = torch.zeros(n_classes).to(device)

    # In your for loop
    y_onehot.scatter_(0, torch.tensor(tensor).to(device), 1)

    return y_onehot


class Block(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size)

    def forward(self, x):
        y = self.linear(x)
        y = torch.relu(y)
        return x + y


# Initialises a leaf node with an embedding
class Embedding(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.n_states = input_size

        self.sequential = nn.Sequential(nn.Linear(input_size, 512),
                                        nn.ReLU(),
                                        Block(512),
                                        nn.Linear(512, output_size))

    def forward(self, x):
        x = one_hot([x], self.n_states)
        return self.sequential(x)


# Guides the search in the Monte Carlo tree
class Policy(nn.Module):
    def __init__(self, embedding_size, output_size):
        super().__init__()
        self.sequential = nn.Sequential(nn.Linear(embedding_size, 512),
                                        nn.ReLU(),
                                        Block(512),
                                        nn.Linear(512, output_size))

    def forward(self, x):
        return self.sequential(x)


# Updates the embedding based on child's new embedding and current node's old embedding
class Backup(nn.Module):
    def __init__(self, embedding_size, n_action):
        super().__init__()

        self.sequential = nn.Sequential(nn.Linear(embedding_size * 2 + n_action + 1, 512),
                                        nn.ReLU(),
                                        Block(512),
                                        nn.Linear(512, embedding_size))
        self.gate = nn.Sequential(nn.Linear(embedding_size * 2 + n_action + 1, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 1))

        self.n_actions = n_action

    def forward(self, h, c, a, r):
        r = torch.tensor([r]).float().to(device)
        a = one_hot([a], self.n_actions)
        phi = torch.cat((h, c, a, r), dim=0)
        return h + torch.sigmoid(self.gate(phi)) * self.sequential(phi)


# Output the policy at the end of the search
class Readout(nn.Module):
    def __init__(self, embedding_size, output_size):
        super().__init__()
        self.sequential = nn.Sequential(nn.Linear(embedding_size, 512),
                                        nn.ReLU(),
                                        Block(512),
                                        nn.Linear(512, output_size))

    def forward(self, x):
        return self.sequential(x)
