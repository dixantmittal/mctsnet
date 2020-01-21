import argparse

import numpy as np
from torch.utils.data import Dataset

from networks import Embedding, Policy, Backup, Readout
from simulator import Gridworld
from trainer import optimise
from tree import Tree


class GridNavigationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_simulations', dest='n_simulations', default=10, type=int, help='Number of tree simulations')
    parser.add_argument('--lr', dest='lr', default=0.0005, help='Learning rate for the training', type=float)
    parser.add_argument('--epochs', dest='epochs', default=50, type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', dest='batch_size', default=20, type=int, help='Batch size')
    parser.add_argument('--gamma', dest='gamma', default=0.5, type=int, help='Value for gamma')
    parser.add_argument('--alpha', dest='alpha', default=1, type=int, help='Value for alpha')
    parser.add_argument('--embedding_size', dest='embedding_size', default=512, type=int, help='Size of Embedding')
    args = parser.parse_args()

    simulator = Gridworld(10)

    # initialise networks
    f_embedding = Embedding(simulator.n_states, args.embedding_size)
    f_policy = Policy(args.embedding_size, simulator.n_actions)
    f_backup = Backup(args.embedding_size, simulator.n_actions)
    f_readout = Readout(args.embedding_size, simulator.n_actions)

    # initialise tree
    search_tree = Tree(simulator, f_embedding, f_policy, f_backup, f_readout)

    y = simulator.solve().reshape(-1)
    X = np.arange(len(y))

    optimise(search_tree, args.n_simulations, GridNavigationDataset(X, y), args.epochs, args.lr, args.batch_size, args.gamma, args.alpha)
