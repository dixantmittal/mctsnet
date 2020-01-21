from math import sqrt

import torch
from recordclass import recordclass

tensor_cache = {}

Vector = recordclass('Vector', 'x y')


def manhattan(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)


def euclidean(a, b):
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def to_one_hot(index, n_classes):
    return tensor_cache['{}_{}'.format(index, n_classes)]


for n_classes in range(1, 20):
    for i in range(n_classes):
        onehot = torch.zeros(n_classes)
        onehot.scatter_(0, torch.tensor(i), 1)
        tensor_cache['{}_{}'.format(i, n_classes)] = onehot
