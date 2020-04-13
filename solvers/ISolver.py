import torch.nn as nn


class ISolver(nn.Module):
    def search(self, belief, args):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass
