import pickle

from sklearn.utils import shuffle


class StateDataset:
    def __init__(self, file=None, dataset=None):
        assert (file is not None or dataset is not None)

        if file is not None:
            file = open(file, 'rb')
            data = pickle.load(file)

            self.X, self.y = shuffle(data['X'], data['y'])

        if dataset is not None:
            self.X, self.y = dataset

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
