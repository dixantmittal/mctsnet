import pickle


class Dataset(list):

    def __init__(self, file):
        super().__init__()
        if file is not None:
            file = open(file, 'rb')
            data = pickle.load(file)

            self.extend(data)
