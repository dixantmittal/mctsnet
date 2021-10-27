import os

import torch as t


class ISolver(t.nn.Module):
    def search(self, belief, args):
        raise NotImplementedError

    def save(self, file):
        if file is None or file == '':
            print('File name empty!!')
            return

        t.save(self.state_dict(), file)

    def load(self, file):
        if file is None or file == '':
            print('File name empty!!')
            return
        if not os.path.exists(file):
            print('File does not exist!!')
            return

        self.load_state_dict(t.load(file, map_location='cpu'))
