class Base:
    def __init__(self):
        self.state = None

    def step(self, action):
        self.state, r, t = self.simulate(self.state, action)
        return self.state, r, t

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    @staticmethod
    def simulate(state, action):
        raise NotImplementedError

    @staticmethod
    def rollout(state, action, use_heuristics=True):
        raise NotImplementedError

    @staticmethod
    def tensor_shape():
        raise NotImplementedError

    @staticmethod
    def state_to_tensor(state):
        raise NotImplementedError
