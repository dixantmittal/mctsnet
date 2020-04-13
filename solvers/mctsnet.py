from collections import namedtuple

import torch as t
from recordclass import recordclass
from torch.distributions import Categorical

from device import Device
from environment import SIMULATOR
from modules import Memory, Policy, Backup, Readout
from solvers.ISolver import ISolver
from utils.prepare_input import prepare_input_for_f_backup

Path = namedtuple('Path', ['node', 'action', 'reward'])
Variables = recordclass('Variables', ['state', 'children'])
Tensors = recordclass('Tensors', ['memory', 'children'])
Node = recordclass('Node', ['variables', 'tensors'])


class MCTSnet(ISolver):
    def __init__(self):
        super().__init__()

        # initialise networks
        self.f_memory = Memory()
        self.f_policy = Policy()
        self.f_backup = Backup()
        self.f_readout = Readout()

        self.tensor_cache = {}

    def state_to_tensor(self, state):
        key = str(state)
        tensor = self.tensor_cache.get(key)
        if tensor is None:
            tensor = SIMULATOR.state_to_tensor(state).to(Device.get_device())
            self.tensor_cache[key] = tensor

        return tensor

    def new_node(self, state):
        variables = Variables(state=state, children={})

        tensor_memory = self.f_memory(self.state_to_tensor(state))

        tensors = Tensors(memory=tensor_memory,
                          children=[t.zeros_like(tensor_memory) for i in range(SIMULATOR.n_actions)])

        return Node(variables=variables, tensors=tensors)

    def search(self, state, args):

        root = self.new_node(state)

        predictions = [self.f_readout(root.tensors.memory)]
        logits = []
        actions = []

        for i in range(args.n_simulations):

            node = root

            path = []
            logits_m = []
            actions_m = []

            # Start simulation and add a new child
            terminal = False
            while not terminal:
                # Choose which branch to explore/exploit based on node memory
                p_actions = self.f_policy(node)
                action = Categorical(logits=p_actions).sample().item()

                # store embedding and action for policy gradient
                if args.training:
                    logits_m.append(p_actions)
                    actions_m.append(action)

                # simulate with action to get next state
                next_state, reward, terminal = SIMULATOR.simulate(node.variables.state, action)
                path.append(Path(node, action, reward))

                # if action and observation branch exists, traverse to the next node and add the new state
                if node.variables.children.get(action) is None:
                    node.variables.children[action] = self.new_node(next_state)
                    break
                # else, create one
                else:
                    node = node.variables.children[action]

            # backup values through the path to root
            for node, action, reward in reversed(path):
                node.tensors.memory = self.f_backup(*prepare_input_for_f_backup(node, action, reward))

                node.tensors.children[action] = node.variables.children[action].tensors.memory

            # store predictions after m_th step
            predictions.append(self.f_readout(root.tensors.memory))

            # store logits and action for the m_th step
            logits.append(logits_m)
            actions.append(actions_m)

        return Categorical(logits=predictions[-1]).sample().item(), (predictions, logits, actions)

    def save(self, file):
        if file is None or file == '':
            return

        t.save(self.state_dict(), file)

    def load(self, file):
        if file is None or file == '':
            return

        self.load_state_dict(t.load(file, map_location='cpu'))
