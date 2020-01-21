import torch
from recordclass import recordclass
from torch.distributions import Categorical

from environment import ENVIRONMENT

Variable = recordclass('Variable', ['state', 'children'])
Tensor = recordclass('Tensor', ['state', 'children', 'memory'])
Node = recordclass('Node', ['variables', 'tensors'])


class MCTSnet(torch.nn.Module):
    def __init__(self, f_memory, f_policy, f_backup, f_readout):
        super().__init__()

        # initialise networks
        self.f_memory = f_memory
        self.f_policy = f_policy
        self.f_backup = f_backup
        self.f_readout = f_readout

    def new_node(self, state):
        variables = Variable(state=state, children={})
        state_tensor = ENVIRONMENT.state_to_tensor(state)
        memory = self.f_memory(state_tensor)
        tensors = Tensor(state=state_tensor,
                         children=[torch.zeros_like(memory)] * ENVIRONMENT.n_actions,
                         memory=memory)

        return Node(variables, tensors)

    def search(self, state, n_simulations, training=False):
        predictions = []
        logits = []
        actions = []

        root = self.new_node(state)

        for i in range(n_simulations):
            node = root

            path = []
            logits_m = []
            actions_m = []

            # Start simulation and add a new child
            while True:
                # Choose which branch to explore/exploit based on node's memory
                memory_state = torch.cat([node.tensors.memory] + node.tensors.children, dim=0)
                p_action = self.f_policy(memory_state)
                action = Categorical(logits=p_action).sample().item()

                # store embedding and action for policy gradient
                if training:
                    logits_m.append(p_action)
                    actions_m.append(action)

                # simulate with action to get next state
                next_state, reward, _ = ENVIRONMENT.simulate(node.variables.state, action)
                path.append((node, action, reward))

                # Traverse to find a leaf node
                if node.variables.children.get(action) is None:
                    # add new child
                    node.variables.children[action] = self.new_node(next_state)
                    break
                else:
                    node = node.variables.children[action]

            # backup values through the path to root
            for node, action, reward in reversed(path):
                node.tensors.memory = self.f_backup(node.tensors.memory, node.variables.children[action].tensors.memory, action, reward)

                node.tensors.children[action] = node.variables.children[action].tensors.memory

            # store predictions after m_th step
            predictions.append(self.f_readout(root.tensors.memory))

            # store logits and action for the m_th step
            logits.append(logits_m)
            actions.append(actions_m)

        # if training mode is on, then return all predictions and embeddings/actions
        if training:
            return predictions, logits, actions
        else:
            return predictions[-1].argmax().item()

    def save(self, file):
        if file == '':
            return

        torch.save(self.state_dict(), file)

    def load(self, file):
        if file == '':
            return

        self.load_state_dict(torch.load(file))
