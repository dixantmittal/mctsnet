import torch
from torch.distributions import Categorical

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Node:
    def __init__(self, state, embedding):
        self.state = state
        self.embedding = embedding
        self.children = {}


class Tree(torch.nn.Module):
    def __init__(self, simulator, f_embedding, f_policy, f_backup, f_readout):
        super().__init__()

        # initialise networks
        self.f_embedding = f_embedding
        self.f_policy = f_policy
        self.f_backup = f_backup
        self.f_readout = f_readout

        # Simulator needs to follow gym env template
        self.simulator = simulator

    def search(self, state, n_simulations, training=False):
        predictions = []
        embeddings = []
        actions = []

        root = Node(state, self.f_embedding(state))
        for i in range(n_simulations):
            node = root
            path = []
            embeddings_m = []
            actions_m = []

            # Start simulation and add a new child
            while True:
                # Choose which branch to explore/exploit based on embedding memory
                logits = self.f_policy(node.embedding)
                action = Categorical(logits=logits).sample().item()

                # store embedding and action for policy gradient
                embeddings_m.append(node.embedding.detach())
                actions_m.append(action)

                # simulate with action to get next state
                next_state, reward, _, _ = self.simulator.step(action, node.state)
                path.append((node, action, reward))

                # keep on traversing if the child exists
                if node.children.get(action) is None:
                    break
                else:
                    node = node.children[action]

            # add new child
            node.children[action] = Node(next_state, self.f_embedding(next_state))

            # backup values through the path to root
            for node, action, reward in reversed(path):
                node.embedding = self.f_backup(node.embedding, node.children[action].embedding, action, reward)

            # store predictions after m_th step
            predictions.append(self.f_readout(root.embedding))

            # store embeddings and action for the m_th step
            embeddings.append(torch.stack(embeddings_m).to(device))
            actions.append(torch.tensor(actions_m).long().to(device))

        # if training mode is on, then return all predictions and embeddings/actions
        if training:
            return torch.stack(predictions), embeddings, actions
        else:
            return predictions[-1]
