import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# For tensorboard
writer = SummaryWriter('runs/mcts_{}'.format(time.time()))


# Find loss on single sample (due to dynamic computation graph)
def loss_on_sample(tree, n_simulations, state, action, gamma=0.1, alpha=1):
    # duplicate action n_simulations times because of anytime nature of MCTS.
    action = torch.tensor([action] * n_simulations).long().to(device)

    # find the predictions, embeddings and sampled actions
    predictions, embeddings, actions = tree.search(state, n_simulations, training=True)

    # Compute cross entropy loss to train differentiable parts
    loss = F.cross_entropy(predictions, action)

    # Compute decrease in loss after each simulation
    r_m = F.cross_entropy(predictions, action, reduction='none').detach()
    r_m[1:] = r_m[:-1] - r_m[1:]
    r_m[0] = -r_m[0]

    # compute geometric sum for difference in loss
    for i in reversed(range(0, n_simulations - 1)):
        r_m[i] = r_m[i] + gamma * r_m[i + 1]

    for l_m, embedding_m, action_m in zip(r_m, embeddings, actions):
        # find logits
        logits = tree.f_policy(embedding_m)

        # calculate entropy for regularisation
        entropy = -torch.sum(F.softmax(logits.detach(), dim=1) * F.log_softmax(logits.detach(), dim=1))

        # find negative likelihood to minimise
        negative_likelihood = F.cross_entropy(logits, action_m, reduction='sum') * (l_m + alpha * entropy)

        # add it to loss
        loss = loss + negative_likelihood

    return loss, predictions[-1].argmax().item()


# optimise the model
def optimise(tree, n_simulations, dataset, epochs, lr, batch_size=1, gamma=0.1, alpha=1):
    optimiser = torch.optim.Adam(params=tree.parameters(), lr=lr)
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    itr = 0
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        tree.to(device)
        tree.train()

        for states, actions in tqdm(data):
            batch_loss = []
            predictions = []

            # iterate through batch and compute batch loss
            for sample in zip(states, actions):
                state, action = sample

                loss, prediction = loss_on_sample(tree, n_simulations, state, action, gamma, alpha)

                batch_loss.append(loss)
                predictions.append(prediction)

            # Backward pass
            batch_loss = torch.stack(batch_loss).mean()
            optimiser.zero_grad()

            # compute gradients
            batch_loss.backward()

            # clip gradient to prevent exploding gradients
            clip_grad_norm_(tree.parameters(), 5)

            # update the model
            optimiser.step()

            # write to tensorboard
            writer.add_scalar('loss', batch_loss.item(), itr)
            writer.add_scalar('accuracy', accuracy_score(actions.cpu(), predictions), itr)
            writer.close()

            itr += 1
