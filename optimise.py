import logging
import math
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import StateDataset
from utils.performance import check_performance

mp.set_start_method('spawn', True)


# Find loss on single sample (due to dynamic computation graph)
def compute_gradient_on_sample(tree, state, action, n_simulations, gamma=0.1):
    # duplicate action n_simulations times because of anytime nature of MCTS.
    action = torch.tensor([action] * n_simulations).long()

    # find the predictions, embeddings and sampled actions
    predictions, logits, actions = tree.search(state, n_simulations, training=True)
    predictions = torch.stack(predictions)

    # Compute cross entropy loss to train differentiable parts
    loss = F.cross_entropy(predictions, action)

    # Compute decrease in loss after each simulation
    r_m = F.cross_entropy(predictions, action, reduction='none').clone().detach()
    r_m[1:] = r_m[:-1] - r_m[1:]
    r_m[0] = -r_m[0]

    # compute geometric sum for difference in loss
    for i in reversed(range(0, n_simulations - 1)):
        r_m[i] = r_m[i] + gamma * r_m[i + 1]

    # calculate loss for tree search actions
    for l_m, logits_m, action_m in zip(r_m, logits, actions):
        action_m = torch.tensor(action_m).long()

        # find logits
        logits_m = torch.stack(logits_m)

        # find negative likelihood to minimise
        negative_likelihood = F.cross_entropy(logits_m, action_m, reduction='sum') * l_m

        # add it to loss
        loss = loss + negative_likelihood

    loss.backward()

    loss = loss.item()
    accuracy = int(predictions[-1].argmax().item() == action[-1].item())

    return loss, accuracy


def worker(idx, model, dataset, args):
    try:
        # define trackers
        writer = SummaryWriter('runs/MCTSnet|worker:{}|{}'.format(idx, datetime.now().strftime("%d:%m|%H:%M")))
        logging.basicConfig(filename='logs/logs_worker_{}.log'.format(idx),
                            filemode='w',
                            format='%(message)s',
                            level=logging.DEBUG)
        track = defaultdict(list)

        optimiser = torch.optim.SGD(params=model.parameters(), lr=args.lr)

        batch_itr = 0
        for epoch in range(args.epochs):
            model.train()

            for state, action in tqdm(zip(dataset.X, dataset.y), position=idx, desc='worker:{:02}|epoch:{:02}'.format(idx, epoch), total=len(dataset)):
                # compute gradient on single sample
                loss, accuracy = compute_gradient_on_sample(model, state, action, args.n_simulations, args.gamma)

                # optimise by taking a gradient step
                clip_grad_norm_(model.parameters(), 10)
                optimiser.step()
                optimiser.zero_grad()

                track['loss'].append(loss)
                track['accuracy'].append(accuracy)

                # write batch metrics to log and tensorboard
                running_loss = np.mean(track['loss'][-100:])
                running_accuracy = np.mean(track['accuracy'][-100:])
                logging.debug('running loss: {}'.format(running_loss))
                logging.debug('running accuracy: {}'.format(running_accuracy))

                writer.add_scalar('sample/loss', running_loss, batch_itr)
                writer.add_scalar('sample/accuracy', running_accuracy, batch_itr)
                writer.close()
                batch_itr += 1

            # check the performance on an episode
            model.eval()
            with torch.no_grad():
                mean_episode_reward = np.mean([check_performance(model, args.n_simulations) for i in range(10)])

            loss = np.mean(track['loss'])
            accuracy = np.mean(track['accuracy'])

            # write epoch's metrics to logs and tensorboard
            logging.debug('\nepoch loss: {}'.format(loss))
            logging.debug('epoch accuracy: {}'.format(accuracy))
            logging.debug('mean episode reward: {}\n'.format(mean_episode_reward))

            writer.add_scalar('epoch/loss', loss, epoch)
            writer.add_scalar('epoch/accuracy', accuracy, epoch)
            writer.add_scalar('epoch/mean_episode_reward', mean_episode_reward, epoch)
            writer.close()

            # clear tracker
            track.clear()
            model.save('models/checkpoint.model')

    except KeyboardInterrupt:
        print('exiting worker:{}'.format(idx))


def optimise(model, dataset, args):
    # put model's weight in shared memory
    model.share_memory()

    # break dataset into chunks
    chunk_size = math.ceil(len(dataset) / args.n_workers)
    data_chunks = [StateDataset(dataset=dataset[i:i + chunk_size]) for i in range(0, len(dataset), chunk_size)]

    # execute in parallel and find gradients
    processes = [mp.Process(target=worker, args=(idx, model, chunk, args)) for idx, chunk in enumerate(data_chunks)]
    [p.start() for p in processes]
    try:
        [p.join() for p in processes]
    except KeyboardInterrupt:
        [p.join() for p in processes]
        raise KeyboardInterrupt
    finally:
        [p.kill() for p in processes]
