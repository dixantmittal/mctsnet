import argparse
import multiprocessing as mp
import os
import sys
from itertools import repeat
from math import ceil

import torch as t
from numpy import mean, std, asarray
from scipy.stats import sem
from tqdm import tqdm

from device import Device
from solvers.mcts import MCTS
from solvers.mctsnet import MCTSnet
from utils.performer import performer

solvers = {'mctsnet': MCTSnet(),
           'mcts': MCTS()}


def worker(idx, solver, args):
    n_gpu = t.cuda.device_count()
    if n_gpu > 0:
        Device.set_device(idx % n_gpu)

    solver.to(Device.get_device())

    rewards = []
    with t.no_grad():
        for _ in tqdm(range(args.n_samples), position=idx, desc='worker_{:02}'.format(idx), file=sys.stdout):
            rewards.append(performer(solver, args))

    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', dest='solver', required=True, help='Solver to use')
    parser.add_argument('--load_model', dest='load_model', default=None, help='Path to load model file')
    parser.add_argument('--n_simulations', dest='n_simulations', default=10, type=int, help='Number of tree simulations')
    parser.add_argument('--n_samples', dest='n_samples', default=10, type=int, help='Number of tree simulations')
    parser.add_argument('--n_workers', dest='n_workers', default=10, type=int, help='Number of tree simulations')
    args = parser.parse_args()
    args.training = False

    model = solvers[args.solver]
    model.load(args.load_model)

    args.n_samples = ceil(args.n_samples / args.n_workers)
    rewards = asarray(mp.Pool(args.n_workers)
                      .starmap_async(func=worker, iterable=zip(range(args.n_workers), repeat(model), repeat(args)))
                      .get())

    os.system('clear')
    print('Rewards in each episode:\n', rewards)
    print('mean: ', mean(rewards))
    print('stderr: ', sem(rewards, axis=None))
    print('stddev: ', std(rewards))
